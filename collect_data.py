# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
IMPROVED Data Collection Script - Diverse Expert Transitions for Offline RL
Combines expert demonstrations with exploration noise for better generalization
- Expert demos from RMPFlow controller (70% of actions)
- Exploration noise (30% of actions) for state coverage
- Multiple random initial positions for cube and target
- Progress-based rewards for better learning signal
- Saves (state, action, reward, next_state) transitions to pickle
"""

# sim app
from isaacsim import SimulationApp

# with ui
simulation_app = SimulationApp({"headless": False})

# np, pickle, sys, os
import numpy as np
import pickle
import sys
import os

# Add ur10e example path to import controller
ur10e_path = "/home/kenpeter/work/isaac-sim-standalone-5.1.0-linux-x86_64/standalone_examples/api/isaacsim.robot.manipulators/ur10e"
sys.path.insert(0, ur10e_path)

# pick and place controller
from controller.pick_place import PickPlaceController

# world
from isaacsim.core.api import World

# pick place
from tasks.pick_place import PickPlace


# === IMPROVED REWARD FUNCTION ===
def compute_reward(
    ee_pos,
    cube_pos,
    target_pos,
    gripper_pos,
    grasped,
    task_completed,
    prev_dist_to_cube=None,
    prev_dist_to_target=None,
):
    """
    Improved reward with progress tracking:
    1. Progress toward cube (delta-based)
    2. Grasp success (binary)
    3. Progress toward target after grasping (delta-based)
    4. Task completion (highest)
    All normalized to [-1, 1] with negative rewards for bad actions
    """

    dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
    dist_to_target = np.linalg.norm(cube_pos - target_pos)

    # Component 1: Progress to cube (before grasp)
    if not grasped:
        if prev_dist_to_cube is not None:
            # Reward for getting closer, penalty for moving away
            progress = prev_dist_to_cube - dist_to_cube
            dist_reward = np.clip(progress * 10, -0.3, 0.3)  # Scale and clip
        else:
            # Initial reward based on distance
            dist_reward = max(-0.3, 0.3 - (dist_to_cube / 2.0))
    else:
        dist_reward = 0.3  # Full points if grasped

    # Component 2: Grasp success (binary)
    grasp_reward = 0.2 if grasped else 0.0

    # Component 3: Progress to target (after grasp)
    if grasped:
        if prev_dist_to_target is not None:
            progress = prev_dist_to_target - dist_to_target
            target_reward = np.clip(progress * 10, -0.3, 0.3)
        else:
            target_reward = max(-0.3, 0.3 - (dist_to_target / 2.0))
    else:
        target_reward = 0.0

    # Component 4: Task completion (highest reward)
    completion_reward = 0.5 if task_completed else 0.0

    # Total reward
    total_reward = dist_reward + grasp_reward + target_reward + completion_reward

    return np.clip(total_reward, -1.0, 1.0), dist_to_cube, dist_to_target


# === RANDOM POSITION GENERATOR ===
def random_cube_position(workspace_center=np.array([0.3, 0.3, 0.3]), radius=0.2):
    """Generate random cube position within workspace"""
    offset = np.random.uniform(-radius, radius, 3)
    offset[2] = abs(offset[2])  # Keep z positive
    pos = workspace_center + offset
    pos[2] = max(0.02575, pos[2])  # Ensure above table
    return pos


def random_target_position(
    workspace_center=np.array([-0.3, 0.6, 0.02575]), radius=0.15
):
    """Generate random target position within workspace"""
    offset = np.random.uniform(-radius, radius, 3)
    offset[2] = 0  # Keep on table
    pos = workspace_center + offset
    pos[2] = 0.02575  # Fixed height for table
    return pos


# === SETUP WORLD AND TASK ===
my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=20 / 200)

# Initial default positions
target_position = np.array([-0.3, 0.6, 0.02575])


# flow: reset world -> add task -> get arm -> get controller -> get cube

# reset
my_world.reset()

# task: task -> no arm, no cube -> reset -> add
my_task = PickPlace(
    name="ur10e_pick_place",
    target_position=target_position,
    cube_size=np.array([0.1, 0.0515, 0.1]),
)
# my world add task
my_world.add_task(my_task)

# get ur10e: get task param -> get ur10e name -> get actual ur10e
task_params = my_world.get_task("ur10e_pick_place").get_params()
ur10e_name = task_params["robot_name"]["value"]
my_ur10e = my_world.scene.get_object(ur10e_name)

# pick place controller
my_controller = PickPlaceController(
    name="controller", robot_articulation=my_ur10e, gripper=my_ur10e.gripper
)
articulation_controller = my_ur10e.get_articulation_controller()

# Get cube object for repositioning
cube_name = task_params["cube_name"]["value"]
cube_obj = my_world.scene.get_object(cube_name)

# === DATA COLLECTION PARAMETERS ===
all_transitions = []
num_episodes = 30  # Multiple episodes with varied positions for DAgger
exploration_prob = 0.0  # PURE BEHAVIOR CLONING: Store exact expert actions only
noise_scale = 0.15  # INCREASED: Higher noise for more exploration (was 0.1)

print("=" * 60)
print("IMPROVED DATA COLLECTION - UR10e Pick and Place")
print(f"Collecting {num_episodes} diverse episodes")
print(f"Exploration probability: {exploration_prob:.1%}")
print("=" * 60)

for episode in range(num_episodes):
    # VARIED positions for diverse dataset (DAgger)
    if episode % 5 == 0:  # Every 5th episode, use default
        cube_start_pos = np.array([0.3, 0.3, 0.3])
        target_pos = np.array([-0.3, 0.6, 0.02575])
    else:
        cube_start_pos = random_cube_position()
        target_pos = random_target_position()

    # Reset world
    my_world.reset()

    # Set random positions
    cube_obj.set_world_pose(position=cube_start_pos)
    cube_obj.set_linear_velocity(np.zeros(3))
    cube_obj.set_angular_velocity(np.zeros(3))

    # Update task target (track it manually)
    current_target = target_pos.copy()

    my_controller.reset()
    transitions = []
    prev_state = None
    prev_action = None
    prev_dist_to_cube = None
    prev_dist_to_target = None
    task_completed = False
    step_count = 0

    print(
        f"\nEpisode {episode+1}: Cube: {cube_start_pos.round(3)} | Target: {target_pos.round(3)}"
    )

    while simulation_app.is_running():
        my_world.step(render=True)

        if my_world.is_playing():
            if my_world.current_time_step_index == 0:
                my_controller.reset()

            observations = my_world.get_observations()

            # Get current state components
            cube_pos = observations[cube_name]["position"]
            joint_positions = observations[ur10e_name]["joint_positions"]
            ee_pos, ee_rot = my_ur10e.end_effector.get_world_pose()
            ee_pos = np.array(ee_pos).flatten()
            ee_rot = np.array(ee_rot).flatten()

            # Gripper position
            gripper_pos = joint_positions[6] if len(joint_positions) > 6 else 0.0

            # Grasp detection
            dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
            grasped = dist_to_cube < 0.15 and gripper_pos < 0.1

            # Task completion check
            dist_to_target = np.linalg.norm(cube_pos - current_target)
            task_completed = grasped and dist_to_target < 0.05

            # Compute reward with progress tracking
            reward, new_dist_cube, new_dist_target = compute_reward(
                ee_pos=ee_pos,
                cube_pos=cube_pos,
                target_pos=current_target,
                gripper_pos=gripper_pos,
                grasped=grasped,
                task_completed=task_completed,
                prev_dist_to_cube=prev_dist_to_cube,
                prev_dist_to_target=prev_dist_to_target,
            )

            # Build state vector: 12 joints + 1 grasped + 3 cube + 3 target + 3 ee + 4 ee_rot = 26
            current_state = np.concatenate(
                [
                    joint_positions[:12],
                    [float(grasped)],
                    cube_pos,
                    current_target,
                    ee_pos,
                    ee_rot,
                ]
            )

            # Get expert action from controller
            expert_actions = my_controller.forward(
                picking_position=cube_pos,
                placing_position=current_target,
                current_joint_positions=joint_positions,
                end_effector_offset=np.array([0, 0, 0.20]),
            )

            # Extract expert action as ABSOLUTE POSITIONS (7-DOF) for exact cloning
            expert_action = np.zeros(7)

            if (
                hasattr(expert_actions, "joint_positions")
                and expert_actions.joint_positions is not None
            ):
                target_positions_full = expert_actions.joint_positions
                if (
                    target_positions_full is not None
                    and len(target_positions_full) >= 6
                    and all(x is not None for x in target_positions_full[:6])
                ):
                    try:
                        # STORE ABSOLUTE TARGET POSITIONS (not deltas!)
                        target_arm = np.array(
                            target_positions_full[:6], dtype=np.float32
                        )

                        current_event = my_controller.get_current_event()
                        if current_event >= 3 and current_event < 7:
                            gripper_action = np.array([0.0], dtype=np.float32)
                        else:
                            gripper_action = np.array([0.628], dtype=np.float32)

                        expert_action = np.concatenate([target_arm, gripper_action])
                    except (TypeError, ValueError):
                        expert_action = np.zeros(7)

            # PURE BEHAVIOR CLONING: Store exact expert action (no exploration noise)
            action = expert_action.copy()

            # Store transition
            if prev_state is not None and prev_action is not None:
                transition = (prev_state, prev_action, reward, current_state)
                transitions.append(transition)

            # Update prev
            prev_state = current_state.copy()
            prev_action = action.copy()
            prev_dist_to_cube = new_dist_cube
            prev_dist_to_target = new_dist_target

            # Apply exact expert action
            articulation_controller.apply_action(expert_actions)

            step_count += 1

            # Log
            if step_count % 100 == 0:
                print(
                    f"  [Step {step_count}] Reward: {reward:.3f} | Grasped: {grasped} | "
                    f"Dist cube: {dist_to_cube:.3f} | Dist target: {dist_to_target:.3f} | Trans: {len(transitions)}"
                )

            # Stop conditions
            if my_controller.is_done() or task_completed:
                print(f"✓ Ep {episode+1} COMPLETE! {len(transitions)} transitions")
                break

            if step_count > 3000:
                print(f"⚠ Ep {episode+1} Max steps! {len(transitions)} transitions")
                break

    all_transitions.extend(transitions)

# === SAVE ===
output_file = "/home/kenpeter/work/robot/transitions.pkl"
with open(output_file, "wb") as f:
    pickle.dump(all_transitions, f)

# Analyze and print statistics
rewards = np.array([t[2] for t in all_transitions])
print(f"\n{'='*60}")
print(f"✓ Saved {len(all_transitions)} diverse transitions to: {output_file}")
print(f"  Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
print(f"  Reward mean: {rewards.mean():.3f} ± {rewards.std():.3f}")
print(f"{'='*60}")

simulation_app.close()
