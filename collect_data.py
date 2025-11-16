# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data Collection Script - Generate Expert Transitions for Offline RL
Based on UR10e pick and place example with RMPFlow controller
Collects (state, action, reward, next_state) transitions and saves to pickle
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import pickle
import sys
import os

# Add ur10e example path to import controller
ur10e_path = "/home/kenpeter/work/isaac-sim-standalone-5.1.0-linux-x86_64/standalone_examples/api/isaacsim.robot.manipulators/ur10e"
sys.path.insert(0, ur10e_path)

from controller.pick_place import PickPlaceController
from isaacsim.core.api import World
from tasks.pick_place import PickPlace

# === SIMPLE REWARD FUNCTION ===
def compute_reward(
    ee_pos,
    cube_pos,
    target_pos,
    gripper_pos,
    grasped,
    task_completed,
):
    """
    Simple normalized reward:
    1. Distance to cube (closer = higher)
    2. Grasp success (binary)
    3. Distance to target after grasping (closer = higher)
    4. Task completion (highest)
    All normalized to [0, 1]
    """

    # Component 1: Distance to cube (before grasp)
    if not grasped:
        dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
        # Normalize: assume max distance ~2m, closer is better
        dist_reward = max(0.0, 1.0 - (dist_to_cube / 2.0))
        dist_reward *= 0.3  # Weight: 30% of total
    else:
        dist_reward = 0.3  # Full points if grasped

    # Component 2: Grasp success (binary)
    grasp_reward = 0.2 if grasped else 0.0  # Weight: 20%

    # Component 3: Distance to target (after grasp)
    if grasped:
        dist_to_target = np.linalg.norm(cube_pos - target_pos)
        # Normalize: assume max distance ~2m
        target_reward = max(0.0, 1.0 - (dist_to_target / 2.0))
        target_reward *= 0.3  # Weight: 30%
    else:
        target_reward = 0.0

    # Component 4: Task completion (highest reward)
    completion_reward = 0.2 if task_completed else 0.0  # Weight: 20%

    # Total normalized reward [0, 1]
    total_reward = dist_reward + grasp_reward + target_reward + completion_reward

    return np.clip(total_reward, 0.0, 1.0)


# === SETUP WORLD AND TASK ===
my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=20 / 200)

# Use EXACT same cube and target positions as pick_up_example.py
target_position = np.array([-0.3, 0.6, 0])
target_position[2] = 0.0515 / 2.0

my_task = PickPlace(
    name="ur10e_pick_place",
    target_position=target_position,
    cube_size=np.array([0.1, 0.0515, 0.1])
)
my_world.add_task(my_task)
my_world.reset()

# Get robot
task_params = my_world.get_task("ur10e_pick_place").get_params()
ur10e_name = task_params["robot_name"]["value"]
my_ur10e = my_world.scene.get_object(ur10e_name)

# Initialize controller
my_controller = PickPlaceController(
    name="controller",
    robot_articulation=my_ur10e,
    gripper=my_ur10e.gripper
)
articulation_controller = my_ur10e.get_articulation_controller()

# === DATA COLLECTION STORAGE ===
transitions = []  # List of (state, action, reward, next_state) tuples
prev_state = None
prev_action = None

print("=" * 60)
print("DATA COLLECTION - UR10e Pick and Place")
print(f"Target Position: {target_position}")
print(f"Collecting transitions to: transitions.pkl")
print("=" * 60)

# === MAIN LOOP ===
my_world.reset()
my_controller.reset()
task_completed = False
step_count = 0
pick_phase = True  # Track which phase we're in
place_phase = False
was_grasped = False  # Track when we transition from pick to place

while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_controller.reset()

        observations = my_world.get_observations()

        # Get current state components
        cube_pos = observations[task_params["cube_name"]["value"]]["position"]
        cube_target_pos = observations[task_params["cube_name"]["value"]]["target_position"]
        joint_positions = observations[task_params["robot_name"]["value"]]["joint_positions"]
        ee_pos, ee_rot = my_ur10e.end_effector.get_world_pose()
        ee_pos = np.array(ee_pos).flatten()
        ee_rot = np.array(ee_rot).flatten()

        # Gripper position (joint 6)
        gripper_pos = joint_positions[6] if len(joint_positions) > 6 else 0.0

        # Grasp detection
        dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
        grasped = (dist_to_cube < 0.15 and gripper_pos > 20.0)

        # Task completion check
        dist_to_target = np.linalg.norm(cube_pos - cube_target_pos)
        task_completed = my_controller.is_done() or (grasped and dist_to_target < 0.05)

        # Compute reward
        reward = compute_reward(
            ee_pos=ee_pos,
            cube_pos=cube_pos,
            target_pos=cube_target_pos,
            gripper_pos=gripper_pos,
            grasped=grasped,
            task_completed=task_completed,
        )

        # Build state vector (same as simple_rl_robot_arm.py)
        # 12 joints + 1 grasped + 3 cube + 3 target + 3 ee + 4 ee_rot = 26 elements
        current_state = np.concatenate([
            joint_positions[:12],  # 12 joints (6 arm + 6 gripper)
            [float(grasped)],      # 1 grasped flag
            cube_pos,              # 3 cube position
            cube_target_pos,       # 3 target position
            ee_pos,                # 3 ee position
            ee_rot,                # 4 ee rotation (quaternion)
        ])

        # Get action from controller
        actions = my_controller.forward(
            picking_position=cube_pos,
            placing_position=cube_target_pos,
            current_joint_positions=joint_positions,
            end_effector_offset=np.array([0, 0, 0.20]),
        )

        # Extract action as DELTA (change in joint positions)
        # This is critical: we want to learn changes, not absolute positions!
        action = np.zeros(6)  # Default fallback

        if hasattr(actions, 'joint_positions') and actions.joint_positions is not None:
            target_positions_full = actions.joint_positions
            # Check if target is valid and contains no None values
            if (target_positions_full is not None and
                len(target_positions_full) >= 6 and
                all(x is not None for x in target_positions_full[:6])):
                try:
                    target_positions = np.array(target_positions_full[:6], dtype=np.float32)
                    current_positions = np.array(joint_positions[:6], dtype=np.float32)
                    action = target_positions - current_positions  # DELTA = target - current
                except (TypeError, ValueError) as e:
                    # Skip this transition if conversion fails
                    action = np.zeros(6)

        # Store transition (state, action, reward, next_state)
        if prev_state is not None and prev_action is not None:
            transition = (prev_state, prev_action, reward, current_state)
            transitions.append(transition)

        # Update previous state and action
        prev_state = current_state.copy()
        prev_action = action.copy() if isinstance(action, np.ndarray) else np.array(action)

        # Apply action
        articulation_controller.apply_action(actions)

        step_count += 1

        # Track phase transitions
        if grasped and not was_grasped:
            pick_phase = False
            place_phase = True
            was_grasped = True
            print(f"\n{'='*60}")
            print(f"🎯 PICK PHASE COMPLETE! Transitioning to PLACE phase at step {step_count}")
            print(f"{'='*60}\n")

        # Log progress
        if step_count % 50 == 0:
            phase = "PLACE" if place_phase else "PICK"
            print(f"[Step {step_count}] Phase: {phase} | Reward: {reward:.3f} | Grasped: {grasped} | "
                  f"Dist to cube: {dist_to_cube:.3f} | Dist to target: {dist_to_target:.3f} | Transitions: {len(transitions)}")

        # Stop after task completion (both pick AND place)
        if my_controller.is_done():
            print(f"\n✓ PICK AND PLACE COMPLETED! Collected {len(transitions)} transitions")
            print(f"   Final cube position: {cube_pos}")
            print(f"   Target position: {cube_target_pos}")
            print(f"   Distance to target: {dist_to_target:.4f}m")
            break

        # Safety: max 5000 steps (pick+place takes longer than just pick)
        if step_count > 5000:
            print(f"\n⚠ Max steps reached without completion! Collected {len(transitions)} transitions")
            print(f"   Controller done: {my_controller.is_done()}")
            print(f"   Grasped: {grasped}")
            print(f"   Distance to target: {dist_to_target:.4f}m")
            break

# === SAVE TRANSITIONS TO PICKLE ===
output_file = "/home/kenpeter/work/robot/transitions.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(transitions, f)

print(f"\n{'='*60}")
print(f"✓ Saved {len(transitions)} transitions to: {output_file}")
print(f"{'='*60}")

simulation_app.close()
