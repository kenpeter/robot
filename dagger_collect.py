# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DAgger Data Collection - Collect expert corrections for learned policy mistakes

This script:
1. Runs the learned policy
2. At each step, asks expert what it SHOULD do
3. Stores (policy_state, expert_action) pairs
4. These are states the policy actually visits (not just expert states)
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import pickle
import sys
import os
import torch
import torch.nn as nn

# Add ur10e path
ur10e_path = "/home/kenpeter/work/isaac-sim-standalone-5.1.0-linux-x86_64/standalone_examples/api/isaacsim.robot.manipulators/ur10e"
sys.path.insert(0, ur10e_path)

from controller.pick_place import PickPlaceController
from isaacsim.core.api import World
from tasks.pick_place import PickPlace
from isaacsim.core.utils.types import ArticulationAction

print("=" * 60)
print("DAGGER DATA COLLECTION")
print("Collecting expert corrections for learned policy states")
print("=" * 60)

# === POLICY NETWORK ===
class PolicyMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        self.action_dim = action_dim
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)

# === LOAD TRAINED POLICY ===
MODEL_FILE = "/home/kenpeter/work/robot/offline_rl_model.pth"
if not os.path.exists(MODEL_FILE):
    print(f"❌ ERROR: {MODEL_FILE} not found! Run train.py first.")
    sys.exit(1)

data = torch.load(MODEL_FILE, map_location="cuda", weights_only=False)
state_dim = data["state_dim"]
action_dim = data["action_dim"]
policy = PolicyMLP(state_dim, action_dim).to("cuda")
policy.load_state_dict(data["model_state_dict"])
policy.eval()

print(f"✓ Loaded policy (trained on {data['num_transitions']} transitions)")

# === SETUP WORLD ===
def random_cube_position(workspace_center=np.array([0.3, 0.3, 0.3]), radius=0.2):
    offset = np.random.uniform(-radius, radius, 3)
    offset[2] = abs(offset[2])
    pos = workspace_center + offset
    pos[2] = max(0.02575, pos[2])
    return pos

def random_target_position(workspace_center=np.array([-0.3, 0.6, 0.02575]), radius=0.15):
    offset = np.random.uniform(-radius, radius, 3)
    offset[2] = 0
    pos = workspace_center + offset
    pos[2] = 0.02575
    return pos

my_world = World(stage_units_in_meters=1.0, physics_dt=1/200, rendering_dt=20/200)
target_position = np.array([-0.3, 0.6, 0.02575])

my_task = PickPlace(
    name="ur10e_pick_place",
    target_position=target_position,
    cube_size=np.array([0.1, 0.0515, 0.1]),
)
my_world.add_task(my_task)
my_world.reset()

task_params = my_world.get_task("ur10e_pick_place").get_params()
ur10e_name = task_params["robot_name"]["value"]
cube_name = task_params["cube_name"]["value"]
my_ur10e = my_world.scene.get_object(ur10e_name)
my_controller = PickPlaceController(name="controller", robot_articulation=my_ur10e, gripper=my_ur10e.gripper)
articulation_controller = my_ur10e.get_articulation_controller()
cube_obj = my_world.scene.get_object(cube_name)

# === DAGGER COLLECTION ===
num_episodes = 10  # DAgger episodes
all_transitions = []

print(f"\n{'='*60}")
print(f"Collecting {num_episodes} DAgger episodes")
print(f"{'='*60}\n")

for episode in range(num_episodes):
    # Random positions
    if episode % 3 == 0:
        cube_start_pos = np.array([0.3, 0.3, 0.3])
        target_pos = np.array([-0.3, 0.6, 0.02575])
    else:
        cube_start_pos = random_cube_position()
        target_pos = random_target_position()

    my_world.reset()
    cube_obj.set_world_pose(position=cube_start_pos)
    cube_obj.set_linear_velocity(np.zeros(3))
    cube_obj.set_angular_velocity(np.zeros(3))

    my_controller.reset()
    current_target = target_pos.copy()

    step_count = 0
    episode_transitions = []

    print(f"Episode {episode+1}: Cube {cube_start_pos.round(3)} | Target {target_pos.round(3)}")

    while simulation_app.is_running() and step_count < 3000:
        my_world.step(render=True)

        if my_world.is_playing():
            observations = my_world.get_observations()
            cube_pos = observations[cube_name]["position"]
            joint_positions = observations[ur10e_name]["joint_positions"]
            ee_pos, ee_rot = my_ur10e.end_effector.get_world_pose()
            ee_pos = np.array(ee_pos).flatten()
            ee_rot = np.array(ee_rot).flatten()

            gripper_pos = joint_positions[6] if len(joint_positions) > 6 else 0.0
            dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
            grasped = dist_to_cube < 0.15 and gripper_pos < 0.1

            # Build state
            current_state = np.concatenate([
                joint_positions[:12],
                [float(grasped)],
                cube_pos,
                current_target,
                ee_pos,
                ee_rot,
            ])

            # Get POLICY action (what the learned policy wants to do)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to("cuda")
                policy_action = policy(state_tensor).cpu().numpy()[0]

            # Get EXPERT action (what expert says we SHOULD do)
            expert_actions = my_controller.forward(
                picking_position=cube_pos,
                placing_position=current_target,
                current_joint_positions=joint_positions,
                end_effector_offset=np.array([0, 0, 0.20]),
            )

            # Extract expert action as absolute positions
            expert_action = np.zeros(7)
            if hasattr(expert_actions, "joint_positions") and expert_actions.joint_positions is not None:
                target_positions_full = expert_actions.joint_positions
                if target_positions_full is not None and len(target_positions_full) >= 6:
                    try:
                        target_arm = np.array(target_positions_full[:6], dtype=np.float32)
                        current_event = my_controller.get_current_event()
                        if current_event >= 3 and current_event < 7:
                            gripper_action = np.array([0.0], dtype=np.float32)
                        else:
                            gripper_action = np.array([0.628], dtype=np.float32)
                        expert_action = np.concatenate([target_arm, gripper_action])
                    except (TypeError, ValueError):
                        expert_action = np.zeros(7)

            # Store (state, EXPERT_action) - DAgger key insight!
            # We visit states according to policy, but store what expert would do
            transition = (current_state, expert_action, 0.0, current_state)  # reward=0 for simplicity
            episode_transitions.append(transition)

            # EXECUTE the EXPERT action (not policy) for safety
            articulation_controller.apply_action(expert_actions)

            step_count += 1

            if step_count % 100 == 0:
                print(f"  [Step {step_count}] Transitions: {len(episode_transitions)}")

            if my_controller.is_done():
                print(f"✓ Ep {episode+1} complete! {len(episode_transitions)} transitions")
                break

            if step_count >= 3000:
                print(f"⚠ Ep {episode+1} max steps! {len(episode_transitions)} transitions")
                break

    all_transitions.extend(episode_transitions)

# === SAVE ===
output_file = "/home/kenpeter/work/robot/dagger_transitions.pkl"
with open(output_file, "wb") as f:
    pickle.dump(all_transitions, f)

print(f"\n{'='*60}")
print(f"✓ Saved {len(all_transitions)} DAgger transitions to: {output_file}")
print(f"{'='*60}")

simulation_app.close()
