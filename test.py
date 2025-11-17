# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test Script - Visual Testing with Exact Same Setup as Collection

Run with:
  /home/kenpeter/work/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh test.py

Tests if trained model can replicate expert behavior in exact same scenario
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import numpy as np
import torch
import torch.nn as nn
import sys
import os

# Add ur10e example path
ur10e_path = "/home/kenpeter/work/isaac-sim-standalone-5.1.0-linux-x86_64/standalone_examples/api/isaacsim.robot.manipulators/ur10e"
sys.path.insert(0, ur10e_path)

from isaacsim.core.api import World
from tasks.pick_place import PickPlace
from isaacsim.core.utils.types import ArticulationAction

print("=" * 60)
print("TESTING MODE - Visual Testing in Isaac Sim")
print("=" * 60)

# === POLICY NETWORK (same as train.py) ===
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

# === SETUP ENVIRONMENT (EXACT SAME AS collect_data.py) ===
my_world = World(stage_units_in_meters=1.0, physics_dt=1/200, rendering_dt=20/200)

# EXACT same positions as collect_data.py
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
articulation_controller = my_ur10e.get_articulation_controller()
cube_obj = my_world.scene.get_object(cube_name)

print("✓ Environment initialized")

# === LOAD MODEL ===
MODEL_FILE = "/home/kenpeter/work/robot/offline_rl_model.pth"

if not os.path.exists(MODEL_FILE):
    print(f"❌ ERROR: Model file not found: {MODEL_FILE}")
    print("   Please run train.py first!")
    simulation_app.close()
    sys.exit(1)

data = torch.load(MODEL_FILE, map_location="cuda", weights_only=False)
state_dim = data["state_dim"]
action_dim = data["action_dim"]

model = PolicyMLP(state_dim, action_dim).to("cuda")
model.load_state_dict(data["model_state_dict"])
model.eval()

# Load normalization stats
if "normalization_stats" in data:
    norm_stats = data["normalization_stats"]
    state_mean = norm_stats["state_mean"]
    state_std = norm_stats["state_std"]
    action_mean = norm_stats["action_mean"]
    action_std = norm_stats["action_std"]
    print(f"✓ Loaded normalization stats")
else:
    state_mean = np.zeros(state_dim)
    state_std = np.ones(state_dim)
    action_mean = np.zeros(action_dim)
    action_std = np.ones(action_dim)
    print(f"⚠ No normalization stats found")

print(f"✓ Model loaded (trained on {data['num_transitions']} transitions)")
print(f"  Final training loss: {data['loss_history'][-1]:.6f}")

# === TESTING LOOP ===
print(f"\n{'='*60}")
print(f"🎬 Testing trained policy - EXACT same scenario as training")
print(f"{'='*60}\n")

num_episodes = 1
for episode in range(num_episodes):
    my_world.play()
    my_world.reset()

    # Set EXACT same initial positions as collect_data.py
    cube_start_pos = np.array([0.3, 0.3, 0.3])
    cube_obj.set_world_pose(position=cube_start_pos)
    cube_obj.set_linear_velocity(np.zeros(3))
    cube_obj.set_angular_velocity(np.zeros(3))

    print(f"Episode {episode+1}: Cube at {cube_start_pos}, Target at {target_position}")

    test_step = 0
    max_test_steps = 5000

    while simulation_app.is_running() and test_step < max_test_steps:
        my_world.step(render=True)

        if my_world.is_playing():
            # Get current state (EXACT same as collect_data.py)
            observations = my_world.get_observations()
            cube_pos = observations[cube_name]["position"]
            joint_positions = observations[ur10e_name]["joint_positions"]
            ee_pos, ee_rot = my_ur10e.end_effector.get_world_pose()
            ee_pos = np.array(ee_pos).flatten()
            ee_rot = np.array(ee_rot).flatten()

            gripper_pos = joint_positions[6] if len(joint_positions) > 6 else 0.0
            dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
            dist_to_target = np.linalg.norm(cube_pos - target_position)
            grasped = dist_to_cube < 0.15 and gripper_pos < 0.1

            # Build state (26D - EXACT same as collect_data.py)
            current_state = np.concatenate([
                joint_positions[:12],
                [float(grasped)],
                cube_pos,
                target_position,
                ee_pos,
                ee_rot,
            ])

            # Get action from trained policy
            with torch.no_grad():
                # Normalize state
                state_norm = (current_state - state_mean) / state_std
                state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to("cuda")
                action_norm = model(state_tensor).cpu().numpy()[0]

                # Denormalize action
                policy_action = action_norm * action_std + action_mean

            # Apply absolute positions (EXACT same as collect_data.py stores)
            arm_positions = policy_action[:6]
            gripper_target = np.clip(policy_action[6], 0, 0.628)

            target_joints = joint_positions.copy()
            target_joints[:6] = arm_positions
            target_joints[6:12] = gripper_target

            # Clip limits
            arm_limits_low = np.array([-2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
            arm_limits_high = np.array([2*np.pi, 0, 0, 2*np.pi, 2*np.pi, 2*np.pi])
            target_joints[:6] = np.clip(target_joints[:6], arm_limits_low, arm_limits_high)
            target_joints[6:12] = np.clip(target_joints[6:12], 0, 0.628)

            # Apply
            action_to_apply = ArticulationAction(joint_positions=target_joints)
            articulation_controller.apply_action(action_to_apply)

            test_step += 1

            if test_step % 100 == 0:
                print(f"  [Step {test_step}]: Dist cube: {dist_to_cube:.3f} | Grasped: {grasped} | Dist target: {dist_to_target:.3f}")
                print(f"      Current joints: {joint_positions[:6].round(3)}")
                print(f"      Policy output:  {arm_positions.round(3)} | Gripper: {gripper_target:.3f}")

            # Early stop
            if grasped and dist_to_target < 0.05:
                print(f"\n✓ Task completed at step {test_step}!")
                break

    print(f"\n✓ Test complete ({test_step} steps)")

print(f"\n{'='*60}")
print(f"✓ Testing finished!")
print(f"{'='*60}")

simulation_app.close()
