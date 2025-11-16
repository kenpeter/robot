# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VISUAL TESTING - Test Trained Model
Load trained model and test it in Isaac Sim with visualization
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import sys

# Add ur10e example path
ur10e_path = "/home/kenpeter/work/isaac-sim-standalone-5.1.0-linux-x86_64/standalone_examples/api/isaacsim.robot.manipulators/ur10e"
sys.path.insert(0, ur10e_path)

from controller.pick_place import PickPlaceController
from isaacsim.core.api import World
from tasks.pick_place import PickPlace

print("=" * 60)
print("VISUAL TESTING - Testing Trained Model")
print("=" * 60)


# === SIMPLE MLP POLICY ===
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


# === SETUP ENVIRONMENT ===
my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=20 / 200)

target_position = np.array([-0.3, 0.6, 0])
target_position[2] = 0.0515 / 2.0

my_task = PickPlace(
    name="ur10e_pick_place",
    target_position=target_position,
    cube_size=np.array([0.1, 0.0515, 0.1])
)
my_world.add_task(my_task)
my_world.reset()

task_params = my_world.get_task("ur10e_pick_place").get_params()
ur10e_name = task_params["robot_name"]["value"]
my_ur10e = my_world.scene.get_object(ur10e_name)
articulation_controller = my_ur10e.get_articulation_controller()

print("✓ Environment initialized")

# === LOAD MODEL ===
MODEL_FILE = "/home/kenpeter/work/robot/offline_rl_model.pth"

if not os.path.exists(MODEL_FILE):
    print(f"❌ ERROR: Model file not found: {MODEL_FILE}")
    print("   Please run train_offline_no_visual.py first!")
    simulation_app.close()
    exit(1)

# Load model to get dimensions
data = torch.load(MODEL_FILE, map_location="cuda", weights_only=False)
state_dim = list(data["model_state_dict"].keys())[0]
state_dim = data["model_state_dict"][state_dim].shape[1]
action_dim = list(data["model_state_dict"].keys())[-1]
action_dim = data["model_state_dict"][action_dim].shape[0]

print(f"✓ Model dimensions: State={state_dim}, Action={action_dim}")

# Create and load model
model = PolicyMLP(state_dim, action_dim).to("cuda")
model.load_state_dict(data["model_state_dict"])
model.eval()

print(f"✓ Model loaded from {MODEL_FILE} (trained {data['step_count']} steps)")
print(f"  Final training loss: {data['loss_history'][-1]:.6f}")

# === TESTING LOOP ===
print(f"\n{'='*60}")
print(f"🎬 Testing trained policy in Isaac Sim")
print(f"{'='*60}\n")

my_world.play()
my_world.reset()

test_step = 0
max_test_steps = 5000

while simulation_app.is_running() and test_step < max_test_steps:
    my_world.step(render=True)

    if my_world.is_playing():
        # Get current state
        observations = my_world.get_observations()
        cube_pos = observations[task_params["cube_name"]["value"]]["position"]
        cube_target_pos = observations[task_params["cube_name"]["value"]]["target_position"]
        joint_positions = observations[task_params["robot_name"]["value"]]["joint_positions"]
        ee_pos, ee_rot = my_ur10e.end_effector.get_world_pose()
        ee_pos = np.array(ee_pos).flatten()
        ee_rot = np.array(ee_rot).flatten()

        gripper_pos = joint_positions[6] if len(joint_positions) > 6 else 0.0
        dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
        dist_to_target = np.linalg.norm(cube_pos - cube_target_pos)
        grasped = (dist_to_cube < 0.15 and gripper_pos < 0.1)

        # Build state
        current_state = np.concatenate([
            joint_positions[:12],
            [float(grasped)],
            cube_pos,
            cube_target_pos,
            ee_pos,
            ee_rot,
        ])

        # Get action from trained policy (7-DOF: 6 arm deltas + 1 gripper)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to("cuda")
            policy_action = model(state_tensor).cpu().numpy()[0]

        # Extract arm deltas (first 6) and gripper position (7th)
        arm_deltas = policy_action[:6]
        gripper_target = policy_action[6]

        # Apply arm deltas to current joint positions
        target_joints = joint_positions.copy()
        target_joints[:6] = joint_positions[:6] + arm_deltas

        # Set gripper position (all gripper joints use same value)
        target_joints[6:12] = gripper_target

        # Clip to safe joint limits
        target_joints[:6] = np.clip(target_joints[:6], -2*np.pi, 2*np.pi)
        target_joints[6:12] = np.clip(target_joints[6:12], 0, 0.628)

        # Apply via articulation controller
        from isaacsim.core.utils.types import ArticulationAction
        action_to_apply = ArticulationAction(joint_positions=target_joints)
        articulation_controller.apply_action(action_to_apply)

        test_step += 1

        if test_step % 100 == 0:
            print(f"  Step {test_step}: Dist to cube: {dist_to_cube:.3f} | Grasped: {grasped} | Dist to target: {dist_to_target:.3f}")
            print(f"             Policy action: arm_delta={arm_deltas[:3]} gripper={gripper_target:.3f}")

        # Early stop if task complete
        if grasped and dist_to_target < 0.05:
            print(f"\n✓ Task completed at step {test_step}!")
            break

print(f"\n{'='*60}")
print(f"✓ Test complete ({test_step} steps)")
print(f"{'='*60}")

simulation_app.close()
