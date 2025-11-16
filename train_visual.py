# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VISUAL TESTING - Test Trained Model
Load trained model and test it in Isaac Sim with visualization
UPDATED: Added initial position prints; Action safeguards (tanh scaling, smoothing); Multi-episode option
FIXED: Added initial world reset after task addition to ensure assets (e.g., cube) are spawned before accessing task params
FIXED: Aligned PolicyMLP architecture to match trained model (removed Dropout layers to fix state_dict key mismatch)
FIXED: Reduced action scale and increased smoothing for stable arm movements (prevents 'crazy' jerky behavior)
FIXED: Added stability configurations from Isaac Sim docs: higher solver iterations, low sleep/stabilization thresholds, joint gains/damping, gripper friction materials, and UR10e-specific joint limits to prevent erratic movements
FIXED: Corrected solver iteration setting to use direct attributes (num_position_iterations, num_velocity_iterations) instead of deprecated/non-existent method; Adjusted prim paths based on logs (/ur instead of /ur10e)
FIXED: Replaced set_prim_property for sleep/stabilization with Articulation API methods (set_sleep_threshold, set_stabilization_threshold) to avoid USD schema errors
FIXED: Replaced non-existent set_joint_position_gain/set_joint_velocity_gain with USD prim properties (drive:stiffness, drive:damping, drive:maxForce) on joint prims for proper gain/effort setting
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
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.prims import set_prim_property
from omni.isaac.core.materials import PhysicsMaterial

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

# FIXED: Stability - Set higher solver iterations globally (direct attributes)
physics_context = my_world.get_physics_context()
physics_context.num_position_iterations = 64
physics_context.num_velocity_iterations = 4

# EXACT same positions as collect_data.py
target_position = np.array([-0.3, 0.6, 0])
target_position[2] = 0.0515 / 2.0

my_task = PickPlace(
    name="ur10e_pick_place",
    target_position=target_position,
    cube_size=np.array([0.1, 0.0515, 0.1]),
)
my_world.add_task(my_task)

# FIXED: Reset world after adding task to spawn assets (cube, robot, etc.) before accessing params
my_world.reset()

task_params = my_world.get_task("ur10e_pick_place").get_params()
ur10e_name = task_params["robot_name"]["value"]
my_ur10e = my_world.scene.get_object(ur10e_name)
articulation_controller = my_ur10e.get_articulation_controller()

# FIXED: Stability - Set low sleep/stabilization thresholds using Articulation API (avoids USD schema issues)
my_ur10e.set_sleep_threshold(0.00005)
my_ur10e.set_stabilization_threshold(0.00001)

# FIXED: Stability - Set joint gains/damping/max_force using USD properties on joint prims (for SingleManipulator/Articulation)
stiffness = 1e5  # High for position accuracy
damping = 1e4  # High for velocity damping to prevent oscillations
arm_max_force = 500  # Reasonable torque limit for arm
gripper_max_force = 200  # For gripper

joint_names = my_ur10e.get_joint_names()  # Get all joint names
for i, joint_name in enumerate(joint_names):
    joint_path = f"{my_ur10e.prim_path}/{joint_name}"
    # Set gains (applies to all joints; adjust per-joint if needed)
    set_prim_property(joint_path, "drive:stiffness", stiffness)
    set_prim_property(joint_path, "drive:damping", damping)
    # Set effort limits (arm: first 6, gripper: index 6)
    max_f = arm_max_force if i < 6 else gripper_max_force
    set_prim_property(joint_path, "drive:maxForce", max_f)

# FIXED: Stability - Add physics materials to gripper fingers for high friction (prevents slip/unstable grasp); Wrapped in try-except to handle path issues
try:
    # Create material
    gripper_material = PhysicsMaterial(
        prim_path="/physics_materials/gripper_material",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    # Apply to finger visuals (logs show visuals/mesh_1 exists; collisions may be separate—focus on visuals for friction)
    gripper_material.apply_to(
        "/ur/ee_link/robotiq_arg2f_base_link/left_inner_finger/visuals/mesh_1"
    )
    gripper_material.apply_to(
        "/ur/ee_link/robotiq_arg2f_base_link/right_inner_finger/visuals/mesh_1"
    )
    print("✓ Gripper friction material applied")
except Exception as e:
    print(f"⚠ Warning: Could not apply gripper material: {e}")
    print("  (Non-fatal; check stage for exact finger prim paths if needed)")

print("✓ Environment initialized with stability configurations")

# === LOAD MODEL ===
MODEL_FILE = "/home/kenpeter/work/robot/offline_rl_model.pth"

if not os.path.exists(MODEL_FILE):
    print(f"❌ ERROR: Model file not found: {MODEL_FILE}")
    print("   Please run train_offline.py first!")
    simulation_app.close()
    sys.exit(1)

# Load model to get dimensions
data = torch.load(MODEL_FILE, map_location="cuda", weights_only=False)
state_dim = data["model_state_dict"][list(data["model_state_dict"].keys())[0]].shape[1]
action_dim = data["model_state_dict"][list(data["model_state_dict"].keys())[-1]].shape[
    0
]

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

num_episodes = 3  # UPDATED: Multi-episode testing (same positions)
for episode in range(num_episodes):
    my_world.play()
    my_world.reset()  # Resets to exact same initial cube/target

    # UPDATED: Print initial positions after reset
    observations = my_world.get_observations()
    initial_cube_pos = observations[task_params["cube_name"]["value"]]["position"]
    initial_target_pos = observations[task_params["cube_name"]["value"]][
        "target_position"
    ]
    print(
        f"Episode {episode+1}: Initial Cube Pos: {initial_cube_pos} | Target: {initial_target_pos}"
    )

    test_step = 0
    max_test_steps = 5000
    smoothed_action = np.zeros(7)  # For smoothing

    while simulation_app.is_running() and test_step < max_test_steps:
        my_world.step(render=True)

        if my_world.is_playing():
            # Get current state
            observations = my_world.get_observations()
            cube_pos = observations[task_params["cube_name"]["value"]]["position"]
            cube_target_pos = observations[task_params["cube_name"]["value"]][
                "target_position"
            ]
            joint_positions = observations[task_params["robot_name"]["value"]][
                "joint_positions"
            ]
            ee_pos, ee_rot = my_ur10e.end_effector.get_world_pose()
            ee_pos = np.array(ee_pos).flatten()
            ee_rot = np.array(ee_rot).flatten()

            gripper_pos = joint_positions[6] if len(joint_positions) > 6 else 0.0
            dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
            dist_to_target = np.linalg.norm(cube_pos - cube_target_pos)
            grasped = (
                dist_to_cube < 0.15 and gripper_pos < 0.1
            )  # Consistent low threshold

            # Build state (26D)
            current_state = np.concatenate(
                [
                    joint_positions[:12],
                    [float(grasped)],
                    cube_pos,
                    cube_target_pos,
                    ee_pos,
                    ee_rot,
                ]
            )

            # Get action from trained policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to("cuda")
                policy_action = model(state_tensor).cpu().numpy()[0]

            # FIXED: Smaller delta scale (pi/8) + stronger smoothing (0.9 prev + 0.1 new) for stability
            arm_deltas_raw = np.tanh(policy_action[:6]) * (
                np.pi / 8
            )  # Bound deltas [-π/8, π/8] ~0.39 rad/step
            gripper_target_raw = np.clip(policy_action[6], 0, 0.628)

            # Smooth
            raw_action = np.concatenate([arm_deltas_raw, [gripper_target_raw]])
            smoothed_action = 0.9 * smoothed_action + 0.1 * raw_action
            arm_deltas = smoothed_action[:6]
            gripper_target = smoothed_action[6]

            # Apply to targets
            target_joints = joint_positions.copy()
            target_joints[:6] = joint_positions[:6] + arm_deltas
            target_joints[6:12] = gripper_target

            # Clip limits (UR10e-specific: J0 ±2π, J1/J2 -2π to 0, J3-5 ±2π)
            arm_limits_low = np.array(
                [-2 * np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi]
            )
            arm_limits_high = np.array(
                [2 * np.pi, 0, 0, 2 * np.pi, 2 * np.pi, 2 * np.pi]
            )
            target_joints[:6] = np.clip(
                target_joints[:6], arm_limits_low, arm_limits_high
            )
            target_joints[6:12] = np.clip(target_joints[6:12], 0, 0.628)

            # Apply
            action_to_apply = ArticulationAction(joint_positions=target_joints)
            articulation_controller.apply_action(action_to_apply)

            test_step += 1

            if test_step % 100 == 0:
                print(
                    f"  [Ep {episode+1} Step {test_step}]: Dist cube: {dist_to_cube:.3f} | Grasped: {grasped} | Dist target: {dist_to_target:.3f}"
                )
                print(
                    f"             Policy raw: {policy_action[:6].round(3)} | Gripper raw: {policy_action[6]:.3f}"
                )
                print(
                    f"             Applied delta: {arm_deltas.round(3)} | Gripper: {gripper_target:.3f}"
                )

            # Early stop
            if grasped and dist_to_target < 0.05:
                print(f"\n✓ Ep {episode+1} Task completed at step {test_step}!")
                break

    print(f"\n✓ Ep {episode+1} Test complete ({test_step} steps)")

print(f"\n{'='*60}")
print(f"✓ All tests complete!")
print(f"{'='*60}")

simulation_app.close()
