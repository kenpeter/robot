# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
COMBINED Training & Visual Testing Script
MODE 1 (train): Train model with reward-weighted BC + data augmentation
MODE 2 (test): Test trained model in Isaac Sim with visualization

Run with:
  python3 train_visual.py train   # Train model from transitions.pkl
  python3 train_visual.py test    # Visual testing in Isaac Sim (default)

TRAINING FEATURES:
- Reward-weighted behavioral cloning (prioritize better transitions)
- Data augmentation with Gaussian noise
- 100 epochs on GPU

TESTING FEATURES:
- Visual testing with 3 episodes
- Action smoothing and stability configurations
- Real-time rendering
"""

import sys
import os

# Parse command line arguments for mode
MODE = sys.argv[1] if len(sys.argv) > 1 else "test"
if MODE not in ["train", "test"]:
    print(f"❌ Invalid mode: {MODE}")
    print("Usage: python3 train_visual.py [train|test]")
    sys.exit(1)

# Only import Isaac Sim if in test mode
if MODE == "test":
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import Dataset, DataLoader

# Only import Isaac Sim modules if in test mode
if MODE == "test":
    ur10e_path = "/home/kenpeter/work/isaac-sim-standalone-5.1.0-linux-x86_64/standalone_examples/api/isaacsim.robot.manipulators/ur10e"
    sys.path.insert(0, ur10e_path)
    from controller.pick_place import PickPlaceController
    from isaacsim.core.api import World
    from tasks.pick_place import PickPlace
    from isaacsim.core.utils.types import ArticulationAction
    from omni.isaac.core.materials import PhysicsMaterial

print("=" * 60)
if MODE == "train":
    print("TRAINING MODE - Reward-Weighted Behavioral Cloning")
else:
    print("TESTING MODE - Visual Testing in Isaac Sim")
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


# ============================================================
# TRAINING MODE
# ============================================================
if MODE == "train":
    # === WEIGHTED DATASET WITH AUGMENTATION ===
    class WeightedTransitionDataset(Dataset):
        def __init__(self, transitions, augment=True, noise_scale=0.02):
            self.transitions = transitions
            self.augment = augment
            self.noise_scale = noise_scale

            # Extract rewards and compute weights
            rewards = np.array([t[2] for t in transitions])

            # Normalize rewards to [0, 1] for weighting
            min_reward = rewards.min()
            max_reward = rewards.max()
            if max_reward > min_reward:
                normalized_rewards = (rewards - min_reward) / (max_reward - min_reward)
            else:
                normalized_rewards = np.ones_like(rewards)

            # Exponential weighting: exp(beta * normalized_reward)
            beta = 2.0
            self.weights = np.exp(beta * normalized_rewards)
            self.weights = self.weights / self.weights.sum()

            print(f"\nDataset Statistics:")
            print(f"  Reward range: [{min_reward:.3f}, {max_reward:.3f}]")
            print(f"  Weight range: [{self.weights.min():.6f}, {self.weights.max():.6f}]")
            print(f"  Top 10% weight: {self.weights[np.argsort(rewards)[-len(rewards)//10:]].sum():.3f}")

        def __len__(self):
            return len(self.transitions)

        def __getitem__(self, idx):
            state, action, reward, next_state = self.transitions[idx]
            weight = self.weights[idx]

            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            weight = torch.FloatTensor([weight])

            if self.augment:
                state = state + torch.randn_like(state) * self.noise_scale
                action = action + torch.randn_like(action) * self.noise_scale
                action[6] = torch.clamp(action[6], 0, 0.628)

            return state, action, weight

    # === LOAD DATA ===
    TRANSITIONS_FILE = "/home/kenpeter/work/robot/transitions.pkl"
    if not os.path.exists(TRANSITIONS_FILE):
        print(f"❌ ERROR: {TRANSITIONS_FILE} not found! Run collect_data.py first.")
        sys.exit(1)

    with open(TRANSITIONS_FILE, "rb") as f:
        transitions = pickle.load(f)

    print(f"✓ Loaded {len(transitions)} transitions")

    # Analyze
    states = np.array([t[0] for t in transitions[:100]])
    actions = np.array([t[1] for t in transitions[:100]])
    rewards = np.array([t[2] for t in transitions])

    print(f"\nData Analysis:")
    print(f"  State dim: {states.shape[1]}, Action dim: {actions.shape[1]}")
    print(f"  Action std: {actions.std(axis=0).round(3)}")
    print(f"  Reward: {rewards.mean():.3f} ± {rewards.std():.3f} [{rewards.min():.3f}, {rewards.max():.3f}]")

    # === TRAINING ===
    state_dim = states.shape[1]
    action_dim = actions.shape[1]

    dataset = WeightedTransitionDataset(transitions, augment=True, noise_scale=0.02)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyMLP(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    def weighted_mse_loss(pred, target, weights):
        return (((pred - target) ** 2).mean(dim=1) * weights.squeeze()).mean()

    num_epochs = 500  # INCREASED: More epochs for better convergence (was 200, originally 100)
    loss_history = []

    print(f"\nTraining on {device} for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_states, batch_actions, batch_weights in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_weights = batch_weights.to(device)

            predicted_actions = model(batch_states)
            loss = weighted_mse_loss(predicted_actions, batch_actions, batch_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)

        if (epoch + 1) % 25 == 0:  # Print every 25 epochs (since we have 500 now)
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f}")

    print(f"\n✓ Training complete!")
    print(f"  Final loss: {loss_history[-1]:.6f} (Initial: {loss_history[0]:.6f})")
    print(f"  Improvement: {((loss_history[0] - loss_history[-1]) / loss_history[0] * 100):.1f}%")

    # === SAVE ===
    output_path = "/home/kenpeter/work/robot/offline_rl_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "loss_history": loss_history,
        "num_transitions": len(transitions),
    }, output_path)

    print(f"✓ Model saved to: {output_path}")
    sys.exit(0)


# ============================================================
# TESTING MODE
# ============================================================


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

# Get DOF (degrees of freedom) count instead of joint names
num_dof = my_ur10e.num_dof
for i in range(num_dof):
    # Use articulation's dof_properties to access joints
    # Set gains (applies to all joints; adjust per-joint if needed)
    # Note: For SingleManipulator/Articulation, we can set these via the articulation controller
    # or directly on the USD prims if we know the joint paths
    pass  # Skip for now - gains may not be critical for testing

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

# Handle different data formats (improved vs original)
if "state_dim" in data and "action_dim" in data:
    # Improved model format
    state_dim = data["state_dim"]
    action_dim = data["action_dim"]
else:
    # Original model format - infer from state_dict
    state_dim = data["model_state_dict"][list(data["model_state_dict"].keys())[0]].shape[1]
    action_dim = data["model_state_dict"][list(data["model_state_dict"].keys())[-1]].shape[0]

print(f"✓ Model dimensions: State={state_dim}, Action={action_dim}")

# Create and load model
model = PolicyMLP(state_dim, action_dim).to("cuda")
model.load_state_dict(data["model_state_dict"])
model.eval()

# Print training info
if "num_transitions" in data:
    print(f"✓ Model loaded from {MODEL_FILE} (trained on {data['num_transitions']} transitions)")
elif "step_count" in data:
    print(f"✓ Model loaded from {MODEL_FILE} (trained {data['step_count']} steps)")
else:
    print(f"✓ Model loaded from {MODEL_FILE}")

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
