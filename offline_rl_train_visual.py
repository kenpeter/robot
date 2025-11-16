# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OFFLINE RL Training with Visualization
Trains offline using expert data, but visualizes progress in Isaac Sim
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import sys
from collections import deque

# Add ur10e example path
ur10e_path = "/home/kenpeter/work/isaac-sim-standalone-5.1.0-linux-x86_64/standalone_examples/api/isaacsim.robot.manipulators/ur10e"
sys.path.insert(0, ur10e_path)

from controller.pick_place import PickPlaceController
from isaacsim.core.api import World
from tasks.pick_place import PickPlace

print("=" * 60)
print("OFFLINE RL TRAINING - With Visualization")
print("Training on expert data + testing in environment")
print("=" * 60)


# === SIMPLE MLP POLICY ===
class PolicyMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        self.action_dim = action_dim

        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.extend([nn.Linear(hidden_dim, action_dim), nn.Tanh()])

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


# === OFFLINE RL AGENT ===
class OfflineAgent:
    def __init__(self, state_dim, action_dim, device="cuda"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.model = PolicyMLP(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.buffer = deque(maxlen=50000)
        self.batch_size = 64

        self.step_count = 0
        self.loss_history = []

    def load_transitions(self, transitions):
        """Load expert transitions into buffer"""
        for transition in transitions:
            state, action, reward, next_state = transition
            self.buffer.append((state, action, reward, next_state))
        print(f"✓ Buffer loaded: {len(self.buffer)} transitions")

    def train_step(self):
        """Single training step"""
        if len(self.buffer) < self.batch_size:
            return None

        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = torch.FloatTensor(np.array([s for s, a, r, ns in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([a for s, a, r, ns in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([r for s, a, r, ns in batch])).to(self.device)

        self.model.train()
        predicted_actions = self.model(states)

        action_diff = predicted_actions - actions
        mse_per_sample = (action_diff ** 2).mean(dim=1)
        reward_weights = torch.softmax(rewards * 5.0, dim=0) * len(rewards)
        loss = (mse_per_sample * reward_weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.loss_history.append(loss.item())
        self.step_count += 1

        return loss.item()

    def get_action(self, state):
        """Get action from policy (for testing)"""
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.model(state_tensor)
        return action.cpu().numpy()[0]

    def save_model(self, filepath):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "loss_history": self.loss_history[-1000:],
        }, filepath)

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            return False
        data = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(data["model_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.step_count = data.get("step_count", 0)
        self.loss_history = data.get("loss_history", [])
        print(f"✓ Model loaded from {filepath} (step {self.step_count})")
        return True


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

# === LOAD DATA ===
TRANSITIONS_FILE = "/home/kenpeter/work/robot/transitions.pkl"
MODEL_FILE = "/home/kenpeter/work/robot/offline_rl_model.pth"

with open(TRANSITIONS_FILE, 'rb') as f:
    transitions = pickle.load(f)
print(f"✓ Loaded {len(transitions)} transitions")

# Verify format
state, action, reward, next_state = transitions[0]
state_dim = len(state)
action_dim = len(action)
print(f"State dim: {state_dim}, Action dim: {action_dim}")

# === INITIALIZE AGENT ===
agent = OfflineAgent(state_dim=state_dim, action_dim=action_dim, device="cuda")
agent.load_model(MODEL_FILE)
agent.load_transitions(transitions)

# === TRAINING LOOP WITH VISUALIZATION ===
NUM_TRAIN_STEPS = 10000
TEST_INTERVAL = 500  # Test policy every 500 training steps
SAVE_INTERVAL = 500

print(f"\n{'='*60}")
print(f"Starting offline training with visualization")
print(f"Train {NUM_TRAIN_STEPS} steps | Test every {TEST_INTERVAL} steps")
print(f"{'='*60}\n")

my_world.play()

train_step_counter = 0
last_test_step = -TEST_INTERVAL  # Force immediate test

while simulation_app.is_running() and train_step_counter < NUM_TRAIN_STEPS:
    # === OFFLINE TRAINING ===
    # Do several training steps before testing
    if train_step_counter < NUM_TRAIN_STEPS:
        loss = agent.train_step()

        if train_step_counter % 100 == 0 and loss is not None:
            avg_loss = np.mean(agent.loss_history[-100:]) if len(agent.loss_history) >= 100 else loss
            print(f"[Train Step {train_step_counter}/{NUM_TRAIN_STEPS}] Loss: {loss:.4f} | Avg: {avg_loss:.4f}")

        if (train_step_counter + 1) % SAVE_INTERVAL == 0:
            agent.save_model(MODEL_FILE)
            print(f"  → Checkpoint saved")

        train_step_counter += 1

    # === POLICY TESTING (VISUALIZATION) ===
    # Test the current policy in the environment
    if train_step_counter - last_test_step >= TEST_INTERVAL or train_step_counter >= NUM_TRAIN_STEPS:
        print(f"\n{'='*60}")
        print(f"🎬 Testing policy at training step {train_step_counter}")
        print(f"{'='*60}")

        # Reset environment
        my_world.reset()

        test_step = 0
        max_test_steps = 200  # Short test episode

        while test_step < max_test_steps:
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
                grasped = (dist_to_cube < 0.15 and gripper_pos > 20.0)

                # Build state
                current_state = np.concatenate([
                    joint_positions[:12],
                    [float(grasped)],
                    cube_pos,
                    cube_target_pos,
                    ee_pos,
                    ee_rot,
                ])

                # Get action from learned policy (6-DOF joint positions)
                policy_action = agent.get_action(current_state)

                # Scale action from [-1, 1] to actual joint ranges
                # The policy outputs tanh actions, need to scale to joint limits
                # UR10e approximate joint limits: [-2π, 2π] for most joints
                scaled_action = policy_action * np.pi

                # Apply learned action to robot's arm joints
                target_joints = joint_positions.copy()
                target_joints[:6] = scaled_action  # Update first 6 joints (arm)

                # Apply via articulation controller
                from isaacsim.core.utils.types import ArticulationAction
                action_to_apply = ArticulationAction(joint_positions=target_joints)
                articulation_controller.apply_action(action_to_apply)

                test_step += 1

                if test_step % 50 == 0:
                    print(f"  Test step {test_step}: Dist to cube: {dist_to_cube:.3f} | Grasped: {grasped}")

        print(f"✓ Test episode complete\n")
        last_test_step = train_step_counter

        # Continue training
        continue

# Final save
agent.save_model(MODEL_FILE)
print(f"\n{'='*60}")
print(f"✓ Training complete! Model saved to {MODEL_FILE}")
print(f"{'='*60}")

simulation_app.close()
