# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OFFLINE RL Training
Loads expert transitions from pickle and trains policy using behavioral cloning
No environment interaction - pure offline learning
"""

import numpy as np
import torch
import torch.nn as nn
import pickle
import os
from collections import deque

print("=" * 60)
print("OFFLINE RL TRAINING - Behavioral Cloning")
print("Loading expert transitions from: transitions.pkl")
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

        self.buffer = deque(maxlen=50000)  # Larger buffer for offline data
        self.batch_size = 64

        self.episode_count = 0
        self.loss_history = []
        self.step_count = 0

    def load_transitions(self, transitions):
        """Load expert transitions into replay buffer"""
        print(f"Loading {len(transitions)} transitions into buffer...")
        for transition in transitions:
            state, action, reward, next_state = transition
            self.buffer.append((state, action, reward, next_state))
        print(f"✓ Buffer loaded: {len(self.buffer)} transitions")

    def train_step(self):
        """Single training step using behavioral cloning"""
        if len(self.buffer) < self.batch_size:
            print(f"⚠ Not enough data in buffer: {len(self.buffer)} < {self.batch_size}")
            return None

        # Sample batch
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = torch.FloatTensor(np.array([s for s, a, r, ns in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([a for s, a, r, ns in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([r for s, a, r, ns in batch])).to(self.device)

        # Behavioral cloning: predict expert actions
        self.model.train()
        predicted_actions = self.model(states)

        # Weighted MSE loss (higher reward = more important)
        action_diff = predicted_actions - actions
        mse_per_sample = (action_diff ** 2).mean(dim=1)

        # Weight by rewards (higher reward transitions are more important)
        reward_weights = torch.softmax(rewards * 5.0, dim=0) * len(rewards)
        loss = (mse_per_sample * reward_weights).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.loss_history.append(loss.item())
        self.step_count += 1

        return loss.item()

    def save_model(self, filepath):
        """Save model checkpoint"""
        model_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode_count": self.episode_count,
            "loss_history": self.loss_history[-1000:],
            "step_count": self.step_count,
        }
        torch.save(model_data, filepath)
        print(f"✓ Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model checkpoint"""
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found. Starting fresh.")
            return False

        model_data = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(model_data["model_state_dict"])
        self.optimizer.load_state_dict(model_data["optimizer_state_dict"])
        self.episode_count = model_data.get("episode_count", 0)
        self.loss_history = model_data.get("loss_history", [])
        self.step_count = model_data.get("step_count", 0)

        print(f"✓ Model loaded from {filepath}")
        print(f"  Episode: {self.episode_count} | Steps: {self.step_count}")
        return True


# === MAIN OFFLINE TRAINING ===
def main():
    # Paths
    TRANSITIONS_FILE = "/home/kenpeter/work/robot/transitions.pkl"
    MODEL_FILE = "/home/kenpeter/work/robot/offline_rl_model.pth"

    # Check if transitions file exists
    if not os.path.exists(TRANSITIONS_FILE):
        print(f"❌ ERROR: Transitions file not found: {TRANSITIONS_FILE}")
        print(f"   Please run collect_data.py first to generate expert transitions!")
        return

    # Load transitions from pickle
    print(f"\nLoading transitions from: {TRANSITIONS_FILE}")
    with open(TRANSITIONS_FILE, 'rb') as f:
        transitions = pickle.load(f)
    print(f"✓ Loaded {len(transitions)} transitions")

    # Verify transition format
    if len(transitions) > 0:
        state, action, reward, next_state = transitions[0]
        state_dim = len(state)
        action_dim = len(action)
        print(f"\nTransition format:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Sample reward: {reward:.3f}")
    else:
        print("❌ ERROR: No transitions found in file!")
        return

    # Initialize agent
    agent = OfflineAgent(state_dim=state_dim, action_dim=action_dim, device="cuda")

    # Load existing model if available
    agent.load_model(MODEL_FILE)

    # Load transitions into buffer
    agent.load_transitions(transitions)

    # Training loop
    NUM_TRAINING_STEPS = 10000
    SAVE_INTERVAL = 500

    print(f"\n{'='*60}")
    print(f"Starting offline training for {NUM_TRAINING_STEPS} steps")
    print(f"{'='*60}\n")

    for step in range(NUM_TRAINING_STEPS):
        loss = agent.train_step()

        # Log progress
        if step % 100 == 0 and loss is not None:
            avg_loss = np.mean(agent.loss_history[-100:]) if len(agent.loss_history) >= 100 else loss
            print(f"[Step {step}/{NUM_TRAINING_STEPS}] Loss: {loss:.4f} | Avg Loss (100): {avg_loss:.4f}")

        # Save checkpoint
        if (step + 1) % SAVE_INTERVAL == 0:
            agent.save_model(MODEL_FILE)
            print(f"  → Checkpoint saved at step {step + 1}")

    # Final save
    agent.save_model(MODEL_FILE)
    print(f"\n{'='*60}")
    print(f"✓ Training complete! Final model saved to: {MODEL_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
