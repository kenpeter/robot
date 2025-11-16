# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PURE OFFLINE RL TRAINING - No Visualization
Fast training without Isaac Sim rendering
"""

import numpy as np
import torch
import torch.nn as nn
import pickle
import os
from collections import deque

print("=" * 60)
print("OFFLINE RL TRAINING - NO VISUALIZATION")
print("Training on expert data (fast, no rendering)")
print("=" * 60)


# === SIMPLE MLP POLICY ===
class PolicyMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        self.action_dim = action_dim

        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        # No final activation - output raw deltas
        layers.append(nn.Linear(hidden_dim, action_dim))

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

# Verify gripper actions are varying
gripper_actions = [t[1][6] for t in transitions]
open_count = sum(1 for g in gripper_actions if g > 0.3)
closed_count = sum(1 for g in gripper_actions if g < 0.1)
print(f"Gripper actions: {open_count} OPEN, {closed_count} CLOSED")

if open_count == 0:
    print("❌ ERROR: All gripper values are CLOSED! Data is bad!")
    exit(1)

# === INITIALIZE AGENT ===
agent = OfflineAgent(state_dim=state_dim, action_dim=action_dim, device="cuda")
agent.load_model(MODEL_FILE)
agent.load_transitions(transitions)

# === TRAINING LOOP ===
NUM_TRAIN_STEPS = 10000
SAVE_INTERVAL = 500

print(f"\n{'='*60}")
print(f"Starting offline training (no visualization)")
print(f"Training for {NUM_TRAIN_STEPS} steps")
print(f"{'='*60}\n")

for train_step in range(NUM_TRAIN_STEPS):
    loss = agent.train_step()

    if train_step % 100 == 0 and loss is not None:
        avg_loss = np.mean(agent.loss_history[-100:]) if len(agent.loss_history) >= 100 else loss
        print(f"[Step {train_step}/{NUM_TRAIN_STEPS}] Loss: {loss:.4f} | Avg: {avg_loss:.4f}")

    if (train_step + 1) % SAVE_INTERVAL == 0:
        agent.save_model(MODEL_FILE)
        print(f"  → Checkpoint saved at step {train_step + 1}")

# Final save
agent.save_model(MODEL_FILE)
print(f"\n{'='*60}")
print(f"✓ Training complete! Model saved to {MODEL_FILE}")
print(f"  Final loss: {agent.loss_history[-1]:.6f}")
print(f"{'='*60}")
