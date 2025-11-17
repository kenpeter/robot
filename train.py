# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Training Script - Pure Behavior Cloning

Run with:
  python3 train.py

Trains a neural network to clone expert behavior from transitions.pkl
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import Dataset, DataLoader

print("=" * 60)
print("TRAINING MODE - Reward-Weighted Behavioral Cloning")
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
DAGGER_FILE = "/home/kenpeter/work/robot/dagger_transitions.pkl"

if not os.path.exists(TRANSITIONS_FILE):
    print(f"❌ ERROR: {TRANSITIONS_FILE} not found! Run collect_data.py first.")
    sys.exit(1)

with open(TRANSITIONS_FILE, "rb") as f:
    transitions = pickle.load(f)

print(f"✓ Loaded {len(transitions)} expert transitions")

# Load DAgger data if available
if os.path.exists(DAGGER_FILE):
    with open(DAGGER_FILE, "rb") as f:
        dagger_transitions = pickle.load(f)
    print(f"✓ Loaded {len(dagger_transitions)} DAgger transitions")
    transitions.extend(dagger_transitions)
    print(f"✓ Combined total: {len(transitions)} transitions")
else:
    print(f"⚠ No DAgger data found (run dagger_collect.py to improve policy)")

# Analyze
states = np.array([t[0] for t in transitions])
actions = np.array([t[1] for t in transitions])
rewards = np.array([t[2] for t in transitions])

print(f"\nData Analysis:")
print(f"  State dim: {states.shape[1]}, Action dim: {actions.shape[1]}")
print(f"  Action std: {actions.std(axis=0).round(3)}")
print(f"  Reward: {rewards.mean():.3f} ± {rewards.std():.3f} [{rewards.min():.3f}, {rewards.max():.3f}]")

# NORMALIZE states and actions to prevent NaN
print(f"\nNormalizing data...")
state_mean = states.mean(axis=0)
state_std = states.std(axis=0)
state_std = np.where(state_std < 1e-6, 1.0, state_std)  # Replace zero std with 1.0
action_mean = actions.mean(axis=0)
action_std = actions.std(axis=0)
action_std = np.where(action_std < 1e-6, 1.0, action_std)

# Normalize
transitions_normalized = []
for state, action, reward, next_state in transitions:
    state_norm = (state - state_mean) / state_std
    action_norm = (action - action_mean) / action_std
    next_state_norm = (next_state - state_mean) / state_std
    transitions_normalized.append((state_norm, action_norm, reward, next_state_norm))

transitions = transitions_normalized
print(f"✓ Data normalized")

# === TRAINING ===
state_dim = states.shape[1]
action_dim = actions.shape[1]

# Save normalization stats for later use
normalization_stats = {
    'state_mean': state_mean,
    'state_std': state_std,
    'action_mean': action_mean,
    'action_std': action_std,
}

# Simple dataset WITHOUT weighting to avoid NaN issues
class SimpleTransitionDataset(Dataset):
    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        state, action, reward, next_state = self.transitions[idx]
        return torch.FloatTensor(state), torch.FloatTensor(action)

dataset = SimpleTransitionDataset(transitions)  # SIMPLE: No weighting, no augmentation
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PolicyMLP(state_dim, action_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)  # Standard LR

def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()

num_epochs = 500  # INCREASED: More epochs for better convergence
loss_history = []

print(f"\nTraining on {device} for {num_epochs} epochs...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch_states, batch_actions in dataloader:
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)

        predicted_actions = model(batch_states)
        loss = mse_loss(predicted_actions, batch_actions)

        # Check for NaN in loss
        if torch.isnan(loss):
            print(f"\n❌ NaN loss detected at epoch {epoch+1}!")
            print(f"   Skipping this batch...")
            continue

        optimizer.zero_grad()
        loss.backward()

        # Check for NaN gradients
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"❌ NaN gradient in {name}")
                has_nan_grad = True

        if has_nan_grad:
            print(f"Skipping update due to NaN gradients")
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Stronger clipping
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    if num_batches > 0:
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)

        if (epoch + 1) % 25 == 0:  # Print every 25 epochs
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f}")
    else:
        print(f"❌ Epoch {epoch+1}: All batches had NaN - training failed!")
        break

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
    "normalization_stats": normalization_stats,  # Save normalization for inference
}, output_path)

print(f"✓ Model saved to: {output_path}")
