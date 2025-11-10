# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
SUPER SIMPLE ONLINE RL Training - Minimal version for controlled training
No vision, short episodes, basic logging. Focus on proprioception + relative pos.
"""
# sim app
from isaacsim import SimulationApp

# Initialize simulation
simulation_app = SimulationApp(
    {
        "headless": False,  # Enable UI
        "width": 1280,
        "height": 720,
        "renderer": "RayTracedLighting",
    }
)
import numpy as np
import sys
import os
import carb
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot_motion.motion_generation import (
    ArticulationMotionPolicy,
    RmpFlow,
)
import random
from isaacsim.core.api.objects import DynamicCuboid

print("Super simple DiT with standard attention - no vision")


# Standard Multi-Head Attention
class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        q = self.Wq(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_probs = F.softmax(attn_scores, dim=-1)
        o = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(B, T, D)
        o = self.Wo(o)
        return o


# Simple DiT Block
class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn = MultiheadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 4 * hidden_dim)
        )

    def forward(self, x, c):
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(
            4, dim=-1
        )
        x_norm1 = self.norm1(x) * (1 + scale_msa.unsqueeze(1))
        attn_out = self.attn(x_norm1)
        x = x + attn_out * gate_msa.unsqueeze(1)
        x_norm2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm2)
        return x


# Simple Diffusion Transformer (no vision)
class DiffusionTransformer(nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_dim=128, num_layers=3, num_heads=4
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, noisy_action, state, timestep):
        t_emb = self.time_embed(timestep)
        s_emb = self.state_encoder(state)
        a_emb = self.action_encoder(noisy_action)
        c = t_emb + s_emb
        x = a_emb.unsqueeze(1)
        for block in self.blocks:
            x = block(x, c)
        x = x.squeeze(1)
        return self.final_layer(x)


# Simple Agent
class DiTAgent:
    def __init__(self, state_dim, action_dim, device="cuda"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.num_diffusion_steps = 10
        self.betas = torch.linspace(
            0.0001, 0.02, self.num_diffusion_steps, dtype=torch.float32
        ).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod = torch.clamp(self.alphas_cumprod, min=1e-8, max=1.0)
        self.model = DiffusionTransformer(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.buffer = deque(maxlen=5000)  # Smaller buffer
        self.batch_size = 32  # Smaller batch
        self.noise_scale = 0.1
        self.step_count = 0
        self.update_freq = 10

    def get_action(self, state, deterministic=False):
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = (
                torch.randn(1, self.action_dim, dtype=torch.float32).to(self.device)
                * 0.5
            )
            for t in reversed(range(self.num_diffusion_steps)):
                timestep = torch.FloatTensor([[t / self.num_diffusion_steps]]).to(
                    self.device
                )
                predicted_noise = self.model(action, state_tensor, timestep)
                if torch.isnan(predicted_noise).any():
                    predicted_noise = torch.zeros_like(predicted_noise)
                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]
                coef1 = 1.0 / torch.sqrt(alpha_t + 1e-8)
                coef2 = beta_t / (torch.sqrt(1.0 - alpha_cumprod_t + 1e-8))
                action = coef1 * (action - coef2 * predicted_noise)
                action = torch.clamp(action, -10.0, 10.0)
                if t > 0:
                    action = action + torch.randn_like(action) * 0.1
            if not deterministic:
                action = action + self.noise_scale * 0.05 * torch.randn_like(action)
            action = torch.clamp(action, -0.5, 0.5)
            if torch.isnan(action).any():
                action = torch.randn_like(action) * 0.3
        return action.cpu().numpy()[0]

    def update(self, state, action, reward, next_state):
        experience = (state, action, reward, next_state)
        self.buffer.append(experience)
        self.step_count += 1
        if (
            len(self.buffer) >= self.batch_size
            and self.step_count % self.update_freq == 0
        ):
            self._train()

    def _train(self):
        self.model.train()
        batch_size = min(self.batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        states = torch.tensor([exp[0] for exp in batch], dtype=torch.float32).to(
            self.device
        )
        actions = torch.tensor([exp[1] for exp in batch], dtype=torch.float32).to(
            self.device
        )
        timesteps_int = torch.randint(0, self.num_diffusion_steps, (batch_size,)).to(
            self.device
        )
        timestep = (timesteps_int.float() / self.num_diffusion_steps).unsqueeze(-1)
        noise = torch.randn(batch_size, self.action_dim).to(self.device)
        sqrt_alpha_cumprod_t = self.alphas_cumprod[timesteps_int].sqrt().unsqueeze(-1)
        sqrt_one_minus_alpha_cumprod_t = (
            (1.0 - self.alphas_cumprod[timesteps_int]).sqrt().unsqueeze(-1)
        )
        noisy_actions = (
            sqrt_alpha_cumprod_t * actions + sqrt_one_minus_alpha_cumprod_t * noise
        )
        predicted_noise = self.model(noisy_actions, states, timestep)
        loss = F.mse_loss(predicted_noise, noise)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Main simulation and training loop
device = "cuda" if torch.cuda.is_available() else "cpu"
state_dim = 17  # 7 joint pos + 7 joint vel + 3 relative pos to target
action_dim = 7  # joint position targets
agent = DiTAgent(state_dim, action_dim, device)

world = World()
world.scene.add_default_ground_plane()
assets_root_path = get_assets_root_path()

# Add Franka robot
franka_asset_path = os.path.join(
    assets_root_path, "Isaac", "Robots", "Franka", "franka.usd"
)
add_reference_to_stage(franka_asset_path, "/World/Franka")

# Setup manipulator (assuming no gripper for simple reaching)
franka = SingleManipulator(
    prim_path="/World/Franka",
    name="franka",
    end_effector_prim_path="/World/Franka/panda_right_hand",  # Adjust if needed
)
world.scene.add(franka)

# Add target object (simple cube using DynamicCuboid) - FIXED
target = DynamicCuboid(
    prim_path="/World/target",
    name="target",
    position=np.array([0.5, 0.0, 0.5]),
    size=0.1,  # Single float value instead of numpy array
)
world.scene.add(target)

# Set camera view
set_camera_view(eye=np.array([1.2, 1.2, 1.0]), target=np.array([0.5, 0.0, 0.5]))

# Reset world
world.reset()

# Home position for Franka (example)
home_pos = np.array(
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
)  # Standard Franka home
franka.set_joint_positions(home_pos)
world.step(render=True)

# Training parameters
num_episodes = 100
max_steps = 200
target_pos = np.array(
    [0.5, 0.0, 0.5]
)  # Fixed target for simplicity; randomize in future
target_orientation = np.array([1.0, 0.0, 0.0, 0.0])

print("Starting training...")
success_count = 0
best_distance = float("inf")

for episode in range(num_episodes):
    world.reset()
    target.set_world_pose(position=target_pos, orientation=target_orientation)
    # Reset to home
    franka.set_joint_positions(home_pos)
    world.step(render=True)

    episode_reward = 0.0
    done = False
    step = 0
    episode_success = False

    while not done and step < max_steps:
        # Get state: joint pos (7), joint vel (7), rel pos (3)
        joint_pos = franka.get_joint_positions()
        joint_vel = franka.get_joint_velocities()
        ee_pose = franka.end_effector.get_world_pose()
        ee_pos = ee_pose[0]
        rel_pos = target_pos - ee_pos
        state = np.concatenate([joint_pos, joint_vel, rel_pos])

        # Gradually reduce exploration noise
        current_noise_scale = max(0.01, agent.noise_scale * (0.99**episode))
        agent.noise_scale = current_noise_scale

        action = agent.get_action(state)

        # Apply action (joint position targets)
        franka.set_joint_positions(action)
        world.step(render=True)

        # Get next state
        next_joint_pos = franka.get_joint_positions()
        next_joint_vel = franka.get_joint_velocities()
        next_ee_pose = franka.end_effector.get_world_pose()
        next_ee_pos = next_ee_pose[0]
        next_rel_pos = target_pos - next_ee_pos
        next_state = np.concatenate([next_joint_pos, next_joint_vel, next_rel_pos])

        # Improved reward function
        dist = np.linalg.norm(next_rel_pos)
        prev_dist = np.linalg.norm(rel_pos)

        # Distance reward + progress reward + success bonus
        reward = -dist * 2.0  # Base distance penalty
        reward += (prev_dist - dist) * 5.0  # Progress bonus

        if dist < 0.05:
            reward += 20.0
            done = True
            episode_success = True
            success_count += 1
            print(f"SUCCESS! Reached target in {step} steps")

        # Penalty for large actions to encourage smooth movements
        action_penalty = np.linalg.norm(action) * 0.01
        reward -= action_penalty

        agent.update(state, action, reward, next_state)
        episode_reward += reward

        # Update best distance
        if dist < best_distance:
            best_distance = dist

        state = next_state
        step += 1

        if step >= max_steps:
            done = True

    success_rate = (success_count / (episode + 1)) * 100
    print(
        f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Final dist = {dist:.3f}, "
        f"Success = {episode_success}, Success Rate = {success_rate:.1f}%"
    )

    # Early stopping if consistently successful
    if success_count >= 10 and episode >= 20:
        print("Good performance achieved! Stopping early.")
        break

print(f"Training complete. Final success rate: {success_rate:.1f}%")
print(f"Best distance achieved: {best_distance:.3f}")
print(f"Total successes: {success_count}/{episode + 1}")

# Test the trained agent
print("\nTesting trained agent...")
for test_episode in range(3):
    world.reset()
    target.set_world_pose(position=target_pos, orientation=target_orientation)
    franka.set_joint_positions(home_pos)
    world.step(render=True)

    test_reward = 0.0
    done = False
    step = 0

    while not done and step < max_steps:
        joint_pos = franka.get_joint_positions()
        joint_vel = franka.get_joint_velocities()
        ee_pose = franka.end_effector.get_world_pose()
        ee_pos = ee_pose[0]
        rel_pos = target_pos - ee_pos
        state = np.concatenate([joint_pos, joint_vel, rel_pos])

        # Use deterministic actions for testing
        action = agent.get_action(state, deterministic=True)
        franka.set_joint_positions(action)
        world.step(render=True)

        dist = np.linalg.norm(rel_pos)
        test_reward -= dist

        if dist < 0.05:
            print(f"Test {test_episode + 1}: SUCCESS in {step} steps!")
            done = True

        step += 1

    if not done:
        print(
            f"Test {test_episode + 1}: Failed to reach target (final dist: {dist:.3f})"
        )

simulation_app.close()
