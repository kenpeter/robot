# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple RL training script for robot arm reaching task.
The robot arm learns to reach a target position.
"""

from isaacsim import SimulationApp

# Initialize simulation
simulation_app = SimulationApp(
    {
        "headless": False,
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
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

# Import Flash Attention 4
try:
    from flash_attention_4_wrapper import flash_attention_4
    USE_FA4 = True
    print("Flash Attention 4 loaded successfully")
except Exception as e:
    USE_FA4 = False
    flash_attention_4 = None
    print(f"Flash Attention 4 not available ({e}), using standard attention")

# Diffusion Transformer for continuous action generation
class DiTBlock(nn.Module):
    """Transformer block with adaptive layer norm and Flash Attention for diffusion timestep conditioning"""
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.norm1 = nn.LayerNorm(hidden_dim)

        # QKV projection for Flash Attention
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        # Adaptive modulation parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)
        )

    def forward(self, x, c):
        """
        x: input tokens [batch, seq_len, hidden_dim]
        c: conditioning (timestep + state) [batch, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # Self-attention with adaptive modulation using Flash Attention 4
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        # QKV projection
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Use Flash Attention 4 if available, otherwise fallback
        if USE_FA4 and torch.cuda.is_available() and flash_attention_4 is not None:
            # Flash Attention 4 with async pipeline and optimized softmax
            attn_out = flash_attention_4(q, k, v)
            attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        else:
            # Fallback: PyTorch's scaled_dot_product_attention (uses FA2/FA3)
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )
            attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)

        attn_out = self.proj(attn_out)

        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP with adaptive modulation
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x

class DiffusionTransformer(nn.Module):
    """Diffusion Transformer for action generation"""
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=4, num_heads=4):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Timestep embedding (for diffusion process)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Noisy action encoder
        self.action_encoder = nn.Linear(action_dim, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        # Output head to predict noise
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, noisy_action, state, timestep):
        """
        noisy_action: [batch, action_dim] - noisy action at timestep t
        state: [batch, state_dim] - robot state
        timestep: [batch, 1] - diffusion timestep (0 to 1)

        Returns: predicted noise [batch, action_dim]
        """
        # Encode inputs
        t_emb = self.time_embed(timestep)  # [batch, hidden_dim]
        s_emb = self.state_encoder(state)   # [batch, hidden_dim]
        a_emb = self.action_encoder(noisy_action)  # [batch, hidden_dim]

        # Conditioning: combine timestep and state
        c = t_emb + s_emb  # [batch, hidden_dim]

        # Action as sequence (can be extended to multiple tokens)
        x = a_emb.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Predict noise
        x = x.squeeze(1)  # [batch, hidden_dim]
        noise_pred = self.final_layer(x)  # [batch, action_dim]

        return noise_pred

class DiTAgent:
    """RL Agent using Diffusion Transformer for action generation"""
    def __init__(self, state_dim, action_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Diffusion hyperparameters
        self.num_diffusion_steps = 10  # Reduced for faster inference
        self.beta_start = 0.0001
        self.beta_end = 0.02

        # Create diffusion schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_diffusion_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Initialize DiT model
        self.model = DiffusionTransformer(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        # Experience replay buffer
        self.buffer = []
        self.buffer_size = 10000
        self.batch_size = 64

        # Training stats
        self.episode_count = 0
        self.total_reward_history = []
        self.noise_scale = 0.3  # Exploration noise

    def get_action(self, state, deterministic=False):
        """Generate action using reverse diffusion process"""
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Start from random noise
            action = torch.randn(1, self.action_dim).to(self.device)

            # Reverse diffusion process
            for t in reversed(range(self.num_diffusion_steps)):
                timestep = torch.FloatTensor([[t / self.num_diffusion_steps]]).to(self.device)

                # Predict noise
                predicted_noise = self.model(action, state_tensor, timestep)

                # Denoise
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                beta = self.betas[t]

                if t > 0:
                    noise = torch.randn_like(action)
                    alpha_cumprod_prev = self.alphas_cumprod[t - 1]
                    sigma = torch.sqrt(beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod))
                else:
                    noise = 0
                    sigma = 0

                # Update action
                action = (action - beta / torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha)
                action = action + sigma * noise

            # Add exploration noise during training
            if not deterministic:
                action = action + self.noise_scale * torch.randn_like(action)

            action = torch.clamp(action, -0.1, 0.1)  # Limit action magnitude

        return action.cpu().numpy()[0]

    def update(self, state, action, reward, next_state):
        """Store experience and train the model"""
        # Add to buffer
        self.buffer.append((state, action, reward, next_state))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # Train if enough samples
        if len(self.buffer) < self.batch_size:
            return

        # Sample batch
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = torch.FloatTensor([s for s, a, r, ns in batch]).to(self.device)
        actions = torch.FloatTensor([a for s, a, r, ns in batch]).to(self.device)
        rewards = torch.FloatTensor([r for s, a, r, ns in batch]).to(self.device)

        # Diffusion training
        self.model.train()

        # Sample random timestep
        t = torch.randint(0, self.num_diffusion_steps, (self.batch_size,)).to(self.device)

        # Add noise to actions
        noise = torch.randn_like(actions)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1)
        noisy_actions = torch.sqrt(alpha_cumprod_t) * actions + torch.sqrt(1 - alpha_cumprod_t) * noise

        # Predict noise
        timesteps = (t.float() / self.num_diffusion_steps).view(-1, 1)
        predicted_noise = self.model(noisy_actions, states, timesteps)

        # Compute loss (MSE between predicted and actual noise)
        loss = F.mse_loss(predicted_noise, noise)

        # Add reward-weighted term
        reward_weights = torch.sigmoid(rewards / 10.0).view(-1, 1)
        weighted_loss = (loss * reward_weights.mean()).mean()

        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Decay exploration noise
        self.noise_scale = max(0.05, self.noise_scale * 0.999)

    def save_model(self, filepath):
        """Save agent state to file"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'total_reward_history': self.total_reward_history,
            'buffer': self.buffer[-1000:],  # Save last 1000 experiences
            'noise_scale': self.noise_scale
        }
        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load agent state from file"""
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found. Starting fresh.")
            return False

        model_data = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(model_data['model_state_dict'])
        self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        self.episode_count = model_data['episode_count']
        self.total_reward_history = model_data['total_reward_history']
        self.buffer = model_data.get('buffer', [])
        self.noise_scale = model_data.get('noise_scale', 0.3)

        print(f"Model loaded from {filepath}")
        print(f"Resuming from episode {self.episode_count}, buffer size: {len(self.buffer)}")
        return True


# Setup scene
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# Set camera view
set_camera_view(
    eye=[2.5, 2.5, 2.0],
    target=[0.0, 0.0, 0.5],
    camera_prim_path="/OmniverseKit_Persp",
)

# Add Franka robot arm
asset_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka")
robot = Articulation(prim_paths_expr="/World/Franka", name="franka_arm")

# Get end-effector link for position tracking
end_effector = RigidPrim("/World/Franka/panda_hand", name="end_effector")

# Add target sphere
from pxr import UsdGeom, Gf, UsdLux

stage = my_world.stage
sphere_path = "/World/Target"
sphere = UsdGeom.Sphere.Define(stage, sphere_path)
sphere.GetRadiusAttr().Set(0.05)
sphere_translate = sphere.AddTranslateOp()
sphere_translate.Set(Gf.Vec3d(0.3, 0.3, 0.5))

# Initialize world
my_world.reset()

# RL Training parameters
MODEL_PATH = "rl_robot_arm_model.pth"
state_dim = 10  # 9 joint positions (7 arm + 2 gripper) + 1 distance to target
agent = DiTAgent(state_dim=state_dim, action_dim=7)  # First 7 joints (excluding grippers)

# Try to load existing model
agent.load_model(MODEL_PATH)

num_episodes = 100
max_steps_per_episode = 200
target_position = np.array([0.3, 0.3, 0.5])
save_interval = 10  # Save model every 10 episodes

print("Starting RL Training...")
print(f"Episodes: {num_episodes}, Max steps per episode: {max_steps_per_episode}")
print(f"Model will be saved to: {MODEL_PATH}")

try:
    for episode in range(agent.episode_count, agent.episode_count + num_episodes):
        # Reset arm to random initial position
        initial_pos = np.random.uniform(-1.0, 1.0, 7)
        robot.set_joint_positions(np.concatenate([initial_pos, [0.04, 0.04]]))

        # Reset target to random position
        target_position = np.array([
            np.random.uniform(0.2, 0.5),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(0.3, 0.7)
        ])
        sphere_translate.Set(Gf.Vec3d(float(target_position[0]),
                                       float(target_position[1]),
                                       float(target_position[2])))

        episode_reward = 0

        for step in range(max_steps_per_episode):
            # Get end effector position
            joint_positions = robot.get_joint_positions()[0]  # Get first element (single robot)
            ee_position, _ = end_effector.get_world_poses()

            # Calculate state (distance to target)
            distance = np.linalg.norm(ee_position - target_position)
            state = np.concatenate([joint_positions, [distance]])

            # Get action from agent
            action = agent.get_action(state)

            # Apply action (update first 7 joints)
            new_positions = joint_positions.copy()
            new_positions[:7] += action
            new_positions[:7] = np.clip(new_positions[:7], -2.5, 2.5)
            robot.set_joint_positions([new_positions])  # Wrap in list for batch format

            # Step simulation
            my_world.step(render=True)

            # Calculate reward
            new_ee_position, _ = end_effector.get_world_poses()
            new_distance = np.linalg.norm(new_ee_position - target_position)
            reward = -new_distance

            # Bonus for reaching target
            if new_distance < 0.1:
                reward += 10.0
                print(f"Episode {episode}: Target reached at step {step}!")
                break

            # Update agent
            next_state = np.concatenate([robot.get_joint_positions()[0], [new_distance]])
            agent.update(state, action, reward, next_state)

            episode_reward += reward

        # Track statistics
        agent.total_reward_history.append(episode_reward)
        agent.episode_count = episode + 1

        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(agent.total_reward_history[-10:]) if len(agent.total_reward_history) >= 10 else np.mean(agent.total_reward_history)
            print(f"Episode {episode}/{agent.episode_count + num_episodes - 1}, Reward: {episode_reward:.2f}, Avg(10): {avg_reward:.2f}, Noise: {agent.noise_scale:.3f}")

        # Save model periodically
        if (episode + 1) % save_interval == 0:
            agent.save_model(MODEL_PATH)
            print(f"Progress saved at episode {episode}")

except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    agent.save_model(MODEL_PATH)
    print("Model saved before exit")
except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()
    agent.save_model(MODEL_PATH)
    print("Model saved after error")
finally:
    print(f"Training complete! Total episodes: {agent.episode_count}")
    print(f"Final model saved to: {MODEL_PATH}")
    agent.save_model(MODEL_PATH)
    simulation_app.close()
