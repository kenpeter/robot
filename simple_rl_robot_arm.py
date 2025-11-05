# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OFFLINE RL Training - Train on expert_dataset.pkl
The robot learns to grasp from pre-collected expert demonstrations.
No online data collection - pure behavioral cloning with Diffusion Policy.
"""

# sim app
from isaacsim import SimulationApp

# Initialize simulation
simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 1280,
        "height": 720,
        # ray trace vs path trace: ray trace -> good performance -> path trace -> more real
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
import pickle
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot_motion.motion_generation import (
    ArticulationMotionPolicy,
    interface_config_loader,
    RmpFlow,
)

# Note: Using Kimi Linear Attention (RecurrentKDA) in DiTBlocks for O(n) complexity
print("Using Kimi Linear Attention (RecurrentKDA) for efficient transformers")


# Kimi Linear Attention (RecurrentKDA)
class RecurrentKDA(nn.Module):
    """Simplified recurrent KDA (Eq. 1) for causal self-attention with O(n) complexity."""

    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wg = nn.Linear(
            d_model, d_model, bias=False
        )  # For alpha (sigmoid -> [0,1])
        self.Wbeta = nn.Linear(d_model, num_heads, bias=False)  # Per-head beta
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, state: torch.Tensor = None):
        B, T, D = x.shape
        if state is None:
            state = torch.zeros(B, self.num_heads, self.d_k, self.d_k, device=x.device)

        x_norm = self.norm(x)
        q = self.Wq(x_norm).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.Wk(x_norm).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.Wv(x_norm).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        g = (
            torch.sigmoid(self.Wg(x_norm))
            .view(B, T, self.num_heads, self.d_k)
            .transpose(1, 2)
        )  # Diag(alpha)
        beta_all = torch.sigmoid(self.Wbeta(x_norm))  # [B, T, num_heads]

        new_states = []
        outputs = []
        for t in range(T):
            # Recurrent step (causal: only up to t)
            q_t = q[:, :, t]  # [B, H, d_k]
            k_t = k[:, :, t]
            v_t = v[:, :, t]
            alpha_t = g[:, :, t]
            beta_t = beta_all[:, t, :].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

            S_t = state + beta_t * torch.einsum(
                "b h d, b h e -> b h d e", k_t, v_t
            )  # Simplified update
            S_t = (
                torch.eye(self.d_k, device=x.device)[None, None]
                - beta_t * torch.einsum("b h d, b h e -> b h d e", k_t, k_t)
            ) @ (alpha_t.unsqueeze(-1) * S_t)
            o_t = torch.einsum("b h d, b h d e -> b h e", q_t, S_t)
            outputs.append(o_t)
            new_states.append(S_t)
            state = S_t  # Update for next

        o = (
            torch.stack(outputs, dim=2).transpose(1, 2).contiguous().view(B, T, D)
        )  # [B,T,D]
        o = self.Wo(o)
        final_state = new_states[-1]  # Last state for next sequence
        return x + o, final_state  # Residual


# Diffusion Transformer for continuous action generation
class DiTBlock(nn.Module):
    """Transformer block with adaptive layer norm and Kimi Linear Attention for diffusion timestep conditioning"""

    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Use Kimi Linear Attention instead of quadratic attention
        self.kimi_attn = RecurrentKDA(hidden_dim, num_heads)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        # Adaptive modulation parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 4 * hidden_dim)  # Reduced from 6 to 4
        )

    def forward(self, x, c, kimi_state=None):
        """
        x: input tokens [batch, seq_len, hidden_dim]
        c: conditioning (timestep + state) [batch, hidden_dim]
        kimi_state: recurrent state for Kimi attention
        """
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(
            4, dim=-1
        )

        # Kimi Linear Attention with adaptive modulation
        x_scaled = x * (1 + scale_msa.unsqueeze(1))
        attn_out, new_kimi_state = self.kimi_attn(x_scaled, kimi_state)
        # attn_out already has residual from kimi_attn, just apply gate
        x = attn_out * gate_msa.unsqueeze(1)

        # MLP with adaptive modulation
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x, new_kimi_state


class SimpleCNN(nn.Module):
    """Simple CNN for visual feature extraction"""

    def __init__(self, output_dim=128, img_size=84):
        super().__init__()
        self.img_size = img_size

        # Simple convolutional layers (using GroupNorm instead of BatchNorm for stability)
        self.conv = nn.Sequential(
            # 84x84x3 -> 42x42x32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 32),  # 8 groups for 32 channels
            # 42x42x32 -> 21x21x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 64),  # 8 groups for 64 channels
            # 21x21x64 -> 10x10x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(16, 128),  # 16 groups for 128 channels
            # 10x10x128 -> 5x5x128
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(16, 128),  # 16 groups for 128 channels
        )

        # Adaptive pooling to ensure consistent output size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Always output 4x4

        # Calculate flattened size: 4x4x128 = 2048
        self.flatten_size = 4 * 4 * 128

        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 256), nn.ReLU(), nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # x: [B, H, W, C] from Isaac Sim -> [B, C, H, W] for PyTorch
        if len(x.shape) == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW

        x = self.conv(x)
        x = self.pool(x)  # Ensure consistent spatial size
        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class DiffusionTransformer(nn.Module):
    """Diffusion Transformer for action generation with vision"""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        use_vision=True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_vision = use_vision

        # Vision encoder (CNN)
        if use_vision:
            self.vision_encoder = SimpleCNN(output_dim=hidden_dim)

        # Timestep embedding (for diffusion process)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        # State encoder (proprioception only, no ball position)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Noisy action encoder
        self.action_encoder = nn.Linear(action_dim, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )

        # Output head to predict noise
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, noisy_action, state, timestep, image=None):
        """
        noisy_action: [batch, action_dim] - noisy action at timestep t
        state: [batch, state_dim] - robot proprioceptive state
        timestep: [batch, 1] - diffusion timestep (0 to 1)
        image: [batch, H, W, 3] - optional RGB image

        Returns: predicted noise [batch, action_dim]
        """
        # Encode inputs
        t_emb = self.time_embed(timestep)  # [batch, hidden_dim]
        s_emb = self.state_encoder(state)  # [batch, hidden_dim]
        a_emb = self.action_encoder(noisy_action)  # [batch, hidden_dim]

        # Conditioning: combine timestep, state, and vision
        c = t_emb + s_emb  # [batch, hidden_dim]

        if self.use_vision and image is not None:
            v_emb = self.vision_encoder(image)  # [batch, hidden_dim]
            c = c + v_emb

        # Action as sequence (can be extended to multiple tokens)
        x = a_emb.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Apply transformer blocks with Kimi linear attention
        kimi_state = None
        for block in self.blocks:
            x, kimi_state = block(x, c, kimi_state)

        # Predict noise
        x = x.squeeze(1)  # [batch, hidden_dim]
        noise_pred = self.final_layer(x)  # [batch, action_dim]

        return noise_pred


# rl agent using DiT
class DiTAgent:
    """RL Agent using Diffusion Transformer for action generation"""

    # self, state dim, action dim, use vision, device cuda
    def __init__(
        self,
        state_dim,
        action_dim,
        use_vision=True,
        device="cuda",
    ):
        # state + action -> new state: state -> action -> new state
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_vision = use_vision
        self.device = device

        # Diffusion hyperparameters (using KDA wrapper settings)
        self.num_diffusion_steps = 5  # Increased for better quality with KDA
        self.beta_start = 0.0001
        self.beta_end = 0.02

        # Create diffusion schedule with careful numerical stability
        self.betas = torch.linspace(
            self.beta_start,
            self.beta_end,
            self.num_diffusion_steps,
            dtype=torch.float32,
        ).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Clamp to avoid numerical issues
        self.alphas_cumprod = torch.clamp(self.alphas_cumprod, min=1e-8, max=1.0)

        # Initialize KDA DiffusionTransformer with vision
        self.model = DiffusionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
            use_vision=use_vision,
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        # Experience replay buffer (using deque for O(1) append/pop)
        self.buffer = deque(maxlen=10000)
        self.buffer_size = 10000
        self.batch_size = 64

        # Expert demonstration tracking
        self.expert_buffer = deque(maxlen=3000)  # Keep last 3000 expert experiences
        self.expert_sample_ratio = 0.3  # 30% of each batch from expert demos

        # Training stats
        self.episode_count = 0
        self.total_reward_history = []
        self.loss_history = []  # Track training loss
        self.noise_scale = 0.3  # Exploration noise
        self.step_count = 0  # Total training steps

    def get_action(self, state, image=None, deterministic=False):
        """Generate action using reverse diffusion process"""
        self.model.eval()
        with torch.no_grad():
            # Check for NaN in state
            if np.any(np.isnan(state)):
                print(f"WARNING: NaN detected in state: {state}")
                state = np.nan_to_num(state, 0.0)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Process image if provided
            image_tensor = None
            if image is not None and self.use_vision:
                # Check for NaN in image
                if np.any(np.isnan(image)):
                    print(f"WARNING: NaN detected in image")
                    image = np.nan_to_num(image, 0.0)
                image_tensor = (
                    torch.FloatTensor(image).unsqueeze(0).to(self.device) / 255.0
                )

            # Start from random noise
            action = (
                torch.randn(1, self.action_dim, dtype=torch.float32).to(self.device)
                * 0.5
            )

            # Reverse diffusion process (DDPM sampling)
            for t in reversed(range(self.num_diffusion_steps)):
                timestep = torch.FloatTensor([[t / self.num_diffusion_steps]]).to(
                    self.device
                )

                # Predict noise
                predicted_noise = self.model(
                    action, state_tensor, timestep, image_tensor
                )

                # Check for NaN in predicted noise
                if torch.isnan(predicted_noise).any():
                    print(f"WARNING: NaN in predicted_noise at step {t}")
                    predicted_noise = torch.zeros_like(predicted_noise)

                # Denoise using DDPM formula with numerical stability
                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]

                # Compute coefficients with clamping
                coef1 = 1.0 / torch.sqrt(alpha_t + 1e-8)
                coef2 = beta_t / (torch.sqrt(1.0 - alpha_cumprod_t + 1e-8))

                # Update action
                action = coef1 * (action - coef2 * predicted_noise)

                # Clamp intermediate values to prevent explosion
                action = torch.clamp(action, -10.0, 10.0)

                # Add noise if not final step
                if t > 0:
                    noise = torch.randn_like(action) * 0.1  # Reduced noise
                    action = action + noise

            # Add exploration noise during training
            if not deterministic:
                action = action + self.noise_scale * 0.1 * torch.randn_like(action)

            action = torch.clamp(action, -1.0, 1.0)  # Keep actions normalized

            # Final NaN check
            if torch.isnan(action).any():
                print("WARNING: NaN in action output, returning random")
                action = torch.randn_like(action) * 0.3

        return action.cpu().numpy()[0]

    def update(self, state, action, reward, next_state, image=None, is_expert=False):
        """Store experience and train the diffusion model"""
        # Only store experiences with valid vision data
        if self.use_vision and (image is None or image.size == 0 or np.all(image == 0)):
            return  # Skip invalid vision experiences

        experience = (state, action, reward, next_state, image)

        # Add to appropriate buffer
        self.buffer.append(experience)
        if is_expert:
            self.expert_buffer.append(experience)  # Also store in expert buffer

        # Train if enough samples
        if len(self.buffer) < self.batch_size:
            return

        # === PRIORITIZED SAMPLING: Mix expert + RL experiences ===
        # Sample 30% from expert buffer, 70% from main buffer
        num_expert_samples = int(self.batch_size * self.expert_sample_ratio)

        batch = []

        # Sample from expert buffer if available
        if len(self.expert_buffer) > 0 and num_expert_samples > 0:
            expert_indices = np.random.choice(
                len(self.expert_buffer),
                min(num_expert_samples, len(self.expert_buffer)),
                replace=False
            )
            batch.extend([self.expert_buffer[i] for i in expert_indices])

        # Fill remaining with RL experiences
        remaining_needed = self.batch_size - len(batch)
        if self.use_vision:
            valid_indices = [
                i for i in range(len(self.buffer)) if self.buffer[i][4] is not None
            ]
            if len(valid_indices) < remaining_needed:
                return  # Not enough valid samples
            indices = np.random.choice(valid_indices, remaining_needed, replace=False)
        else:
            indices = np.random.choice(len(self.buffer), remaining_needed, replace=False)
        batch.extend([self.buffer[i] for i in indices])

        # Convert to tensors
        states = torch.FloatTensor(np.array([s for s, a, r, ns, img in batch])).to(
            self.device
        )
        actions = torch.FloatTensor(np.array([a for s, a, r, ns, img in batch])).to(
            self.device
        )
        rewards = torch.FloatTensor(np.array([r for s, a, r, ns, img in batch])).to(
            self.device
        )

        # Process images if using vision
        images_tensor = None
        if self.use_vision:
            images = [img for s, a, r, ns, img in batch]
            # All images should be valid now due to filtering in update()
            images_tensor = torch.FloatTensor(np.stack(images)).to(self.device) / 255.0

        # Diffusion training
        self.model.train()

        # Sample random timesteps
        t = torch.randint(
            0, self.num_diffusion_steps, (self.batch_size,), dtype=torch.long
        ).to(self.device)

        # Add noise to actions (forward diffusion process)
        noise = torch.randn_like(actions)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1)

        # x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        noisy_actions = (
            torch.sqrt(alpha_cumprod_t + 1e-8) * actions
            + torch.sqrt(1.0 - alpha_cumprod_t + 1e-8) * noise
        )

        # Predict noise
        timesteps = (t.float() / self.num_diffusion_steps).view(-1, 1)
        predicted_noise = self.model(noisy_actions, states, timesteps, images_tensor)

        # Compute per-sample loss (MSE between predicted and actual noise)
        per_sample_loss = F.mse_loss(predicted_noise, noise, reduction="none").mean(
            dim=1
        )

        # Add reward weighting (prioritize good experiences)
        reward_weights = torch.sigmoid(rewards / 10.0)
        weighted_loss = (per_sample_loss * reward_weights).mean()

        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Track training metrics
        self.loss_history.append(weighted_loss.item())
        self.step_count += 1

        # Decay exploration noise - floor at 0.10 for bold exploration
        self.noise_scale = max(0.10, self.noise_scale * 0.999)

        # Return loss for logging
        return weighted_loss.item()

    def save_model(self, filepath):
        """Save agent state to file"""
        model_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode_count": self.episode_count,
            "total_reward_history": self.total_reward_history,
            "loss_history": self.loss_history[-1000:],  # Save last 1000 losses
            "buffer": list(self.buffer)[
                -1000:
            ],  # Convert deque to list, save last 1000
            "expert_buffer": list(self.expert_buffer),  # Save ALL expert demos
            "noise_scale": self.noise_scale,
            "step_count": self.step_count,
        }
        torch.save(model_data, filepath)
        print(f"Model saved to {filepath} (including {len(self.expert_buffer)} expert demos)")

    def load_model(self, filepath):
        """Load agent state from file"""
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found. Starting fresh.")
            return False

        model_data = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(model_data["model_state_dict"])
        self.optimizer.load_state_dict(model_data["optimizer_state_dict"])
        self.episode_count = model_data["episode_count"]
        self.total_reward_history = model_data["total_reward_history"]
        self.loss_history = model_data.get("loss_history", [])
        self.step_count = model_data.get("step_count", 0)

        # Load buffer and convert back to deque
        buffer_list = model_data.get("buffer", [])
        self.buffer = deque(buffer_list, maxlen=self.buffer_size)

        # Load expert buffer
        expert_buffer_list = model_data.get("expert_buffer", [])
        self.expert_buffer = deque(expert_buffer_list, maxlen=3000)

        self.noise_scale = model_data.get("noise_scale", 0.3)

        print(f"Model loaded from {filepath}")
        print(
            f"Resuming from episode {self.episode_count}, step {self.step_count}, buffer size: {len(self.buffer)}, expert demos: {len(self.expert_buffer)}"
        )
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

# === UR10e ROBOT SETUP (from test_grasp_official.py) ===
# Add UR10e robot arm with Robotiq gripper
asset_path = (
    assets_root_path
    + "/Isaac/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd"
)
robot_prim = add_reference_to_stage(usd_path=asset_path, prim_path="/World/ur")

# Configure Robotiq 2F-140 gripper (parallel jaw)
gripper = ParallelGripper(
    end_effector_prim_path="/World/ur/ee_link/robotiq_arg2f_base_link",
    joint_prim_names=["finger_joint"],
    joint_opened_positions=np.array([0]),
    joint_closed_positions=np.array([40]),
    action_deltas=np.array([-40]),
    use_mimic_joints=True,
)

# Create SingleManipulator
robot = SingleManipulator(
    prim_path="/World/ur",
    name="ur10_robot",
    end_effector_prim_path="/World/ur/ee_link/robotiq_arg2f_base_link",
    gripper=gripper,
)

# Add wrist camera for vision
from isaacsim.core.utils.prims import create_prim
from isaacsim.sensors.camera import Camera
from pxr import Gf, UsdLux

# === OVERHEAD CAMERA (Eye-to-Hand) - GitHub best practice ===
# Fixed camera above workspace for global view (like visual-pushing-grasping)
# Workspace bounds: X[-0.6, 0.8], Y[-0.6, 0.6], Z[0.05, 1.0]
# Center camera over workspace center: X=0.1, Y=0.0
camera_prim_path = "/World/OverheadCamera"
create_prim(camera_prim_path, "Camera")

# Use set_camera_view to properly position and orient camera
# Lower camera height to make objects larger in view (closer = bigger objects)
eye = Gf.Vec3d(0.1, 0.0, 1.2)  # 1.2m high - closer to objects for larger view
target = Gf.Vec3d(0.1, 0.0, 0.0)  # Look at ground at workspace center
set_camera_view(eye=eye, target=target, camera_prim_path=camera_prim_path)

# Adjust horizontal aperture for much wider field of view
camera_prim = my_world.stage.GetPrimAtPath(camera_prim_path)
camera_prim.GetAttribute("horizontalAperture").Set(
    80.0
)  # Much wider FOV for entire workspace
camera_prim.GetAttribute("verticalAperture").Set(
    80.0
)  # Match vertical for square aspect ratio

overhead_camera = Camera(
    prim_path=camera_prim_path,
    resolution=(84, 84),  # Small resolution for faster processing
)
overhead_camera.initialize()

# === RED CUBE SETUP (matching test_grasp_official.py) ===
# Add red cube with same setup as test_grasp_official.py
from isaacsim.core.api.objects import DynamicCuboid

# Use exact same cube size as test_grasp_official.py
cube_size = 0.0515  # 5.15cm cube (same as test_grasp_official.py)
cube_initial_x = 0.5  # Same as test_grasp_official.py
cube_initial_y = 0.2  # Same as test_grasp_official.py
cube_initial_z = cube_size / 2.0  # On ground plane

print(f"\n=== RED CUBE SETUP (matching test_grasp_official.py) ===")
print(
    f"Cube initial position: [{cube_initial_x:.3f}, {cube_initial_y:.3f}, {cube_initial_z:.3f}]"
)

cube = DynamicCuboid(
    name="red_cube",
    position=np.array([cube_initial_x, cube_initial_y, cube_initial_z]),
    orientation=np.array([1, 0, 0, 0]),
    prim_path="/World/Cube",
    scale=np.array([cube_size, cube_size, cube_size]),
    size=1.0,
    color=np.array([1.0, 0.0, 0.0]),  # RED cube
)
my_world.scene.add(cube)

stage = my_world.stage
sphere_path = "/World/Cube"  # Keep variable name for compatibility
sphere_translate = None  # Will be set during reset

# Cube already has RED color from DynamicCuboid - no need for extra material

# Create RigidPrim reference for tracking cube position
ball = cube  # Keep variable name "ball" for code compatibility

# === ADD LIGHTING (CRITICAL for camera vision) ===
# Add bright dome light for even illumination
dome_light_path = "/World/DomeLight"
dome_light = UsdLux.DomeLight.Define(stage, dome_light_path)
dome_light.CreateIntensityAttr(1000.0)  # Bright lighting

# Add directional light from above
distant_light_path = "/World/DistantLight"
distant_light = UsdLux.DistantLight.Define(stage, distant_light_path)
distant_light.CreateIntensityAttr(2000.0)
distant_light_xform = distant_light.AddRotateXYZOp()
distant_light_xform.Set(Gf.Vec3f(-45, 0, 0))  # Angle from above

print("=" * 50 + "\n")

# Initialize world
my_world.reset()

# Initialize robot articulation (CRITICAL - must be after world reset)
robot.initialize()
ball.initialize()
# SingleManipulator has built-in end_effector property (initialized automatically)

# === Initialize RMPflow for UR10e (from test_grasp_official.py approach) ===
print("Initializing RMPflow ArticulationMotionPolicy for UR10e...")

# Use custom RMPflow config for UR10e (same as test_grasp_official.py)
import isaacsim.robot_motion.motion_generation as mg

rmpflow_dir = os.path.join(os.path.dirname(__file__), "rmpflow")
rmp_flow = mg.lula.motion_policies.RmpFlow(
    robot_description_path=os.path.join(rmpflow_dir, "robot_descriptor.yaml"),
    rmpflow_config_path=os.path.join(rmpflow_dir, "ur10e_rmpflow_common.yaml"),
    urdf_path=os.path.join(rmpflow_dir, "ur10e.urdf"),
    end_effector_frame_name="ee_link_robotiq_arg2f_base_link",
    maximum_substep_size=0.00334,
)

# Create motion policy with physics timestep
physics_dt = 1.0 / 60.0  # 60Hz control loop
motion_policy = ArticulationMotionPolicy(robot, rmp_flow, physics_dt)
print("✓ RMPflow initialized - robot can now accurately reach and grasp the cube!")

# RL Training parameters
MODEL_PATH = "rl_robot_arm_model.pth"

# Using Cartesian control: (x, y, z, gripper) - GitHub best practice for grasping
# Simpler 4D action space instead of 12D joint control
# UR10e has 12 DOF: 6 arm joints + 6 gripper joints (with mimic)
state_dim = 13  # 12 joints + 1 cube_grasped
action_dim = 4  # dx, dy, dz, gripper

agent = DiTAgent(state_dim=state_dim, action_dim=action_dim, use_vision=True)

# Try to load existing model
agent.load_model(MODEL_PATH)

# === LOAD EXPERT DATASET ===
DATASET_PATH = "expert_dataset.pkl"
print("\n" + "=" * 70)
print(" LOADING EXPERT DATASET FOR OFFLINE TRAINING")
print("=" * 70)

if not os.path.exists(DATASET_PATH):
    print(f"\n❌ ERROR: Dataset not found at {DATASET_PATH}")
    print("Please run: python collect_expert_dataset.py")
    print("=" * 70)
    simulation_app.close()
    sys.exit(1)

print(f"\nLoading dataset from {DATASET_PATH}...")
with open(DATASET_PATH, 'rb') as f:
    expert_dataset = pickle.load(f)

print(f"✓ Dataset loaded successfully!")
print(f"  Total experiences: {len(expert_dataset['states'])}")
print(f"  State shape: {expert_dataset['states'].shape}")
print(f"  Action shape: {expert_dataset['actions'].shape}")
print(f"  Image shape: {expert_dataset['images'].shape}")
print("=" * 70 + "\n")

# Old expert seeding functions removed - using pre-collected dataset instead

# Training parameters
NUM_EPOCHS = 200
VISUALIZATION_INTERVAL = 20  # Visualize learned policy every 20 epochs
save_interval = 10  # Save model every 10 epochs

print("\n" + "=" * 70)
print(" STARTING OFFLINE TRAINING")
print("=" * 70)
print(f"Training on {len(expert_dataset['states'])} expert experiences")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Visualization every: {VISUALIZATION_INTERVAL} epochs")
print(f"Model will be saved to: {MODEL_PATH}\n")

try:
    for epoch in range(NUM_EPOCHS):
        # Train on entire expert dataset
        num_samples = len(expert_dataset['states'])
        indices = np.random.permutation(num_samples)
        epoch_losses = []

        # Train on batches from expert dataset
        for start_idx in range(0, num_samples, agent.batch_size):
            end_idx = min(start_idx + agent.batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            # Get batch from expert dataset
            for idx in batch_indices:
                state = expert_dataset['states'][idx]
                action = expert_dataset['actions'][idx]
                reward = expert_dataset['rewards'][idx]
                next_state = expert_dataset['next_states'][idx]
                image = expert_dataset['images'][idx]

                # Train on this experience (is_expert=True puts it in expert_buffer)
                loss = agent.update(state, action, reward, next_state, image, is_expert=True)
                if loss is not None:
                    epoch_losses.append(loss)

        # Log progress
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.6f} | Expert buffer: {len(agent.expert_buffer)}")

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            agent.save_model(MODEL_PATH)
            print(f"  → Checkpoint saved at epoch {epoch+1}\n")

        # Visualize learned policy in Isaac Sim
        if (epoch + 1) % VISUALIZATION_INTERVAL == 0:
            print(f"\n🎬 VISUALIZATION Epoch {epoch+1}: Testing learned policy...")

            # Reset robot
            initial_pos = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
            gripper_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            robot.set_joint_positions(np.concatenate([initial_pos, gripper_open]))

            # Reset cube to random position
            cube_size = 0.0515
            cube_x = np.random.uniform(0.4, 0.6)
            cube_y = np.random.uniform(-0.3, 0.3)
            cube_z = cube_size / 2.0
            ball.set_world_pose(position=np.array([cube_x, cube_y, cube_z]))

            # Stabilize
            for _ in range(10):
                my_world.step(render=False)

            # Run 100 steps with learned policy (visualization only)
            for viz_step in range(100):
                my_world.step(render=True)

                # Get current state and image
                overhead_camera.get_current_frame()
                rgba_data = overhead_camera.get_rgba()
                if rgba_data is not None and rgba_data.size > 0:
                    if len(rgba_data.shape) == 1:
                        rgba_data = rgba_data.reshape(84, 84, 4)
                    rgb_image = rgba_data[:, :, :3].astype(np.uint8)
                else:
                    rgb_image = np.zeros((84, 84, 3), dtype=np.uint8)

                joint_positions_raw = robot.get_joint_positions()
                if isinstance(joint_positions_raw, tuple):
                    joint_positions = joint_positions_raw[0]
                else:
                    joint_positions = joint_positions_raw

                ball_pos, _ = ball.get_world_pose()
                ball_pos = np.array(ball_pos).flatten()
                ee_pos, _ = robot.end_effector.get_world_pose()
                ee_pos = np.array(ee_pos).flatten()
                ball_dist = np.linalg.norm(ee_pos - ball_pos)
                gripper_pos = joint_positions[6] if len(joint_positions) > 6 else 0.0
                grasped = float(ball_dist < 0.15 and gripper_pos > 0.02)
                state = np.concatenate([joint_positions, [grasped]])

                # Get action from learned policy (deterministic=True for testing)
                action = agent.get_action(state, image=rgb_image, deterministic=True)

                # Execute action with RMPflow
                delta_pos = action[:3] * 0.05
                target_position = ee_pos + delta_pos
                target_position = np.clip(target_position, [-0.6, -0.6, 0.05], [0.8, 0.6, 1.0])
                rmp_flow.set_end_effector_target(target_position=target_position, target_orientation=None)
                actions = motion_policy.get_next_articulation_action(1.0/60.0)
                robot.apply_action(actions)

                # Gripper control
                gripper_action = np.clip(action[3], -1.0, 1.0)
                current_joints_raw = robot.get_joint_positions()
                if isinstance(current_joints_raw, tuple):
                    current_joints = current_joints_raw[0].copy()
                else:
                    current_joints = current_joints_raw.copy()
                if len(current_joints) > 6:
                    current_gripper = current_joints[6]
                    target_gripper = np.clip(current_gripper + gripper_action * 0.01, 0.0, 0.04)
                    current_joints[6] = target_gripper
                    robot.set_joint_positions(current_joints)

            print(f"✓ Visualization complete (Cube at [{cube_x:.2f}, {cube_y:.2f}])\n")

except KeyboardInterrupt:
    print("\nOffline training interrupted by user")
    agent.save_model(MODEL_PATH)
    print("Model saved before exit")
except Exception as e:
    print(f"Error during training: {e}")
    import traceback

    traceback.print_exc()
    agent.save_model(MODEL_PATH)
    print("Model saved after error")
finally:
    print(f"\n✓ Offline training complete!")
    print(f"✓ Final model saved to: {MODEL_PATH}")
    agent.save_model(MODEL_PATH)

    # Explicit cleanup to avoid shutdown crash
    try:
        my_world.stop()
        my_world.clear()
    except:
        pass

    simulation_app.close()
