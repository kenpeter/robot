# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple RL training script for robot arm reaching task.
The robot arm learns to reach a target position.
"""

# sim app
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
from collections import deque
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


class DiTAgent:
    """RL Agent using Diffusion Transformer for action generation"""

    def __init__(
        self,
        state_dim,
        action_dim,
        use_vision=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_vision = use_vision
        self.device = device

        # Diffusion hyperparameters (using KDA wrapper settings)
        self.num_diffusion_steps = 10  # Increased for better quality with KDA
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

    def update(self, state, action, reward, next_state, image=None):
        """Store experience and train the diffusion model"""
        # Only store experiences with valid vision data
        if self.use_vision and (image is None or image.size == 0 or np.all(image == 0)):
            return  # Skip invalid vision experiences

        # Add to buffer (deque auto-evicts oldest when full)
        self.buffer.append((state, action, reward, next_state, image))

        # Train if enough samples
        if len(self.buffer) < self.batch_size:
            return

        # Sample batch - only from valid vision experiences
        if self.use_vision:
            valid_indices = [
                i for i in range(len(self.buffer)) if self.buffer[i][4] is not None
            ]
            if len(valid_indices) < self.batch_size:
                return  # Not enough valid samples
            indices = np.random.choice(valid_indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

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

        # Decay exploration noise - reduce to 0.05 for better signal-to-noise ratio
        self.noise_scale = max(0.05, self.noise_scale * 0.999)

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
            "noise_scale": self.noise_scale,
            "step_count": self.step_count,
        }
        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")

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
        self.noise_scale = model_data.get("noise_scale", 0.3)

        print(f"Model loaded from {filepath}")
        print(
            f"Resuming from episode {self.episode_count}, step {self.step_count}, buffer size: {len(self.buffer)}"
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
from pxr import UsdGeom, Gf, UsdLux, UsdPhysics, UsdShade

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

# === RED CUBE SETUP (randomized position) ===
# Add red cube (instead of ball) with random initial position
from isaacsim.core.api.objects import DynamicCuboid

# Randomize cube initial position within reachable workspace
cube_size = 0.0515  # 5.15cm cube (same as test_grasp_official.py)
cube_initial_x = np.random.uniform(0.2, 0.5)  # Reachable by UR10e
cube_initial_y = np.random.uniform(-0.2, 0.2)
cube_initial_z = cube_size / 2.0  # On ground plane

print(f"\n=== RANDOMIZED SCENE ===")
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

# === RANDOMIZED GOAL/TARGET LOCATION ===
# Add goal location marker (green box) with random position
goal_x = np.random.uniform(-0.4, -0.1)  # Different area than cube
goal_y = np.random.uniform(-0.2, 0.2)
goal_z = 0.075  # Half of green marker size

print(f"Goal position: [{goal_x:.3f}, {goal_y:.3f}, {goal_z:.3f}]")
print("=" * 50 + "\n")

goal_path = "/World/Goal"
goal_cube = UsdGeom.Cube.Define(stage, goal_path)
goal_cube.GetSizeAttr().Set(0.15)
goal_translate = goal_cube.AddTranslateOp()
goal_translate.Set(Gf.Vec3d(goal_x, goal_y, goal_z))

# Add GREEN material to goal marker
from pxr import Sdf

goal_material_path = "/World/Looks/GreenMaterial"
goal_material = UsdShade.Material.Define(stage, goal_material_path)
goal_shader = UsdShade.Shader.Define(stage, goal_material_path + "/Shader")
goal_shader.CreateIdAttr("UsdPreviewSurface")
goal_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
    Gf.Vec3f(0.0, 1.0, 0.0)
)  # Pure green
goal_material.CreateSurfaceOutput().ConnectToSource(
    goal_shader.ConnectableAPI(), "surface"
)
goal_prim = stage.GetPrimAtPath(goal_path)
UsdShade.MaterialBindingAPI.Apply(goal_prim).Bind(goal_material)

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

num_episodes = 2000  # Train much longer for complex vision task
# 500 step for reach grasp delivery
max_steps_per_episode = 1000
save_interval = 10  # Save model every 10 episodes
vision_debug_saved = False  # Flag to save one camera image for debugging

# NOTE: Goal position is now RANDOMIZED per episode (see episode loop below)
# No static goal marker created here - goal visualization moved to episode loop

print("Starting RL Training...")
print(f"Episodes: {num_episodes}, Max steps per episode: {max_steps_per_episode}")
print(f"Model will be saved to: {MODEL_PATH}")

try:
    for episode in range(agent.episode_count, agent.episode_count + num_episodes):
        # Reset arm to UPRIGHT vertical position for UR10e
        # UR10e has 12 DOF total: 6 arm joints + 6 gripper joints (with mimic)
        # Arm joints: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
        # Gripper joints: finger_joint, left_inner_finger, left_inner_knuckle,
        #                 right_inner_finger, right_inner_knuckle, right_outer_knuckle
        # ALL ZEROS = arm points straight up (perpendicular to floor)

        # Reset arm to UPRIGHT vertical position for UR10e (suggested safe initial pose)
        initial_pos = np.deg2rad(
            np.array([0.0, 45.0, 90.0, 45.0, 90.0, 0.0])
        )  # Degrees to radians
        initial_pos += np.random.uniform(-0.05, 0.05, 6)  # Tiny perturbation

        # Gripper open position: all joints at 0
        gripper_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 6 gripper joints
        robot.set_joint_positions(np.concatenate([initial_pos, gripper_open]))

        # Step world multiple times to stabilize physics
        for _ in range(10):
            my_world.step(render=False)

        # === CURRICULUM LEARNING: Start easy, gradually increase difficulty ===
        # progress: agent step -> progress -> harder
        curriculum_progress = min(agent.step_count / 500000.0, 1.0)

        # progress: ball closer -> ball further
        # Early training: ball closer (0.2-0.4m), Later: ball farther (0.2-0.6m)
        min_distance = 0.2
        max_distance = 0.2 + (
            0.4 * curriculum_progress
        )  # 0.2->0.6 as training progresses

        # === DOMAIN RANDOMIZATION (GitHub best practice - improves generalization) ===
        # Reset RED CUBE to random position - FARTHER from robot base (at origin)
        cube_x = np.random.uniform(0.4, 0.6)  # Farther in positive X direction
        cube_y = np.random.uniform(-0.3, 0.3)  # Wider Y range
        cube_z = cube_size / 2.0  # On ground plane

        cube_position = np.array([cube_x, cube_y, cube_z], dtype=np.float32).flatten()
        ball.set_world_pose(
            position=cube_position
        )  # ball variable = cube (uses singular)

        # Randomize GOAL position - FARTHER on opposite side
        goal_x = np.random.uniform(-0.6, -0.3)  # Farther in negative X direction
        goal_y = np.random.uniform(-0.3, 0.3)  # Wider Y range
        goal_z = 0.075  # Half of green marker size
        goal_position = np.array([goal_x, goal_y, goal_z], dtype=np.float32)

        # Update goal marker position
        goal_translate.Set(Gf.Vec3d(goal_x, goal_y, goal_z))

        episode_reward = 0
        ball_grasped = False
        last_ball_visible = False  # Track previous visibility for smoothing
        closest_distance = float("inf")  # Track closest distance for HER
        step = 0  # Initialize step counter

        for step in range(max_steps_per_episode):
            # Step simulation FIRST to update physics
            my_world.step(render=True)

            # THEN capture camera image (synchronized with current state)
            overhead_camera.get_current_frame()
            rgba_data = overhead_camera.get_rgba()

            # Check if data is valid
            if rgba_data is None or rgba_data.size == 0:
                # Use dummy image if camera not ready
                rgb_image = np.zeros((84, 84, 3), dtype=np.uint8)
            else:
                # Reshape from flat array to image if needed
                if len(rgba_data.shape) == 1:
                    rgba_data = rgba_data.reshape(84, 84, 4)
                rgb_image = rgba_data[:, :, :3].astype(
                    np.uint8
                )  # Get RGB only (84x84x3)

            # Get robot state (after step, synchronized with image)
            joint_positions_raw = robot.get_joint_positions()
            # Handle both tuple and direct array returns
            if isinstance(joint_positions_raw, tuple):
                joint_positions = joint_positions_raw[0]
            else:
                joint_positions = joint_positions_raw

            ball_pos, _ = ball.get_world_pose()  # DynamicCuboid uses singular
            ball_pos = np.array(ball_pos, dtype=np.float32).flatten()
            ee_position, _ = robot.end_effector.get_world_pose()
            ee_position = np.array(ee_position, dtype=np.float32).flatten()

            # Calculate distances (for reward only, not in state!)
            ball_distance = np.linalg.norm(ee_position - ball_pos)
            goal_distance = np.linalg.norm(ball_pos - goal_position)

            # Gripper state (UR10e has 6 gripper joints, but finger_joint is main control)
            # joint_positions[6] = finger_joint: 0 = open, 40 = closed
            if len(joint_positions) > 6:
                gripper_position = joint_positions[6]  # Main gripper joint
            else:
                print(
                    f"Warning: joint_positions has only {len(joint_positions)} elements"
                )
                gripper_position = 0.0

            # Check if cube is grasped (cube is easier than ball)
            # Distance < 0.15m and gripper is closing (position > 0.02)
            if ball_distance < 0.15 and gripper_position > 0.02:
                ball_grasped = True

            # Build state: joints + cube_grasped (NO cube position!)
            state = np.concatenate(
                [joint_positions, [float(ball_grasped)]]  # 12 joints  # 1 grasped flag
            )  # Total: 13

            # Get action from agent WITH VISION
            action = agent.get_action(state, image=rgb_image)

            # Vision debugging: Check if RED CUBE is visible in camera
            ball_visible = False  # Keep variable name for compatibility
            ball_pixel_count = 0
            ball_centroid_x = 0.0
            ball_centroid_y = 0.0
            if (
                rgb_image is not None
                and rgb_image.size > 0
                and not np.all(rgb_image == 0)
            ):
                # With overhead camera, detect RED CUBE (renders as red with high lighting)
                # Look for reddish pixels that stand out from blue background
                # Cube appears as R>140 with washed out colors due to lighting
                ball_mask = (
                    (rgb_image[:, :, 0] > 140)  # Reddish (higher than background)
                    & (rgb_image[:, :, 1] < 210)  # Not pure white
                    & (rgb_image[:, :, 2] < 210)  # Not pure white
                )
                ball_pixel_count = np.sum(ball_mask)

                # Overhead view: cube should be 5-300 pixels (5.15cm cube from above at 1.2m height)
                ball_visible = (ball_pixel_count > 5) and (ball_pixel_count < 300)

                # Calculate cube centroid in image (for visual servoing reward)
                if ball_visible:
                    y_coords, x_coords = np.where(ball_mask)
                    if len(x_coords) > 0:
                        ball_centroid_x = np.mean(x_coords) / 84.0  # Normalize to 0-1
                        ball_centroid_y = np.mean(y_coords) / 84.0  # Normalize to 0-1

                # Temporal smoothing: If cube was visible last frame and close in distance, keep visible
                if not ball_visible and last_ball_visible and ball_distance < 0.3:
                    ball_visible = True  # Assume still visible (avoid flicker)

                # Save one debug image (first valid frame) showing cube and goal marker
                if not vision_debug_saved and ball_pixel_count > 0:
                    try:
                        import cv2

                        # Detect green bucket pixels
                        green_mask = (
                            (rgb_image[:, :, 0] < 100)  # Low red
                            & (rgb_image[:, :, 1] > 100)  # High green
                            & (rgb_image[:, :, 2] < 100)  # Low blue
                        )
                        green_pixel_count = np.sum(green_mask)

                        # Create debug image with both cube and goal marker highlighted
                        debug_img = rgb_image.copy()
                        # Highlight cube pixels in bright green
                        debug_img[ball_mask] = [0, 255, 0]
                        # Highlight green goal marker pixels in yellow
                        debug_img[green_mask] = [255, 255, 0]

                        # Save raw camera view
                        cv2.imwrite(
                            "rl_camera_raw.png",
                            cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
                        )
                        # Save annotated view with cube and goal marker highlighted
                        cv2.imwrite(
                            "rl_camera_annotated.png",
                            cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR),
                        )
                        print(f"\n=== VISION DEBUG: Saved camera images ===")
                        print(
                            f"    Cube pixels: {ball_pixel_count}, Cube visible: {ball_visible}"
                        )
                        print(f"    Green goal marker pixels: {green_pixel_count}")
                        print(f"    Saved: rl_camera_raw.png")
                        print(f"    Saved: rl_camera_annotated.png")
                        vision_debug_saved = True
                    except Exception as e:
                        print(f"Failed to save debug images: {e}")
                        pass  # OpenCV not available, skip image save

            # Update last visibility for next frame
            last_ball_visible = ball_visible

            # Debug: Print action and vision info every 50 steps
            if step % 50 == 0:
                print(
                    f"Step {step}: Action range [{action.min():.3f}, {action.max():.3f}], Mean: {action.mean():.3f}"
                )
                print(f"  Cube dist: {ball_distance:.3f}, EE: {ee_position}")
                print(
                    f"  Vision: Cube pixels={ball_pixel_count}, Cube visible={ball_visible}"
                )

            # === CARTESIAN ACTION EXECUTION ===
            # Action = [dx, dy, dz, gripper] - 4D end-effector control
            new_positions = joint_positions.copy()

            # Extract Cartesian delta from action
            delta_pos = (
                action[:3] * 0.05
            )  # Cartesian movements (5cm max per step - increased for faster learning)

            # Calculate target end-effector position
            target_position = ee_position + delta_pos
            target_position = np.clip(
                target_position, [-0.6, -0.6, 0.05], [0.8, 0.6, 1.0]
            )

            # === USE RMPFLOW MOTION POLICY FOR ACCURATE REACHING ===
            # RMPflow provides accurate task-space control for physical grasping
            # RL learns high-level strategy (where to move), RMPflow handles precise low-level control

            # Set target on RmpFlow object (not on ArticulationMotionPolicy!)
            rmp_flow.set_end_effector_target(
                target_position=target_position,
                target_orientation=None,  # Let RMPflow handle orientation
            )

            # Get next action from motion policy wrapper (only arm joints - 7 DOF)
            actions = motion_policy.get_next_articulation_action(physics_dt)

            # Apply RMPflow arm motion
            robot.apply_action(actions)

            # Gripper control (action[3] from RL agent)
            # UR10e has 6 gripper joints, but we control via finger_joint (index 6)
            # The mimic joints will automatically follow
            gripper_action = np.clip(action[3], -1.0, 1.0)

            # Get current joint positions with safe handling
            current_joints_raw = robot.get_joint_positions()
            if isinstance(current_joints_raw, tuple):
                current_joints = current_joints_raw[0].copy()
            else:
                current_joints = current_joints_raw.copy()

            if len(current_joints) > 6:
                current_gripper = current_joints[6]  # finger_joint

                # Map action to gripper position change: -1 = open, +1 = close
                # Scale factor 0.01 for smooth control
                target_gripper = np.clip(
                    current_gripper + gripper_action * 0.01, 0.0, 0.04
                )

                # Apply velocity-limited gripper motion
                max_gripper_vel = 0.005  # 0.5cm/step maximum velocity
                gripper_delta = np.clip(
                    target_gripper - current_gripper, -max_gripper_vel, max_gripper_vel
                )

                # Set gripper joint (RMPflow doesn't control gripper)
                # Only update finger_joint - mimic joints auto-follow
                current_joints[6] = np.clip(current_gripper + gripper_delta, 0.0, 0.04)
                robot.set_joint_positions(current_joints)
            else:
                print(
                    f"Warning: Cannot control gripper, only {len(current_joints)} joints available"
                )

            # === Enhanced State Tracking and Distance Calculation ===
            # Get current poses with error handling
            try:
                new_ball_pos, _ = ball.get_world_pose()  # DynamicCuboid uses singular
                new_ball_pos = np.array(new_ball_pos, dtype=np.float32).flatten()
                new_ee_position, _ = robot.end_effector.get_world_pose()
                new_ee_position = np.array(new_ee_position, dtype=np.float32).flatten()
            except Exception as e:
                print(f"Error getting poses: {e}")
                # Use previous values if there's an error
                new_ball_pos = ball_pos
                new_ee_position = ee_position

            # Calculate distances with stability checks
            try:
                # End-effector to ball distance
                new_ball_distance = np.linalg.norm(new_ee_position - new_ball_pos)
                if np.isnan(new_ball_distance) or np.isinf(new_ball_distance):
                    print("Warning: Invalid ball distance, using previous value")
                    new_ball_distance = ball_distance

                # Ball to goal distance
                new_goal_distance = np.linalg.norm(new_ball_pos - goal_position)
                if np.isnan(new_goal_distance) or np.isinf(new_goal_distance):
                    print("Warning: Invalid goal distance, using previous value")
                    new_goal_distance = goal_distance
            except Exception as e:
                print(f"Error calculating distances: {e}")
                new_ball_distance = ball_distance
                new_goal_distance = goal_distance

            # Track closest distance this episode (for HER)
            if new_ball_distance < closest_distance:
                closest_distance = new_ball_distance

            reward = 0

            # === ENHANCED REWARD STRUCTURE WITH PHASED OBJECTIVES ===

            reward = 0.0

            # Phase 1: Initial Approach (>0.3m)
            if new_ball_distance > 0.3:
                # Aggressive distance reduction with velocity bonus
                distance_improvement = ball_distance - new_ball_distance
                reward += np.clip(
                    distance_improvement * 15.0, -0.8, 0.8
                )  # Stronger shaping

                # Bonus for maintaining speed during approach
                if distance_improvement > 0.02:  # Moving at least 2cm/step
                    reward += 0.2

            # Phase 2: Precision Control (0.12-0.3m)
            elif new_ball_distance > 0.12:  # Above grasp threshold
                # Fine-grained distance control
                distance_improvement = ball_distance - new_ball_distance
                reward += np.clip(
                    distance_improvement * 20.0, -1.0, 1.0
                )  # Very sensitive to progress

                # Strong gradient for getting closer
                proximity_factor = (
                    1.0 - (new_ball_distance - 0.12) / 0.18
                )  # Normalized 0-1
                reward += proximity_factor * 0.8  # Up to 0.8 reward for being close

                # Stability bonus - penalize jerky motion
                if abs(distance_improvement) < 0.01:  # Smooth motion
                    reward += 0.3

            # Phase 3: Final Approach & Grasp (<0.12m)
            else:
                # Ultra-precise control near ball
                distance_improvement = ball_distance - new_ball_distance
                reward += np.clip(
                    distance_improvement * 30.0, -1.5, 1.5
                )  # Extremely sensitive

                # Strong success gradient
                grasp_progress = 1.0 - (new_ball_distance / 0.12)  # 0-1 normalized
                reward += grasp_progress * 1.5  # Up to 1.5 reward

                # Bonus for achieving grasp
                if new_ball_distance < 0.06:  # Within grasp threshold
                    reward += 2.0  # Major milestone reward
                ee_velocity = np.linalg.norm(new_ee_position - ee_position) / 0.01
                if ee_velocity > 0.5:  # Too fast when close
                    # Normalized velocity penalty: clip to -0.2
                    velocity_penalty = -np.clip((ee_velocity - 0.5) * 0.4, 0, 0.2)
                    reward += velocity_penalty
                else:
                    # Small bonus for controlled approach
                    reward += 0.1

            # 4. Visual servoing rewards (normalized)
            if ball_visible:
                # Basic visibility bonus
                reward += 0.05

                # Visual centering (align ball with image center)
                center_distance = np.sqrt(
                    (ball_centroid_x - 0.5) ** 2 + (ball_centroid_y - 0.5) ** 2
                )
                centering_reward = (0.707 - center_distance) / 0.707
                reward += centering_reward * 0.2  # Scale to max +0.2

                # Size-based (closer = larger in view)
                size_reward = min(ball_pixel_count / 200.0, 1.0)
                reward += size_reward * 0.1  # Scale to max +0.1
            else:
                # Penalty for losing visual tracking
                if last_ball_visible:
                    reward -= 0.2

            # 5. Safety: Ground avoidance (normalized)
            if new_ee_position[2] < 0.05:
                reward -= 0.5
            elif new_ee_position[2] < 0.1:
                reward -= (0.1 - new_ee_position[2]) * 2.0  # Max -0.1

            # 6. Gripper control (normalized) - UR10e uses single joint
            new_joint_positions_raw = robot.get_joint_positions()
            if isinstance(new_joint_positions_raw, tuple):
                new_joint_positions = new_joint_positions_raw[0]
            else:
                new_joint_positions = new_joint_positions_raw

            # UR10e: gripper position at index 6 (0 = open, 0.04 = closed)
            if len(new_joint_positions) > 6:
                new_gripper_position = new_joint_positions[6]
            else:
                new_gripper_position = 0.0

            # Encourage gripper closing when near cube
            if new_ball_distance < 0.15:
                # For UR10e: gripper closes from 0 to 0.04
                gripper_close_action = 0.04 - new_gripper_position
                if gripper_close_action > 0:
                    proximity_factor = (0.15 - new_ball_distance) / 0.15
                    # Normalize to max +0.3
                    reward += gripper_close_action * 3.0 * proximity_factor * 0.1

            # === MILESTONE REWARDS (5-10x larger than per-step rewards) ===

            # MILESTONE 1: Successful grasp (5x per-step reward)
            # Cube is 5.15cm, gripper closes to 0.02 when grasping
            if new_ball_distance < 0.10 and new_gripper_position > 0.02:
                reward += 5.0  # Normalized milestone reward
                ball_grasped = True

                # Shaping after grasp: Move ball to goal (normalized)
                goal_improvement = goal_distance - new_goal_distance
                reward += np.clip(goal_improvement * 10.0, -0.5, 0.5)

                # MILESTONE 2: Successful delivery (10x per-step reward)
                if new_goal_distance < 0.15:
                    reward += 10.0  # Normalized final milestone reward
                    print(f"Episode {episode}: Ball delivered to goal at step {step}!")
                    break
            else:
                ball_grasped = False

            # Update agent (state WITHOUT cube position, using vision!)
            new_ball_grasped = new_ball_distance < 0.15 and new_gripper_position > 0.02
            next_state = np.concatenate(
                [
                    new_joint_positions,
                    [float(new_ball_grasped)],
                ]  # 12 joints  # 1 grasped
            )  # Total: 13

            # Update agent and track loss
            loss = agent.update(state, action, reward, next_state, image=rgb_image)

            episode_reward += reward

        # === HINDSIGHT EXPERIENCE REPLAY (HER) - Research-backed 2024 ===
        # If episode failed to grasp, relabel experiences as "reaching" successes
        # This turns sparse rewards into dense learning signal
        if not ball_grasped and len(agent.buffer) > 0 and closest_distance < 0.5:
            # HER insight: Even failed episodes have value if they got close
            # Relabel close approaches as successes for "reaching" goal

            # Sample last 50 experiences from this episode for relabeling
            relabel_count = min(50, step + 1)
            for i in range(relabel_count):
                if len(agent.buffer) > i:
                    # Get recent experience
                    idx = len(agent.buffer) - 1 - i
                    exp = agent.buffer[idx]

                    # Relabel reward: if got close, add bonus (hindsight success)
                    # This creates alternative "goals" - pretending closeness was the objective
                    s, a, r, ns, img = exp
                    # Bonus based on closest distance achieved
                    bonus_reward = max(
                        0, (0.5 - closest_distance) * 2.0
                    )  # Max +1.0 for 0m
                    if bonus_reward > 0.1:
                        # Store relabeled experience (augments learning from failures)
                        agent.buffer.append((s, a, r + bonus_reward, ns, img))

        # Track statistics
        agent.total_reward_history.append(episode_reward)
        agent.episode_count = episode + 1

        # Print progress with detailed metrics (every episode)
        avg_reward = (
            np.mean(agent.total_reward_history[-10:])
            if len(agent.total_reward_history) >= 10
            else np.mean(agent.total_reward_history)
        )
        avg_loss = (
            np.mean(agent.loss_history[-100:])
            if len(agent.loss_history) >= 100
            else (np.mean(agent.loss_history) if agent.loss_history else 0.0)
        )
        # Calculate curriculum difficulty for display
        curr_progress = min(agent.step_count / 500000.0, 1.0)
        curr_max_dist = 0.2 + (0.4 * curr_progress)

        print(
            f"Ep {episode:4d} | "
            f"R: {episode_reward:7.2f} | Avg: {avg_reward:7.2f} | "
            f"Loss: {avg_loss:.4f} | Steps: {agent.step_count:6d} | "
            f"Noise: {agent.noise_scale:.3f} | Buf: {len(agent.buffer):5d} | "
            f"Curriculum: {curr_progress*100:.0f}% (max_dist={curr_max_dist:.2f}m)"
        )

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

    # Explicit cleanup to avoid shutdown crash
    try:
        my_world.stop()
        my_world.clear()
    except:
        pass

    simulation_app.close()
