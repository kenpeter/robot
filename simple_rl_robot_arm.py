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
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

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

        # Decay exploration noise
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

# Add Franka robot arm
asset_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka")
robot = Articulation(prim_paths_expr="/World/Franka", name="franka_arm")

# Get end-effector link for position tracking
end_effector = RigidPrim("/World/Franka/panda_hand", name="end_effector")

# Add wrist camera for vision
from isaacsim.core.utils.prims import create_prim
from isaacsim.sensors.camera import Camera

# === OVERHEAD CAMERA (Eye-to-Hand) - GitHub best practice ===
# Fixed camera above workspace for global view (like visual-pushing-grasping)
camera_prim_path = "/World/OverheadCamera"
create_prim(camera_prim_path, "Camera")
overhead_camera = Camera(
    prim_path=camera_prim_path,
    resolution=(84, 84),  # Small resolution for faster processing
    position=np.array([0.0, 0.0, 1.2]),  # 1.2m above origin (bird's eye view)
    orientation=np.array(
        [0.7071, 0, 0, 0.7071]
    ),  # Look straight down (90deg rotation around X-axis)
)
overhead_camera.initialize()

# Add target sphere (ball to pick up) with physics
from pxr import UsdGeom, Gf, UsdLux, UsdPhysics, UsdShade

stage = my_world.stage
sphere_path = "/World/Target"
sphere = UsdGeom.Sphere.Define(stage, sphere_path)
sphere.GetRadiusAttr().Set(0.05)
sphere_translate = sphere.AddTranslateOp()
sphere_translate.Set(Gf.Vec3d(0.3, 0.3, 0.05))  # Ball on floor (z = radius)

# Add RED material to ball (so camera can detect it)
from pxr import Sdf

material_path = "/World/Looks/RedMaterial"
material = UsdShade.Material.Define(stage, material_path)
shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
    Gf.Vec3f(1.0, 0.0, 0.0)
)  # Pure red
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

# Bind material to ball
sphere_prim = stage.GetPrimAtPath(sphere_path)
binding_api = UsdShade.MaterialBindingAPI.Apply(sphere_prim)
binding_api.Bind(material)

# Add physics to ball (kinematic - won't fall)
rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(sphere_prim)
rigid_body_api.CreateKinematicEnabledAttr(True)  # Make it kinematic (stationary)
UsdPhysics.CollisionAPI.Apply(sphere_prim)
UsdPhysics.MassAPI.Apply(sphere_prim).CreateMassAttr(0.05)  # 50g ball

# Create RigidPrim for tracking
ball = RigidPrim(sphere_path, name="ball")

# Add goal location marker (green box)
goal_path = "/World/Goal"
goal_cube = UsdGeom.Cube.Define(stage, goal_path)
goal_cube.GetSizeAttr().Set(0.15)
goal_translate = goal_cube.AddTranslateOp()
goal_translate.Set(
    Gf.Vec3d(-0.3, 0.3, 0.075)
)  # Goal position on floor (half cube size)

# Initialize world
my_world.reset()

# Initialize robot articulation (CRITICAL - must be after world reset)
robot.initialize()
ball.initialize()
end_effector.initialize()

# RL Training parameters
MODEL_PATH = "rl_robot_arm_model.pth"

# Using Cartesian control: (x, y, z, gripper) - GitHub best practice for grasping
# Simpler 4D action space instead of 8D joint control
state_dim = 10  # 9 joints + 1 ball_grasped
action_dim = 4  # dx, dy, dz, gripper

agent = DiTAgent(state_dim=state_dim, action_dim=action_dim, use_vision=True)

# Try to load existing model
agent.load_model(MODEL_PATH)

num_episodes = 1000
# 500 step for reach grasp delivery
max_steps_per_episode = 500
goal_position = np.array([-0.3, 0.3, 0.05])  # Goal location on floor
save_interval = 10  # Save model every 10 episodes
vision_debug_saved = False  # Flag to save one camera image for debugging

# Create visual goal bucket - simple ring (no top, open for ball to drop in)
# Just a visual marker - ball detection is done by distance check in reward function
bucket_path = "/World/GoalBucket"
bucket_ring = UsdGeom.Cylinder.Define(stage, bucket_path)
bucket_ring.GetRadiusAttr().Set(0.10)  # 10cm radius
bucket_ring.GetHeightAttr().Set(0.15)  # 15cm tall
bucket_ring.GetAxisAttr().Set("Z")  # Upright
bucket_ring_translate = bucket_ring.AddTranslateOp()
bucket_ring_translate.Set(
    Gf.Vec3d(goal_position[0], goal_position[1], 0.075)
)  # Half height above floor

# Add BRIGHT GREEN semi-transparent material (can see through to know it's hollow)
bucket_material_path = "/World/Looks/GreenBucketMaterial"
bucket_material = UsdShade.Material.Define(stage, bucket_material_path)
bucket_shader = UsdShade.Shader.Define(stage, bucket_material_path + "/Shader")
bucket_shader.CreateIdAttr("UsdPreviewSurface")
bucket_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
    Gf.Vec3f(0.2, 1.0, 0.3)
)  # Very bright green
bucket_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
    0.4
)  # Semi-glossy plastic
bucket_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)  # Non-metallic
bucket_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
    0.5
)  # 50% transparent - clearly see it's open/hollow
bucket_material.CreateSurfaceOutput().ConnectToSource(
    bucket_shader.ConnectableAPI(), "surface"
)

# Bind material to bucket
bucket_prim = stage.GetPrimAtPath(bucket_path)
bucket_binding = UsdShade.MaterialBindingAPI.Apply(bucket_prim)
bucket_binding.Bind(bucket_material)

# No collision - this is just a visual marker, not a physical container
# Ball "delivery" is detected by goal_distance < 0.15 in reward function

print("Starting RL Training...")
print(f"Episodes: {num_episodes}, Max steps per episode: {max_steps_per_episode}")
print(f"Model will be saved to: {MODEL_PATH}")

try:
    for episode in range(agent.episode_count, agent.episode_count + num_episodes):
        # Reset arm to safe initial position (very conservative range)
        initial_pos = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.2, 0.0])  # Known safe pose
        # Add small random perturbation
        initial_pos += np.random.uniform(-0.1, 0.1, 7)
        robot.set_joint_positions(np.concatenate([initial_pos, [0.04, 0.04]]))

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
        # Randomize ball size for robust policy (sim-to-real transfer)
        ball_radius_variation = np.random.uniform(0.9, 1.1)  # ±10% size
        ball_z = 0.05 * ball_radius_variation  # Adjust height

        # Reset ball to random position on floor using RigidPrim
        target_position = np.array(
            [
                np.random.uniform(
                    min_distance, max_distance
                ),  # x: curriculum-based distance
                np.random.uniform(-0.3, 0.3),  # y: left/right
                ball_z,  # z: randomized based on size
            ],
            dtype=np.float32,
        ).flatten()  # Ensure 1D
        ball.set_world_poses(positions=np.array([target_position]))

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
            joint_positions = robot.get_joint_positions()[0]
            ball_pos, _ = ball.get_world_poses()
            ball_pos = np.array(ball_pos, dtype=np.float32).flatten()
            ee_position, _ = end_effector.get_world_poses()
            ee_position = np.array(ee_position, dtype=np.float32).flatten()

            # Calculate distances (for reward only, not in state!)
            ball_distance = np.linalg.norm(ee_position - ball_pos)
            goal_distance = np.linalg.norm(ball_pos - goal_position)

            # Gripper width
            gripper_width = joint_positions[7] + joint_positions[8]

            # Check if ball is grasped
            if ball_distance < 0.08 and gripper_width < 0.02:
                ball_grasped = True

            # Build state: joints + ball_grasped (NO ball position!)
            state = np.concatenate(
                [joint_positions, [float(ball_grasped)]]  # 9  # 1
            )  # Total: 10

            # Get action from agent WITH VISION
            action = agent.get_action(state, image=rgb_image)

            # Vision debugging: Check if ball is visible in camera
            ball_visible = False
            ball_pixel_count = 0
            ball_centroid_x = 0.0
            ball_centroid_y = 0.0
            if (
                rgb_image is not None
                and rgb_image.size > 0
                and not np.all(rgb_image == 0)
            ):
                # With overhead camera, detect RED ball
                # Look for bright red pixels: high R, low G, low B
                red_mask = (
                    (rgb_image[:, :, 0] > 200)  # High red channel
                    & (rgb_image[:, :, 1] < 100)  # Low green
                    & (rgb_image[:, :, 2] < 100)  # Low blue
                )
                ball_pixel_count = np.sum(red_mask)

                # Overhead view: ball should be 100-800 pixels (small sphere from above)
                ball_visible = (ball_pixel_count > 100) and (ball_pixel_count < 800)

                # Calculate ball centroid in image (for visual servoing reward)
                if ball_visible:
                    y_coords, x_coords = np.where(red_mask)
                    if len(x_coords) > 0:
                        ball_centroid_x = np.mean(x_coords) / 84.0  # Normalize to 0-1
                        ball_centroid_y = np.mean(y_coords) / 84.0  # Normalize to 0-1

                # Temporal smoothing: If ball was visible last frame and close in distance, keep visible
                if not ball_visible and last_ball_visible and ball_distance < 0.3:
                    ball_visible = True  # Assume still visible (avoid flicker)

                # Save one debug image (first valid frame)
                if not vision_debug_saved and ball_pixel_count > 0:
                    try:
                        import cv2

                        debug_img = rgb_image.copy()
                        # Draw mask overlay
                        debug_img[red_mask] = [
                            0,
                            255,
                            0,
                        ]  # Highlight detected red pixels in green
                        cv2.imwrite(
                            "debug_camera_view.png",
                            cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
                        )
                        cv2.imwrite(
                            "debug_camera_mask.png",
                            cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR),
                        )
                        print(f"\n=== VISION DEBUG: Saved debug images ===")
                        print(
                            f"    Red pixels: {ball_pixel_count}, Ball visible: {ball_visible}"
                        )
                        vision_debug_saved = True
                    except:
                        pass  # OpenCV not available, skip image save

            # Update last visibility for next frame
            last_ball_visible = ball_visible

            # Debug: Print action and vision info every 50 steps
            if step % 50 == 0:
                print(
                    f"Step {step}: Action range [{action.min():.3f}, {action.max():.3f}], Mean: {action.mean():.3f}"
                )
                print(f"  Ball dist: {ball_distance:.3f}, EE: {ee_position}")
                print(
                    f"  Vision: Red pixels={ball_pixel_count}, Ball visible={ball_visible}"
                )

            # === CARTESIAN ACTION EXECUTION ===
            # Action = [dx, dy, dz, gripper] - 4D end-effector control
            new_positions = joint_positions.copy()

            # Extract Cartesian delta from action
            delta_pos = action[:3] * 0.02  # Cartesian movements (2cm max per step)

            # Calculate target end-effector position
            target_position = ee_position + delta_pos
            target_position = np.clip(
                target_position, [-0.6, -0.6, 0.05], [0.8, 0.6, 1.0]
            )

            # Simplified differential IK (proportional control toward target)
            # Maps desired end-effector motion to joint space
            delta_ee = (target_position - ee_position) * 3.0

            # Distribute motion across joints (simplified Jacobian approximation)
            new_positions[:3] += delta_ee * 0.5  # First 3 joints control position

            # Repeat delta for remaining 4 joints (orientation control)
            delta_repeat = np.tile(delta_ee, 2)[:4]  # Repeat and take first 4 elements
            new_positions[3:7] += delta_repeat * 0.3

            new_positions[:7] = np.clip(new_positions[:7], -2.8, 2.8)

            # Gripper control (action[3])
            gripper_action = np.clip(action[3], -1.0, 1.0)
            new_positions[7:9] = np.clip(
                new_positions[7:9] + gripper_action * 0.005, 0.0, 0.04
            )

            # Apply the positions (position control mode)
            robot.set_joint_positions(new_positions)

            # Multi-stage reward function (step happens at start of next loop)
            new_ball_pos, _ = ball.get_world_poses()
            new_ball_pos = np.array(new_ball_pos, dtype=np.float32).flatten()
            new_ee_position, _ = end_effector.get_world_poses()
            new_ee_position = np.array(new_ee_position, dtype=np.float32).flatten()
            new_ball_distance = np.linalg.norm(new_ee_position - new_ball_pos)
            new_goal_distance = np.linalg.norm(new_ball_pos - goal_position)

            # Track closest distance this episode (for HER)
            if new_ball_distance < closest_distance:
                closest_distance = new_ball_distance

            reward = 0

            # === Normalized Reward Structure ===
            # Shaped rewards guide learning, but FINAL GOALS (grasp + delivery) have biggest rewards

            # closer more reward: closer -> more reward

            # Shaping: Getting closer to ball (linear - guides early learning)
            distance_improvement = ball_distance - new_ball_distance
            reward += distance_improvement * 10.0  # Dense signal for learning

            # closer more reward: closer -> more reward

            # EXPONENTIAL proximity reward (research-backed: reward grows as distance shrinks)
            # This creates a strong gradient near the ball, solving sparse reward problem
            if new_ball_distance < 0.5:
                # Exponential: r = e^(-k*distance) scaled to 0-10 range
                # At 0.5m: ~0, At 0.1m: ~5, At 0.05m: ~7, At 0.01m: ~9
                exponential_reward = 10.0 * (
                    1.0 - np.exp(-5.0 * (0.5 - new_ball_distance))
                )
                reward += exponential_reward

            # === LYAPUNOV-BASED PHYSICS-INFORMED REWARD ===

            # closer slow down: closer -> slow down

            # Penalize high velocity when close to ball (encourages stable approach)
            # Lyapunov function: V = distance^2 + velocity^2
            # Stable if dV/dt < 0 (distance decreasing faster than velocity increasing)
            if new_ball_distance < 0.3:
                # Estimate end-effector velocity from position change
                ee_velocity = (
                    np.linalg.norm(new_ee_position - ee_position) / 0.01
                )  # dt ~0.01s per step
                # Penalize high velocity near target (encourages smooth, stable approach)
                if ee_velocity > 0.5:  # Too fast when close
                    velocity_penalty = -(ee_velocity - 0.5) * 0.5  # Max penalty ~-1.0
                    reward += velocity_penalty
                else:
                    # Bonus for slow, controlled approach
                    reward += 0.2

            # === RESEARCH-BACKED VISUAL SERVOING REWARDS ===
            # From 2024 papers: reward visual alignment, not just visibility
            if ball_visible:
                # 1. Basic visibility reward (small)
                reward += 0.1

                # 2. Visual centering reward (align ball with image center)
                # Image center is (0.5, 0.5), reward when ball is centered
                center_distance = np.sqrt(
                    (ball_centroid_x - 0.5) ** 2 + (ball_centroid_y - 0.5) ** 2
                )
                # Max distance from center is ~0.707 (corner to center)
                # Reward ranges from 0 (corner) to +1.0 (perfectly centered)
                centering_reward = (0.707 - center_distance) / 0.707
                reward += centering_reward * 0.5  # Scale to max +0.5

                # 3. Size-based reward (closer objects appear larger)
                # More pixels = closer to ball (correlates with distance)
                # Encourage keeping ball in view AND getting closer
                size_reward = min(ball_pixel_count / 1000.0, 0.5)  # Max +0.5
                reward += size_reward * 0.3  # Scale to max +0.15
            else:
                # Penalty for losing visual tracking (important for visual servoing)
                if last_ball_visible:
                    reward -= 0.3  # Lost tracking

            # Penalty: Ground avoidance
            if new_ee_position[2] < 0.05:
                reward -= 1.0
            elif new_ee_position[2] < 0.1:
                reward -= (0.1 - new_ee_position[2]) * 2.0  # Max -0.1

            # === GRIPPER CONTROL SHAPING (Research-backed 2024) ===
            new_joint_positions = robot.get_joint_positions()[0]
            new_gripper_width = new_joint_positions[7] + new_joint_positions[8]

            # Encourage gripper closing when near ball (learns closing behavior)
            if new_ball_distance < 0.15:
                # Reward for closing gripper when close to ball
                gripper_close_action = (
                    0.08 - new_gripper_width
                )  # How much gripper closed
                if gripper_close_action > 0:
                    # Scaled reward: closer to ball = more reward for closing
                    proximity_factor = (0.15 - new_ball_distance) / 0.15  # 0-1
                    reward += gripper_close_action * 5.0 * proximity_factor  # Max ~+2.5

            # === FINAL GOAL 1: GRASP (Big reward!) ===
            if new_ball_distance < 0.08 and new_gripper_width < 0.02:
                reward += 50.0  # MAJOR reward for successful grasp!
                ball_grasped = True

                # Shaping after grasp: Move ball to goal
                goal_improvement = goal_distance - new_goal_distance
                reward += goal_improvement * 10.0

                # === FINAL GOAL 2: DELIVERY (Biggest reward!) ===
                if new_goal_distance < 0.15:
                    reward += 100.0  # MASSIVE reward for completing task!
                    print(f"Episode {episode}: Ball delivered to goal at step {step}!")
                    break
            else:
                ball_grasped = False

            # Update agent (state WITHOUT ball position, using vision!)
            new_ball_grasped = new_ball_distance < 0.08 and new_gripper_width < 0.02
            next_state = np.concatenate(
                [new_joint_positions, [float(new_ball_grasped)]]  # 9  # 1
            )  # Total: 10

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
