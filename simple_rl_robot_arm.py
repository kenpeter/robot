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
        self.Wg = nn.Linear(d_model, d_model, bias=False)  # For alpha (sigmoid -> [0,1])
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
        g = torch.sigmoid(self.Wg(x_norm)).view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # Diag(alpha)
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

            S_t = state + beta_t * torch.einsum('b h d, b h e -> b h d e', k_t, v_t)  # Simplified update
            S_t = (torch.eye(self.d_k, device=x.device)[None, None] - beta_t * torch.einsum('b h d, b h e -> b h d e', k_t, k_t)) @ (alpha_t.unsqueeze(-1) * S_t)
            o_t = torch.einsum('b h d, b h d e -> b h e', q_t, S_t)
            outputs.append(o_t)
            new_states.append(S_t)
            state = S_t  # Update for next

        o = torch.stack(outputs, dim=2).transpose(1, 2).contiguous().view(B, T, D)  # [B,T,D]
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
        scale_msa, gate_msa, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(4, dim=-1)
        )

        # Kimi Linear Attention with adaptive modulation
        x_scaled = x * (1 + scale_msa.unsqueeze(1))
        attn_out, new_kimi_state = self.kimi_attn(x_scaled, kimi_state)
        x = x + gate_msa.unsqueeze(1) * (attn_out - x_scaled)  # Residual already in kimi_attn

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

        # Simple convolutional layers
        self.conv = nn.Sequential(
            # 84x84x3 -> 42x42x32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # 42x42x32 -> 21x21x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # 21x21x64 -> 10x10x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            # 10x10x128 -> 5x5x128
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        # Adaptive pooling to ensure consistent output size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Always output 4x4

        # Calculate flattened size: 4x4x128 = 2048
        self.flatten_size = 4 * 4 * 128

        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
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
        self, state_dim, action_dim, hidden_dim=128, num_layers=4, num_heads=4, use_vision=True
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

        # Diffusion hyperparameters (simplified for stability)
        self.num_diffusion_steps = 5  # Reduced from 10 for faster inference
        self.beta_start = 0.0001
        self.beta_end = 0.02

        # Create diffusion schedule with careful numerical stability
        self.betas = torch.linspace(
            self.beta_start, self.beta_end, self.num_diffusion_steps, dtype=torch.float32
        ).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Clamp to avoid numerical issues
        self.alphas_cumprod = torch.clamp(self.alphas_cumprod, min=1e-8, max=1.0)

        # Initialize DiT model with vision
        self.model = DiffusionTransformer(
            state_dim, action_dim, hidden_dim=128, num_layers=3, num_heads=4, use_vision=use_vision
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        # Experience replay buffer
        self.buffer = []
        self.buffer_size = 10000
        self.batch_size = 64

        # Training stats
        self.episode_count = 0
        self.total_reward_history = []
        self.noise_scale = 0.3  # Exploration noise

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
                image_tensor = torch.FloatTensor(image).unsqueeze(0).to(self.device) / 255.0

            # Start from random noise
            action = torch.randn(1, self.action_dim, dtype=torch.float32).to(self.device) * 0.5

            # Reverse diffusion process (DDPM sampling)
            for t in reversed(range(self.num_diffusion_steps)):
                timestep = torch.FloatTensor([[t / self.num_diffusion_steps]]).to(self.device)

                # Predict noise
                predicted_noise = self.model(action, state_tensor, timestep, image_tensor)

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
        # Add to buffer
        self.buffer.append((state, action, reward, next_state, image))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # Train if enough samples
        if len(self.buffer) < self.batch_size:
            return

        # Sample batch
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        # Convert to tensors
        states = torch.FloatTensor(np.array([s for s, a, r, ns, img in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([a for s, a, r, ns, img in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([r for s, a, r, ns, img in batch])).to(self.device)

        # Process images if using vision
        images_tensor = None
        if self.use_vision:
            images = [img for s, a, r, ns, img in batch if img is not None]
            if images:
                images_tensor = torch.FloatTensor(np.stack(images)).to(self.device) / 255.0

        # Diffusion training
        self.model.train()

        # Sample random timesteps
        t = torch.randint(0, self.num_diffusion_steps, (self.batch_size,), dtype=torch.long).to(self.device)

        # Add noise to actions (forward diffusion process)
        noise = torch.randn_like(actions)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1)

        # x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        noisy_actions = torch.sqrt(alpha_cumprod_t + 1e-8) * actions + torch.sqrt(1.0 - alpha_cumprod_t + 1e-8) * noise

        # Predict noise
        timesteps = (t.float() / self.num_diffusion_steps).view(-1, 1)
        predicted_noise = self.model(noisy_actions, states, timesteps, images_tensor)

        # Compute loss (MSE between predicted and actual noise)
        loss = F.mse_loss(predicted_noise, noise)

        # Add reward weighting (prioritize good experiences)
        reward_weights = torch.sigmoid(rewards / 10.0)
        weighted_loss = (loss * reward_weights.mean())

        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Decay exploration noise
        self.noise_scale = max(0.05, self.noise_scale * 0.999)

    def save_model(self, filepath):
        """Save agent state to file"""
        model_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode_count": self.episode_count,
            "total_reward_history": self.total_reward_history,
            "buffer": self.buffer[-1000:],  # Save last 1000 experiences
            "noise_scale": self.noise_scale,
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
        self.buffer = model_data.get("buffer", [])
        self.noise_scale = model_data.get("noise_scale", 0.3)

        print(f"Model loaded from {filepath}")
        print(
            f"Resuming from episode {self.episode_count}, buffer size: {len(self.buffer)}"
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

camera_prim_path = "/World/Franka/panda_hand/camera"
create_prim(camera_prim_path, "Camera")
wrist_camera = Camera(
    prim_path=camera_prim_path,
    resolution=(84, 84),  # Small resolution for faster processing
    position=np.array([0.05, 0.0, 0.05]),  # Offset from end-effector
    orientation=np.array([0.7071, 0, 0.7071, 0])  # Point forward
)
wrist_camera.initialize()

# Add target sphere (ball to pick up) with physics
from pxr import UsdGeom, Gf, UsdLux, UsdPhysics

stage = my_world.stage
sphere_path = "/World/Target"
sphere = UsdGeom.Sphere.Define(stage, sphere_path)
sphere.GetRadiusAttr().Set(0.05)
sphere_translate = sphere.AddTranslateOp()
sphere_translate.Set(Gf.Vec3d(0.3, 0.3, 0.5))

# Add physics to ball
sphere_prim = stage.GetPrimAtPath(sphere_path)
UsdPhysics.RigidBodyAPI.Apply(sphere_prim)
UsdPhysics.CollisionAPI.Apply(sphere_prim)
UsdPhysics.MassAPI.Apply(sphere_prim).CreateMassAttr(0.05)  # 50g ball

# Create RigidPrim for tracking
ball = RigidPrim(sphere_path, name="ball")

# Add goal location marker (green box)
goal_path = "/World/Goal"
goal_cube = UsdGeom.Cube.Define(stage, goal_path)
goal_cube.GetSizeAttr().Set(0.15)
goal_translate = goal_cube.AddTranslateOp()
goal_translate.Set(Gf.Vec3d(-0.3, 0.3, 0.3))  # Goal position

# Initialize world
my_world.reset()

# Initialize robot articulation (CRITICAL - must be after world reset)
robot.initialize()
ball.initialize()
end_effector.initialize()

# RL Training parameters
MODEL_PATH = "rl_robot_arm_model.pth"
state_dim = 10  # 9 joints + 1 ball_grasped (NO ball position, using vision instead!)
agent = DiTAgent(
    state_dim=state_dim, action_dim=8, use_vision=True
)  # 7 arm joints + 1 gripper

# Try to load existing model
agent.load_model(MODEL_PATH)

num_episodes = 1000
max_steps_per_episode = 500
goal_position = np.array([-0.3, 0.3, 0.3])  # Fixed goal location
save_interval = 10  # Save model every 10 episodes

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

        # Reset ball to random position using RigidPrim
        target_position = np.array([
            np.random.uniform(0.2, 0.5),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(0.3, 0.7),
        ], dtype=np.float32).flatten()  # Ensure 1D
        ball.set_world_poses(positions=np.array([target_position]))

        episode_reward = 0
        ball_grasped = False

        # Step simulation once to initialize camera
        my_world.step(render=True)

        for step in range(max_steps_per_episode):
            # Capture camera image
            wrist_camera.get_current_frame()
            rgba_data = wrist_camera.get_rgba()

            # Check if data is valid
            if rgba_data is None or rgba_data.size == 0:
                # Use dummy image if camera not ready
                rgb_image = np.zeros((84, 84, 3), dtype=np.uint8)
            else:
                # Reshape from flat array to image if needed
                if len(rgba_data.shape) == 1:
                    rgba_data = rgba_data.reshape(84, 84, 4)
                rgb_image = rgba_data[:, :, :3].astype(np.uint8)  # Get RGB only (84x84x3)

            # Get robot state
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
            state = np.concatenate([
                joint_positions,                    # 9
                [float(ball_grasped)]              # 1
            ])  # Total: 10

            # Get action from agent WITH VISION
            action = agent.get_action(state, image=rgb_image)

            # Debug: Print action every 50 steps
            if step % 50 == 0:
                print(f"Step {step}: Action range [{action.min():.3f}, {action.max():.3f}], Mean: {action.mean():.3f}")
                print(f"  Ball distance: {ball_distance:.3f}, EE pos: {ee_position}")

            # Apply action: first 7 = arm joints, last 1 = gripper
            # Use moderate action scaling for smooth but effective movement
            new_positions = joint_positions.copy()
            new_positions[:7] += action[:7] * 0.05  # Balanced scaling
            new_positions[:7] = np.clip(new_positions[:7], -2.8, 2.8)  # Franka joint limits

            # Gripper control: positive = close, negative = open
            gripper_action = np.clip(action[7], -1.0, 1.0)
            new_positions[7:9] = np.clip(new_positions[7:9] + gripper_action * 0.005, 0.0, 0.04)  # Slower gripper

            # Apply the positions (position control mode)
            robot.set_joint_positions(new_positions)

            # Step simulation
            my_world.step(render=True)

            # Multi-stage reward function
            new_ball_pos, _ = ball.get_world_poses()
            new_ball_pos = np.array(new_ball_pos, dtype=np.float32).flatten()
            new_ee_position, _ = end_effector.get_world_poses()
            new_ee_position = np.array(new_ee_position, dtype=np.float32).flatten()
            new_ball_distance = np.linalg.norm(new_ee_position - new_ball_pos)
            new_goal_distance = np.linalg.norm(new_ball_pos - goal_position)

            reward = 0

            # Stage 1: Reach the ball
            reward -= new_ball_distance * 2.0

            # Stage 2: Grasp the ball
            new_joint_positions = robot.get_joint_positions()[0]
            new_gripper_width = new_joint_positions[7] + new_joint_positions[8]
            if new_ball_distance < 0.08 and new_gripper_width < 0.02:
                reward += 5.0  # Grasping bonus
                ball_grasped = True

                # Stage 3: Move ball to goal
                reward -= new_goal_distance * 3.0

                # Stage 4: Success - ball at goal
                if new_goal_distance < 0.15:
                    reward += 20.0
                    print(f"Episode {episode}: Ball delivered to goal at step {step}!")
                    break
            else:
                ball_grasped = False

            # Update agent (state WITHOUT ball position, using vision!)
            new_ball_grasped = new_ball_distance < 0.08 and new_gripper_width < 0.02
            next_state = np.concatenate([
                new_joint_positions,              # 9
                [float(new_ball_grasped)]        # 1
            ])  # Total: 10

            agent.update(state, action, reward, next_state, image=rgb_image)

            episode_reward += reward

        # Track statistics
        agent.total_reward_history.append(episode_reward)
        agent.episode_count = episode + 1

        # Print progress
        if episode % 10 == 0:
            avg_reward = (
                np.mean(agent.total_reward_history[-10:])
                if len(agent.total_reward_history) >= 10
                else np.mean(agent.total_reward_history)
            )
            print(
                f"Episode {episode}/{agent.episode_count + num_episodes - 1}, Reward: {episode_reward:.2f}, Avg(10): {avg_reward:.2f}, Noise: {agent.noise_scale:.3f}"
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
    simulation_app.close()
