# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ONLINE RL Training - Robot learns from online interactions
The robot learns to grasp through online experience collection with reward weighting.
Uses Conditional MLP for diffusion (no attention overhead - simple and efficient).
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
import cv2  # For video recording
from collections import deque
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import re
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

# Using Conditional MLP for Diffusion (no attention - honest and fast!)
print("=" * 60)
print("DIFFUSION MODEL: Conditional MLP (no transformer)")
print("Why: Single action vector doesn't need attention")
print("=" * 60)


# Conditional MLP for diffusion - simpler and more honest than fake transformer
class ConditionalDiffusionMLP(nn.Module):
    """Conditional MLP for diffusion-based action generation (no attention waste!)"""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        num_layers=4,
        use_vision=False,  # Disabled vision
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.use_vision = False  # Force disable vision

        # Input: concatenate [noisy_action, state, timestep]
        input_dim = action_dim + state_dim + 1

        # Build MLP layers
        layers = []

        # First layer: input → hidden
        layers.extend(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
            ]
        )

        # Middle layers with residual connections
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                ]
            )

        # Final layer: hidden → action (noise prediction)
        layers.extend(
            [
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, action_dim),
            ]
        )

        self.network = nn.Sequential(*layers)

        print(f"ConditionalDiffusionMLP initialized:")
        print(
            f"  Input dim: {input_dim} (state={state_dim} + action={action_dim} + timestep=1)"
        )
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num layers: {num_layers}")
        print(f"  Output dim: {action_dim}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, noisy_action, state, timestep, image=None):
        """
        noisy_action: [batch, action_dim] - noisy action at timestep t
        state: [batch, state_dim] - robot proprioceptive state
        timestep: [batch, 1] - diffusion timestep (0 to 1)
        image: IGNORED - no vision

        Returns: predicted noise [batch, action_dim]
        """
        # Simply concatenate all inputs
        x = torch.cat(
            [noisy_action, state, timestep], dim=-1
        )  # [batch, state_dim + action_dim + 1]

        # Pass through MLP
        noise_pred = self.network(x)  # [batch, action_dim]

        return noise_pred


# === VLM-BASED REWARD SHAPING ===
class VLMRewardHelper:
    """Uses Qwen3-VL to provide vision-based reward feedback"""

    def __init__(self, model_path):
        print(f"[VLM] Loading vision-language model from {model_path}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and processor (auto-detect Qwen3-VL model type)
        # Load directly to device to avoid meta tensor issues
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map=self.device  # Load directly to device
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.prompt = """You are analyzing a robot grasping task.
The robot (UR10 arm) needs to reach and grasp the red cube.

Score the robot's performance from 0-10:
- 0-2: Robot far from cube (> 1m away)
- 3-4: Robot approaching cube (0.5-1m)
- 5-6: Robot near cube (< 0.5m)
- 7-8: Robot touching/grasping cube
- 9-10: Robot successfully grasped cube

Return ONLY a single number 0-10."""

        print(f"[VLM] Model loaded on {self.device}")

    def get_reward(self, image_bgr, ee_pos, ball_pos, grasped, target_pos, joint_positions, action=None):
        """Get VLM-based reward from camera image + full state info + action"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Calculate distance
        distance_to_cube = np.linalg.norm(ee_pos - ball_pos)

        # SIMPLIFIED prompt - SmolVLM can't handle long text
        state_info = f"Distance: {distance_to_cube:.2f}m. Grasped: {'Yes' if grasped > 0.5 else 'No'}."

        # Simplified prompt that fits in SmolVLM's context
        simple_prompt = f"<image>Rate robot grasping red cube (0-10). {state_info}"

        full_prompt = simple_prompt

        # LOG THE PROMPT
        print(f"\n[VLM PROMPT] → {full_prompt}")
        print(f"[VLM STATE] EE: {ee_pos}, Ball: {ball_pos}, Dist: {distance_to_cube:.3f}m, Grasped: {grasped:.2f}")

        # SmolVLM uses simpler API - direct text and image input
        inputs = self.processor(
            text=full_prompt,
            images=pil_image,
            return_tensors="pt"
        ).to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1
            )

        # Decode response
        response = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # LOG THE RAW RESPONSE
        print(f"[VLM RAW RESPONSE] ← {repr(response)}")

        # Extract score (look for number 0-10)
        try:
            # Try to find a number in the response
            numbers = re.findall(r'\b([0-9]|10)\b', response)
            if numbers:
                score = float(numbers[0])
                print(f"[VLM REWARD] ✓ Score: {score}/10")
                return score
            else:
                print(f"[VLM WARNING] ✗ Could not parse score from: {response}")
                return 0.0
        except Exception as e:
            print(f"[VLM ERROR] {e}")
            return 0.0


# rl agent using DiT
class DiTAgent:
    """RL Agent using Diffusion Transformer for action generation"""

    # self, state dim, action dim, NO vision, device cuda
    def __init__(
        self,
        state_dim,
        action_dim,
        use_vision=False,  # Disabled
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

        # Initialize ConditionalDiffusionMLP (simpler than fake transformer)
        self.model = ConditionalDiffusionMLP(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,  # Larger hidden dim since we removed attention overhead
            num_layers=4,
            use_vision=False,  # No vision
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
        """Store experience and train the diffusion model with reward weighting"""
        # No vision - ignore image
        experience = (state, action, reward, next_state, None)

        # Add to buffer
        self.buffer.append(experience)

        # Train if enough samples
        if len(self.buffer) < self.batch_size:
            return

        # FILTER: Only train on experiences with reward > 0 (positive experiences only!)
        positive_indices = [
            i for i in range(len(self.buffer)) if self.buffer[i][2] > 0.0
        ]

        # If not enough positive experiences, use all experiences (fall back)
        if len(positive_indices) < self.batch_size:
            # Print only every 100 steps to avoid spam
            if self.step_count % 100 == 0:
                print(
                    f"[INFO] Only {len(positive_indices)} positive experiences, using all {len(self.buffer)} experiences"
                )
            indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        else:
            # Print only every 1000 steps when using positive-only filter
            if self.step_count % 1000 == 0:
                print(
                    f"[INFO] Training on {len(positive_indices)} positive experiences only!"
                )
            indices = np.random.choice(positive_indices, self.batch_size, replace=False)

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

        # No vision - images_tensor is always None
        images_tensor = None

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

        # Reward-based weighting: high reward = more weight
        # Normalize rewards to [0, 1], then scale by 2, then apply softmax
        reward_normalized = (rewards - rewards.min()) / (
            rewards.max() - rewards.min() + 1e-8
        )
        reward_weights = F.softmax(reward_normalized * 2.0, dim=0) * len(
            rewards
        )  # Scale by batch size
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

        # DEBUG: Print diffusion MLP stats every 100 steps
        if self.step_count % 100 == 0:
            print(f"\n[DIFFUSION MLP DEBUG - Step {self.step_count}]")
            print(f"  Actual noise magnitude: {noise.abs().mean().item():.4f}")
            print(
                f"  Predicted noise magnitude: {predicted_noise.abs().mean().item():.4f}"
            )
            print(
                f"  Noise prediction error (MAE): {(noise - predicted_noise).abs().mean().item():.4f} ← Should DECREASE"
            )
            print(f"  Loss (weighted): {weighted_loss.item():.6f} ← Should DECREASE")
            print(
                f"  Reward range: [{rewards.min().item():.2f}, {rewards.max().item():.2f}]"
            )
            print(
                f"  Alpha_cumprod range: [{alpha_cumprod_t.min().item():.4f}, {alpha_cumprod_t.max().item():.4f}]"
            )
            print(f"  Exploration noise scale: {self.noise_scale:.3f}")

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
from pxr import Gf, UsdLux

# === OVERHEAD CAMERA (Eye-to-Hand) - For RL policy (small, fast) ===
camera_prim_path = "/World/OverheadCamera"
create_prim(camera_prim_path, "Camera")
eye = Gf.Vec3d(0.1, 0.0, 1.2)
target = Gf.Vec3d(0.1, 0.0, 0.0)
set_camera_view(eye=eye, target=target, camera_prim_path=camera_prim_path)
camera_prim = my_world.stage.GetPrimAtPath(camera_prim_path)
camera_prim.GetAttribute("horizontalAperture").Set(80.0)
camera_prim.GetAttribute("verticalAperture").Set(80.0)
overhead_camera = Camera(
    prim_path=camera_prim_path,
    resolution=(84, 84),  # Small for fast RL training
)
overhead_camera.initialize()

# === SIDE VIEW CAMERA - For video recording (high-res, better angle) ===
side_camera_prim_path = "/World/SideCamera"
create_prim(side_camera_prim_path, "Camera")
# Position camera MUCH farther back to see entire scene
eye_side = Gf.Vec3d(3.5, 3.5, 2.5)  # Very far back and high - full scene view
target_side = Gf.Vec3d(0.0, 0.0, 0.5)  # Look at center of robot
set_camera_view(
    eye=eye_side, target=target_side, camera_prim_path=side_camera_prim_path
)
camera_prim_side = my_world.stage.GetPrimAtPath(side_camera_prim_path)
camera_prim_side.GetAttribute("horizontalAperture").Set(50.0)  # Very wide FOV
camera_prim_side.GetAttribute("verticalAperture").Set(37.5)  # 16:9 aspect ratio
side_camera = Camera(
    prim_path=side_camera_prim_path,
    resolution=(1280, 720),  # HD resolution for clear video
)
side_camera.initialize()
print("✓ Cameras initialized: Overhead (84x84) for RL, Side (1280x720) for video")

# === RED CUBE SETUP - FIXED POSITION ===
# Add red cube with FIXED position for testing diffusion transformer
from isaacsim.core.api.objects import DynamicCuboid

# FIXED cube position
cube_size = 0.0515  # 5.15cm cube
FIXED_CUBE_X = 0.5  # FIXED
FIXED_CUBE_Y = 0.2  # FIXED
FIXED_CUBE_Z = cube_size / 2.0  # On ground plane

print(f"\n=== RED CUBE SETUP - FIXED POSITION ===")
print(
    f"Cube FIXED position: [{FIXED_CUBE_X:.3f}, {FIXED_CUBE_Y:.3f}, {FIXED_CUBE_Z:.3f}]"
)

cube = DynamicCuboid(
    name="red_cube",
    position=np.array([FIXED_CUBE_X, FIXED_CUBE_Y, FIXED_CUBE_Z]),
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

# FIXED STATE DIMENSION: Now includes spatial awareness!
# Old (broken): 13D = 12 joints + 1 grasped (model was blind!)
# New (fixed): 22D = 12 joints + 1 grasped + 3 cube_pos + 3 ee_pos + 3 target_pos
state_dim = 22  # 12 joints + 1 grasped + 3 cube + 3 gripper + 3 target
action_dim = 4  # dx, dy, dz, gripper

agent = DiTAgent(state_dim=state_dim, action_dim=action_dim, use_vision=False)

# Try to load existing model
agent.load_model(MODEL_PATH)

# === ONLINE TRAINING PARAMETERS ===
print("\n" + "=" * 70)
print(" STARTING ONLINE RL TRAINING")
print("=" * 70)
print("Robot will learn from online interactions")
print("High-reward experiences will be weighted more during training")
print(f"Model will be saved to: {MODEL_PATH}\n")

# Training parameters
MAX_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 1000
SAVE_INTERVAL = 10  # Save model every 10 episodes
VIDEO_INTERVAL = 20  # Record video every 20 episodes
VIDEO_PATH = "/home/kenpeter/work/robot/training_video.avi"  # Single file, overwrite

# Start simulation
print("[DEBUG] Starting simulation timeline...")
my_world.play()
print(f"[DEBUG] World is playing: {my_world.is_playing()}")
print("✓ Simulation timeline started\n")


# FIXED target position for robot end-effector
FIXED_TARGET_X = 0.5
FIXED_TARGET_Y = 0.2
FIXED_TARGET_Z = 0.3  # Above the cube

# === INITIALIZE VLM REWARD HELPER (REQUIRED) ===
VLM_REWARD_INTERVAL = 10  # Get VLM reward every N steps (to save compute)

print("=" * 70)
print("LOADING VISION-LANGUAGE MODEL FOR REWARD COMPUTATION")
print("=" * 70)

# Load VLM - using lightweight SmolVLM-256M (faster than Qwen3-VL-2B)
vlm_helper = VLMRewardHelper(
    model_path="/home/kenpeter/.cache/huggingface/hub/SmolVLM-256M-Instruct"
)
print(f"[VLM] ✓ Model loaded successfully!")
print(f"[VLM] Will compute VLM reward every {VLM_REWARD_INTERVAL} steps")
print(f"[VLM] PURE VISION-BASED REWARDS - NO HAND-CRAFTED RULES!")
print("=" * 70)


# Helper function to reset environment
def reset_environment():
    """Reset robot and cube to FIXED positions"""
    # Reset robot to initial position
    initial_pos = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
    gripper_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    robot.set_joint_positions(np.concatenate([initial_pos, gripper_open]))

    # Reset cube to FIXED position (no randomization)
    ball.set_world_pose(position=np.array([FIXED_CUBE_X, FIXED_CUBE_Y, FIXED_CUBE_Z]))

    # Stabilize
    for _ in range(10):
        my_world.step(render=False)

    return FIXED_CUBE_X, FIXED_CUBE_Y


# Helper function to compute reward (PURE VLM - NO FALLBACK)
def compute_reward(ee_pos, ball_pos, grasped, prev_dist, vlm_reward=None):
    """Compute reward using VLM ONLY - no hand-crafted rules!"""
    ball_dist = np.linalg.norm(ee_pos - ball_pos)

    # VLM reward is REQUIRED - will be None only on non-VLM steps
    if vlm_reward is not None:
        reward = vlm_reward  # Pure vision-based reward (0-10 scale)
    else:
        # On non-VLM steps, use 0 reward (only train on VLM-scored steps)
        reward = 0.0

    return reward, ball_dist


try:
    for episode in range(MAX_EPISODES):
        print(f"\n===== Episode {episode+1}/{MAX_EPISODES} =====")

        # Reset environment
        cube_x, cube_y = reset_environment()

        episode_reward = 0.0
        episode_loss = []
        prev_dist = 999.0

        # Run episode
        for step in range(MAX_STEPS_PER_EPISODE):
            my_world.step(render=True)

            # Get current state (NO camera/vision)
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

            # FIXED STATE: Include cube position, ee position, and target position
            # This gives the model spatial awareness!
            target_pos = np.array([FIXED_TARGET_X, FIXED_TARGET_Y, FIXED_TARGET_Z])
            state = np.concatenate(
                [
                    joint_positions,  # 12D: robot joint angles
                    [grasped],  # 1D: is cube grasped?
                    ball_pos,  # 3D: WHERE IS THE CUBE? (was missing!)
                    ee_pos,  # 3D: WHERE IS THE GRIPPER? (was missing!)
                    target_pos,  # 3D: WHERE SHOULD WE GO? (was missing!)
                ]
            )  # Total: 22D (was 13D)

            # Get action from policy (NO image)
            action = agent.get_action(state, image=None, deterministic=False)

            # Execute action with RMPflow (INCREASED from 0.05 to 0.15 for faster movement)
            delta_pos = action[:3] * 0.15  # 15cm max movement per step
            target_position = ee_pos + delta_pos
            target_position = np.clip(
                target_position, [-0.6, -0.6, 0.05], [0.8, 0.6, 1.0]
            )
            rmp_flow.set_end_effector_target(
                target_position=target_position, target_orientation=None
            )
            actions = motion_policy.get_next_articulation_action(1.0 / 60.0)
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
                target_gripper = np.clip(
                    current_gripper + gripper_action * 0.01, 0.0, 0.04
                )
                current_joints[6] = target_gripper
                robot.set_joint_positions(current_joints)

            # Step simulation to get next state
            my_world.step(render=False)

            # Get next state
            joint_positions_raw_next = robot.get_joint_positions()
            if isinstance(joint_positions_raw_next, tuple):
                joint_positions_next = joint_positions_raw_next[0]
            else:
                joint_positions_next = joint_positions_raw_next

            ball_pos_next, _ = ball.get_world_pose()
            ball_pos_next = np.array(ball_pos_next).flatten()
            ee_pos_next, _ = robot.end_effector.get_world_pose()
            ee_pos_next = np.array(ee_pos_next).flatten()
            ball_dist_next = np.linalg.norm(ee_pos_next - ball_pos_next)
            gripper_pos_next = (
                joint_positions_next[6] if len(joint_positions_next) > 6 else 0.0
            )
            grasped_next = float(ball_dist_next < 0.15 and gripper_pos_next > 0.02)

            # FIXED NEXT STATE: Include spatial information
            target_pos_next = np.array([FIXED_TARGET_X, FIXED_TARGET_Y, FIXED_TARGET_Z])
            next_state = np.concatenate(
                [
                    joint_positions_next,  # 12D
                    [grasped_next],  # 1D
                    ball_pos_next,  # 3D: cube position
                    ee_pos_next,  # 3D: gripper position
                    target_pos_next,  # 3D: target position
                ]
            )  # Total: 22D

            # Compute VLM reward (every VLM_REWARD_INTERVAL steps to save compute)
            vlm_reward_value = None
            if vlm_helper is not None and step % VLM_REWARD_INTERVAL == 0:
                # Capture image from side camera for VLM
                image_side = side_camera.get_rgba()[:, :, :3]  # RGB only
                # Pass FULL context: image, complete state, and action taken
                vlm_reward_value = vlm_helper.get_reward(
                    image_side,
                    ee_pos,
                    ball_pos,
                    grasped,
                    target_pos,
                    joint_positions,
                    action=action
                )
                if step % 100 == 0:  # Print occasionally
                    print(f"  [VLM Step {step}] Score: {vlm_reward_value}/10 | Dist to cube: {ball_dist:.3f}m")

            # Compute reward using VLM
            reward, ball_dist = compute_reward(ee_pos, ball_pos, grasped, prev_dist, vlm_reward=vlm_reward_value)
            prev_dist = ball_dist
            episode_reward += reward

            # Store experience and train (online training with reward weighting, NO image)
            loss = agent.update(state, action, reward, next_state, image=None)
            if loss is not None:
                episode_loss.append(loss)

        # Episode summary
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        print(
            f"Episode {episode+1}/{MAX_EPISODES} | Reward: {episode_reward:.2f} | Loss: {avg_loss:.6f} | Buffer: {len(agent.buffer)}"
        )
        agent.episode_count += 1
        agent.total_reward_history.append(episode_reward)

        # Save checkpoint
        if (episode + 1) % SAVE_INTERVAL == 0:
            agent.save_model(MODEL_PATH)
            print(f"  → Checkpoint saved at episode {episode+1}\n")

        # Record video (SINGLE FILE - overwrite)
        if (episode + 1) % VIDEO_INTERVAL == 0:
            print(f"\n📹 Recording video at episode {episode+1}...")

            # Reset for video recording
            cube_x, cube_y = reset_environment()

            video_frames = []
            for step in range(200):  # Record 200 steps
                my_world.step(render=True)

                # Get state and action (no image needed)
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

                # Fixed state for video recording
                target_pos = np.array([FIXED_TARGET_X, FIXED_TARGET_Y, FIXED_TARGET_Z])
                state = np.concatenate(
                    [joint_positions, [grasped], ball_pos, ee_pos, target_pos]
                )  # 22D

                action = agent.get_action(state, image=None, deterministic=True)

                # Execute action (increased scaling for faster movement)
                delta_pos = action[:3] * 0.15
                target_position = ee_pos + delta_pos
                target_position = np.clip(
                    target_position, [-0.6, -0.6, 0.05], [0.8, 0.6, 1.0]
                )
                rmp_flow.set_end_effector_target(
                    target_position=target_position, target_orientation=None
                )
                actions = motion_policy.get_next_articulation_action(1.0 / 60.0)
                robot.apply_action(actions)

                gripper_action = np.clip(action[3], -1.0, 1.0)
                current_joints_raw = robot.get_joint_positions()
                if isinstance(current_joints_raw, tuple):
                    current_joints = current_joints_raw[0].copy()
                else:
                    current_joints = current_joints_raw.copy()
                if len(current_joints) > 6:
                    current_gripper = current_joints[6]
                    target_gripper = np.clip(
                        current_gripper + gripper_action * 0.01, 0.0, 0.04
                    )
                    current_joints[6] = target_gripper
                    robot.set_joint_positions(current_joints)

                # Capture frame from side camera
                side_camera.get_current_frame()
                side_rgba = side_camera.get_rgba()
                if side_rgba is not None and side_rgba.size > 0:
                    if len(side_rgba.shape) == 1:
                        side_rgba = side_rgba.reshape(720, 1280, 4)
                    side_rgb = side_rgba[:, :, :3].astype(np.uint8)
                    side_bgr = cv2.cvtColor(side_rgb, cv2.COLOR_RGB2BGR)
                    video_frames.append(side_bgr)

            # Save video (OVERWRITE single file)
            if len(video_frames) > 0:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                video_writer = cv2.VideoWriter(VIDEO_PATH, fourcc, 30.0, (1280, 720))
                if video_writer.isOpened():
                    for frame in video_frames:
                        video_writer.write(frame)
                    video_writer.release()
                    print(f"✓ Video saved (overwritten): {VIDEO_PATH}\n")

except KeyboardInterrupt:
    print("\nOnline training interrupted by user")
    agent.save_model(MODEL_PATH)
    print("Model saved before exit")
except Exception as e:
    print(f"Error during training: {e}")
    import traceback

    traceback.print_exc()
    agent.save_model(MODEL_PATH)
    print("Model saved after error")
finally:
    print(f"\n✓ Online training complete!")
    print(f"✓ Final model saved to: {MODEL_PATH}")
    agent.save_model(MODEL_PATH)

    # Explicit cleanup to avoid shutdown crash
    try:
        my_world.stop()
        my_world.clear()
    except:
        pass

    simulation_app.close()
