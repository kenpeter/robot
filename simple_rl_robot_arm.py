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
        "width": 500,
        "height": 200,
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

# Using MLP for online RL (simple and fast)
print("=" * 60)
print("SIMPLE MLP AGENT: Direct state-to-action mapping")
print("Why: Fast learning for sparse rewards")
print("=" * 60)


# Simple MLP policy network
class PolicyMLP(nn.Module):
    """Simple MLP that maps state directly to actions"""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        num_layers=4,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Build MLP layers
        layers = []
        layers.extend([nn.Linear(state_dim, hidden_dim), nn.ReLU()])

        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )

        layers.extend(
            [
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),  # Output in [-1, 1]
            ]
        )

        self.network = nn.Sequential(*layers)

        print(f"PolicyMLP initialized:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num layers: {num_layers}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, state):
        """Returns: action [batch, action_dim]"""
        return self.network(state)


# Simple MLP agent for online RL
class MLPAgent:
    """Simple MLP Agent for online RL with direct state-to-action mapping"""

    def __init__(
        self,
        state_dim,
        action_dim,
        device="cuda",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Initialize simple PolicyMLP
        self.model = PolicyMLP(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,
            num_layers=3,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        # Experience replay buffer (using deque for O(1) append/pop)
        self.buffer = deque(maxlen=10000)
        self.buffer_size = 10000
        self.batch_size = 64

        # Training stats
        self.episode_count = 0
        self.total_reward_history = []
        self.loss_history = []  # Track training loss
        self.noise_scale = 0.1  # REDUCED exploration noise (was 0.3, too high!)
        self.step_count = 0  # Total training steps

    def get_action(self, state, image=None, deterministic=False):
        """Generate action directly from MLP policy"""
        self.model.eval()
        with torch.no_grad():
            # Check for NaN in state
            if np.any(np.isnan(state)):
                print(f"WARNING: NaN detected in state: {state}")
                state = np.nan_to_num(state, 0.0)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Get action directly from policy network (single forward pass!)
            action = self.model(state_tensor)

            # Add exploration noise during training
            if not deterministic:
                noise = torch.randn_like(action) * self.noise_scale
                action = action + noise
                action = torch.clamp(action, -1.0, 1.0)

            # Final NaN check
            if torch.isnan(action).any():
                print("WARNING: NaN in action output, returning random")
                action = torch.randn_like(action) * 0.3

        return action.cpu().numpy()[0]

    def update(self, state, action, reward, next_state, image=None):
        """Store experience and train with simple supervised learning (behavioral cloning on good actions)"""
        experience = (state, action, reward, next_state)

        # Add to buffer
        self.buffer.append(experience)

        # Train if enough samples
        if len(self.buffer) < self.batch_size:
            return

        # Sample batch
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        # Convert to tensors
        states = torch.FloatTensor(np.array([s for s, a, r, ns in batch])).to(
            self.device
        )
        actions = torch.FloatTensor(np.array([a for s, a, r, ns in batch])).to(
            self.device
        )
        rewards = torch.FloatTensor(np.array([r for s, a, r, ns in batch])).to(
            self.device
        )

        # IMPROVED: Only learn from good experiences (reward > 0.1)
        self.model.train()
        predicted_actions = self.model(states)

        # Filter to meaningful experiences only
        good_mask = rewards > 0.1

        if good_mask.sum() > 10:  # Need at least 10 good samples
            # Only train on good experiences
            good_predicted = predicted_actions[good_mask]
            good_actions = actions[good_mask]
            good_rewards = rewards[good_mask]

            # Weighted loss: better actions get more weight
            weights = torch.softmax(good_rewards * 3.0, dim=0) * len(good_rewards)
            per_sample_loss = F.mse_loss(good_predicted, good_actions, reduction="none").mean(dim=1)
            loss = (per_sample_loss * weights).mean()
        else:
            # Not enough good data yet, uniform loss on all
            per_sample_loss = F.mse_loss(predicted_actions, actions, reduction="none").mean(dim=1)
            loss = per_sample_loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Track training metrics
        self.loss_history.append(loss.item())
        self.step_count += 1

        # Decay exploration noise more aggressively
        self.noise_scale = max(0.02, self.noise_scale * 0.998)  # Faster decay, lower floor

        # DEBUG: Print stats every 100 steps
        if self.step_count % 100 == 0:
            print(f"\n[MLP TRAINING - Step {self.step_count}]")
            print(
                f"  Action prediction error (MSE): {per_sample_loss.mean().item():.6f} ← Should DECREASE"
            )
            print(f"  Loss (weighted): {loss.item():.6f} ← Should DECREASE")
            print(
                f"  Reward range: [{rewards.min().item():.2f}, {rewards.max().item():.2f}]"
            )
            print(f"  Exploration noise scale: {self.noise_scale:.3f}")

        return loss.item()

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

# FIXED cube position - IN FRONT of robot (easier reach)
cube_size = 0.0515  # 5.15cm cube
# Robot base is at origin (0, 0, 0)
# UR10e reach is ~1.3m, so place cube well within comfortable reach
# Place cube IN FRONT (positive X direction, Y=0 center)
FIXED_CUBE_X = 0.6  # 0.6m in front of robot - comfortable reach
FIXED_CUBE_Y = 0.0  # CENTER (directly in front)
FIXED_CUBE_Z = (
    cube_size / 2.0 + 0.3
)  # 30cm above ground - table height

print(f"\n=== RED CUBE SETUP - IN FRONT OF ROBOT ===")
print(f"Cube position: [{FIXED_CUBE_X:.3f}, {FIXED_CUBE_Y:.3f}, {FIXED_CUBE_Z:.3f}]")
print(f"(0.6m IN FRONT of robot, centered, table height - easy reach!)")

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

# SIMPLIFIED STATE: Remove target position, keep it simple!
# State: 12 joints + 1 grasped + 3 cube_pos + 3 ee_pos = 19D
state_dim = 19  # 12 joints + 1 grasped + 3 cube + 3 ee
action_dim = 4  # dx, dy, dz, gripper

agent = MLPAgent(state_dim=state_dim, action_dim=action_dim, device="cuda")

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
MAX_STEPS_PER_EPISODE = 2000
SAVE_INTERVAL = 1  # Save model every 10 episodes
VIDEO_INTERVAL = 1  # Record video every 20 episodes
VIDEO_PATH = "/home/kenpeter/work/robot/training_video.avi"  # Single file, overwrite

# Start simulation
print("[DEBUG] Starting simulation timeline...")
my_world.play()
print(f"[DEBUG] World is playing: {my_world.is_playing()}")
print("✓ Simulation timeline started\n")


# FIXED target position for robot end-effector (just above the cube)
FIXED_TARGET_X = -1.0  # Match cube X (behind robot)
FIXED_TARGET_Y = 0.0  # Match cube Y (centered)
FIXED_TARGET_Z = 0.2  # 10cm above the cube for pre-grasp position

# === STAGE-BASED DENSE REWARD (6 STAGES) ===
print("=" * 70)
print("USING 6-STAGE DENSE REWARD (ALL NORMALIZED 0-1)")
print("=" * 70)
print("Stage 0 (>2.0m):    Very far - initial movement        (max ~0.15)")
print("Stage 1 (1.5-2.0m): Far checkpoint - sustained motion  (max ~0.18)")
print("Stage 2 (1.0-1.5m): Medium far - acceleration          (max ~0.22)")
print("Stage 3 (0.5-1.0m): Medium close - focused approach    (max ~0.33)")
print("Stage 4 (0.1-0.5m): Close - precision positioning      (max ~0.45)")
print("Stage 5 (<0.1m):    Grasping range - completion        (max 1.00)")
print("Progress bonuses at EVERY stage for moving closer!")
print("=" * 70)


# Helper function to reset environment
def reset_environment():
    """Reset robot and cube to FIXED positions"""
    # Reset robot to NATURAL initial position (arm relaxed, ready to reach forward)
    # Cube is at [0.6, 0.0, 0.33] - IN FRONT of robot at table height
    #
    # UR10e joint configuration for natural "ready" pose:
    # - shoulder_pan: 0° (facing forward, positive X direction)
    # - shoulder_lift: -60° (arm slightly raised from horizontal)
    # - elbow: 90° (bent upward for comfortable reach)
    # - wrist_1: -30° (slight downward tilt)
    # - wrist_2: 0° (neutral)
    # - wrist_3: 0° (neutral)
    initial_pos = np.array(
        [
            0.0,            # shoulder_pan: 0° = face forward (positive X)
            -np.pi / 3,     # shoulder_lift: -60° = arm raised comfortably
            np.pi / 2,      # elbow: 90° = bent upward
            -np.pi / 6,     # wrist_1: -30° = slight downward tilt
            0.0,            # wrist_2: neutral
            0.0,            # wrist_3: neutral
        ]
    )
    gripper_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    robot.set_joint_positions(np.concatenate([initial_pos, gripper_open]))

    # Reset cube to FIXED position (no randomization)
    ball.set_world_pose(position=np.array([FIXED_CUBE_X, FIXED_CUBE_Y, FIXED_CUBE_Z]))

    # Stabilize
    for _ in range(10):
        my_world.step(render=False)

    return FIXED_CUBE_X, FIXED_CUBE_Y


# Helper function to compute reward (STAGE-BASED DENSE REWARD - OPTIMIZED for close start)
def compute_reward(
    ee_pos, ball_pos, grasped, prev_distance=None, prev_ee_pos=None, episode_step=0, gripper_pos=0.0, ee_rot=None
):
    """
    STAGE-BASED DENSE REWARD - OPTIMIZED FOR CLOSE STARTING POSITION
    Robot starts at 0.3-0.5m away, so we need VERY DENSE rewards in this range!

    TWO-PHASE REWARD:
    1. ARM POSITIONING: Get end-effector close to cube (for far distances)
    2. GRIPPER POSITIONING: Get gripper CENTER (between fingers) close to cube

    IMPORTANT: We measure distance from GRIPPER CENTER (not end-effector base!)
    - Gripper center is ~10cm forward from end-effector base (in gripper's local Z direction)
    - This ensures we reward gripper fingers getting close, not just the wrist

    - Stage 0 (>0.4m):    Initial close range - arm positioning
    - Stage 1 (0.25-0.4m): Very close - arm positioning
    - Stage 2 (0.15-0.25m): Pre-grasp - arm positioning complete
    - Stage 3 (0.08-0.15m): Grasp preparation - gripper fingers approaching
    - Stage 4 (<0.08m):    Grasping range - gripper fingers must close on cube

    ALL REWARDS NORMALIZED TO 0-1 RANGE with STRONG PROGRESS BONUSES
    """
    # Calculate gripper center position (fingers extend in local +Z direction from ee_link)
    if ee_rot is not None:
        # Convert quaternion to rotation matrix to get forward direction
        # quaternion format: [w, x, y, z]
        w, x, y, z = ee_rot[0], ee_rot[1], ee_rot[2], ee_rot[3]

        # Rotation matrix (extract Z axis - forward direction of gripper)
        forward_x = 2.0 * (x*z + w*y)
        forward_y = 2.0 * (y*z - w*x)
        forward_z = 1.0 - 2.0 * (x*x + y*y)
        forward_dir = np.array([forward_x, forward_y, forward_z])
        forward_dir = forward_dir / (np.linalg.norm(forward_dir) + 1e-8)  # normalize

        # Gripper center is ~10cm forward from end-effector base
        gripper_forward_offset = 0.10
        gripper_center = ee_pos + forward_dir * gripper_forward_offset
    else:
        # Fallback: simple offset (if rotation not provided)
        gripper_forward_offset = 0.10
        gripper_center = ee_pos + np.array([0, 0, gripper_forward_offset])

    # Calculate current distance (gripper center to cube center)
    distance = np.linalg.norm(gripper_center - ball_pos)

    # Normalize gripper position to [0, 1] range (0 = open, 1 = closed)
    # gripper_pos ranges from 0.0 (open) to 0.04 (closed)
    gripper_normalized = np.clip(gripper_pos / 0.04, 0.0, 1.0)

    # === STAGE 0: INITIAL CLOSE RANGE (distance > 0.4m) ===
    if distance > 0.4:
        # Moderate reward for being reasonably close
        distance_reward = np.exp(-3.0 * distance) * 0.18  # Max ~0.18

        # Strong progress bonus to encourage movement
        progress_bonus = 0.0
        if prev_distance is not None and prev_distance > distance:
            improvement = prev_distance - distance
            progress_bonus = min(0.15, improvement * 25.0)  # Max 0.15

        # Small gripper reward (keep it open at this stage)
        gripper_reward = 0.02 * (1.0 - gripper_normalized)  # Max 0.02 for staying open

        reward = distance_reward + progress_bonus + gripper_reward  # Max ~0.35

    # === STAGE 1: VERY CLOSE (0.25m < distance <= 0.4m) ===
    elif distance > 0.25:
        # Strong exponential reward for getting very close
        distance_reward = np.exp(-5.0 * distance) * 0.23  # Max ~0.23

        # Very strong progress bonus
        progress_bonus = 0.0
        if prev_distance is not None and prev_distance > distance:
            improvement = prev_distance - distance
            progress_bonus = min(0.20, improvement * 30.0)  # Max 0.20

        # Small gripper reward (keep it open, approaching)
        gripper_reward = 0.02 * (1.0 - gripper_normalized)  # Max 0.02 for staying open

        reward = distance_reward + progress_bonus + gripper_reward  # Max ~0.45

    # === STAGE 2: PRE-GRASP (0.15m < distance <= 0.25m) ===
    elif distance > 0.15:
        # ARM POSITIONING REWARD: Strong exponential for precise positioning
        distance_reward = np.exp(-6.0 * distance) * 0.25  # Max ~0.25

        # Progress bonus for arm movement
        progress_bonus = 0.0
        if prev_distance is not None and prev_distance > distance:
            improvement = prev_distance - distance
            progress_bonus = min(0.25, improvement * 40.0)  # Max 0.25

        # Keep gripper OPEN for alignment (arm positioning priority)
        gripper_reward = 0.05 * (1.0 - gripper_normalized)  # Max 0.05 for staying open

        reward = distance_reward + progress_bonus + gripper_reward  # Max ~0.55

    # === STAGE 3: GRASP PREPARATION (0.08m < distance <= 0.15m) ===
    elif distance > 0.08:
        # ARM POSITIONING REWARD: Good position maintained
        distance_reward = np.exp(-8.0 * distance) * 0.20  # Max ~0.20

        # Progress bonus for getting even closer
        progress_bonus = 0.0
        if prev_distance is not None and prev_distance > distance:
            improvement = prev_distance - distance
            progress_bonus = min(0.20, improvement * 50.0)  # Max 0.20

        # GRIPPER CLOSING GUIDANCE: Start closing when arm is positioned!
        # The closer the arm, the MORE the gripper should close
        # At 0.15m: gripper should be ~30% closed (fingers approaching)
        # At 0.08m: gripper should be ~60% closed (fingers very close to cube)
        desired_gripper = 0.3 + (0.15 - distance) / (0.15 - 0.08) * 0.3  # 0.3 to 0.6
        gripper_error = abs(gripper_normalized - desired_gripper)
        gripper_reward = 0.25 * (1.0 - gripper_error)  # Max 0.25 - HIGH WEIGHT!

        reward = distance_reward + progress_bonus + gripper_reward  # Max ~0.65

    # === STAGE 4: GRASPING RANGE (distance <= 0.08m) ===
    else:
        # ARM POSITIONING REWARD: Excellent position, slightly scale with closeness
        # At 0.08m: 0.15, at 0.00m: 0.20
        proximity_reward = 0.15 + (0.08 - min(distance, 0.08)) / 0.08 * 0.05  # 0.15 to 0.20

        # GRIPPER CLOSING REWARD: VERY STRONG - this is the main action now!
        # Arm is positioned well, now gripper fingers MUST close to grasp
        # At 0.08m: expect 60% closed (fingers touching cube sides)
        # At 0.04m: expect 80% closed (fingers wrapping around)
        # At 0.00m: expect 100% closed (full grasp!)
        desired_gripper = 0.6 + (0.08 - min(distance, 0.08)) / 0.08 * 0.4  # 0.6 to 1.0
        gripper_error = abs(gripper_normalized - desired_gripper)
        gripper_closing_reward = 0.40 * (1.0 - gripper_error)  # Max 0.40 - VERY HIGH WEIGHT!

        # HUGE bonus for successful grasp (actual contact + gripper closed)
        if grasped:
            grasp_bonus = 0.40  # Massive success signal!
        else:
            grasp_bonus = 0.0

        reward = proximity_reward + gripper_closing_reward + grasp_bonus  # Max 1.00 (perfect!)

    # Ensure final reward is in 0-1 range
    reward = np.clip(reward, 0.0, 1.0)

    return reward, distance


try:
    for episode in range(MAX_EPISODES):
        print(f"\n===== Episode {episode+1}/{MAX_EPISODES} =====")

        # Reset environment
        cube_x, cube_y = reset_environment()

        episode_reward = 0.0
        episode_loss = []
        prev_distance = None  # Track previous distance for progress reward
        prev_ee_pos = None  # Track previous ee position for direction detection

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
            ee_pos, ee_rot = robot.end_effector.get_world_pose()
            ee_pos = np.array(ee_pos).flatten()
            ee_rot = np.array(ee_rot).flatten()  # quaternion [w, x, y, z]
            ball_dist = np.linalg.norm(ee_pos - ball_pos)
            gripper_pos = joint_positions[6] if len(joint_positions) > 6 else 0.0
            grasped = float(ball_dist < 0.15 and gripper_pos > 0.02)

            # SIMPLIFIED STATE: Just joints, grasp status, cube position, and ee position
            state = np.concatenate(
                [
                    joint_positions,  # 12D: robot joint angles
                    [grasped],  # 1D: is cube grasped?
                    ball_pos,  # 3D: cube position
                    ee_pos,  # 3D: end-effector position
                ]
            )  # Total: 19D

            # Get action from policy (NO image)
            action = agent.get_action(state, image=None, deterministic=False)

            # TRUST THE LEARNED POLICY with SMALL GUIDANCE BIAS
            # Scale actions to meters
            delta_pos = action[:3] * 0.5

            # Add small bias toward cube (10% guidance) to prevent random drift
            direction_to_cube = ball_pos - ee_pos
            direction_norm = np.linalg.norm(direction_to_cube)
            if direction_norm > 0.01:
                cube_bias = (direction_to_cube / direction_norm) * 0.05  # 5cm bias toward cube
                delta_pos = delta_pos * 0.9 + cube_bias  # 90% policy + 10% guidance

            target_position = ee_pos + delta_pos
            # Workspace limits for cube at [0.6, 0.0, 0.33] (in front of robot)
            target_position = np.clip(
                target_position, [-0.2, -0.8, 0.05], [1.0, 0.8, 0.8]
            )
            rmp_flow.set_end_effector_target(
                target_position=target_position, target_orientation=None
            )
            actions = motion_policy.get_next_articulation_action(1.0 / 60.0)
            robot.apply_action(actions)

            # Gripper control - IMPROVED for better grasping
            gripper_action = np.clip(action[3], -1.0, 1.0)
            current_joints_raw = robot.get_joint_positions()
            if isinstance(current_joints_raw, tuple):
                current_joints = current_joints_raw[0].copy()
            else:
                current_joints = current_joints_raw.copy()
            if len(current_joints) > 6:
                current_gripper = current_joints[6]

                # SMART GRIPPER: Auto-close when near cube
                if ball_dist < 0.1:  # Very close to cube - try to grasp!
                    # Force close gripper aggressively
                    target_gripper = np.clip(current_gripper + 0.05, 0.0, 0.04)
                else:
                    # Normal gripper control with faster movement
                    target_gripper = np.clip(
                        current_gripper + gripper_action * 0.02, 0.0, 0.04
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

            # SIMPLIFIED NEXT STATE
            next_state = np.concatenate(
                [
                    joint_positions_next,  # 12D
                    [grasped_next],  # 1D
                    ball_pos_next,  # 3D: cube position
                    ee_pos_next,  # 3D: ee position
                ]
            )  # Total: 19D

            # Compute distance-based reward with directional awareness and penalties
            reward, current_distance = compute_reward(
                ee_pos,
                ball_pos,
                grasped,
                prev_distance=prev_distance,
                prev_ee_pos=prev_ee_pos,
                episode_step=step,
                gripper_pos=gripper_pos,
                ee_rot=ee_rot,
            )
            prev_distance = current_distance  # Update for next iteration
            prev_ee_pos = ee_pos.copy()  # Update previous position

            # Log progress every 100 steps
            if step % 100 == 0:
                print(
                    f"  [Step {step}] Reward: {reward:.3f} | Dist to cube: {ball_dist:.3f}m"
                )
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

                # Simplified state for video recording
                state = np.concatenate(
                    [joint_positions, [grasped], ball_pos, ee_pos]
                )  # 19D

                action = agent.get_action(state, image=None, deterministic=True)

                # TRUST THE LEARNED POLICY: Use agent's actions directly
                delta_pos = action[:3] * 0.5

                target_position = ee_pos + delta_pos
                # Workspace limits for cube in front
                target_position = np.clip(
                    target_position, [-0.2, -0.8, 0.05], [1.0, 0.8, 0.8]
                )
                rmp_flow.set_end_effector_target(
                    target_position=target_position, target_orientation=None
                )
                actions = motion_policy.get_next_articulation_action(1.0 / 60.0)
                robot.apply_action(actions)

                # Gripper control - IMPROVED for better grasping
                gripper_action = np.clip(action[3], -1.0, 1.0)
                current_joints_raw = robot.get_joint_positions()
                if isinstance(current_joints_raw, tuple):
                    current_joints = current_joints_raw[0].copy()
                else:
                    current_joints = current_joints_raw.copy()
                if len(current_joints) > 6:
                    current_gripper = current_joints[6]

                    # SMART GRIPPER: Auto-close when near cube
                    if ball_dist < 0.1:
                        target_gripper = np.clip(current_gripper + 0.05, 0.0, 0.04)
                    else:
                        target_gripper = np.clip(
                            current_gripper + gripper_action * 0.02, 0.0, 0.04
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
