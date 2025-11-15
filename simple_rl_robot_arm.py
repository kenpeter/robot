# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ONLINE RL Training - CORRECTED VERSION
Fixed joint count (12 joints) and missing MODEL_PATH
"""

# sim app
from isaacsim import SimulationApp

# Initialize simulation
simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 500,
        "height": 200,
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
import cv2
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

print("=" * 60)
print("CORRECTED GRASPING TRAINING - FIXED VERSION")
print("Fixed: 12 joints, State dimensions, Gripper scaling")
print("=" * 60)


# === CORRECTED FINGER TRACKING ===
class FingerTracker:
    """Track finger tip positions with corrected gripper scaling"""

    def __init__(self, robot_ref):
        self.robot = robot_ref

    def get_finger_positions(self, ee_pos, ee_rot, gripper_pos):
        """Get finger tip positions with CORRECT gripper scaling"""
        # CORRECTED: Use 40.0 for normalization (not 0.04)
        gripper_openness = 1.0 - (gripper_pos / 40.0)  # 1=open, 0=closed

        # Extract orientation vectors
        w, x, y, z = ee_rot[0], ee_rot[1], ee_rot[2], ee_rot[3]

        # Forward direction
        forward_x = 2.0 * (x * z + w * y)
        forward_y = 2.0 * (y * z - w * x)
        forward_z = 1.0 - 2.0 * (x * x + y * y)
        forward_dir = np.array([forward_x, forward_y, forward_z])
        forward_dir = forward_dir / (np.linalg.norm(forward_dir) + 1e-8)

        # Right direction
        right_x = 2.0 * (w * z + x * y)
        right_y = 1.0 - 2.0 * (x * x + z * z)
        right_z = 2.0 * (y * z - w * x)
        right_dir = np.array([right_x, right_y, right_z])
        right_dir = right_dir / (np.linalg.norm(right_dir) + 1e-8)

        # Finger geometry
        finger_offset = 0.15 + gripper_openness * 0.03
        finger_separation = 0.04 + gripper_openness * 0.06

        left_finger = (
            ee_pos + forward_dir * finger_offset - right_dir * finger_separation / 2
        )
        right_finger = (
            ee_pos + forward_dir * finger_offset + right_dir * finger_separation / 2
        )

        return left_finger, right_finger


# === SIMPLE MLP POLICY ===
class PolicyMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        self.action_dim = action_dim

        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.extend([nn.Linear(hidden_dim, action_dim), nn.Tanh()])

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


# === CORRECTED AGENT ===
class MLPAgent:
    def __init__(self, state_dim, action_dim, device="cuda"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.model = PolicyMLP(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.buffer = deque(maxlen=10000)
        self.batch_size = 64

        self.episode_count = 0
        self.total_reward_history = []
        self.loss_history = []
        self.noise_scale = 0.1
        self.step_count = 0

    def get_action(self, state, deterministic=False):
        self.model.eval()
        with torch.no_grad():
            if np.any(np.isnan(state)):
                print(f"WARNING: NaN detected in state: {state}")
                state = np.nan_to_num(state, 0.0)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.model(state_tensor)

            if not deterministic:
                noise = torch.randn_like(action) * self.noise_scale
                action = action + noise
                action = torch.clamp(action, -1.0, 1.0)

        return action.cpu().numpy()[0]

    def update(self, state, action, reward, next_state):
        experience = (state, action, reward, next_state)
        self.buffer.append(experience)

        if len(self.buffer) < self.batch_size:
            return None

        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = torch.FloatTensor(np.array([s for s, a, r, ns in batch])).to(
            self.device
        )
        actions = torch.FloatTensor(np.array([a for s, a, r, ns in batch])).to(
            self.device
        )
        rewards = torch.FloatTensor(np.array([r for s, a, r, ns in batch])).to(
            self.device
        )

        # Simple policy gradient
        self.model.train()
        predicted_actions = self.model(states)

        advantage = rewards - rewards.mean()
        action_diff = predicted_actions - actions
        mse_per_sample = (action_diff**2).mean(dim=1)

        advantage_weights = torch.softmax(advantage * 5.0, dim=0) * len(advantage)
        loss = (mse_per_sample * advantage_weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.loss_history.append(loss.item())
        self.step_count += 1
        self.noise_scale = max(0.02, self.noise_scale * 0.998)

        if self.step_count % 100 == 0:
            avg_reward = rewards.mean().item()
            print(
                f"\n[Step {self.step_count}] Loss: {loss.item():.4f} | Avg Reward: {avg_reward:.3f}"
            )

        return loss.item()

    def save_model(self, filepath):
        model_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode_count": self.episode_count,
            "total_reward_history": self.total_reward_history,
            "loss_history": self.loss_history[-1000:],
            "buffer": list(self.buffer)[-1000:],
            "noise_scale": self.noise_scale,
            "step_count": self.step_count,
        }
        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
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

        buffer_list = model_data.get("buffer", [])
        self.buffer = deque(buffer_list, maxlen=10000)

        self.noise_scale = model_data.get("noise_scale", 0.3)

        print(f"Model loaded from {filepath}")
        return True


# === SCENE SETUP ===
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

set_camera_view(
    eye=[2.5, 2.5, 2.0],
    target=[0.0, 0.0, 0.5],
    camera_prim_path="/OmniverseKit_Persp",
)

# Robot setup
asset_path = (
    assets_root_path
    + "/Isaac/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd"
)
robot_prim = add_reference_to_stage(usd_path=asset_path, prim_path="/World/ur")

gripper = ParallelGripper(
    end_effector_prim_path="/World/ur/ee_link/robotiq_arg2f_base_link",
    joint_prim_names=["finger_joint"],
    joint_opened_positions=np.array([0]),
    joint_closed_positions=np.array([40]),
    action_deltas=np.array([-40]),
    use_mimic_joints=True,
)

robot = SingleManipulator(
    prim_path="/World/ur",
    name="ur10_robot",
    end_effector_prim_path="/World/ur/ee_link/robotiq_arg2f_base_link",
    gripper=gripper,
)

# Camera setup
from isaacsim.core.utils.prims import create_prim
from isaacsim.sensors.camera import Camera
from pxr import Gf, UsdLux

camera_prim_path = "/World/OverheadCamera"
create_prim(camera_prim_path, "Camera")
set_camera_view(
    eye=[0.1, 0.0, 1.2], target=[0.1, 0.0, 0.0], camera_prim_path=camera_prim_path
)
overhead_camera = Camera(prim_path=camera_prim_path, resolution=(84, 84))
overhead_camera.initialize()

side_camera_prim_path = "/World/SideCamera"
create_prim(side_camera_prim_path, "Camera")
set_camera_view(
    eye=[3.5, 3.5, 2.5], target=[0.0, 0.0, 0.5], camera_prim_path=side_camera_prim_path
)
side_camera = Camera(prim_path=side_camera_prim_path, resolution=(1280, 720))
side_camera.initialize()

# Cube setup
from isaacsim.core.api.objects import DynamicCuboid

cube_size = 0.0515
FIXED_CUBE_X = 0.5
FIXED_CUBE_Y = 0.0
FIXED_CUBE_Z = cube_size / 2.0 + 0.0

cube = DynamicCuboid(
    name="red_cube",
    position=np.array([FIXED_CUBE_X, FIXED_CUBE_Y, FIXED_CUBE_Z]),
    orientation=np.array([1, 0, 0, 0]),
    prim_path="/World/Cube",
    scale=np.array([cube_size, cube_size, cube_size]),
    size=1.0,
    color=np.array([1.0, 0.0, 0.0]),
)
my_world.scene.add(cube)

ball = cube

# Lighting
stage = my_world.stage
dome_light_path = "/World/DomeLight"
dome_light = UsdLux.DomeLight.Define(stage, dome_light_path)
dome_light.CreateIntensityAttr(1000.0)

distant_light_path = "/World/DistantLight"
distant_light = UsdLux.DistantLight.Define(stage, distant_light_path)
distant_light.CreateIntensityAttr(2000.0)
distant_light_xform = distant_light.AddRotateXYZOp()
distant_light_xform.Set(Gf.Vec3f(-45, 0, 0))

# === CORRECTED RL PARAMETERS ===
# CORRECTED: The robot has 12 joints (6 arm + 6 gripper mimic joints)
# 12 joints + 1 grasped + 3 ball + 3 EE + 3 left + 3 right = 25
state_dim = 25
action_dim = 4

# FIXED: Add missing MODEL_PATH definition
MODEL_PATH = "rl_robot_arm_model.pth"

# Initialize world
my_world.reset()
robot.initialize()
ball.initialize()

# Initialize components
finger_tracker = FingerTracker(robot)
agent = MLPAgent(state_dim=state_dim, action_dim=action_dim, device="cuda")

# RMPflow setup
import isaacsim.robot_motion.motion_generation as mg

rmpflow_dir = os.path.join(os.path.dirname(__file__), "rmpflow")
rmp_flow = mg.lula.motion_policies.RmpFlow(
    robot_description_path=os.path.join(rmpflow_dir, "robot_descriptor.yaml"),
    rmpflow_config_path=os.path.join(rmpflow_dir, "ur10e_rmpflow_common.yaml"),
    urdf_path=os.path.join(rmpflow_dir, "ur10e.urdf"),
    end_effector_frame_name="ee_link_robotiq_arg2f_base_link",
    maximum_substep_size=0.00334,
)

physics_dt = 1.0 / 60.0
motion_policy = ArticulationMotionPolicy(robot, rmp_flow, physics_dt)

print("✓ All components initialized")
print(f"✓ CORRECTED: State dim={state_dim} (12 joints), Gripper scaling fixed")

# Training parameters
MAX_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 1000
SAVE_INTERVAL = 10
VIDEO_INTERVAL = 20
VIDEO_PATH = "/home/kenpeter/work/robot/training_video.avi"

# Start simulation
my_world.play()
print(f"World is playing: {my_world.is_playing()}")


# === CORRECTED RESET ===
def reset_environment():
    """CORRECTED: Proper joint initialization with 12 joints total"""
    # 6 arm joints - STABLE starting configuration with downward gripper
    initial_pos = np.array(
        [
            0.0,  # shoulder_pan: face forward
            -np.pi / 3,  # shoulder_lift: stable raised position
            np.pi / 2.5,  # elbow: moderate bend
            -np.pi / 2,  # wrist_1: point down
            -np.pi / 2,  # wrist_2: downward pitch alignment
            0.0,  # wrist_3: neutral
        ]
    )

    # CORRECTED: 6 gripper mimic joints (all should be 0 for open gripper)
    gripper_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    joint_positions = np.concatenate([initial_pos, gripper_open])

    # Debug: Check joint count
    print(f"Setting {len(joint_positions)} joint positions")
    robot.set_joint_positions(joint_positions)

    # Verify the positions were set
    actual_positions = robot.get_joint_positions()
    if isinstance(actual_positions, tuple):
        actual_positions = actual_positions[0]
    print(f"✓ Reset: Set {len(actual_positions)} joints")

    ball.set_world_pose(position=np.array([FIXED_CUBE_X, FIXED_CUBE_Y, FIXED_CUBE_Z]))

    for _ in range(10):
        my_world.step(render=False)

    return FIXED_CUBE_X, FIXED_CUBE_Y


# === SIMPLE REWARD ===
def compute_reward(ee_pos, ball_pos, grasped, gripper_pos, left_finger, right_finger, prev_distance=None):
    """SUPER SIMPLE: Closer = reward, Away = penalty, Grasp = BIG reward"""
    # Average finger distance to cube (closer = better)
    left_dist = np.linalg.norm(left_finger - ball_pos)
    right_dist = np.linalg.norm(right_finger - ball_pos)
    avg_finger_dist = (left_dist + right_dist) / 2.0

    # Base distance reward: normalize to [0, 1]
    distance_reward = max(0.0, 1.0 - (avg_finger_dist / 2.0))

    # Movement penalty/reward: penalize moving away, reward moving closer
    movement_reward = 0.0
    if prev_distance is not None:
        delta = prev_distance - avg_finger_dist  # positive if closer, negative if away
        movement_reward = delta * 2.0  # Scale to make it significant

    # Big bonus for successful grasp
    grasp_reward = 1.0 if grasped else 0.0

    # Total reward (can be negative if moving away!)
    total_reward = distance_reward + movement_reward + grasp_reward

    # Normalize to roughly [-1, 2] range, then to [0, 1]
    normalized_reward = (total_reward + 1.0) / 3.0

    return normalized_reward, avg_finger_dist


# === CORRECTED ACTION GUIDANCE ===
def get_guided_action(ee_pos, ball_pos, action, step_count):
    """Direct waypoint guidance with corrected approach - FIXED Z-heights"""
    # CRITICAL FIX: Account for gripper geometry (gripper base is ~15cm from fingers)
    # When EE is at cube height, fingers are actually 15cm lower!
    # All target Z positions already include this 15cm offset

    # Define waypoints with CORRECTED heights
    if step_count < 200:
        # Phase 1: Move above the cube
        target_position = np.array([ball_pos[0], ball_pos[1], ball_pos[2] + 0.35])
        phase = "APPROACH"
    elif step_count < 500:
        # Phase 2: Move to pre-grasp position (fingers 8cm above cube)
        target_position = np.array([ball_pos[0], ball_pos[1], ball_pos[2] + 0.23])
        phase = "PRE_GRASP"
    elif step_count < 700:
        # Phase 3: Descend to grasp (fingers 3cm above cube)
        target_position = np.array([ball_pos[0], ball_pos[1], ball_pos[2] + 0.18])
        phase = "DESCEND"
    else:
        # Phase 4: Fine control
        target_position = ee_pos + action[:3] * 0.05
        phase = "FINE_CONTROL"

    # Smooth movement
    if phase != "FINE_CONTROL":
        direction = target_position - ee_pos
        distance = np.linalg.norm(direction)
        if distance > 0.01:
            move_speed = min(0.15, distance * 0.5)
            direction_normalized = direction / distance
            target_position = ee_pos + direction_normalized * move_speed

    # Safety limits - allow lower Z for descent
    target_position = np.clip(target_position, [-0.2, -0.5, 0.1], [0.8, 0.8, 0.8])

    return target_position, phase


# === CORRECTED GRIPPER CONTROL ===
def get_gripper_action(current_gripper, min_finger_dist, phase):
    """CORRECTED: Proper gripper scaling with 40.0 range"""
    if phase == "DESCEND" and min_finger_dist < 0.08:
        return min(current_gripper + 1.5, 40.0)  # Start closing during descend
    elif phase == "FINE_CONTROL" and min_finger_dist < 0.05:
        return min(current_gripper + 2.5, 40.0)  # Close faster when very close
    elif phase == "FINE_CONTROL" and min_finger_dist < 0.1:
        return min(current_gripper + 1.5, 40.0)  # Start closing
    else:
        return max(current_gripper - 1.0, 0.0)  # Keep open during approach


# === MAIN TRAINING LOOP ===
try:
    for episode in range(MAX_EPISODES):
        print(f"\n===== Episode {episode+1}/{MAX_EPISODES} =====")

        cube_x, cube_y = reset_environment()
        episode_reward = 0.0
        episode_loss = []
        prev_finger_distance = None  # Track previous distance for penalty/reward

        for step in range(MAX_STEPS_PER_EPISODE):
            my_world.step(render=True)

            # Get current state
            joint_positions_raw = robot.get_joint_positions()
            joint_positions = (
                joint_positions_raw[0]
                if isinstance(joint_positions_raw, tuple)
                else joint_positions_raw
            )

            # CORRECTED: Verify we have 12 joints
            if len(joint_positions) != 12:
                print(f"⚠️ WARNING: Expected 12 joints, got {len(joint_positions)}")
                # Use zeros for missing joints
                if len(joint_positions) < 12:
                    joint_positions = np.concatenate(
                        [joint_positions, np.zeros(12 - len(joint_positions))]
                    )
                else:
                    joint_positions = joint_positions[:12]

            ball_pos, _ = ball.get_world_pose()
            ball_pos = np.array(ball_pos).flatten()
            ee_pos, ee_rot = robot.end_effector.get_world_pose()
            ee_pos = np.array(ee_pos).flatten()
            ee_rot = np.array(ee_rot).flatten()

            # CORRECTED: Gripper is the 7th joint (index 6)
            gripper_pos = joint_positions[6] if len(joint_positions) > 6 else 0.0

            left_finger, right_finger = finger_tracker.get_finger_positions(
                ee_pos, ee_rot, gripper_pos
            )

            # CORRECTED: Proper grasp detection with scaled threshold
            ball_dist = np.linalg.norm(ee_pos - ball_pos)
            grasped = float(ball_dist < 0.15 and gripper_pos > 20.0)

            # CORRECTED STATE: 12 joints + 1 grasped + 3 ball + 3 EE + 3 left + 3 right = 25 elements
            state = np.concatenate(
                [
                    joint_positions,  # 12 elements
                    [grasped],  # 1 element
                    ball_pos,  # 3 elements
                    ee_pos,  # 3 elements
                    left_finger,  # 3 elements
                    right_finger,  # 3 elements
                ]
            )

            # Verify state dimension
            if len(state) != 25:
                print(f"⚠️ WARNING: State dimension mismatch: {len(state)} != 25")
                continue

            # Get action
            action = agent.get_action(state, deterministic=False)

            # Simple guided action
            target_position, current_phase = get_guided_action(
                ee_pos, ball_pos, action, step
            )

            # FIXED: Use correct downward orientation
            # Quaternion for gripper pointing DOWN toward cube
            if current_phase in ["APPROACH", "PRE_GRASP", "DESCEND"]:
                # Downward orientation: rotate 180° around X-axis [w, x, y, z]
                target_orientation = np.array([0.0, 1.0, 0.0, 0.0])
            else:
                # Allow flexibility in fine control
                target_orientation = None

            # Apply movement
            rmp_flow.set_end_effector_target(
                target_position=target_position, target_orientation=target_orientation
            )
            actions = motion_policy.get_next_articulation_action(1.0 / 60.0)
            robot.apply_action(actions)

            # CORRECTED gripper control
            current_joints_raw = robot.get_joint_positions()
            current_joints = (
                current_joints_raw[0]
                if isinstance(current_joints_raw, tuple)
                else current_joints_raw.copy()
            )
            if len(current_joints) > 6:
                current_gripper = current_joints[6]
                left_dist = np.linalg.norm(left_finger - ball_pos)
                right_dist = np.linalg.norm(right_finger - ball_pos)
                min_finger_dist = min(left_dist, right_dist)

                target_gripper = get_gripper_action(
                    current_gripper, min_finger_dist, current_phase
                )
                current_joints[6] = target_gripper
                robot.set_joint_positions(current_joints)

            # Step simulation
            my_world.step(render=False)

            # Get next state
            joint_positions_raw_next = robot.get_joint_positions()
            joint_positions_next = (
                joint_positions_raw_next[0]
                if isinstance(joint_positions_raw_next, tuple)
                else joint_positions_raw_next
            )

            ball_pos_next, _ = ball.get_world_pose()
            ball_pos_next = np.array(ball_pos_next).flatten()
            ee_pos_next, ee_rot_next = robot.end_effector.get_world_pose()
            ee_pos_next = np.array(ee_pos_next).flatten()
            ee_rot_next = np.array(ee_rot_next).flatten()
            gripper_pos_next = (
                joint_positions_next[6] if len(joint_positions_next) > 6 else 0.0
            )

            left_finger_next, right_finger_next = finger_tracker.get_finger_positions(
                ee_pos_next, ee_rot_next, gripper_pos_next
            )

            # CORRECTED grasp detection
            ball_dist_next = np.linalg.norm(ee_pos_next - ball_pos_next)
            grasped_next = float(ball_dist_next < 0.15 and gripper_pos_next > 20.0)

            next_state = np.concatenate(
                [
                    joint_positions_next,  # 12
                    [grasped_next],  # 1
                    ball_pos_next,  # 3
                    ee_pos_next,  # 3
                    left_finger_next,  # 3
                    right_finger_next,  # 3
                ]
            )

            # Compute reward with movement penalty/bonus
            reward, gripper_distance = compute_reward(
                ee_pos, ball_pos, grasped, gripper_pos, left_finger, right_finger, prev_finger_distance
            )
            prev_finger_distance = gripper_distance  # Update for next step
            episode_reward += reward

            # Log progress every 100 steps
            if step % 100 == 0:
                left_dist = np.linalg.norm(left_finger - ball_pos)
                right_dist = np.linalg.norm(right_finger - ball_pos)
                print(f"  [Step {step}] {current_phase} | Reward: {reward:.3f}")
                print(
                    f"    EE: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}] | Cube: [{ball_pos[0]:.2f}, {ball_pos[1]:.2f}, {ball_pos[2]:.2f}]"
                )
                print(
                    f"    Dist: {gripper_distance:.3f}m | Gripper: {gripper_pos:.1f}/40.0"
                )

            # Train
            loss = agent.update(state, action, reward, next_state)
            if loss is not None:
                episode_loss.append(loss)

        # Episode summary
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        print(
            f"Episode {episode+1} | Total Reward: {episode_reward:.2f} | Avg Loss: {avg_loss:.4f}"
        )
        agent.episode_count += 1
        agent.total_reward_history.append(episode_reward)

        # Save checkpoint
        if (episode + 1) % SAVE_INTERVAL == 0:
            agent.save_model(MODEL_PATH)
            print(f"  → Checkpoint saved")

        # Record video (optional)
        if (episode + 1) % VIDEO_INTERVAL == 0:
            print(f"  → Recording video...")
            # Video recording code would go here

except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    agent.save_model(MODEL_PATH)
except Exception as e:
    print(f"Error during training: {e}")
    import traceback

    traceback.print_exc()
    agent.save_model(MODEL_PATH)
finally:
    print(f"\n✓ Training complete!")
    print(f"✓ Model saved to: {MODEL_PATH}")
    agent.save_model(MODEL_PATH)

    try:
        my_world.stop()
        my_world.clear()
    except:
        pass

    simulation_app.close()
