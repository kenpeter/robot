# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
IsaacLab-Style Lift Task with DDPG
Exact implementation of IsaacLab's reward structure and observations
"""

from isaacsim import SimulationApp

# Initialize simulation
simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 800,
        "height": 600,
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
import random
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


# === DDPG AGENT ===
class Actor(nn.Module):
    """Actor network with IsaacLab-style compact architecture"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.elu(self.l1(state))
        x = F.elu(self.l2(x))
        return torch.tanh(self.l3(x))


class Critic(nn.Module):
    """Critic network with IsaacLab-style compact architecture"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))
        return self.l3(x)


class DDPGAgent:
    def __init__(self, state_dim, action_dim, device="cuda"):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor (Policy)
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        # Critic (Q-Value)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.buffer = deque(maxlen=100000)
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.005

        self.noise = 0.2
        self.step_count = 0
        self.episode_count = 0

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()

        if not deterministic:
            noise = np.random.normal(0, self.noise, size=self.action_dim)
            action = action + noise

        return np.clip(action, -1.0, 1.0)

    def update(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

        if len(self.buffer) < self.batch_size:
            return None

        # Sample Batch
        batch = random.sample(self.buffer, self.batch_size)
        state_batch = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        action_batch = torch.FloatTensor(np.array([x[1] for x in batch])).to(self.device)
        reward_batch = torch.FloatTensor(np.array([x[2] for x in batch])).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([x[3] for x in batch])).to(self.device)
        done_batch = torch.FloatTensor(np.array([x[4] for x in batch])).unsqueeze(1).to(self.device)

        # Train Critic
        with torch.no_grad():
            next_action = self.actor_target(next_state_batch)
            target_Q = self.critic_target(next_state_batch, next_action)
            target_Q = reward_batch + (1 - done_batch) * self.gamma * target_Q

        current_Q = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Train Actor
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Soft Update Target Networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Decay noise
        self.noise = max(0.05, self.noise * 0.9999)
        self.step_count += 1

        return critic_loss.item()

    def save_model(self, filepath):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "episode": self.episode_count,
            },
            filepath,
        )

    def load_model(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.actor.load_state_dict(checkpoint["actor"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.episode_count = checkpoint["episode"]
            print(f"Loaded model from {filepath}")


# === IsaacLab-Style Reward Functions ===
def object_ee_distance(ee_pos, object_pos, std=0.1):
    """Reward for reaching the object using tanh-kernel (IsaacLab exact)"""
    distance = np.linalg.norm(object_pos - ee_pos)
    return 1.0 - np.tanh(distance / std)


def object_is_lifted(object_pos, minimal_height=0.04):
    """Reward for lifting the object above minimal height (IsaacLab exact)"""
    return 1.0 if object_pos[2] > minimal_height else 0.0


def object_goal_distance(object_pos, goal_pos, minimal_height=0.04, std=0.3):
    """Reward for tracking goal pose (IsaacLab exact)"""
    distance = np.linalg.norm(goal_pos - object_pos)
    # Only reward if object is lifted
    if object_pos[2] > minimal_height:
        return 1.0 - np.tanh(distance / std)
    return 0.0


def compute_isaaclab_reward(ee_pos, object_pos, goal_pos, action, prev_action, joint_vel):
    """
    Compute reward using IsaacLab's exact configuration:
    - reaching_object: weight=1.0, std=0.1
    - lifting_object: weight=15.0, minimal_height=0.04
    - object_goal_tracking: weight=16.0, std=0.3
    - object_goal_tracking_fine_grained: weight=5.0, std=0.05
    - action_rate: weight=-1e-4
    - joint_vel: weight=-1e-4
    """
    # Reaching reward
    reaching = object_ee_distance(ee_pos, object_pos, std=0.1) * 1.0

    # Lifting reward
    lifting = object_is_lifted(object_pos, minimal_height=0.04) * 15.0

    # Goal tracking (coarse)
    goal_tracking = object_goal_distance(object_pos, goal_pos, minimal_height=0.04, std=0.3) * 16.0

    # Goal tracking (fine-grained)
    goal_tracking_fine = object_goal_distance(object_pos, goal_pos, minimal_height=0.04, std=0.05) * 5.0

    # Action rate penalty (L2 norm)
    action_rate_penalty = -1e-4 * np.sum((action - prev_action) ** 2)

    # Joint velocity penalty (L2 norm)
    joint_vel_penalty = -1e-4 * np.sum(joint_vel ** 2)

    total_reward = reaching + lifting + goal_tracking + goal_tracking_fine + action_rate_penalty + joint_vel_penalty

    return total_reward, {
        'reaching': reaching,
        'lifting': lifting,
        'goal_tracking': goal_tracking,
        'goal_tracking_fine': goal_tracking_fine,
        'action_rate': action_rate_penalty,
        'joint_vel': joint_vel_penalty
    }


# === SCENE SETUP ===
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

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

set_camera_view(
    eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.0], camera_prim_path="/OmniverseKit_Persp"
)

# Cube setup
from isaacsim.core.api.objects import DynamicCuboid

cube_size = 0.05
ball = DynamicCuboid(
    name="red_cube",
    position=np.array([0.5, 0.0, 0.055]),
    prim_path="/World/Cube",
    scale=np.array([cube_size, cube_size, cube_size]),
    color=np.array([1.0, 0.0, 0.0]),
    mass=0.1,
)
my_world.scene.add(ball)

# RMPFlow
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

# IsaacLab-style observations:
# joint_pos_rel (12) + joint_vel_rel (12) + object_position (3) + target_object_position (3) + actions (4) = 34
state_dim = 34
action_dim = 4  # dx, dy, dz, gripper
agent = DDPGAgent(state_dim, action_dim)
MODEL_PATH = "isaaclab_lift_ddpg.pth"

my_world.reset()
robot.initialize()
ball.initialize()

print("Stabilizing physics...")
for _ in range(60):
    my_world.step(render=False)

# Default joint positions for relative observations
default_joint_pos = np.array([0, -1.57, 1.57, -1.57, -1.57, 0, 0, 0, 0, 0, 0, 0])
robot.set_joint_positions(default_joint_pos)
for _ in range(30):
    my_world.step(render=False)

# === TRAINING LOOP ===
MAX_EPISODES = 1000
MAX_STEPS = 250  # 5 seconds at 50Hz (decimation=2, sim=100Hz)

print("Starting IsaacLab-Style DDPG Training...")
print(f"State dim: {state_dim}, Action dim: {action_dim}")
print("Reward structure:")
print("  - reaching_object: weight=1.0, std=0.1")
print("  - lifting_object: weight=15.0")
print("  - object_goal_tracking: weight=16.0, std=0.3")
print("  - object_goal_tracking_fine: weight=5.0, std=0.05")
print("  - action_rate: weight=-1e-4")
print("  - joint_vel: weight=-1e-4")

for episode in range(MAX_EPISODES):
    # Reset
    # Randomize Cube position (IsaacLab range: x:(-0.1, 0.1), y:(-0.25, 0.25))
    cube_x = 0.5 + np.random.uniform(-0.1, 0.1)
    cube_y = np.random.uniform(-0.25, 0.25)

    # Sample goal position (IsaacLab range: x:(0.4, 0.6), y:(-0.25, 0.25), z:(0.25, 0.5))
    goal_x = np.random.uniform(0.4, 0.6)
    goal_y = np.random.uniform(-0.25, 0.25)
    goal_z = np.random.uniform(0.25, 0.5)
    goal_pos = np.array([goal_x, goal_y, goal_z])

    # Teleport cube
    from pxr import Gf, UsdGeom
    cube_prim = my_world.stage.GetPrimAtPath("/World/Cube")
    xform = UsdGeom.Xformable(cube_prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(cube_x, cube_y, 0.055))
            break

    # Reset Robot (IsaacLab uses scale factor 0.5-1.5 for randomization)
    scale = np.random.uniform(0.5, 1.5, 12)
    reset_joint_pos = default_joint_pos * scale
    robot.set_joint_positions(reset_joint_pos)

    for _ in range(10):
        my_world.step(render=False)

    episode_reward = 0
    prev_action = np.zeros(4)

    for step in range(MAX_STEPS):
        # 1. Get State (IsaacLab-style)
        joint_pos = robot.get_joint_positions()
        if len(joint_pos) < 12:
            joint_pos = np.pad(joint_pos, (0, 12 - len(joint_pos)))
        joint_pos = joint_pos[:12]

        joint_vel = robot.get_joint_velocities()
        if len(joint_vel) < 12:
            joint_vel = np.pad(joint_vel, (0, 12 - len(joint_vel)))
        joint_vel = joint_vel[:12]

        ee_pos, ee_rot = robot.end_effector.get_world_pose()
        ball_pos, _ = ball.get_world_pose()

        # IsaacLab observations
        joint_pos_rel = joint_pos - default_joint_pos  # Relative to default
        joint_vel_rel = joint_vel  # Already relative
        object_position = ball_pos - np.array([0, 0, 0])  # In world frame (simplified)
        target_object_position = goal_pos  # Target position

        # Build state vector
        state = np.concatenate([
            joint_pos_rel,          # 12
            joint_vel_rel,          # 12
            object_position,        # 3
            target_object_position, # 3
            prev_action             # 4
        ])  # Total: 34

        # 2. Get Action
        action = agent.get_action(state)

        # 3. Execute Action (IsaacLab-style: scale=0.5)
        pos_action = action[:3] * 0.5 * 0.02  # Scale by 0.5, then by movement size
        target_pos = ee_pos + pos_action

        # Workspace clamping
        target_pos[0] = np.clip(target_pos[0], 0.2, 0.7)
        target_pos[1] = np.clip(target_pos[1], -0.4, 0.4)
        target_pos[2] = np.clip(target_pos[2], 0.02, 0.6)

        rmp_flow.set_end_effector_target(
            target_position=target_pos, target_orientation=None
        )

        # Gripper action
        grip_action = action[3] * 5.0
        current_joints = robot.get_joint_positions()
        if len(current_joints) > 6:
            new_grip = np.clip(current_joints[6] + grip_action, 0, 40)
            current_joints[6] = new_grip
            robot.set_joint_positions(current_joints)

        robot.apply_action(motion_policy.get_next_articulation_action(physics_dt))
        my_world.step(render=True)

        # 4. Get Next State
        next_joint_pos = robot.get_joint_positions()[:12]
        next_joint_vel = robot.get_joint_velocities()[:12]
        next_ee_pos, _ = robot.end_effector.get_world_pose()
        next_ball_pos, _ = ball.get_world_pose()

        next_joint_pos_rel = next_joint_pos - default_joint_pos
        next_object_position = next_ball_pos

        next_state = np.concatenate([
            next_joint_pos_rel,
            next_joint_vel,
            next_object_position,
            target_object_position,
            action
        ])

        # 5. Calculate Reward (IsaacLab-style)
        reward, reward_breakdown = compute_isaaclab_reward(
            next_ee_pos, next_ball_pos, goal_pos, action, prev_action, next_joint_vel
        )

        # Done condition
        done = False
        if step == MAX_STEPS - 1:
            done = True
        # Termination: object dropping below -0.05 (IsaacLab)
        if next_ball_pos[2] < -0.05:
            done = True

        # 6. Update Agent
        loss = agent.update(state, action, reward, next_state, float(done))

        episode_reward += reward
        prev_action = action.copy()

        if step % 50 == 0:
            print(f"  Step {step} | Reward: {reward:.3f} | Reaching: {reward_breakdown['reaching']:.3f} | Lifting: {reward_breakdown['lifting']:.3f}")

        if done:
            break

    agent.episode_count += 1
    print(f"Episode {episode} | Total Reward: {episode_reward:.2f} | Loss: {loss if loss else 0:.4f} | Noise: {agent.noise:.4f}")

    # Save periodically
    if episode % 50 == 0 and episode > 0:
        agent.save_model(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

print("Training complete!")
agent.save_model(MODEL_PATH)
simulation_app.close()
