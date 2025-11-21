# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ONLINE RL Training - DDPG VERSION
Fixes wandering by using Actor-Critic logic and Relative Coordinates
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


# === DDPG AGENT (Fixes Wandering) ===
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        # Tanh bounds output to [-1, 1]
        return torch.tanh(self.l3(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Critic takes (State + Action) and outputs Q-Value
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


# deep determine policy agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, device="cuda"):
        # device, state dim, action dim
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor (Policy)
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        # Critic (Q-Value)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.buffer = deque(maxlen=100000)
        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 0.005  # Soft update rate

        self.noise = 0.2
        self.step_count = 0
        self.episode_count = 0
        self.loss_history = []
        self.total_reward_history = []

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()

        if not deterministic:
            # Add exploration noise
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
        action_batch = torch.FloatTensor(np.array([x[1] for x in batch])).to(
            self.device
        )
        reward_batch = (
            torch.FloatTensor(np.array([x[2] for x in batch]))
            .unsqueeze(1)
            .to(self.device)
        )
        next_state_batch = torch.FloatTensor(np.array([x[3] for x in batch])).to(
            self.device
        )
        done_batch = (
            torch.FloatTensor(np.array([x[4] for x in batch]))
            .unsqueeze(1)
            .to(self.device)
        )

        # 1. Train Critic
        with torch.no_grad():
            next_action = self.actor_target(next_state_batch)
            target_Q = self.critic_target(next_state_batch, next_action)
            target_Q = reward_batch + (1 - done_batch) * self.gamma * target_Q

        current_Q = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # Clip gradients
        self.critic_optimizer.step()

        # 2. Train Actor
        # The actor tries to maximize the Critic's estimated Q-value
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # Clip gradients
        self.actor_optimizer.step()

        # 3. Soft Update Target Networks
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # Decay noise
        self.noise = max(0.05, self.noise * 0.99995)
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
    position=np.array([0.5, 0.0, 0.025]),
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

# State: 12 joints + 3 delta_pos + 3 ball_pos + 3 ee_pos + 1 gripper = 22
state_dim = 22
action_dim = 4  # dx, dy, dz, gripper
agent = DDPGAgent(state_dim, action_dim)
MODEL_PATH = "ddpg_robot_arm.pth"

my_world.reset()
robot.initialize()
ball.initialize()

print("Stabilizing physics...")
for _ in range(60):
    my_world.step(render=False)


def get_fingers(ee_pos, ee_rot, gripper_val):
    # Simplified approximate finger positions based on EE
    # gripper_val: 0 (open) to 40 (closed)
    openness = 1.0 - (gripper_val / 40.0)

    # Local offset approx
    # Typically Z is forward for gripper, Y is grasp axis
    # We just need relative cloud for now, exact math not critical for dense reward if close
    return ee_pos  # Just return EE center for simple tracking in this snippet


def compute_reward(ee_pos, ball_pos, gripper_val):
    dist = np.linalg.norm(ee_pos - ball_pos)

    # Distance-based reward: closer = higher reward (stable, no velocity dependence)
    # Map distance [0, 1] to reward [10, -10] roughly
    # At dist=0: reward=10, at dist=0.5: reward=0, at dist>=1: reward=-10
    dist_reward = 10.0 - 20.0 * min(dist, 1.0)

    # Small bonus for being very close
    if dist < 0.1:
        dist_reward += 2.0

    # Grasp bonus
    if dist < 0.1 and gripper_val > 20:
        dist_reward += 3.0
        if ball_pos[2] > 0.05:  # Lifted
            dist_reward += 5.0

    # Clip to [-10, 10]
    return np.clip(dist_reward, -10.0, 10.0)


# === TRAINING LOOP ===
MAX_EPISODES = 1000
MAX_STEPS = 2000

print("Starting DDPG Training...")

for episode in range(MAX_EPISODES):
    # Reset
    # Randomize Cube
    cube_x = 0.4 + np.random.uniform(-0.1, 0.1)
    cube_y = np.random.uniform(-0.1, 0.1)

    # Use Teleport to reset cube
    from pxr import Gf, UsdGeom

    cube_prim = my_world.stage.GetPrimAtPath("/World/Cube")
    xform = UsdGeom.Xformable(cube_prim)
    # Find translate op
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(cube_x, cube_y, 0.025))
            break

    # Reset Robot
    robot.set_joint_positions(
        np.array([0, -1.57, 1.57, -1.57, -1.57, 0, 0, 0, 0, 0, 0, 0])
    )
    for _ in range(10):
        my_world.step(render=False)

    episode_reward = 0
    prev_action = np.zeros(4)  # For action smoothing

    for step in range(MAX_STEPS):
        # 1. Get State
        joint_pos = robot.get_joint_positions()
        if len(joint_pos) < 12:
            joint_pos = np.pad(joint_pos, (0, 12 - len(joint_pos)))
        joint_pos = joint_pos[:12]

        ee_pos, ee_rot = robot.end_effector.get_world_pose()
        ball_pos, _ = ball.get_world_pose()

        # CRITICAL FIX: Relative Coordinates
        # The network learns "how far to go" much faster than "where to go in absolute space"
        delta_pos = ball_pos - ee_pos

        gripper_val = joint_pos[6] if len(joint_pos) > 6 else 0

        # Normalize observations for stable training
        state = np.concatenate(
            [
                joint_pos / np.pi,  # 12 - normalized to ~[-1, 1]
                delta_pos * 2.0,  # 3 - scale relative position
                ball_pos * 2.0,  # 3 - normalized positions
                ee_pos * 2.0,  # 3 - normalized positions
                [gripper_val / 40.0],  # 1 (Normalized gripper)
            ]
        )  # Total 22

        # 2. Get Action
        action = agent.get_action(state)

        # 3. Execute Action (With Clamping)
        # Action is [-1, 1]. Scale to speed.
        pos_action = action[:3] * 0.02  # 2cm max per step (smoother)

        target_pos = ee_pos + pos_action

        # CRITICAL FIX: Workspace Clamping
        # Don't let the robot wander off the table!
        target_pos[0] = np.clip(target_pos[0], 0.2, 0.7)  # X bounds
        target_pos[1] = np.clip(target_pos[1], -0.4, 0.4)  # Y bounds
        target_pos[2] = np.clip(target_pos[2], 0.02, 0.6)  # Z bounds

        rmp_flow.set_end_effector_target(
            target_position=target_pos, target_orientation=None
        )

        # Gripper action
        grip_action = action[3] * 5.0  # speed
        current_joints = robot.get_joint_positions()
        if len(current_joints) > 6:
            new_grip = np.clip(current_joints[6] + grip_action, 0, 40)
            current_joints[6] = new_grip
            robot.set_joint_positions(current_joints)

        robot.apply_action(motion_policy.get_next_articulation_action(physics_dt))
        my_world.step(render=True)

        # 4. Get Next State
        next_joint_pos = robot.get_joint_positions()[:12]
        next_ee_pos, _ = robot.end_effector.get_world_pose()
        next_ball_pos, _ = ball.get_world_pose()
        next_delta = next_ball_pos - next_ee_pos
        next_grip = next_joint_pos[6] if len(next_joint_pos) > 6 else 0

        # Normalize next_state the same way
        next_state = np.concatenate(
            [next_joint_pos / np.pi, next_delta * 2.0, next_ball_pos * 2.0, next_ee_pos * 2.0, [next_grip / 40.0]]
        )

        # 5. Calculate Reward
        reward = compute_reward(next_ee_pos, next_ball_pos, next_grip)

        # Action smoothing penalty - penalize jerky movements
        action_penalty = -0.5 * np.sum((action - prev_action) ** 2)
        reward += action_penalty
        reward = np.clip(reward, -10.0, 10.0)
        prev_action = action.copy()

        # Done condition
        done = False
        if step == MAX_STEPS - 1:
            done = True

        # 6. Update Agent
        loss = agent.update(state, action, reward, next_state, float(done))

        episode_reward += reward

        if step % 50 == 0:
            print(
                f"  Step {step} | Dist: {np.linalg.norm(delta_pos):.3f} | Rew: {reward:.3f}"
            )

    print(
        f"Episode {episode} | Total Reward: {episode_reward:.2f} | Loss: {loss if loss else 0:.4f}"
    )

    if episode % 10 == 0:
        agent.save_model(MODEL_PATH)

simulation_app.close()
