# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SUPER SIMPLE ONLINE RL Training - Minimal version for controlled training
No vision, short episodes, basic logging. Focus on proprioception + relative pos.
"""

# sim app
from isaacsim import SimulationApp

# Initialize simulation
simulation_app = SimulationApp(
    {
        "headless": False,  # Enable UI
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
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot_motion.motion_generation import (
    ArticulationMotionPolicy,
    RmpFlow,
)

print("Super simple DiT with standard attention - no vision")


# Standard Multi-Head Attention
class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        q = self.Wq(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_probs = F.softmax(attn_scores, dim=-1)
        o = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(B, T, D)
        o = self.Wo(o)
        return o


# Simple DiT Block
class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn = MultiheadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 4 * hidden_dim)
        )

    def forward(self, x, c):
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(
            4, dim=-1
        )
        x_norm1 = self.norm1(x) * (1 + scale_msa.unsqueeze(1))
        attn_out = self.attn(x_norm1)
        x = x + attn_out * gate_msa.unsqueeze(1)
        x_norm2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm2)
        return x


# Simple Diffusion Transformer (no vision)
class DiffusionTransformer(nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_dim=128, num_layers=3, num_heads=4
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, noisy_action, state, timestep):
        t_emb = self.time_embed(timestep)
        s_emb = self.state_encoder(state)
        a_emb = self.action_encoder(noisy_action)
        c = t_emb + s_emb
        x = a_emb.unsqueeze(1)
        for block in self.blocks:
            x = block(x, c)
        x = x.squeeze(1)
        return self.final_layer(x)


# Simple Agent
class DiTAgent:
    def __init__(self, state_dim, action_dim, device="cuda"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.num_diffusion_steps = 10
        self.betas = torch.linspace(
            0.0001, 0.02, self.num_diffusion_steps, dtype=torch.float32
        ).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod = torch.clamp(self.alphas_cumprod, min=1e-8, max=1.0)

        self.model = DiffusionTransformer(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.buffer = deque(maxlen=5000)  # Smaller buffer
        self.batch_size = 32  # Smaller batch
        self.noise_scale = 0.1
        self.step_count = 0

    def get_action(self, state, deterministic=False):
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = (
                torch.randn(1, self.action_dim, dtype=torch.float32).to(self.device)
                * 0.5
            )
            for t in reversed(range(self.num_diffusion_steps)):
                timestep = torch.FloatTensor([[t / self.num_diffusion_steps]]).to(
                    self.device
                )
                predicted_noise = self.model(action, state_tensor, timestep)
                if torch.isnan(predicted_noise).any():
                    predicted_noise = torch.zeros_like(predicted_noise)
                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]
                coef1 = 1.0 / torch.sqrt(alpha_t + 1e-8)
                coef2 = beta_t / (torch.sqrt(1.0 - alpha_cumprod_t + 1e-8))
                action = coef1 * (action - coef2 * predicted_noise)
                action = torch.clamp(action, -10.0, 10.0)
                if t > 0:
                    action = action + torch.randn_like(action) * 0.1
            if not deterministic:
                action = action + self.noise_scale * 0.05 * torch.randn_like(action)
            action = torch.clamp(action, -0.5, 0.5)
            if torch.isnan(action).any():
                action = torch.randn_like(action) * 0.3
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
        self.model.train()
        t = torch.randint(
            0, self.num_diffusion_steps, (self.batch_size,), dtype=torch.long
        ).to(self.device)
        noise = torch.randn_like(actions)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1)
        noisy_actions = (
            torch.sqrt(alpha_cumprod_t + 1e-8) * actions
            + torch.sqrt(1.0 - alpha_cumprod_t + 1e-8) * noise
        )
        timesteps = (t.float() / self.num_diffusion_steps).view(-1, 1)
        predicted_noise = self.model(noisy_actions, states, timesteps)
        per_sample_loss = F.mse_loss(predicted_noise, noise, reduction="none").mean(
            dim=1
        )
        reward_weights = torch.sigmoid(rewards / 10.0)
        weighted_loss = (per_sample_loss * reward_weights).mean()
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.step_count += 1
        self.noise_scale = max(0.05, self.noise_scale * 0.995)
        return weighted_loss.item()

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        if os.path.exists(filepath):
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            print(f"Model loaded from {filepath}")
            return True
        return False


# Setup scene (minimal)
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

set_camera_view(
    eye=[2.5, 2.5, 2.0], target=[0.0, 0.0, 0.5], camera_prim_path="/OmniverseKit_Persp"
)

# UR10e setup
asset_path = (
    assets_root_path
    + "/Isaac/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd"
)
add_reference_to_stage(usd_path=asset_path, prim_path="/World/ur")

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

from isaacsim.core.api.objects import DynamicCuboid
from pxr import Gf, UsdLux
import isaacsim.robot_motion.motion_generation as mg

# Cube
cube_size = 0.0515
cube = DynamicCuboid(
    name="red_cube",
    position=np.array([0.5, 0.2, cube_size / 2.0]),
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
dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(1000.0)
distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
distant_light.CreateIntensityAttr(2000.0)
distant_light_xform = distant_light.AddRotateXYZOp()
distant_light_xform.Set(Gf.Vec3f(-45, 0, 0))

my_world.reset()
robot.initialize()
ball.initialize()

# RMPflow
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

# Params
MODEL_PATH = "simple_model.pth"
state_dim = 16  # 12 joints + 3 rel_pos + 1 grasped
action_dim = 4
agent = DiTAgent(state_dim, action_dim)
agent.load_model(MODEL_PATH)

NUM_EPISODES = 200  # Short for control
EPISODE_LENGTH = 200  # Short episodes

print("\n=== SIMPLE TRAINING START ===")
print(f"Episodes: {NUM_EPISODES}, Length: {EPISODE_LENGTH}")
my_world.play()


def compute_reward(ee_pos, ball_pos, gripper_pos, prev_dist):
    dist = np.linalg.norm(ee_pos - ball_pos)
    reward = -dist * 0.1
    if dist < 0.05 and gripper_pos > 0.02:
        reward += 10.0
    reward += (prev_dist - dist) * 0.01 - 0.01
    return reward, dist


try:
    for episode in range(NUM_EPISODES):
        # Reset
        initial_pos = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
        gripper_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot.set_joint_positions(np.concatenate([initial_pos, gripper_open]))
        cube_x = np.random.uniform(0.45, 0.55)  # Closer spawn for control
        cube_y = np.random.uniform(-0.1, 0.1)
        ball.set_world_pose(position=np.array([cube_x, cube_y, cube_size / 2.0]))
        for _ in range(10):
            my_world.step(render=False)

        episode_reward = 0
        prev_dist = 1.0
        deterministic = episode % 50 == 0  # Eval every 50

        my_world.step(render=True)  # First step

        for step in range(EPISODE_LENGTH):
            # Current obs
            joint_positions = robot.get_joint_positions()
            if isinstance(joint_positions, tuple):
                joint_positions = joint_positions[0]
            ball_pos = np.array(ball.get_world_pose()[0]).flatten()
            ee_pos = np.array(robot.end_effector.get_world_pose()[0]).flatten()
            gripper_pos = joint_positions[6] if len(joint_positions) > 6 else 0.0
            rel_pos = ball_pos - ee_pos
            grasped = float(
                np.linalg.norm(ee_pos - ball_pos) < 0.15 and gripper_pos > 0.02
            )
            state = np.concatenate([joint_positions, rel_pos, [grasped]])

            # Act
            action = agent.get_action(state, deterministic=deterministic)

            # Execute
            delta_pos = action[:3] * 0.03
            target_position = ee_pos + delta_pos
            target_position = np.clip(
                target_position, [-0.6, -0.6, 0.05], [0.8, 0.6, 1.0]
            )
            rmp_flow.set_end_effector_target(
                target_position=target_position, target_orientation=None
            )
            actions = motion_policy.get_next_articulation_action(physics_dt)
            robot.apply_action(actions)

            # Gripper
            gripper_action = np.clip(action[3], -1.0, 1.0)
            current_joints = robot.get_joint_positions()
            if isinstance(current_joints, tuple):
                current_joints = current_joints[0].copy()
            else:
                current_joints = current_joints.copy()
            if len(current_joints) > 6:
                current_gripper = current_joints[6]
                target_gripper = np.clip(
                    current_gripper + gripper_action * 0.01, 0.0, 0.04
                )
                current_joints[6] = target_gripper
                robot.set_joint_positions(current_joints)

            # Step & next obs
            my_world.step(render=True)
            next_joint_positions = robot.get_joint_positions()
            if isinstance(next_joint_positions, tuple):
                next_joint_positions = next_joint_positions[0]
            next_ball_pos = np.array(ball.get_world_pose()[0]).flatten()
            next_ee_pos = np.array(robot.end_effector.get_world_pose()[0]).flatten()
            next_gripper_pos = (
                next_joint_positions[6] if len(next_joint_positions) > 6 else 0.0
            )
            next_rel_pos = next_ball_pos - next_ee_pos
            next_grasped = float(
                np.linalg.norm(next_ee_pos - next_ball_pos) < 0.15
                and next_gripper_pos > 0.02
            )
            next_state = np.concatenate(
                [next_joint_positions, next_rel_pos, [next_grasped]]
            )

            # Reward
            reward, curr_dist = compute_reward(
                next_ee_pos, next_ball_pos, next_gripper_pos, prev_dist
            )
            prev_dist = curr_dist

            # Update
            loss = agent.update(state, action, reward, next_state)
            if step % 50 == 0:
                print(
                    f"Ep {episode+1} Step {step}: R={reward:.2f} D={curr_dist:.2f} L={loss:.4f if loss else 'N/A'}"
                )

            episode_reward += reward

        avg_reward = episode_reward / EPISODE_LENGTH
        print(f"Episode {episode+1}: Avg R = {avg_reward:.3f}")

        if (episode + 1) % 50 == 0:
            agent.save_model(MODEL_PATH)

except KeyboardInterrupt:
    print("\nInterrupted")
    agent.save_model(MODEL_PATH)
finally:
    my_world.stop()
    my_world.clear()
    simulation_app.close()
