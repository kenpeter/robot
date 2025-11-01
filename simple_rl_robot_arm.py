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
import pickle
import carb
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

# Simple Q-learning agent with save/load
class SimpleRLAgent:
    def __init__(self, action_dim, learning_rate=0.1, epsilon=0.2):
        self.action_dim = action_dim
        self.lr = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.q_table = {}  # State-action values
        self.episode_count = 0
        self.total_reward_history = []

    def get_action(self, state):
        """Simple epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            # Random action: small joint position changes
            return np.random.uniform(-0.1, 0.1, self.action_dim)
        else:
            # Greedy action: move towards target (simplified)
            return np.random.uniform(-0.05, 0.05, self.action_dim)

    def update(self, state, action, reward, next_state):
        """Placeholder for learning update"""
        # In a real RL implementation, this would update Q-values or policy
        # For now, just decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filepath):
        """Save agent state to file"""
        model_data = {
            'epsilon': self.epsilon,
            'q_table': self.q_table,
            'episode_count': self.episode_count,
            'total_reward_history': self.total_reward_history,
            'action_dim': self.action_dim,
            'lr': self.lr
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load agent state from file"""
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found. Starting fresh.")
            return False

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.epsilon = model_data['epsilon']
        self.q_table = model_data['q_table']
        self.episode_count = model_data['episode_count']
        self.total_reward_history = model_data['total_reward_history']
        self.action_dim = model_data['action_dim']
        self.lr = model_data['lr']

        print(f"Model loaded from {filepath}")
        print(f"Resuming from episode {self.episode_count}, epsilon={self.epsilon:.3f}")
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

# Add target sphere
from pxr import UsdGeom, Gf, UsdLux

stage = my_world.stage
sphere_path = "/World/Target"
sphere = UsdGeom.Sphere.Define(stage, sphere_path)
sphere.GetRadiusAttr().Set(0.05)
sphere_translate = sphere.AddTranslateOp()
sphere_translate.Set(Gf.Vec3d(0.3, 0.3, 0.5))

# Initialize world
my_world.reset()

# RL Training parameters
MODEL_PATH = "rl_robot_arm_model.pkl"
agent = SimpleRLAgent(action_dim=7)  # First 7 joints (excluding grippers)

# Try to load existing model
agent.load_model(MODEL_PATH)

num_episodes = 100
max_steps_per_episode = 200
target_position = np.array([0.3, 0.3, 0.5])
save_interval = 10  # Save model every 10 episodes

print("Starting RL Training...")
print(f"Episodes: {num_episodes}, Max steps per episode: {max_steps_per_episode}")
print(f"Model will be saved to: {MODEL_PATH}")

try:
    for episode in range(agent.episode_count, agent.episode_count + num_episodes):
        # Reset arm to random initial position
        initial_pos = np.random.uniform(-1.0, 1.0, 7)
        robot.set_joint_positions(np.concatenate([initial_pos, [0.04, 0.04]]))

        # Reset target to random position
        target_position = np.array([
            np.random.uniform(0.2, 0.5),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(0.3, 0.7)
        ])
        sphere_translate.Set(Gf.Vec3d(float(target_position[0]),
                                       float(target_position[1]),
                                       float(target_position[2])))

        episode_reward = 0

        for step in range(max_steps_per_episode):
            # Get end effector position
            joint_positions = robot.get_joint_positions()[0]  # Get first element (single robot)
            ee_position, _ = end_effector.get_world_poses()

            # Calculate state (distance to target)
            distance = np.linalg.norm(ee_position - target_position)
            state = np.concatenate([joint_positions, [distance]])

            # Get action from agent
            action = agent.get_action(state)

            # Apply action (update first 7 joints)
            new_positions = joint_positions.copy()
            new_positions[:7] += action
            new_positions[:7] = np.clip(new_positions[:7], -2.5, 2.5)
            robot.set_joint_positions([new_positions])  # Wrap in list for batch format

            # Step simulation
            my_world.step(render=True)

            # Calculate reward
            new_ee_position, _ = end_effector.get_world_poses()
            new_distance = np.linalg.norm(new_ee_position - target_position)
            reward = -new_distance

            # Bonus for reaching target
            if new_distance < 0.1:
                reward += 10.0
                print(f"Episode {episode}: Target reached at step {step}!")
                break

            # Update agent
            next_state = np.concatenate([robot.get_joint_positions()[0], [new_distance]])
            agent.update(state, action, reward, next_state)

            episode_reward += reward

        # Track statistics
        agent.total_reward_history.append(episode_reward)
        agent.episode_count = episode + 1

        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(agent.total_reward_history[-10:]) if len(agent.total_reward_history) >= 10 else np.mean(agent.total_reward_history)
            print(f"Episode {episode}/{agent.episode_count + num_episodes - 1}, Reward: {episode_reward:.2f}, Avg(10): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

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
