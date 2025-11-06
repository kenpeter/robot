#!/usr/bin/env python3
"""
Collect synthetic expert dataset - exact copy of pick_up_example.py logic
but running 10 episodes with data collection.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import pickle
import sys
import os

# Add Isaac Sim path to import the controller and task
isaac_path = "/home/kenpeter/work/isaac-sim-standalone-5.1.0-linux-x86_64/standalone_examples/api/isaacsim.robot.manipulators/ur10e"
sys.path.insert(0, isaac_path)

from controller.pick_place import PickPlaceController
from isaacsim.core.api import World
from tasks.pick_place import PickPlace
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf

# Configuration
NUM_EPISODES = 10
SAVE_PATH = "/home/kenpeter/work/robot/synthetic_dataset.pkl"

# Dataset storage
dataset = {
    'states': [],
    'actions': [],
    'next_states': [],
    'images': [],
    'rewards': [],
}

print("\n" + "=" * 70)
print(f" SYNTHETIC DATASET COLLECTION - {NUM_EPISODES} Episodes")
print("=" * 70)

successful_demos = 0

# Create world - EXACT same parameters as pick_up_example.py
my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=20 / 200)

# Create initial task - EXACT same setup
target_position = np.array([-0.3, 0.6, 0])
target_position[2] = 0.0515 / 2.0
my_task = PickPlace(name="ur10e_pick_place", target_position=target_position, cube_size=np.array([0.1, 0.0515, 0.1]))
my_world.add_task(my_task)
my_world.reset()

# Get task objects - EXACT same as pick_up_example.py
task_params = my_world.get_task("ur10e_pick_place").get_params()
ur10e_name = task_params["robot_name"]["value"]
cube_name = task_params["cube_name"]["value"]
my_ur10e = my_world.scene.get_object(ur10e_name)

# Create controller - EXACT same as pick_up_example.py
my_controller = PickPlaceController(name="controller", robot_articulation=my_ur10e, gripper=my_ur10e.gripper)
articulation_controller = my_ur10e.get_articulation_controller()

# Set up overhead camera for data collection
camera_prim_path = "/World/OverheadCamera"
create_prim(camera_prim_path, "Camera")
eye = Gf.Vec3d(0.0, 0.0, 1.2)
target_vec = Gf.Vec3d(0.0, 0.0, 0.0)
set_camera_view(eye=eye, target=target_vec, camera_prim_path=camera_prim_path)
camera_prim = my_world.stage.GetPrimAtPath(camera_prim_path)
camera_prim.GetAttribute("horizontalAperture").Set(80.0)
camera_prim.GetAttribute("verticalAperture").Set(80.0)
overhead_camera = Camera(prim_path=camera_prim_path, resolution=(84, 84))
overhead_camera.initialize()

# Episode tracking
current_episode = 0
episode_data = []
step_count = 0

# EXACT same control flow as pick_up_example.py
reset_needed = False
task_completed = False

try:
    while simulation_app.is_running() and current_episode < NUM_EPISODES:
        my_world.step(render=True)

        if my_world.is_playing():
            if reset_needed:
                # Episode just finished - process data
                if task_completed and len(episode_data) > 0:
                    print(f"  ✓ Episode {current_episode} completed with {len(episode_data)} steps")

                    # Post-process to fix next_state
                    for i in range(len(episode_data) - 1):
                        episode_data[i]['next_state'] = episode_data[i + 1]['state']

                    # Save episode data
                    for exp in episode_data:
                        dataset['states'].append(exp['state'])
                        dataset['actions'].append(exp['action'])
                        dataset['next_states'].append(exp['next_state'])
                        dataset['images'].append(exp['image'])
                        dataset['rewards'].append(exp['reward'])

                    successful_demos += 1
                else:
                    print(f"  ✗ Episode {current_episode} failed")

                # Start new episode
                current_episode += 1
                if current_episode < NUM_EPISODES:
                    print(f"\n--- Episode {current_episode + 1}/{NUM_EPISODES} ---")
                    episode_data = []
                    step_count = 0

                my_world.reset()
                reset_needed = False
                my_controller.reset()
                task_completed = False

            if my_world.current_time_step_index == 0:
                my_controller.reset()
                if current_episode == 0:
                    print(f"\n--- Episode 1/{NUM_EPISODES} ---")

            # Get observations - EXACT same as pick_up_example.py
            observations = my_world.get_observations()

            # Capture camera image
            overhead_camera.get_current_frame()
            rgba_data = overhead_camera.get_rgba()
            if rgba_data is None or rgba_data.size == 0:
                rgb_image = np.zeros((84, 84, 3), dtype=np.uint8)
            else:
                if len(rgba_data.shape) == 1:
                    rgba_data = rgba_data.reshape(84, 84, 4)
                rgb_image = rgba_data[:, :, :3].astype(np.uint8)

            # Build state
            current_joints = observations[ur10e_name]["joint_positions"]
            cube_pos = observations[cube_name]["position"]
            ee_pos, _ = my_ur10e.end_effector.get_world_pose()
            ee_pos = np.array(ee_pos)
            cube_distance = np.linalg.norm(ee_pos - cube_pos)

            # Get gripper joint for grasped detection
            gripper_joint = current_joints[6] if len(current_joints) > 6 else 0.0
            grasped = float(cube_distance < 0.15 and gripper_joint > 20.0)

            state = np.concatenate([current_joints, [grasped]])

            # Forward the observation values to the controller - EXACT same as pick_up_example.py
            actions = my_controller.forward(
                picking_position=observations[cube_name]["position"],
                placing_position=observations[cube_name]["target_position"],
                current_joint_positions=observations[ur10e_name]["joint_positions"],
                end_effector_offset=np.array([0, 0, 0.20]),
            )

            # Check if done - EXACT same as pick_up_example.py
            if my_controller.is_done() and not task_completed:
                print(f"  ✓ done picking and placing at step {step_count}")
                task_completed = True

            # Apply actions - EXACT same as pick_up_example.py
            articulation_controller.apply_action(actions)

            # Stop this episode after task completion to save data
            if task_completed:
                my_world.stop()
                reset_needed = True

            # Simple RL action encoding (direction towards target)
            target_pos = observations[cube_name]["target_position"]
            direction_to_target = target_pos - cube_pos
            direction_to_target = direction_to_target / (np.linalg.norm(direction_to_target) + 1e-6)
            rl_action = np.zeros(4, dtype=np.float32)
            rl_action[:3] = direction_to_target[:3] * 0.1
            rl_action[3] = 1.0 if grasped else -1.0

            # Reward calculation
            reward = -cube_distance * 1.0
            if grasped:
                reward += 5.0
            if task_completed:
                reward += 10.0

            episode_data.append({
                'state': state,
                'action': rl_action,
                'next_state': state,  # Will fix in post-processing
                'image': rgb_image,
                'reward': reward,
            })

            step_count += 1

            # Debug output every 100 steps
            if step_count % 100 == 0:
                print(f"    Step {step_count}: Distance: {cube_distance:.3f}, Grasped: {grasped}")

        if my_world.is_stopped():
            reset_needed = True

except KeyboardInterrupt:
    print("\n\n⚠ Collection interrupted by user")

# Convert to numpy arrays
dataset['states'] = np.array(dataset['states'], dtype=np.float32) if dataset['states'] else np.array([])
dataset['actions'] = np.array(dataset['actions'], dtype=np.float32) if dataset['actions'] else np.array([])
dataset['next_states'] = np.array(dataset['next_states'], dtype=np.float32) if dataset['next_states'] else np.array([])
dataset['images'] = np.array(dataset['images'], dtype=np.uint8) if dataset['images'] else np.array([])
dataset['rewards'] = np.array(dataset['rewards'], dtype=np.float32) if dataset['rewards'] else np.array([])

# Save dataset
with open(SAVE_PATH, 'wb') as f:
    pickle.dump(dataset, f)

print("\n" + "=" * 70)
print(f"✓ Dataset collection complete!")
print(f"  Successful demos: {successful_demos}/{NUM_EPISODES}")
print(f"  Total transitions: {len(dataset['states'])}")
print(f"  Saved to: {SAVE_PATH}")
print("=" * 70)

simulation_app.close()
