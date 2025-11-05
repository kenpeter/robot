#!/usr/bin/env python3
"""
Collect expert dataset using test_grasp_official.py controller.
Self-contained - full copy of test_grasp_official.py with data collection added.

Collects randomized pick-place demonstrations and saves to expert_dataset.pkl
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # Show GUI for visualization

import numpy as np
import os
import pickle
from typing import Optional

# Isaac Sim imports
import isaacsim.core.api.tasks as tasks
import isaacsim.robot.manipulators.controllers as manipulators_controllers
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf


# ============================================================================
# RMPFlow Controller Class (from test_grasp_official.py)
# ============================================================================
class RMPFlowController(mg.MotionPolicyController):
    """RMPFlow motion controller for UR10e."""

    def __init__(
        self,
        name: str,
        robot_articulation: Articulation,
        physics_dt: float = 1.0 / 60.0,
    ) -> None:
        rmpflow_dir = os.path.join(os.path.dirname(__file__), "rmpflow")

        self.rmpflow = mg.lula.motion_policies.RmpFlow(
            robot_description_path=os.path.join(rmpflow_dir, "robot_descriptor.yaml"),
            rmpflow_config_path=os.path.join(rmpflow_dir, "ur10e_rmpflow_common.yaml"),
            urdf_path=os.path.join(rmpflow_dir, "ur10e.urdf"),
            end_effector_frame_name="ee_link_robotiq_arg2f_base_link",
            maximum_substep_size=0.00334,
        )

        self.articulation_rmp = mg.ArticulationMotionPolicy(
            robot_articulation, self.rmpflow, physics_dt
        )

        mg.MotionPolicyController.__init__(
            self, name=name, articulation_motion_policy=self.articulation_rmp
        )
        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )


# ============================================================================
# Pick-Place Controller Class (from test_grasp_official.py)
# ============================================================================
class PickPlaceController(manipulators_controllers.PickPlaceController):
    """Pick and place controller for UR10e - from test_grasp_official.py"""

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        events_dt=None,
    ) -> None:
        if events_dt is None:
            events_dt = [
                0.008,  # Phase 0: Move above
                0.003,  # Phase 1: Lower
                0.15,   # Phase 2: Settle
                0.08,   # Phase 3: Close
                0.002,  # Phase 4: Lift
                0.001,  # Phase 5: Move to place XY
                0.002,  # Phase 6: Move to place height
                0.8,    # Phase 7: Open gripper
                0.008,  # Phase 8: Lift after release
                0.008,  # Phase 9: Return to start
            ]
        manipulators_controllers.PickPlaceController.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation
            ),
            gripper=gripper,
            events_dt=events_dt,
            end_effector_initial_height=0.5,
        )


# ============================================================================
# Red Cube Pick-Place Task Class (from test_grasp_official.py)
# ============================================================================
class RedCubePickPlace(tasks.PickPlace):
    """Simple pick and place task with a RED cube instead of blue."""

    def __init__(
        self,
        name: str = "ur10e_red_cube_pick_place",
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.PickPlace.__init__(
            self,
            name=name,
            cube_initial_position=cube_initial_position,
            cube_initial_orientation=cube_initial_orientation,
            target_position=target_position,
            cube_size=cube_size,
            offset=offset,
        )

    def set_robot(self) -> SingleManipulator:
        """Set up the UR10e robot with gripper."""
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise Exception("Could not find Isaac Sim assets folder")

        asset_path = (
            assets_root_path
            + "/Isaac/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd"
        )
        add_reference_to_stage(usd_path=asset_path, prim_path="/ur")

        gripper = ParallelGripper(
            end_effector_prim_path="/ur/ee_link/robotiq_arg2f_base_link",
            joint_prim_names=["finger_joint"],
            joint_opened_positions=np.array([0]),
            joint_closed_positions=np.array([40]),
            action_deltas=np.array([-40]),
            use_mimic_joints=True,
        )

        manipulator = SingleManipulator(
            prim_path="/ur",
            name="ur10_robot",
            end_effector_prim_path="/ur/ee_link/robotiq_arg2f_base_link",
            gripper=gripper,
        )
        return manipulator

    def set_up_scene(self, scene) -> None:
        """Set up the scene - parent creates blue cube, we change to red."""
        super().set_up_scene(scene)
        from isaacsim.core.api.materials import PreviewSurface

        red_material = PreviewSurface(
            prim_path="/World/Looks/RedMaterial", color=np.array([1.0, 0.0, 0.0])
        )
        self._cube.apply_visual_material(red_material)


# ============================================================================
# DATA COLLECTION MAIN
# ============================================================================
print("\n" + "=" * 70)
print(" EXPERT DATASET COLLECTION - test_grasp_official.py Controller")
print("=" * 70)

# Configuration
NUM_EPISODES = 10
SAVE_PATH = "expert_dataset.pkl"
MAX_STEPS_PER_EPISODE = 2000

# Dataset storage
dataset = {
    'states': [],
    'actions': [],
    'next_states': [],
    'images': [],
    'rewards': [],
}

print(f"\nCollecting {NUM_EPISODES} expert demonstrations...")
print(f"Will save to: {SAVE_PATH}\n")

successful_demos = 0

# Create world ONCE (not per episode)
cube_size = np.array([0.0515, 0.0515, 0.0515])
my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=20 / 200)

# Create initial task
cube_initial_position = np.array([0.5, 0.0, cube_size[2] / 2.0])
target_position = np.array([-0.5, 0.0, cube_size[2] / 2.0])

my_task = RedCubePickPlace(
    name="ur10e_red_cube_pick_place",
    cube_initial_position=cube_initial_position,
    target_position=target_position,
    cube_size=cube_size,
)
my_world.add_task(my_task)
my_world.reset()

# Set up overhead camera ONCE
camera_prim_path = "/World/OverheadCamera"
create_prim(camera_prim_path, "Camera")
eye = Gf.Vec3d(0.1, 0.0, 1.2)
target_vec = Gf.Vec3d(0.1, 0.0, 0.0)
set_camera_view(eye=eye, target=target_vec, camera_prim_path=camera_prim_path)
camera_prim = my_world.stage.GetPrimAtPath(camera_prim_path)
camera_prim.GetAttribute("horizontalAperture").Set(80.0)
camera_prim.GetAttribute("verticalAperture").Set(80.0)
overhead_camera = Camera(prim_path=camera_prim_path, resolution=(84, 84))
overhead_camera.initialize()

# Get task objects ONCE
task_params = my_world.get_task("ur10e_red_cube_pick_place").get_params()
ur10e_name = task_params["robot_name"]["value"]
cube_name = task_params["cube_name"]["value"]
my_ur10e = my_world.scene.get_object(ur10e_name)
my_cube = my_world.scene.get_object(cube_name)

# Create controller ONCE
my_controller = PickPlaceController(
    name="controller", robot_articulation=my_ur10e, gripper=my_ur10e.gripper
)
articulation_controller = my_ur10e.get_articulation_controller()

try:
    for episode in range(NUM_EPISODES):
        print(f"\n--- Episode {episode+1}/{NUM_EPISODES} ---")

        # Randomize cube and target positions
        cube_x = np.random.uniform(0.3, 0.6)
        cube_y = np.random.uniform(-0.3, 0.3)
        cube_initial_position = np.array([cube_x, cube_y, cube_size[2] / 2.0])

        target_x = np.random.uniform(-0.6, -0.3)
        target_y = np.random.uniform(-0.3, 0.3)
        target_position = np.array([target_x, target_y, cube_size[2] / 2.0])

        print(f"  Cube: ({cube_x:.2f}, {cube_y:.2f})")
        print(f"  Target: ({target_x:.2f}, {target_y:.2f})")

        # Reset first to initialize physics
        my_world.reset()

        # AFTER reset, update positions using set_world_pose (recommended by Isaac Sim docs)
        my_cube.set_world_pose(position=cube_initial_position, orientation=np.array([1.0, 0.0, 0.0, 0.0]))

        # Update target position in observations dict (internal task state)
        # The target isn't a physical object, so we manually update the observations
        my_task._target_position = target_position

        # Reset controller with new positions
        my_controller.reset()

        # Episode data storage
        episode_data = []
        task_completed = False
        step_count = 0
        reset_needed = False

        # Run episode (same pattern as test_grasp_official.py)
        while step_count < MAX_STEPS_PER_EPISODE:
            my_world.step(render=True)

            if my_world.is_playing():
                if reset_needed:
                    my_world.reset()
                    reset_needed = False
                    my_controller.reset()

                if my_world.current_time_step_index == 0:
                    my_controller.reset()

                # Get current state
                observations = my_world.get_observations()
                current_joints = observations[ur10e_name]["joint_positions"]
                cube_pos = observations[cube_name]["position"]

                # Capture camera image
                overhead_camera.get_current_frame()
                rgba_data = overhead_camera.get_rgba()
                if rgba_data is None or rgba_data.size == 0:
                    rgb_image = np.zeros((84, 84, 3), dtype=np.uint8)
                else:
                    if len(rgba_data.shape) == 1:
                        rgba_data = rgba_data.reshape(84, 84, 4)
                    rgb_image = rgba_data[:, :, :3].astype(np.uint8)

                # Build state (13D: 12 joints + 1 grasped flag)
                ee_pos, _ = my_ur10e.end_effector.get_world_pose()
                ee_pos = np.array(ee_pos)
                cube_distance = np.linalg.norm(ee_pos - cube_pos)
                gripper_pos = current_joints[6] if len(current_joints) > 6 else 0.0
                grasped = float(cube_distance < 0.15 and gripper_pos > 0.02)
                state = np.concatenate([current_joints, [grasped]])

                # Get expert action from official controller
                actions = my_controller.forward(
                    picking_position=observations[cube_name]["position"],
                    placing_position=observations[cube_name]["target_position"],
                    current_joint_positions=current_joints,
                    end_effector_offset=np.array([0, 0, 0.20]),
                )

                # Apply action
                articulation_controller.apply_action(actions)

                # Step and get next state
                my_world.step(render=False)
                next_observations = my_world.get_observations()
                next_joints = next_observations[ur10e_name]["joint_positions"]
                next_cube_pos = next_observations[cube_name]["position"]
                next_ee_pos, _ = my_ur10e.end_effector.get_world_pose()
                next_ee_pos = np.array(next_ee_pos)
                next_cube_distance = np.linalg.norm(next_ee_pos - next_cube_pos)
                next_gripper_pos = next_joints[6] if len(next_joints) > 6 else 0.0
                next_grasped = float(next_cube_distance < 0.15 and next_gripper_pos > 0.02)
                next_state = np.concatenate([next_joints, [next_grasped]])

                # Compute reward
                reward = max(0, cube_distance - next_cube_distance) * 10.0
                if next_grasped:
                    reward += 5.0

                # Convert action to RL format (dx, dy, dz, gripper)
                # Approximate from joint changes
                rl_action = np.zeros(4, dtype=np.float32)
                if len(actions.joint_positions) >= 6:
                    joint_delta = next_joints[:6] - current_joints[:6]
                    rl_action[:3] = joint_delta[:3] * 0.1
                    rl_action[3] = (next_gripper_pos - gripper_pos) * 10.0
                rl_action = np.clip(rl_action, -1.0, 1.0)

                # Store experience
                episode_data.append({
                    'state': state,
                    'action': rl_action,
                    'next_state': next_state,
                    'image': rgb_image,
                    'reward': reward,
                })

                # Check if done
                if my_controller.is_done():
                    task_completed = True
                    break

                step_count += 1

            if my_world.is_stopped():
                reset_needed = True

        # Add to dataset if successful OR if we have useful data (robot was grasping)
        # Check if robot ever grasped the cube during the episode
        had_grasp = any(exp['state'][12] > 0.5 for exp in episode_data) if episode_data else False

        if (task_completed or had_grasp) and len(episode_data) > 0:
            for exp in episode_data:
                dataset['states'].append(exp['state'])
                dataset['actions'].append(exp['action'])
                dataset['next_states'].append(exp['next_state'])
                dataset['images'].append(exp['image'])
                dataset['rewards'].append(exp['reward'])
            successful_demos += 1
            status = "Complete" if task_completed else "Partial (grasped)"
            print(f"  ✓ {status} ({len(episode_data)} steps)")
        else:
            print(f"  ✗ Failed (steps: {step_count}, experiences: {len(episode_data)})")

except KeyboardInterrupt:
    print("\n\n⚠ Collection interrupted by user")
    print(f"Partial dataset: {len(dataset['states'])} experiences from {successful_demos} demos")

# Convert to numpy arrays (save whatever we have)
dataset['states'] = np.array(dataset['states'], dtype=np.float32)
dataset['actions'] = np.array(dataset['actions'], dtype=np.float32)
dataset['next_states'] = np.array(dataset['next_states'], dtype=np.float32)
dataset['images'] = np.array(dataset['images'], dtype=np.uint8)
dataset['rewards'] = np.array(dataset['rewards'], dtype=np.float32)

# Save dataset BEFORE closing simulation to avoid loss on crash
with open(SAVE_PATH, 'wb') as f:
    pickle.dump(dataset, f)

print("\n" + "=" * 70)
print(f"✓ Dataset collection complete!")
print(f"  Successful demos: {successful_demos}/{NUM_EPISODES}")
print(f"  Total experiences: {len(dataset['states'])}")
print(f"  Saved to: {SAVE_PATH}")
print(f"  Dataset size: {os.path.getsize(SAVE_PATH) / 1024 / 1024:.1f} MB")
print("=" * 70)

# Close simulation (may crash on exit, but dataset is already saved)
try:
    simulation_app.close()
except:
    pass  # Ignore shutdown crashes
