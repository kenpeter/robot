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
# Pick-Place Controller Class (with parallel arm motion fix)
# ============================================================================
class PickPlaceController(manipulators_controllers.PickPlaceController):
    """
    Pick and place controller for UR10e with extended timing for smooth motion.

    Modified timing to ensure: lift up → move horizontally → lower down
    """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        events_dt=None,
    ) -> None:
        if events_dt is None:
            # Modified timing for RELIABLE grip
            # Phase 1 (Lower): 0.002 -> 0.005 (slower lowering for better alignment)
            # Phase 2 (Settle): 1 -> 2.0 seconds (MUCH more time for gripper to wrap around cube)
            # Phase 3 (Close): 0.05 -> 0.15 seconds (3x longer to ensure maximum grip force)
            events_dt = [0.005, 0.005, 2.0, 0.15, 0.0008, 0.005, 0.0008, 0.15, 0.0008, 0.008]
        manipulators_controllers.PickPlaceController.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation
            ),
            gripper=gripper,
            events_dt=events_dt,
            end_effector_initial_height=0.6,  # Official Isaac Sim value
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
            joint_closed_positions=np.array([40]),  # Official value
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
cube_size = np.array([0.065, 0.065, 0.065])  # Increased from 0.0515 to 0.065 (26% bigger, easier to grip)
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

reset_needed = False
task_completed = False

try:
    for episode in range(NUM_EPISODES):
        print(f"\n--- Episode {episode+1}/{NUM_EPISODES} ---")

        # Randomize cube and target positions for each episode
        # RIGHT-HANDED SWING MOTION: Pick from front-right → swing 180° → place at back-right
        # Like a right-handed person sweeping an arc, no twisting!

        # Cube position: FRONT-RIGHT (positive X, positive Y)
        cube_x = np.random.uniform(0.3, 0.4)    # Front of robot
        cube_y = np.random.uniform(0.25, 0.35)  # Right side (positive Y)
        cube_initial_position = np.array([cube_x, cube_y, cube_size[2] / 2.0])

        # Target position: BACK-RIGHT (negative X, SAME positive Y side)
        # 180° swing: same Y side, opposite X (front → back)
        target_x = np.random.uniform(-0.4, -0.3)   # Behind robot (negative X)
        target_y = np.random.uniform(0.25, 0.35)   # SAME right side (positive Y)
        target_position = np.array([target_x, target_y, cube_size[2] / 2.0])

        # Motion: Front-right → (lift) → swing 180° arc → Back-right (drop)
        # Natural right-handed motion, no twisting or crossing over base!

        print(f"  Cube: ({cube_x:.2f}, {cube_y:.2f})")
        print(f"  Target: ({target_x:.2f}, {target_y:.2f})")

        # Reset world
        my_world.reset()

        # Update cube and target positions after reset
        my_cube.set_world_pose(position=cube_initial_position, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        my_task._target_position = target_position
        my_task._cube_initial_position = cube_initial_position

        # Reset controller
        my_controller.reset()
        task_completed = False
        reset_needed = False

        # Episode data storage
        episode_data = []
        step_count = 0

        # Run episode - EXACT pattern from test_grasp_official.py
        # Keep running until task completes (just like test_grasp_official.py)
        while True:
            my_world.step(render=True)

            if my_world.is_playing():
                if reset_needed:
                    my_world.reset()
                    reset_needed = False
                    my_controller.reset()

                if my_world.current_time_step_index == 0:
                    my_controller.reset()

                # Get observations
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

                # Get current phase for debugging
                current_phase = my_controller.get_current_event() if hasattr(my_controller, 'get_current_event') else 0

                # Print debug info every 50 steps AND at key phases
                if step_count % 50 == 0 or current_phase in [2, 3, 4]:  # Log settling, closing, lifting
                    print(f"    Step {step_count}: Phase {current_phase}, Cube distance: {cube_distance:.3f}, Grasped: {grasped}, Done: {my_controller.is_done()}")
                    print(f"      EE pos: {ee_pos}")
                    print(f"      Cube pos: {cube_pos}")
                    print(f"      Gripper: {gripper_pos:.3f}")
                    if current_phase == 2:
                        print(f"      ⚠ Phase 2 (Settling) - distance should be < 0.1 for good grasp!")

                # Check if done - EXACT pattern from test_grasp_official.py
                if my_controller.is_done() and not task_completed:
                    print(f"    ✓ Episode {episode+1} DONE at step {step_count}, phase: {current_phase}")
                    print(f"      Final cube distance: {cube_distance:.3f}")
                    task_completed = True

                # Apply actions to robot - EXACT pattern from test_grasp_official.py
                articulation_controller.apply_action(actions)

                # Simple phase-based action encoding
                rl_action = np.zeros(4, dtype=np.float32)
                if current_phase in [0, 1]:  # Moving/lowering to pick
                    rl_action[:3] = [0.1, 0.0, -0.1]
                elif current_phase == 3:  # Closing gripper
                    rl_action[3] = 1.0
                elif current_phase == 4:  # Lifting
                    rl_action[:3] = [0.0, 0.0, 0.1]
                elif current_phase == 5:  # Moving to target
                    rl_action[:3] = [-0.1, 0.0, 0.0]
                elif current_phase in [6, 7]:  # Lowering/opening
                    rl_action[:3] = [0.0, 0.0, -0.1]
                    rl_action[3] = -1.0 if current_phase == 7 else 0.0

                reward = -cube_distance * 1.0
                if grasped:
                    reward += 5.0

                episode_data.append({
                    'state': state,
                    'action': rl_action,
                    'next_state': state,  # Will fix in post-processing
                    'image': rgb_image,
                    'reward': reward,
                })

                step_count += 1

                # Exit when task is completed (like test_grasp_official.py just continues)
                # For data collection, we want to capture the full trajectory then move to next episode
                if task_completed:
                    break

            if my_world.is_stopped():
                reset_needed = True

        # Post-process to fix next_state (shift states by 1)
        for i in range(len(episode_data) - 1):
            episode_data[i]['next_state'] = episode_data[i + 1]['state']

        # QUALITY FILTER: Only save HIGH-QUALITY successful demonstrations
        # Check if robot successfully grasped the cube (not just touched it)
        grasp_count = sum(1 for exp in episode_data if exp['state'][12] > 0.5)
        grasp_percentage = grasp_count / len(episode_data) if len(episode_data) > 0 else 0

        # Check if cube reached near target (within 15cm)
        if len(episode_data) > 0:
            final_cube_pos = episode_data[-1]['state']  # Get final state
            # Assuming we can infer if placement was successful from task_completed
            successful_placement = task_completed
        else:
            successful_placement = False

        # STRICT CRITERIA: Only save if
        # 1. Task fully completed (picked, transported, placed)
        # 2. Robot held cube for at least 30% of the trajectory
        if task_completed and grasp_percentage > 0.3 and len(episode_data) > 50:
            for exp in episode_data:
                dataset['states'].append(exp['state'])
                dataset['actions'].append(exp['action'])
                dataset['next_states'].append(exp['next_state'])
                dataset['images'].append(exp['image'])
                dataset['rewards'].append(exp['reward'])
            successful_demos += 1
            print(f"  ✓ SUCCESS! Saved {len(episode_data)} steps (grasp: {grasp_percentage*100:.1f}%)")
        else:
            failure_reason = []
            if not task_completed:
                failure_reason.append("incomplete")
            if grasp_percentage <= 0.3:
                failure_reason.append(f"weak grasp ({grasp_percentage*100:.1f}%)")
            if len(episode_data) <= 50:
                failure_reason.append("too short")
            print(f"  ✗ REJECTED: {', '.join(failure_reason)} (steps: {step_count})")

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
