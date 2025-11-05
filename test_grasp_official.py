#!/usr/bin/env python3
"""
UR10e RED CUBE pick-place using OFFICIAL pattern
Based on Isaac Sim standalone_examples/api/isaacsim.robot.manipulators/ur10e/pick_up_example.py

All-in-one file - simplified version using red cube.
"""

# main app
from isaacsim import SimulationApp

# main app
simulation_app = SimulationApp({"headless": False})

import numpy as np
import os
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


# ============================================================================
# RMPFlow Controller Class
# ============================================================================
class RMPFlowController(mg.MotionPolicyController):
    """RMPFlow motion controller for UR10e."""

    def __init__(
        self,
        name: str,
        robot_articulation: Articulation,
        physics_dt: float = 1.0 / 60.0,
    ) -> None:
        # Get the directory where rmpflow config files are located
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
# Pick-Place Controller Class
# ============================================================================
class PickPlaceController(manipulators_controllers.PickPlaceController):
    """
    Pick and place controller for UR10e to grasp a ball.
    Uses the official PickPlaceController state machine with custom timing.
    """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        events_dt=None,
    ) -> None:
        # Official Isaac Sim UR10e timing
        if events_dt is None:
            events_dt = [0.005, 0.002, 1, 0.05, 0.0008, 0.005, 0.0008, 0.1, 0.0008, 0.008]
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
# Red Cube Pick-Place Task Class
# ============================================================================
class RedCubePickPlace(tasks.PickPlace):
    """
    Simple pick and place task with a RED cube instead of blue.
    """

    def __init__(
        self,
        name: str = "ur10e_red_cube_pick_place",
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        # Initialize parent class normally
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

        # Define the gripper
        gripper = ParallelGripper(
            end_effector_prim_path="/ur/ee_link/robotiq_arg2f_base_link",
            joint_prim_names=["finger_joint"],
            joint_opened_positions=np.array([0]),
            joint_closed_positions=np.array([40]),
            action_deltas=np.array([-40]),
            use_mimic_joints=True,
        )

        # Define the manipulator
        manipulator = SingleManipulator(
            prim_path="/ur",
            name="ur10_robot",
            end_effector_prim_path="/ur/ee_link/robotiq_arg2f_base_link",
            gripper=gripper,
        )
        return manipulator

    def set_up_scene(self, scene) -> None:
        """Set up the scene - parent creates blue cube, we just change it to red."""
        # Let parent create everything (robot, cube, ground plane)
        super().set_up_scene(scene)

        # Simply change the existing cube's color to red
        # The cube already exists from parent's set_up_scene
        from isaacsim.core.api.materials import PreviewSurface

        # Create a red material
        red_material = PreviewSurface(
            prim_path="/World/Looks/RedMaterial", color=np.array([1.0, 0.0, 0.0])
        )

        # Apply it to the cube
        self._cube.apply_visual_material(red_material)


# ============================================================================
# MAIN SCRIPT
# ============================================================================
print("\n" + "=" * 70)
print(" UR10e RED CUBE PICK & PLACE - OFFICIAL PATTERN")
print("=" * 70)

# Create world with same parameters as official example
my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=20 / 200)

# Set up RED cube pick and place task
# Use official example structure - let task set default cube position
cube_size = np.array([0.0515, 0.0515, 0.0515])
target_position = np.array([-0.3, 0.3, 0])
target_position[2] = cube_size[2] / 2.0

print(f"\nTask setup:")
print(f"  Cube size: {cube_size[0]*100:.1f}cm")
print(f"  Target position: {target_position}")

# Create the RED cube pick-place task
my_task = RedCubePickPlace(
    name="ur10e_red_cube_pick_place",
    target_position=target_position,
    cube_size=cube_size,
)
my_world.add_task(my_task)
my_world.reset()

# Get task parameters
task_params = my_world.get_task("ur10e_red_cube_pick_place").get_params()
ur10e_name = task_params["robot_name"]["value"]
cube_name = task_params["cube_name"]["value"]

print(f"\nRobot name: {ur10e_name}")
print(f"Cube name: {cube_name}")

# Get robot from scene
my_ur10e = my_world.scene.get_object(ur10e_name)

# Initialize the pick-place controller
my_controller = PickPlaceController(
    name="controller", robot_articulation=my_ur10e, gripper=my_ur10e.gripper
)

articulation_controller = my_ur10e.get_articulation_controller()

reset_needed = False
task_completed = False

print("\nStarting simulation...")
print("The robot will:")
print("  1. Move above the cube")
print("  2. Lower down to grasp position")
print("  3. Close gripper around cube")
print("  4. Lift the cube")
print("  5. Move to target position")
print("  6. Place the cube")
print("  7. Return to start\n")

step_count = 0
last_event = -1

# Main simulation loop - follows official pattern exactly
while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False
            my_controller.reset()
            task_completed = False

        if my_world.current_time_step_index == 0:
            my_controller.reset()

        # Get observations
        observations = my_world.get_observations()

        # Forward the observation values to the controller
        actions = my_controller.forward(
            picking_position=observations[cube_name]["position"],
            placing_position=observations[cube_name]["target_position"],
            current_joint_positions=observations[ur10e_name]["joint_positions"],
            # Offset needs tuning as well
            end_effector_offset=np.array([0, 0, 0.20]),
        )

        # Check for event changes to print progress
        current_event = my_controller.get_current_event()
        if current_event != last_event:
            event_names = [
                "Moving above cube",
                "Lowering to cube",
                "Settling",
                "Closing gripper",
                "Lifting cube",
                "Moving to target XY",
                "Lowering to target",
                "Opening gripper",
                "Lifting after release",
                "Returning to start",
            ]
            if current_event < len(event_names):
                print(f"Phase {current_event}: {event_names[current_event]}")
            last_event = current_event

        # Check if done
        if my_controller.is_done() and not task_completed:
            print("\n" + "=" * 70)
            print("DONE! Cube pick-place sequence completed")
            print("=" * 70)
            task_completed = True

        # Apply actions to robot
        articulation_controller.apply_action(actions)

        step_count += 1

    if my_world.is_stopped():
        reset_needed = True

simulation_app.close()
