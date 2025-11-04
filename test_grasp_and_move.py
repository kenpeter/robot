#!/usr/bin/env python3
"""
Test complete grasp-and-move workflow with RMPflow.
Tests: reach ball → grasp → lift → move to goal
This script uses a robust kinematic-to-dynamic switching method for the target object.
"""

import sys
import numpy as np
from isaacsim import SimulationApp

# It's recommended to test with headless=False first. Once confirmed, use headless=True for speed.
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper

# <<< FIX: Correct import path for ArticulationAction
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import (
    ArticulationMotionPolicy,
    interface_config_loader,
    RmpFlow,
)
from pxr import UsdGeom, Gf, UsdPhysics, UsdShade, Sdf

print("\n" + "=" * 70, flush=True)
print(" FULL GRASP-AND-MOVE TEST (WITH KINEMATIC-TO-DYNAMIC SWITCH)")
print("=" * 70, flush=True)

# Create world
my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# Add Franka robot with gripper
assets_root_path = get_assets_root_path()
asset_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka")

# Debug: Print the structure of the loaded robot
stage = my_world.stage
robot_prim = stage.GetPrimAtPath("/World/Franka")
if robot_prim:
    print("\nAvailable robot prims:")
    for child in robot_prim.GetChildren():
        print(f"- {child.GetPath()}")
        for grandchild in child.GetChildren():
            print(f"  - {grandchild.GetPath()}")

# Configure gripper
gripper = ParallelGripper(
    end_effector_prim_path="/World/Franka/panda_rightfinger",  # Use the right finger as end effector
    joint_prim_names=[
        "panda_finger_joint1",
        "panda_finger_joint2",
    ],  # Joint names only (not full paths)
    joint_opened_positions=np.array([0.04, 0.04]),
    joint_closed_positions=np.array([0.0, 0.0]),
    action_deltas=np.array([-0.1, -0.1]),
)

robot = my_world.scene.add(
    SingleManipulator(
        prim_path="/World/Franka",
        name="franka_arm",
        end_effector_prim_path="/World/Franka/panda_rightfinger",
        gripper=gripper,
    )
)

# Add ball (0.04m radius = 0.08m diameter - max grippable size)
stage = my_world.stage
ball_path = "/World/Ball"
ball_sphere = UsdGeom.Sphere.Define(stage, ball_path)
ball_sphere.GetRadiusAttr().Set(0.04)
ball_translate = ball_sphere.AddTranslateOp()
ball_position = Gf.Vec3d(0.3, 0.0, 0.04)
ball_translate.Set(ball_position)

# Add red material
material_path = "/World/Looks/RedMaterial"
material = UsdShade.Material.Define(stage, material_path)
shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
    Gf.Vec3f(1.0, 0.0, 0.0)
)
material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
ball_prim = stage.GetPrimAtPath(ball_path)
UsdShade.MaterialBindingAPI.Apply(ball_prim).Bind(material)

# Add physics to ball
rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(ball_prim)
# Start the ball as KINEMATIC so it won't move during the robot's approach
rigid_body_api.CreateKinematicEnabledAttr(True)
UsdPhysics.CollisionAPI.Apply(ball_prim)
UsdPhysics.MassAPI.Apply(ball_prim).CreateMassAttr(0.05)  # 50g ball

ball = RigidPrim(ball_path, name="ball")
my_world.scene.add(ball)

# Reset world (this initializes the robot internally)
my_world.reset()

gripper_dof_indices = robot.gripper.joint_dof_indicies.astype(np.int32)
arm_num_dof = robot.num_dof - len(gripper_dof_indices)

# Set gripper joint properties directly using USD APIs
for joint_name in gripper.joint_prim_names:
    joint_path = robot.prim_path + "/panda_hand/" + joint_name
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim:
        raise RuntimeError(f"Joint prim not found at path: {joint_path}")

    # Try to get the drive API - could be 'linear' for prismatic or 'angular' for revolute
    drive = UsdPhysics.DriveAPI.Get(joint_prim, "linear")
    if not drive:
        drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
    if not drive:
        # If no named drive exists, try to apply it
        joint_type = joint_prim.GetTypeName()
        print(f"Joint {joint_path} type: {joint_type}, attempting to apply DriveAPI")
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")

    if drive:
        drive.GetStiffnessAttr().Set(0)
        drive.GetDampingAttr().Set(1e6)
        joint_prim.GetAttribute("physxJoint:maxJointVelocity").Set(1.0)

robot.gripper.open()
# Set a good starting pose for the arm
robot.set_joint_positions(
    np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.2, 0.0]), joint_indices=range(arm_num_dof)
)
for _ in range(10):
    my_world.step(render=True)

print("\n✓ World, robot, and ball initialized", flush=True)

# Initialize RMPflow
print("\nInitializing RMPflow...", flush=True)
rmp_flow_config = interface_config_loader.load_supported_motion_policy_config(
    "Franka", "RMPflow"
)
rmp_flow = RmpFlow(**rmp_flow_config)
physics_dt = 1.0 / 60.0
motion_policy = ArticulationMotionPolicy(robot, rmp_flow, physics_dt)
print("✓ RMPflow initialized", flush=True)

print("\n" + "=" * 70, flush=True)
print("TESTING GRASP-AND-MOVE WORKFLOW")
print("=" * 70, flush=True)

# Get initial ball position
ball_pos, _ = ball.get_world_poses()
ball_pos = np.array(ball_pos, dtype=np.float32).flatten()
print(f"\n1. Ball position: {ball_pos} (kinematic)", flush=True)

# PHASE 1: Reach above ball
print("\n2. PHASE 1: Moving above ball...", flush=True)
above_ball = ball_pos.copy()
above_ball[2] += 0.15  # 15cm above ball

for step in range(100):
    rmp_flow.set_end_effector_target(
        target_position=above_ball, target_orientation=None
    )
    # Get action for the arm only
    arm_action = motion_policy.get_next_articulation_action(physics_dt)

    # Create a joint velocity command for the full robot
    full_dof_velocities = np.zeros(robot.num_dof)
    full_dof_velocities[:arm_num_dof] = arm_action.joint_velocities
    # Gripper velocity is 0 (keep it open)
    full_dof_velocities[gripper_dof_indices] = 0.0

    # Create a complete action for the full robot
    robot.apply_action(ArticulationAction(joint_velocities=full_dof_velocities))
    my_world.step(render=True)
    if step % 25 == 0:
        ee_pos, _ = robot.end_effector.get_world_pose()
        distance = np.linalg.norm(np.array(ee_pos) - above_ball)
        print(f"   Step {step}: distance to above-ball = {distance:.4f}m", flush=True)

ee_pos, _ = robot.end_effector.get_world_pose()
distance = np.linalg.norm(np.array(ee_pos) - above_ball)
print(f"✓ Reached above ball (distance: {distance:.4f}m)", flush=True)

# PHASE 2: Lower to ball
print("\n3. PHASE 2: Lowering to ball...", flush=True)
grasp_target = ball_pos.copy()
grasp_target[2] += 0.005  # Target slightly above center for better contact

for step in range(100):
    rmp_flow.set_end_effector_target(
        target_position=grasp_target, target_orientation=None
    )
    arm_action = motion_policy.get_next_articulation_action(physics_dt)

    full_dof_velocities = np.zeros(robot.num_dof)
    full_dof_velocities[:arm_num_dof] = arm_action.joint_velocities
    full_dof_velocities[gripper_dof_indices] = 0.0  # Keep gripper open

    robot.apply_action(ArticulationAction(joint_velocities=full_dof_velocities))
    my_world.step(render=True)
    if step % 30 == 0:
        ee_pos_debug, _ = robot.end_effector.get_world_pose()
        distance_debug = np.linalg.norm(np.array(ee_pos_debug) - grasp_target)
        print(f"   Step {step}: distance to grasp = {distance_debug:.4f}m", flush=True)

ee_pos, _ = robot.end_effector.get_world_pose()
ee_to_ball = np.linalg.norm(np.array(ee_pos) - np.array(ball_pos))
print(f"✓ Lowered to ball (EE-to-ball distance: {ee_to_ball:.4f}m)", flush=True)

# CRITICAL STEP: Switch ball from kinematic to dynamic right before grasping
print("\n4. ACTIVATING BALL PHYSICS FOR GRASP...", flush=True)
ball_prim = my_world.stage.GetPrimAtPath(ball_path)
rigid_body_api = UsdPhysics.RigidBodyAPI(ball_prim)
rigid_body_api.GetKinematicEnabledAttr().Set(False)
# Give the physics engine a moment to process the change
for _ in range(5):
    my_world.step(render=True)
print("✓ Ball is now a dynamic object.", flush=True)

# PHASE 3: Close gripper
print("\n5. PHASE 3: Closing gripper...", flush=True)
for step in range(40):
    ee_pos, _ = robot.end_effector.get_world_pose()
    rmp_flow.set_end_effector_target(target_position=ee_pos, target_orientation=None)
    arm_action = motion_policy.get_next_articulation_action(physics_dt)

    full_dof_velocities = np.zeros(robot.num_dof)
    full_dof_velocities[:arm_num_dof] = arm_action.joint_velocities
    full_dof_velocities[gripper_dof_indices] = -0.1  # Close gripper

    robot.apply_action(ArticulationAction(joint_velocities=full_dof_velocities))
    my_world.step(render=True)

joint_pos = robot.get_joint_positions()
gripper_width = joint_pos[gripper_dof_indices].sum()
print(f"✓ Gripper closed (width: {gripper_width:.4f}m)", flush=True)

# Check if ball is grasped
ee_pos, _ = robot.end_effector.get_world_pose()
ball_pos, _ = ball.get_world_poses()
ball_pos = np.array(ball_pos).flatten()
ee_to_ball = np.linalg.norm(np.array(ee_pos) - ball_pos)
ball_grasped = (
    ee_to_ball < 0.08 and gripper_width < 0.08
)  # Ball radius is 0.04 -> diameter is 0.08
if ball_grasped:
    print(
        f"✓✓✓ BALL GRASPED! (distance: {ee_to_ball:.4f}m, gripper: {gripper_width:.4f}m)",
        flush=True,
    )
else:
    print(
        f"⚠ Grasp uncertain (distance: {ee_to_ball:.4f}m, gripper: {gripper_width:.4f}m)",
        flush=True,
    )

# PHASE 4: Lift ball
print("\n6. PHASE 4: Lifting ball...", flush=True)
ball_pos_before_lift, _ = ball.get_world_poses()
ball_pos_before_lift = np.array(ball_pos_before_lift).flatten()
lift_target = ball_pos_before_lift.copy()
lift_target[2] += 0.20  # Lift 20cm

for step in range(80):
    rmp_flow.set_end_effector_target(
        target_position=lift_target, target_orientation=None
    )
    arm_action = motion_policy.get_next_articulation_action(physics_dt)

    full_dof_velocities = np.zeros(robot.num_dof)
    full_dof_velocities[:arm_num_dof] = arm_action.joint_velocities
    full_dof_velocities[gripper_dof_indices] = -0.1  # Keep closed

    robot.apply_action(ArticulationAction(joint_velocities=full_dof_velocities))
    my_world.step(render=True)

ball_pos_after_lift, _ = ball.get_world_poses()
ball_pos_after_lift = np.array(ball_pos_after_lift).flatten()
lift_height = ball_pos_after_lift[2] - ball_pos_before_lift[2]
print(f"✓ Lifted (ball moved up {lift_height:.4f}m)", flush=True)
lift_success = lift_height > 0.15  # A little tolerance for physics settling
if lift_success:
    print(f"✓✓✓ LIFT SUCCESSFUL! Ball moved {lift_height:.4f}m up", flush=True)
else:
    print(f"⚠ Lift failed or ball slipped (only moved {lift_height:.4f}m)", flush=True)

# PHASE 5: Move to goal
print("\n7. PHASE 5: Moving ball to goal...", flush=True)
goal_position = np.array([-0.3, 0.3, 0.20])

for step in range(120):
    rmp_flow.set_end_effector_target(
        target_position=goal_position, target_orientation=None
    )
    arm_action = motion_policy.get_next_articulation_action(physics_dt)

    full_dof_velocities = np.zeros(robot.num_dof)
    full_dof_velocities[:arm_num_dof] = arm_action.joint_velocities
    full_dof_velocities[gripper_dof_indices] = -0.1  # Keep closed

    robot.apply_action(ArticulationAction(joint_velocities=full_dof_velocities))
    my_world.step(render=True)
    if step % 30 == 0:
        ball_pos, _ = ball.get_world_poses()
        ball_pos = np.array(ball_pos).flatten()
        distance_to_goal = np.linalg.norm(ball_pos - goal_position)
        print(
            f"   Step {step}: ball distance to goal = {distance_to_goal:.4f}m",
            flush=True,
        )

ball_pos_final, _ = ball.get_world_poses()
ball_pos_final = np.array(ball_pos_final).flatten()
distance_to_goal = np.linalg.norm(ball_pos_final - goal_position)
print(f"✓ Moved ball (final distance to goal: {distance_to_goal:.4f}m)", flush=True)
delivery_success = distance_to_goal < 0.1
if delivery_success:
    print(f"✓✓✓ DELIVERY SUCCESSFUL! Ball at goal", flush=True)
else:
    print(f"⚠ Delivery incomplete (ball {distance_to_goal:.4f}m from goal)", flush=True)

# PHASE 6: Release ball
print("\n8. PHASE 6: Opening gripper to release...", flush=True)
for step in range(30):
    ee_pos, _ = robot.end_effector.get_world_pose()
    rmp_flow.set_end_effector_target(target_position=ee_pos, target_orientation=None)
    arm_action = motion_policy.get_next_articulation_action(physics_dt)

    full_dof_velocities = np.zeros(robot.num_dof)
    full_dof_velocities[:arm_num_dof] = arm_action.joint_velocities
    full_dof_velocities[gripper_dof_indices] = 0.1  # Open gripper

    robot.apply_action(ArticulationAction(joint_velocities=full_dof_velocities))
    my_world.step(render=True)
print("✓ Gripper opened", flush=True)

# Let ball fall and settle
for _ in range(60):
    my_world.step(render=True)
ball_pos_released, _ = ball.get_world_poses()
ball_pos_released = np.array(ball_pos_released).flatten()
print(f"✓ Ball released at: {ball_pos_released}", flush=True)

# FINAL VERDICT
print("\n" + "=" * 70, flush=True)
print("FINAL VERDICT")
print("=" * 70, flush=True)
if ball_grasped and lift_success and delivery_success:
    print("✓✓✓ COMPLETE SUCCESS! Robot can:", flush=True)
    print("  1. Reach and grasp the ball", flush=True)
    print("  2. Lift the ball off the ground", flush=True)
    print("  3. Move the ball to the goal location", flush=True)
    print("\n✓✓✓ Training script WILL WORK for full pick-and-place task!", flush=True)
elif ball_grasped and lift_success:
    print(
        "✓✓ PARTIAL SUCCESS - Can grasp and lift, but delivery needs tuning", flush=True
    )
    print("Training script should work but may need reward adjustments", flush=True)
elif ball_grasped:
    print("✓ PARTIAL SUCCESS - Can grasp but cannot lift", flush=True)
    print("Check:", flush=True)
    print("  - Ball mass (too heavy?)", flush=True)
    print("  - Gripper force/drive settings (too weak?)", flush=True)
    print("  - Friction settings on gripper/ball", flush=True)
else:
    print("✗ FAILED - Cannot reliably grasp the ball", flush=True)
    print("Check:", flush=True)
    print("  - Ball size vs gripper opening", flush=True)
    print("  - Final EE-to-ball distance before grasp", flush=True)
    print("  - RMPflow accuracy or drive settings", flush=True)
print("=" * 70, flush=True)

simulation_app.close()
