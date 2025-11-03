"""Test script to verify overhead camera can see the red ball"""

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 1280,
        "height": 720,
    }
)

import numpy as np
import cv2
from isaacsim.core.api import World
from isaacsim.core.utils.prims import create_prim
from isaacsim.sensors.camera import Camera
from pxr import UsdGeom, Gf, UsdLux, UsdPhysics, UsdShade, Sdf

# Create world
my_world = World(stage_units_in_meters=1.0)

# Create ground plane
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
ground_plane_path = "/World/GroundPlane"
add_reference_to_stage(usd_path=f"{get_assets_root_path()}/Isaac/Environments/Grid/default_environment.usd", prim_path="/World")

# Reset world to initialize
my_world.reset()

stage = my_world.stage

# === CREATE OVERHEAD CAMERA ===
from isaacsim.core.utils.viewports import set_camera_view

camera_prim_path = "/World/OverheadCamera"
create_prim(camera_prim_path, "Camera")

# Use set_camera_view to properly position and orient camera
# Lower camera height to make objects larger in view (closer = bigger objects)
eye = Gf.Vec3d(0.1, 0.0, 1.2)  # 1.2m high - closer to objects for larger view
target = Gf.Vec3d(0.1, 0.0, 0.0)  # Look at ground at workspace center
set_camera_view(eye=eye, target=target, camera_prim_path=camera_prim_path)

# Adjust horizontal aperture for much wider field of view
camera_prim = stage.GetPrimAtPath(camera_prim_path)
camera_prim.GetAttribute("horizontalAperture").Set(80.0)  # Much wider FOV for entire workspace
camera_prim.GetAttribute("verticalAperture").Set(80.0)  # Match vertical for square aspect ratio

overhead_camera = Camera(
    prim_path=camera_prim_path,
    resolution=(84, 84),
)
overhead_camera.initialize()

# === CREATE RED BALL ===
sphere_path = "/World/Target"
sphere = UsdGeom.Sphere.Define(stage, sphere_path)
sphere.GetRadiusAttr().Set(0.05)
sphere_translate = sphere.AddTranslateOp()
sphere_translate.Set(Gf.Vec3d(0.3, 0.3, 0.05))  # Ball spawn location

# Add RED material
material_path = "/World/Looks/RedMaterial"
material = UsdShade.Material.Define(stage, material_path)
shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

sphere_prim = stage.GetPrimAtPath(sphere_path)
binding_api = UsdShade.MaterialBindingAPI.Apply(sphere_prim)
binding_api.Bind(material)

# === CREATE GOAL MARKER (GREEN) ===
goal_path = "/World/Goal"
goal_sphere = UsdGeom.Sphere.Define(stage, goal_path)
goal_sphere.GetRadiusAttr().Set(0.03)
goal_translate = goal_sphere.AddTranslateOp()
goal_translate.Set(Gf.Vec3d(-0.3, 0.3, 0.05))  # Goal location

goal_material_path = "/World/Looks/GreenMaterial"
goal_material = UsdShade.Material.Define(stage, goal_material_path)
goal_shader = UsdShade.Shader.Define(stage, goal_material_path + "/Shader")
goal_shader.CreateIdAttr("UsdPreviewSurface")
goal_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 1.0, 0.0))
goal_material.CreateSurfaceOutput().ConnectToSource(goal_shader.ConnectableAPI(), "surface")

goal_prim = stage.GetPrimAtPath(goal_path)
goal_binding = UsdShade.MaterialBindingAPI.Apply(goal_prim)
goal_binding.Bind(goal_material)

# === ADD LIGHTING ===
dome_light_path = "/World/DomeLight"
dome_light = UsdLux.DomeLight.Define(stage, dome_light_path)
dome_light.CreateIntensityAttr(1000.0)

distant_light_path = "/World/DistantLight"
distant_light = UsdLux.DistantLight.Define(stage, distant_light_path)
distant_light.CreateIntensityAttr(2000.0)
distant_light_xform = distant_light.AddRotateXYZOp()
distant_light_xform.Set(Gf.Vec3f(-45, 0, 0))

print("=" * 60)
print("OVERHEAD CAMERA TEST")
print("=" * 60)
print(f"Camera position: [0.1, 0.0, 1.5]")
print(f"Ball position: [0.3, 0.3, 0.05] (RED)")
print(f"Goal position: [-0.3, 0.3, 0.05] (GREEN)")
print(f"Workspace: X[-0.6, 0.8], Y[-0.6, 0.6]")
print("=" * 60)

# Run simulation steps
for i in range(10):
    my_world.step(render=True)

# Capture camera image
overhead_camera.get_current_frame()
rgba_data = overhead_camera.get_rgba()

if rgba_data is not None and rgba_data.size > 0:
    # Convert to RGB
    rgb_image = (rgba_data[:, :, :3] * 255).astype(np.uint8)

    # Detect red pixels
    red_mask = (
        (rgb_image[:, :, 0] > 200)  # High red
        & (rgb_image[:, :, 1] < 100)  # Low green
        & (rgb_image[:, :, 2] < 100)  # Low blue
    )
    red_pixel_count = np.sum(red_mask)

    # Detect green pixels
    green_mask = (
        (rgb_image[:, :, 0] < 100)  # Low red
        & (rgb_image[:, :, 1] > 200)  # High green
        & (rgb_image[:, :, 2] < 100)  # Low blue
    )
    green_pixel_count = np.sum(green_mask)

    print(f"\n✓ Camera capture successful!")
    print(f"  Image shape: {rgb_image.shape}")
    print(f"  Red pixels detected: {red_pixel_count}")
    print(f"  Green pixels detected: {green_pixel_count}")

    # Check if ball is visible
    ball_visible = (red_pixel_count > 100) and (red_pixel_count < 800)
    goal_visible = (green_pixel_count > 50)

    print(f"\n{'✓' if ball_visible else '✗'} Ball visible: {ball_visible}")
    print(f"{'✓' if goal_visible else '✗'} Goal visible: {goal_visible}")

    # Save debug images
    debug_img = rgb_image.copy()
    debug_img[red_mask] = [0, 255, 0]  # Highlight red pixels in green
    debug_img[green_mask] = [255, 255, 0]  # Highlight green pixels in yellow

    cv2.imwrite("test_camera_raw.png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite("test_camera_mask.png", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

    print(f"\n✓ Saved: test_camera_raw.png")
    print(f"✓ Saved: test_camera_mask.png")

    if ball_visible and goal_visible:
        print(f"\n{'='*60}")
        print("✓✓✓ SUCCESS! Camera can see both ball and goal!")
        print(f"{'='*60}")
    elif ball_visible:
        print(f"\n⚠ WARNING: Ball visible but goal not visible")
    elif goal_visible:
        print(f"\n⚠ WARNING: Goal visible but ball not visible")
    else:
        print(f"\n✗ FAILURE: Cannot see ball or goal")
else:
    print("✗ Camera capture failed - no data returned")

# Close after brief delay
import time
time.sleep(2)
simulation_app.close()
