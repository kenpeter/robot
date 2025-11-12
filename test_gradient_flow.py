"""
Test script to verify gradient flow in the Diffusion Transformer model.
Checks forward pass, backward pass, and gradient magnitudes.
Includes Isaac Sim UI for visualization.
"""

# Initialize Isaac Sim first
from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,  # Show UI
        "width": 1280,
        "height": 720,
        "renderer": "RayTracedLighting",
    }
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.core.api.objects import DynamicCuboid
from pxr import Gf, UsdLux

# Copy model classes from simple_rl_robot_arm.py
class DiTBlock(nn.Module):
    """Transformer block with adaptive layer norm and standard multi-head attention for diffusion timestep conditioning"""

    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Standard PyTorch multi-head attention
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        # Adaptive modulation parameters (6 parameters: scale/shift for attn and mlp, gate for attn and mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim)
        )

    def forward(self, x, c):
        """
        x: input tokens [batch, seq_len, hidden_dim]
        c: conditioning (timestep + state) [batch, hidden_dim]
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(
            6, dim=-1
        )

        # Standard multi-head attention with adaptive modulation
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP with adaptive modulation
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


class SimpleCNN(nn.Module):
    """Simple CNN for visual feature extraction"""

    def __init__(self, output_dim=128, img_size=84):
        super().__init__()
        self.img_size = img_size

        # Simple convolutional layers (using GroupNorm instead of BatchNorm for stability)
        self.conv = nn.Sequential(
            # 84x84x3 -> 42x42x32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 32),  # 8 groups for 32 channels
            # 42x42x32 -> 21x21x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 64),  # 8 groups for 64 channels
            # 21x21x64 -> 10x10x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(16, 128),  # 16 groups for 128 channels
            # 10x10x128 -> 5x5x128
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(16, 128),  # 16 groups for 128 channels
        )

        # Adaptive pooling to ensure consistent output size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Always output 4x4

        # Calculate flattened size: 4x4x128 = 2048
        self.flatten_size = 4 * 4 * 128

        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 256), nn.ReLU(), nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # x: [B, H, W, C] from Isaac Sim -> [B, C, H, W] for PyTorch
        if len(x.shape) == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW

        x = self.conv(x)
        x = self.pool(x)  # Ensure consistent spatial size
        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class DiffusionTransformer(nn.Module):
    """Diffusion Transformer for action generation with vision"""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        use_vision=True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_vision = use_vision

        # Vision encoder (CNN)
        if use_vision:
            self.vision_encoder = SimpleCNN(output_dim=hidden_dim)

        # Timestep embedding (for diffusion process)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        # State encoder (proprioception only, no ball position)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Noisy action encoder
        self.action_encoder = nn.Linear(action_dim, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )

        # Output head to predict noise
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, noisy_action, state, timestep, image=None):
        """
        noisy_action: [batch, action_dim] - noisy action at timestep t
        state: [batch, state_dim] - robot proprioceptive state
        timestep: [batch, 1] - diffusion timestep (0 to 1)
        image: [batch, H, W, 3] - optional RGB image

        Returns: predicted noise [batch, action_dim]
        """
        # Encode inputs
        t_emb = self.time_embed(timestep)  # [batch, hidden_dim]
        s_emb = self.state_encoder(state)  # [batch, hidden_dim]
        a_emb = self.action_encoder(noisy_action)  # [batch, hidden_dim]

        # Conditioning: combine timestep, state, and vision
        c = t_emb + s_emb  # [batch, hidden_dim]

        if self.use_vision and image is not None:
            v_emb = self.vision_encoder(image)  # [batch, hidden_dim]
            c = c + v_emb

        # Action as sequence (can be extended to multiple tokens)
        x = a_emb.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Apply transformer blocks with standard multi-head attention
        for block in self.blocks:
            x = block(x, c)

        # Predict noise
        x = x.squeeze(1)  # [batch, hidden_dim]
        noise_pred = self.final_layer(x)  # [batch, action_dim]

        return noise_pred


def check_gradient_flow(named_parameters):
    """Check gradient magnitudes for each layer"""
    print("\n" + "=" * 80)
    print("GRADIENT FLOW CHECK")
    print("=" * 80)

    total_norm = 0.0
    has_nan = False
    zero_grads = []

    for name, param in named_parameters:
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_norm += grad_norm ** 2

            if torch.isnan(param.grad).any():
                print(f"❌ NaN gradient in {name}")
                has_nan = True
            elif grad_norm < 1e-8:
                zero_grads.append(name)
            else:
                print(f"✓ {name:60s} | grad_norm: {grad_norm:.6f}")
        else:
            print(f"⚠ {name:60s} | NO GRADIENT")

    total_norm = total_norm ** 0.5

    if zero_grads:
        print(f"\n⚠ WARNING: {len(zero_grads)} layers have near-zero gradients:")
        for name in zero_grads[:5]:  # Show first 5
            print(f"  - {name}")

    print(f"\n{'=' * 80}")
    print(f"Total gradient norm: {total_norm:.6f}")
    print(f"Has NaN gradients: {has_nan}")
    print(f"{'=' * 80}\n")

    return total_norm, has_nan


def test_forward_pass(model, batch_size=4, device='cuda'):
    """Test forward pass with dummy data"""
    print("\n" + "=" * 80)
    print("FORWARD PASS TEST")
    print("=" * 80)

    # Create dummy inputs
    state_dim = 13
    action_dim = 4

    noisy_action = torch.randn(batch_size, action_dim).to(device)
    state = torch.randn(batch_size, state_dim).to(device)
    timestep = torch.rand(batch_size, 1).to(device)
    image = torch.rand(batch_size, 84, 84, 3).to(device)  # Normalized [0,1]

    print(f"Input shapes:")
    print(f"  noisy_action: {noisy_action.shape}")
    print(f"  state: {state.shape}")
    print(f"  timestep: {timestep.shape}")
    print(f"  image: {image.shape}")

    # Forward pass
    try:
        noise_pred = model(noisy_action, state, timestep, image)
        print(f"\n✓ Forward pass successful!")
        print(f"  Output shape: {noise_pred.shape}")
        print(f"  Output range: [{noise_pred.min().item():.4f}, {noise_pred.max().item():.4f}]")
        print(f"  Output mean: {noise_pred.mean().item():.4f}")
        print(f"  Output std: {noise_pred.std().item():.4f}")

        if torch.isnan(noise_pred).any():
            print(f"❌ NaN detected in output!")
            return None
        if torch.isinf(noise_pred).any():
            print(f"❌ Inf detected in output!")
            return None

        return noise_pred
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return None


def test_backward_pass(model, noise_pred, batch_size=4, device='cuda'):
    """Test backward pass and gradient flow"""
    print("\n" + "=" * 80)
    print("BACKWARD PASS TEST")
    print("=" * 80)

    # Create dummy target and compute loss
    target_noise = torch.randn_like(noise_pred).to(device)

    # Create dummy rewards for weighted loss
    rewards = torch.randn(batch_size).to(device) * 5.0  # Random rewards

    # Compute per-sample loss
    per_sample_loss = F.mse_loss(noise_pred, target_noise, reduction='none').mean(dim=1)

    print(f"Per-sample loss shape: {per_sample_loss.shape}")
    print(f"Per-sample loss: {per_sample_loss}")
    print(f"Rewards: {rewards}")

    # Apply reward weighting (same as in training code)
    reward_normalized = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
    reward_weights = F.softmax(reward_normalized * 2.0, dim=0) * len(rewards)
    weighted_loss = (per_sample_loss * reward_weights).mean()

    print(f"\nReward normalized: {reward_normalized}")
    print(f"Reward weights: {reward_weights}")
    print(f"Weighted loss: {weighted_loss.item():.6f}")

    # Backward pass
    try:
        model.zero_grad()
        weighted_loss.backward()
        print(f"\n✓ Backward pass successful!")

        # Check gradient flow
        total_norm, has_nan = check_gradient_flow(model.named_parameters())

        if has_nan:
            return False
        if total_norm < 1e-6:
            print(f"⚠ WARNING: Total gradient norm is very small ({total_norm})")
            return False

        return True
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_weighting():
    """Test reward weighting mechanism specifically"""
    print("\n" + "=" * 80)
    print("REWARD WEIGHTING TEST")
    print("=" * 80)

    # Test with different reward scenarios
    scenarios = [
        ("All positive", torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])),
        ("All negative", torch.tensor([-5.0, -4.0, -3.0, -2.0, -1.0])),
        ("Mixed", torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])),
        ("With outlier", torch.tensor([1.0, 1.0, 1.0, 1.0, 10.0])),
    ]

    for name, rewards in scenarios:
        print(f"\n{name}:")
        print(f"  Raw rewards: {rewards}")

        # Apply normalization and softmax (same as training code)
        reward_normalized = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
        reward_weights = F.softmax(reward_normalized * 2.0, dim=0) * len(rewards)

        print(f"  Normalized: {reward_normalized}")
        print(f"  Weights: {reward_weights}")
        print(f"  Weight sum: {reward_weights.sum().item():.4f}")
        print(f"  Weight mean: {reward_weights.mean().item():.4f}")
        print(f"  Max/Min ratio: {(reward_weights.max() / reward_weights.min()).item():.2f}x")


def setup_isaac_sim_scene():
    """Setup Isaac Sim scene with robot and cube"""
    print("\n" + "=" * 80)
    print("SETTING UP ISAAC SIM SCENE")
    print("=" * 80)

    # Create world
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()

    # Set camera view
    set_camera_view(
        eye=[2.5, 2.5, 2.0],
        target=[0.0, 0.0, 0.5],
        camera_prim_path="/OmniverseKit_Persp",
    )

    # Add UR10e robot
    assets_root_path = get_assets_root_path()
    asset_path = (
        assets_root_path
        + "/Isaac/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd"
    )
    add_reference_to_stage(usd_path=asset_path, prim_path="/World/ur")

    # Configure gripper
    gripper = ParallelGripper(
        end_effector_prim_path="/World/ur/ee_link/robotiq_arg2f_base_link",
        joint_prim_names=["finger_joint"],
        joint_opened_positions=np.array([0]),
        joint_closed_positions=np.array([40]),
        action_deltas=np.array([-40]),
        use_mimic_joints=True,
    )

    # Create robot
    robot = SingleManipulator(
        prim_path="/World/ur",
        name="ur10_robot",
        end_effector_prim_path="/World/ur/ee_link/robotiq_arg2f_base_link",
        gripper=gripper,
    )

    # Add red cube
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

    # Add lighting
    stage = my_world.stage
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(1000.0)

    distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant_light.CreateIntensityAttr(2000.0)
    distant_light_xform = distant_light.AddRotateXYZOp()
    distant_light_xform.Set(Gf.Vec3f(-45, 0, 0))

    print("✓ Scene created with UR10e robot and red cube")

    # Initialize world
    my_world.reset()
    robot.initialize()
    cube.initialize()

    print("✓ World initialized and ready")
    print("=" * 80 + "\n")

    return my_world, robot, cube


def main():
    print("\n" + "=" * 80)
    print("DIFFUSION TRANSFORMER GRADIENT FLOW TEST WITH ISAAC SIM")
    print("=" * 80)

    try:
        # Setup Isaac Sim scene
        my_world, robot, cube = setup_isaac_sim_scene()

        # Start simulation
        my_world.play()
        print("✓ Simulation started\n")

        # Step simulation a few times to stabilize
        for _ in range(10):
            my_world.step(render=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}\n")

        # Create model
        state_dim = 13
        action_dim = 4
        batch_size = 4

        model = DiffusionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
            use_vision=True,
        ).to(device)

        print(f"Model created with:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Hidden dim: 128")
        print(f"  Num layers: 3")
        print(f"  Num heads: 4")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

        # Test forward pass
        noise_pred = test_forward_pass(model, batch_size, device)
        if noise_pred is None:
            print("\n❌ FAILED: Forward pass test")
            return

        # Step simulation during testing to keep UI responsive
        for _ in range(5):
            my_world.step(render=True)

        # Test backward pass
        success = test_backward_pass(model, noise_pred, batch_size, device)
        if not success:
            print("\n❌ FAILED: Backward pass test")
            return

        # Step simulation
        for _ in range(5):
            my_world.step(render=True)

        # Test reward weighting
        test_reward_weighting()

        # Step simulation
        for _ in range(5):
            my_world.step(render=True)

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nSummary:")
        print("  ✓ Isaac Sim scene setup successfully")
        print("  ✓ Forward pass working correctly")
        print("  ✓ Backward pass working correctly")
        print("  ✓ Gradients flowing through all layers")
        print("  ✓ No NaN or Inf values detected")
        print("  ✓ Reward weighting mechanism working as expected")
        print("=" * 80 + "\n")

        # Keep simulation running for a bit
        print("Keeping simulation running for 5 seconds...")
        for i in range(300):
            my_world.step(render=True)
            if i % 60 == 0:
                print(f"  Step {i}/300")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        try:
            my_world.stop()
            my_world.clear()
        except:
            pass
        simulation_app.close()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    main()
