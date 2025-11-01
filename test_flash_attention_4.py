#!/usr/bin/env python3
"""
Test Flash Attention 4 on RTX 4070 Ti
"""

import torch
import time
import numpy as np

def test_cuda_setup():
    """Verify CUDA setup"""
    print("="*60)
    print("CUDA Setup Check")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

        capability = torch.cuda.get_device_capability(0)
        print(f"Compute capability: {capability[0]}.{capability[1]}")

        if capability[0] >= 8:
            print("✓ Your GPU supports Flash Attention 4!")
        else:
            print("✗ Your GPU does not support Flash Attention 4 (requires SM 8.0+)")
    else:
        print("✗ CUDA is not available")
        return False

    print()
    return True


def benchmark_attention(batch_size=4, num_heads=8, seq_len=512, head_dim=64, num_runs=100):
    """Benchmark Flash Attention 4 vs standard attention"""
    print("="*60)
    print(f"Benchmarking Attention")
    print("="*60)
    print(f"Config: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")
    print()

    device = 'cuda'
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    # Standard PyTorch attention
    print("Testing standard PyTorch attention...")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out_std = torch.matmul(attn, v)
    torch.cuda.synchronize()
    std_time = (time.time() - start) / num_runs * 1000
    print(f"  Time: {std_time:.3f} ms")

    # PyTorch scaled_dot_product_attention (FA2/FA3)
    print("Testing F.scaled_dot_product_attention (FA2/FA3)...")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        out_sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    sdpa_time = (time.time() - start) / num_runs * 1000
    print(f"  Time: {sdpa_time:.3f} ms")
    print(f"  Speedup vs standard: {std_time/sdpa_time:.2f}x")

    # Flash Attention 4 (if available)
    try:
        from flash_attention_4_wrapper import flash_attention_4
        print("Testing Flash Attention 4...")

        # Warmup
        for _ in range(10):
            _ = flash_attention_4(q, k, v)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            out_fa4 = flash_attention_4(q, k, v)
        torch.cuda.synchronize()
        fa4_time = (time.time() - start) / num_runs * 1000
        print(f"  Time: {fa4_time:.3f} ms")
        print(f"  Speedup vs standard: {std_time/fa4_time:.2f}x")
        print(f"  Speedup vs SDPA: {sdpa_time/fa4_time:.2f}x")

        # Verify correctness
        diff = (out_fa4 - out_sdpa).abs().max().item()
        print(f"  Max diff vs SDPA: {diff:.6f}")

        if diff < 1e-2:
            print("  ✓ Flash Attention 4 output matches reference")
        else:
            print("  ✗ Flash Attention 4 output differs from reference")

    except ImportError as e:
        print(f"Flash Attention 4 not available: {e}")
    except Exception as e:
        print(f"Error testing Flash Attention 4: {e}")
        import traceback
        traceback.print_exc()

    print()


def test_different_sizes():
    """Test various sequence lengths"""
    print("="*60)
    print("Testing Different Sequence Lengths")
    print("="*60)

    try:
        from flash_attention_4_wrapper import flash_attention_4

        for seq_len in [64, 128, 256, 512, 1024]:
            q = torch.randn(2, 4, seq_len, 64, device='cuda')
            k = torch.randn(2, 4, seq_len, 64, device='cuda')
            v = torch.randn(2, 4, seq_len, 64, device='cuda')

            # FA4
            torch.cuda.synchronize()
            start = time.time()
            out_fa4 = flash_attention_4(q, k, v)
            torch.cuda.synchronize()
            fa4_time = (time.time() - start) * 1000

            # SDPA
            torch.cuda.synchronize()
            start = time.time()
            out_sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            sdpa_time = (time.time() - start) * 1000

            speedup = sdpa_time / fa4_time
            diff = (out_fa4 - out_sdpa).abs().max().item()

            print(f"seq_len={seq_len:4d}: FA4={fa4_time:6.2f}ms, SDPA={sdpa_time:6.2f}ms, "
                  f"speedup={speedup:.2f}x, diff={diff:.6f}")

    except Exception as e:
        print(f"Error: {e}")

    print()


def test_robot_arm_integration():
    """Test FA4 with the robot arm DiT"""
    print("="*60)
    print("Testing Robot Arm DiT Integration")
    print("="*60)

    try:
        # Import the DiT agent
        import sys
        sys.path.insert(0, '/home/kenpeter/work/robot')

        # Suppress Isaac Sim imports for testing
        import unittest.mock as mock
        with mock.patch.dict('sys.modules', {'isaacsim': mock.MagicMock()}):
            from simple_rl_robot_arm import DiTAgent, USE_FA4

            print(f"Flash Attention 4 enabled: {USE_FA4}")

            if USE_FA4:
                print("✓ DiTAgent will use Flash Attention 4")
            else:
                print("✗ DiTAgent will use fallback attention")

            # Create agent
            state_dim = 8
            action_dim = 7
            agent = DiTAgent(state_dim, action_dim)

            print(f"Agent created successfully")
            print(f"  State dim: {state_dim}")
            print(f"  Action dim: {action_dim}")
            print(f"  Device: {agent.device}")

            # Test action generation
            test_state = torch.randn(state_dim).numpy()
            action = agent.get_action(test_state)

            print(f"  Generated action shape: {action.shape}")
            print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
            print("✓ Robot arm integration test passed")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*12 + "Flash Attention 4 Test Suite" + " "*18 + "║")
    print("║" + " "*15 + "RTX 4070 Ti (Ada Lovelace)" + " "*16 + "║")
    print("╚" + "="*58 + "╝")
    print("\n")

    # Run tests
    if not test_cuda_setup():
        print("CUDA setup failed. Exiting.")
        exit(1)

    benchmark_attention(batch_size=4, num_heads=8, seq_len=512, head_dim=64, num_runs=100)
    test_different_sizes()
    test_robot_arm_integration()

    print("="*60)
    print("All tests complete!")
    print("="*60)
