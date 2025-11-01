#!/bin/bash
# Install Flash Attention 4 for Isaac Sim (bypass conda)

ISAAC_SIM=/home/kenpeter/work/isaac-sim-standalone-5.1.0-linux-x86_64

echo "=== Installing Flash Attention 4 for Isaac Sim ==="
echo ""

# Unset conda variables temporarily
unset CONDA_DEFAULT_ENV
unset CONDA_PREFIX

# Install ninja
echo "Installing ninja..."
$ISAAC_SIM/python.sh -m pip install ninja --user

echo ""
echo "Testing PyTorch CUDA..."
$ISAAC_SIM/python.sh -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "To test Flash Attention 4:"
echo "  $ISAAC_SIM/python.sh test_flash_attention_4.py"
echo ""
echo "To run robot training:"
echo "  $ISAAC_SIM/python.sh simple_rl_robot_arm.py"
