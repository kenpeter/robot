#!/bin/bash
# Step 2: Test the trained model with visualization

echo "========================================="
echo "STEP 2: VISUAL TESTING (Isaac Sim UI)"
echo "========================================="
echo ""

# Verify model exists
if [ ! -f offline_rl_model.pth ]; then
    echo "❌ ERROR: offline_rl_model.pth not found!"
    echo "   Please run ./1_TRAIN.sh first"
    exit 1
fi

echo "✅ Found trained model"
echo ""
echo "Starting Isaac Sim with visualization..."
echo "The robot will execute the trained policy"
echo ""

# Run visual testing
/home/kenpeter/work/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh test_visual.py
