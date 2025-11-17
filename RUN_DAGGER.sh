#!/bin/bash

# DAgger Training Workflow
# Run this script to train a robot policy with DAgger

set -e  # Exit on error

ISAAC_PYTHON="/home/kenpeter/work/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh"

echo "============================================================"
echo "DAGGER TRAINING WORKFLOW"
echo "============================================================"
echo ""

# Step 1: Collect expert data
echo "[Step 1/5] Collecting 30 diverse expert episodes..."
$ISAAC_PYTHON collect_data.py
echo "✓ Expert data collected"
echo ""

# Step 2: Train initial policy
echo "[Step 2/5] Training initial policy on expert data..."
python3 train.py
echo "✓ Initial policy trained"
echo ""

# Step 3: DAgger collection
echo "[Step 3/5] Running DAgger - collecting expert corrections..."
$ISAAC_PYTHON dagger_collect.py
echo "✓ DAgger data collected"
echo ""

# Step 4: Retrain with combined data
echo "[Step 4/5] Retraining with combined expert + DAgger data..."
python3 train.py
echo "✓ Policy retrained with DAgger"
echo ""

# Step 5: Test
echo "[Step 5/5] Testing improved policy..."
echo "Opening Isaac Sim for visual testing..."
$ISAAC_PYTHON test.py

echo ""
echo "============================================================"
echo "✓ DAGGER WORKFLOW COMPLETE!"
echo "============================================================"
