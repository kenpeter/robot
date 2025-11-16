#!/bin/bash
# Step 1: Train the model (no visualization, fast)

echo "========================================="
echo "STEP 1: OFFLINE TRAINING (No Visualization)"
echo "========================================="
echo ""

# Verify data exists
if [ ! -f transitions.pkl ]; then
    echo "❌ ERROR: transitions.pkl not found!"
    echo "   Please run collect_data.py first"
    exit 1
fi

# Verify data is good
python3 << 'EOF'
import pickle
with open('transitions.pkl', 'rb') as f:
    t = pickle.load(f)
g = [x[1][6] for x in t]
open_count = sum(1 for x in g if x>0.3)
if open_count == 0:
    print("❌ ERROR: Bad data (all gripper values are 0)!")
    exit(1)
print(f"✅ Data verified: {len(t)} transitions, {open_count} OPEN gripper actions")
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo "Starting training (no Isaac Sim window)..."
echo "This will take ~2-3 minutes"
echo ""

# Delete old model to start fresh
rm -f offline_rl_model.pth

# Run training (no visualization)
python3 train_offline_no_visual.py

echo ""
echo "✅ Training complete!"
echo "   Next: Run ./2_TEST_VISUAL.sh to see the trained robot"
