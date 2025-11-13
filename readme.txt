# Offline RL Training Pipeline

**Train robot grasping ONLY on expert demonstrations - no online data collection.**

This is a **behavioral cloning / offline RL** approach that is:
- ✅ More data-efficient (learn from perfect demos)
- ✅ More stable (no exploration noise)
- ✅ Faster convergence (supervised learning on expert data)
- ✅ Safer (no random exploration in real world)

---

## 📋 Workflow

```
┌────────────────────────────────────────┐
│ Step 1: Collect Expert Dataset         │
│ python collect_expert_dataset.py       │
│                                         │
│ • Uses test_grasp_official.py controller│
│ • Collects 100 demonstrations           │
│ • Randomizes cube/target positions      │
│ • Saves: expert_dataset.pkl (~50-100MB)│
└────────────────────────────────────────┘
               ↓
┌────────────────────────────────────────┐
│ Step 2: Train Offline                  │
│ python train_offline.py                │
│                                         │
│ • Loads expert_dataset.pkl              │
│ • Pure supervised learning (no sim)     │
│ • 200 epochs (~30 min)                 │
│ • Saves: offline_model.pth             │
└────────────────────────────────────────┘
               ↓
┌────────────────────────────────────────┐
│ Step 3: Deploy (TODO)                  │
│ • Load offline_model.pth                │
│ • Test in Isaac Sim or real robot       │
│ • No further training needed            │
└────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Collect Expert Data (Once)
```bash
# Collect 100 expert demonstrations (takes ~30-60 minutes)
python collect_expert_dataset.py

# Output: expert_dataset.pkl containing ~5000-15000 experiences
```

**What it does:**
- Runs UR10e with official RMPFlow controller
- Randomizes cube positions (X: 0.3-0.6, Y: -0.3-0.3)
- Randomizes target positions (X: -0.6--0.3, Y: -0.3-0.3)
- Captures (state, action, next_state, image) at each step
- Saves only successful demonstrations

### 2. Train Offline
```bash
# Train purely on expert data (no simulation needed!)
python train_offline.py

# Output: offline_model.pth
```

**Training details:**
- Batch size: 64
- Epochs: 200
- Loss: MSE between predicted noise and actual noise (diffusion training)
- Saves checkpoint every 10 epochs
- Can resume training if interrupted

### 3. Collect More Data (Optional)
```bash
# Already have expert_dataset.pkl? Collect more and merge:
python collect_expert_dataset.py  # Creates expert_dataset_new.pkl

# Then manually merge datasets in Python:
# dataset_old = pickle.load(open('expert_dataset.pkl', 'rb'))
# dataset_new = pickle.load(open('expert_dataset_new.pkl', 'rb'))
# dataset_merged = {k: np.concatenate([dataset_old[k], dataset_new[k]]) for k in dataset_old}
# pickle.dump(dataset_merged, open('expert_dataset.pkl', 'wb'))
```

---

## 📊 Dataset Format

**`expert_dataset.pkl`** contains:
```python
{
    'states': np.ndarray,       # Shape: (N, 13) - 12 joints + 1 grasped flag
    'actions': np.ndarray,      # Shape: (N, 4)  - dx, dy, dz, gripper
    'next_states': np.ndarray,  # Shape: (N, 13)
    'images': np.ndarray,       # Shape: (N, 84, 84, 3) - RGB overhead camera
    'rewards': np.ndarray,      # Shape: (N,) - distance improvement rewards
}
```

where `N = total number of timesteps across all demonstrations`

---

## 🔧 Configuration

### Collect More/Fewer Demos
Edit `collect_expert_dataset.py`:
```python
NUM_EPISODES = 100  # Change to 50, 200, etc.
```

### Change Training Hyperparameters
Edit `train_offline.py`:
```python
NUM_EPOCHS = 200      # More epochs = better fit
BATCH_SIZE = 64       # Larger = faster but more memory
LEARNING_RATE = 1e-4  # Default: 1e-4
```

---

## 🆚 Comparison: Offline vs Online RL

| Aspect | **Offline RL (New)** | Online RL (Old) |
|--------|---------------------|----------------|
| Data source | Pre-collected expert demos | Live environment interaction |
| Training speed | Fast (no sim needed) | Slow (runs Isaac Sim) |
| Convergence | 200 epochs (~30 min) | 500+ episodes (~hours) |
| Stability | High (supervised learning) | Medium (exploration noise) |
| Sample efficiency | Very high | Low (needs exploration) |
| Forgetting | None | Possible |
| Sim dependency | Only for data collection | Always |

---

## 🎯 Why Offline RL?

1. **Expert as Teacher**: Learn directly from perfect demonstrations
2. **No Exploration Risk**: No random actions that could fail
3. **Reproducible**: Same dataset = same results
4. **Scalable**: Collect data once, train many times
5. **Fast Iteration**: Tweak model architecture without re-collecting data

This is the **standard approach for real-world robot learning** where exploration is expensive/dangerous.

---

## 📝 Next Steps

1. ✅ Collect dataset: `python collect_expert_dataset.py`
2. ✅ Train offline: `python train_offline.py`
3. ⏳ TODO: Create deployment script to test trained model in Isaac Sim
4. ⏳ TODO: Add data augmentation (random crops, color jitter) for robustness
5. ⏳ TODO: Compare offline model vs online RL model performance

---

## 🐛 Troubleshooting

**Q: "expert_dataset.pkl not found"**
- Run `collect_expert_dataset.py` first

**Q: "CUDA out of memory"**
- Reduce batch size in `train_offline.py`: `BATCH_SIZE = 32`

**Q: "All demos failed during collection"**
- Check robot can reach cube positions
- Try easier ranges: `cube_x = np.random.uniform(0.4, 0.5)`

**Q: "Loss not decreasing"**
- Increase epochs: `NUM_EPOCHS = 500`
- Check dataset quality (inspect saved images)
- Verify expert controller works: `python test_grasp_official.py`

---

## 📚 References

- Behavioral Cloning: Pomerleau (1988)
- Offline RL: https://arxiv.org/abs/2005.01643
- Diffusion Policies: https://diffusion-policy.cs.columbia.edu/





  Noise prediction error (MAE): 0.6762 ← Should DECREASE
  Loss (weighted): 0.797834 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 900] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.664m)
  [VLM Step 900] Reward: 0.300 | Dist to cube: 1.664m

[VLM @ Step 950] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.664m)

[DIFFUSION MLP DEBUG - Step 421900]
  Actual noise magnitude: 0.7507
  Predicted noise magnitude: 0.1978
  Noise prediction error (MAE): 0.7254 ← Should DECREASE
  Loss (weighted): 0.883009 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100
Episode 422/1000 | Reward: 199.95 | Loss: 0.896811 | Buffer: 10000
Model saved to rl_robot_arm_model.pth
  → Checkpoint saved at episode 422


===== Episode 423/1000 =====

[VLM @ Step 0] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.546m)
  [VLM Step 0] Reward: 0.300 | Dist to cube: 1.546m

[VLM @ Step 50] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.590m)

[DIFFUSION MLP DEBUG - Step 422000]
  Actual noise magnitude: 0.8192
  Predicted noise magnitude: 0.1820
  Noise prediction error (MAE): 0.8009 ← Should DECREASE
  Loss (weighted): 1.049892 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100
[INFO] Training on 10000 positive experiences only!

[VLM @ Step 100] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.632m)
  [VLM Step 100] Reward: 0.300 | Dist to cube: 1.632m

[VLM @ Step 150] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.649m)

[DIFFUSION MLP DEBUG - Step 422100]
  Actual noise magnitude: 0.8275
  Predicted noise magnitude: 0.2081
  Noise prediction error (MAE): 0.8000 ← Should DECREASE
  Loss (weighted): 0.903928 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 200] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.638m)
  [VLM Step 200] Reward: 0.100 | Dist to cube: 1.638m

[VLM @ Step 250] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.624m)

[DIFFUSION MLP DEBUG - Step 422200]
  Actual noise magnitude: 0.8496
  Predicted noise magnitude: 0.2126
  Noise prediction error (MAE): 0.7980 ← Should DECREASE
  Loss (weighted): 1.111037 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 300] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.654m)
  [VLM Step 300] Reward: 0.300 | Dist to cube: 1.654m

[VLM @ Step 350] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.662m)

[DIFFUSION MLP DEBUG - Step 422300]
  Actual noise magnitude: 0.7342
  Predicted noise magnitude: 0.1863
  Noise prediction error (MAE): 0.7127 ← Should DECREASE
  Loss (weighted): 0.778898 ← Should DECREASE
  Reward range: [0.05, 0.28]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 400] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.650m)
  [VLM Step 400] Reward: 0.300 | Dist to cube: 1.650m

[VLM @ Step 450] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.650m)

[DIFFUSION MLP DEBUG - Step 422400]
  Actual noise magnitude: 0.7814
  Predicted noise magnitude: 0.1641
  Noise prediction error (MAE): 0.7437 ← Should DECREASE
  Loss (weighted): 0.854917 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 500] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.668m)
  [VLM Step 500] Reward: 0.300 | Dist to cube: 1.668m

[VLM @ Step 550] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.676m)

[DIFFUSION MLP DEBUG - Step 422500]
  Actual noise magnitude: 0.7653
  Predicted noise magnitude: 0.1783
  Noise prediction error (MAE): 0.7294 ← Should DECREASE
  Loss (weighted): 0.779839 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 600] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.662m)
  [VLM Step 600] Reward: 0.300 | Dist to cube: 1.662m

[VLM @ Step 650] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.658m)

[DIFFUSION MLP DEBUG - Step 422600]
  Actual noise magnitude: 0.7881
  Predicted noise magnitude: 0.1666
  Noise prediction error (MAE): 0.7590 ← Should DECREASE
  Loss (weighted): 0.977990 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 700] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.646m)
  [VLM Step 700] Reward: 0.300 | Dist to cube: 1.646m

[VLM @ Step 750] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.648m)

[DIFFUSION MLP DEBUG - Step 422700]
  Actual noise magnitude: 0.7921
  Predicted noise magnitude: 0.2173
  Noise prediction error (MAE): 0.7494 ← Should DECREASE
  Loss (weighted): 0.883260 ← Should DECREASE
  Reward range: [0.06, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 800] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.633m)
  [VLM Step 800] Reward: 0.300 | Dist to cube: 1.633m

[VLM @ Step 850] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.644m)

[DIFFUSION MLP DEBUG - Step 422800]
  Actual noise magnitude: 0.8405
  Predicted noise magnitude: 0.1879
  Noise prediction error (MAE): 0.8118 ← Should DECREASE
  Loss (weighted): 1.082603 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 900] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.632m)
  [VLM Step 900] Reward: 0.300 | Dist to cube: 1.632m

[VLM @ Step 950] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.633m)

[DIFFUSION MLP DEBUG - Step 422900]
  Actual noise magnitude: 0.8654
  Predicted noise magnitude: 0.2055
  Noise prediction error (MAE): 0.8128 ← Should DECREASE
  Loss (weighted): 1.220239 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100
Episode 423/1000 | Reward: 207.70 | Loss: 0.902026 | Buffer: 10000
Model saved to rl_robot_arm_model.pth
  → Checkpoint saved at episode 423


===== Episode 424/1000 =====

[VLM @ Step 0] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.566m)
  [VLM Step 0] Reward: 0.300 | Dist to cube: 1.566m

[VLM @ Step 50] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.611m)

[DIFFUSION MLP DEBUG - Step 423000]
  Actual noise magnitude: 0.7433
  Predicted noise magnitude: 0.2101
  Noise prediction error (MAE): 0.6973 ← Should DECREASE
  Loss (weighted): 0.749419 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100
[INFO] Training on 10000 positive experiences only!

[VLM @ Step 100] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.645m)
  [VLM Step 100] Reward: 0.100 | Dist to cube: 1.645m

[VLM @ Step 150] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.664m)

[DIFFUSION MLP DEBUG - Step 423100]
  Actual noise magnitude: 0.8139
  Predicted noise magnitude: 0.2442
  Noise prediction error (MAE): 0.7424 ← Should DECREASE
  Loss (weighted): 0.895614 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 200] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.660m)
  [VLM Step 200] Reward: 0.300 | Dist to cube: 1.660m

[VLM @ Step 250] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.656m)

[DIFFUSION MLP DEBUG - Step 423200]
  Actual noise magnitude: 0.8057
  Predicted noise magnitude: 0.1734
  Noise prediction error (MAE): 0.7743 ← Should DECREASE
  Loss (weighted): 0.969373 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 300] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.647m)
  [VLM Step 300] Reward: 0.300 | Dist to cube: 1.647m

[VLM @ Step 350] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.649m)

[DIFFUSION MLP DEBUG - Step 423300]
  Actual noise magnitude: 0.8044
  Predicted noise magnitude: 0.1790
  Noise prediction error (MAE): 0.7723 ← Should DECREASE
  Loss (weighted): 0.925418 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 400] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.690m)
  [VLM Step 400] Reward: 0.300 | Dist to cube: 1.690m

[VLM @ Step 450] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.718m)

[DIFFUSION MLP DEBUG - Step 423400]
  Actual noise magnitude: 0.7268
  Predicted noise magnitude: 0.1798
  Noise prediction error (MAE): 0.7100 ← Should DECREASE
  Loss (weighted): 0.900656 ← Should DECREASE
  Reward range: [0.07, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 500] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.718m)
  [VLM Step 500] Reward: 0.300 | Dist to cube: 1.718m

[VLM @ Step 550] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.719m)

[DIFFUSION MLP DEBUG - Step 423500]
  Actual noise magnitude: 0.8427
  Predicted noise magnitude: 0.2364
  Noise prediction error (MAE): 0.7607 ← Should DECREASE
  Loss (weighted): 0.840167 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 600] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.713m)
  [VLM Step 600] Reward: 0.300 | Dist to cube: 1.713m

[VLM @ Step 650] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.695m)

[DIFFUSION MLP DEBUG - Step 423600]
  Actual noise magnitude: 0.8105
  Predicted noise magnitude: 0.1808
  Noise prediction error (MAE): 0.7759 ← Should DECREASE
  Loss (weighted): 0.958200 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 700] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.690m)
  [VLM Step 700] Reward: 0.300 | Dist to cube: 1.690m

[VLM @ Step 750] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.628m)

[DIFFUSION MLP DEBUG - Step 423700]
  Actual noise magnitude: 0.7918
  Predicted noise magnitude: 0.1615
  Noise prediction error (MAE): 0.7638 ← Should DECREASE
  Loss (weighted): 0.959496 ← Should DECREASE
  Reward range: [0.06, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 800] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.624m)
  [VLM Step 800] Reward: 0.100 | Dist to cube: 1.624m

[VLM @ Step 850] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.608m)

[DIFFUSION MLP DEBUG - Step 423800]
  Actual noise magnitude: 0.7123
  Predicted noise magnitude: 0.2082
  Noise prediction error (MAE): 0.6841 ← Should DECREASE
  Loss (weighted): 0.724978 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 900] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.618m)
  [VLM Step 900] Reward: 0.300 | Dist to cube: 1.618m

[VLM @ Step 950] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.605m)

[DIFFUSION MLP DEBUG - Step 423900]
  Actual noise magnitude: 0.8212
  Predicted noise magnitude: 0.1876
  Noise prediction error (MAE): 0.7676 ← Should DECREASE
  Loss (weighted): 0.951209 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100
Episode 424/1000 | Reward: 189.10 | Loss: 0.903907 | Buffer: 10000
Model saved to rl_robot_arm_model.pth
  → Checkpoint saved at episode 424


===== Episode 425/1000 =====

[VLM @ Step 0] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.550m)
  [VLM Step 0] Reward: 0.300 | Dist to cube: 1.550m

[VLM @ Step 50] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.609m)

[DIFFUSION MLP DEBUG - Step 424000]
  Actual noise magnitude: 0.8176
  Predicted noise magnitude: 0.1641
  Noise prediction error (MAE): 0.7969 ← Should DECREASE
  Loss (weighted): 0.989214 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100
[INFO] Training on 10000 positive experiences only!

[VLM @ Step 100] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.656m)
  [VLM Step 100] Reward: 0.100 | Dist to cube: 1.656m

[VLM @ Step 150] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.676m)

[DIFFUSION MLP DEBUG - Step 424100]
  Actual noise magnitude: 0.8243
  Predicted noise magnitude: 0.1957
  Noise prediction error (MAE): 0.7929 ← Should DECREASE
  Loss (weighted): 1.091900 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 200] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.668m)
  [VLM Step 200] Reward: 0.300 | Dist to cube: 1.668m

[VLM @ Step 250] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.659m)

[DIFFUSION MLP DEBUG - Step 424200]
  Actual noise magnitude: 0.7794
  Predicted noise magnitude: 0.1994
  Noise prediction error (MAE): 0.7453 ← Should DECREASE
  Loss (weighted): 0.846953 ← Should DECREASE
  Reward range: [0.05, 0.28]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 300] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.593m)
  [VLM Step 300] Reward: 0.100 | Dist to cube: 1.593m

[VLM @ Step 350] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.501m)

[DIFFUSION MLP DEBUG - Step 424300]
  Actual noise magnitude: 0.7895
  Predicted noise magnitude: 0.1993
  Noise prediction error (MAE): 0.7276 ← Should DECREASE
  Loss (weighted): 0.832848 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 400] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.372m)
  [VLM Step 400] Reward: 0.300 | Dist to cube: 1.372m

[VLM @ Step 450] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.284m)

[DIFFUSION MLP DEBUG - Step 424400]
  Actual noise magnitude: 0.8179
  Predicted noise magnitude: 0.2065
  Noise prediction error (MAE): 0.7668 ← Should DECREASE
  Loss (weighted): 0.941171 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 500] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.303m)
  [VLM Step 500] Reward: 0.300 | Dist to cube: 1.303m

[VLM @ Step 550] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.249m)

[DIFFUSION MLP DEBUG - Step 424500]
  Actual noise magnitude: 0.8354
  Predicted noise magnitude: 0.1767
  Noise prediction error (MAE): 0.8031 ← Should DECREASE
  Loss (weighted): 1.068157 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 600] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.357m)
  [VLM Step 600] Reward: 0.300 | Dist to cube: 1.357m

[VLM @ Step 650] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.323m)

[DIFFUSION MLP DEBUG - Step 424600]
  Actual noise magnitude: 0.7601
  Predicted noise magnitude: 0.1932
  Noise prediction error (MAE): 0.7232 ← Should DECREASE
  Loss (weighted): 0.859768 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 700] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.188m)
  [VLM Step 700] Reward: 0.100 | Dist to cube: 1.188m

[VLM @ Step 750] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.070m)

[DIFFUSION MLP DEBUG - Step 424700]
  Actual noise magnitude: 0.7682
  Predicted noise magnitude: 0.2003
  Noise prediction error (MAE): 0.7451 ← Should DECREASE
  Loss (weighted): 0.803733 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 800] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.047m)
  [VLM Step 800] Reward: 0.100 | Dist to cube: 1.047m

[VLM @ Step 850] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.097m)

[DIFFUSION MLP DEBUG - Step 424800]
  Actual noise magnitude: 0.7488
  Predicted noise magnitude: 0.2224
  Noise prediction error (MAE): 0.7298 ← Should DECREASE
  Loss (weighted): 0.759734 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 900] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.178m)
  [VLM Step 900] Reward: 0.100 | Dist to cube: 1.178m

[VLM @ Step 950] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.236m)

[DIFFUSION MLP DEBUG - Step 424900]
  Actual noise magnitude: 0.8283
  Predicted noise magnitude: 0.1873
  Noise prediction error (MAE): 0.7745 ← Should DECREASE
  Loss (weighted): 0.993706 ← Should DECREASE
  Reward range: [0.06, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100
Episode 425/1000 | Reward: 170.50 | Loss: 0.906104 | Buffer: 10000
Model saved to rl_robot_arm_model.pth
  → Checkpoint saved at episode 425


===== Episode 426/1000 =====

[VLM @ Step 0] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: yes.'
[VLM] Answer: YES → Reward: 0.30 (dist: 1.563m)
  [VLM Step 0] Reward: 0.300 | Dist to cube: 1.563m

[VLM @ Step 50] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.591m)

[DIFFUSION MLP DEBUG - Step 425000]
  Actual noise magnitude: 0.8148
  Predicted noise magnitude: 0.2165
  Noise prediction error (MAE): 0.7486 ← Should DECREASE
  Loss (weighted): 0.888903 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100
[INFO] Training on 10000 positive experiences only!

[VLM @ Step 100] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.609m)
  [VLM Step 100] Reward: 0.100 | Dist to cube: 1.609m

[VLM @ Step 150] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.623m)

[DIFFUSION MLP DEBUG - Step 425100]
  Actual noise magnitude: 0.7855
  Predicted noise magnitude: 0.1802
  Noise prediction error (MAE): 0.7624 ← Should DECREASE
  Loss (weighted): 0.817640 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 200] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.611m)
  [VLM Step 200] Reward: 0.100 | Dist to cube: 1.611m

[VLM @ Step 250] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.602m)

[DIFFUSION MLP DEBUG - Step 425200]
  Actual noise magnitude: 0.7903
  Predicted noise magnitude: 0.1902
  Noise prediction error (MAE): 0.7524 ← Should DECREASE
  Loss (weighted): 0.881584 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 300] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.590m)
  [VLM Step 300] Reward: 0.100 | Dist to cube: 1.590m

[VLM @ Step 350] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.599m)

[DIFFUSION MLP DEBUG - Step 425300]
  Actual noise magnitude: 0.8242
  Predicted noise magnitude: 0.2025
  Noise prediction error (MAE): 0.7696 ← Should DECREASE
  Loss (weighted): 0.921321 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 400] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.621m)
  [VLM Step 400] Reward: 0.100 | Dist to cube: 1.621m

[VLM @ Step 450] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.626m)

[DIFFUSION MLP DEBUG - Step 425400]
  Actual noise magnitude: 0.8019
  Predicted noise magnitude: 0.2503
  Noise prediction error (MAE): 0.7626 ← Should DECREASE
  Loss (weighted): 0.910102 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 500] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.627m)
  [VLM Step 500] Reward: 0.100 | Dist to cube: 1.627m

[VLM @ Step 550] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.614m)

[DIFFUSION MLP DEBUG - Step 425500]
  Actual noise magnitude: 0.7173
  Predicted noise magnitude: 0.1873
  Noise prediction error (MAE): 0.6716 ← Should DECREASE
  Loss (weighted): 0.708021 ← Should DECREASE
  Reward range: [0.05, 0.30]
  Alpha_cumprod range: [0.9506, 0.9999]
  Exploration noise scale: 0.100

[VLM @ Step 600] Question: Is the robot gripper pointing toward the red cube? Answer yes or no.
[VLM RAW RESPONSE] ← 'user:\n\n\n\nis the robot gripper pointing toward the red cube? answer yes or no.\nassistant: no.'
[VLM] Answer: NO → Reward: 0.10 (dist: 1.628m)
  [VLM Step 600] Reward: 0.100 | Dist to cube: 1.628m
^C(base) kenpeter@kenpeter-ubuntu:~/work/robot$ 

