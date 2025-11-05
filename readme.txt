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
