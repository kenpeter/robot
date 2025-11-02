Final Architecture with O(n) Complexity:
┌─────────────────────────────────────────────┐
│          ROBOT ARM PICK & PLACE             │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
    [Vision]              [Proprioception]
   84×84 RGB              9 joints + grasp flag
        │                       │
        ↓                       ↓
┌─────────────────┐     ┌──────────────┐
│ Vision Trans    │     │ State MLP    │
│ (ViT w/ Kimi)   │     │              │
├─────────────────┤     └──────────────┘
│ Patch: 14×14    │            │
│ Patches: 6×6    │            │
│ + CLS token     │            │
│                 │            │
│ RecurrentKDA ×2 │────────────┤
│ (O(n) attn)     │            │
│                 │            │
│ Output: 128-dim │            │
└─────────────────┘            │
        │                      │
        └──────────┬───────────┘
                   │
             [Combined 128-dim]
                   │
                   ↓
        ┌──────────────────────┐
        │  Diffusion Policy    │
        │  (DiT w/ Kimi)       │
        ├──────────────────────┤
        │  DiTBlock ×4         │
        │  ├─ RecurrentKDA     │
        │  │  (O(n) attn)      │
        │  ├─ Adaptive LayerN  │
        │  └─ MLP              │
        │                      │
        │  Output: 8D action   │
        │  (7 arm + 1 gripper) │
        └──────────────────────┘
                   │
                   ↓
            [Robot Action]
