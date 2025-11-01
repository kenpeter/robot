import torch
import torch.nn as nn
import math

class RecurrentKDA(nn.Module):
    """Simplified recurrent KDA (Eq. 1) for causal self-attention."""
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wg = nn.Linear(d_model, d_model, bias=False)  # For alpha (sigmoid -> [0,1])
        self.Wbeta = nn.Linear(d_model, num_heads, bias=False)  # Per-head beta
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, state: torch.Tensor = None):
        B, T, D = x.shape
        if state is None:
            state = torch.zeros(B, self.num_heads, self.d_k, self.d_k, device=x.device)

        x_norm = self.norm(x)
        q = self.Wq(x_norm).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.Wk(x_norm).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.Wv(x_norm).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        g = torch.sigmoid(self.Wg(x_norm)).view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # Diag(alpha)
        beta = torch.sigmoid(self.Wbeta(x_norm))[:, None, :, None, None]  # [B,1,H,1,1]

        new_states = []
        outputs = []
        for t in range(T):
            # Recurrent step (causal: only up to t)
            q_t = q[:, :, t]  # [B, H, d_k]
            k_t = k[:, :, t]
            v_t = v[:, :, t]
            alpha_t = g[:, :, t]
            S_t = state + beta * torch.einsum('b h d, b h e -> b h d e', k_t, v_t)  # Simplified update
            S_t = (torch.eye(self.d_k, device=x.device)[None, None] - beta * torch.einsum('b h d, b h e -> b h d e', k_t, k_t)) @ (alpha_t.unsqueeze(-1) * S_t)
            o_t = torch.einsum('b h d, b h d e -> b h e', q_t, S_t)
            outputs.append(o_t)
            new_states.append(S_t)
            state = S_t  # Update for next

        o = torch.stack(outputs, dim=2).transpose(1, 2).contiguous().view(B, T, D)  # [B,T,D]
        o = self.Wo(o)
        final_state = new_states[-1]  # Last state for next sequence
        return x + o, final_state  # Residual

class DiffusionKDATransformer(nn.Module):
    """KDA-based diffusion policy for denoising robot actions."""
    def __init__(self, action_dim: int = 7, obs_dim: int = 10, embed_dim: int = 256, num_layers: int = 4,
                 num_heads: int = 4, diffusion_steps: int = 50, horizon: int = 8):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.diffusion_steps = diffusion_steps
        self.horizon = horizon

        # Embeddings
        self.obs_proj = nn.Linear(obs_dim, embed_dim)
        self.time_proj = nn.Sequential(
            nn.Embedding(diffusion_steps, embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.action_proj = nn.Linear(action_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(horizon, embed_dim) * 0.02)  # Positional (optional, since KDA has implicit)

        # KDA layers (3 KDA : 1 full attn, but simplify to all KDA for toy)
        self.layers = nn.ModuleList([RecurrentKDA(embed_dim, num_heads) for _ in range(num_layers)])

        # Output: Predict noise (epsilon)
        self.noise_pred = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, action_dim)
        )

    def forward(self, obs: torch.Tensor, t: torch.Tensor, noisy_actions: torch.Tensor):
        """
        Args:
            obs: [B, obs_dim] - Robot state/image embedding.
            t: [B] - Timestep in [0, T).
            noisy_actions: [B, horizon, action_dim] - Noised action sequence.
        Returns:
            predicted_noise: [B, horizon, action_dim]
        """
        B = obs.shape[0]

        # Embed
        obs_emb = self.obs_proj(obs).unsqueeze(1).expand(-1, self.horizon, -1)  # Broadcast [B, horizon, embed]
        t_emb = self.time_proj(t).unsqueeze(1).expand(-1, self.horizon, -1)
        act_emb = self.action_proj(noisy_actions) + self.pos_embed.unsqueeze(0)  # [B, horizon, embed]

        x = act_emb + obs_emb + t_emb  # Conditioned input

        # Transformer layers
        state = None
        for layer in self.layers:
            x, state = layer(x, state)  # Recurrent, causal

        # Predict noise
        noise = self.noise_pred(x)  # [B, horizon, action_dim]
        return noise

    def sample_action(self, obs: torch.Tensor, num_steps: int = 50):
        """Inference: Denoise from noise to action sequence."""
        B = obs.shape[0]
        actions = torch.randn(B, self.horizon, self.action_dim)  # Start from noise
        for step in reversed(range(num_steps)):
            t = torch.full((B,), step, dtype=torch.long, device=obs.device)
            pred_noise = self.forward(obs, t, actions)
            actions = self.denoise_step(actions, pred_noise, t, step == 0)  # DDPM formula
        return actions

    def denoise_step(self, x_t, pred_noise, t, is_last):
        """DDPM denoising step (simplified)."""
        alpha_bar = torch.tensor(0.99 ** (t + 1)).to(x_t.device)  # Example scheduler
        if is_last:
            return x_t - pred_noise / math.sqrt(alpha_bar)
        else:
            noise = torch.randn_like(x_t)
            return (x_t - pred_noise / math.sqrt(alpha_bar)) / math.sqrt(alpha_bar) + noise / math.sqrt(alpha_bar)

# Example Usage
if __name__ == "__main__":
    model = DiffusionKDATransformer()
    obs = torch.randn(2, 10)  # Batch of 2 robot states
    t = torch.randint(0, 50, (2,))  # Random timesteps
    clean_actions = torch.randn(2, 8, 7)  # Clean 8-step actions
    # Noising (simplified)
    beta_t = 0.1 * t.float().unsqueeze(-1).unsqueeze(-1)  # Variance schedule
    noisy_actions = clean_actions + torch.sqrt(beta_t) * torch.randn_like(clean_actions)
    pred_noise = model(obs, t, noisy_actions)
    print(f"Predicted noise shape: {pred_noise.shape}")  # [2, 8, 7]

    # Inference example
    sampled = model.sample_action(obs)
    print(f"Sampled actions shape: {sampled.shape}")
