"""
RWKV-7 TTT Phantom: Combining RWKV-7 with Test-Time Training and GhostRNN efficiency
Author: RWKV-TTT-Phantom Team
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


@dataclass
class RWKV7PhantomConfig:
    """Configuration for RWKV-7 TTT Phantom model"""

    # Model dimensions
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    head_size: int = 64
    vocab_size: int = 65536

    # GhostRNN decomposition
    ghost_ratio: float = 0.8  # Proportion of dimensions in ghost state
    core_ratio: float = 0.2  # Proportion in intrinsic core

    # TTT (Test-Time Training) parameters
    ttt_enabled: bool = True
    ttt_lr: float = 0.01  # Learning rate for test-time updates
    ttt_steps: int = 1  # Number of gradient steps per token
    ttt_loss_weight: float = 0.1  # Weight for TTT loss

    # Recurrent depth parameters
    max_depth_iter: int = 1  # Maximum depth iterations (1 = standard)
    adaptive_depth: bool = False  # Whether to use adaptive depth

    # Compression options (RWKV-lite style)
    use_svd_compression: bool = False
    svd_rank_ratio: float = 0.5
    use_sparsity: bool = False
    sparsity_ratio: float = 0.3

    # Training parameters
    dropout: float = 0.0
    layer_norm_eps: float = 64e-5

    # Hardware optimization
    use_cuda_kernel: bool = True  # Use optimized CUDA kernels when available


class GhostLinear(nn.Module):
    """Ghost Linear layer that decomposes computation into core and ghost parts"""

    def __init__(self, in_features: int, out_features: int, config: RWKV7PhantomConfig):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Calculate core and ghost dimensions
        self.core_out = int(out_features * config.core_ratio)
        self.ghost_out = out_features - self.core_out

        # Core transformation (gets full updates)
        self.core_weight = nn.Parameter(torch.empty(self.core_out, in_features))

        # Ghost transformation (cheap linear projection from core)
        self.ghost_proj = nn.Parameter(torch.empty(self.ghost_out, self.core_out))

        # Optional bias
        self.bias = nn.Parameter(torch.zeros(out_features))

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.core_weight, gain=1.0)
        nn.init.xavier_uniform_(self.ghost_proj, gain=0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute core features
        core_out = F.linear(x, self.core_weight)

        # Generate ghost features via cheap projection
        ghost_out = F.linear(core_out, self.ghost_proj)

        # Combine core and ghost
        output = torch.cat([core_out, ghost_out], dim=-1)
        output = output + self.bias

        return output, core_out


class TTTLayer(nn.Module):
    """Test-Time Training layer that updates hidden state via gradient descent"""

    def __init__(self, hidden_size: int, config: RWKV7PhantomConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config

        # Small prediction head for TTT loss
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size),
        )

        # Learnable step size for gradient updates
        self.step_size = nn.Parameter(torch.tensor(config.ttt_lr))

        # Stability regularizer
        self.stability_lambda = 0.01

    def compute_ttt_loss(
        self, state: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss for TTT"""
        pred = self.pred_head(state)
        loss = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        return loss

    def forward(self, state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Perform TTT update on state"""
        if not self.config.ttt_enabled:
            return state

        # Save original state for stability regularization
        orig_state = state.detach()

        for _ in range(self.config.ttt_steps):
            # Enable gradients for state
            state = state.detach().requires_grad_(True)

            # Compute TTT loss
            loss = self.compute_ttt_loss(state, target)

            # Add stability regularization
            if self.stability_lambda > 0:
                stability_loss = (
                    self.stability_lambda * (state - orig_state).pow(2).mean()
                )
                loss = loss + stability_loss

            # Compute gradients
            grad = torch.autograd.grad(loss.mean(), state, retain_graph=True)[0]

            # Update state via gradient descent
            with torch.no_grad():
                state = state - self.step_size * grad

        return state


class RWKV7TimeMixing(nn.Module):
    """RWKV-7 Time Mixing with GhostRNN decomposition and TTT"""

    def __init__(self, layer_id: int, config: RWKV7PhantomConfig):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_size = config.head_size

        # Calculate core dimensions for GhostRNN
        self.core_dim = int(self.n_embd * config.core_ratio)
        self.ghost_dim = self.n_embd - self.core_dim

        # Token mixing parameters (lerp factors)
        self.x_r = nn.Parameter(torch.empty(1, 1, self.n_embd))
        self.x_w = nn.Parameter(torch.empty(1, 1, self.n_embd))
        self.x_k = nn.Parameter(torch.empty(1, 1, self.n_embd))
        self.x_v = nn.Parameter(torch.empty(1, 1, self.n_embd))
        self.x_a = nn.Parameter(torch.empty(1, 1, self.n_embd))
        self.x_g = nn.Parameter(torch.empty(1, 1, self.n_embd))

        # Low-rank MLPs for parameter generation
        TIME_MIX_EXTRA_DIM = 32
        self.w0 = nn.Parameter(torch.empty(1, 1, self.n_embd))
        self.w1 = nn.Parameter(torch.empty(self.n_embd, TIME_MIX_EXTRA_DIM))
        self.w2 = nn.Parameter(torch.empty(TIME_MIX_EXTRA_DIM, self.n_embd))

        self.a0 = nn.Parameter(torch.empty(1, 1, self.n_embd))
        self.a1 = nn.Parameter(torch.empty(self.n_embd, TIME_MIX_EXTRA_DIM))
        self.a2 = nn.Parameter(torch.empty(TIME_MIX_EXTRA_DIM, self.n_embd))

        # Value update parameters (for layers > 0)
        if layer_id > 0:
            self.v0 = nn.Parameter(torch.empty(1, 1, self.n_embd))
            self.v1 = nn.Parameter(torch.empty(self.n_embd, TIME_MIX_EXTRA_DIM))
            self.v2 = nn.Parameter(torch.empty(TIME_MIX_EXTRA_DIM, self.n_embd))

        # Gate parameters
        self.g1 = nn.Parameter(torch.empty(self.n_embd, TIME_MIX_EXTRA_DIM * 4))
        self.g2 = nn.Parameter(torch.empty(TIME_MIX_EXTRA_DIM * 4, self.n_embd))

        # Key scaling parameters
        self.k_k = nn.Parameter(torch.empty(1, 1, self.n_embd))
        self.k_a = nn.Parameter(torch.empty(1, 1, self.n_embd))
        self.r_k = nn.Parameter(torch.empty(self.n_head, self.head_size))

        # GhostRNN decomposed weight matrices
        self.receptance = GhostLinear(self.n_embd, self.n_embd, config)
        self.key = GhostLinear(self.n_embd, self.n_embd, config)
        self.value = GhostLinear(self.n_embd, self.n_embd, config)
        self.output = GhostLinear(self.n_embd, self.n_embd, config)

        # Group normalization for output
        self.ln_x = nn.GroupNorm(self.n_head, self.n_embd, eps=config.layer_norm_eps)

        # TTT module for core state
        if config.ttt_enabled:
            self.ttt = TTTLayer(self.core_dim, config)

        self._init_weights()

    def _init_weights(self):
        # Initialize mixing parameters
        nn.init.xavier_uniform_(self.x_r, gain=0.1)
        nn.init.xavier_uniform_(self.x_w, gain=0.1)
        nn.init.xavier_uniform_(self.x_k, gain=0.1)
        nn.init.xavier_uniform_(self.x_v, gain=0.1)
        nn.init.xavier_uniform_(self.x_a, gain=0.1)
        nn.init.xavier_uniform_(self.x_g, gain=0.1)

        # Initialize MLP parameters
        nn.init.zeros_(self.w0)
        nn.init.uniform_(self.w1, -0.01, 0.01)
        nn.init.zeros_(self.w2)
        nn.init.zeros_(self.a0)
        nn.init.uniform_(self.a1, -0.01, 0.01)
        nn.init.zeros_(self.a2)

        # Initialize other parameters
        nn.init.ones_(self.k_k)
        nn.init.ones_(self.k_a)
        nn.init.zeros_(self.r_k)

        nn.init.zeros_(self.g1)
        nn.init.orthogonal_(self.g2, gain=0.5)

    def forward(
        self,
        x: torch.Tensor,
        x_prev: torch.Tensor,
        state: torch.Tensor,
        v_first: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        H, N = self.n_head, self.head_size

        # Token shift mixing
        xx = x_prev - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        # Compute receptance, key, value with GhostRNN decomposition
        r, r_core = self.receptance(xr)
        k, k_core = self.key(xk)
        v, v_core = self.value(xv)

        # Compute weights and activations
        w = torch.tanh(xw @ self.w1) @ self.w2
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        # Normalize keys
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        # Value update for layers > 0
        if self.layer_id == 0:
            v_first = v
        else:
            v_offset = torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
            v = v + (v_first - v) * v_offset

        # Compute decay
        w = torch.exp(-0.606531 * torch.sigmoid(self.w0 + w))

        # Reshape for matrix operations
        r = r.view(B, T, H, N)
        k = k.view(B, T, H, N)
        v = v.view(B, T, H, N)
        kk = kk.view(B, T, H, N)
        a = a.view(B, T, H, N)
        w = w.view(B, T, H, N)

        # Process state updates (simplified for clarity - full CUDA kernel
        # would be used in production)
        output = []
        for t in range(T):
            # Extract core state for TTT
            if hasattr(self, "ttt") and self.config.ttt_enabled:
                # Avoid dimension mismatch if core_dim exceeds state size
                core_dim = min(self.core_dim, state.size(-1))
                if core_dim > 0:
                    core_state = state[:, :, :core_dim, :core_dim].clone()
                    core_state_flat = core_state.view(B, -1)
                    target = v_core[:, t, :core_dim].detach()
                    core_state_flat = self.ttt(core_state_flat, target.view(B, -1))
                    state[:, :, :core_dim, :core_dim] = core_state_flat.view(
                        B, H, core_dim, core_dim
                    )

            # RWKV-7 state update with generalized delta rule
            vk = v[:, t].unsqueeze(-1) @ k[:, t].unsqueeze(-2)
            ab = (-kk[:, t]).unsqueeze(-1) @ (kk[:, t] * a[:, t]).unsqueeze(-2)
            state = state * w[:, t].unsqueeze(-1) + state @ ab + vk

            # Compute output
            y = (state @ r[:, t].unsqueeze(-1)).squeeze(-1)
            output.append(y)

        output = torch.stack(output, dim=1)
        output = output.view(B, T, C)

        # Apply group norm and final processing
        output = self.ln_x(output)
        output = output + (r * k * self.r_k.unsqueeze(0).unsqueeze(0)).view(
            B, T, H, N
        ).sum(dim=-1, keepdim=True).expand(-1, -1, -1, N).contiguous().view(
            B, T, C
        ) * v.view(
            B, T, C
        )

        # Output projection with ghost decomposition
        output, _ = self.output(output * g)

        return output, x[:, -1], state, v_first


class RWKV7ChannelMixing(nn.Module):
    """RWKV-7 Channel Mixing (FFN) with ReLU² and GhostRNN"""

    def __init__(self, config: RWKV7PhantomConfig):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd

        # Token mixing for channel mix
        self.x_k = nn.Parameter(torch.empty(1, 1, self.n_embd))

        # GhostRNN decomposed FFN
        self.key = GhostLinear(self.n_embd, self.n_embd * 4, config)
        self.value = GhostLinear(self.n_embd * 4, self.n_embd, config)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.x_k, gain=0.1)

    def forward(
        self, x: torch.Tensor, x_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Token shift
        xx = x_prev - x
        xk = x + xx * self.x_k

        # ReLU² activation with ghost decomposition
        k, _ = self.key(xk)
        k = torch.relu(k) ** 2
        output, _ = self.value(k)

        return output, x


class RWKV7Block(nn.Module):
    """Single RWKV-7 block with optional depth iteration"""

    def __init__(self, layer_id: int, config: RWKV7PhantomConfig):
        super().__init__()
        self.layer_id = layer_id
        self.config = config

        self.ln1 = LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.ln2 = LayerNorm(config.n_embd, eps=config.layer_norm_eps)

        self.time_mixing = RWKV7TimeMixing(layer_id, config)
        self.channel_mixing = RWKV7ChannelMixing(config)

        if config.dropout > 0:
            self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_prev_tm: torch.Tensor,
        x_prev_cm: torch.Tensor,
        tm_state: torch.Tensor,
        v_first: Optional[torch.Tensor] = None,
        depth_iter: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Support for recurrent depth
        for d in range(depth_iter):
            # Time mixing
            x_norm = self.ln1(x)
            dx, x_prev_tm, tm_state, v_first = self.time_mixing(
                x_norm, x_prev_tm, tm_state, v_first
            )
            x = x + dx

            if hasattr(self, "dropout"):
                x = self.dropout(x)

            # Channel mixing
            x_norm = self.ln2(x)
            dx, x_prev_cm = self.channel_mixing(x_norm, x_prev_cm)
            x = x + dx

            if hasattr(self, "dropout"):
                x = self.dropout(x)

        return x, x_prev_tm, x_prev_cm, tm_state, v_first


class RWKV7TTTPhantom(nn.Module):
    """RWKV-7 TTT Phantom model.

    Combines RWKV-7, GhostRNN, and Test-Time Training.
    """

    def __init__(self, config: RWKV7PhantomConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)

        # Layer 0 normalization (as per RWKV-7)
        self.ln0 = LayerNorm(config.n_embd, eps=config.layer_norm_eps)

        # RWKV-7 blocks
        self.blocks = nn.ModuleList(
            [RWKV7Block(i, config) for i in range(config.n_layer)]
        )

        # Output layers
        self.ln_out = LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

        # Optional: Apply SVD compression if enabled
        if config.use_svd_compression:
            self._apply_svd_compression()

    def _init_weights(self):
        # Initialize embeddings
        nn.init.uniform_(self.emb.weight, -1e-4, 1e-4)

        # Initialize output head
        if self.config.vocab_size > self.config.n_embd:
            gain = 0.5 * math.sqrt(self.config.vocab_size / self.config.n_embd)
        else:
            gain = 0.5
        nn.init.orthogonal_(self.head.weight, gain=gain)

        # Apply layer-wise scaling to ln_x in each block
        for i, block in enumerate(self.blocks):
            layer_scale = (1 + i) / self.config.n_layer
            with torch.no_grad():
                block.time_mixing.ln_x.weight.data *= layer_scale**0.7

    def _apply_svd_compression(self):
        """Apply SVD compression to weight matrices"""
        # This would compress large matrices using SVD
        # Implementation depends on specific requirements
        pass

    def forward(
        self,
        idx: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None,
        return_state: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        B, T = idx.shape

        # Initialize state if not provided
        if state is None:
            state = self.init_state(B)

        # Token embeddings
        x = self.emb(idx)
        x = self.ln0(x)

        # Process through blocks
        new_state = []
        v_first = None

        for i, block in enumerate(self.blocks):
            # Extract states for this layer
            x_prev_tm = state[i * 3]
            x_prev_cm = state[i * 3 + 1]
            tm_state = state[i * 3 + 2]

            # Determine depth iterations (could be adaptive based on complexity)
            depth_iter = self.config.max_depth_iter

            # Forward through block
            x, x_prev_tm, x_prev_cm, tm_state, v_first = block(
                x, x_prev_tm, x_prev_cm, tm_state, v_first, depth_iter
            )

            # Save new states
            new_state.extend([x_prev_tm, x_prev_cm, tm_state])

        # Output projection
        x = self.ln_out(x)
        logits = self.head(x)

        if return_state:
            return logits, new_state
        return logits, None

    def init_state(self, batch_size: int) -> List[torch.Tensor]:
        """Initialize model state"""
        state = []
        for i in range(self.config.n_layer):
            # x_prev for time mixing
            state.append(
                torch.zeros(
                    batch_size,
                    1,
                    self.config.n_embd,
                    dtype=torch.float32,
                    device=self.emb.weight.device,
                )
            )
            # x_prev for channel mixing
            state.append(
                torch.zeros(
                    batch_size,
                    1,
                    self.config.n_embd,
                    dtype=torch.float32,
                    device=self.emb.weight.device,
                )
            )
            # time mixing state (matrix-valued)
            state.append(
                torch.zeros(
                    batch_size,
                    self.config.n_head,
                    self.config.head_size,
                    self.config.head_size,
                    dtype=torch.float32,
                    device=self.emb.weight.device,
                )
            )
        return state

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Generate text autoregressively"""
        state = None

        for _ in range(max_new_tokens):
            # Get logits
            logits, state = self.forward(idx[:, -1:], state)
            logits = logits[:, -1, :] / temperature

            # Apply top-k and top-p filtering
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def create_model(config: Optional[RWKV7PhantomConfig] = None) -> RWKV7TTTPhantom:
    """Create RWKV-7 TTT Phantom model with default or custom config"""
    if config is None:
        config = RWKV7PhantomConfig()
    return RWKV7TTTPhantom(config)


# Example usage and testing
if __name__ == "__main__":
    # Create a small model for testing
    config = RWKV7PhantomConfig(
        n_layer=4,
        n_embd=256,
        n_head=4,
        head_size=64,
        vocab_size=1000,
        ghost_ratio=0.75,
        ttt_enabled=True,
        max_depth_iter=2,
    )

    model = create_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    batch_size = 2
    seq_len = 10
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, state = model(x)
    print(f"Output shape: {logits.shape}")
    print(f"Number of state tensors: {len(state)}")

    # Test generation
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    generated = model.generate(prompt, max_new_tokens=10)
    print(f"Generated sequence: {generated}")
