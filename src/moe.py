"""
Sparse Mixture of Experts (SMoE) — PyTorch Implementation
==========================================================
Implements the core architecture behind Mixtral 8x7B:
  - TopKRouter       : differentiable top-k gating network
  - SparseMoELayer   : masked-matmul token routing over N experts
  - MixtralBlock     : pre-norm Attention + SparseMoE residual block

Reference: Jiang et al. (2024) "Mixtral of Experts"
           https://arxiv.org/abs/2401.04088
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "d_model": 128,               # Hidden / embedding dimension
    "n_heads": 4,                 # Attention heads
    "seq_len": 16,                # Sequence length (used in tests)
    "batch_size": 2,              # Batch size (used in tests)
    "num_experts": 8,             # Total number of expert FFNs
    "num_experts_per_token": 2,   # Top-K experts selected per token
    "expert_hidden_dim": 256,     # Inner dimension inside each SwiGLU expert
}


# ---------------------------------------------------------------------------
# SwiGLU Expert  (single "expert" network — same as a Llama 3 FFN)
# ---------------------------------------------------------------------------
class SwiGLUExpert(nn.Module):
    """
    A single SwiGLU feed-forward expert.

    Architecture:
        y = W_out( SiLU(x W_gate) ⊙ (x W_in) )

    Identical to the FFN used inside Llama 3; used here as the building
    block for each slot in the expert bank.

    Args:
        d_model     : Input / output dimension.
        hidden_dim  : Inner projection dimension.
    """

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_in   = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_out  = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.silu(self.w_gate(x)) * self.w_in(x))


# ---------------------------------------------------------------------------
# TopKRouter  (the gating / routing network)
# ---------------------------------------------------------------------------
class TopKRouter(nn.Module):
    """
    Differentiable top-K router.

    Projects each token's hidden state to a logit over N experts, selects
    the top-K experts, and re-normalises their scores via softmax so that
    the K routing weights sum to exactly 1.0.

    Key design choice — softmax is applied *after* top-k selection:
        If softmax were applied to all N logits first, the top-K weights
        would not sum to 1 because the remaining (N-K) experts absorb
        probability mass.  Applying softmax only to the K selected logits
        re-normalises over the chosen subset, matching the Mixtral paper.

    Args:
        d_model     : Input dimension.
        num_experts : Total number of experts (N).
        top_k       : Number of experts selected per token (K).
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.gate  = nn.Linear(d_model, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : Hidden states, shape (B, T, D).

        Returns:
            routing_weights  : (B, T, top_k)  — normalised expert weights.
            selected_indices : (B, T, top_k)  — expert indices 0 … N-1.
        """
        # (B, T, N)
        gate_logits = self.gate(x)

        # Select the top-K logits and their expert indices along the last dim.
        # top_k_values    : (B, T, top_k)
        # selected_indices: (B, T, top_k)
        top_k_values, selected_indices = torch.topk(
            gate_logits, self.top_k, dim=-1
        )

        # Normalise only the selected K logits → weights sum to 1 per token.
        routing_weights = F.softmax(top_k_values, dim=-1)  # (B, T, top_k)

        return routing_weights, selected_indices


# ---------------------------------------------------------------------------
# SparseMoELayer  (the core Sparse Mixture-of-Experts layer)
# ---------------------------------------------------------------------------
class SparseMoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts feed-forward layer.

    Replaces the single dense FFN in a standard Transformer block with a
    bank of N expert FFNs, where only K experts process each token.

    Routing strategy — Masked Matmul (loops over experts, not tokens):
        For expert i in [0, N):
          1. Build a boolean mask of shape (B, T): which tokens chose i?
          2. Extract the scalar routing weight for expert i per token.
          3. Zero out tokens not routed to i, then run all (B, T, D)
             through the expert in one batched call.
          4. Scale by routing weight and accumulate into the output.

        This avoids an inner Python loop over B×T tokens, keeping the
        loop bounded by N (typically 8), and — crucially — preserves the
        full differentiable computation graph so gradients reach the
        router's gate weights.

    Args:
        config : Dict with keys d_model, num_experts,
                 num_experts_per_token, expert_hidden_dim.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.num_experts = config["num_experts"]

        self.router = TopKRouter(
            config["d_model"],
            self.num_experts,
            config["num_experts_per_token"],
        )

        # ModuleList ensures PyTorch registers all expert parameters.
        self.experts = nn.ModuleList([
            SwiGLUExpert(config["d_model"], config["expert_hidden_dim"])
            for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Hidden states, shape (B, T, D).

        Returns:
            Tensor of shape (B, T, D) — weighted sum of active expert outputs.
        """
        B, T, D = x.shape

        routing_weights, selected_indices = self.router(x)
        # routing_weights : (B, T, top_k)
        # selected_indices: (B, T, top_k)

        final_output = torch.zeros_like(x)

        for i, expert_layer in enumerate(self.experts):
            # ── Step 1: Boolean mask ────────────────────────────────────────
            # True at [b, t] if expert i appears anywhere in token (b,t)'s
            # top-K selection.
            # (B, T, top_k) → .any(dim=-1) → (B, T)  bool
            mask = (selected_indices == i).any(dim=-1)

            # ── Step 2: Scalar routing weight ───────────────────────────────
            # Isolate the weight for expert i; 0.0 where i was not selected.
            # (B, T, top_k) * (B, T, top_k) → .sum(dim=-1) → (B, T)  float
            expert_weight = (
                routing_weights * (selected_indices == i).float()
            ).sum(dim=-1)

            # ── Step 3: Mask the input ──────────────────────────────────────
            # Zero out tokens not routed to expert i.
            # mask.unsqueeze(-1): (B, T, 1) broadcasts over D
            masked_x = x * mask.unsqueeze(-1)  # (B, T, D)

            # ── Step 4: Expert forward pass ─────────────────────────────────
            expert_out = expert_layer(masked_x)  # (B, T, D)

            # ── Step 5: Weighted accumulation ──────────────────────────────
            # expert_weight.unsqueeze(-1): (B, T, 1) broadcasts over D
            final_output = final_output + expert_out * expert_weight.unsqueeze(-1)

        return final_output


# ---------------------------------------------------------------------------
# MixtralBlock  (full Attention + SparseMoE transformer block)
# ---------------------------------------------------------------------------
class MixtralBlock(nn.Module):
    """
    Pre-norm Transformer block with a Sparse MoE FFN.

    Architecture (mirrors Mixtral / Llama 3 block design):
        x ← x + Attention( LayerNorm(x) )
        x ← x + SparseMoE( LayerNorm(x) )

    The only difference from a vanilla Transformer block is substituting
    SparseMoELayer for the single dense SwiGLU FFN.

    Args:
        config : Same config dict used by SparseMoELayer.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(config["d_model"])
        self.attn = nn.MultiheadAttention(
            config["d_model"], config["n_heads"], batch_first=True
        )
        self.ln2  = nn.LayerNorm(config["d_model"])
        self.ffn  = SparseMoELayer(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual — Attention
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Pre-norm residual — Sparse MoE FFN
        x = x + self.ffn(self.ln2(x))
        return x
