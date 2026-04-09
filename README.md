# Mixtral MoE from Scratch

A clean, well-documented PyTorch implementation of **Sparse Mixture of Experts (SMoE)** — the core architecture behind [Mixtral 8x7B](https://arxiv.org/abs/2401.04088), Grok-1, and Databricks DBRX.

This project builds the architecture piece by piece, from a single SwiGLU expert up to a full pre-norm Transformer block with sparse routing, and verifies end-to-end differentiability with a gradient check.

---

## What's inside

```
mixtral-moe-from-scratch/
├── src/
│   └── moe.py          # All four components (SwiGLUExpert, TopKRouter,
│                       #   SparseMoELayer, MixtralBlock)
├── tests/
│   └── test_moe.py     # 10 pytest unit tests
├── notebooks/
│   └── walkthrough.ipynb   # Step-by-step Jupyter notebook
├── demo.py             # Standalone end-to-end demo (run this first)
└── requirements.txt
```

---

## Architecture overview

### The problem with dense FFNs

In a standard Transformer (e.g. Llama 3), every token passes through the **same** large feed-forward network. Scaling up the model means making that FFN bigger — which increases compute cost quadratically.

### The MoE solution

Instead of one massive FFN, the model holds **N smaller "expert" FFNs**. For every token, a **router** dynamically selects only the top-K experts to activate. Parameters scale with N, but compute scales with K — they are decoupled.

```
Token hidden state  x  (shape: D)
        │
        ▼
   ┌─────────┐
   │  Router  │  Linear(D → N)  →  top-K selection  →  Softmax over K
   └─────────┘
        │   indices + weights
        ▼
 ┌──────────────────────────────────┐
 │  Expert 0  │ Expert 1 │ ... │ Expert N-1  │
 └──────────────────────────────────┘
        │   (only K experts active per token)
        ▼
   weighted sum  →  output  (shape: D)
```

### The routing formula

For token hidden state **x**:

```
gate_logits = x · Wg                          # (N,)
top_k_values, top_k_idx = topk(gate_logits)   # (K,)
weights = Softmax(top_k_values)               # (K,)  — sum to 1

output = Σ  weights[i] · Expert_i(x)
          i ∈ top_k_idx
```

Key insight: Softmax is applied **after** top-k selection so the K weights sum to exactly 1.0, not ~K/N.

---

## Components

### `SwiGLUExpert`
Single expert FFN using the SwiGLU activation (same as Llama 3):
```
y = W_out( SiLU(x·W_gate) ⊙ (x·W_in) )
```

### `TopKRouter`
Linear projection → top-K selection → re-normalised softmax. Returns `routing_weights (B, T, K)` and `selected_indices (B, T, K)`.

### `SparseMoELayer`
The core challenge. Uses a **Masked Matmul** approach that loops over N experts (not B×T tokens):

| Step | Operation | Shape |
|------|-----------|-------|
| Boolean mask | `(selected_indices == i).any(dim=-1)` | `(B, T)` |
| Routing weight | `(routing_weights * (selected_indices==i).float()).sum(dim=-1)` | `(B, T)` |
| Mask input | `x * mask.unsqueeze(-1)` | `(B, T, D)` |
| Expert forward | `expert_layer(masked_x)` | `(B, T, D)` |
| Accumulate | `final_output += expert_out * expert_weight.unsqueeze(-1)` | `(B, T, D)` |

The boolean mask zeros out non-selected tokens — no integer indexing — so the **computational graph stays intact** and gradients flow back to the router's gate weights.

### `MixtralBlock`
Standard pre-norm residual block with SparseMoE replacing the dense FFN:
```
x ← x + Attention( LayerNorm(x) )
x ← x + SparseMoE( LayerNorm(x) )
```

---

## Quickstart

```bash
git clone https://github.com/<your-username>/mixtral-moe-from-scratch
cd mixtral-moe-from-scratch
pip install -r requirements.txt

# End-to-end demo with gradient verification
python demo.py

# Unit tests
python -m pytest tests/ -v
```

### Expected demo output

```
Device: cpu

=======================================================
Part A — TopKRouter
=======================================================
  routing_weights  shape : (2, 16, 2)  (expected B,T,K)
  selected_indices shape : (2, 16, 2)  (expected B,T,K)
  ✓ Weights sum to 1.0  — Router test passed!

=======================================================
Part B — SparseMoELayer
=======================================================
  Output shape : (2, 16, 128)
  ✓ Shape correct, not identity — MoE layer test passed!

=======================================================
Part C — MixtralBlock (Attention + SparseMoE)
=======================================================
  Output shape : (2, 16, 128)
  ✓ Forward pass OK — MixtralBlock test passed!

=======================================================
Part D — Toy Training Loop & Gradient Verification
=======================================================
  Initial router gate weight norm : 1.6309

  Step 1: Loss = 2.089525 | Router Grad Norm = 0.031791
  Step 2: Loss = 2.059690 | Router Grad Norm = 0.030281
  Step 3: Loss = 2.029687 | Router Grad Norm = 0.028661
  Step 4: Loss = 2.000725 | Router Grad Norm = 0.030021
  Step 5: Loss = 1.971650 | Router Grad Norm = 0.032049

  ✓ SUCCESS: Loss decreased and gradients flow through the router!
```

The non-zero `Router Grad Norm` at every step is the key result — it proves the masked routing is fully differentiable.

---

## Configuration

```python
from src.moe import DEFAULT_CONFIG, MixtralBlock

config = {
    "d_model": 128,
    "n_heads": 4,
    "seq_len": 16,
    "batch_size": 2,
    "num_experts": 8,             # N — total expert count
    "num_experts_per_token": 2,   # K — active experts per token
    "expert_hidden_dim": 256,
}

block = MixtralBlock(config)
```

---

## Why is this hard to implement?

The tricky part is **routing tokens to different experts inside a single batch** without:

1. Breaking PyTorch's autograd graph (no `.detach()`, `.numpy()`, or integer-index scattering)
2. Looping over every token individually (too slow for large B×T)

The Masked Matmul approach handles both: it loops over N experts (small, fixed), uses floating-point masking to zero out non-selected tokens, and never touches the graph with non-differentiable operations.

---

## References

| Paper | Notes |
|-------|-------|
| [Mixtral of Experts (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088) | This implementation follows the routing formulation in Section 2 |
| [Outrageously Large Neural Networks (Shazeer et al., 2017)](https://arxiv.org/abs/1701.06538) | Original Noisy Top-K Gating paper |
| [Switch Transformers (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961) | Top-1 routing + Load Balancing Loss |
| [Hugging Face MoE Blog](https://huggingface.co/blog/moe) | Accessible visual overview |

---

## License

MIT
