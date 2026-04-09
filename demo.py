"""
demo.py — Run the full Sparse MoE pipeline end-to-end.

Usage:
    python demo.py

Prints per-step loss and router gradient norms to confirm the routing
mechanism is fully differentiable.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.moe import DEFAULT_CONFIG, TopKRouter, SparseMoELayer, MixtralBlock

# ── Reproducibility ────────────────────────────────────────────────────────
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

config = DEFAULT_CONFIG.copy()

# ── Part A: Router smoke-test ──────────────────────────────────────────────
print("=" * 55)
print("Part A — TopKRouter")
print("=" * 55)
dummy = torch.randn(config["batch_size"], config["seq_len"], config["d_model"]).to(device)
router = TopKRouter(config["d_model"], config["num_experts"], config["num_experts_per_token"]).to(device)
weights, indices = router(dummy)

print(f"  routing_weights  shape : {tuple(weights.shape)}  (expected B,T,K)")
print(f"  selected_indices shape : {tuple(indices.shape)}  (expected B,T,K)")
assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones(config["batch_size"], config["seq_len"]).to(device),
    atol=1e-5,
)
print("  ✓ Weights sum to 1.0  — Router test passed!\n")

# ── Part B: SparseMoELayer smoke-test ─────────────────────────────────────
print("=" * 55)
print("Part B — SparseMoELayer")
print("=" * 55)
moe = SparseMoELayer(config).to(device)
moe_out = moe(dummy)
assert moe_out.shape == dummy.shape
assert not torch.allclose(moe_out, dummy)
print(f"  Output shape : {tuple(moe_out.shape)}")
print("  ✓ Shape correct, not identity — MoE layer test passed!\n")

# ── Part C: MixtralBlock smoke-test ───────────────────────────────────────
print("=" * 55)
print("Part C — MixtralBlock (Attention + SparseMoE)")
print("=" * 55)
block = MixtralBlock(config).to(device)
block_out = block(dummy)
assert block_out.shape == dummy.shape
print(f"  Output shape : {tuple(block_out.shape)}")
print("  ✓ Forward pass OK — MixtralBlock test passed!\n")

# ── Part D: Toy training loop & gradient verification ─────────────────────
print("=" * 55)
print("Part D — Toy Training Loop & Gradient Verification")
print("=" * 55)

block      = MixtralBlock(config).to(device)
optimizer  = optim.AdamW(block.parameters(), lr=3e-4)
loss_fn    = nn.MSELoss()

x      = torch.randn(config["batch_size"], config["seq_len"], config["d_model"]).to(device)
target = torch.roll(x, shifts=1, dims=1)

block.train()
init_norm = block.ffn.router.gate.weight.norm().item()
print(f"  Initial router gate weight norm : {init_norm:.4f}\n")

losses = []
for step in range(1, 6):
    optimizer.zero_grad()
    output = block(x)
    loss   = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    grad_norm = block.ffn.router.gate.weight.grad.norm().item()
    losses.append(loss.item())
    print(f"  Step {step}: Loss = {loss.item():.6f} | Router Grad Norm = {grad_norm:.6f}")

print(f"\n  Final router gate weight norm   : {block.ffn.router.gate.weight.norm():.4f}")
assert losses[-1] < losses[0], "Loss did not decrease!"
assert block.ffn.router.gate.weight.grad is not None
assert block.ffn.router.gate.weight.grad.norm().item() > 0.0

print("\n  ✓ SUCCESS: Loss decreased and gradients flow through the router!")
