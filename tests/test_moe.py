"""
Unit tests for the Sparse MoE implementation.
Run with:  python -m pytest tests/ -v
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.moe import (
    DEFAULT_CONFIG,
    SwiGLUExpert,
    TopKRouter,
    SparseMoELayer,
    MixtralBlock,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def config():
    return DEFAULT_CONFIG.copy()

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def dummy_input(config, device):
    torch.manual_seed(42)
    return torch.randn(
        config["batch_size"], config["seq_len"], config["d_model"]
    ).to(device)


# ---------------------------------------------------------------------------
# SwiGLUExpert
# ---------------------------------------------------------------------------
class TestSwiGLUExpert:
    def test_output_shape(self, config, device):
        expert = SwiGLUExpert(config["d_model"], config["expert_hidden_dim"]).to(device)
        x = torch.randn(2, 16, config["d_model"]).to(device)
        out = expert(x)
        assert out.shape == x.shape, "Expert must preserve (B, T, D) shape."

    def test_not_identity(self, config, device):
        expert = SwiGLUExpert(config["d_model"], config["expert_hidden_dim"]).to(device)
        x = torch.randn(2, 16, config["d_model"]).to(device)
        assert not torch.allclose(expert(x), x), "Expert should transform the input."


# ---------------------------------------------------------------------------
# TopKRouter
# ---------------------------------------------------------------------------
class TestTopKRouter:
    def test_output_shapes(self, config, device, dummy_input):
        router = TopKRouter(
            config["d_model"], config["num_experts"], config["num_experts_per_token"]
        ).to(device)
        weights, indices = router(dummy_input)

        B, T = config["batch_size"], config["seq_len"]
        K    = config["num_experts_per_token"]
        assert weights.shape == (B, T, K), f"Expected ({B},{T},{K}), got {weights.shape}"
        assert indices.shape == (B, T, K), f"Expected ({B},{T},{K}), got {indices.shape}"

    def test_weights_sum_to_one(self, config, device, dummy_input):
        router = TopKRouter(
            config["d_model"], config["num_experts"], config["num_experts_per_token"]
        ).to(device)
        weights, _ = router(dummy_input)
        sums = weights.sum(dim=-1)
        expected = torch.ones(config["batch_size"], config["seq_len"]).to(device)
        assert torch.allclose(sums, expected, atol=1e-5), \
            "Routing weights must sum to 1 over the top-k dimension."

    def test_indices_in_range(self, config, device, dummy_input):
        router = TopKRouter(
            config["d_model"], config["num_experts"], config["num_experts_per_token"]
        ).to(device)
        _, indices = router(dummy_input)
        assert indices.min() >= 0
        assert indices.max() < config["num_experts"]


# ---------------------------------------------------------------------------
# SparseMoELayer
# ---------------------------------------------------------------------------
class TestSparseMoELayer:
    def test_output_shape(self, config, device, dummy_input):
        moe = SparseMoELayer(config).to(device)
        out = moe(dummy_input)
        assert out.shape == dummy_input.shape, "MoE layer must preserve input shape."

    def test_not_identity(self, config, device, dummy_input):
        moe = SparseMoELayer(config).to(device)
        out = moe(dummy_input)
        assert not torch.allclose(out, dummy_input), \
            "MoE layer should not be an identity function."

    def test_gradients_flow_to_router(self, config, device, dummy_input):
        """Critical: router gate must receive non-zero gradients after backward."""
        moe = SparseMoELayer(config).to(device)
        out = moe(dummy_input)
        loss = out.sum()
        loss.backward()
        grad = moe.router.gate.weight.grad
        assert grad is not None, "Router gate gradient is None — graph is broken."
        assert grad.norm().item() > 0.0, "Router gate gradient norm is zero."


# ---------------------------------------------------------------------------
# MixtralBlock
# ---------------------------------------------------------------------------
class TestMixtralBlock:
    def test_output_shape(self, config, device, dummy_input):
        block = MixtralBlock(config).to(device)
        out = block(dummy_input)
        assert out.shape == dummy_input.shape, "Block must preserve (B, T, D) shape."

    def test_training_loop_loss_decreases(self, config, device):
        """5-step toy training loop: verify loss falls and gradients are non-zero."""
        torch.manual_seed(42)
        block = MixtralBlock(config).to(device)
        optimizer = optim.AdamW(block.parameters(), lr=3e-4)
        loss_fn = nn.MSELoss()

        x      = torch.randn(config["batch_size"], config["seq_len"], config["d_model"]).to(device)
        target = torch.roll(x, shifts=1, dims=1)

        block.train()
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            out  = block(x)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should be lower at step 5 than step 1
        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses}"

        # Router must have received a gradient
        assert block.ffn.router.gate.weight.grad is not None
        assert block.ffn.router.gate.weight.grad.norm().item() > 0.0
