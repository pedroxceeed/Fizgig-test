"""Focused regression tests for the DoRA Linear adapter."""
import unittest

import torch
import torch.nn.functional as F

from fizgig.families.lora import DoRAAdapter


class DoRAAdapterTests(unittest.TestCase):
    def test_zero_up_projection_starts_as_identity(self):
        torch.manual_seed(7)
        base = torch.nn.Linear(12, 9, bias=False)
        adapter = DoRAAdapter(12, 9, 4, base.weight, trainable=True)
        x = torch.randn(3, 5, 12)
        delta = adapter.forward_delta(x, base.weight, 0.25)
        self.assertTrue(torch.allclose(delta, torch.zeros_like(delta), atol=1e-6))

    def test_forward_matches_weight_decomposition(self):
        torch.manual_seed(11)
        base = torch.nn.Linear(12, 9, bias=False)
        adapter = DoRAAdapter(12, 9, 4, base.weight, trainable=True)
        with torch.no_grad():
            adapter.up.weight.normal_(std=0.03)
            adapter.magnitude.mul_(1.07)
        x = torch.randn(2, 6, 12)
        strength = 0.65
        actual = F.linear(x, base.weight) + adapter.forward_delta(x, base.weight, strength)
        adapted = base.weight.detach().float() + strength * (
            adapter.up.weight.detach().float() @ adapter.down.weight.detach().float()
        )
        norm = adapted.norm(dim=1).clamp_min(1e-12)
        expected_weight = adapter.magnitude.detach().float()[:, None] * adapted / norm[:, None]
        expected = F.linear(x, expected_weight.to(x.dtype))
        self.assertTrue(torch.allclose(actual, expected, atol=2e-5, rtol=2e-5))

    def test_gradients_reach_low_rank_and_magnitude(self):
        torch.manual_seed(19)
        base = torch.nn.Linear(12, 9, bias=False)
        adapter = DoRAAdapter(12, 9, 4, base.weight, trainable=True)
        with torch.no_grad():
            adapter.up.weight.normal_(std=0.02)
        x = torch.randn(2, 12)
        adapter.forward_delta(x, base.weight).square().mean().backward()
        self.assertIsNotNone(adapter.down.weight.grad)
        self.assertIsNotNone(adapter.up.weight.grad)
        self.assertIsNotNone(adapter.magnitude.grad)
        self.assertGreater(adapter.magnitude.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
