"""Tests for danger-tilted sampling weights.

The tilted sampler draws from q proportional to p * exp(J / beta). p stays the
model's own probability -- tilting reshapes only what gets drawn, never the
likelihood the sample is scored with -- so realism accounting is unaffected and
importance weights w = p/q remain exact.
"""
import math

import pytest
import torch

from smart.safety.tilting import tilt_weights


def test_large_beta_leaves_weights_proportionally_unchanged():
    w = torch.tensor([[0.5, 0.3, 0.2]])
    danger = torch.tensor([[1.0, -2.0, 0.5]])

    tilted = tilt_weights(w, danger, beta=1e9)

    # normalise both and compare
    a = tilted / tilted.sum(-1, keepdim=True)
    b = w / w.sum(-1, keepdim=True)
    assert torch.allclose(a, b, atol=1e-4)


def test_tilting_upweights_the_more_dangerous_candidate():
    w = torch.tensor([[0.5, 0.5]])
    danger = torch.tensor([[0.0, 2.0]])           # second is more dangerous

    tilted = tilt_weights(w, danger, beta=1.0)

    assert tilted[0, 1] > tilted[0, 0]


def test_small_beta_concentrates_on_maximum_danger():
    w = torch.tensor([[0.5, 0.4, 0.1]])
    danger = torch.tensor([[0.0, 1.0, 3.0]])      # third is most dangerous

    tilted = tilt_weights(w, danger, beta=0.05)
    probs = tilted / tilted.sum(-1, keepdim=True)

    assert probs[0, 2] > 0.99


def test_zero_probability_candidates_stay_zero():
    """Tilting must not resurrect tokens the model assigned no mass."""
    w = torch.tensor([[0.7, 0.0, 0.3]])
    danger = torch.tensor([[0.0, 5.0, 0.0]])      # dangerous but impossible

    tilted = tilt_weights(w, danger, beta=0.5)

    assert tilted[0, 1].item() == 0.0
