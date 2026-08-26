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


def test_restrict_to_topk_keeps_only_the_k_largest():
    from smart.safety.tilting import restrict_to_topk
    w = torch.tensor([[0.1, 0.5, 0.2, 0.05, 0.15]])

    kept = restrict_to_topk(w, k=2)

    assert kept[0, 1].item() == pytest.approx(0.5)   # largest kept
    assert kept[0, 2].item() == pytest.approx(0.2)   # 2nd kept
    assert kept[0, 0].item() == 0.0                  # rest zeroed
    assert kept[0, 3].item() == 0.0
    assert kept[0, 4].item() == 0.0


def test_restrict_to_topk_is_a_noop_when_k_exceeds_width():
    from smart.safety.tilting import restrict_to_topk
    w = torch.tensor([[0.3, 0.7]])
    assert torch.equal(restrict_to_topk(w, k=10), w)


def test_tilt_survives_when_the_max_danger_candidate_is_masked_out():
    """restrict_to_topk can zero the single most dangerous candidate. The tilt
    must then normalise against the surviving candidates, not the masked max,
    or every survivor underflows to zero and multinomial has nothing to draw."""
    # float32 with a realistic danger gap, as in the rollout: exp underflows
    # to exactly zero for every survivor if the masked max is the reference.
    w = torch.tensor([[0.0, 0.5, 0.5]], dtype=torch.float32)
    danger = torch.tensor([[200.0, 1.0, 0.0]], dtype=torch.float32)

    out = tilt_weights(w, danger, beta=0.1)

    assert torch.isfinite(out).all()
    assert out.sum().item() > 0
    assert out[0, 0].item() == 0.0            # masked candidate stays zero
