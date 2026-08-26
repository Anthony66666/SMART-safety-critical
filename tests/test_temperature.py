"""Sampling temperature as a continuous pressure knob.

A Pareto frontier needs realism to respond smoothly to a scalar, not just to
separate a few categories. Temperature is the cheapest such scalar: it
reshapes the sampling distribution without touching the model, so log p stays
the model's own verdict on what was sampled.
"""
import pytest
import torch

from tests.test_inference_likelihood import _load


def test_temperature_one_is_the_untouched_sampler():
    """Guards the default path: adding the knob must not move existing results."""
    model, data = _load()
    torch.manual_seed(0)
    with torch.no_grad():
        plain = model.inference(data)

    _, fresh = _load()
    torch.manual_seed(0)
    with torch.no_grad():
        explicit = model.inference(fresh, temperature=1.0)

    assert torch.equal(plain['next_token_idx'], explicit['next_token_idx'])


def test_hotter_sampling_produces_less_likely_scenarios():
    """The knob has to move realism in the expected direction, or it is no
    use as a stand-in for adversarial pressure."""
    model, data = _load()
    torch.manual_seed(0)
    with torch.no_grad():
        cold = model.inference(data, temperature=0.5)

    _, fresh = _load()
    torch.manual_seed(0)
    with torch.no_grad():
        hot = model.inference(fresh, temperature=3.0)

    assert hot['log_p'].mean().item() < cold['log_p'].mean().item()


def test_log_p_stays_the_untempered_model_verdict():
    """log p must be read off the T=1 softmax whatever the sampler did; a
    tempered log p would measure the sampler, not the model."""
    model, data = _load()
    torch.manual_seed(0)
    with torch.no_grad():
        hot = model.inference(data, temperature=3.0)

    _, fresh = _load()
    with torch.no_grad():
        rescored = model.inference(fresh, forced_tokens=hot['next_token_idx'])

    assert torch.allclose(rescored['log_p'], hot['log_p'], atol=1e-5)
