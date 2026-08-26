"""Tests for exact sequence likelihood accounting during rollout.

The model distribution p is the full 2048-way softmax; the sampling
distribution q is p restricted to the top-k candidates and renormalised.
Both must be tracked separately so importance weights w = p/q are exact.
"""
import math

import pytest

import torch

from smart.safety.likelihood import SequenceLikelihood


def test_accumulates_log_p_across_steps():
    lik = SequenceLikelihood(num_agents=1)

    lik.update(p_chosen=torch.tensor([0.5]),
               q_chosen=torch.tensor([0.5]),
               valid=torch.tensor([True]))
    lik.update(p_chosen=torch.tensor([0.25]),
               q_chosen=torch.tensor([0.25]),
               valid=torch.tensor([True]))

    assert lik.log_p.item() == pytest.approx(math.log(0.5) + math.log(0.25))


def test_log_q_renormalises_by_topk_mass():
    """q is p restricted to the top-k, so log q = log p - log(top-k mass)."""
    lik = SequenceLikelihood(num_agents=1)

    lik.update(p_chosen=torch.tensor([0.5]),
               q_chosen=torch.tensor([0.5 / 0.8]),
               valid=torch.tensor([True]))

    assert lik.log_q.item() == pytest.approx(math.log(0.5) - math.log(0.8))


def test_invalid_agents_contribute_nothing():
    """Agents masked out -- notably an externally controlled ego -- must not
    enter the likelihood, or every downstream importance weight is wrong."""
    lik = SequenceLikelihood(num_agents=2)

    lik.update(p_chosen=torch.tensor([0.5, 0.5]),
               q_chosen=torch.tensor([0.625, 0.625]),
               valid=torch.tensor([True, False]))

    assert lik.log_p[1].item() == 0.0
    assert lik.log_q[1].item() == 0.0


def test_full_support_makes_q_identical_to_p():
    """With no truncation the top-k mass is 1, so q collapses onto p."""
    lik = SequenceLikelihood(num_agents=1)

    lik.update(p_chosen=torch.tensor([0.3]),
               q_chosen=torch.tensor([0.3]),
               valid=torch.tensor([True]))

    assert lik.log_q.item() == pytest.approx(lik.log_p.item())


def test_truncation_makes_q_larger_than_p():
    """Truncation concentrates mass, so a sampled token is more likely under q."""
    lik = SequenceLikelihood(num_agents=1)

    lik.update(p_chosen=torch.tensor([0.3]),
               q_chosen=torch.tensor([0.5]),
               valid=torch.tensor([True]))

    assert lik.log_q.item() > lik.log_p.item()


def test_log_q_follows_the_sampler_even_when_it_is_not_p_renormalised():
    """Temperature, tilting and guidance all reshape the sampling distribution
    into something that is not the model's own mass renormalised. q has to be
    supplied by whoever sampled, not inferred from p."""
    lik = SequenceLikelihood(num_agents=1)

    lik.update(p_chosen=torch.tensor([0.2]),
               q_chosen=torch.tensor([0.9]),
               valid=torch.tensor([True]))

    assert lik.log_p.item() == pytest.approx(math.log(0.2))
    assert lik.log_q.item() == pytest.approx(math.log(0.9))
