"""Tests for judge-model construction.

The judge exists to break the circularity of scoring SMART samples with
SMART. That argument collapses silently if the judge ends up being the same
model, so the difference is asserted rather than assumed.
"""
import pytest

from smart.safety.judge import assert_distinct_architecture, split_spec_from_config
from smart.utils.config import load_config_act

GENERATOR = "configs/train/train_scalable.yaml"


def _arch(**overrides):
    arch = dict(hidden_dim=128, num_heads=8, head_dim=16,
                num_agent_layers=6, num_map_layers=3, seed=2)
    arch.update(overrides)
    return arch


def test_identical_architecture_is_rejected():
    with pytest.raises(ValueError, match="indistinguishable"):
        assert_distinct_architecture(_arch(), _arch())


def test_same_architecture_with_a_different_seed_is_still_rejected():
    """A reseeded twin shares inductive biases; reviewers will not accept it
    as an independent density model."""
    with pytest.raises(ValueError, match="indistinguishable"):
        assert_distinct_architecture(_arch(seed=99), _arch())


def test_differing_width_and_depth_is_accepted():
    assert_distinct_architecture(
        _arch(hidden_dim=192, num_heads=6, head_dim=32,
              num_agent_layers=4, num_map_layers=2, seed=99),
        _arch())


def test_split_spec_is_read_from_config():
    cfg = load_config_act("configs/judge/judge_server.yaml")
    spec = split_spec_from_config(cfg)

    assert spec.salt
    assert sum(spec.fractions.values()) == pytest.approx(1.0)


def test_generator_architecture_is_read_from_the_checkpoint():
    """The config file may have drifted; the checkpoint records what was
    actually trained, so the independence check reads from there."""
    from smart.safety.judge import generator_architecture_from_checkpoint

    arch = generator_architecture_from_checkpoint("checkpoints/epoch=31.ckpt")

    assert arch['hidden_dim'] == 128
    assert arch['num_heads'] == 8
    assert arch['head_dim'] == 16
    assert arch['num_agent_layers'] == 6
    assert arch['num_map_layers'] == 3


def test_shipped_judge_config_is_distinct_from_the_trained_generator():
    """Guards the actual config we will train with, not a hypothetical one."""
    from smart.safety.judge import (generator_architecture_from_checkpoint,
                                    judge_architecture_from_config)

    judge = judge_architecture_from_config(
        load_config_act("configs/judge/judge_server.yaml"))
    generator = generator_architecture_from_checkpoint("checkpoints/epoch=31.ckpt")

    assert_distinct_architecture(judge, generator)
