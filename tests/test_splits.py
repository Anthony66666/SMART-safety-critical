"""Tests for deterministic dataset partitioning.

The judge model must never see the generator's training data, or realism
scores become circular. Splits are assigned by hashing the scenario id, so
membership is reproducible across machines and runs without shipping a
manifest of half a million filenames.
"""
import pytest

from smart.safety.splits import assign_split, SplitSpec


SPEC = SplitSpec(fractions={'judge_train': 0.6, 'judge_val': 0.1, 'eval': 0.3},
                 salt='m1')


def test_assignment_is_deterministic():
    assert assign_split('100006d9c3e93b6e', SPEC) == assign_split('100006d9c3e93b6e', SPEC)


def test_assignment_only_returns_declared_splits():
    names = {assign_split(f'scenario_{i}', SPEC) for i in range(200)}
    assert names <= set(SPEC.fractions)


def test_salt_changes_the_partition():
    other = SplitSpec(fractions=SPEC.fractions, salt='different')
    ids = [f'scenario_{i}' for i in range(200)]
    assert [assign_split(i, SPEC) for i in ids] != [assign_split(i, other) for i in ids]


def test_proportions_track_the_requested_fractions():
    ids = [f'scenario_{i}' for i in range(20000)]
    counts = {name: 0 for name in SPEC.fractions}
    for i in ids:
        counts[assign_split(i, SPEC)] += 1
    for name, frac in SPEC.fractions.items():
        assert counts[name] / len(ids) == pytest.approx(frac, abs=0.02)


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        SplitSpec(fractions={'a': 0.5, 'b': 0.2}, salt='x')
