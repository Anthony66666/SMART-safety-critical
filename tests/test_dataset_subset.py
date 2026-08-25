"""Tests for restricting a dataset to a split.

The judge and the generator must be trainable on different scenario sets from
the same directory, so the dataset has to accept an explicit id subset.
"""
import os

from smart.datasets.scalable_dataset import MultiDataset

DEMO = ["data/valid_demo"]


def _ids():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(DEMO[0]))


def _dataset(**kwargs):
    return MultiDataset(root=None, split='val', raw_dir=DEMO, **kwargs)


def test_unrestricted_dataset_sees_every_scenario():
    """Default behaviour must not change for existing callers."""
    assert len(_dataset()) == len(_ids())


def test_dataset_can_be_restricted_to_a_subset():
    keep = set(_ids()[:3])
    assert len(_dataset(scenario_ids=keep)) == 3


def test_restricted_dataset_loads_only_the_requested_scenarios():
    keep = set(_ids()[:3])
    ds = _dataset(scenario_ids=keep)
    loaded = {os.path.splitext(os.path.basename(p))[0] for p in ds.raw_paths}
    assert loaded == keep
