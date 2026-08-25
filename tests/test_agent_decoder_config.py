"""Tests for rollout configuration on SMARTAgentDecoder.

beam_size controls top-k truncation during sampling. It must be configurable:
unbiased rare-event estimation needs full support (beam_size = token_size),
while the default has to stay 5 so the existing val.py path is unchanged.
"""
import pickle

import torch

from smart.modules.agent_decoder import SMARTAgentDecoder

TOKEN_PATH = "smart/tokens/cluster_frame_5_2048.pkl"


def _build_decoder(**overrides):
    with open(TOKEN_PATH, "rb") as f:
        token_data = pickle.load(f)
    kwargs = dict(
        dataset="waymo", input_dim=2, hidden_dim=16, num_historical_steps=11,
        time_span=30, pl2a_radius=30.0, a2a_radius=60.0, num_freq_bands=64,
        num_layers=1, num_heads=2, head_dim=8, dropout=0.1,
        token_data=token_data, token_size=2048,
    )
    kwargs.update(overrides)
    return SMARTAgentDecoder(**kwargs)


def test_beam_size_defaults_to_five():
    """The default must not change, or existing val.py results shift."""
    assert _build_decoder().beam_size == 5


def test_beam_size_is_configurable():
    """Full support is required for unbiased importance weights."""
    assert _build_decoder(beam_size=2048).beam_size == 2048
