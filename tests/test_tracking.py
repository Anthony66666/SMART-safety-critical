"""Tests for object permanence across occlusion gaps."""
import pytest

from smart.occlusion.tracking import (
    CONSTANT_VELOCITY,
    HOLD,
    TrackObservation,
    TrackingBuffer,
)


def obs(track_id='a', x=0.0, y=0.0, vx=0.0, vy=0.0):
    return TrackObservation(track_id=track_id, x=x, y=y, heading=0.0,
                            velocity_x=vx, velocity_y=vy, width=2.0, length=4.5)


def test_visible_object_is_reported_as_observed():
    buffer = TrackingBuffer()
    [estimate] = buffer.update(0.0, [obs(x=10.0)])
    assert estimate.observed
    assert estimate.time_since_observed == 0.0
    assert estimate.x == 10.0


def test_occluded_object_survives_and_coasts():
    buffer = TrackingBuffer(memory_horizon=3.0, propagate=CONSTANT_VELOCITY)
    buffer.update(0.0, [obs(x=10.0, vx=5.0)])

    [estimate] = buffer.update(0.5, [])
    assert not estimate.observed
    assert estimate.time_since_observed == pytest.approx(0.5)
    assert estimate.x == pytest.approx(12.5)   # coasted at 5 m/s for 0.5 s
    assert estimate.velocity_x == 5.0          # last observed velocity is kept


def test_hold_mode_does_not_move_an_unseen_track():
    buffer = TrackingBuffer(memory_horizon=3.0, propagate=HOLD)
    buffer.update(0.0, [obs(x=10.0, vx=5.0)])
    [estimate] = buffer.update(0.5, [])
    assert estimate.x == 10.0
    assert estimate.time_since_observed == pytest.approx(0.5)


def test_track_is_forgotten_past_the_memory_horizon():
    buffer = TrackingBuffer(memory_horizon=1.0)
    buffer.update(0.0, [obs()])
    assert len(buffer.update(1.0, [])) == 1     # exactly at the horizon, still held
    assert buffer.update(1.5, []) == []


def test_forgotten_track_does_not_come_back_on_its_own():
    buffer = TrackingBuffer(memory_horizon=1.0)
    buffer.update(0.0, [obs()])
    buffer.update(2.0, [])
    assert buffer.update(2.5, []) == []


def test_reappearance_resets_staleness():
    buffer = TrackingBuffer(memory_horizon=3.0)
    buffer.update(0.0, [obs(x=0.0)])
    buffer.update(1.0, [])
    [estimate] = buffer.update(2.0, [obs(x=42.0)])
    assert estimate.observed
    assert estimate.time_since_observed == 0.0
    assert estimate.x == 42.0


def test_never_seen_object_is_never_reported():
    buffer = TrackingBuffer()
    assert buffer.update(0.0, []) == []
    assert buffer.update(1.0, []) == []


def test_output_order_is_independent_of_observation_order():
    forward, backward = TrackingBuffer(), TrackingBuffer()
    a, b, c = obs('a', x=1.0), obs('b', x=2.0), obs('c', x=3.0)
    ids = lambda es: [e.track_id for e in es]
    assert ids(forward.update(0.0, [a, b, c])) == ids(backward.update(0.0, [c, b, a]))
    assert ids(forward.update(0.0, [a, b, c])) == ['a', 'b', 'c']


def test_partially_occluded_scene_keeps_both_kinds():
    buffer = TrackingBuffer(memory_horizon=3.0)
    buffer.update(0.0, [obs('a', x=1.0, vx=1.0), obs('b', x=2.0, vx=2.0)])
    estimates = {e.track_id: e for e in buffer.update(1.0, [obs('a', x=5.0)])}
    assert estimates['a'].observed and estimates['a'].x == 5.0
    assert not estimates['b'].observed and estimates['b'].x == pytest.approx(4.0)


def test_reset_clears_memory():
    buffer = TrackingBuffer()
    buffer.update(0.0, [obs()])
    buffer.reset()
    assert buffer.update(0.1, []) == []


def test_zero_horizon_drops_the_moment_sight_is_lost():
    buffer = TrackingBuffer(memory_horizon=0.0)
    assert len(buffer.update(0.0, [obs()])) == 1
    assert buffer.update(0.1, []) == []


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        TrackingBuffer(memory_horizon=-1.0)
    with pytest.raises(ValueError):
        TrackingBuffer(propagate='teleport')
