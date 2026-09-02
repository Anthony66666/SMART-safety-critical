"""Tests for the nuPlan observation that hides what the ego cannot see.

Skipped where nuplan-devkit is absent; the geometry and the memory policy this
wrapper composes are tested on their own in test_visibility.py and
test_tracking.py, which need no devkit.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip('nuplan')

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)

from smart.nuplan.occluded_observation import OccludedObservation

VEHICLE = TrackedObjectType.VEHICLE
PEDESTRIAN = TrackedObjectType.PEDESTRIAN


def agent(track, x, y, kind=VEHICLE, vx=0.0, vy=0.0, length=5.0, width=2.0):
    box = OrientedBox(StateSE2(x, y, 0.0), length, width, 1.6)
    metadata = SceneObjectMetadata(timestamp_us=0, token=track, track_token=track, track_id=1)
    return Agent(kind, box, StateVector2D(vx, vy), metadata)


class FakeObservation:
    """Stands in for whatever observation is being wrapped."""

    def __init__(self, objects):
        self.objects = objects
        self.initialized = False

    def initialize(self):
        self.initialized = True

    def reset(self):
        pass

    def get_observation(self):
        return DetectionsTracks(TrackedObjects(list(self.objects)))

    def update_observation(self, iteration, next_iteration, history):
        pass


def ego_at(x=0.0, y=0.0, time_s=0.0):
    return SimpleNamespace(center=StateSE2(x, y, 0.0),
                           time_point=SimpleNamespace(time_s=time_s))


def scenario_with_ego(ego):
    return SimpleNamespace(get_ego_state_at_iteration=lambda i: ego)


def step(observation, time_s, ego):
    """Advance one simulation step, mirroring nuPlan's propagate() ordering."""
    history = SimpleNamespace(current_state=(ego, None))
    iteration = SimulationIteration(TimePoint(int(time_s * 1e6)), int(time_s * 10))
    observation.update_observation(iteration, iteration, history)


def ids(detections):
    return sorted(o.track_token for o in detections.tracked_objects)


def make(objects, **kwargs):
    wrapped = FakeObservation(objects)
    occluded = OccludedObservation(wrapped, scenario_with_ego(ego_at()), **kwargs)
    occluded.initialize()
    return wrapped, occluded


def test_clear_line_of_sight_passes_objects_through_untouched():
    target = agent('a', 10.0, 20.0)
    _, occluded = make([target])
    [seen] = occluded.get_observation().tracked_objects
    assert seen is target


def test_object_hidden_from_the_start_is_never_reported():
    # The blocker at x=10 subtends a wider angle from the ego than the target
    # at x=30 does, so the target sits entirely inside its shadow.
    _, occluded = make([agent('blocker', 10.0, 0.0), agent('hidden', 30.0, 0.0)])
    assert ids(occluded.get_observation()) == ['blocker']


def test_object_is_remembered_after_it_becomes_occluded():
    target = agent('target', 30.0, 0.0, vx=2.0)
    wrapped, occluded = make([target])
    assert ids(occluded.get_observation()) == ['target']

    wrapped.objects = [agent('blocker', 10.0, 0.0), target]
    step(occluded, 0.5, ego_at())
    believed = {o.track_token: o for o in occluded.get_observation().tracked_objects}
    assert set(believed) == {'blocker', 'target'}
    assert believed['target'] is not target                      # rebuilt, not the live box
    assert believed['target'].center.x == pytest.approx(31.0)    # coasted 2 m/s for 0.5 s


def test_memory_expires():
    target = agent('target', 30.0, 0.0)
    wrapped, occluded = make([target], memory_horizon=1.0)
    occluded.get_observation()
    wrapped.objects = [agent('blocker', 10.0, 0.0), target]
    step(occluded, 2.0, ego_at())
    assert ids(occluded.get_observation()) == ['blocker']


def test_pedestrians_do_not_occlude_by_default():
    _, occluded = make([agent('blocker', 10.0, 0.0, kind=PEDESTRIAN),
                        agent('behind', 30.0, 0.0)])
    assert ids(occluded.get_observation()) == ['behind', 'blocker']


def test_occluder_types_are_configurable():
    _, occluded = make([agent('blocker', 10.0, 0.0, kind=PEDESTRIAN),
                        agent('behind', 30.0, 0.0)],
                       occluder_types=(VEHICLE, PEDESTRIAN))
    assert ids(occluded.get_observation()) == ['blocker']


def test_repeated_calls_within_a_step_do_not_age_memory():
    target = agent('target', 30.0, 0.0, vx=2.0)
    wrapped, occluded = make([target])
    occluded.get_observation()
    wrapped.objects = [agent('blocker', 10.0, 0.0), target]
    step(occluded, 0.5, ego_at())

    first = {o.track_token: o.center.x for o in occluded.get_observation().tracked_objects}
    second = {o.track_token: o.center.x for o in occluded.get_observation().tracked_objects}
    assert first == second


def test_radius_limits_range():
    _, occluded = make([agent('near', 10.0, 20.0), agent('far', 10.0, 90.0)],
                       radius=55.0)
    assert ids(occluded.get_observation()) == ['near']


def test_empty_scene_is_handled():
    _, occluded = make([])
    assert occluded.get_observation().tracked_objects.tracked_objects == []


def test_reset_clears_memory_and_requires_reinitialisation():
    target = agent('target', 30.0, 0.0)
    wrapped, occluded = make([target])
    occluded.get_observation()
    occluded.reset()
    with pytest.raises(RuntimeError):
        occluded.get_observation()


def test_initialize_forwards_to_the_wrapped_observation():
    wrapped, _ = make([agent('a', 10.0, 20.0)])
    assert wrapped.initialized


def test_memory_never_outlives_the_underlying_observation():
    """The occluded view must stay a subset of the fully observable one.

    A buffer cannot tell an occluded object from one the underlying observation
    has stopped reporting -- an object that left sensor range, or the log's
    detection set. Bridging the second kind hands the planner a ghost, and the
    occluded condition then carries information the full-observability baseline
    it is compared against does not have, which inverts the whole experiment.
    """
    near = agent('near', 10.0, 0.0)
    leaving = agent('leaving', 10.0, 20.0)
    wrapped, occluded = make([near, leaving], memory_horizon=5.0)

    # Both in clear view to begin with, so both are genuinely observed.
    assert ids(occluded.get_observation()) == ['leaving', 'near']

    # The underlying observation stops reporting one -- it drove out of range,
    # not behind something. Memory must drop it too, well inside the horizon.
    wrapped.objects = [near]
    step(occluded, 0.1, ego_at(0.0, 0.0, 0.1))
    assert ids(occluded.get_observation()) == ['near']


def test_output_is_always_a_subset_of_the_input():
    blocker = agent('blocker', 10.0, 0.0)
    hidden = agent('hidden', 20.0, 0.0)
    other = agent('other', -10.0, 0.0)
    wrapped, occluded = make([blocker, hidden, other], memory_horizon=5.0)

    for time_s, objects in ((0.0, [blocker, hidden, other]),
                            (0.1, [blocker, hidden]),
                            (0.2, [hidden]),
                            (0.3, [])):
        wrapped.objects = objects
        given = set(ids(occluded.get_observation()))
        assert given <= {o.track_token for o in objects}
        step(occluded, time_s + 0.1, ego_at(0.0, 0.0, time_s + 0.1))


def _count(detections):
    return len(list(detections.tracked_objects))


def test_randomised_control_withholds_the_same_number():
    """The control's whole purpose is to remove as much and no more.

    If it withheld a different amount, a difference between the conditions
    could be explained by information volume rather than by which objects went
    missing, and the control would prove nothing.
    """
    blocker = agent('blocker', 10.0, 0.0)
    hidden = agent('hidden', 20.0, 0.0)
    hidden2 = agent('hidden2', 30.0, 0.0)
    clear = agent('clear', 0.0, 20.0)
    objects = [blocker, hidden, hidden2, clear]

    _, occluded = make(objects, memory_horizon=0.0)
    _, randomised = make(objects, memory_horizon=0.0, randomise=True, seed=1)

    assert _count(occluded.get_observation()) == _count(randomised.get_observation())


def test_randomised_control_stays_a_subset():
    objects = [agent('a', 10.0, 0.0), agent('b', 20.0, 0.0),
               agent('c', 30.0, 0.0), agent('d', 0.0, 15.0)]
    _, randomised = make(objects, memory_horizon=0.0, randomise=True, seed=3)
    given = set(ids(randomised.get_observation()))
    assert given <= {o.track_token for o in objects}


def test_randomised_control_is_reproducible():
    objects = [agent(f'a{i}', 10.0 + 6 * i, 0.0) for i in range(6)]
    _, first = make(objects, memory_horizon=0.0, randomise=True, seed=7)
    _, second = make(objects, memory_horizon=0.0, randomise=True, seed=7)
    assert ids(first.get_observation()) == ids(second.get_observation())


def test_randomised_control_differs_from_the_sight_lines():
    """Over enough objects the two should disagree about who is missing.

    Identical output would mean the control is not controlling for anything.
    """
    objects = [agent(f'a{i}', 10.0 + 6 * i, 0.0) for i in range(8)]
    _, occluded = make(objects, memory_horizon=0.0)
    disagreed = False
    for seed in range(8):
        _, randomised = make(objects, memory_horizon=0.0, randomise=True, seed=seed)
        if ids(occluded.get_observation()) != ids(randomised.get_observation()):
            disagreed = True
            break
    assert disagreed
