"""Tests for the SMART-driven traffic observation.

These exercise the bookkeeping around the model -- the history window, the
substitution of the simulated ego for the logged one, the split between agents
SMART drives and static objects replayed from the log -- and stub the model
itself. The model is a 85 MB checkpoint and a GPU; what can go wrong here and
stay silent is the plumbing, not the network.
"""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('nuplan')
pytest.importorskip('torch_geometric')

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import (StateSE2, StateVector2D,
                                                            TimePoint)
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)

from smart.nuplan.smart_agents import HISTORY_STEPS, SMART_STEP_S, SMARTAgents

VEHICLE = TrackedObjectType.VEHICLE
CONE = TrackedObjectType.TRAFFIC_CONE


def agent(track, x, y, kind=VEHICLE):
    box = OrientedBox(StateSE2(x, y, 0.0), 5.0, 2.0, 1.6)
    metadata = SceneObjectMetadata(timestamp_us=0, token=track, track_token=track, track_id=1)
    return Agent(kind, box, StateVector2D(0.0, 0.0), metadata)


def cone(track, x, y):
    box = OrientedBox(StateSE2(x, y, 0.0), 0.3, 0.3, 0.5)
    metadata = SceneObjectMetadata(timestamp_us=0, token=track, track_token=track, track_id=1)
    return StaticObject(CONE, box, metadata)


def ego_at(x=0.0, y=0.0):
    footprint = SimpleNamespace(oriented_box=OrientedBox(StateSE2(x, y, 0.0), 5.0, 2.0, 1.7))
    return SimpleNamespace(
        center=StateSE2(x, y, 0.0),
        car_footprint=footprint,
        dynamic_car_state=SimpleNamespace(center_velocity_2d=StateVector2D(1.0, 0.0)))


class FakeScenario:
    """Enough of a scenario for the observation to seed itself from.

    `database_interval` is the simulation's step, and it is not a constant:
    nuPlan's official scenario builder subsamples the 20 Hz logs to 10 Hz while
    the converter's scenarios stay at 20 Hz. The observation has to read it
    rather than assume it.
    """

    def __init__(self, objects, database_interval=0.1):
        self.objects = objects
        self.database_interval = database_interval
        self.map_api = SimpleNamespace(map_name='fake')

    def get_ego_state_at_iteration(self, iteration):
        return ego_at(float(iteration), 0.0)

    def get_tracked_objects_at_iteration(self, iteration):
        return DetectionsTracks(TrackedObjects(list(self.objects)))

    def get_ego_past_trajectory(self, iteration, time_horizon, num_samples=None):
        return (ego_at(-float(i), 0.0) for i in range(num_samples, 0, -1))

    def get_past_tracked_objects(self, iteration, time_horizon, num_samples=None,
                                 future_trajectory_sampling=None):
        return (DetectionsTracks(TrackedObjects(list(self.objects)))
                for _ in range(num_samples))


class StubModel:
    """Sends every agent 100 m north of wherever it was asked about.

    Mimics the two prep hooks the real SMART LightningModule exposes so the
    observation's rollout path can be exercised without either. Both are pure
    passthroughs here -- the training-time map masking they do in the model is
    not what these tests are about.
    """

    def __init__(self):
        self.calls = 0

    def match_token_map(self, data):
        return data

    def sample_pt_pred(self, data):
        return data

    def inference(self, data):
        self.calls += 1
        count = int(data['agent']['num_nodes'])
        traj = torch.zeros(count, 80, 2)
        traj[:, :, 1] = 100.0
        return {'pred_traj': traj, 'pred_head': torch.zeros(count, 80)}


def make(objects, database_interval=0.1, **kwargs):
    """An observation whose model and map are stubs.

    The map needs a real nuPlan map API and the model needs an 85 MB
    checkpoint; neither is what breaks silently. Tokenising is stubbed with it,
    so `_agent_tensors` -- the part that has to match the layout the checkpoint
    was trained on -- is tested directly instead, below.
    """
    scenario = FakeScenario(objects, database_interval)
    observation = SMARTAgents(StubModel(), scenario, device='cpu', **kwargs)
    observation._build_map = lambda: {}

    def build_input(ego_state):
        agent = observation._agent_tensors()
        return None if agent is None else {'agent': agent}

    observation._build_input = build_input
    return observation


def step(observation, index, ego):
    history = SimpleNamespace(current_state=(ego, None))
    at = lambda i: SimulationIteration(TimePoint(int(i * 5e4)), i)
    observation.update_observation(at(index), at(index + 1), history)


def ids(detections):
    return sorted(o.track_token for o in detections.tracked_objects)


def test_history_is_seeded_to_a_full_window():
    observation = make([agent('a', 10.0, 0.0)])
    observation.initialize()
    assert len(observation._history) == HISTORY_STEPS


def test_history_never_grows_past_the_window():
    observation = make([agent('a', 10.0, 0.0)])
    observation.initialize()
    for index in range(40):
        step(observation, index, ego_at(float(index), 0.0))
    assert len(observation._history) == HISTORY_STEPS


def test_history_records_the_simulated_ego_not_the_logged_one():
    """The whole point of a reactive model: it must see where the ego went."""
    observation = make([agent('a', 10.0, 0.0)])
    observation.initialize()
    # A pose the logged ego never takes -- FakeScenario only ever moves along y=0.
    step(observation, 0, ego_at(3.0, 77.0))
    assert observation._history[-1][0].center.y == 77.0


def test_every_step_is_recorded_when_the_simulation_runs_at_smart_rate():
    """nuPlan's closed-loop challenges step at 10 Hz, which is SMART's rate.

    Assuming the 20 Hz of the raw logs instead samples the history at 5 Hz and
    advances the agents at half speed, and neither shows up as an error.
    """
    observation = make([agent('a', 10.0, 0.0)], database_interval=0.1)
    observation.initialize()
    assert observation._stride == 1
    step(observation, 1, ego_at(50.0, 0.0))
    assert observation._history[-1][0].center.x == 50.0


def test_history_is_thinned_when_the_simulation_runs_faster_than_smart():
    observation = make([agent('a', 10.0, 0.0)], database_interval=0.05)
    observation.initialize()
    assert observation._stride == 2
    before = observation._history[-1][0].center.x
    step(observation, 1, ego_at(50.0, 0.0))   # odd index: not a SMART frame
    assert observation._history[-1][0].center.x == before
    step(observation, 2, ego_at(60.0, 0.0))   # even index: recorded
    assert observation._history[-1][0].center.x == 60.0


def test_the_history_window_spans_one_second_whatever_the_rate():
    """SMART reads ten past steps at 10 Hz; the span is what matters."""
    for interval in (0.05, 0.1):
        observation = make([agent('a', 10.0, 0.0)], database_interval=interval)
        observation.initialize()
        assert len(observation._history) == HISTORY_STEPS
        assert (HISTORY_STEPS - 1) * SMART_STEP_S == pytest.approx(1.0)


def test_agents_move_to_the_predicted_pose():
    observation = make([agent('a', 10.0, 0.0)])
    observation.initialize()
    step(observation, 0, ego_at())
    moved = observation.get_observation().tracked_objects
    predicted = [o for o in moved if o.track_token == 'a'][0]
    assert predicted.box.center.y == pytest.approx(100.0)


def test_agents_keep_their_identity_and_size_through_a_rollout():
    original = agent('a', 10.0, 0.0)
    observation = make([original])
    observation.initialize()
    step(observation, 0, ego_at())
    moved = [o for o in observation.get_observation().tracked_objects
             if o.track_token == 'a'][0]
    assert moved.track_token == original.track_token
    assert moved.box.length == original.box.length
    assert moved.box.width == original.box.width
    assert moved.tracked_object_type == original.tracked_object_type


def test_static_objects_are_replayed_rather_than_predicted():
    """Cones have no token vocabulary; asking SMART to predict them is a bug."""
    observation = make([agent('a', 10.0, 0.0), cone('c', 4.0, 1.0)])
    observation.initialize()
    step(observation, 0, ego_at())
    emitted = observation.get_observation().tracked_objects
    assert ids(DetectionsTracks(TrackedObjects(list(emitted)))) == ['a', 'c']
    replayed = [o for o in emitted if o.track_token == 'c'][0]
    assert replayed.box.center.y == pytest.approx(1.0)


def test_the_ego_is_not_emitted_as_traffic():
    """SMART predicts the av row too; handing it back would duplicate the ego."""
    observation = make([agent('a', 10.0, 0.0)])
    observation.initialize()
    step(observation, 0, ego_at())
    assert 'ego' not in ids(observation.get_observation())


def test_reset_clears_the_history():
    observation = make([agent('a', 10.0, 0.0)])
    observation.initialize()
    observation.reset()
    assert observation._history == []
    assert observation._plan == {}


def test_the_ego_occupies_the_last_row():
    """WOMD puts the av last and the checkpoint was trained that way."""
    observation = make([agent('a', 10.0, 0.0), agent('b', 20.0, 0.0)])
    observation.initialize()
    agents = observation._agent_tensors()
    assert agents['av_index'] == agents['num_nodes'] - 1
    assert agents['id'][-1] == 'ego'


def test_positions_are_recentred_on_the_origin():
    """UTM northings in float32 quantise to 0.25 m; recentring is not optional."""
    observation = make([agent('a', 10.0, 0.0)])
    observation.initialize()
    agents = observation._agent_tensors()
    av = agents['av_index']
    # The current ego sits at the origin regardless of where the past frames
    # were; recentring subtracts the same origin from every step.
    assert float(agents['position'][av, HISTORY_STEPS - 1, 0]) == pytest.approx(0.0)


def test_invalid_slots_stay_exactly_zero():
    """A non-zero invalid slot is read as a position to interpolate from."""
    observation = make([agent('a', 10.0, 0.0)])
    observation.initialize()
    agents = observation._agent_tensors()
    invalid = ~agents['valid_mask']
    assert torch.count_nonzero(agents['position'][invalid]) == 0


def test_only_the_history_window_is_marked_valid():
    """Everything past the window is what the model is being asked to produce."""
    observation = make([agent('a', 10.0, 0.0)])
    observation.initialize()
    agents = observation._agent_tensors()
    assert bool(agents['valid_mask'][:, :HISTORY_STEPS].all())
    assert not bool(agents['valid_mask'][:, HISTORY_STEPS:].any())


def test_static_objects_get_no_row():
    """SMART has token embeddings for three types and nothing for cones."""
    observation = make([agent('a', 10.0, 0.0), cone('c', 4.0, 1.0)])
    observation.initialize()
    agents = observation._agent_tensors()
    assert agents['id'] == ['a', 'ego']


def test_inference_runs_once_per_replan_interval():
    """Half a second between rollouts, counted in seconds rather than steps."""
    observation = make([agent('a', 10.0, 0.0)], replan_seconds=0.5)
    observation.initialize()
    assert observation._replan_steps == 5      # 0.5 s at 10 Hz
    for index in range(20):
        step(observation, index, ego_at(float(index), 0.0))
    assert observation._model.calls == 4


def test_agents_move_smoothly_between_predicted_poses():
    """SMART predicts at 10 Hz, nuPlan steps at 20 Hz.

    Holding each predicted pose for both steps makes half of them a zero-length
    move, so the traffic stutters and anything reading speed off consecutive
    frames sees agents alternating between stopped and double speed. The
    symptom is a median vehicle speed of exactly zero.
    """
    observation = make([agent('a', 10.0, 0.0)])

    class RampModel(StubModel):
        """A trajectory that actually goes somewhere, one metre per 10 Hz step."""

        def inference(self, data):
            count = int(data['agent']['num_nodes'])
            traj = torch.zeros(count, 80, 2)
            traj[:, :, 1] = torch.arange(1, 81, dtype=torch.float)
            return {'pred_traj': traj, 'pred_head': torch.zeros(count, 80)}

    observation._model = RampModel()
    observation.initialize()
    seen = []
    for index in range(6):
        step(observation, index, ego_at(float(index), 0.0))
        moved = [o for o in observation.get_observation().tracked_objects
                 if o.track_token == 'a'][0]
        seen.append(moved.box.center.y)
    steps = [abs(b - a) for a, b in zip(seen, seen[1:])]
    assert all(s > 0.0 for s in steps), f'stalled on some step: {steps}'


def test_velocity_is_recomputed_from_the_motion():
    """Carrying the template's velocity pins every agent to its t=0 state.

    SMART reads velocity back as its own input, and the history ablation says
    velocity is the only part of the history that matters, so a stale value
    feeds straight into the next rollout. The planner reads it too.
    """
    observation = make([agent('a', 10.0, 0.0)])

    class RampModel(StubModel):
        def inference(self, data):
            count = int(data['agent']['num_nodes'])
            traj = torch.zeros(count, 80, 2)
            traj[:, :, 1] = torch.arange(1, 81, dtype=torch.float)
            return {'pred_traj': traj, 'pred_head': torch.zeros(count, 80)}

    observation._model = RampModel()
    observation.initialize()
    for index in range(3):
        step(observation, index, ego_at(float(index), 0.0))
    moved = [o for o in observation.get_observation().tracked_objects
             if o.track_token == 'a'][0]
    # The stub advances one metre per 10 Hz step, so 10 m/s northwards.
    assert moved.velocity.y == pytest.approx(10.0, abs=0.5)
    assert moved.velocity.x == pytest.approx(0.0, abs=0.5)
