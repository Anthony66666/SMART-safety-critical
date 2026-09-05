"""Background traffic driven by SMART instead of IDM.

nuPlan's reactive challenge drives other vehicles with IDM, which follows a lead
vehicle and cannot react to anything in an adjacent lane. Replacing it with a
learned traffic model is not our idea -- Hagedorn et al. did it at ICRA 2026 and
released a SMART checkpoint trained on nuPlan -- and this benchmark adopts it as
infrastructure rather than claiming it. Their implementation is AGPL-3.0, so
this is written against SMART's own API instead of derived from theirs; only the
checkpoint is shared, and that is data the user fetches themselves.

The ego is the point of the exercise. SMART predicts every agent including the
one at av_index, but the ego is the planner's to control, so its predictions are
discarded there and the *simulated* ego pose is written into the history the
model reads. That is what makes the traffic reactive: the background cars see
where the planner actually went, not where the log says it went.

Two rates have to be reconciled. SMART was trained at 10 Hz with a five-step
token, nuPlan simulates at 20 Hz. History is therefore sampled every second
frame, and each predicted 10 Hz pose is held for two simulation steps. Inference
is not run every step -- it rolls out a whole future at once, and re-running it
constantly would be wasteful and would also make the traffic jitter as
successive rollouts disagree.
"""
import copy
import math
import os
from typing import Dict, List, Optional, Type

import torch

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObject
from nuplan.common.actor_state.state_representation import (Point2D, StateSE2,
                                                            StateVector2D)
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation

from torch_geometric.data import HeteroData

from smart.datasets.preprocess import TokenProcessor
from smart.nuplan.converter import (AGENT_TYPE, AGENT_VEHICLE, WOMD_CURRENT_STEP,
                                    WOMD_STEPS, _convert_map,
                                    to_nuplan_checkpoint_semantics)

# SMART's history window: the current step plus everything before it.
HISTORY_STEPS = WOMD_CURRENT_STEP + 1

# SMART's own step. The simulation's step is a separate number and the two are
# not reliably equal: nuPlan's raw logs are 20 Hz, but the official scenario
# builder subsamples to 10 Hz, and the converter's scenarios do not. Assuming
# either one is how the traffic ends up at half speed. The ratio is read from
# the scenario at initialize time instead.
SMART_STEP_S = 0.1

# Only these three have a token vocabulary; everything else nuPlan tracks --
# cones, barriers, debris -- is static and is replayed from the log untouched.
SIMULATED_TYPES = (TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN,
                   TrackedObjectType.BICYCLE)


def _track_id(tracked_object) -> str:
    return str(tracked_object.track_token or tracked_object.token)


class SMARTAgents(AbstractObservation):
    """Drives non-ego agents with SMART, replaying static objects from the log.

    Args:
        model: a SMART LightningModule with weights already loaded.
        scenario: used for the map, for the static objects, and to seed the
            history before the simulation has one of its own.
        replan_seconds: seconds between rollouts. Half a second is what the
            reference implementation uses. Lower is not obviously better:
            successive rollouts disagree, and re-running constantly makes the
            traffic jitter.
        map_radius: metres of map around the starting ego handed to the model.
        seed: makes the rollout reproducible, and identical between the
            occluded and unoccluded runs of the same scenario.
        checkpoint_semantics: relabel map types to the numbering the
            nuPlan-trained checkpoint saw. Turn it off for a model trained
            through this repo's own WOMD preprocessing.
    """

    def __init__(self, model, scenario, replan_seconds: float = 0.5,
                 map_radius: float = 150.0, seed: int = 0,
                 checkpoint_semantics: bool = True, device: str = 'cuda'):
        self._model = model
        self._scenario = scenario
        self._replan_seconds = replan_seconds
        # Filled in at initialize, once the scenario can be asked its rate.
        self._step_time = SMART_STEP_S
        self._stride = 1
        self._replan_steps = 1
        self._map_radius = map_radius
        self._seed = seed
        self._device = device
        self._checkpoint_semantics = checkpoint_semantics
        self._token_processor = TokenProcessor(2048)

        self._map: Optional[Dict] = None
        self._origin = (0.0, 0.0)
        self._shift = (0.0, 0.0)
        self._iteration = 0
        # (ego_state, {track_id: tracked_object}) at 10 Hz, oldest first.
        self._history: List = []
        self._templates: Dict[str, object] = {}
        self._current: Dict[str, object] = {}
        self._plan: Dict[str, torch.Tensor] = {}
        self._plan_start = 0

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks

    def initialize(self) -> None:
        """Inherited, see superclass."""
        self.reset()
        ego = self._scenario.get_ego_state_at_iteration(0)
        self._origin = (ego.center.x, ego.center.y)
        # Where the map query is centred and how the coordinates are shifted
        # are separate questions. The checkpoint was trained on data that was
        # never recentred, so shifting to a local frame puts it out of
        # distribution -- worth 7.49% against 23.62% next-token top-1. See
        # converter.convert_scenario.
        self._shift = (0.0, 0.0) if self._checkpoint_semantics else self._origin
        self._map = self._build_map()

        # Seed the window from the scenario's *past*. Before the first step
        # there is no simulated history, and SMART needs a full window to
        # condition on -- but it has to be the past. Reading forward from
        # iteration 0 instead would open the simulation with a model that has
        # already been shown two seconds of the ego's logged future and has
        # started the traffic from where the log ends up, which is the kind of
        # leak that makes a closed-loop number meaningless.
        # One SMART step per `stride` simulation steps. A 10 Hz simulation
        # gives 1, a 20 Hz one gives 2.
        step_time = getattr(self._scenario, 'database_interval', SMART_STEP_S)
        self._step_time = step_time
        self._stride = max(1, int(round(SMART_STEP_S / step_time)))
        self._replan_steps = max(1, int(round(self._replan_seconds / step_time)))
        horizon = (HISTORY_STEPS - 1) * SMART_STEP_S
        past_egos = list(self._scenario.get_ego_past_trajectory(
            0, horizon, HISTORY_STEPS - 1))
        past_objects = list(self._scenario.get_past_tracked_objects(
            0, horizon, HISTORY_STEPS - 1))
        for ego_state, objects in zip(past_egos, past_objects):
            self._remember(ego_state, objects)
        self._remember(self._scenario.get_ego_state_at_iteration(0),
                       self._scenario.get_tracked_objects_at_iteration(0))

        # A scenario anchored near the start of a log has less past than the
        # window wants. Repeating the oldest frame keeps the window full, which
        # matters more than it looks: SMART reads the step at HISTORY_STEPS - 1
        # to decide which agents to predict at all, and that step only lands
        # there if the earlier slots are occupied. A short window silently
        # predicts nothing and the traffic freezes.
        while len(self._history) < HISTORY_STEPS:
            self._history.insert(0, self._history[0])
        self._current = dict(self._history[-1][1])

    def _build_map(self) -> Dict:
        """Lanes and crosswalks around the start, taken once.

        The map does not change during a simulation and the query is expensive
        -- nuPlan map lookups are this benchmark's measured bottleneck -- so it
        is built at initialise time and reused. The radius is generous enough
        that the ego cannot drive out of it in one scenario.
        """
        raw = _convert_map(self._scenario, Point2D(*self._origin),
                           self._map_radius, True, self._shift)
        # The checkpoint this observation loads was trained through an
        # Argoverse-shaped intermediate, which numbers map types differently
        # from the WOMD preprocessing our converter follows. Left alone, every
        # point type we emit lands on an embedding row that checkpoint never
        # trained. See to_nuplan_checkpoint_semantics -- including the
        # measurement showing this does not actually move next-token accuracy.
        return to_nuplan_checkpoint_semantics(raw) if self._checkpoint_semantics else raw

    def reset(self) -> None:
        """Inherited, see superclass."""
        self._iteration = 0
        self._shift = (0.0, 0.0)
        self._history = []
        self._templates = {}
        self._current = {}
        self._plan = {}
        self._plan_start = 0

    def _remember(self, ego_state, detections: DetectionsTracks) -> None:
        """Append one 10 Hz frame, keeping only the window SMART reads."""
        agents = {}
        for obj in detections.tracked_objects:
            if obj.tracked_object_type in SIMULATED_TYPES:
                agents[_track_id(obj)] = obj
                self._templates.setdefault(_track_id(obj), obj)
        self._history.append((ego_state, agents))
        del self._history[:-HISTORY_STEPS]

    def update_observation(self, iteration, next_iteration, history) -> None:
        """Inherited, see superclass."""
        self._iteration = next_iteration.index
        ego_state, _ = history.current_state

        # The history SMART conditions on is at its own rate, so only every
        # `stride`-th simulation step contributes -- and the ego written in is
        # the simulated one, which is the whole point of a reactive model.
        if iteration.index % self._stride == 0:
            self._remember(ego_state, DetectionsTracks(TrackedObjects(
                list(self._current.values()))))

        if self._iteration - self._plan_start >= self._replan_steps or not self._plan:
            self._rollout(ego_state)
            self._plan_start = self._iteration
        self._advance()

    @torch.no_grad()
    def _rollout(self, ego_state) -> None:
        """Run SMART once and keep the predicted trajectories.

        The two preparation steps are the model's own: `validation_step` calls
        `match_token_map` then `sample_pt_pred` before `inference`, so this is
        the path the checkpoint was evaluated on. `sample_pt_pred` masks a third
        of the map points at random -- a training-time augmentation the authors
        left in the rollout path, and not ours to remove.

        It does mean the traffic depends on the RNG, which for a paired
        comparison is a problem: the same scenario run with and without
        occlusion would draw different map masks and the difference between the
        two conditions would include that noise. Seeding on the step index
        makes the traffic reproducible and identical across conditions, so what
        is left between them is the occlusion.
        """
        data = self._build_input(ego_state)
        if data is None:
            return
        torch.manual_seed(self._seed + self._iteration)
        data = self._model.match_token_map(data)
        data = self._model.sample_pt_pred(data)
        prediction = self._model.inference(data)
        traj = prediction['pred_traj'].cpu()          # (N, T, 2), recentred
        head = prediction['pred_head'].cpu()          # (N, T)
        self._plan = {track: (traj[row], head[row])
                      for track, row in self._rows.items()}

    def _build_input(self, ego_state):
        """The HeteroData SMART expects: live agent tensors plus the map."""
        agent = self._agent_tensors()
        if agent is None:
            return None
        data = {'scenario_id': 'live', 'city': self._scenario.map_api.map_name,
                'origin': self._origin, 'agent': agent}
        # A fresh copy every rollout: TokenProcessor tokenises the map in place,
        # so handing it the cached dict would let the first rollout consume it
        # and leave every later one reading its own leftovers.
        data.update(copy.deepcopy(self._map))
        # HeteroData directly, *not* through WaymoTargetBuilder. That transform
        # is what the training dataset uses, and it decides which vehicles to
        # predict by counting their valid *future* steps -- then randomly
        # subsamples the survivors. Here the future is invalid by construction,
        # because producing it is the whole job, so the transform would quietly
        # select nothing and every agent would stand still.
        return HeteroData(self._token_processor.preprocess(data)).to(self._device)

    def _agent_tensors(self):
        """Dense [agent, time] tensors from the live history, ego last.

        Laid out exactly as converter._convert_agents does, because that is the
        layout the checkpoint was trained on: WOMD's 91 steps with the ego at
        av_index. Only the first HISTORY_STEPS are filled -- the rest is what
        the model is being asked to produce.
        """
        tracks = sorted({t for _, agents in self._history for t in agents})
        if not tracks:
            return None
        self._rows = {track: index for index, track in enumerate(tracks)}
        av_index = len(tracks)
        count = len(tracks) + 1

        position = torch.zeros(count, WOMD_STEPS, 3, dtype=torch.float64)
        heading = torch.zeros(count, WOMD_STEPS, dtype=torch.float64)
        velocity = torch.zeros(count, WOMD_STEPS, 3, dtype=torch.float64)
        shape = torch.zeros(count, WOMD_STEPS, 3, dtype=torch.float64)
        valid = torch.zeros(count, WOMD_STEPS, dtype=torch.bool)
        kinds = torch.full((count,), AGENT_VEHICLE, dtype=torch.uint8)
        # 1 is an ordinary tracked agent, 3 marks the one WOMD scores. The ego
        # gets 3 because that is what the checkpoint saw at that row.
        category = torch.ones(count, dtype=torch.uint8)
        category[av_index] = 3

        for step, (ego, agents) in enumerate(self._history):
            position[av_index, step, 0] = ego.center.x
            position[av_index, step, 1] = ego.center.y
            heading[av_index, step] = ego.center.heading
            velocity[av_index, step, 0] = ego.dynamic_car_state.center_velocity_2d.x
            velocity[av_index, step, 1] = ego.dynamic_car_state.center_velocity_2d.y
            box = ego.car_footprint.oriented_box
            shape[av_index, step] = torch.tensor([box.length, box.width, box.height])
            valid[av_index, step] = True

            for track, obj in agents.items():
                row = self._rows[track]
                position[row, step, 0] = obj.box.center.x
                position[row, step, 1] = obj.box.center.y
                heading[row, step] = obj.box.center.heading
                speed = getattr(obj, 'velocity', None)
                if speed is not None:
                    velocity[row, step, 0] = speed.x
                    velocity[row, step, 1] = speed.y
                shape[row, step] = torch.tensor(
                    [obj.box.length, obj.box.width, obj.box.height])
                valid[row, step] = True
                kinds[row] = AGENT_TYPE.get(obj.tracked_object_type, 0)

        position[:, :, 0] -= self._shift[0]
        position[:, :, 1] -= self._shift[1]
        # Invalid slots must stay exactly zero: SMART's preprocessing treats a
        # non-zero position in an invalid slot as something to interpolate from,
        # and the contamination spreads to agents that were fine.
        position[~valid] = 0.0

        agent = {'num_nodes': count, 'av_index': av_index, 'valid_mask': valid,
                 'predict_mask': valid.clone(), 'id': list(tracks) + ['ego'],
                 'type': kinds, 'category': category,
                 'position': position, 'heading': heading,
                 'velocity': velocity, 'shape': shape}
        for key in ('position', 'heading', 'velocity', 'shape'):
            agent[key] = agent[key].float()

        return agent

    def _advance(self) -> None:
        """Move each agent onto the pose its rollout says it should be at.

        Where the simulation runs faster than SMART, a predicted pose covers
        several simulation steps. Holding it for all of them makes most steps a
        zero-length move: the traffic stutters, and a planner reading velocity
        or time-to-collision off consecutive frames sees agents that alternate
        between stopped and travelling at several times their real speed. It
        shows up as a median vehicle speed of exactly zero. Interpolating
        between the two poses instead costs nothing and gives smooth motion.
        At the usual 10 Hz the stride is 1 and this reduces to indexing.
        """
        elapsed = (self._iteration - self._plan_start) / self._stride
        index = int(elapsed)
        fraction = elapsed - index
        moved = {}
        for track, (traj, head) in self._plan.items():
            template = self._templates.get(track)
            if template is None or index >= len(traj):
                continue
            x, y = float(traj[index, 0]), float(traj[index, 1])
            heading = float(head[index])
            if fraction > 0.0 and index + 1 < len(traj):
                x += fraction * (float(traj[index + 1, 0]) - x)
                y += fraction * (float(traj[index + 1, 1]) - y)
                # Headings wrap, so interpolate the shorter way round rather
                # than sweeping the long way through pi.
                delta = float(head[index + 1]) - heading
                heading += fraction * math.atan2(math.sin(delta), math.cos(delta))
            pose = StateSE2(x + self._shift[0], y + self._shift[1], heading)
            previous = self._current.get(track)
            velocity = None
            if previous is not None and self._step_time > 0.0:
                velocity = StateVector2D(
                    (pose.x - previous.box.center.x) / self._step_time,
                    (pose.y - previous.box.center.y) / self._step_time)
            moved[track] = self._at_pose(template, pose, velocity)
        if moved:
            self._current = moved

    @staticmethod
    def _at_pose(original, pose: StateSE2, velocity=None):
        """A copy of `original` moved to `pose`, keeping its identity and size.

        The velocity has to be recomputed from the motion, not carried over.
        `original` is the object as first seen, so reusing its velocity pins
        every agent to whatever it was doing at t=0 for the rest of the
        scenario.

        Not for SMART's benefit -- it never reads the field. The model outputs
        positions and headings only, and derives motion from the position
        history: `agent_token_embedding` assigns `token_velocity` to a local
        and never uses it, and `inference` recomputes velocity from
        `motion_vector_a`. Feeding it a stale velocity changes nothing.

        It matters because everything downstream reads it. nuPlan's
        time-to-collision projects agents forward along their reported
        velocity, and a planner deciding whether to yield reads it too, so an
        agent that has been driving for ten seconds while reporting the
        velocity it had at t=0 is invisible to exactly the machinery this
        benchmark measures.
        """
        box = OrientedBox.from_new_pose(original.box, pose)
        if isinstance(original, Agent):
            return Agent(original.tracked_object_type, box,
                         velocity if velocity is not None else original.velocity,
                         original.metadata, original.angular_velocity)
        if isinstance(original, StaticObject):
            return StaticObject(original.tracked_object_type, box, original.metadata)
        return SceneObject(original.tracked_object_type, box, original.metadata)

    def get_observation(self) -> DetectionsTracks:
        """Inherited, see superclass.

        Simulated agents plus the log's static objects. Cones and barriers have
        no token vocabulary and do not move, so replaying them is both correct
        and cheaper than asking a model to predict that they stay put.
        """
        static = [obj for obj in
                  self._scenario.get_tracked_objects_at_iteration(
                      self._iteration).tracked_objects
                  if obj.tracked_object_type not in SIMULATED_TYPES]
        return DetectionsTracks(TrackedObjects(list(self._current.values()) + static))


def resolve_checkpoint(checkpoint_path: str) -> str:
    """Absolute path to a checkpoint given relative to the repository.

    nuPlan's runner is a hydra application, and hydra changes the working
    directory to the run's output folder before anything is built. A relative
    path in a config therefore resolves somewhere inside the results tree and
    the only clue is a FileNotFoundError with no filename in it. Relative paths
    are resolved against the repository root so a config can stay readable.
    """
    if os.path.isabs(checkpoint_path) or os.path.exists(checkpoint_path):
        return checkpoint_path
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidate = os.path.join(root, checkpoint_path)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f'SMART checkpoint not found: tried {checkpoint_path!r} from '
        f'{os.getcwd()!r} and {candidate!r}. It is not in the repository -- '
        f'fetch it from the release the paper points at, or set '
        f'SMART_CHECKPOINT to an absolute path.')


_MODEL_CACHE: Dict[tuple, object] = {}


def load_smart(checkpoint_path: str, device: str = 'cuda'):
    """Build SMART from a checkpoint, using the config stored inside it.

    Cached per (checkpoint, device) for the life of the process. nuPlan builds
    an observation per scenario, so without this a ray worker reloads the same
    85 MB checkpoint and re-uploads it to the card for every one of the ~220
    scenarios it handles. The model is read-only here -- eval mode, no_grad,
    no buffers written -- so one copy serves every scenario in the worker.

    Lightning saves the model config under `hyper_parameters`, so the
    architecture is read from the same file as the weights. Passing a separate
    yaml instead is how you end up silently loading a model whose shapes agree
    but whose token vocabulary does not.
    """
    from smart.model import SMART

    checkpoint_path = resolve_checkpoint(checkpoint_path)
    cached = _MODEL_CACHE.get((checkpoint_path, device))
    if cached is not None:
        return cached
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = SMART(checkpoint['hyper_parameters']['model_config'])
    missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f'checkpoint does not match the model it declares: {len(missing)} '
            f'missing and {len(unexpected)} unexpected parameters')
    model = model.eval().to(device)
    _MODEL_CACHE[(checkpoint_path, device)] = model
    return model


def build_smart_agents(scenario, checkpoint_path: str, device: str = 'cuda',
                       **kwargs) -> SMARTAgents:
    """Hydra entry point: load SMART, then wrap it as an observation.

    nuPlan builds observations with `instantiate(cfg, scenario=scenario)`, so
    the scenario arrives as a keyword and the model has to be constructed here.
    """
    return SMARTAgents(load_smart(checkpoint_path, device), scenario,
                       device=device, **kwargs)
