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
from typing import Dict, List, Optional, Type

import torch

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObject
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation

from torch_geometric.data import HeteroData

from smart.datasets.preprocess import TokenProcessor
from smart.nuplan.converter import (AGENT_TYPE, AGENT_VEHICLE, NUPLAN_STRIDE, WOMD_CURRENT_STEP,
                                    WOMD_STEPS, _convert_map)

# SMART's history window: the current step plus everything before it.
HISTORY_STEPS = WOMD_CURRENT_STEP + 1

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
        replan_steps: simulation steps between rollouts. At 20 Hz, 10 is half a
            second of predicted motion consumed per inference.
        map_radius: metres of map around the starting ego handed to the model.
        seed: makes the rollout reproducible, and identical between the
            occluded and unoccluded runs of the same scenario.
    """

    def __init__(self, model, scenario, replan_steps: int = 10,
                 map_radius: float = 150.0, seed: int = 0, device: str = 'cuda'):
        self._model = model
        self._scenario = scenario
        self._replan_steps = replan_steps
        self._map_radius = map_radius
        self._seed = seed
        self._device = device
        self._token_processor = TokenProcessor(2048)

        self._map: Optional[Dict] = None
        self._origin = (0.0, 0.0)
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
        self._map = self._build_map()

        # Seed the window from the scenario's *past*. Before the first step
        # there is no simulated history, and SMART needs a full window to
        # condition on -- but it has to be the past. Reading forward from
        # iteration 0 instead would open the simulation with a model that has
        # already been shown two seconds of the ego's logged future and has
        # started the traffic from where the log ends up, which is the kind of
        # leak that makes a closed-loop number meaningless.
        step_time = getattr(self._scenario, 'database_interval', 0.05)
        horizon = (HISTORY_STEPS - 1) * NUPLAN_STRIDE * step_time
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
        return _convert_map(self._scenario, Point2D(*self._origin),
                            self._map_radius, True, self._origin)

    def reset(self) -> None:
        """Inherited, see superclass."""
        self._iteration = 0
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

        # The history SMART conditions on is at 10 Hz, so only every second
        # simulation step contributes -- and the ego written in is the simulated
        # one, which is the whole point of a reactive model.
        if iteration.index % NUPLAN_STRIDE == 0:
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

        position[:, :, 0] -= self._origin[0]
        position[:, :, 1] -= self._origin[1]
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
        """Move each agent onto the pose its rollout says it should be at."""
        offset = (self._iteration - self._plan_start) // NUPLAN_STRIDE
        moved = {}
        for track, (traj, head) in self._plan.items():
            template = self._templates.get(track)
            if template is None or offset >= len(traj):
                continue
            pose = StateSE2(float(traj[offset, 0]) + self._origin[0],
                            float(traj[offset, 1]) + self._origin[1],
                            float(head[offset]))
            moved[track] = self._at_pose(template, pose)
        if moved:
            self._current = moved

    @staticmethod
    def _at_pose(original, pose: StateSE2):
        """A copy of `original` moved to `pose`, keeping its identity and size."""
        box = OrientedBox.from_new_pose(original.box, pose)
        if isinstance(original, Agent):
            return Agent(original.tracked_object_type, box, original.velocity,
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


def load_smart(checkpoint_path: str, device: str = 'cuda'):
    """Build SMART from a checkpoint, using the config stored inside it.

    Lightning saves the model config under `hyper_parameters`, so the
    architecture is read from the same file as the weights. Passing a separate
    yaml instead is how you end up silently loading a model whose shapes agree
    but whose token vocabulary does not.
    """
    from smart.model import SMART

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = SMART(checkpoint['hyper_parameters']['model_config'])
    missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f'checkpoint does not match the model it declares: {len(missing)} '
            f'missing and {len(unexpected)} unexpected parameters')
    return model.eval().to(device)


def build_smart_agents(scenario, checkpoint_path: str, device: str = 'cuda',
                       **kwargs) -> SMARTAgents:
    """Hydra entry point: load SMART, then wrap it as an observation.

    nuPlan builds observations with `instantiate(cfg, scenario=scenario)`, so
    the scenario arrives as a keyword and the model has to be constructed here.
    """
    return SMARTAgents(load_smart(checkpoint_path, device), scenario,
                       device=device, **kwargs)
