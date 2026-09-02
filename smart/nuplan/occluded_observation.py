"""A nuPlan observation that hides what the ego has no line of sight to.

Every closed-loop planning benchmark hands the planner ground-truth boxes for
every agent in the scene. This wraps any nuPlan observation and takes that away,
leaving the planner with what the ego could actually see: agents in direct line
of sight, plus agents it saw recently enough to still be tracking.

The wrapper is deliberately thin. Sight lines come from smart.occlusion
.visibility, which is parameter-free geometry, and memory from smart.occlusion
.tracking, which is not nuPlan-specific and is tested on synthetic sequences.
What lives here is only the adaptation between those and nuPlan's types.

Three choices are worth stating plainly, because they are assumptions the
benchmark imposes rather than facts about the world:

- Only vehicles occlude, by default. nuPlan also tracks barriers, cones and
  czone signs, and a construction barrier really does block a sight line -- but
  the visibility model is 2D and has no notion of height, so admitting 0.4 m
  roadside objects as occluders would hide cars behind traffic cones. Vehicles
  are the defensible default; the rest is left to an ablation.

- The ego pose used to cast sight lines is one simulation step stale. nuPlan
  updates observations before the ego state for that step reaches the history
  buffer, so the freshest ego an observation can see is the previous one. The
  devkit's own IDMAgents has the same constraint and resolves it the same way.
  At 0.1 s steps this is around a metre.

- Objects the ego can currently see are passed through untouched, identity and
  metadata intact. Only remembered ones are rebuilt, at the pose the tracking
  buffer propagated them to, keeping their original stale timestamp -- which is
  the honest thing for a measurement that is no longer fresh.
"""
import random
from typing import Optional, Sequence, Type

import torch

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObject
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation

from smart.occlusion.tracking import CONSTANT_VELOCITY, TrackObservation, TrackingBuffer
from smart.occlusion.visibility import agent_visibility

DEFAULT_OCCLUDER_TYPES = (TrackedObjectType.VEHICLE,)


def _track_id(tracked_object) -> str:
    """Stable identity for an object across frames."""
    return str(tracked_object.track_token or tracked_object.token)


def _velocity(tracked_object):
    """Velocity of an object, or zero for the static ones that have none."""
    velocity = getattr(tracked_object, 'velocity', None)
    return (velocity.x, velocity.y) if velocity is not None else (0.0, 0.0)


class OccludedObservation(AbstractObservation):
    """Filters another observation down to what the ego can see or remembers.

    Args:
        observation: the observation to wrap -- log replay, IDM agents, or a
            learned traffic model. Occlusion is applied on top of whatever it
            produces, so the two concerns stay separable.
        scenario: used only to seed the ego pose before the first simulation
            step; afterwards the ego comes from the simulated history, so this
            stays correct in closed loop where the ego leaves the logged route.
        occluder_types: which object types block a sight line.
        memory_horizon: seconds an unseen track survives. See
            smart.occlusion.tracking for why this is a declared assumption.
        propagate: how an unseen track is carried forward.
        radius: optional range limit in metres. Left off by default so the
            benchmark does not silently impose a sensor range on top of the
            geometry; set it to model one deliberately.
        visibility_threshold: fraction of an object's extent that must be
            visible for it to count as seen. Zero means any sliver counts.
        randomise: withhold the same *number* of objects the sight lines would
            have hidden, but choose them uniformly at random. This is the
            control, not a second benchmark: it removes exactly as much
            information per frame, so anything the occluded condition does that
            this one does not is attributable to *which* objects go missing
            rather than to how many. Without it, a planner that simply degrades
            under any perturbation looks the same as one that is specifically
            defeated by things hiding behind other things.
        seed: makes the random choice reproducible.
    """

    def __init__(self,
                 observation: AbstractObservation,
                 scenario,
                 occluder_types: Sequence[TrackedObjectType] = DEFAULT_OCCLUDER_TYPES,
                 memory_horizon: float = 3.0,
                 propagate: str = CONSTANT_VELOCITY,
                 radius: Optional[float] = None,
                 visibility_threshold: float = 0.0,
                 randomise: bool = False,
                 seed: int = 0):
        self._observation = observation
        self._scenario = scenario
        self._occluder_types = set(occluder_types)
        self._radius = radius
        self._visibility_threshold = visibility_threshold
        self._randomise = randomise
        self._random = random.Random(seed)
        self._seed = seed
        self._buffer = TrackingBuffer(memory_horizon=memory_horizon, propagate=propagate)
        self._ego_state = None
        self._time_s = 0.0
        self._last_seen = {}
        self._cached = None
        self._cached_time = None

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks

    def initialize(self) -> None:
        """Inherited, see superclass."""
        self._observation.initialize()
        self.reset()
        self._ego_state = self._scenario.get_ego_state_at_iteration(0)
        self._time_s = self._ego_state.time_point.time_s

    def reset(self) -> None:
        """Inherited, see superclass."""
        self._observation.reset()
        self._buffer.reset()
        # Reseed so a scenario draws the same sequence however many scenarios
        # ran before it, which keeps a rerun of one scenario reproducible.
        self._random = random.Random(self._seed)
        self._ego_state = None
        self._time_s = 0.0
        self._last_seen = {}
        self._cached = None
        self._cached_time = None

    def update_observation(self, iteration, next_iteration, history) -> None:
        """Inherited, see superclass."""
        self._observation.update_observation(iteration, next_iteration, history)
        self._ego_state, _ = history.current_state
        self._time_s = next_iteration.time_s
        self._cached = None

    def get_observation(self) -> DetectionsTracks:
        """The visible and remembered part of the wrapped observation.

        The simulation may ask more than once per step, so the filtered result
        is cached: folding the same step into the tracking buffer twice would
        age every remembered track by an extra step.
        """
        if self._cached is None or self._cached_time != self._time_s:
            self._cached = self._occlude(self._observation.get_observation())
            self._cached_time = self._time_s
        return self._cached

    def _occlude(self, detections: DetectionsTracks) -> DetectionsTracks:
        objects = list(detections.tracked_objects)
        if self._ego_state is None:
            raise RuntimeError('OccludedObservation.initialize() was never called')
        if not objects:
            self._buffer.update(self._time_s, [])
            return DetectionsTracks(TrackedObjects([]))

        origin = torch.tensor([self._ego_state.center.x, self._ego_state.center.y])
        boxes = torch.tensor([[o.box.center.x, o.box.center.y, o.box.center.heading,
                               o.box.width, o.box.length] for o in objects])

        within = ((boxes[:, :2] - origin).norm(dim=-1) <= self._radius
                  if self._radius is not None
                  else torch.ones(len(objects), dtype=torch.bool))
        occluders = torch.tensor([o.tracked_object_type in self._occluder_types
                                  for o in objects]) & within
        fraction = agent_visibility(origin, boxes, occluder_mask=occluders)

        hidden = [i for i in range(len(objects))
                  if within[i] and fraction[i] <= self._visibility_threshold]
        if self._randomise:
            # Same count, different choice. Drawing from the in-range objects
            # keeps the two conditions comparable: occlusion never withholds
            # something out of range either.
            candidates = [i for i in range(len(objects)) if within[i]]
            hidden = self._random.sample(candidates, min(len(hidden), len(candidates)))
        withheld = set(hidden)

        seen = []
        for i, tracked_object in enumerate(objects):
            if not within[i] or i in withheld:
                continue
            track_id = _track_id(tracked_object)
            self._last_seen[track_id] = tracked_object
            velocity_x, velocity_y = _velocity(tracked_object)
            seen.append(TrackObservation(
                track_id=track_id,
                x=float(boxes[i, 0]), y=float(boxes[i, 1]), heading=float(boxes[i, 2]),
                velocity_x=velocity_x, velocity_y=velocity_y,
                width=float(boxes[i, 3]), length=float(boxes[i, 4])))

        estimates = self._buffer.update(self._time_s, seen)

        # Memory may only bridge an occlusion gap, never a detection-range one.
        # The underlying observation drops an object once it leaves the scene,
        # and a buffer cannot tell that apart from the object being hidden -- so
        # left alone it hands the planner ghosts, and the occluded condition
        # ends up with *more* information than the fully observable one it is
        # being compared against. Restricting the output to objects the base
        # observation still reports keeps it a strict subset, which is the
        # invariant the whole comparison rests on.
        present = {_track_id(o) for o in objects}
        estimates = [e for e in estimates if e.track_id in present]

        believed = []
        for estimate in estimates:
            original = self._last_seen[estimate.track_id]
            believed.append(original if estimate.observed
                            else self._at_new_pose(original, estimate))
        self._last_seen = {estimate.track_id: self._last_seen[estimate.track_id]
                           for estimate in estimates}
        return DetectionsTracks(TrackedObjects(believed))

    def _at_new_pose(self, original, estimate):
        """Rebuild a remembered object where the buffer believes it now is."""
        box = OrientedBox.from_new_pose(
            original.box, StateSE2(estimate.x, estimate.y, estimate.heading))
        if isinstance(original, Agent):
            return Agent(original.tracked_object_type, box, original.velocity,
                         original.metadata, original.angular_velocity)
        if isinstance(original, StaticObject):
            return StaticObject(original.tracked_object_type, box, original.metadata)
        return SceneObject(original.tracked_object_type, box, original.metadata)


def build_occluded_observation(observation, scenario, **kwargs):
    """Hydra entry point for nuPlan's own simulation runner.

    nuPlan builds observations with `instantiate(cfg, scenario=scenario)`, so
    the scenario arrives as a keyword and any nested observation in the config
    never receives one. Rather than teach OccludedObservation about hydra, this
    factory instantiates the wrapped observation with the scenario and hands
    both to the wrapper. Configure it with `_recursive_: false` so the inner
    node arrives here as config rather than as a half-built object.

    Args:
        observation: config of the observation to wrap, e.g. TracksObservation.
        scenario: injected by nuPlan's observation builder.
        **kwargs: passed to OccludedObservation.
    """
    from hydra.utils import instantiate

    wrapped = instantiate(observation, scenario=scenario)
    return OccludedObservation(wrapped, scenario, **kwargs)
