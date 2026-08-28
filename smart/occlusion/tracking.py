"""Object permanence for a planner that cannot see everything.

Dropping an occluded agent from the observation the instant its sight line
breaks is not what perception does, and a planner fed that signal sees agents
flicker in and out of existence. It would then be judged on an interface
artefact rather than on how it handles occlusion. Real trackers remember: they
keep propagating a track for a while after losing sight of it, and only give up
once the estimate has gone stale.

This is that memory, kept deliberately free of any nuPlan types so the policy
can be tested on synthetic sequences. The nuPlan side adapts its objects into
`TrackObservation` and reads `TrackEstimate` back out.

Two things here are perception *assumptions* the benchmark imposes on every
planner, not physics, and both must be reported alongside any result:

- `memory_horizon`: how long a track survives unseen. This is the one free
  parameter in the occlusion stack -- the geometry in visibility.py has none.
- `propagate`: whether an unseen track coasts at its last velocity or freezes.
  Real trackers coast, which is the default, but freezing is kept because the
  difference between the two is a measurable claim about how much a planner
  leans on remembered motion.

A track is only ever remembered if it was seen at least once. Nothing here
invents objects the ego never observed; reasoning about what might be hiding in
an unobserved region is a planner's job, not the benchmark's.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence

HOLD = 'hold'
CONSTANT_VELOCITY = 'constant_velocity'


@dataclass(frozen=True)
class TrackObservation:
    """One currently-visible object, as handed in by the visibility layer."""

    track_id: str
    x: float
    y: float
    heading: float
    velocity_x: float
    velocity_y: float
    width: float
    length: float


@dataclass(frozen=True)
class TrackEstimate:
    """What the planner is told about one object.

    `observed` distinguishes a live measurement from a remembered one, and
    `time_since_observed` says how stale a remembered one is. A planner that
    ignores both fields sees exactly the flicker-free track list it would have
    got from a conventional benchmark; one that reads them can be appropriately
    unsure about what it can no longer see.
    """

    track_id: str
    x: float
    y: float
    heading: float
    velocity_x: float
    velocity_y: float
    width: float
    length: float
    observed: bool
    time_since_observed: float


class TrackingBuffer:
    """Carries tracks forward across occlusion gaps.

    Args:
        memory_horizon: seconds a track survives without being seen. A track
            unseen for longer is forgotten, and would have to be re-observed to
            come back.
        propagate: `CONSTANT_VELOCITY` coasts an unseen track at its last
            observed velocity; `HOLD` leaves it where it was last seen.
    """

    def __init__(self,
                 memory_horizon: float = 3.0,
                 propagate: str = CONSTANT_VELOCITY):
        if memory_horizon < 0:
            raise ValueError('memory_horizon must be non-negative')
        if propagate not in (HOLD, CONSTANT_VELOCITY):
            raise ValueError(f'unknown propagate mode: {propagate!r}')
        self.memory_horizon = memory_horizon
        self.propagate = propagate
        self._last_seen: Dict[str, float] = {}
        self._state: Dict[str, TrackObservation] = {}

    def reset(self) -> None:
        """Forget everything. Call between scenarios."""
        self._last_seen.clear()
        self._state.clear()

    def update(self,
               timestamp: float,
               observations: Sequence[TrackObservation]) -> List[TrackEstimate]:
        """Fold this step's visible objects in and report the believed world.

        Args:
            timestamp: current time in seconds; must not go backwards.
            observations: the objects visible right now. Anything absent from
                this list is treated as unseen this step, whether it is occluded
                or has left the scene -- the buffer cannot tell those apart, and
                neither can a real tracker.

        Returns:
            Estimates for every track still remembered, ordered by track id so
            the output does not depend on the order observations arrived in.
        """
        for observation in observations:
            self._state[observation.track_id] = observation
            self._last_seen[observation.track_id] = timestamp

        estimates = []
        for track_id in sorted(self._state):
            elapsed = timestamp - self._last_seen[track_id]
            if elapsed > self.memory_horizon:
                continue
            estimates.append(self._estimate(self._state[track_id], elapsed))

        self._forget(timestamp)
        return estimates

    def _estimate(self, state: TrackObservation, elapsed: float) -> TrackEstimate:
        dt = elapsed if self.propagate == CONSTANT_VELOCITY else 0.0
        return TrackEstimate(
            track_id=state.track_id,
            x=state.x + state.velocity_x * dt,
            y=state.y + state.velocity_y * dt,
            heading=state.heading,
            velocity_x=state.velocity_x,
            velocity_y=state.velocity_y,
            width=state.width,
            length=state.length,
            observed=elapsed == 0.0,
            time_since_observed=elapsed,
        )

    def _forget(self, timestamp: float) -> None:
        stale = [track_id for track_id, seen in self._last_seen.items()
                 if timestamp - seen > self.memory_horizon]
        for track_id in stale:
            del self._last_seen[track_id]
            del self._state[track_id]
