"""Ego collisions classified the way nuPlan classifies them.

Counting the fraction of timesteps in which the ego overlaps something is not
what nuPlan measures, and using it makes an agent model look far worse than the
official metric would. Two differences matter:

- nuPlan counts a *new* collision per track. Once the ego has touched a given
  vehicle that pair is remembered and not counted again, so two cars locked
  together for eight seconds is one collision, not eighty.

- nuPlan only holds the ego responsible for some of them. Being rear-ended, or
  being hit while stopped, is not a planner failure, and
  `no_ego_at_fault_collisions` excludes both. Only driving into something --
  front-bumper contact, or hitting a stopped track -- counts against the ego.

Both rules are reproduced here from
nuplan/planning/metrics/evaluation_metrics/common/no_ego_at_fault_collisions.py
so the same taxonomy applies to trajectories held as tensors, without building
a full SimulationHistory.

One narrowing is deliberate. nuPlan also counts a lateral collision as at fault
when the ego was straddling lanes or outside the drivable area, which needs
route and roadblock lookups this does not do. Lateral collisions are reported
separately rather than folded in, so the at-fault count here is a lower bound
and is labelled as one.
"""
import math

import torch

from smart.metrics.collision import boxes_overlap
from smart.occlusion.visibility import boxes_to_corners, segments_intersect

STOPPED_SPEED = 5e-2
BEHIND_ANGLE = math.radians(150.0)

STOPPED_EGO, STOPPED_TRACK, ACTIVE_FRONT, ACTIVE_REAR, ACTIVE_LATERAL = range(5)
NAMES = ['stopped ego', 'stopped track', 'active front', 'active rear', 'active lateral']
# nuplan: ACTIVE_FRONT_COLLISION and STOPPED_TRACK_COLLISION are always at
# fault; ACTIVE_LATERAL_COLLISION is too, but only off-route, which is not
# reproduced here.
AT_FAULT = (STOPPED_TRACK, ACTIVE_FRONT)


def _speed(position, valid):
    """Per-step speed, holding the first step's value at the start."""
    steps = (position[:, 1:] - position[:, :-1]).norm(dim=-1) / 0.1
    both = valid[:, 1:] & valid[:, :-1]
    speed = torch.zeros_like(valid, dtype=torch.float32)
    speed[:, 1:] = steps * both
    speed[:, 0] = speed[:, 1]
    return speed


def _classify(ego_box, ego_speed, track_box, track_speed):
    """nuPlan's collision taxonomy for one ego/track pair."""
    if ego_speed <= STOPPED_SPEED:
        return STOPPED_EGO
    if track_speed <= STOPPED_SPEED:
        return STOPPED_TRACK

    # Behind is measured from the ego's heading to the bearing of the track.
    offset = track_box[:2] - ego_box[:2]
    bearing = torch.atan2(offset[1], offset[0])
    relative = torch.remainder(bearing - ego_box[2] + math.pi, 2 * math.pi) - math.pi
    if abs(float(relative)) > BEHIND_ANGLE:
        return ACTIVE_REAR

    # Front-bumper contact: the segment joining the ego's two front corners.
    ego_corners = boxes_to_corners(*ego_box[:2], ego_box[2], ego_box[3], ego_box[4])
    track_corners = boxes_to_corners(*track_box[:2], track_box[2],
                                     track_box[3], track_box[4])
    bumper_a, bumper_b = ego_corners[0], ego_corners[1]
    edges = torch.stack([track_corners, track_corners.roll(-1, dims=0)], dim=1)
    crosses = bool(segments_intersect(bumper_a.expand(4, 2), bumper_b.expand(4, 2),
                                      edges[:, 0], edges[:, 1]).any())
    # shapely's `intersects` is true for containment too, and a bumper can lie
    # wholly inside a larger box without crossing any of its edges -- which is
    # exactly what a car driving into the side of a lorry looks like.
    inside = _point_in_box(bumper_a, track_box) or _point_in_box(bumper_b, track_box)
    return ACTIVE_FRONT if crosses or inside else ACTIVE_LATERAL


def _point_in_box(point, box):
    """Whether a point lies within an oriented box."""
    offset = point - box[:2]
    cos, sin = torch.cos(box[2]), torch.sin(box[2])
    longitudinal = offset[0] * cos + offset[1] * sin
    lateral = -offset[0] * sin + offset[1] * cos
    return bool(abs(longitudinal) <= 0.5 * box[4] and abs(lateral) <= 0.5 * box[3])


def ego_collisions(position, heading, shape, valid, ego):
    """Classify every new ego collision over a rollout.

    Args:
        position: (N, T, 2), heading: (N, T), shape: (N, 3) as
            (length, width, height), valid: (N, T) bool, ego: row index.

    Returns:
        (counts, events) where counts is a list of five ints indexed by
        collision type and events is a list of (step, track, type).
    """
    counts = [0] * 5
    events = []
    speed = _speed(position, valid)
    seen = set()

    for t in range(position.shape[1]):
        if not bool(valid[ego, t]):
            continue
        ego_box = torch.tensor([position[ego, t, 0], position[ego, t, 1],
                                heading[ego, t], shape[ego, 1], shape[ego, 0]])
        others = (valid[:, t].clone().index_fill_(
            0, torch.tensor([ego]), False)).nonzero(as_tuple=True)[0]
        others = torch.tensor([i for i in others.tolist() if i not in seen])
        if not len(others):
            continue

        boxes = torch.stack([position[others, t, 0], position[others, t, 1],
                             heading[others, t], shape[others, 1],
                             shape[others, 0]], dim=-1)
        hit = boxes_overlap(ego_box.expand(len(others), 5), boxes)
        for index in hit.nonzero(as_tuple=True)[0].tolist():
            track = int(others[index])
            seen.add(track)
            kind = _classify(ego_box, float(speed[ego, t]),
                             boxes[index], float(speed[track, t]))
            counts[kind] += 1
            events.append((t, track, kind))
    return counts, events


def at_fault(counts):
    """Collisions nuPlan would hold the ego responsible for."""
    return sum(counts[kind] for kind in AT_FAULT)
