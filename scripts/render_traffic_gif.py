"""Animate the same scenario under different traffic models, side by side.

render_nuplan_gif.py answers a different question -- what the planner was and
was not shown -- and colours agents by whether they reached it. Here the
observation is the thing under test: log replay is the recorded truth, IDM is
what nuPlan's reactive challenge uses, and SMART is the learned model. What
matters is whether the learned traffic looks like driving: cars on lanes, in
their lane direction, at plausible speeds, not sliding sideways or drifting
into buildings.

Judgement still has to be a person's, so this draws the lanes underneath and
marks agents whose motion is inconsistent with their own heading, which is the
failure the numbers hide.

Usage:
    PYTHONPATH=. python scripts/render_traffic_gif.py \
        --runs exp_local/exp/simulation/.../baseline/2026-... \
               exp_local/exp/simulation/.../smart_agents_observation/2026-... \
        --labels "log replay" "SMART" --out-dir traffic_gifs
"""
import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.simulation_log import SimulationLog

from smart.occlusion.visibility import boxes_to_corners

EGO = '#d62728'
VEHICLE = '#4c72b0'
PEDESTRIAN = '#dd8452'
STATIC = '#b0b0b0'
# An agent moving in a direction its own heading does not support is the
# giveaway for a traffic model that has come off the rails, and it is invisible
# in any aggregate.
SIDEWAYS = '#000000'
LANE = '#d0d0d0'


def find_logs(experiment_dir):
    pattern = os.path.join(experiment_dir, 'simulation_log', '**', '*.msgpack.xz')
    return {os.path.basename(p).split('.')[0]: p
            for p in glob.glob(pattern, recursive=True)}


def lane_paths(scenario, centre, radius):
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    try:
        nearby = scenario.map_api.get_proximal_map_objects(centre, radius, layers)
    except Exception:
        return []
    paths = []
    for layer in layers:
        for lane in nearby.get(layer, []):
            try:
                path = lane.baseline_path.discrete_path
            except Exception:
                continue
            paths.append([(s.x, s.y) for s in path])
    return paths


def frame_state(sample):
    """Boxes, types and ids for one frame of a simulation history."""
    objects = list(sample.observation.tracked_objects)
    if not objects:
        return torch.zeros(0, 5), [], []
    boxes = torch.tensor([[o.box.center.x, o.box.center.y, o.box.center.heading,
                           o.box.width, o.box.length] for o in objects])
    ids = [o.track_token or o.token for o in objects]
    return boxes, ids, [o.tracked_object_type for o in objects]


def sideways(boxes, ids, previous, threshold=0.25):
    """Ids whose displacement is mostly perpendicular to their own heading.

    A car that translates sideways faster than it goes forward is not driving.
    The threshold ignores anything nearly stationary, where heading is noise.
    """
    bad = set()
    for i, track in enumerate(ids):
        if track not in previous:
            continue
        dx = float(boxes[i, 0]) - previous[track][0]
        dy = float(boxes[i, 1]) - previous[track][1]
        step = (dx * dx + dy * dy) ** 0.5
        if step < threshold:
            continue
        heading = float(boxes[i, 2])
        forward = dx * np.cos(heading) + dy * np.sin(heading)
        lateral = abs(-dx * np.sin(heading) + dy * np.cos(heading))
        if lateral > abs(forward):
            bad.add(track)
    return bad


def render(logs, labels, out_path, radius, fps, stride):
    histories = [log.simulation_history.data for log in logs]
    frames = list(range(0, min(len(h) for h in histories), stride))
    if not frames:
        return None, {}

    ego0 = histories[0][0].ego_state.center
    paths = lane_paths(logs[0].scenario, Point2D(ego0.x, ego0.y), radius * 2.5)

    figure, axes = plt.subplots(1, len(logs), figsize=(6.4 * len(logs), 6.6), dpi=85)
    axes = np.atleast_1d(axes)
    previous = [{} for _ in logs]
    counts = [0 for _ in logs]

    def draw(index):
        for panel, (ax, history, label) in enumerate(zip(axes, histories, labels)):
            ax.clear()
            sample = history[index]
            ego = sample.ego_state
            for path in paths:
                ax.plot([p[0] for p in path], [p[1] for p in path],
                        color=LANE, linewidth=0.6, zorder=0)

            boxes, ids, kinds = frame_state(sample)
            odd = sideways(boxes, ids, previous[panel]) if index else set()
            counts[panel] += len(odd)
            if len(boxes):
                corners = boxes_to_corners(boxes[:, 0], boxes[:, 1], boxes[:, 2],
                                           boxes[:, 3], boxes[:, 4])
                near = (boxes[:, :2] - torch.tensor([ego.center.x, ego.center.y])
                        ).norm(dim=-1) <= radius
                for i in range(len(boxes)):
                    if not near[i]:
                        continue
                    colour = (SIDEWAYS if ids[i] in odd
                              else VEHICLE if kinds[i] == TrackedObjectType.VEHICLE
                              else PEDESTRIAN if kinds[i] == TrackedObjectType.PEDESTRIAN
                              else STATIC)
                    ax.add_patch(Polygon(corners[i].numpy(), closed=True,
                                         facecolor=colour, edgecolor='#333333',
                                         linewidth=0.4, alpha=0.85, zorder=2))
                previous[panel] = {t: (float(boxes[i, 0]), float(boxes[i, 1]))
                                   for i, t in enumerate(ids)}

            ego_box = torch.tensor([[ego.center.x, ego.center.y, ego.center.heading,
                                     ego.car_footprint.width, ego.car_footprint.length]])
            ec = boxes_to_corners(ego_box[:, 0], ego_box[:, 1], ego_box[:, 2],
                                  ego_box[:, 3], ego_box[:, 4])[0]
            ax.add_patch(Polygon(ec.numpy(), closed=True, facecolor=EGO,
                                 edgecolor='white', linewidth=1.0, zorder=3))

            moving = int((boxes[:, :2].shape[0] if len(boxes) else 0))
            ax.set_title(f'{label}   {moving} agents   '
                         f'{len(odd)} 侧滑   t={index * 0.1:.1f}s', fontsize=10)
            ax.set_xlim(ego.center.x - radius, ego.center.x + radius)
            ax.set_ylim(ego.center.y - radius, ego.center.y + radius)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])

    animation = FuncAnimation(figure, draw, frames=frames, interval=1000 / fps)
    animation.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    return len(frames), dict(zip(labels, counts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True, help='experiment dirs')
    parser.add_argument('--labels', nargs='+', default=None)
    parser.add_argument('--out-dir', default='traffic_gifs')
    parser.add_argument('--scenarios', type=int, default=3)
    parser.add_argument('--radius', type=float, default=55.0)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--stride', type=int, default=2)
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(r.rstrip('/')) for r in args.runs]
    found = [find_logs(r) for r in args.runs]
    shared = set(found[0])
    for f in found[1:]:
        shared &= set(f)
    if not shared:
        raise SystemExit('the runs have no scenario tokens in common')

    os.makedirs(args.out_dir, exist_ok=True)
    for token in sorted(shared)[:args.scenarios]:
        logs = [SimulationLog.load_data(Path(f[token])) for f in found]
        out = os.path.join(args.out_dir, f'{token}.gif')
        count, odd = render(logs, labels, out, args.radius, args.fps, args.stride)
        summary = '  '.join(f'{k}: {v} 次侧滑' for k, v in odd.items())
        print(f'{token}: {count} frames -> {out}   {summary}')


if __name__ == '__main__':
    main()
