"""Animate what a nuPlan planner was and was not shown.

nuBoard is the devkit's own visualisation and it works on these runs already --
but it draws whatever observation it is given and has no notion of occlusion, so
side by side the occluded run simply has fewer cars in it and never says why.
Its frame-by-frame video export is a nuBoard button behind selenium and headless
Chrome, which needs a browser installed.

This animates the same simulation logs with the one thing nuBoard cannot show:
the sight lines. Agents are coloured by whether the planner actually received
them, and every occluding vehicle casts its shadow, so a withheld agent can be
checked against the wedge that hides it rather than taken on trust. The drawing
reuses the bird's-eye renderer already in this repo.

It reads the two runs' `simulation_log` output, so the frames are the real
closed-loop simulations that produced the metrics, not a replay staged for the
picture.

Usage:
    PYTHONPATH=. python scripts/render_nuplan_gif.py \
        --baseline /path/to/exp/.../2026.08.31.15.46.32 \
        --occluded /path/to/exp/.../2026.08.31.15.49.59 \
        --out-dir /tmp/occlusion_gifs
"""
import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.simulation_log import SimulationLog

from scripts.render_occlusion import EGO, HIDDEN, PARTIAL, VISIBLE, shadow_polygon
from smart.occlusion.visibility import boxes_to_corners

WITHHELD = HIDDEN
SEEN = VISIBLE
# Remembered objects are fully occluded right now -- the planner has them only
# because the tracking buffer is still carrying a stale measurement. Painting
# them the same green as directly visible agents makes the picture look wrong:
# cars sitting in an obvious shadow appear to have been handed over. They were,
# but from memory, and that distinction is the whole point of the buffer.
REMEMBERED = PARTIAL
LANE_COLOUR = '#c8c8c8'


def find_logs(experiment_dir):
    """Simulation logs in one experiment run, keyed by scenario token."""
    pattern = os.path.join(experiment_dir, 'simulation_log', '**', '*.msgpack.xz')
    return {os.path.basename(path).split('.')[0]: path
            for path in glob.glob(pattern, recursive=True)}


def track_ids(detections):
    return {o.track_token or o.token for o in detections.tracked_objects}


def boxes_of(detections):
    """(N, 5) boxes and the matching track ids, in one frame."""
    objects = list(detections.tracked_objects)
    if not objects:
        return torch.zeros(0, 5), []
    boxes = torch.tensor([[o.box.center.x, o.box.center.y, o.box.center.heading,
                           o.box.width, o.box.length] for o in objects])
    return boxes, [o.track_token or o.token for o in objects]


def lane_paths(scenario, centre, radius):
    """Lane centrelines near the scenario, queried once and drawn statically."""
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
            paths.append([(state.x, state.y) for state in path])
    return paths


def render(baseline_log, occluded_log, out_path, radius, fps, stride):
    """Write one GIF comparing what was there against what was shown."""
    baseline = baseline_log.simulation_history.data
    occluded = occluded_log.simulation_history.data
    frames = list(range(0, min(len(baseline), len(occluded)), stride))
    if not frames:
        return None

    ego0 = baseline[0].ego_state.center
    from nuplan.common.actor_state.state_representation import Point2D
    paths = lane_paths(baseline_log.scenario, Point2D(ego0.x, ego0.y), radius * 3)

    figure, ax = plt.subplots(figsize=(7.5, 7.5), dpi=90)

    def draw(index):
        ax.clear()
        sample = baseline[index]
        ego = sample.ego_state
        # Sight lines are cast from the pose the wrapper actually used, which
        # is the previous step's: nuPlan updates observations before the ego
        # state for the step reaches the history buffer. Drawing from the
        # current pose instead leaves about 1.4% of agents contradicting their
        # own shadow at the boundary -- measured, and exactly zero once the
        # right pose is used. The half-metre offset is invisible at this scale,
        # and the ego box itself is still drawn where the ego really is.
        viewpoint = baseline[max(index - 1, 0)].ego_state.center
        origin = torch.tensor([viewpoint.x, viewpoint.y])

        for path in paths:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, color=LANE_COLOUR, linewidth=0.6, zorder=0)

        boxes, ids = boxes_of(sample.observation)
        given = track_ids(occluded[index].observation)
        # An object the wrapper rebuilt from memory keeps its original, stale
        # timestamp; one that was actually seen carries the current frame's.
        now = max((o.metadata.timestamp_us
                   for o in occluded[index].observation.tracked_objects), default=0)
        fresh = {o.track_token or o.token
                 for o in occluded[index].observation.tracked_objects
                 if o.metadata.timestamp_us == now}

        if len(boxes):
            near = ((boxes[:, :2] - origin).norm(dim=-1) <= radius)
            corners = boxes_to_corners(boxes[:, 0], boxes[:, 1], boxes[:, 2],
                                       boxes[:, 3], boxes[:, 4])
            # Shadows come from the vehicles, which is what the wrapper treats
            # as occluders; drawing them is what makes a withheld agent checkable
            # rather than something to take on trust.
            vehicles = [i for i, o in enumerate(sample.observation.tracked_objects)
                        if o.tracked_object_type == TrackedObjectType.VEHICLE]
            for i in vehicles:
                if not near[i]:
                    continue
                ax.add_patch(Polygon(shadow_polygon(origin, corners[i], radius * 2.2),
                                     closed=True, facecolor='#000000', alpha=0.10,
                                     edgecolor='none', zorder=1))

            for i in range(len(boxes)):
                if not near[i]:
                    continue
                colour = (SEEN if ids[i] in fresh
                          else REMEMBERED if ids[i] in given else WITHHELD)
                ax.add_patch(Polygon(corners[i].numpy(), closed=True,
                                     facecolor=colour, edgecolor='#333333',
                                     linewidth=0.4, alpha=0.85, zorder=2))
                # A 0.8 m pedestrian is a couple of pixels across a 120 m view
                # and vanishes exactly where it matters -- people stepping out
                # from behind a parked car are the case this benchmark is for.
                # A ring keeps them findable without inflating their footprint.
                if float(boxes[i, 4]) < 2.0:
                    ax.plot(float(boxes[i, 0]), float(boxes[i, 1]), 'o',
                            markersize=5, markerfacecolor='none',
                            markeredgecolor=colour, markeredgewidth=1.2, zorder=2)

        ego_box = torch.tensor([[ego.center.x, ego.center.y, ego.center.heading,
                                 ego.car_footprint.width, ego.car_footprint.length]])
        ego_corners = boxes_to_corners(ego_box[:, 0], ego_box[:, 1], ego_box[:, 2],
                                       ego_box[:, 3], ego_box[:, 4])[0]
        ax.add_patch(Polygon(ego_corners.numpy(), closed=True, facecolor=EGO,
                             edgecolor='white', linewidth=1.0, zorder=3))

        counts = [0, 0, 0]  # seen, remembered, withheld
        if len(boxes):
            for i, token in enumerate(ids):
                if not near[i]:
                    continue
                counts[0 if token in fresh else 1 if token in given else 2] += 1
        ax.set_title(f'seen {counts[0]}   remembered {counts[1]}   '
                     f'withheld {counts[2]}   t={index * 0.1:.1f}s', fontsize=10)
        ax.set_xlim(float(origin[0]) - radius, float(origin[0]) + radius)
        ax.set_ylim(float(origin[1]) - radius, float(origin[1]) + radius)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    animation = FuncAnimation(figure, draw, frames=frames, interval=1000 / fps)
    animation.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    return len(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True, help='experiment dir, full observability')
    parser.add_argument('--occluded', required=True, help='experiment dir, occluded')
    parser.add_argument('--out-dir', default='occlusion_gifs')
    parser.add_argument('--scenarios', type=int, default=3)
    parser.add_argument('--tokens', default=None,
                        help='comma-separated scenario tokens to render instead '
                             'of the first few shared ones')
    parser.add_argument('--radius', type=float, default=60.0)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--stride', type=int, default=2,
                        help='keep every Nth frame; 2 halves a 20 Hz log')
    args = parser.parse_args()

    baseline_logs = find_logs(args.baseline)
    occluded_logs = find_logs(args.occluded)
    shared = set(baseline_logs) & set(occluded_logs)
    if not shared:
        raise SystemExit('the two runs have no scenario tokens in common')

    if args.tokens:
        # Keep the order given: these are usually ranked worst-first, and the
        # point is to look at them in that order.
        wanted = [t.strip() for t in args.tokens.split(',') if t.strip()]
        missing = [t for t in wanted if t not in shared]
        if missing:
            raise SystemExit('not in both runs: ' + ', '.join(missing))
        chosen = wanted
    else:
        chosen = sorted(shared)[:args.scenarios]

    os.makedirs(args.out_dir, exist_ok=True)
    for token in chosen:
        baseline = SimulationLog.load_data(Path(baseline_logs[token]))
        occluded = SimulationLog.load_data(Path(occluded_logs[token]))
        out_path = os.path.join(args.out_dir, f'{token}.gif')
        count = render(baseline, occluded, out_path, args.radius, args.fps, args.stride)
        print(f'{token}: {count} frames -> {out_path}')


if __name__ == '__main__':
    main()
