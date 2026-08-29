"""Bird's-eye rendering of what the ego can and cannot see.

Every closed-loop planning benchmark hands the planner ground-truth boxes for
all agents. This script draws the same scene the way perception would actually
deliver it: agents coloured by how much of their silhouette has a clear sight
line from the ego, with the shadow each occluder casts drawn behind it.

The point is to eyeball whether the geometry in smart/occlusion/visibility.py
behaves the way a person would judge by looking -- the unit tests pin
correctness against an independent implementation, but only a picture shows
whether the resulting occlusion is *interesting* on real traffic.

Usage (the package is not installed in the `smart` env, hence PYTHONPATH):
    PYTHONPATH=. python scripts/render_occlusion.py --data-dir data/valid_demo --timestep 50
"""
import argparse
import glob
import math
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Polygon

from smart.occlusion.visibility import agent_visibility, boxes_to_corners

VEHICLE = 0

HIDDEN = '#d1495b'
PARTIAL = '#edae49'
VISIBLE = '#4c9f70'
EGO = '#2364aa'


def load_boxes(scenario, timestep, radius):
    """Boxes of every agent valid at `timestep`, plus the ego's row index.

    Returns (boxes, types, ego_row) where boxes is (N, 5) rows of
    [x, y, heading, width, length]. Agents beyond `radius` from the ego are
    dropped: they are neither interesting to look at nor able to occlude
    anything near the ego.
    """
    agent = scenario['agent']
    valid = agent['valid_mask'][:, timestep]
    av_index = agent['av_index']
    if not bool(valid[av_index]):
        return None, None, None

    position = agent['position'][:, timestep, :2]
    within = (position - position[av_index]).norm(dim=-1) <= radius
    keep = (valid & within).nonzero(as_tuple=True)[0]

    shape = agent['shape'][:, timestep, :]  # [length, width, height]
    boxes = torch.stack([position[keep, 0],
                         position[keep, 1],
                         agent['heading'][keep, timestep],
                         shape[keep, 1],
                         shape[keep, 0]], dim=-1)
    ego_row = int((keep == av_index).nonzero(as_tuple=True)[0])
    return boxes, agent['type'][keep], ego_row


def shadow_polygon(origin, corners, reach):
    """The wedge an occluder hides, as a polygon.

    The two silhouette corners are the angular extremes of the box as seen from
    `origin`; the wedge runs from them outward to `reach`. Angles are measured
    relative to the direction of the box centre so the extremes are unambiguous
    without worrying about the pi/-pi seam.
    """
    offsets = corners - origin
    centre_angle = math.atan2(float(offsets[:, 1].mean()), float(offsets[:, 0].mean()))
    relative = torch.atan2(offsets[:, 1], offsets[:, 0]) - centre_angle
    relative = (relative + math.pi) % (2 * math.pi) - math.pi

    near, far = corners[int(relative.argmin())], corners[int(relative.argmax())]
    extend = lambda c: origin + (c - origin) / (c - origin).norm() * reach
    return torch.stack([near, extend(near), extend(far), far]).numpy()


def draw_scene(ax, scenario, timestep, radius=55.0, occluder_types=(VEHICLE,)):
    boxes, types, ego_row = load_boxes(scenario, timestep, radius)
    if boxes is None:
        ax.set_axis_off()
        return None

    origin = boxes[ego_row, :2]

    # Only vehicle-sized agents block sight, and the ego cannot occlude itself.
    is_occluder = torch.zeros(len(boxes), dtype=torch.bool)
    for t in occluder_types:
        is_occluder |= (types == t)
    is_occluder[ego_row] = False

    fraction = agent_visibility(origin, boxes, occluder_mask=is_occluder)
    corners = boxes_to_corners(boxes[:, 0], boxes[:, 1], boxes[:, 2],
                               boxes[:, 3], boxes[:, 4])

    # nuPlan fixtures carry no road geometry -- the maps are a separate, large
    # download -- so the scene is still worth drawing without it.
    if 'map_point' in scenario:
        map_xy = scenario['map_point']['position'][:, :2]
        near_map = (map_xy - origin).norm(dim=-1) <= radius
        ax.scatter(map_xy[near_map, 0], map_xy[near_map, 1],
                   s=0.25, c='#c8c8c8', linewidths=0, zorder=0)

    for i in torch.nonzero(is_occluder, as_tuple=True)[0].tolist():
        ax.add_patch(Polygon(shadow_polygon(origin, corners[i], radius * 2.2),
                             closed=True, facecolor='#000000', alpha=0.055,
                             edgecolor='none', zorder=1))

    for i in range(len(boxes)):
        if i == ego_row:
            continue
        f = float(fraction[i])
        colour = VISIBLE if f == 1.0 else (HIDDEN if f == 0.0 else PARTIAL)
        ax.add_patch(Polygon(corners[i].numpy(), closed=True, facecolor=colour,
                             edgecolor='#333333', linewidth=0.5, alpha=0.9, zorder=3))

    ax.add_patch(Polygon(corners[ego_row].numpy(), closed=True, facecolor=EGO,
                         edgecolor='black', linewidth=0.9, zorder=4))
    ax.plot(*origin.tolist(), marker='*', color='white', markersize=6,
            markeredgecolor='black', markeredgewidth=0.4, zorder=5)

    others = torch.cat([fraction[:ego_row], fraction[ego_row + 1:]])
    counts = (int((others == 0).sum()), int(((others > 0) & (others < 1)).sum()),
              int((others == 1).sum()))

    ax.set_xlim(float(origin[0]) - radius, float(origin[0]) + radius)
    ax.set_ylim(float(origin[1]) - radius, float(origin[1]) + radius)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{scenario['scenario_id']}  t={timestep}\n"
                 f"hidden {counts[0]} / partial {counts[1]} / visible {counts[2]}",
                 fontsize=8)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/valid_demo')
    parser.add_argument('--timestep', type=int, default=50)
    parser.add_argument('--radius', type=float, default=55.0)
    parser.add_argument('--num-scenarios', type=int, default=6)
    parser.add_argument('--out', default='occlusion_demo.png')
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, '*.pkl')))[:args.num_scenarios]
    if not paths:
        raise SystemExit(f'no scenarios found in {args.data_dir}')

    cols = min(3, len(paths))
    rows = math.ceil(len(paths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.4 * rows))
    axes = [axes] if len(paths) == 1 else list(axes.flat)

    totals = [0, 0, 0]
    for ax, path in zip(axes, paths):
        with open(path, 'rb') as f:
            scenario = pickle.load(f)
        counts = draw_scene(ax, scenario, args.timestep, args.radius)
        if counts is not None:
            totals = [a + b for a, b in zip(totals, counts)]
        print(f'{os.path.basename(path):24s} hidden/partial/visible = {counts}')
    for ax in axes[len(paths):]:
        ax.set_axis_off()

    fig.suptitle(f'Ego line of sight at t={args.timestep}  '
                 f'(blue = ego, red = fully hidden, amber = partial, green = fully visible)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches='tight')
    print(f'\ntotal hidden/partial/visible = {tuple(totals)}  -> {args.out}')


if __name__ == '__main__':
    main()
