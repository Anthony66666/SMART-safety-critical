"""Draw what the model did against what the log did.

Aggregate numbers say a rollout is wrong; a picture says how. Each panel shows
one scenario: lane centrelines underneath, then each agent's history, the
logged future, and the generated future, so a failure mode that a collision
rate can only hint at -- everything snapping onto a centreline, agents frozen,
trajectories fired off in one direction -- is visible directly.

Usage:
    PYTHONPATH=. python scripts/render_rollout.py --rollouts <dir> --out roll.png
"""
import argparse
import glob
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

HISTORY = '#9a9a9a'
LOG = '#1f6fb4'
PREDICTION = '#d1495b'


def draw(ax, rollout, radius):
    """One scenario: lanes, history, logged future, generated future."""
    ego = rollout['av_index']
    history, history_valid = rollout['history'], rollout['history_valid']
    origin = history[ego, -1]

    lanes = rollout['lanes']
    near = (lanes - origin).norm(dim=-1) <= radius
    ax.scatter(lanes[near, 0], lanes[near, 1], s=0.2, c='#d8d8d8',
               linewidths=0, zorder=0)

    predicted, truth, valid = rollout['pred'], rollout['gt'], rollout['valid']
    shown = 0
    for i in range(len(predicted)):
        if not bool(history_valid[i, -1]):
            continue
        if float((history[i, -1] - origin).norm()) > radius:
            continue
        shown += 1

        past = history[i][history_valid[i]]
        if len(past) > 1:
            ax.plot(past[:, 0], past[:, 1], c=HISTORY, lw=0.7, zorder=1)

        live = valid[i]
        if live.any():
            ax.plot(truth[i][live][:, 0], truth[i][live][:, 1],
                    c=LOG, lw=0.9, alpha=0.85, zorder=2)
            ax.plot(predicted[i][live][:, 0], predicted[i][live][:, 1],
                    c=PREDICTION, lw=0.9, alpha=0.85, zorder=3)

    ax.plot(*history[ego, -1], marker='*', ms=11, c='#111111', zorder=5)
    ax.set_xlim(float(origin[0]) - radius, float(origin[0]) + radius)
    ax.set_ylim(float(origin[1]) - radius, float(origin[1]) + radius)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{rollout['scenario_id'][:16]}  {shown} agents shown", fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', required=True)
    parser.add_argument('--out', default='rollout.png')
    parser.add_argument('--radius', type=float, default=60.0)
    parser.add_argument('--num', type=int, default=6)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.rollouts, '*.pkl')))[:args.num]
    if not paths:
        raise SystemExit(f'no rollouts in {args.rollouts}')

    columns = min(3, len(paths))
    rows = (len(paths) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.4 * rows))
    axes = [axes] if len(paths) == 1 else list(axes.flat)

    for ax, path in zip(axes, paths):
        with open(path, 'rb') as f:
            draw(ax, pickle.load(f), args.radius)
    for ax in axes[len(paths):]:
        ax.axis('off')

    figure.suptitle('grey = history,  blue = logged future,  red = generated future',
                    fontsize=10)
    figure.tight_layout()
    figure.savefig(args.out, dpi=140, bbox_inches='tight')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
