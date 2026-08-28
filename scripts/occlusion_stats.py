"""How much of the scene the ego loses to occlusion, and how much memory wins back.

Two numbers matter for the benchmark and neither should be guessed at. The first
is how much raw line-of-sight occlusion costs, broken down by distance -- an
aggregate is misleading, because agents far away are both the most occluded and
the least relevant to avoiding a collision. The second is how much of that a
tracking buffer recovers, since a planner that remembers a car it saw two
seconds ago is not really blind to it.

Reported per distance band:
    hidden / partial / visible   raw line-of-sight outcome
    remembered                   hidden agents the buffer still knows about
    unknown                      hidden agents the ego has no belief about at all

`unknown` is the number the benchmark actually rests on: those are the agents a
planner cannot account for by any means short of reasoning about empty space.

Usage:
    PYTHONPATH=. python scripts/occlusion_stats.py --data-dir data/valid_demo
"""
import argparse
import glob
import os
import pickle

import torch

from smart.occlusion.tracking import TrackObservation, TrackingBuffer
from smart.occlusion.visibility import agent_visibility

VEHICLE = 0
WOMD_DT = 0.1


def scenario_stats(scenario, bands, radius, buffer):
    """Accumulate per-band counts over every timestep of one scenario."""
    agent = scenario['agent']
    av_index = agent['av_index']
    counts = {b: [0, 0, 0, 0, 0] for b in bands}  # hidden, partial, visible, remembered, unknown
    buffer.reset()

    for t in range(agent['valid_mask'].shape[1]):
        valid = agent['valid_mask'][:, t]
        if not bool(valid[av_index]):
            continue

        position = agent['position'][:, t, :2]
        distance = (position - position[av_index]).norm(dim=-1)
        keep = (valid & (distance <= radius)).nonzero(as_tuple=True)[0]
        shape = agent['shape'][:, t, :]
        boxes = torch.stack([position[keep, 0], position[keep, 1],
                             agent['heading'][keep, t],
                             shape[keep, 1], shape[keep, 0]], dim=-1)

        ego_row = int((keep == av_index).nonzero(as_tuple=True)[0])
        occluders = (agent['type'][keep] == VEHICLE).clone()
        occluders[ego_row] = False
        fraction = agent_visibility(boxes[ego_row, :2], boxes, occluder_mask=occluders)

        velocity = agent['velocity'][:, t, :2]
        seen = [TrackObservation(track_id=str(int(keep[j])),
                                 x=float(boxes[j, 0]), y=float(boxes[j, 1]),
                                 heading=float(boxes[j, 2]),
                                 velocity_x=float(velocity[keep[j], 0]),
                                 velocity_y=float(velocity[keep[j], 1]),
                                 width=float(boxes[j, 3]), length=float(boxes[j, 4]))
                for j in range(len(keep)) if j != ego_row and fraction[j] > 0]
        remembered_ids = {e.track_id for e in buffer.update(t * WOMD_DT, seen)}

        for j in range(len(keep)):
            if j == ego_row:
                continue
            d = float(distance[keep[j]])
            band = next((b for b in bands if b[0] <= d < b[1]), None)
            if band is None:
                continue
            f = float(fraction[j])
            row = counts[band]
            row[0 if f == 0 else (2 if f == 1 else 1)] += 1
            if f == 0:
                row[3 if str(int(keep[j])) in remembered_ids else 4] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/valid_demo')
    parser.add_argument('--radius', type=float, default=55.0)
    parser.add_argument('--memory-horizon', type=float, default=3.0)
    args = parser.parse_args()

    bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, args.radius)]
    totals = {b: [0, 0, 0, 0, 0] for b in bands}
    buffer = TrackingBuffer(memory_horizon=args.memory_horizon)

    paths = sorted(glob.glob(os.path.join(args.data_dir, '*.pkl')))
    if not paths:
        raise SystemExit(f'no scenarios found in {args.data_dir}')
    for path in paths:
        with open(path, 'rb') as f:
            scenario = pickle.load(f)
        for band, row in scenario_stats(scenario, bands, args.radius, buffer).items():
            totals[band] = [a + b for a, b in zip(totals[band], row)]

    print(f'{len(paths)} scenarios, radius {args.radius:g} m, '
          f'memory horizon {args.memory_horizon:g} s\n')
    header = f"{'band (m)':>10} {'hidden':>8} {'partial':>8} {'visible':>8} " \
             f"{'hidden%':>8} {'remembered':>11} {'unknown':>8} {'unknown%':>9}"
    print(header)
    print('-' * len(header))
    grand = [0, 0, 0, 0, 0]
    for band in bands:
        h, p, v, r, u = totals[band]
        n = h + p + v
        grand = [a + b for a, b in zip(grand, totals[band])]
        if n:
            print(f'{str(band):>10} {h:>8} {p:>8} {v:>8} {h / n:>7.1%} '
                  f'{r:>11} {u:>8} {u / n:>8.1%}')
    h, p, v, r, u = grand
    n = h + p + v
    print('-' * len(header))
    print(f"{'all':>10} {h:>8} {p:>8} {v:>8} {h / n:>7.1%} {r:>11} {u:>8} {u / n:>8.1%}")


if __name__ == '__main__':
    main()
