"""Roll the WOMD checkpoint out on converted nuPlan scenarios, without training.

The plan's first transfer gate: does a model trained entirely on Waymo produce
sane traffic on nuPlan geometry? If it does, fine-tuning and retraining -- and
re-clustering the token vocabulary -- can be skipped, which is the difference
between a day and a fortnight.

What is reported is deliberately about plausibility rather than accuracy.
Nobody expects a zero-shot model to reproduce the log; the question is whether
what it produces is physically possible traffic. So: acceleration (a car that
pulls 15 m/s^2 is not driving), speed, collisions between generated agents, and
how far agents stray from any lane centreline.

Ground truth is rolled out through the same measurements, because several of
these numbers are only interpretable against what the real scene does. nuPlan
tracks are perception output, not physics -- they jitter -- so a collision rate
or an acceleration tail means little until you know the log's own.

Usage:
    PYTHONPATH=. python scripts/nuplan_zeroshot.py \
        --data-dir data/nuplan_converted --ckpt checkpoints/epoch=31.ckpt
"""
import argparse
import os
import pickle

import torch

from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMART
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging

DT = 0.1


def rollout(model, data, seed):
    """One scenario through the inference path validation_step uses."""
    # sample_pt_pred drops map points at random -- a training augmentation that
    # is still live at inference. Unseeded it makes every rollout, and every
    # number below, irreproducible.
    torch.manual_seed(seed)
    data = model.match_token_map(data)
    data = model.sample_pt_pred(data)
    if 'ptr' in data['agent']:
        data['agent']['av_index'] += data['agent']['ptr'][:-1]
    with torch.no_grad():
        pred = model.inference(data)
    return data, pred


def kinematics(trajectory, valid):
    """Speed and acceleration magnitudes over a [agent, time, 2] trajectory."""
    if trajectory.shape[1] < 3:
        return torch.zeros(0), torch.zeros(0)
    steps = trajectory[:, 1:] - trajectory[:, :-1]
    speed = steps.norm(dim=-1) / DT
    acceleration = (speed[:, 1:] - speed[:, :-1]).abs() / DT
    keep = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    return speed[:, 1:][keep], acceleration[keep]


def collisions(trajectory, valid, shape, radius=None):
    """Fraction of timesteps where an agent overlaps another, by circle test.

    A circle of the box's half-length is a coarse proxy for the box, and it
    over-reports for long vehicles side by side. It is applied identically to
    prediction and to ground truth, so the comparison between them holds even
    though neither number is an exact box-overlap rate.
    """
    hits = 0
    total = 0
    for t in range(trajectory.shape[1]):
        live = valid[:, t].nonzero(as_tuple=True)[0]
        if len(live) < 2:
            continue
        centres = trajectory[live, t]
        extent = shape[live, 0] * 0.5
        gap = torch.cdist(centres, centres)
        threshold = extent[:, None] + extent[None, :]
        overlap = (gap < threshold).fill_diagonal_(False)
        hits += int(overlap.any(dim=1).sum())
        total += len(live)
    return hits / max(total, 1)


def offroad(trajectory, valid, centreline):
    """Distance from each agent to the nearest lane centreline point."""
    if not len(centreline):
        return torch.zeros(0)
    distances = []
    for t in range(trajectory.shape[1]):
        live = valid[:, t]
        if not live.any():
            continue
        distances.append(torch.cdist(trajectory[live, t], centreline).min(dim=1).values)
    return torch.cat(distances) if distances else torch.zeros(0)


def summarise(name, speed, acceleration, collision_rate, distance):
    def q(values, p):
        return float(values.quantile(p)) if len(values) else float('nan')

    print(f'  {name:12s} '
          f'speed p50 {q(speed, 0.5):5.1f}  p95 {q(speed, 0.95):5.1f} m/s   '
          f'accel p50 {q(acceleration, 0.5):5.2f}  p95 {q(acceleration, 0.95):6.2f}  '
          f'p99 {q(acceleration, 0.99):7.2f} m/s^2   '
          f'collision {collision_rate:5.1%}   '
          f'lane dist p50 {q(distance, 0.5):4.1f}  p95 {q(distance, 0.95):5.1f} m')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/nuplan_converted')
    parser.add_argument('--ckpt', default='checkpoints/epoch=31.ckpt')
    parser.add_argument('--config', default='configs/validation/validation_scalable.yaml')
    parser.add_argument('--out', default=None, help='directory to save rollouts in')
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()

    config = load_config_act(args.config)
    model = SMART(config.Model)
    model.load_params_from_file(filename=args.ckpt, logger=Logging().log(level='INFO'))
    model = model.eval().cuda()

    dataset = MultiDataset(
        root=None, split='val', raw_dir=[args.data_dir], processed_dir=None,
        transform=WaymoTargetBuilder(config.Model.num_historical_steps,
                                     config.Model.decoder.num_future_steps))
    history = config.Model.num_historical_steps

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    totals = {'pred': [[], [], [], []], 'log': [[], [], [], []]}
    for index in range(min(args.limit, len(dataset))):
        data = dataset[index].cuda()
        scenario_id = data['scenario_id']
        try:
            data, pred = rollout(model, data, args.seed)
        except RuntimeError as error:
            print(f'{scenario_id}  FAILED  {error}'.replace('\n', ' ')[:150])
            continue

        predicted = pred['pred_traj'][..., :2].cpu()
        # pred['gt'] is the tokenized ground truth, so its accelerations are
        # token-grid snapping rather than motion -- they come out as repeated
        # values like 6.25 and 25.77 m/s^2. The raw logged positions are the
        # only honest kinematic baseline.
        truth = data['agent']['position'][:, history:, :2].cpu()
        valid = data['agent']['valid_mask'][:, history:].cpu()
        shape = data['agent']['shape'][:, history, :2].cpu()
        centre = data['map_point']['position'][:, :2].cpu()
        lanes = centre[data['map_point']['type'].cpu() == 16]

        print(f"{scenario_id}  {data['agent']['num_nodes']:4d} agents  "
              f"pred {tuple(predicted.shape)}")
        for label, trajectory in (('prediction', predicted), ('log', truth)):
            speed, acceleration = kinematics(trajectory, valid)
            rate = collisions(trajectory, valid, shape)
            distance = offroad(trajectory, valid, lanes)
            summarise(label, speed, acceleration, rate, distance)
            bucket = totals['pred' if label == 'prediction' else 'log']
            bucket[0].append(speed)
            bucket[1].append(acceleration)
            bucket[2].append(torch.tensor([rate]))
            bucket[3].append(distance)

        if args.out:
            with open(os.path.join(args.out, f'{scenario_id}.pkl'), 'wb') as f:
                pickle.dump({'scenario_id': scenario_id,
                             'pred': predicted, 'gt': truth, 'valid': valid,
                             'shape': shape, 'lanes': lanes,
                             'history': data['agent']['position'][:, :history, :2].cpu(),
                             'history_valid': data['agent']['valid_mask'][:, :history].cpu(),
                             'type': data['agent']['type'].cpu(),
                             'av_index': int(data['agent']['av_index'])}, f)

    print('\naggregate')
    for label, key in (('prediction', 'pred'), ('log', 'log')):
        speed, acceleration, rate, distance = (torch.cat(v) if v else torch.zeros(0)
                                               for v in totals[key])
        summarise(label, speed, acceleration, float(rate.mean()) if len(rate) else 0.0,
                  distance)


if __name__ == '__main__':
    main()
