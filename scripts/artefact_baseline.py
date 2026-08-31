"""How much bad traffic the agent model produces on its own.

Before occlusion can be blamed for anything, the noise floor has to be known.
CRAFT reports that correcting an autoregressive traffic model cuts collisions
by 31% and violations by 33%, which read backwards says roughly a third of what
such a model produces is its own artefact. If that is not measured first, an
S1 result cannot distinguish "the planner failed because it could not see" from
"a background car drove into it".

So: ordinary scenarios, no adversarial pressure, no occlusion. Three
configurations, all scored with the same exact box-overlap test.

    log         every agent on its logged trajectory. This is nuPlan's own
                floor, and it is not zero -- these are perception boxes, and
                they overlap each other in the data.
    generated   every agent driven by the model.
    replay ego  the ego held on its logged trajectory while the model drives
                everything else. This is the configuration a planner benchmark
                actually runs, so the ego-involved rate here is the number that
                contaminates an S1 result.

The ego-involved column is the one that matters. A generated car drifting into
another generated car is background noise; a generated car driving into the
ego is indistinguishable, at scoring time, from the ego's own failure.

Usage:
    PYTHONPATH=. python scripts/artefact_baseline.py --limit 12
"""
import argparse

import torch

from scripts.nuplan_zeroshot import rollout
from smart.datasets.scalable_dataset import MultiDataset
from smart.metrics.at_fault import NAMES, at_fault, ego_collisions
from smart.metrics.collision import overlap_matrix
from smart.model import SMART
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging

VEHICLE = 0


def boxes_at(step, position, heading, shape, live):
    return torch.stack([position[live, step, 0], position[live, step, 1],
                        heading[live, step], shape[live, 1], shape[live, 0]], dim=-1)


def background_collision_rate(position, heading, shape, valid, ego):
    """Share of non-ego agent-timesteps spent overlapping another agent.

    This is not nuPlan's metric and is not meant to be. nuPlan scores the ego;
    this says how much the generated background is tangled up with itself,
    which is what tells you whether the traffic is plausible at all.
    """
    hits = total = 0
    for t in range(position.shape[1]):
        live = valid[:, t].nonzero(as_tuple=True)[0]
        if len(live) < 2:
            continue
        boxes = torch.stack([position[live, t, 0], position[live, t, 1],
                             heading[live, t], shape[live, 1], shape[live, 0]], dim=-1)
        struck = overlap_matrix(boxes).any(dim=1)
        keep = live != ego
        hits += int(struck[keep].sum())
        total += int(keep.sum())
    return hits / max(total, 1)


def offroad_rate(position, valid, kinds, lanes, tolerance):
    """Share of vehicle-timesteps further than `tolerance` from any centreline.

    Only vehicles are counted. A pedestrian on a pavement is not a violation,
    and nuPlan logs plenty of them.
    """
    if not len(lanes):
        return 0.0
    off = total = 0
    vehicles = kinds == VEHICLE
    for t in range(position.shape[1]):
        live = valid[:, t] & vehicles
        if not live.any():
            continue
        distance = torch.cdist(position[live, t], lanes).min(dim=1).values
        off += int((distance > tolerance).sum())
        total += int(live.sum())
    return off / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/nuplan_converted')
    parser.add_argument('--ckpt', default='checkpoints/epoch=31.ckpt')
    parser.add_argument('--config', default='configs/validation/validation_scalable.yaml')
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--offroad-tolerance', type=float, default=3.0,
                        help='metres from a lane centreline before a vehicle '
                             'counts as off-road; roughly half a lane plus margin')
    args = parser.parse_args()

    config = load_config_act(args.config)
    model = SMART(config.Model)
    model.load_params_from_file(filename=args.ckpt, logger=Logging().log(level='ERROR'))
    model = model.eval().cuda()

    dataset = MultiDataset(
        root=None, split='val', raw_dir=[args.data_dir], processed_dir=None,
        transform=WaymoTargetBuilder(config.Model.num_historical_steps,
                                     config.Model.decoder.num_future_steps))
    history = config.Model.num_historical_steps

    totals = {name: [0, 0, 0.0, 0.0, 0, [0] * 5]
              for name in ('log', 'generated', 'replay ego')}
    for index in range(min(args.limit, len(dataset))):
        data = dataset[index].cuda()
        scenario_id = data['scenario_id']
        try:
            data, pred = rollout(model, data, args.seed)
        except RuntimeError as error:
            print(f'{scenario_id}  FAILED  {error}'.replace(chr(10), " ")[:120])
            continue

        ego = int(data['agent']['av_index'])
        valid = data['agent']['valid_mask'][:, history:].cpu()
        shape = data['agent']['shape'][:, history, :].cpu()
        kinds = data['agent']['type'].cpu()
        lanes = data['map_point']['position'][:, :2].cpu()[
            data['map_point']['type'].cpu() == 16]

        logged = data['agent']['position'][:, history:, :2].cpu()
        logged_heading = data['agent']['heading'][:, history:].cpu()
        generated = pred['pred_traj'][..., :2].cpu()
        generated_heading = pred['pred_head'].cpu()

        replay = generated.clone()
        replay[ego] = logged[ego]
        replay_heading = generated_heading.clone()
        replay_heading[ego] = logged_heading[ego]

        print(f'{scenario_id}  {int(valid[:, 0].sum())} agents')
        for name, position, heads in (('log', logged, logged_heading),
                                      ('generated', generated, generated_heading),
                                      ('replay ego', replay, replay_heading)):
            counts, _ = ego_collisions(position, heads, shape, valid, ego)
            blame = at_fault(counts)
            background = background_collision_rate(position, heads, shape, valid, ego)
            off = offroad_rate(position, valid, kinds, lanes, args.offroad_tolerance)
            detail = ', '.join(f'{NAMES[k]} {c}' for k, c in enumerate(counts) if c)
            print(f'  {name:11s} ego collisions {sum(counts)}  at fault {blame}'
                  f'{"  (" + detail + ")" if detail else ""}   '
                  f'background {background:6.2%}   offroad {off:6.2%}')
            bucket = totals[name]
            bucket[0] += sum(counts)
            bucket[1] += blame
            bucket[2] += background
            bucket[3] += off
            bucket[4] += 1
            for k, c in enumerate(counts):
                bucket[5][k] += c

    print('\naggregate over scenarios')
    for name, bucket in totals.items():
        if not bucket[4]:
            continue
        detail = ', '.join(f'{NAMES[k]} {c}' for k, c in enumerate(bucket[5]) if c)
        print(f'  {name:11s} ego collisions {bucket[0]:3d}  at fault {bucket[1]:3d}   '
              f'background {bucket[2] / bucket[4]:6.2%}   '
              f'offroad {bucket[3] / bucket[4]:6.2%}')
        if detail:
            print(f'{"":15s}{detail}')


if __name__ == '__main__':
    main()
