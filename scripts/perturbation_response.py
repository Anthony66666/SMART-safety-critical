"""Does the realism statistic respond to a single agent going wrong?

Separating four categories of scrambled scenario is not enough for a Pareto
frontier: that needs realism to fall smoothly as adversarial pressure rises.
This probes the smallest version of that question. One agent is slid sideways
by a growing distance while its token sequence is held fixed, so its motion is
untouched and only its place in the world changes -- first off its lane, then
off the road entirely.

If the likelihood does not track that, it cannot measure what adversarial
generation needs it to measure.

    python scripts/perturbation_response.py --num_scenarios 20 --allow_self_judge
"""
import argparse
import json
import os
import sys

import torch
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMART
from smart.safety.scoring import (SelfJudgeError, lateral_offset,
                                  prepare_scenario, score_tokens)
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging

DELTAS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]


def load_model(config, ckpt, beam_size):
    config.Model.decoder.beam_size = beam_size
    model = SMART(config.Model)
    model.load_params_from_file(filename=ckpt, logger=Logging().log(level='DEBUG'))
    return model.eval()


def pick_target(data, eval_mask):
    """A vehicle near the AV: on-road to begin with, so sliding it off means
    something. Never the AV itself."""
    av = int(data['agent']['av_index'])
    is_veh = data['agent']['type'] == 0
    candidates = eval_mask & is_veh
    candidates[av] = False
    if not candidates.any():
        return None
    pos = data['agent']['token_pos']
    dist = (pos[:, 1, :2] - pos[av, 1, :2]).norm(dim=-1)
    dist[~candidates] = float('inf')
    return int(dist.argmin())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/validation/validation_scalable.yaml')
    parser.add_argument('--generator_ckpt', type=str, default='checkpoints/epoch=31.ckpt')
    parser.add_argument('--judge_ckpt', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--num_scenarios', type=int, default=20)
    parser.add_argument('--beam_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--allow_self_judge', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    judge_ckpt = args.judge_ckpt or args.generator_ckpt
    if judge_ckpt == args.generator_ckpt and not args.allow_self_judge:
        raise SelfJudgeError(
            'judge and generator are the same checkpoint; pass --allow_self_judge '
            'to measure the pipeline rather than realism')

    config = load_config_act(args.config)
    judge = load_model(config, judge_ckpt, args.beam_size).to(args.device)

    dc = config.Dataset
    raw_dir = [args.data_dir] if args.data_dir else dc.val_raw_dir
    dataset = MultiDataset(root=dc.root, split='val', raw_dir=raw_dir,
                           processed_dir=dc.val_processed_dir,
                           transform=WaymoTargetBuilder(
                               config.Model.num_historical_steps,
                               config.Model.decoder.num_future_steps))

    hist = config.Model.num_historical_steps
    shift = 5
    offset = (hist - 1) // shift
    rows = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0)
    for i, batch in enumerate(loader):
        if len(rows) >= args.num_scenarios:
            break
        data = prepare_scenario(judge, batch.to(args.device), seed=args.seed)
        eval_mask = data['agent']['valid_mask'][:, hist - 1]
        target = pick_target(data, eval_mask)
        if target is None:
            continue

        num_steps = data['agent']['token_idx'].shape[1] - offset
        logged = data['agent']['token_idx'][:, offset:offset + num_steps].contiguous()
        base_token_pos = data['agent']['token_pos'].clone()
        others = eval_mask.clone()
        others[target] = False

        row = {'scenario': str(data['scenario_id'][0]),
               'target': target, 'num_agents': int(eval_mask.sum()), 'curve': []}
        for delta in DELTAS:
            data['agent']['token_pos'] = lateral_offset(
                base_token_pos, data['agent']['token_heading'], target, delta)
            log_p = score_tokens(judge, data, logged) / num_steps
            row['curve'].append({
                'delta': delta,
                'target': float(log_p[target]),
                'scene_min': float(log_p[eval_mask].min()),
                'others_mean': float(log_p[others].mean()) if others.any() else None,
            })
        data['agent']['token_pos'] = base_token_pos
        rows.append(row)

    print(f'\nscenarios: {len(rows)}   judge: {os.path.basename(judge_ckpt)}   '
          f'beam_size: {args.beam_size}')
    print(f'\n  {"offset (m)":>10} {"target agent":>14} {"scene min":>11} {"others":>10}')
    for k, delta in enumerate(DELTAS):
        tgt = sum(r['curve'][k]['target'] for r in rows) / len(rows)
        mn = sum(r['curve'][k]['scene_min'] for r in rows) / len(rows)
        oth = [r['curve'][k]['others_mean'] for r in rows
               if r['curve'][k]['others_mean'] is not None]
        print(f'  {delta:>10.1f} {tgt:>14.3f} {mn:>11.3f} '
              f'{sum(oth)/len(oth) if oth else float("nan"):>10.3f}')

    if judge_ckpt == args.generator_ckpt:
        print('\n  !! CIRCULAR: scored by the generating model.')

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'judge_ckpt': judge_ckpt, 'deltas': DELTAS, 'rows': rows}, f, indent=2)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
