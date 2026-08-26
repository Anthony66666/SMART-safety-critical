"""Does realism fall smoothly as a single scalar turns up the pressure?

Separating categories of wrong scenario is not enough for a Pareto frontier;
that needs a continuous knob and a smooth response. Temperature is the
cheapest stand-in for adversarial pressure: it reshapes the sampling
distribution without touching the model, so log p remains the model's own
verdict on what came out.

If realism does not track temperature monotonically, a beta-tilted sampler has
no reason to behave either.

    python scripts/temperature_sweep.py --num_scenarios 40 --allow_self_judge
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
from smart.safety.scoring import SelfJudgeError, prepare_scenario, score_tokens
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging

TEMPERATURES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/validation/validation_scalable.yaml')
    parser.add_argument('--generator_ckpt', type=str, default='checkpoints/epoch=31.ckpt')
    parser.add_argument('--judge_ckpt', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--num_scenarios', type=int, default=40)
    parser.add_argument('--beam_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--allow_self_judge', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    judge_ckpt = args.judge_ckpt or args.generator_ckpt
    if judge_ckpt == args.generator_ckpt and not args.allow_self_judge:
        raise SelfJudgeError('judge and generator are the same checkpoint; '
                             'pass --allow_self_judge to exercise the pipeline')

    config = load_config_act(args.config)
    config.Model.decoder.beam_size = args.beam_size
    generator = SMART(config.Model)
    generator.load_params_from_file(filename=args.generator_ckpt,
                                    logger=Logging().log(level='DEBUG'))
    generator = generator.eval().to(args.device)
    judge = generator if judge_ckpt == args.generator_ckpt else None
    if judge is None:
        judge = SMART(config.Model)
        judge.load_params_from_file(filename=judge_ckpt,
                                    logger=Logging().log(level='DEBUG'))
        judge = judge.eval().to(args.device)

    dc = config.Dataset
    raw_dir = [args.data_dir] if args.data_dir else dc.val_raw_dir
    dataset = MultiDataset(root=dc.root, split='val', raw_dir=raw_dir,
                           processed_dir=dc.val_processed_dir,
                           transform=WaymoTargetBuilder(
                               config.Model.num_historical_steps,
                               config.Model.decoder.num_future_steps))

    hist = config.Model.num_historical_steps
    offset = (hist - 1) // 5
    rows = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0)
    for i, batch in enumerate(loader):
        if len(rows) >= args.num_scenarios:
            break
        data = prepare_scenario(generator, batch.to(args.device), seed=args.seed)
        eval_mask = data['agent']['valid_mask'][:, hist - 1]
        if not eval_mask.any():
            continue

        num_steps = data['agent']['token_idx'].shape[1] - offset
        logged = data['agent']['token_idx'][:, offset:offset + num_steps].contiguous()
        logged_score = float(
            (score_tokens(judge, data, logged)[eval_mask] / num_steps).mean())

        row = {'scenario': str(data['scenario_id'][0]),
               'num_agents': int(eval_mask.sum()),
               'logged': logged_score, 'curve': []}
        for temp in TEMPERATURES:
            torch.manual_seed(args.seed + i)
            with torch.no_grad():
                rollout = generator.inference(data, temperature=temp)
            per_step = score_tokens(judge, data, rollout['next_token_idx'])[eval_mask]
            per_step = per_step / rollout['next_token_idx'].shape[1]
            row['curve'].append({
                'temperature': temp,
                'mean': float(per_step.mean()),
                'min': float(per_step.min()),
                'gap_vs_logged': float(per_step.mean()) - logged_score,
            })
        rows.append(row)

    print(f'\nscenarios: {len(rows)}   judge: {os.path.basename(judge_ckpt)}   '
          f'beam_size: {args.beam_size}')
    logged_mean = sum(r['logged'] for r in rows) / len(rows)
    print(f'  real traffic reference: {logged_mean:+.3f} log p / agent-step\n')
    print(f'  {"T":>6} {"mean":>9} {"min":>9} {"gap vs real":>13}')
    for k, temp in enumerate(TEMPERATURES):
        mean = sum(r['curve'][k]['mean'] for r in rows) / len(rows)
        mn = sum(r['curve'][k]['min'] for r in rows) / len(rows)
        gap = sum(r['curve'][k]['gap_vs_logged'] for r in rows) / len(rows)
        print(f'  {temp:>6.2f} {mean:>9.3f} {mn:>9.3f} {gap:>13.3f}')

    if judge_ckpt == args.generator_ckpt:
        print('\n  !! CIRCULAR: scored by the generating model.')

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'judge_ckpt': judge_ckpt, 'temperatures': TEMPERATURES,
                       'rows': rows}, f, indent=2)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
