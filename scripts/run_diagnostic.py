"""End-to-end realism diagnostic: generate, then score under a judge.

Reports the judge's log-likelihood for generated scenarios against the same
judge's log-likelihood for the logged ones. The logged scenarios are the
reference scale: they are what real traffic looks like to the judge, so the
gap is what "how realistic is this generator" means numerically.

Running with the generator as its own judge is circular and produces no
evidence, but it does exercise the whole chain. It must be requested with
--allow_self_judge, and every result is stamped accordingly.

    python scripts/run_diagnostic.py --num_scenarios 11 --allow_self_judge
"""
import json
import os
import sys
from argparse import ArgumentParser

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMART
from smart.safety.scoring import (RealismReport, bits_per_dim, borrow_tokens,
                                  permute_agents, prepare_scenario, score_tokens)
from smart.safety.splits import SplitSpec, assign_split
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging


def load_model(config, ckpt, device='cuda'):
    model = SMART(config.Model)
    model.load_params_from_file(filename=ckpt, logger=Logging().log(level='DEBUG'))
    model.eval()
    return model.to(device)


def mean_per_step(log_p, eval_mask, num_steps):
    """Log-likelihood per simulated agent-step, comparable across scenarios."""
    simulated = log_p[eval_mask]
    if simulated.numel() == 0:
        return None
    return float(simulated.sum() / (simulated.numel() * num_steps))


def stats_per_step(log_p, eval_mask, num_steps):
    """Mean plus lower-tail statistics of the per-agent likelihood.

    Averaging over every agent dilutes the few that are actually implausible,
    which is precisely the situation in an adversarial scenario. The tail
    statistics are kept so that aggregation choices can be compared without
    re-running generation.
    """
    simulated = log_p[eval_mask] / num_steps
    if simulated.numel() == 0:
        return None
    q = torch.tensor([0.10, 0.25, 0.50], device=simulated.device)
    p10, p25, median = torch.quantile(simulated.float(), q).tolist()
    return {'mean': float(simulated.mean()), 'min': float(simulated.min()),
            'p10': p10, 'p25': p25, 'median': median}


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/validation/validation_scalable.yaml')
    parser.add_argument('--generator_ckpt', type=str, default='checkpoints/epoch=31.ckpt')
    parser.add_argument('--judge_ckpt', type=str, default='')
    parser.add_argument('--num_scenarios', type=int, default=11)
    parser.add_argument('--data_dir', type=str, default='',
                        help='override the config validation directory')
    parser.add_argument('--fraction', type=float, default=0.0,
                        help='deterministic hash subset of the directory, e.g. 0.1')
    parser.add_argument('--allow_self_judge', action='store_true')
    parser.add_argument('--beam_size', type=int, default=0,
                        help='override sampling truncation; 0 keeps the config value')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='dataloader workers; preprocessing dominates runtime')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='')
    args = parser.parse_args()

    judge_ckpt = args.judge_ckpt or args.generator_ckpt
    report = RealismReport.create(generator_ckpt=args.generator_ckpt,
                                  judge_ckpt=judge_ckpt,
                                  allow_self_judge=args.allow_self_judge)

    config = load_config_act(args.config)
    if args.beam_size:
        config.Model.decoder.beam_size = args.beam_size
    dc = config.Dataset
    raw_dir = [args.data_dir] if args.data_dir else dc.val_raw_dir
    scenario_ids = None
    if args.fraction:
        # Hash-based so the same subset comes back on any machine.
        spec = SplitSpec(fractions={'keep': args.fraction, 'drop': 1.0 - args.fraction},
                         salt='diagnostic-v1')
        scenario_ids = {
            os.path.splitext(f)[0]
            for d in raw_dir for f in os.listdir(d)
            if assign_split(os.path.splitext(f)[0], spec) == 'keep'}
        print(f'subset: {len(scenario_ids)} scenarios ({args.fraction:.0%} of {raw_dir})')
    dataset = MultiDataset(root=dc.root, split='val', raw_dir=raw_dir,
                           processed_dir=dc.val_processed_dir,
                           transform=WaymoTargetBuilder(config.Model.num_historical_steps,
                                                        config.Model.decoder.num_future_steps),
                           scenario_ids=scenario_ids)

    generator = load_model(config, args.generator_ckpt, args.device)
    judge = generator if report.self_judged else load_model(config, judge_ckpt, args.device)

    hist = config.Model.num_historical_steps
    shift = 5
    offset = (hist - 1) // shift

    donor = None
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0)
    for i, batch in enumerate(loader):
        if i >= args.num_scenarios:
            break
        # One prepared scenario for both models: preparing separately would
        # give them different map context and incomparable likelihoods.
        data = prepare_scenario(generator, batch.to(args.device), seed=args.seed)
        eval_mask = data['agent']['valid_mask'][:, hist - 1]

        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            rollout = generator.inference(data)
        num_steps = rollout['next_token_idx'].shape[1]

        logged_tokens = data['agent']['token_idx'][:, offset:offset + num_steps].contiguous()

        row = {
            'scenario': str(data['scenario_id'][0]) if 'scenario_id' in data else str(i),
            'num_agents': int(eval_mask.sum()),
        }
        raw = {
            'logged': score_tokens(judge, data, logged_tokens),
            'generated': score_tokens(judge, data, rollout['next_token_idx']),
            'permuted': score_tokens(judge, data, permute_agents(logged_tokens, seed=args.seed)),
        }
        if donor is not None:
            raw['borrowed'] = score_tokens(judge, data, borrow_tokens(donor, eval_mask.shape[0]))
        for name, value in raw.items():
            row[name] = mean_per_step(value, eval_mask, num_steps)
            row[name + '_stats'] = stats_per_step(value, eval_mask, num_steps)
        donor = logged_tokens
        report.scores.append(row)

    usable = [s for s in report.scores if s['generated'] is not None]

    def mean_of(key):
        vals = [s[key] for s in usable if s.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    beam = getattr(config.Model.decoder, 'beam_size', 5)
    print(f'\nscenarios: {len(usable)}   judge: {os.path.basename(judge_ckpt)}   beam_size: {beam}')
    log = mean_of('logged')
    labels = [('logged', 'real traffic (reference)'),
              ('generated', 'this generator'),
              ('permuted', 'own tokens, wrong agents'),
              ('borrowed', 'another scenario entirely')]
    def stat_of(key, agg):
        vals = [s[key + '_stats'][agg] for s in usable if s.get(key + '_stats')]
        return sum(vals) / len(vals) if vals else None

    # The lower tail is the headline: the mean cancels a positive tail effect
    # against a negative median effect and separates nothing. See the note in
    # smart/safety/scoring.py for the measured comparison.
    print(f'  {"":<26} {"min":>8} {"p10":>8} {"mean":>9} {"bpd":>7}   {"tail vs logged":>14}')
    log_tail = stat_of('logged', 'min')
    for key, label in labels:
        value = mean_of(key)
        if value is None:
            continue
        tail = stat_of(key, 'min')
        p10 = stat_of(key, 'p10')
        delta = '' if key == 'logged' else f'{tail - log_tail:+14.3f}'
        print(f'  {label:<26} {tail:>8.3f} {p10:>8.3f} {value:>9.3f} '
              f'{bits_per_dim(value):>7.3f}   {delta:>14}')
    caveat = report.caveat()
    if caveat:
        print(f'\n  !! {caveat}')

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'generator_ckpt': report.generator_ckpt,
                       'judge_ckpt': report.judge_ckpt,
                       'self_judged': report.self_judged,
                       'caveat': caveat,
                       'means': {k: mean_of(k) for k, _ in labels},
                       'tail_min': {k: stat_of(k, 'min') for k, _ in labels},
                       'tail_p10': {k: stat_of(k, 'p10') for k, _ in labels},
                       'scores': report.scores}, f, indent=2)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
