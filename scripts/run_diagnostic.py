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
from smart.safety.scoring import RealismReport, prepare_scenario, score_tokens
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging


def load_model(config, ckpt):
    model = SMART(config.Model)
    model.load_params_from_file(filename=ckpt, logger=Logging().log(level='DEBUG'))
    model.eval()
    return model


def mean_per_step(log_p, eval_mask, num_steps):
    """Log-likelihood per simulated agent-step, comparable across scenarios."""
    simulated = log_p[eval_mask]
    if simulated.numel() == 0:
        return None
    return float(simulated.sum() / (simulated.numel() * num_steps))


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/validation/validation_scalable.yaml')
    parser.add_argument('--generator_ckpt', type=str, default='checkpoints/epoch=31.ckpt')
    parser.add_argument('--judge_ckpt', type=str, default='')
    parser.add_argument('--num_scenarios', type=int, default=11)
    parser.add_argument('--allow_self_judge', action='store_true')
    parser.add_argument('--beam_size', type=int, default=0,
                        help='override sampling truncation; 0 keeps the config value')
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
    dataset = MultiDataset(root=dc.root, split='val', raw_dir=dc.val_raw_dir,
                           processed_dir=dc.val_processed_dir,
                           transform=WaymoTargetBuilder(config.Model.num_historical_steps,
                                                        config.Model.decoder.num_future_steps))

    generator = load_model(config, args.generator_ckpt)
    judge = generator if report.self_judged else load_model(config, judge_ckpt)

    hist = config.Model.num_historical_steps
    shift = 5
    offset = (hist - 1) // shift

    for i, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
        if i >= args.num_scenarios:
            break
        # One prepared scenario for both models: preparing separately would
        # give them different map context and incomparable likelihoods.
        data = prepare_scenario(generator, batch, seed=args.seed)
        eval_mask = data['agent']['valid_mask'][:, hist - 1]

        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            rollout = generator.inference(data)
        num_steps = rollout['next_token_idx'].shape[1]

        logged_tokens = data['agent']['token_idx'][:, offset:offset + num_steps].contiguous()

        generated_score = score_tokens(judge, data, rollout['next_token_idx'])
        logged_score = score_tokens(judge, data, logged_tokens)

        report.scores.append({
            'scenario': str(data['scenario_id'][0]) if 'scenario_id' in data else str(i),
            'num_agents': int(eval_mask.sum()),
            'generated': mean_per_step(generated_score, eval_mask, num_steps),
            'logged': mean_per_step(logged_score, eval_mask, num_steps),
        })

    usable = [s for s in report.scores if s['generated'] is not None]
    gen = sum(s['generated'] for s in usable) / len(usable)
    log = sum(s['logged'] for s in usable) / len(usable)

    beam = getattr(config.Model.decoder, 'beam_size', 5)
    print(f'\nscenarios: {len(usable)}   judge: {os.path.basename(judge_ckpt)}   beam_size: {beam}')
    print(f'  logged    log p / agent-step: {log:+.4f}')
    print(f'  generated log p / agent-step: {gen:+.4f}')
    print(f'  gap (generated - logged):     {gen - log:+.4f}')
    caveat = report.caveat()
    if caveat:
        print(f'\n  !! {caveat}')

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'generator_ckpt': report.generator_ckpt,
                       'judge_ckpt': report.judge_ckpt,
                       'self_judged': report.self_judged,
                       'caveat': caveat,
                       'mean_logged': log, 'mean_generated': gen,
                       'scores': report.scores}, f, indent=2)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
