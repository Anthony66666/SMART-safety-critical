"""Coarse beta grid for danger-tilted sampling: does a usable frontier exist?

The plan flags this as the check to run before committing to the full Track A
experiment. Temperature was monotone in 60/60 scenarios, but it is a global
perturbation; a danger tilt concentrates its effect on the adversary near the
victim, so its response could be far noisier. This sweeps beta from off to
aggressive and reports, per beta, the danger the adversary achieves against the
victim and the realism cost it pays (the model's own log p of the adversary
trajectory -- tilting does not change that accounting).

    python scripts/tilt_sweep.py --num_scenarios 40 --allow_self_judge

Result on 60 WOMD validation scenarios, self-judged, full support (mean curve):

      beta   danger (m)   adv log p/step   collisions
       off       -4.015           -3.728      5/60
      5.00       -3.676           -3.828      6/60
      2.00       -3.158           -3.994     12/60
      1.00       -2.690           -4.135     15/60
      0.50       -2.262           -4.208     18/60
      0.25       -0.745           -4.297     32/60
      0.10       -0.344           -4.493     37/60

The mean frontier is monotone in both axes at all seven points: danger rises
from -4.0 m to contact while adversary realism falls only ~0.77 nats. Tilting
drives the collision rate from 8% to 62% for under one nat of the adversary's
own log p -- for contrast the temperature sweep gave up 4+ nats to move realism
a comparable amount, because it degraded the whole scene rather than one agent.
That gap is the structural advantage: danger bought cheaply by reweighting a
single agent on the token manifold.

Per scenario the response is noisy, as anticipated for a concentrated tilt:
danger is fully monotone in only 30/60, realism in 9/60 (against 60/60 for the
global temperature knob). A frontier point must therefore average over
scenarios -- which is how the curve above is built.
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
from smart.safety.objectives import proximity_danger
from smart.safety.scoring import SelfJudgeError, prepare_scenario
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging

BETAS = [1e9, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1]
NOM_LEN, NOM_WID = 4.8, 2.0


def boxes(centroids, headings):
    """(A, T, 2) + (A, T) -> (A, T, 4, 2) nominal vehicle boxes."""
    c, s = headings.cos(), headings.sin()
    hl, hw = NOM_LEN / 2, NOM_WID / 2
    corners = torch.stack([
        torch.stack([hl * c - hw * s, hl * s + hw * c], dim=-1),
        torch.stack([hl * c + hw * s, hl * s - hw * c], dim=-1),
        torch.stack([-hl * c + hw * s, -hl * s - hw * c], dim=-1),
        torch.stack([-hl * c - hw * s, -hl * s + hw * c], dim=-1),
    ], dim=-2)
    return centroids[..., None, :] + corners


def achieved_danger(pred, adv, victim):
    ab = boxes(pred['pred_traj'][adv][None], pred['pred_head'][adv][None])
    vb = boxes(pred['pred_traj'][victim][None], pred['pred_head'][victim][None])
    return float(proximity_danger(ab, vb))


def pick_adversary(data, victim):
    is_veh = data['agent']['type'] == 0
    valid = data['agent']['valid_mask'][:, 10]
    cand = is_veh & valid
    cand[victim] = False
    if not cand.any():
        return None
    pos = data['agent']['token_pos']
    d = (pos[:, 1, :2] - pos[victim, 1, :2]).norm(dim=-1)
    d[~cand] = float('inf')
    return int(d.argmin())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/validation/validation_scalable.yaml')
    parser.add_argument('--generator_ckpt', type=str, default='checkpoints/epoch=31.ckpt')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--num_scenarios', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--allow_self_judge', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    if not args.allow_self_judge:
        raise SelfJudgeError('coarse tilt sweep is self-judged; pass --allow_self_judge')

    config = load_config_act(args.config)
    config.Model.decoder.beam_size = 2048          # tilting needs full support
    model = SMART(config.Model)
    model.load_params_from_file(filename=args.generator_ckpt,
                                logger=Logging().log(level='DEBUG'))
    model = model.eval().to(args.device)

    dc = config.Dataset
    raw_dir = [args.data_dir] if args.data_dir else dc.val_raw_dir
    dataset = MultiDataset(root=dc.root, split='val', raw_dir=raw_dir,
                           processed_dir=dc.val_processed_dir,
                           transform=WaymoTargetBuilder(
                               config.Model.num_historical_steps,
                               config.Model.decoder.num_future_steps))

    rows = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0)
    for i, batch in enumerate(loader):
        if len(rows) >= args.num_scenarios:
            break
        data = prepare_scenario(model, batch.to(args.device), seed=args.seed)
        victim = int(data['agent']['av_index'])
        adv = pick_adversary(data, victim)
        if adv is None:
            continue
        mask = torch.zeros(data['agent'].num_nodes, dtype=torch.bool)
        mask[adv] = True
        num_steps = data['agent']['token_idx'].shape[1] - 2

        curve = []
        for beta in BETAS:
            torch.manual_seed(args.seed + i)
            with torch.no_grad():
                pred = model.inference(data, tilt_beta=beta,
                                       adversary_mask=mask, victim_index=victim)
            curve.append({
                'beta': beta,
                'danger': achieved_danger(pred, adv, victim),
                'adv_logp': float(pred['log_p'][adv] / num_steps),
            })
        rows.append({'scenario': str(data['scenario_id'][0]),
                     'adversary': adv, 'victim': victim, 'curve': curve})

    print(f'\nscenarios: {len(rows)}   adversary = nearest vehicle to the AV\n')
    print(f'  {"beta":>8} {"danger (m)":>12} {"adv log p/step":>16}')
    for k, beta in enumerate(BETAS):
        dg = sum(r['curve'][k]['danger'] for r in rows) / len(rows)
        lp = sum(r['curve'][k]['adv_logp'] for r in rows) / len(rows)
        tag = '  (off)' if beta >= 1e8 else ''
        print(f'  {beta:>8.2f} {dg:>12.3f} {lp:>16.3f}{tag}')

    print('\n  !! CIRCULAR: self-judged. danger is separation-based: '
          'higher = closer, >0 means boxes overlap (collision).')

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'betas': BETAS, 'rows': rows}, f, indent=2)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
