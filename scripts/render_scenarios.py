"""Render danger-tilted scenarios: bird's-eye, off vs tilted, side by side.

Turns the frontier numbers into something you can look at: the same scenario
sampled with tilting off and with a small beta, so the adversary (red) can be
seen pressing into the victim (blue).
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMART
from smart.safety.objectives import proximity_danger
from smart.safety.scoring import prepare_scenario
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging

NOM_LEN, NOM_WID = 4.8, 2.0


def boxes(centroids, headings):
    c, s = headings.cos(), headings.sin()
    hl, hw = NOM_LEN / 2, NOM_WID / 2
    corners = torch.stack([
        torch.stack([hl * c - hw * s, hl * s + hw * c], dim=-1),
        torch.stack([hl * c + hw * s, hl * s - hw * c], dim=-1),
        torch.stack([-hl * c + hw * s, -hl * s - hw * c], dim=-1),
        torch.stack([-hl * c - hw * s, -hl * s + hw * c], dim=-1),
    ], dim=-2)
    return centroids[..., None, :] + corners


def closest_step(pred, adv, victim):
    ab = boxes(pred['pred_traj'][adv][None], pred['pred_head'][adv][None])
    vb = boxes(pred['pred_traj'][victim][None], pred['pred_head'][victim][None])
    from smart.safety.objectives import box_separation
    sep = box_separation(ab, vb)[0]           # (T,)
    return int(sep.argmin()), float(-sep.min())


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


def draw_panel(ax, data, pred, adv, victim, title):
    mp = data['map_point']['position'][:, :2].cpu().numpy()
    # window around the victim's path
    vt = pred['pred_traj'][victim].cpu().numpy()
    cx, cy = vt[:, 0].mean(), vt[:, 1].mean()
    R = 40.0
    m = (np.abs(mp[:, 0] - cx) < R) & (np.abs(mp[:, 1] - cy) < R)
    ax.scatter(mp[m, 0], mp[m, 1], s=0.5, c='#cccccc', linewidths=0, zorder=0)

    eval_mask = data['agent']['valid_mask'][:, 10]
    traj = pred['pred_traj'].cpu().numpy()
    for i in torch.nonzero(eval_mask).flatten().tolist():
        if i in (adv, victim):
            continue
        ax.plot(traj[i, :, 0], traj[i, :, 1], c='#9fb6cf', lw=0.8, zorder=1)

    step, danger = closest_step(pred, adv, victim)
    for idx, col, name in [(victim, '#1f6feb', 'victim (AV)'), (adv, '#e5484d', 'adversary')]:
        t = traj[idx]
        ax.plot(t[:, 0], t[:, 1], c=col, lw=2.2, zorder=3, label=name)
        ax.scatter(t[0, 0], t[0, 1], c=col, s=28, marker='o', zorder=4)
        box = boxes(pred['pred_traj'][idx][step][None], pred['pred_head'][idx][step][None])[0].cpu().numpy()
        ax.add_patch(MplPolygon(box, closed=True, facecolor=col, alpha=0.35,
                                edgecolor=col, lw=1.4, zorder=3))
    ax.set_xlim(cx - R, cx + R); ax.set_ylim(cy - R, cy + R)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    verdict = 'COLLISION' if danger > 0 else f'min gap {-danger:.1f} m'
    ax.set_title(f'{title}\n{verdict}', fontsize=11)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/validation/validation_scalable.yaml')
    parser.add_argument('--generator_ckpt', type=str, default='checkpoints/epoch=31.ckpt')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--tilt_topk', type=int, default=None,
                        help='restrict tilt to top-k for plausible motion')
    parser.add_argument('--scan', type=int, default=30, help='scenarios to search')
    parser.add_argument('--render', type=int, default=3, help='best to render')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()

    config = load_config_act(args.config)
    config.Model.decoder.beam_size = 2048
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

    os.makedirs(args.outdir, exist_ok=True)
    found = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    for i, batch in enumerate(loader):
        if i >= args.scan or len(found) >= args.render:
            break
        data = prepare_scenario(model, batch.to(args.device), seed=args.seed)
        victim = int(data['agent']['av_index'])
        adv = pick_adversary(data, victim)
        if adv is None:
            continue
        mask = torch.zeros(data['agent'].num_nodes, dtype=torch.bool); mask[adv] = True

        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            off = model.inference(data, tilt_beta=1e9, adversary_mask=mask, victim_index=victim)
        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            on = model.inference(data, tilt_beta=args.beta, adversary_mask=mask,
                                 victim_index=victim, tilt_topk=args.tilt_topk)

        _, d_off = closest_step(off, adv, victim)
        _, d_on = closest_step(on, adv, victim)
        if not (d_off < 0 and d_on > 0):     # want: safe when off, collision when tilted
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        draw_panel(axes[0], data, off, adv, victim, 'tilting off (beta = inf)')
        ttl = f'tilted (beta={args.beta}' + (f', top-{args.tilt_topk})' if args.tilt_topk else ')')
        draw_panel(axes[1], data, on, adv, victim, ttl)
        sid = str(data['scenario_id'][0])
        fig.suptitle(f'scenario {sid[:12]}   adversary log p/step: '
                     f'{float(off["log_p"][adv])/16:.2f} -> {float(on["log_p"][adv])/16:.2f}',
                     fontsize=12)
        fig.tight_layout()
        path = os.path.join(args.outdir, f'scenario_{len(found)}_{sid[:10]}.png')
        fig.savefig(path, dpi=120, bbox_inches='tight'); plt.close(fig)
        found.append(path)
        print(f'rendered {path}  (off gap {-d_off:.1f}m -> tilted danger {d_on:+.1f})')

    print(f'\ndone: {len(found)} scenarios')


if __name__ == '__main__':
    main()
