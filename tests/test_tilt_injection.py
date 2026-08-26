"""End-to-end tilted sampling inside the rollout.

Turning beta down must make the designated adversary press toward the victim,
while the likelihood stays the model's own verdict on whatever was sampled --
tilting reshapes q, not p.
"""
import pytest
import torch

from smart.safety.objectives import proximity_danger
from tests.test_inference_likelihood import _load

NOM_LEN, NOM_WID = 4.8, 2.0


def _boxes(centroids, headings):
    """(T,2)+(T,) -> (T,4,2) nominal vehicle boxes, corner order lf,rf,rb,lb."""
    c, s = headings.cos(), headings.sin()
    hl, hw = NOM_LEN / 2, NOM_WID / 2
    offs = torch.tensor([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    out = []
    for k in range(centroids.shape[0]):
        R = torch.tensor([[c[k], -s[k]], [s[k], c[k]]])
        out.append(centroids[k] + offs @ R.T)
    return torch.stack(out)


def _pick_adversary(data, victim):
    is_veh = data['agent']['type'] == 0
    valid = data['agent']['valid_mask'][:, 10]
    cand = is_veh & valid
    cand[victim] = False
    pos = data['agent']['token_pos']
    d = (pos[:, 1, :2] - pos[victim, 1, :2]).norm(dim=-1)
    d[~cand] = float('inf')
    return int(d.argmin())


def _achieved_danger(pred, adv, victim):
    adv_box = _boxes(pred['pred_traj'][adv], pred['pred_head'][adv])
    vic_box = _boxes(pred['pred_traj'][victim], pred['pred_head'][victim])
    return proximity_danger(adv_box, vic_box).item()


@pytest.fixture(scope="module")
def tilted_pair():
    model, data = _load(beam_size=2048)
    victim = int(data['agent']['av_index'])
    adv = _pick_adversary(data, victim)
    mask = torch.zeros(data['agent'].num_nodes, dtype=torch.bool)
    mask[adv] = True

    torch.manual_seed(0)
    with torch.no_grad():
        off = model.inference(data, tilt_beta=1e9, adversary_mask=mask, victim_index=victim)
    _, data2 = _load(beam_size=2048)
    torch.manual_seed(0)
    with torch.no_grad():
        on = model.inference(data2, tilt_beta=0.5, adversary_mask=mask, victim_index=victim)
    return off, on, adv, victim


def test_tilting_increases_adversary_danger(tilted_pair):
    off, on, adv, victim = tilted_pair
    assert _achieved_danger(on, adv, victim) > _achieved_danger(off, adv, victim)


def test_tilting_preserves_the_model_likelihood(tilted_pair):
    """log p of the tilted sample must equal a teacher-forced re-score: tilting
    changed what was drawn, not the model's probability for it."""
    off, on, adv, victim = tilted_pair
    model, data = _load(beam_size=2048)
    with torch.no_grad():
        re = model.inference(data, forced_tokens=on['next_token_idx'])
    assert torch.allclose(re['log_p'][adv], on['log_p'][adv], atol=1e-4)
