"""Animate the log's traffic against SMART's, with and without agents entering.

The question this answers is what a closed population looks like. The agent set
used to be fixed at t=0 -- the history window decided it, and the history was
written from the simulation's own agents -- so no car could join a scenario
already in progress, and over fifteen seconds the traffic thinned to less than
half of what the log has.

Three panels, one variable. The ego follows the log in all of them, so nothing
differs except how the background traffic is produced: the log itself, SMART
with the population frozen at t=0, and SMART with entry and retirement. Driving
the observation directly rather than through a simulation keeps it that way --
a closed-loop planner would react to the traffic and the ego paths would
diverge, which is a fair comparison of outcomes but a useless one of pictures.

    PYTHONPATH=. python scripts/render_traffic_comparison.py --scenarios 3
"""
import argparse
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

from nuplan.common.actor_state.state_representation import Point2D, TimePoint
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration)

from smart.nuplan.scenarios import build_scenario, find_scenarios
from smart.nuplan.smart_agents import SMARTAgents, load_smart
from smart.occlusion.visibility import boxes_to_corners

DATA = '/mnt/e/nuplan-mini/nuplan-v1.1_mini/data/cache/mini'
MAPS = '/mnt/e/nuplan-mini/nuplan-maps-v1.0/maps'
LANE = '#c9c9c9'
EGO = '#1f77b4'
KEPT = '#4c9f70'      # present from the start
ENTERED = '#d1495b'   # joined partway through -- the case that was missing
MOVING = (TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN,
          TrackedObjectType.BICYCLE)


def boxes(objects):
    objects = [o for o in objects if o.tracked_object_type in MOVING]
    if not objects:
        return torch.zeros(0, 5), []
    data = torch.tensor([[o.box.center.x, o.box.center.y, o.box.center.heading,
                          o.box.width, o.box.length] for o in objects])
    return data, [str(o.track_token or o.token) for o in objects]


def lanes(scenario, centre, radius):
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    try:
        nearby = scenario.map_api.get_proximal_map_objects(centre, radius, layers)
    except Exception:
        return []
    out = []
    for layer in layers:
        for lane in nearby.get(layer, []):
            try:
                out.append([(s.x, s.y) for s in lane.baseline_path.discrete_path])
            except Exception:
                pass
    return out


def roll(scenario, model, steps, closed):
    """Drive the observation forward, returning per-step objects and ego."""
    observation = SMARTAgents(model, scenario, device='cuda')
    if closed:
        # The old behaviour, reproduced by disabling the two new steps rather
        # than by checking out the old file: same code path otherwise.
        observation._admit_entering_agents = lambda: None
        observation._retire_distant_agents = lambda ego: None
    observation.initialize()
    frames = []
    for index in range(steps):
        ego = scenario.get_ego_state_at_iteration(index)
        observation.update_observation(
            SimulationIteration(TimePoint(int(index * 1e5)), index),
            SimulationIteration(TimePoint(int((index + 1) * 1e5)), index + 1),
            SimpleNamespace(current_state=(ego, None)))
        frames.append(list(observation.get_observation().tracked_objects))
    return frames


def render(scenario, panels, out_path, steps, radius, fps, stride):
    ego0 = scenario.get_ego_state_at_iteration(0).center
    paths = lanes(scenario, Point2D(ego0.x, ego0.y), radius * 2.5)
    figure, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.6), dpi=95)
    if len(panels) == 1:
        axes = [axes]

    def draw(index):
        ego = scenario.get_ego_state_at_iteration(index).center
        for ax, (title, frames, first_seen) in zip(axes, panels):
            ax.clear()
            for path in paths:
                ax.plot([p[0] for p in path], [p[1] for p in path],
                        color=LANE, linewidth=0.6, zorder=0)
            data, ids = boxes(frames[index])
            entered = 0
            if len(data):
                near = (data[:, :2] - torch.tensor([ego.x, ego.y])).norm(dim=-1) <= radius
                corners = boxes_to_corners(data[:, 0], data[:, 1], data[:, 2],
                                           data[:, 3], data[:, 4])
                for i, track in enumerate(ids):
                    if not near[i]:
                        continue
                    late = track not in first_seen
                    entered += late
                    ax.add_patch(Polygon(corners[i].numpy(), closed=True,
                                         facecolor=ENTERED if late else KEPT,
                                         edgecolor='#333', linewidth=0.4,
                                         alpha=0.85, zorder=2))
            ego_box = torch.tensor([[ego.x, ego.y, ego.heading, 2.3, 5.2]])
            ec = boxes_to_corners(ego_box[:, 0], ego_box[:, 1], ego_box[:, 2],
                                  ego_box[:, 3], ego_box[:, 4])[0]
            ax.add_patch(Polygon(ec.numpy(), closed=True, facecolor=EGO,
                                 edgecolor='white', linewidth=1.0, zorder=3))
            shown = int(near.sum()) if len(data) else 0
            ax.set_title(f'{title}\n{shown} agents, {entered} joined later',
                         fontsize=10)
            ax.set_xlim(ego.x - radius, ego.x + radius)
            ax.set_ylim(ego.y - radius, ego.y + radius)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        figure.suptitle(f't = {index * 0.1:.1f} s', fontsize=11)

    frames = list(range(0, steps, stride))
    FuncAnimation(figure, draw, frames=frames, interval=1000 / fps).save(
        out_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    return len(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', type=int, default=3)
    parser.add_argument('--tag', default='traversing_intersection')
    parser.add_argument('--steps', type=int, default=140)
    parser.add_argument('--radius', type=float, default=70.0)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--out-dir', default='traffic_gifs')
    parser.add_argument('--checkpoint', default='checkpoints/bosch_nuplan_smart.ckpt')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_smart(args.checkpoint, 'cuda')
    for i, entry in enumerate(find_scenarios(DATA, args.tag, args.scenarios,
                                             duration=(args.steps + 20) * 0.1)):
        scenario = build_scenario(entry, DATA, MAPS,
                                  duration=(args.steps + 20) * 0.1,
                                  scenario_type=args.tag)
        log = [list(scenario.get_tracked_objects_at_iteration(k).tracked_objects)
               for k in range(args.steps)]
        closed = roll(scenario, model, args.steps, closed=True)
        opened = roll(scenario, model, args.steps, closed=False)

        # Anything not present in the first frame joined later; colouring by
        # that is what makes the difference between the panels visible at all.
        start = {t for _, ids in [boxes(log[0])] for t in ids}
        panels = [('log', log, start),
                  ('SMART, population fixed at t=0', closed, start),
                  ('SMART, agents enter and leave', opened, start)]
        out = os.path.join(args.out_dir, f'{scenario.token}.gif')
        n = render(scenario, panels, out, args.steps, args.radius,
                   args.fps, args.stride)
        print(f'{scenario.token}: {n} frames -> {out}')


if __name__ == '__main__':
    main()
