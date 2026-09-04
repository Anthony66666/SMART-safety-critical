"""Is the simulated traffic plausible? Measured against the recorded traffic.

The question a GIF answers by eye, in numbers. Log replay is the control: it is
the traffic that actually happened, so whatever it scores is what "plausible"
looks like on this metric, and a traffic model is doing well when it lands near
it rather than near zero. A model that freezes every car would score perfectly
on sideways motion and off-road rate, which is why those are reported next to
how far the agents actually travelled.

Only vehicles are judged on heading consistency. A pedestrian stepping
sideways is a pedestrian, not a failure, and including them buries the signal:
that alone was the difference between log replay looking broken and looking
fine.

Usage:
    PYTHONPATH=. python scripts/traffic_stats.py \
        --runs <dir> <dir> --labels "log replay" "SMART"
"""
import argparse
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.simulation_log import SimulationLog

STEP_S = 0.1


def find_logs(experiment_dir):
    pattern = os.path.join(experiment_dir, 'simulation_log', '**', '*.msgpack.xz')
    return {os.path.basename(p).split('.')[0]: p
            for p in glob.glob(pattern, recursive=True)}


def track_states(history):
    """{track id: [(x, y, heading, t)]} for vehicles, over one simulation."""
    tracks = defaultdict(list)
    for step, sample in enumerate(history):
        for o in sample.observation.tracked_objects:
            if o.tracked_object_type != TrackedObjectType.VEHICLE:
                continue
            tracks[o.track_token or o.token].append(
                (o.box.center.x, o.box.center.y, o.box.center.heading, step))
    return tracks


def measure(log, map_api):
    history = log.simulation_history.data
    tracks = track_states(history)

    lateral, forward, speeds, jumps = [], [], [], 0
    for states in tracks.values():
        for (x0, y0, h0, t0), (x1, y1, h1, t1) in zip(states, states[1:]):
            if t1 - t0 != 1:
                continue
            dx, dy = x1 - x0, y1 - y0
            step = (dx * dx + dy * dy) ** 0.5
            speeds.append(step / STEP_S)
            if step < 0.05:          # heading is meaningless when standing still
                continue
            forward.append(dx * np.cos(h0) + dy * np.sin(h0))
            lateral.append(abs(-dx * np.sin(h0) + dy * np.cos(h0)))
            if step / STEP_S > 40.0:  # 144 km/h between two frames
                jumps += 1

    lateral = np.array(lateral)
    forward = np.array(forward)
    speeds = np.array(speeds)
    # Sideways: lateral displacement exceeding forward, while actually moving.
    sideways = float((lateral > np.abs(forward)).mean()) if len(lateral) else 0.0
    reversing = float((forward < 0).mean()) if len(forward) else 0.0

    # Off-road, sampled at the last frame to keep the map queries bounded.
    off_road, checked = 0, 0
    last = history[-1]
    for o in last.observation.tracked_objects:
        if o.tracked_object_type != TrackedObjectType.VEHICLE:
            continue
        checked += 1
        try:
            from nuplan.common.actor_state.state_representation import Point2D
            point = Point2D(o.box.center.x, o.box.center.y)
            on = (map_api.is_in_layer(point, SemanticMapLayer.LANE)
                  or map_api.is_in_layer(point, SemanticMapLayer.INTERSECTION))
        except Exception:
            continue
        off_road += 0 if on else 1

    return {
        'vehicles': len(tracks),
        'median speed (m/s)': float(np.median(speeds)) if len(speeds) else 0.0,
        'p95 speed (m/s)': float(np.percentile(speeds, 95)) if len(speeds) else 0.0,
        'sideways %': sideways * 100,
        'reversing %': reversing * 100,
        'teleports': jumps,
        'off-road % (final frame)': (off_road / checked * 100) if checked else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', default=None)
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(r.rstrip('/')) for r in args.runs]
    found = [find_logs(r) for r in args.runs]
    shared = sorted(set(found[0]).intersection(*[set(f) for f in found[1:]]))
    if not shared:
        raise SystemExit('the runs have no scenario tokens in common')

    totals = {label: defaultdict(list) for label in labels}
    for token in shared:
        for label, f in zip(labels, found):
            log = SimulationLog.load_data(Path(f[token]))
            for key, value in measure(log, log.scenario.map_api).items():
                totals[label][key].append(value)

    keys = list(next(iter(totals.values())).keys())
    width = max(len(k) for k in keys) + 2
    print(f'{len(shared)} 个场景，按场景取平均\n')
    print(' ' * width + ''.join(f'{l:>22s}' for l in labels))
    for key in keys:
        row = ''.join(f'{np.mean(totals[l][key]):>22.2f}' for l in labels)
        print(f'{key:<{width}}' + row)


if __name__ == '__main__':
    main()
