"""Convert nuPlan logs into scenarios SMART can read.

Writes one pickle per scenario in the WOMD-derived schema, ready for
TokenProcessor. Conversion needs nuplan-devkit; tokenizing and training need
torch_geometric. Those rarely live in the same environment, and they do not
have to -- the output is a plain dict of tensors, so the two halves can be run
from different interpreters.

Usage:
    PYTHONPATH=. python scripts/convert_nuplan.py --out data/nuplan_converted -n 20
"""
import argparse
import glob
import os
import pickle
import sqlite3
import traceback

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo

from smart.nuplan.converter import NUPLAN_STRIDE, WOMD_STEPS, convert_scenario


def scenario_starts(db_path, steps, stride, spacing):
    """Lidar frames to start scenarios at, spaced out along the log."""
    connection = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    frames = connection.execute(
        'select token, timestamp from lidar_pc order by timestamp').fetchall()
    # `location` is a loose label; `map_version` is the name maps are filed
    # under, and the two differ in Las Vegas.
    map_name = connection.execute('select map_version from log').fetchone()[0]
    connection.close()

    needed = steps * stride
    return [(token.hex(), timestamp, map_name)
            for token, timestamp in frames[:max(0, len(frames) - needed):spacing]]


def build(db_path, data_root, map_root, token, timestamp, map_name, duration):
    return NuPlanScenario(
        data_root=data_root,
        log_file_load_path=db_path,
        initial_lidar_token=token,
        initial_lidar_timestamp=timestamp,
        scenario_type='unknown',
        map_root=map_root,
        map_version='nuplan-maps-v1.0',
        map_name=map_name,
        scenario_extraction_info=ScenarioExtractionInfo(
            scenario_name='converted', scenario_duration=duration,
            extraction_offset=0.0, subsample_ratio=1.0),
        ego_vehicle_parameters=get_pacifica_parameters(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/mnt/e/nuplan-mini/nuplan-v1.1_mini/data/cache/mini')
    parser.add_argument('--map-root', default='/mnt/e/nuplan-mini/nuplan-maps-v1.0/maps')
    parser.add_argument('--out', default='data/nuplan_converted')
    parser.add_argument('-n', '--num-scenarios', type=int, default=20)
    parser.add_argument('--steps', type=int, default=WOMD_STEPS)
    parser.add_argument('--stride', type=int, default=NUPLAN_STRIDE)
    parser.add_argument('--spacing', type=int, default=400,
                        help='lidar frames between consecutive scenario starts')
    parser.add_argument('--map-radius', type=float, default=150.0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    duration = args.steps * args.stride * 0.05 + 1.0

    written, failed = 0, 0
    for db_path in sorted(glob.glob(os.path.join(args.data_root, '*.db'))):
        if written >= args.num_scenarios:
            break
        for token, timestamp, map_name in scenario_starts(
                db_path, args.steps, args.stride, args.spacing):
            if written >= args.num_scenarios:
                break
            try:
                scenario = build(db_path, args.data_root, args.map_root,
                                 token, timestamp, map_name, duration)
                data = convert_scenario(scenario, args.steps, args.stride,
                                        args.map_radius)
                path = os.path.join(args.out, f"{data['scenario_id']}.pkl")
                with open(path, 'wb') as f:
                    pickle.dump(data, f)
                written += 1
                agent = data['agent']
                print(f"{data['scenario_id']}  {agent['num_nodes']:4d} agents  "
                      f"{data['map_polygon']['num_nodes']:4d} polygons  "
                      f"{data['map_point']['num_nodes']:6d} points  {data['city']}")
            except Exception as error:
                failed += 1
                print(f'  skipped {os.path.basename(db_path)} @ {timestamp}: '
                      f'{type(error).__name__}: {error}')
                if os.environ.get('CONVERT_TRACEBACK'):
                    traceback.print_exc()

    print(f'\nwrote {written} scenarios to {args.out}'
          + (f', {failed} skipped' if failed else ''))


if __name__ == '__main__':
    main()
