"""Run the occluded observation over a real nuPlan scenario through the devkit.

The unit tests build nuPlan objects by hand, which proves the wrapper's logic
but not that it survives contact with the real devkit: real scenarios, real
DetectionsTracks, the real observation interface with its history buffer. This
does that, and reports what the planner would actually have been told.

Seen and remembered objects are told apart by measurement age. The wrapper
gives a remembered object its original, now-stale timestamp rather than the
current one, so an object stamped with the current frame was actually seen and
anything older is memory. Python object identity looks like the more direct
test and is not one: the underlying observation rebuilds its objects on every
call, so nothing survives to be identical to anything.

Usage:
    PYTHONPATH=. python scripts/nuplan_occlusion_check.py \
        --data-root /mnt/e/nuplan-mini/nuplan-v1.1_mini/data/cache/mini \
        --map-root /mnt/e/nuplan-mini/nuplan-maps-v1.0/maps
"""
import argparse
import glob
import math
import os
import sqlite3
from collections import deque

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

from smart.nuplan.occluded_observation import OccludedObservation


def densest_log(data_root):
    """The log with the most tracked boxes per frame.

    Picking a log by file size finds the emptiest scenes -- the smallest logs
    are quiet roads with under two boxes a frame, where occlusion has nothing
    to hide behind and the check would pass without testing anything.
    """
    best = None
    for path in sorted(glob.glob(os.path.join(data_root, '*.db'))):
        connection = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        frames = connection.execute('select count(*) from lidar_pc').fetchone()[0]
        boxes = connection.execute('select count(*) from lidar_box').fetchone()[0]
        connection.close()
        if frames and (best is None or boxes / frames > best[0]):
            best = (boxes / frames, path)
    if best is None:
        raise SystemExit(f'no .db files under {data_root}')
    return best[1], best[0]


def open_scenario(db_path, data_root, map_root, duration):
    """Build a NuPlanScenario starting at the log's first lidar frame."""
    connection = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    token, timestamp = connection.execute(
        'select token, timestamp from lidar_pc order by timestamp limit 1').fetchone()
    # `location` is a loose label -- Las Vegas logs carry "las_vegas", which no
    # map is filed under. `map_version` holds the real map name. The two agree
    # in Pittsburgh and Boston, so using the wrong one stays invisible until
    # something actually loads the map.
    location, map_name = connection.execute(
        'select location, map_version from log').fetchone()
    connection.close()

    return NuPlanScenario(
        data_root=data_root,
        log_file_load_path=db_path,
        initial_lidar_token=token.hex(),
        initial_lidar_timestamp=timestamp,
        scenario_type='unknown',
        map_root=map_root,
        map_version='nuplan-maps-v1.0',
        map_name=map_name,
        scenario_extraction_info=ScenarioExtractionInfo(
            scenario_name='occlusion_check', scenario_duration=duration,
            extraction_offset=0.0, subsample_ratio=1.0),
        ego_vehicle_parameters=get_pacifica_parameters(),
    ), location


def step_through(scenario, observation, steps, counted=None, radius=None):
    """Drive the observation over the log and count what it hides.

    `counted` restricts the tally to certain object types. Occlusion still
    applies to everything -- a parked car hides what is behind it whether or
    not it is being counted -- but a 0.4 m cone going unseen says nothing about
    a planner, and static objects that never move are trivially remembered
    forever, which flatters the memory figure.

    `radius` must match the wrapper's own range limit. Without it, objects
    beyond sensor range fall into the hidden column, which conflates "too far
    away" with "behind something" -- and what is being measured here is
    occlusion, not range.
    """
    underlying = observation._observation
    observation.initialize()

    def keep(objects, ego):
        objects = [o for o in objects
                   if counted is None or o.tracked_object_type in counted]
        if radius is None:
            return objects
        return [o for o in objects
                if math.hypot(o.center.x - ego.center.x,
                              o.center.y - ego.center.y) <= radius]

    totals = [0, 0, 0]  # seen, remembered, hidden
    for index in range(min(steps, scenario.get_number_of_iterations() - 1)):
        ego = scenario.get_ego_state_at_iteration(index)
        full = keep(underlying.get_observation().tracked_objects, ego)
        given = keep(observation.get_observation().tracked_objects, ego)
        now = max((o.metadata.timestamp_us for o in full), default=0)

        seen = sum(1 for o in given if o.metadata.timestamp_us == now)
        totals[0] += seen
        totals[1] += len(given) - seen
        totals[2] += len(full) - seen

        ego_states = deque([ego])
        observations = deque([underlying.get_observation()])
        history = SimulationHistoryBuffer(ego_states, observations,
                                          sample_interval=scenario.database_interval)
        observation.update_observation(
            SimulationIteration(scenario.get_time_point(index), index),
            SimulationIteration(scenario.get_time_point(index + 1), index + 1),
            history)
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/mnt/e/nuplan-mini/nuplan-v1.1_mini/data/cache/mini')
    parser.add_argument('--map-root', default='/mnt/e/nuplan-mini/nuplan-maps-v1.0/maps')
    parser.add_argument('--db', default=None, help='defaults to the densest log')
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--duration', type=float, default=20.0)
    parser.add_argument('--memory-horizon', type=float, default=3.0)
    parser.add_argument('--radius', type=float, default=None,
                        help='optional sensor range in metres; off by default')
    parser.add_argument('--all-types', action='store_true',
                        help='count cones, barriers and debris as targets too')
    args = parser.parse_args()

    if args.db:
        db_path, density = args.db, float('nan')
    else:
        db_path, density = densest_log(args.data_root)

    scenario, location = open_scenario(db_path, args.data_root, args.map_root, args.duration)
    observation = OccludedObservation(TracksObservation(scenario), scenario,
                                      memory_horizon=args.memory_horizon,
                                      radius=args.radius)

    counted = None if args.all_types else {
        TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE}
    seen, remembered, hidden = step_through(scenario, observation, args.steps,
                                            counted, args.radius)
    total = seen + remembered + hidden

    print(f'{os.path.basename(db_path)}  ({location}, {density:.0f} boxes/frame)')
    print(f'{scenario.get_number_of_iterations()} iterations available, '
          f'stepped {min(args.steps, scenario.get_number_of_iterations() - 1)}\n')
    print(f'  {"seen":>12}  {seen:>7}  {seen / total:6.1%}   direct line of sight')
    print(f'  {"remembered":>12}  {remembered:>7}  {remembered / total:6.1%}   '
          f'occluded, still tracked')
    print(f'  {"hidden":>12}  {hidden:>7}  {hidden / total:6.1%}   '
          f'withheld from the planner')
    print(f'\n  a conventional benchmark would have handed over all {total}.')


if __name__ == '__main__':
    main()
