"""Select nuPlan scenarios by the dataset's own scenario types.

Picking a starting frame arbitrarily gets you a stopped car. `stationary` is the
most common tag in the mini split by a wide margin -- 62k tagged frames against
20k for `traversing_intersection` -- so an unguided choice lands on an ego
waiting at a light, where the expert itself moves under four metres and no
perception assumption can change any outcome.

nuPlan tags every lidar frame with the situations it belongs to, and published
nuPlan results are reported per scenario type. Selecting the same way keeps this
benchmark comparable with them, and lets occlusion be measured where it should
matter -- crossing an intersection, or approaching a crosswalk -- rather than
averaged into a corpus of parked cars.

The `.db` files are SQLite, so the tags are read directly rather than through
the devkit's scenario builder, which wants a hydra configuration and a worker
pool to do the same query.
"""
import glob
import os
import sqlite3

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo

# Types where a planner is moving and something could be hidden from it. The
# occlusion-relevant ones first: an intersection is where sight lines are broken
# by cross traffic, and a crosswalk is where a pedestrian steps out from behind
# something. `stationary` is deliberately absent.
OCCLUSION_RELEVANT = (
    'traversing_intersection',
    'traversing_traffic_light_intersection',
    'near_pedestrian_on_crosswalk',
    'high_magnitude_speed',
    'following_lane_with_lead',
)

DATABASE_INTERVAL = 0.05  # nuPlan logs run at 20 Hz


def _frames(connection):
    """Every lidar frame in the log, in time order, with its index."""
    return connection.execute(
        'select token, timestamp from lidar_pc order by timestamp').fetchall()


def find_scenarios(data_root, scenario_type, count, duration,
                   history_seconds=2.0, spacing=200):
    """Locate frames tagged `scenario_type` with room to simulate around them.

    Args:
        data_root: directory of `.db` logs.
        scenario_type: one of nuPlan's tags, e.g. `traversing_intersection`.
        count: how many scenarios to return.
        duration: seconds of simulation needed after the start frame.
        history_seconds: seconds of past the simulation preloads before it will
            step. A scenario anchored too near the start of a log cannot fill
            that buffer and the devkit refuses to run it.
        spacing: minimum frames between two scenarios from the same log, so a
            single tagged stretch does not supply all of them.

    Returns:
        List of (db_path, token_hex, timestamp, map_name).
    """
    needed_after = int(duration / DATABASE_INTERVAL) + 1
    needed_before = int(history_seconds / DATABASE_INTERVAL) + 1

    found = []
    for path in sorted(glob.glob(os.path.join(data_root, '*.db'))):
        if len(found) >= count:
            break
        connection = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        try:
            tagged = {row[0] for row in connection.execute(
                'select lidar_pc_token from scenario_tag where type = ?',
                (scenario_type,))}
            if not tagged:
                continue
            frames = _frames(connection)
            # `location` is a loose label; `map_version` is what maps are filed
            # under, and the two differ in Las Vegas.
            map_name = connection.execute(
                'select map_version from log').fetchone()[0]
        finally:
            connection.close()

        last = -spacing
        for index, (token, timestamp) in enumerate(frames):
            if len(found) >= count:
                break
            if (token not in tagged or index - last < spacing
                    or index < needed_before
                    or index + needed_after >= len(frames)):
                continue
            found.append((path, token.hex(), timestamp, map_name))
            last = index
    return found


def build_scenario(entry, data_root, map_root, duration, scenario_type='unknown'):
    """A NuPlanScenario for one entry from `find_scenarios`."""
    path, token, timestamp, map_name = entry
    return NuPlanScenario(
        data_root=data_root,
        log_file_load_path=path,
        initial_lidar_token=token,
        initial_lidar_timestamp=timestamp,
        scenario_type=scenario_type,
        map_root=map_root,
        map_version='nuplan-maps-v1.0',
        map_name=map_name,
        scenario_extraction_info=ScenarioExtractionInfo(
            scenario_name=scenario_type, scenario_duration=duration,
            extraction_offset=0.0, subsample_ratio=1.0),
        ego_vehicle_parameters=get_pacifica_parameters(),
    )


def expert_distance(scenario):
    """How far the logged ego travels over the scenario.

    A scenario where the expert barely moves cannot show anything about
    occlusion, whatever it is tagged as, so this is worth checking before
    spending a simulation on it.
    """
    first = scenario.get_ego_state_at_iteration(0).center
    last = scenario.get_ego_state_at_iteration(
        scenario.get_number_of_iterations() - 1).center
    return float(((last.x - first.x) ** 2 + (last.y - first.y) ** 2) ** 0.5)
