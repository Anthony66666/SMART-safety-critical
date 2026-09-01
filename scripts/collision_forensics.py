"""Find what the ego hit, and whether it had been told about it.

A scenario that scores zero on no_ego_at_fault_collisions says a collision
happened, not what it means. The claim this benchmark makes is that occlusion
caused it, and that stands or falls on one fact: was the object the ego struck
in the observation the planner received, or was it withheld?

If it was withheld, the ego drove into something it had no way of knowing was
there. If it was handed over, occlusion is not the explanation and something
else is wrong -- which is worth finding out before the number is quoted.

Collision geometry is the devkit's own `in_collision`, not a reimplementation,
so this agrees with the metric that flagged the scenario in the first place.

    PYTHONPATH=. python scripts/collision_forensics.py \
        --baseline <run_dir> --occluded <run_dir> --tokens a,b,c
"""
import argparse
from pathlib import Path

from nuplan.common.actor_state.oriented_box import in_collision
from nuplan.planning.simulation.simulation_log import SimulationLog

from scripts.render_nuplan_gif import find_logs, track_ids


def first_collision(samples):
    """The first frame where the ego box overlaps a tracked object."""
    for index, sample in enumerate(samples):
        ego_box = sample.ego_state.car_footprint.oriented_box
        for obj in sample.observation.tracked_objects:
            if in_collision(ego_box, obj.box):
                return index, obj
    return None, None


def report(token, baseline_log, occluded_log):
    base = baseline_log.simulation_history.data
    occ = occluded_log.simulation_history.data
    scenario = occluded_log.scenario

    # The occluded run's observation is the filtered one, so a collision has to
    # be looked for against the ground truth the log holds, not against what
    # the planner was shown.
    truth = [scenario.get_tracked_objects_at_iteration(i) for i in range(len(occ))]
    samples = [type('S', (), {'ego_state': occ[i].ego_state,
                              'observation': truth[i]})()
               for i in range(len(occ))]

    index, hit = first_collision(samples)
    print(f'\n=== {token} ===')
    if hit is None:
        print('  no ego/object overlap found in the occluded run')
        return

    given = track_ids(occ[index].observation)
    identity = hit.track_token or hit.token
    was_given = identity in given
    ego = occ[index].ego_state

    print(f'  collision at step {index}  (t={index * 0.1:.1f}s)')
    print(f'  struck a {str(hit.tracked_object_type).split(".")[-1]}')
    print(f'  ego speed {ego.dynamic_car_state.speed:.2f} m/s')
    print(f'  that object was {"HANDED OVER" if was_given else "WITHHELD"} '
          f'from the planner at that step')

    # Was the same object visible in the fully observable run at that step?
    if index < len(base):
        base_given = track_ids(base[index].observation)
        print(f'  in the baseline run it was '
              f'{"present" if identity in base_given else "absent"}')
        a, b = base[index].ego_state.center, ego.center
        print(f'  ego positions differ by '
              f'{((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5:.2f} m by then')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--occluded', required=True)
    parser.add_argument('--tokens', required=True)
    args = parser.parse_args()

    baseline_logs = find_logs(args.baseline)
    occluded_logs = find_logs(args.occluded)
    for token in [t.strip() for t in args.tokens.split(',') if t.strip()]:
        if token not in baseline_logs or token not in occluded_logs:
            print(f'{token}: not in both runs')
            continue
        report(token,
               SimulationLog.load_data(Path(baseline_logs[token])),
               SimulationLog.load_data(Path(occluded_logs[token])))


if __name__ == '__main__':
    main()
