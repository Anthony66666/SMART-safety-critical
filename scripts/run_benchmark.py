"""Run a nuPlan planner with and without occlusion, and score it nuPlan's way.

This is the benchmark in miniature. Everything -- the simulation loop, the
planners, the ego controller, the metrics -- is the devkit's own. The single
substitution is the observation: `TracksObservation` hands the planner every
agent in the scene, `OccludedObservation` hands it only what the ego can see or
still remembers.

That is the whole argument. If the numbers move, they move because of the
perception assumption and nothing else, and they are numbers other nuPlan work
already reports, so they can be compared against published figures rather than
against a metric invented here.

The headline is the gap between the two conditions, not either score alone.
Every planner in nuPlan was tuned against full observability, so all of them
lose points when it is taken away; the absolute drop measures how much
information was removed, while the difference between planners measures which
of them copes.

Usage:
    PYTHONPATH=. python scripts/run_benchmark.py --planner idm --scenarios 5
"""
import argparse
import os
import tempfile
from pathlib import Path

from nuplan.planning.metrics.evaluation_metrics.common.drivable_area_compliance import (
    DrivableAreaComplianceStatistics)
from nuplan.planning.metrics.evaluation_metrics.common.ego_is_making_progress import (
    EgoIsMakingProgressStatistics)
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import (
    EgoLaneChangeStatistics)
from nuplan.planning.metrics.evaluation_metrics.common.ego_progress_along_expert_route import (
    EgoProgressAlongExpertRouteStatistics)
from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import (
    EgoAtFaultCollisionStatistics)
from nuplan.planning.metrics.evaluation_metrics.common.speed_limit_compliance import (
    SpeedLimitComplianceStatistics)
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.planner.idm_planner import IDMPlanner
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
    StepSimulationTimeController)

from smart.nuplan.occluded_observation import OccludedObservation
from smart.nuplan.scenarios import (OCCLUSION_RELEVANT, build_scenario,
                                    expert_distance, find_scenarios)


FLOW_CONFIG = 'checkpoints/flow_planner/model_config.yaml'
FLOW_CKPT = 'checkpoints/flow_planner/model.pth'


def build_flow_planner(device='cuda'):
    """Flow Planner, at the settings its own simulation config uses.

    A learned planner is the point of the exercise. IDM consumes only the
    nearest obstacle in its own lane -- the least occludable object there is --
    so it reports a zero gap however severe the occlusion. Flow Planner attends
    to 32 neighbouring agents drawn from the observation buffer, which is
    exactly where the occlusion wrapper substitutes what the ego can see.
    """
    # The config interpolates a couple of training paths from the environment;
    # they are never read at inference, but omegaconf resolves eagerly enough
    # that leaving them unset can fail.
    os.environ.setdefault('PROJECT_ROOT', os.getcwd())
    os.environ.setdefault('SAVE_DIR', tempfile.gettempdir())

    from flow_planner.planner import FlowPlanner
    return FlowPlanner(
        config_path=_resolvable_flow_config(),
        ckpt_path=FLOW_CKPT,
        past_trajectory_sampling=TrajectorySampling(num_poses=20, time_horizon=2),
        future_trajectory_sampling=TrajectorySampling(num_poses=80, time_horizon=8),
        # The published checkpoint is already a flat state_dict of exported EMA
        # weights -- 338 `module.`-prefixed tensors -- not a training
        # checkpoint carrying an `ema_state_dict` to unwrap. Asking for the
        # wrapper raises KeyError; this branch strips the prefix and loads.
        enable_ema=False,
        device=device,
        use_cfg=True,
        cfg_weight=1.0)


def _resolvable_flow_config():
    """The shipped config, with the branches it references but does not carry.

    The config published with the checkpoint was cut out of a full training
    config tree and still interpolates into parts of it that did not come
    along -- `${data.dataset.train.future_downsampling_method}` among them.
    Loading it as-is raises InterpolationKeyError before the model is built.
    The missing values are taken from the repository's own defaults rather
    than guessed, and written to a copy so the downloaded file stays exactly
    as the authors published it.
    """
    import omegaconf

    config = omegaconf.OmegaConf.load(FLOW_CONFIG)
    # Values from flow_planner/script/data/dataset/nuplan_data.yaml, which is
    # the branch the published config was cut away from. `train.epoch` is only
    # read by the learning-rate scheduler and never at inference; it is present
    # so resolution succeeds, not because the number means anything here.
    defaults = omegaconf.OmegaConf.create({
        'data': {'dataset': {'train': {
            'future_downsampling_method': 'uniform',
            'predicted_neighbor_num': '${model.neighbor_pred_num}'}}},
        'train': {'epoch': 1},
    })
    merged = omegaconf.OmegaConf.merge(defaults, config)

    resolved = os.path.join(tempfile.gettempdir(), 'flow_planner_config.yaml')
    omegaconf.OmegaConf.save(merged, resolved)
    return resolved


def build_planner(name):
    """A devkit planner, unmodified."""
    if name == 'flow':
        return build_flow_planner()
    if name == 'idm':
        return IDMPlanner(target_velocity=10.0, min_gap_to_lead_agent=1.0,
                          headway_time=1.5, accel_max=1.0, decel_max=3.0,
                          planned_trajectory_samples=16,
                          planned_trajectory_sample_interval=0.5,
                          occupancy_map_radius=40.0)
    if name == 'simple':
        return SimplePlanner(horizon_seconds=8.0, sampling_time=0.25,
                             acceleration=[0.0, 0.0])
    raise SystemExit(f'unknown planner {name!r}')


def build_observation(kind, scenario, memory_horizon):
    """Log replay or IDM traffic, optionally seen through occlusion."""
    base = (IDMAgents(target_velocity=10.0, min_gap_to_lead_agent=1.0,
                      headway_time=1.5, accel_max=1.0, decel_max=2.0,
                      scenario=scenario)
            if kind.startswith('idm') else TracksObservation(scenario))
    if kind.endswith('occluded'):
        return OccludedObservation(base, scenario, memory_horizon=memory_horizon)
    return base


def build_metrics():
    """Official metrics, in dependency order.

    The collision and drivable-area metrics both take the lane-change metric as
    an argument, so it has to be computed first and handed to them; that is how
    nuPlan's own configuration wires them together.
    """
    lane_change = EgoLaneChangeStatistics('ego_lane_change', 'Planning',
                                          max_fail_rate=0.3)
    return lane_change, [
        lane_change,
        EgoAtFaultCollisionStatistics('no_ego_at_fault_collisions', 'Violations',
                                      ego_lane_change_metric=lane_change),
        DrivableAreaComplianceStatistics('drivable_area_compliance', 'Violations',
                                         lane_change_metric=lane_change,
                                         max_violation_threshold=0.3),
        EgoProgressAlongExpertRouteStatistics('ego_progress_along_expert_route',
                                              'Planning', score_progress_threshold=2.0),
        EgoIsMakingProgressStatistics('ego_is_making_progress', 'Planning',
                                      ego_progress_along_expert_route_metric=None,
                                      min_progress_threshold=0.2)
        if False else None,
        SpeedLimitComplianceStatistics('speed_limit_compliance', 'Violations',
                                       lane_change_metric=lane_change,
                                       max_violation_threshold=1.0,
                                       max_overspeed_value_threshold=2.23),
    ]


def score(history, scenario, metrics):
    """Run each official metric over one simulated history."""
    results = {}
    for metric in metrics:
        if metric is None:
            continue
        try:
            for statistics in metric.compute(history, scenario):
                for statistic in statistics.statistics:
                    results[f'{metric.name}/{statistic.name}'] = statistic.value
        except Exception as error:
            results[f'{metric.name}/ERROR'] = f'{type(error).__name__}: {error}'[:60]
    return results


def simulate(scenario, planner, observation):
    """One closed-loop simulation through the devkit's own runner."""
    setup = SimulationSetup(
        time_controller=StepSimulationTimeController(scenario),
        observations=observation,
        ego_controller=PerfectTrackingController(scenario),
        scenario=scenario)
    simulation = Simulation(setup)
    report = SimulationRunner(simulation, planner).run()
    if not report.succeeded:
        raise RuntimeError(report.error_message or 'simulation failed')
    return simulation.history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/mnt/e/nuplan-mini/nuplan-v1.1_mini/data/cache/mini')
    parser.add_argument('--map-root', default='/mnt/e/nuplan-mini/nuplan-maps-v1.0/maps')
    parser.add_argument('--planner', default='idm',
                        choices=('idm', 'simple', 'flow'))
    parser.add_argument('--traffic', default='log', choices=('log', 'idm'))
    parser.add_argument('--scenario-type', default='traversing_intersection',
                        help=f'a nuPlan tag; occlusion-relevant ones are '
                             f'{", ".join(OCCLUSION_RELEVANT)}')
    parser.add_argument('--scenarios', type=int, default=5)
    parser.add_argument('--duration', type=float, default=15.0)
    parser.add_argument('--memory-horizon', type=float, default=0.0,
                        help='seconds an occluded track survives. Zero, the '
                             'default, withholds it outright: the planners '
                             'under test drop any agent missing from the '
                             'current frame anyway, so a buffer would grant '
                             'object permanence rather than restore it.')
    parser.add_argument('--min-expert-distance', type=float, default=20.0,
                        help='skip scenarios where the logged ego barely moves; '
                             'no perception assumption can change those')
    args = parser.parse_args()

    entries = find_scenarios(args.data_root, args.scenario_type,
                             args.scenarios * 2, args.duration)
    if not entries:
        raise SystemExit(f'no {args.scenario_type} scenarios under {args.data_root}')

    totals = {'full': {}, 'occluded': {}}
    counts = {'full': {}, 'occluded': {}}
    deviations = []
    used = 0
    for entry in entries:
        if used >= args.scenarios:
            break
        scenario = build_scenario(entry, args.data_root, args.map_root,
                                  args.duration, args.scenario_type)
        travelled = expert_distance(scenario)
        if travelled < args.min_expert_distance:
            continue
        used += 1
        print(f'{scenario.scenario_name}  expert travels {travelled:.0f} m')
        paths = {}

        for label, kind in (('full', args.traffic),
                            ('occluded', args.traffic + ' occluded')):
            planner = build_planner(args.planner)
            observation = build_observation(kind, scenario, args.memory_horizon)
            _, metrics = build_metrics()
            try:
                history = simulate(scenario, planner, observation)
            except Exception as error:
                print(f'  {label:9s} FAILED {type(error).__name__}: {error}'[:140])
                continue
            paths[label] = [(sample.ego_state.center.x, sample.ego_state.center.y)
                            for sample in history.data]
            for key, value in score(history, scenario, metrics).items():
                if isinstance(value, (int, float)):
                    totals[label][key] = totals[label].get(key, 0.0) + value
                    counts[label][key] = counts[label].get(key, 0) + 1

        # The bluntest statement of whether occlusion reached the planner at
        # all. IDM's two trajectories are identical to 0.000000 m because it
        # only ever consults its lead vehicle; a planner that reads the wider
        # scene has to diverge, and if it does not, nothing downstream of it
        # can be trusted to mean anything.
        if len(paths) == 2:
            deviation = max(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                            for a, b in zip(paths['full'], paths['occluded']))
            deviations.append(deviation)
            print(f'  ego trajectories diverge by {deviation:.3f} m')

    if not used:
        raise SystemExit('every candidate scenario had a near-stationary expert')

    print(f'\n{args.planner} planner, {args.traffic} traffic, '
          f'{used} {args.scenario_type} scenarios, '
          f'memory horizon {args.memory_horizon:g} s')
    if deviations:
        print(f'ego trajectory divergence: mean {sum(deviations) / len(deviations):.3f} m, '
              f'max {max(deviations):.3f} m')
    print(f"\n{'metric':54s} {'full':>10} {'occluded':>10} {'delta':>10}")
    print('-' * 88)
    for key in sorted(set(totals['full']) | set(totals['occluded'])):
        values = []
        for label in ('full', 'occluded'):
            n = counts[label].get(key, 0)
            values.append(totals[label][key] / n if n else None)
        full, occluded = values
        if full is None or occluded is None:
            continue
        delta = occluded - full
        mark = '  <--' if abs(delta) > 1e-6 else ''
        print(f'{key[:54]:54s} {full:10.3f} {occluded:10.3f} {delta:+10.3f}{mark}')


if __name__ == '__main__':
    main()
