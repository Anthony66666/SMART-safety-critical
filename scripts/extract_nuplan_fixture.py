"""Pull a window of a nuPlan log into the same shape as a WOMD scenario.

nuPlan logs are plain SQLite, so this reads them directly rather than through
nuplan-devkit. That is not a shortcut around the devkit -- the simulation side
will need it -- but it means occlusion geometry can be developed and tested
against real nuPlan traffic without the devkit, its dependency pins, or the
map files being installed anywhere.

The output deliberately mirrors the WOMD scenario dict this repo already uses
(position / heading / shape / velocity / valid_mask / type / av_index), so
visibility.py, occlusion_stats.py and render_occlusion.py run on nuPlan data
unchanged, and the occlusion statistics of the two datasets can be put side by
side without a second code path to trust.

Two nuPlan specifics are handled here rather than being left to callers:

- `ego_pose` stores the *rear axle*, not the box centre. On a 5.18 m Pacifica
  that is a 1.461 m error in where the ego is, which would bias every sight
  line cast from it, so the centre is reconstructed.
- Logs are 20 Hz. They are subsampled to 10 Hz by default to match WOMD, so a
  timestep means the same thing in both.

Usage:
    python scripts/extract_nuplan_fixture.py --db some_log.db --out fixture.pkl
"""
import argparse
import math
import os
import pickle
import sqlite3

import torch

# nuplan.common.actor_state.vehicle_parameters.get_pacifica_parameters()
EGO_WIDTH = 1.1485 * 2.0
EGO_FRONT_LENGTH = 4.049
EGO_REAR_LENGTH = 1.127
EGO_LENGTH = EGO_FRONT_LENGTH + EGO_REAR_LENGTH
EGO_HEIGHT = 1.777
REAR_AXLE_TO_CENTER = EGO_LENGTH / 2.0 - EGO_REAR_LENGTH

# Ordered so vehicle/pedestrian/bicycle keep the low indices WOMD gives them;
# the static classes have no WOMD counterpart and are appended.
TYPE_NAMES = ['vehicle', 'pedestrian', 'bicycle',
              'traffic_cone', 'barrier', 'czone_sign', 'generic_object']
TYPE_INDEX = {name: i for i, name in enumerate(TYPE_NAMES)}
VEHICLE = TYPE_INDEX['vehicle']


def quaternion_yaw(qw, qx, qy, qz):
    """Heading of a pose stored as a quaternion."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def read_frames(connection, start, count, stride):
    """The chosen lidar_pc frames, each with its ego pose, in time order."""
    rows = connection.execute(
        'select lidar_pc.token, ego_pose.x, ego_pose.y, ego_pose.qw, ego_pose.qx, '
        '       ego_pose.qy, ego_pose.qz, ego_pose.vx, ego_pose.vy, lidar_pc.timestamp '
        'from lidar_pc join ego_pose on lidar_pc.ego_pose_token = ego_pose.token '
        'order by lidar_pc.timestamp').fetchall()
    return rows[start:start + count * stride:stride]


def read_boxes(connection, frame_tokens):
    """Every tracked box in the chosen frames, keyed by frame then track."""
    types = {token: TYPE_INDEX[name] for token, name
             in connection.execute('select token, name from category')}
    track_type = {token: types[category] for token, category
                  in connection.execute('select token, category_token from track')}

    by_frame = {token: {} for token in frame_tokens}
    placeholders = ','.join('?' * len(frame_tokens))
    rows = connection.execute(
        f'select lidar_pc_token, track_token, x, y, yaw, width, length, height, vx, vy '
        f'from lidar_box where lidar_pc_token in ({placeholders})', list(frame_tokens))
    for frame, track, x, y, yaw, width, length, height, vx, vy in rows:
        by_frame[frame][track] = (x, y, yaw, length, width, height, vx, vy)
    return by_frame, track_type


def build_scenario(frames, by_frame, track_type):
    """Pack frames into dense [agent, time] tensors with the ego at av_index."""
    tracks = sorted({t for frame in frames for t in by_frame[frame[0]]})
    index = {track: i for i, track in enumerate(tracks)}
    av_index = len(tracks)
    n, steps = len(tracks) + 1, len(frames)

    position = torch.zeros(n, steps, 3)
    heading = torch.zeros(n, steps)
    shape = torch.zeros(n, steps, 3)
    velocity = torch.zeros(n, steps, 2)
    valid = torch.zeros(n, steps, dtype=torch.bool)
    kinds = torch.zeros(n, dtype=torch.long)

    for track, i in index.items():
        kinds[i] = track_type[track]
    kinds[av_index] = VEHICLE

    for t, (frame_token, ex, ey, qw, qx, qy, qz, evx, evy, _) in enumerate(frames):
        yaw = quaternion_yaw(qw, qx, qy, qz)
        position[av_index, t, 0] = ex + REAR_AXLE_TO_CENTER * math.cos(yaw)
        position[av_index, t, 1] = ey + REAR_AXLE_TO_CENTER * math.sin(yaw)
        heading[av_index, t] = yaw
        shape[av_index, t] = torch.tensor([EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT])
        velocity[av_index, t] = torch.tensor([evx, evy])
        valid[av_index, t] = True

        for track, (x, y, box_yaw, length, width, height, vx, vy) in by_frame[frame_token].items():
            i = index[track]
            position[i, t, 0], position[i, t, 1] = x, y
            heading[i, t] = box_yaw
            shape[i, t] = torch.tensor([length, width, height])
            velocity[i, t] = torch.tensor([vx, vy])
            valid[i, t] = True

    return {'agent': {'position': position, 'heading': heading, 'shape': shape,
                      'velocity': velocity, 'valid_mask': valid, 'type': kinds,
                      'av_index': av_index, 'type_names': TYPE_NAMES}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--start', type=int, default=0, help='first lidar_pc frame')
    parser.add_argument('--steps', type=int, default=91, help='timesteps to keep')
    parser.add_argument('--stride', type=int, default=2,
                        help='2 turns nuPlan 20 Hz into WOMD 10 Hz')
    args = parser.parse_args()

    connection = sqlite3.connect(f'file:{args.db}?mode=ro', uri=True)
    frames = read_frames(connection, args.start, args.steps, args.stride)
    if len(frames) < args.steps:
        raise SystemExit(f'log has only {len(frames)} frames from --start {args.start}')
    by_frame, track_type = read_boxes(connection, [f[0] for f in frames])
    scenario = build_scenario(frames, by_frame, track_type)
    location = connection.execute('select location from log').fetchone()[0]
    log_name = os.path.basename(args.db)[:-3]
    scenario['scenario_id'] = f'{log_name}@{args.start}'
    scenario['location'] = location
    scenario['source'] = os.path.basename(args.db)

    with open(args.out, 'wb') as f:
        pickle.dump(scenario, f)

    agent = scenario['agent']
    counts = torch.bincount(agent['type'], minlength=len(TYPE_NAMES))
    print(f"{args.out}: {agent['position'].shape[0]} agents x {args.steps} steps  ({location})")
    print(f"  mean visible per step: {agent['valid_mask'].sum(0).float().mean():.1f}")
    print('  ' + '  '.join(f'{name}={int(count)}'
                           for name, count in zip(TYPE_NAMES, counts) if count))


if __name__ == '__main__':
    main()
