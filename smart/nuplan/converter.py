"""nuPlan scenarios in the schema SMART was trained on.

SMART reads a WOMD-derived dict: agents as dense [agent, time] tensors, and the
map as polygons carrying polylines of typed points, wired together by
point-to-polygon and polygon-to-polygon edges. This produces exactly that from
a nuPlan scenario, so `TokenProcessor.preprocess` and the model itself need no
nuPlan-specific code path.

The schema constants here are not invented. They are read off the WOMD
preprocessing in data_preprocess.py at the repo root -- `_point_types`,
`_polygon_types`, `_polygon_light_type`, `_polygon_to_polygon_types` -- because
the checkpoint's embeddings were trained against those exact indices.
nn.Embedding sizes pin the ranges: map point types must stay under 17, polygon
types, sides, light states and agent types under 4. A nuPlan concept that does
not fit one of those slots cannot simply be given a new index.

Where the two datasets genuinely disagree, the choice is made here and stated:

- nuPlan records lane boundaries as geometry with no paint semantics. WOMD
  distinguishes eleven kinds of painted line. Boundaries therefore come across
  as EDGE rather than as an invented paint type, which loses information but
  fabricates none.

- WOMD marks a handful of agents as tracks_to_predict, which is what its
  `category` field carries. nuPlan has no such designation, so the ego gets
  category 3 and everything else 1. The field is currently inert -- every
  `agent_category == 3` test in the model is commented out -- so this costs
  nothing today, but it is a guess and would need revisiting if that changes.

- nuPlan logs run at 20 Hz and WOMD at 10 Hz, so iterations are taken with a
  stride of 2. The token vocabulary is built for 10 Hz motion; feeding it 20 Hz
  steps would halve every distance it sees.
"""
import math

import torch

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType

# Indices from data_preprocess.py `_polygon_types`.
POLYGON_VEHICLE, POLYGON_BIKE, POLYGON_BUS, POLYGON_PEDESTRIAN = 0, 1, 2, 3
# `_polygon_light_type`.
LIGHT_STOP, LIGHT_GO, LIGHT_CAUTION, LIGHT_UNKNOWN = 0, 1, 2, 3
# `_point_types`; only the three nuPlan can honestly fill are named.
PT_EDGE, PT_CROSSWALK, PT_CENTERLINE = 12, 15, 16
# `_polygon_to_polygon_types`.
PL2PL_PRED, PL2PL_SUCC, PL2PL_LEFT, PL2PL_RIGHT = 1, 2, 3, 4
# `_agent_types`.
AGENT_VEHICLE, AGENT_PEDESTRIAN, AGENT_CYCLIST, AGENT_BACKGROUND = 0, 1, 2, 3

AGENT_TYPE = {
    TrackedObjectType.VEHICLE: AGENT_VEHICLE,
    TrackedObjectType.PEDESTRIAN: AGENT_PEDESTRIAN,
    TrackedObjectType.BICYCLE: AGENT_CYCLIST,
}

LIGHT_STATE = {
    TrafficLightStatusType.RED: LIGHT_STOP,
    TrafficLightStatusType.GREEN: LIGHT_GO,
    TrafficLightStatusType.YELLOW: LIGHT_CAUTION,
}

WOMD_STEPS = 91
WOMD_CURRENT_STEP = 10
NUPLAN_STRIDE = 2


def convert_scenario(scenario, num_steps=WOMD_STEPS, stride=NUPLAN_STRIDE,
                     map_radius=150.0, include_boundaries=True):
    """Convert one nuPlan scenario into the dict SMART's preprocessing expects.

    Args:
        scenario: an AbstractScenario, normally a NuPlanScenario.
        num_steps: timesteps to emit. 91 matches WOMD, which is what the token
            vocabulary and the model's time axis were built for.
        stride: iterations per emitted step; 2 turns nuPlan's 20 Hz into 10 Hz.
        map_radius: metres around the ego at the current step to take map from.
        include_boundaries: emit lane boundaries as EDGE polylines as well as
            centrelines.

    Returns:
        A dict with the keys TokenProcessor.preprocess reads.
    """
    agent = _convert_agents(scenario, num_steps, stride)
    centre = agent['position'][agent['av_index'], WOMD_CURRENT_STEP, :2]
    origin = (float(centre[0]), float(centre[1]))
    map_data = _convert_map(scenario, Point2D(*origin), map_radius,
                            include_boundaries, origin)

    # Everything above is in global UTM and float64. Casting UTM straight to
    # float32 is what silently ruins this conversion: a northing near 4e6 has a
    # float32 spacing of 0.25 m, so lane geometry and agent positions get
    # quantised to a quarter of a metre, and consecutive polyline points
    # collapse onto each other. WOMD ships local coordinates in the low
    # thousands and never had the problem. Recentring on the ego puts nuPlan in
    # the same regime, where the spacing is well under a millimetre.
    agent['position'][..., 0] -= origin[0]
    agent['position'][..., 1] -= origin[1]
    # Invalid slots must stay exactly zero. They were never written, so the
    # subtraction above would leave them at minus the origin -- a position
    # several million metres away that looks like data. SMART's preprocessing
    # reads `position[:, current_step, 0] != 0` as "invalid but positioned" and
    # interpolates from it, so a non-zero invalid slot does not fail loudly: it
    # feeds the tokenizer nonsense and the contamination spreads to agents that
    # were fine.
    agent['position'][~agent['valid_mask']] = 0.0

    for key in ('position', 'heading', 'velocity', 'shape'):
        agent[key] = agent[key].float()

    data = {'scenario_id': scenario.scenario_name, 'city': scenario.map_api.map_name,
            'origin': origin, 'agent': agent}
    data.update(map_data)
    return data


def _convert_agents(scenario, num_steps, stride):
    """Dense [agent, time] tensors with the ego last, as WOMD lays them out."""
    available = scenario.get_number_of_iterations()
    iterations = [t * stride for t in range(num_steps)]
    if iterations[-1] >= available:
        raise ValueError(f'scenario has {available} iterations, need '
                         f'{iterations[-1] + 1} for {num_steps} steps at stride {stride}')

    frames = []
    order = {}
    for t, iteration in enumerate(iterations):
        objects = {}
        for tracked in scenario.get_tracked_objects_at_iteration(iteration).tracked_objects:
            # Cones, barriers and roadside debris are not agents. SMART has a
            # token embedding for vehicles, pedestrians and cyclists and none
            # for anything else, and tokenize_agent masks on exactly those
            # three types -- so a fourth type is not modelled badly, it is
            # skipped in silence and decodes to a degenerate trajectory pinned
            # to the nearest lane. In Las Vegas these outnumber the real agents,
            # which is enough to dominate any collision rate measured over them.
            # They still occlude; the occlusion layer reads nuPlan objects
            # directly and is unaffected by this.
            if tracked.tracked_object_type not in AGENT_TYPE:
                continue
            token = tracked.track_token or tracked.token
            objects[token] = tracked
            order.setdefault(token, len(order))
        frames.append((scenario.get_ego_state_at_iteration(iteration), objects))

    av_index = len(order)
    num_nodes = av_index + 1

    position = torch.zeros(num_nodes, num_steps, 3, dtype=torch.float64)
    heading = torch.zeros(num_nodes, num_steps, dtype=torch.float64)
    velocity = torch.zeros(num_nodes, num_steps, 3, dtype=torch.float64)
    shape = torch.zeros(num_nodes, num_steps, 3, dtype=torch.float64)
    valid = torch.zeros(num_nodes, num_steps, dtype=torch.bool)
    types = torch.full((num_nodes,), AGENT_VEHICLE, dtype=torch.uint8)
    category = torch.ones(num_nodes, dtype=torch.uint8)

    types[av_index] = AGENT_VEHICLE
    category[av_index] = 3

    for t, (ego, objects) in enumerate(frames):
        box = ego.car_footprint.oriented_box
        position[av_index, t] = torch.tensor([ego.center.x, ego.center.y, 0.0])
        heading[av_index, t] = ego.center.heading
        velocity[av_index, t] = torch.tensor([
            ego.dynamic_car_state.center_velocity_2d.x,
            ego.dynamic_car_state.center_velocity_2d.y, 0.0])
        shape[av_index, t] = torch.tensor([box.length, box.width, box.height])
        valid[av_index, t] = True

        for token, tracked in objects.items():
            i = order[token]
            types[i] = AGENT_TYPE[tracked.tracked_object_type]
            centre = tracked.box.center
            position[i, t] = torch.tensor([centre.x, centre.y, 0.0])
            heading[i, t] = centre.heading
            shape[i, t] = torch.tensor([tracked.box.length, tracked.box.width,
                                        tracked.box.height])
            speed = getattr(tracked, 'velocity', None)
            if speed is not None:
                velocity[i, t] = torch.tensor([speed.x, speed.y, 0.0])
            valid[i, t] = True

    return {'num_nodes': num_nodes, 'av_index': av_index,
            'valid_mask': valid, 'predict_mask': valid.clone(),
            'id': list(order) + ['ego'], 'type': types, 'category': category,
            'position': position, 'heading': heading,
            'velocity': velocity, 'shape': shape}


MIN_SEGMENT = 1e-3


def _polyline(states):
    """Positions, headings, segment lengths and rises along a discrete path.

    WOMD drops the last point of every polyline, because each point carries the
    vector leaving it. Following that keeps the point and edge counts consistent
    with what the model saw in training.

    Consecutive duplicate points are dropped first. nuPlan's boundaries contain
    them -- a couple of hundred per scenario -- and WOMD's do not, so the map
    tokenizer never had to cope: it interpolates by arc length and divides by
    the segment length, turning a zero-length segment into NaN. Those NaNs
    reach map_save.traj_pos without any error being raised, which is the worst
    way for this to fail.
    """
    if len(states) < 2:
        return None
    xy = [[states[0].x, states[0].y, 0.0]]
    for state in states[1:]:
        if math.hypot(state.x - xy[-1][0], state.y - xy[-1][1]) >= MIN_SEGMENT:
            xy.append([state.x, state.y, 0.0])
    if len(xy) < 2:
        return None
    xy = torch.tensor(xy, dtype=torch.float64)
    vectors = xy[1:] - xy[:-1]
    orientation = torch.atan2(vectors[:, 1], vectors[:, 0])
    magnitude = vectors[:, :2].norm(dim=-1)
    return xy[:-1], orientation, magnitude, vectors[:, 2]


def _traffic_lights(scenario):
    """Light state per lane connector at the current step, if the log has any."""
    states = {}
    try:
        iteration = WOMD_CURRENT_STEP * NUPLAN_STRIDE
        for light in scenario.get_traffic_light_status_at_iteration(iteration):
            states[str(light.lane_connector_id)] = LIGHT_STATE.get(
                light.status, LIGHT_UNKNOWN)
    except (AttributeError, NotImplementedError):
        pass
    return states


def _convert_map(scenario, centre, radius, include_boundaries, origin):
    """Lanes, their boundaries and crosswalks as typed polygons of points."""
    map_api = scenario.map_api
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR,
              SemanticMapLayer.CROSSWALK]
    nearby = map_api.get_proximal_map_objects(centre, radius, layers)
    lights = _traffic_lights(scenario)

    polygon_type, polygon_light = [], []
    positions, orientations, magnitudes, heights, point_type = [], [], [], [], []
    lane_index = {}

    def add(points, pt_type, pl_type, light=LIGHT_UNKNOWN):
        line = _polyline(points)
        if line is None:
            return None
        xy, orientation, magnitude, rise = line
        index = len(polygon_type)
        polygon_type.append(pl_type)
        polygon_light.append(light)
        positions.append(xy)
        orientations.append(orientation)
        magnitudes.append(magnitude)
        heights.append(rise)
        point_type.append(torch.full((len(xy),), pt_type, dtype=torch.uint8))
        return index

    lanes = (nearby.get(SemanticMapLayer.LANE, [])
             + nearby.get(SemanticMapLayer.LANE_CONNECTOR, []))
    for lane in lanes:
        light = lights.get(lane.id, LIGHT_UNKNOWN)
        index = add(lane.baseline_path.discrete_path, PT_CENTERLINE,
                    POLYGON_VEHICLE, light)
        if index is None:
            continue
        lane_index[lane.id] = index
        if include_boundaries:
            for boundary in (lane.left_boundary, lane.right_boundary):
                add(boundary.discrete_path, PT_EDGE, POLYGON_VEHICLE)

    for crosswalk in nearby.get(SemanticMapLayer.CROSSWALK, []):
        add(_polygon_outline(crosswalk.polygon), PT_CROSSWALK, POLYGON_PEDESTRIAN)

    if not polygon_type:
        raise ValueError('no map objects within the radius of the ego')

    # Recentre in float64, then cast. See convert_scenario for why.
    position = torch.cat(positions)
    position[:, 0] -= origin[0]
    position[:, 1] -= origin[1]

    edges, edge_types = _lane_graph(lanes, lane_index)
    num_points = torch.tensor([len(p) for p in positions])

    return {
        'map_polygon': {
            'num_nodes': len(polygon_type),
            'type': torch.tensor(polygon_type, dtype=torch.uint8),
            'light_type': torch.tensor(polygon_light, dtype=torch.uint8),
        },
        'map_point': {
            'num_nodes': int(num_points.sum()),
            'position': position.float(),
            'orientation': torch.cat(orientations).float(),
            'magnitude': torch.cat(magnitudes).float(),
            'height': torch.cat(heights).float(),
            'type': torch.cat(point_type),
        },
        ('map_point', 'to', 'map_polygon'): {
            'edge_index': torch.stack([
                torch.arange(int(num_points.sum())),
                torch.arange(len(polygon_type)).repeat_interleave(num_points)]),
        },
        ('map_polygon', 'to', 'map_polygon'): {
            'edge_index': edges,
            'type': edge_types,
        },
    }


def _polygon_outline(polygon):
    """A shapely polygon's exterior as pose-like points, headings included."""
    coords = list(polygon.exterior.coords)

    class _Point:
        __slots__ = ('x', 'y')

        def __init__(self, x, y):
            self.x, self.y = x, y

    return [_Point(x, y) for x, y in coords]


def _lane_graph(lanes, lane_index):
    """Predecessor, successor and adjacency edges between lane polygons.

    Only lanes that made it into the output are wired up; a neighbour outside
    the map radius has no polygon to point at.
    """
    sources, targets, kinds = [], [], []

    def connect(other_id, index, kind):
        other = lane_index.get(other_id)
        if other is not None:
            sources.append(other)
            targets.append(index)
            kinds.append(kind)

    for lane in lanes:
        index = lane_index.get(lane.id)
        if index is None:
            continue
        for edge in lane.incoming_edges:
            connect(edge.id, index, PL2PL_PRED)
        for edge in lane.outgoing_edges:
            connect(edge.id, index, PL2PL_SUCC)
        left, right = lane.adjacent_edges
        if left is not None:
            connect(left.id, index, PL2PL_LEFT)
        if right is not None:
            connect(right.id, index, PL2PL_RIGHT)

    if not sources:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, dtype=torch.uint8)
    return (torch.tensor([sources, targets], dtype=torch.long),
            torch.tensor(kinds, dtype=torch.uint8))
