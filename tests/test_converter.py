"""Tests for the nuPlan to SMART schema conversion.

The parts worth testing here are the ones a real scenario would not exercise
loudly: polyline geometry, the lane graph when neighbours fall outside the map
radius, and the type mappings that have to land inside the checkpoint's
embedding ranges. Whether the whole thing runs on real data is answered by
scripts/nuplan_occlusion_check.py and by tokenizing a converted scenario, not
by mocking a NuPlanScenario in full.
"""
import math
import os

import pytest
import torch

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from smart.nuplan.converter import (
    AGENT_BACKGROUND,
    AGENT_TYPE,
    LIGHT_STATE,
    LIGHT_UNKNOWN,
    PL2PL_LEFT,
    PL2PL_PRED,
    PL2PL_RIGHT,
    PL2PL_SUCC,
    PT_CENTERLINE,
    PT_CROSSWALK,
    PT_EDGE,
    WOMD_CURRENT_STEP,
    _lane_graph,
    _polyline,
)


class Pose:
    def __init__(self, x, y):
        self.x, self.y = x, y


class Lane:
    """Enough of a nuPlan lane for the graph builder."""

    def __init__(self, id, incoming=(), outgoing=(), left=None, right=None):
        self.id = id
        self.incoming_edges = [Lane(i) for i in incoming]
        self.outgoing_edges = [Lane(o) for o in outgoing]
        self.adjacent_edges = (Lane(left) if left else None,
                               Lane(right) if right else None)


def test_polyline_drops_the_last_point():
    """WOMD stores the vector leaving each point, so the final point has none."""
    xy, orientation, magnitude, rise = _polyline([Pose(0, 0), Pose(1, 0), Pose(2, 0)])
    assert len(xy) == 2
    # float64 until convert_scenario recentres and casts; see the UTM test.
    assert xy.dtype == torch.float64
    assert torch.allclose(xy[:, 0], torch.tensor([0.0, 1.0], dtype=torch.float64))


def test_polyline_orientation_and_magnitude():
    xy, orientation, magnitude, rise = _polyline([Pose(0, 0), Pose(0, 3)])
    assert orientation[0] == pytest.approx(math.pi / 2)
    assert magnitude[0] == pytest.approx(3.0)


def test_polyline_needs_two_points():
    assert _polyline([Pose(0, 0)]) is None
    assert _polyline([]) is None


def test_lane_graph_wires_all_four_relations():
    lanes = [Lane('b', incoming=['a'], outgoing=['c'], left='d', right='e')]
    index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    edges, kinds = _lane_graph(lanes, index)

    assert set(kinds.tolist()) == {PL2PL_PRED, PL2PL_SUCC, PL2PL_LEFT, PL2PL_RIGHT}
    assert (edges[1] == 1).all()          # every edge points at lane b
    by_kind = dict(zip(kinds.tolist(), edges[0].tolist()))
    assert by_kind[PL2PL_PRED] == 0 and by_kind[PL2PL_SUCC] == 2


def test_lane_graph_skips_neighbours_outside_the_radius():
    """A lane beyond the map radius has no polygon for an edge to point at."""
    lanes = [Lane('b', incoming=['far'], outgoing=['c'])]
    edges, kinds = _lane_graph(lanes, {'b': 0, 'c': 1})
    assert kinds.tolist() == [PL2PL_SUCC]


def test_lane_graph_skips_lanes_that_produced_no_polygon():
    """A lane too short to make a polyline is absent from the index."""
    edges, kinds = _lane_graph([Lane('dropped', outgoing=['c'])], {'c': 0})
    assert edges.shape == (2, 0)
    assert kinds.numel() == 0


def test_agent_types_stay_inside_the_embedding():
    """type_a_emb is nn.Embedding(4), so every value must be under 4."""
    assert all(v < 4 for v in AGENT_TYPE.values())
    assert AGENT_BACKGROUND < 4
    assert AGENT_TYPE[TrackedObjectType.VEHICLE] == 0


def test_map_types_stay_inside_the_embedding():
    """type_pt_emb is nn.Embedding(17), light_pl_emb nn.Embedding(4)."""
    assert max(PT_EDGE, PT_CROSSWALK, PT_CENTERLINE) < 17
    assert max(LIGHT_STATE.values()) < 4
    assert LIGHT_UNKNOWN < 4


def test_unknown_object_types_fall_back_to_background():
    """Cones, barriers and debris have no WOMD counterpart."""
    for kind in (TrackedObjectType.TRAFFIC_CONE, TrackedObjectType.BARRIER,
                 TrackedObjectType.GENERIC_OBJECT):
        assert AGENT_TYPE.get(kind, AGENT_BACKGROUND) == AGENT_BACKGROUND


def test_utm_in_float32_loses_a_quarter_metre():
    """Why convert_scenario recentres before casting.

    nuPlan is in global UTM, where a northing sits near 4e6. float32 spacing
    there is 0.25 m, so casting straight from UTM quantises lane geometry to a
    quarter of a metre and collapses nearby polyline points onto each other --
    which then divides by zero in the map tokenizer and puts NaN into
    map_save.traj_pos with no error raised. WOMD ships local coordinates in the
    low thousands and never hit this.
    """
    northing = 3999080.2
    assert float(torch.tensor(northing).float()) != northing
    lost = abs(float(torch.tensor(northing).float()) - northing)
    assert lost > 0.01

    local = northing - 3999000.0
    assert float(torch.tensor(local).float()) == pytest.approx(local, abs=1e-5)


def test_polyline_drops_points_a_repeated_point_would_duplicate():
    """nuPlan boundaries repeat points; a zero-length segment is a NaN later."""
    xy, orientation, magnitude, rise = _polyline(
        [Pose(0, 0), Pose(0, 0), Pose(1, 0), Pose(1, 0), Pose(2, 0)])
    assert len(xy) == 2
    assert (magnitude > 0).all()


def test_polyline_rejects_a_path_that_is_all_one_point():
    assert _polyline([Pose(5, 5), Pose(5, 5), Pose(5, 5)]) is None


def test_nuplan_checkpoint_semantics_uses_the_rows_that_were_trained():
    """The released nuPlan checkpoint numbers map types differently from WOMD.

    Its type embedding gives this away: only rows 0, 8 and 11 carry trained
    weights, everything else sits at its initialisation. Our converter follows
    the WOMD numbering, so without this relabelling every point type we emit --
    centreline included -- lands on a row that checkpoint never saw.
    """
    from smart.nuplan.converter import (PT_CENTERLINE, PT_CROSSWALK, PT_EDGE,
                                        POLYGON_PEDESTRIAN, PL2PL_PRED,
                                        to_nuplan_checkpoint_semantics)

    data = {
        'map_point': {'type': torch.tensor([PT_CENTERLINE, PT_EDGE, PT_CROSSWALK])},
        'map_polygon': {'type': torch.tensor([POLYGON_PEDESTRIAN])},
        ('map_polygon', 'to', 'map_polygon'): {'type': torch.tensor([PL2PL_PRED])},
    }
    out = to_nuplan_checkpoint_semantics(data)
    assert out['map_point']['type'].tolist() == [11, 0, 10]
    assert out['map_polygon']['type'].tolist() == [2]
    assert out[('map_polygon', 'to', 'map_polygon')]['type'].tolist() == [0]
    # The original must survive: the WOMD-trained checkpoint still needs it.
    assert data['map_point']['type'].tolist() == [PT_CENTERLINE, PT_EDGE, PT_CROSSWALK]


MINI_DATA = '/mnt/e/nuplan-mini/nuplan-v1.1_mini/data/cache/mini'
MINI_MAPS = '/mnt/e/nuplan-mini/nuplan-maps-v1.0/maps'


@pytest.mark.skipif(not os.path.isdir(MINI_DATA), reason='needs nuPlan mini')
def test_recentre_false_leaves_the_scene_in_global_coordinates():
    """The coordinate frame has to match the checkpoint, not the numerics.

    Recentring is better conditioned -- UTM northings quantise to about half a
    metre once they are cast to float32, which is what
    test_utm_in_float32_loses_a_quarter_metre is about. It is still wrong for
    the nuPlan-trained checkpoint, which was trained on data that was never
    recentred: clean local coordinates are out of distribution for it and cost
    7.49% against 23.62% next-token top-1 over 28 scenarios. Whichever frame is
    chosen, the agents and the map have to be in the same one, which is the
    part that would break silently.
    """
    from smart.nuplan.converter import convert_scenario
    from smart.nuplan.scenarios import build_scenario, find_scenarios

    entries = find_scenarios(MINI_DATA, 'traversing_intersection', 1, duration=12.0)
    scenario = build_scenario(entries[0], MINI_DATA, MINI_MAPS, duration=12.0,
                              scenario_type='traversing_intersection')
    local = convert_scenario(scenario, recentre=True)
    world = convert_scenario(scenario, recentre=False)

    av = local['agent']['av_index']
    assert abs(float(local['agent']['position'][av, WOMD_CURRENT_STEP, 0])) < 1.0
    assert abs(float(world['agent']['position'][av, WOMD_CURRENT_STEP, 0])) > 1000.0
    # The map must travel with the agents; a mismatch here puts every agent
    # kilometres from its own lane and nothing raises.
    assert float(world['map_point']['position'][:, 0].abs().min()) > 1000.0
    assert float(local['map_point']['position'][:, 0].abs().min()) < 1000.0
    # The reported origin stays the ego either way -- it is where the map query
    # was centred, which is a separate question from the frame.
    assert local['origin'] == world['origin']
