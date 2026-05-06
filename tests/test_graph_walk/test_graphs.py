"""Tests for graph construction and random walks.

These tests are designed to catch subtle bugs in adjacency logic,
especially the tricky hex graph even/odd row handling.
"""

import math
import random

import pytest

from causalab.tasks.graph_walk.graphs import (
    Graph,
    build_graph,
    make_cylinder_graph,
    make_grid_graph,
    make_hex_graph,
    make_ring_graph,
    random_neighbor_pairs,
    random_walk,
)


# ---------------------------------------------------------------------------
# Grid graph
# ---------------------------------------------------------------------------


class TestGridGraph:
    def test_node_count(self):
        for m in [1, 2, 3, 5, 10]:
            g = make_grid_graph(m)
            assert g.n_nodes == m * m

    def test_corner_nodes_have_two_neighbors(self):
        g = make_grid_graph(5)
        corners = [0, 4, 20, 24]  # (0,0), (0,4), (4,0), (4,4)
        for c in corners:
            assert len(g.adjacency[c]) == 2, (
                f"Corner node {c} has {len(g.adjacency[c])} neighbors"
            )

    def test_edge_nodes_have_three_neighbors(self):
        g = make_grid_graph(5)
        # Node 1 = (0,1) — top edge, not corner
        assert len(g.adjacency[1]) == 3
        # Node 5 = (1,0) — left edge, not corner
        assert len(g.adjacency[5]) == 3

    def test_interior_nodes_have_four_neighbors(self):
        g = make_grid_graph(5)
        # Node 6 = (1,1) — interior
        assert len(g.adjacency[6]) == 4
        # Node 12 = (2,2) — center
        assert len(g.adjacency[12]) == 4

    def test_adjacency_is_symmetric(self):
        """If A is neighbor of B, then B must be neighbor of A."""
        g = make_grid_graph(5)
        for node, neighbors in g.adjacency.items():
            for nb in neighbors:
                assert node in g.adjacency[nb], (
                    f"Asymmetric adjacency: {node} -> {nb} but {nb} -/-> {node}"
                )

    def test_no_self_loops(self):
        g = make_grid_graph(5)
        for node, neighbors in g.adjacency.items():
            assert node not in neighbors, f"Self-loop at node {node}"

    def test_coordinates_match_grid_position(self):
        g = make_grid_graph(4)
        # Node 0 = (0,0), Node 5 = (1,1), Node 15 = (3,3)
        assert g.coordinates[0] == (0.0, 0.0)
        assert g.coordinates[5] == (1.0, 1.0)
        assert g.coordinates[15] == (3.0, 3.0)

    def test_specific_neighbors(self):
        """Check exact neighbors for a known node."""
        g = make_grid_graph(3)
        # Node 4 = center (1,1), neighbors: (0,1)=1, (2,1)=7, (1,0)=3, (1,2)=5
        assert set(g.adjacency[4]) == {1, 3, 5, 7}

    def test_size_one(self):
        """1x1 grid: single node, no neighbors."""
        g = make_grid_graph(1)
        assert g.n_nodes == 1
        assert g.adjacency[0] == []

    def test_coordinate_names(self):
        g = make_grid_graph(3)
        assert g.coordinate_names == ["row", "col"]


# ---------------------------------------------------------------------------
# Ring graph
# ---------------------------------------------------------------------------


class TestRingGraph:
    def test_node_count(self):
        for n in [3, 5, 10, 50]:
            g = make_ring_graph(n)
            assert g.n_nodes == n

    def test_all_nodes_have_exactly_two_neighbors(self):
        g = make_ring_graph(10)
        for node in range(10):
            assert len(g.adjacency[node]) == 2

    def test_wraparound(self):
        """Node 0 and node n-1 must be neighbors."""
        g = make_ring_graph(10)
        assert 9 in g.adjacency[0]
        assert 0 in g.adjacency[9]

    def test_adjacency_is_symmetric(self):
        g = make_ring_graph(10)
        for node, neighbors in g.adjacency.items():
            for nb in neighbors:
                assert node in g.adjacency[nb]

    def test_no_self_loops(self):
        g = make_ring_graph(10)
        for node, neighbors in g.adjacency.items():
            assert node not in neighbors

    def test_specific_neighbors(self):
        g = make_ring_graph(5)
        assert set(g.adjacency[0]) == {4, 1}
        assert set(g.adjacency[2]) == {1, 3}
        assert set(g.adjacency[4]) == {3, 0}

    def test_angles_evenly_spaced(self):
        g = make_ring_graph(4)
        angles = [g.coordinates[i][0] for i in range(4)]
        expected = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
        for a, e in zip(angles, expected):
            assert abs(a - e) < 1e-10

    def test_coordinate_names(self):
        g = make_ring_graph(5)
        assert g.coordinate_names == ["angle"]

    def test_small_ring(self):
        """Ring of 3: each node is neighbor of both others."""
        g = make_ring_graph(3)
        assert set(g.adjacency[0]) == {2, 1}
        assert set(g.adjacency[1]) == {0, 2}
        assert set(g.adjacency[2]) == {1, 0}


# ---------------------------------------------------------------------------
# Hex graph
# ---------------------------------------------------------------------------


class TestHexGraph:
    def test_node_count(self):
        for m in [2, 3, 5]:
            g = make_hex_graph(m)
            assert g.n_nodes == m * m

    def test_adjacency_is_symmetric(self):
        """Critical: the even/odd row logic must produce symmetric adjacency."""
        for m in [2, 3, 4, 5]:
            g = make_hex_graph(m)
            for node, neighbors in g.adjacency.items():
                for nb in neighbors:
                    assert node in g.adjacency[nb], (
                        f"Hex m={m}: asymmetric adjacency {node} -> {nb} "
                        f"but {nb}'s neighbors are {g.adjacency[nb]}"
                    )

    def test_no_self_loops(self):
        g = make_hex_graph(5)
        for node, neighbors in g.adjacency.items():
            assert node not in neighbors

    def test_no_out_of_range_neighbors(self):
        for m in [2, 3, 4, 5]:
            g = make_hex_graph(m)
            for node, neighbors in g.adjacency.items():
                for nb in neighbors:
                    assert 0 <= nb < g.n_nodes, (
                        f"Hex m={m}: node {node} has out-of-range neighbor {nb}"
                    )

    def test_interior_nodes_have_six_neighbors(self):
        """Interior hex nodes should have exactly 6 neighbors."""
        g = make_hex_graph(5)
        # Node at (2,2) = 12 is interior (not on any edge)
        # For a hex grid, interior nodes should have 6 neighbors
        assert len(g.adjacency[12]) == 6, (
            f"Interior node 12 has {len(g.adjacency[12])} neighbors, expected 6"
        )

    def test_corner_nodes_have_fewer_neighbors(self):
        """Corner nodes should have 2-3 neighbors."""
        g = make_hex_graph(5)
        # Node 0 = (0,0) is a corner
        assert len(g.adjacency[0]) <= 3

    def test_no_duplicate_neighbors(self):
        for m in [2, 3, 4, 5]:
            g = make_hex_graph(m)
            for node, neighbors in g.adjacency.items():
                assert len(neighbors) == len(set(neighbors)), (
                    f"Hex m={m}: node {node} has duplicate neighbors: {neighbors}"
                )

    def test_coordinate_names(self):
        g = make_hex_graph(3)
        assert g.coordinate_names == ["x", "y"]

    def test_odd_row_offset(self):
        """Odd rows should have x-coordinate shifted by 0.5."""
        g = make_hex_graph(3)
        # Node 3 = (1,0) — odd row, col 0
        assert g.coordinates[3][0] == 0.5
        # Node 0 = (0,0) — even row, col 0
        assert g.coordinates[0][0] == 0.0


# ---------------------------------------------------------------------------
# Cylinder graph
# ---------------------------------------------------------------------------


class TestCylinderGraph:
    def test_node_count(self):
        for n_ring, n_height in [(4, 3), (6, 5), (8, 2)]:
            g = make_cylinder_graph(n_ring, n_height)
            assert g.n_nodes == n_ring * n_height

    def test_ring_neighbors_are_periodic(self):
        g = make_cylinder_graph(6, 3)
        # Node 0 at (h=0, r=0) should connect to r=5 and r=1 on same ring
        assert 5 in g.adjacency[0]  # (h=0, r=5)
        assert 1 in g.adjacency[0]  # (h=0, r=1)

    def test_vertical_neighbors_not_periodic(self):
        g = make_cylinder_graph(6, 3)
        # Bottom ring (h=0) nodes should NOT connect to top ring (h=2)
        for r in range(6):
            bottom_node = r  # h=0
            top_node = 2 * 6 + r  # h=2
            assert top_node not in g.adjacency[bottom_node]
            assert bottom_node not in g.adjacency[top_node]

    def test_interior_nodes_have_four_neighbors(self):
        """Interior nodes (not top/bottom ring) have 2 ring + 2 vertical = 4."""
        g = make_cylinder_graph(6, 3)
        for r in range(6):
            node = 1 * 6 + r  # h=1 (middle ring)
            assert len(g.adjacency[node]) == 4

    def test_boundary_nodes_have_three_neighbors(self):
        """Top/bottom ring nodes have 2 ring + 1 vertical = 3."""
        g = make_cylinder_graph(6, 3)
        for r in range(6):
            bottom = r  # h=0
            top = 2 * 6 + r  # h=2
            assert len(g.adjacency[bottom]) == 3
            assert len(g.adjacency[top]) == 3

    def test_adjacency_is_symmetric(self):
        g = make_cylinder_graph(6, 4)
        for node, neighbors in g.adjacency.items():
            for nb in neighbors:
                assert node in g.adjacency[nb], (
                    f"Asymmetric: {node} -> {nb} but {nb} -/-> {node}"
                )

    def test_no_self_loops(self):
        g = make_cylinder_graph(6, 4)
        for node, neighbors in g.adjacency.items():
            assert node not in neighbors

    def test_coordinate_names(self):
        g = make_cylinder_graph(6, 3)
        assert g.coordinate_names == ["height", "angle"]

    def test_angles_evenly_spaced(self):
        g = make_cylinder_graph(4, 2)
        angles = [g.coordinates[r][1] for r in range(4)]  # h=0 ring
        expected = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
        for a, e in zip(angles, expected):
            assert abs(a - e) < 1e-10

    def test_height_coordinates(self):
        g = make_cylinder_graph(4, 3)
        for h in range(3):
            node = h * 4  # r=0 at each height
            assert g.coordinates[node][0] == float(h)


# ---------------------------------------------------------------------------
# build_graph dispatcher
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def test_valid_types(self):
        assert build_graph("grid", 3).n_nodes == 9
        assert build_graph("ring", 5).n_nodes == 5
        assert build_graph("hex", 3).n_nodes == 9
        assert build_graph("cylinder", 6, 4).n_nodes == 24

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown graph type"):
            build_graph("triangle", 3)


# ---------------------------------------------------------------------------
# Random walks
# ---------------------------------------------------------------------------


class TestRandomWalk:
    def test_walk_length(self):
        g = make_ring_graph(5)
        path = random_walk(g, 10, start=0)
        assert len(path) == 11  # length + 1

    def test_walk_starts_at_given_node(self):
        g = make_ring_graph(10)
        for start in [0, 3, 7]:
            path = random_walk(g, 5, start=start)
            assert path[0] == start

    def test_every_step_is_valid(self):
        """Each consecutive pair must be neighbors."""
        g = make_grid_graph(5)
        rng = random.Random(42)
        for _ in range(20):
            path = random_walk(g, 50, rng=rng)
            for i in range(len(path) - 1):
                assert path[i + 1] in g.adjacency[path[i]], (
                    f"Invalid step: {path[i]} -> {path[i + 1]}, "
                    f"valid neighbors: {g.adjacency[path[i]]}"
                )

    def test_walks_on_hex_are_valid(self):
        """Hex graph walks should also produce valid steps."""
        g = make_hex_graph(4)
        rng = random.Random(123)
        for _ in range(50):
            path = random_walk(g, 30, rng=rng)
            for i in range(len(path) - 1):
                assert path[i + 1] in g.adjacency[path[i]]

    def test_determinism_with_seed(self):
        g = make_ring_graph(10)
        path1 = random_walk(g, 20, start=0, rng=random.Random(42))
        path2 = random_walk(g, 20, start=0, rng=random.Random(42))
        assert path1 == path2

    def test_random_start_is_in_range(self):
        g = make_grid_graph(5)
        rng = random.Random(99)
        for _ in range(100):
            path = random_walk(g, 5, rng=rng)
            assert 0 <= path[0] < g.n_nodes

    def test_zero_length_walk(self):
        g = make_ring_graph(5)
        path = random_walk(g, 0, start=3)
        assert path == [3]

    def test_no_backtrack_prevents_immediate_reversal(self):
        """With no_backtrack, the walk should never go A -> B -> A."""
        g = make_grid_graph(5)
        rng = random.Random(42)
        for _ in range(50):
            path = random_walk(g, 100, rng=rng, no_backtrack=True)
            for i in range(len(path) - 2):
                assert path[i + 2] != path[i], (
                    f"Backtrack at step {i}: {path[i]} -> {path[i + 1]} -> {path[i + 2]}"
                )

    def test_no_backtrack_all_steps_valid(self):
        """no_backtrack walks must still only visit valid neighbors."""
        g = make_hex_graph(4)
        rng = random.Random(99)
        for _ in range(50):
            path = random_walk(g, 50, rng=rng, no_backtrack=True)
            for i in range(len(path) - 1):
                assert path[i + 1] in g.adjacency[path[i]]

    def test_no_backtrack_ring_is_unidirectional(self):
        """On a ring, no-backtrack forces a single direction (always CW or CCW)."""
        g = make_ring_graph(10)
        path = random_walk(g, 20, start=0, rng=random.Random(42), no_backtrack=True)
        # After the first step picks a direction, all subsequent steps
        # must continue in that direction (since each node has only 2
        # neighbors and one is the previous node).
        direction = path[1] - path[0]  # +1 or -1 (mod n)
        for i in range(1, len(path) - 1):
            diff = (path[i + 1] - path[i]) % 10
            expected = direction % 10
            assert diff == expected, (
                f"Direction changed at step {i}: {path[i]} -> {path[i + 1]}, "
                f"expected direction {expected}"
            )

    def test_no_backtrack_fallback_when_only_one_neighbor(self):
        """On a path graph (dead-end), no_backtrack falls back to allowing backtrack."""
        # Create a simple 3-node path: 0 -- 1 -- 2
        g = Graph(
            adjacency={0: [1], 1: [0, 2], 2: [1]},
            coordinates={0: (0.0,), 1: (1.0,), 2: (2.0,)},
            coordinate_names=["x"],
            n_nodes=3,
        )
        # Start at 0, must go to 1, then can go to 2, then must backtrack to 1
        path = random_walk(g, 4, start=0, rng=random.Random(42), no_backtrack=True)
        assert path[0] == 0
        assert path[1] == 1
        # At node 2 with prev=1, only neighbor is 1 — fallback allows backtrack
        # Walk should not crash


class TestRandomNeighborPairs:
    def test_all_pairs_are_valid_edges(self):
        g = make_grid_graph(5)
        rng = random.Random(42)
        pairs = random_neighbor_pairs(g, 100, rng=rng)
        assert len(pairs) == 100
        for node, nb in pairs:
            assert nb in g.adjacency[node], (
                f"Invalid pair: ({node}, {nb}), neighbors: {g.adjacency[node]}"
            )

    def test_determinism(self):
        g = make_ring_graph(10)
        p1 = random_neighbor_pairs(g, 10, rng=random.Random(42))
        p2 = random_neighbor_pairs(g, 10, rng=random.Random(42))
        assert p1 == p2
