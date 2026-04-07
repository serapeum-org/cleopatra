"""Tests for cleopatra.mesh_glyph.MeshGlyph.

Covers mesh data plotting (tripcolor, tricontourf), wireframe rendering,
fan triangulation, face-to-triangle value mapping, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleopatra.mesh_glyph import MeshGlyph


@pytest.fixture(scope="module")
def triangle_glyph():
    """MeshGlyph with 2 triangular faces.

    Layout::

         2---3
        / \\ /
       0---1
    """
    node_x = np.array([0.0, 1.0, 0.5, 1.5])
    node_y = np.array([0.0, 0.0, 1.0, 1.0])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    return MeshGlyph(node_x, node_y, faces)


@pytest.fixture(scope="module")
def mixed_glyph():
    """MeshGlyph with 1 quad + 2 triangles (mixed mesh)."""
    node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    faces = np.array(
        [
            [0, 1, 4, 3],
            [1, 2, 5, -1],
            [1, 5, 4, -1],
        ]
    )
    return MeshGlyph(node_x, node_y, faces, fill_value=-1)


@pytest.fixture(scope="module")
def quad_with_edges():
    """MeshGlyph with 1 quad and explicit edge connectivity."""
    node_x = np.array([0.0, 1.0, 1.0, 0.0])
    node_y = np.array([0.0, 0.0, 1.0, 1.0])
    faces = np.array([[0, 1, 2, 3]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    return MeshGlyph(node_x, node_y, faces, edge_node_connectivity=edges)


class TestConstructorValidation:
    """Tests for MeshGlyph constructor input validation."""

    def test_mismatched_node_lengths_raises(self):
        """Test that mismatched node_x and node_y raises ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            MeshGlyph(
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0, 2.0]),
                np.array([[0, 1, 2]]),
            )

    def test_non_1d_node_x_raises(self):
        """Test that 2D node_x raises ValueError."""
        with pytest.raises(ValueError, match="1D"):
            MeshGlyph(
                np.array([[0.0, 1.0]]),
                np.array([[0.0, 1.0]]),
                np.array([[0, 1, 2]]),
            )

    def test_1d_face_connectivity_raises(self):
        """Test that 1D face_node_connectivity raises ValueError."""
        with pytest.raises(ValueError, match="2D"):
            MeshGlyph(
                np.array([0.0, 1.0, 2.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([0, 1, 2]),
            )

    def test_out_of_range_face_indices_raises(self):
        """Test that out-of-range node indices raise ValueError."""
        with pytest.raises(ValueError, match="indices must be in"):
            MeshGlyph(
                np.array([0.0, 1.0]),
                np.array([0.0, 0.0]),
                np.array([[0, 1, 99]]),
            )

    def test_wrong_edge_shape_raises(self):
        """Test that edge_node_connectivity with wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="shape.*n_edges, 2"):
            MeshGlyph(
                np.array([0.0, 1.0, 2.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([[0, 1, 2]]),
                edge_node_connectivity=np.array([0, 1, 2]),
            )


class TestMeshGlyphProperties:
    """Tests for MeshGlyph basic properties."""

    def test_n_nodes(self, triangle_glyph):
        """Test node count property."""
        assert triangle_glyph.n_nodes == 4, f"Expected 4, got {triangle_glyph.n_nodes}"

    def test_n_faces(self, triangle_glyph):
        """Test face count property."""
        assert triangle_glyph.n_faces == 2, f"Expected 2, got {triangle_glyph.n_faces}"

    def test_n_edges_without_edges(self, triangle_glyph):
        """Test edge count returns 0 when no edge connectivity."""
        assert triangle_glyph.n_edges == 0, f"Expected 0, got {triangle_glyph.n_edges}"

    def test_n_edges_with_edges(self, quad_with_edges):
        """Test edge count when edge connectivity provided."""
        assert (
            quad_with_edges.n_edges == 4
        ), f"Expected 4, got {quad_with_edges.n_edges}"

    def test_nodes_per_face_triangular(self, triangle_glyph):
        """Test nodes per face for pure triangular mesh."""
        counts = triangle_glyph.nodes_per_face
        np.testing.assert_array_equal(counts, [3, 3])

    def test_nodes_per_face_mixed(self, mixed_glyph):
        """Test nodes per face for mixed mesh."""
        counts = mixed_glyph.nodes_per_face
        np.testing.assert_array_equal(counts, [4, 3, 3])


class TestTriangulation:
    """Tests for fan triangulation."""

    def test_triangular_mesh(self, triangle_glyph):
        """Test triangulation of pure triangular mesh produces 2 triangles."""
        tri = triangle_glyph.triangulation
        assert tri.triangles.shape == (
            2,
            3,
        ), f"Expected (2, 3), got {tri.triangles.shape}"

    def test_mixed_mesh(self, mixed_glyph):
        """Test triangulation of mixed mesh: 1 quad (2 tri) + 2 tri = 4."""
        tri = mixed_glyph.triangulation
        assert (
            tri.triangles.shape[0] == 4
        ), f"Expected 4 triangles, got {tri.triangles.shape[0]}"

    def test_cached(self, triangle_glyph):
        """Test triangulation is cached."""
        t1 = triangle_glyph.triangulation
        t2 = triangle_glyph.triangulation
        assert t1 is t2, "Triangulation should be cached"

    def test_empty_mesh_raises(self):
        """Test that mesh with no valid faces raises ValueError."""
        mg = MeshGlyph(
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([[0, 1, -1]], dtype=np.intp),
            fill_value=-1,
        )
        with pytest.raises(ValueError, match="no faces with 3"):
            _ = mg.triangulation


class TestMapFaceToTriangleValues:
    """Tests for face-to-triangle value mapping."""

    def test_pure_triangles(self, triangle_glyph):
        """Test 1:1 mapping for triangular mesh."""
        values = np.array([10.0, 20.0])
        result = triangle_glyph._map_face_to_triangle_values(values)
        assert len(result) == 2, f"Expected 2, got {len(result)}"
        assert result[0] == 10.0 and result[1] == 20.0

    def test_mixed_mesh(self, mixed_glyph):
        """Test quad value duplicated across its 2 triangles."""
        values = np.array([10.0, 20.0, 30.0])
        result = mixed_glyph._map_face_to_triangle_values(values)
        assert len(result) == 4, f"Expected 4, got {len(result)}"
        assert result[0] == 10.0 and result[1] == 10.0


class TestPlot:
    """Tests for MeshGlyph.plot()."""

    def test_face_plot(self, triangle_glyph):
        """Test face-centered data plot returns Figure and Axes."""
        data = np.array([1.0, 2.0])
        fig, ax = triangle_glyph.plot(data, location="face")
        assert fig is not None, "Should return a Figure"
        assert ax is not None, "Should return Axes"

    def test_node_plot(self, triangle_glyph):
        """Test node-centered data plot."""
        data = np.array([0.0, 1.0, 2.0, 3.0])
        fig, ax = triangle_glyph.plot(data, location="node")
        assert fig is not None, "Should return a Figure"

    def test_mixed_face_plot(self, mixed_glyph):
        """Test face plot on mixed mesh."""
        data = np.array([10.0, 20.0, 30.0])
        fig, ax = mixed_glyph.plot(data, location="face")
        assert fig is not None, "Should return a Figure"

    def test_with_title_and_options(self, triangle_glyph):
        """Test plot with title, colorbar, and custom cmap."""
        data = np.array([1.0, 2.0])
        fig, ax = triangle_glyph.plot(
            data,
            title="Test",
            colorbar=True,
            cmap="coolwarm",
            vmin=0.0,
            vmax=3.0,
        )
        assert ax.get_title() == "Test", f"Expected 'Test', got '{ax.get_title()}'"

    def test_invalid_location_raises(self, triangle_glyph):
        """Test that invalid location raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            triangle_glyph.plot(np.array([1.0]), location="edge")

    def test_face_data_wrong_length_raises(self, triangle_glyph):
        """Test that wrong data length for face location raises ValueError."""
        with pytest.raises(ValueError, match="data length"):
            triangle_glyph.plot(np.array([1.0, 2.0, 3.0]), location="face")

    def test_node_data_wrong_length_raises(self, triangle_glyph):
        """Test that wrong data length for node location raises ValueError."""
        with pytest.raises(ValueError, match="data length"):
            triangle_glyph.plot(np.array([1.0, 2.0]), location="node")

    def test_existing_axes(self, triangle_glyph):
        """Test plotting on user-provided Axes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        _, ax_out = triangle_glyph.plot(np.array([1.0, 2.0]), ax=ax)
        assert ax_out is ax, "Should plot on the provided Axes"

    def test_node_plot_custom_levels(self, triangle_glyph):
        """Test that user can override levels via kwargs without TypeError."""
        data = np.array([0.0, 1.0, 2.0, 3.0])
        fig, ax = triangle_glyph.plot(data, location="node", levels=5)
        assert fig is not None

    def test_colorbar_false(self, triangle_glyph):
        """Test plot with colorbar=False."""
        fig, ax = triangle_glyph.plot(
            np.array([1.0, 2.0]), colorbar=False
        )
        assert fig is not None


class TestPlotOutline:
    """Tests for MeshGlyph.plot_outline()."""

    def test_wireframe_no_edges(self, triangle_glyph):
        """Test wireframe without explicit edge connectivity."""
        fig, ax = triangle_glyph.plot_outline()
        assert fig is not None, "Should return a Figure"
        assert len(ax.collections) == 1, "Should have 1 LineCollection"

    def test_wireframe_with_edges(self, quad_with_edges):
        """Test wireframe with explicit edge connectivity."""
        fig, ax = quad_with_edges.plot_outline(color="red", linewidth=1.0)
        assert fig is not None, "Should return a Figure"

    def test_wireframe_mixed_mesh(self, mixed_glyph):
        """Test wireframe on mixed mesh."""
        fig, ax = mixed_glyph.plot_outline()
        assert fig is not None, "Should return a Figure"

    def test_existing_axes(self, triangle_glyph):
        """Test wireframe on user-provided Axes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        _, ax_out = triangle_glyph.plot_outline(ax=ax)
        assert ax_out is ax, "Should use the provided Axes"


class TestEdgeSegments:
    """Tests for _build_edge_segments internal method."""

    def test_segments_from_faces(self, triangle_glyph):
        """Test edge segments derived from face connectivity."""
        segs = triangle_glyph._build_edge_segments()
        assert segs.shape[0] > 0, "Should produce edge segments"
        assert segs.shape[1:] == (2, 2), "Each segment should be [[x1,y1],[x2,y2]]"

    def test_segments_from_edges(self, quad_with_edges):
        """Test edge segments from explicit edge connectivity."""
        segs = quad_with_edges._build_edge_segments()
        assert segs.shape[0] == 4, f"Expected 4 segments, got {segs.shape[0]}"

    def test_no_duplicate_edges(self, triangle_glyph):
        """Test that face-derived edges are deduplicated."""
        segs = triangle_glyph._build_edge_segments()
        edge_set = set()
        for seg in segs:
            key = (tuple(seg[0]), tuple(seg[1]))
            assert key not in edge_set, f"Duplicate edge: {key}"
            edge_set.add(key)
