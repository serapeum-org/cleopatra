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

    def test_all_fill_values_passes_validation(self):
        """Test that face_nodes with all fill_values passes index validation.

        Test scenario:
            When all entries are fill_value, valid_indices is empty so
            index range check is skipped.
        """
        mg = MeshGlyph(
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([[-1, -1, -1]], dtype=np.intp),
            fill_value=-1,
        )
        assert mg.n_faces == 1, f"Expected 1 face, got {mg.n_faces}"

    def test_negative_index_without_fill_raises(self):
        """Test that negative indices (not fill_value) raise ValueError.

        Test scenario:
            fill_value=0, so -1 is treated as a real index, which is
            out of range.
        """
        with pytest.raises(ValueError, match="indices must be in"):
            MeshGlyph(
                np.array([0.0, 1.0, 2.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([[0, 1, -1]]),
                fill_value=0,
            )


class TestMeshGlyphProperties:
    """Tests for MeshGlyph basic properties."""

    def test_node_x_property(self, triangle_glyph):
        """Test node_x property returns the x-coordinates array."""
        np.testing.assert_array_equal(triangle_glyph.node_x, [0.0, 1.0, 0.5, 1.5])

    def test_node_y_property(self, triangle_glyph):
        """Test node_y property returns the y-coordinates array."""
        np.testing.assert_array_equal(triangle_glyph.node_y, [0.0, 0.0, 1.0, 1.0])

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

    def test_fan_triangles_cached(self):
        """Test that _fan_triangles returns cached array on second call."""
        mg = MeshGlyph(
            np.array([0.0, 1.0, 0.5, 1.5, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0]),
            np.array([[0, 1, 3, 2], [2, 3, 5, 4]]),
        )
        first = mg._fan_triangles()
        second = mg._fan_triangles()
        assert first is second, "Should return cached array"

    def test_mixed_mesh_with_degenerate_faces(self):
        """Test that faces with < 3 valid nodes are skipped in triangulation."""
        mg = MeshGlyph(
            np.array([0.0, 1.0, 0.5, 2.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([[0, 1, 2, -1], [0, 3, -1, -1]], dtype=np.intp),
            fill_value=-1,
        )
        tri = mg.triangulation
        assert (
            tri.triangles.shape[0] == 1
        ), f"Expected 1 triangle (degenerate skipped), got {tri.triangles.shape[0]}"

    def test_triangle_node_indices_correct(self, triangle_glyph):
        """Test that triangle node indices match input face connectivity."""
        tri = triangle_glyph.triangulation
        expected = np.array([[0, 1, 2], [1, 3, 2]])
        np.testing.assert_array_equal(tri.triangles, expected)

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

    def test_node_plot_with_vmin_vmax(self, triangle_glyph):
        """Test node plot with explicit vmin and vmax parameters."""
        data = np.array([0.0, 1.0, 2.0, 3.0])
        fig, ax = triangle_glyph.plot(data, location="node", vmin=0.0, vmax=3.0)
        assert fig is not None, "Should return a Figure"

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
        fig, ax = triangle_glyph.plot(np.array([1.0, 2.0]), colorbar=False)
        assert fig is not None


class TestPlotOutline:
    """Tests for MeshGlyph.plot_outline()."""

    def test_wireframe_no_edges(self):
        """Test wireframe without explicit edge connectivity."""
        mg = _make_tri_mg()
        fig, ax = mg.plot_outline()
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

    def test_empty_edges_from_faces(self):
        """Test _build_edge_segments returns empty array when no valid edges.

        Test scenario:
            All face entries are fill_value so no edges can be derived.
        """
        mg = MeshGlyph(
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([[-1, -1, -1]], dtype=np.intp),
            fill_value=-1,
        )
        segs = mg._build_edge_segments()
        assert segs.shape == (
            0,
            2,
            2,
        ), f"Expected empty segments, got shape {segs.shape}"


def _make_tri_mg():
    """Create a fresh 4-node, 2-face triangular MeshGlyph."""
    return MeshGlyph(
        np.array([0.0, 1.0, 0.5, 1.5]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([[0, 1, 2], [1, 3, 2]]),
    )


class TestColorScales:
    """Tests for color scale support in MeshGlyph.plot()."""

    def test_linear_scale(self):
        """Test default linear color scale."""
        fig, ax = _make_tri_mg().plot(np.array([1.0, 2.0]), color_scale="linear")
        assert fig is not None, "Should return a Figure"

    def test_power_scale(self):
        """Test power color scale with custom gamma."""
        fig, ax = _make_tri_mg().plot(
            np.array([1.0, 2.0]),
            color_scale="power",
            gamma=0.3,
        )
        assert fig is not None, "Should return a Figure"

    def test_sym_lognorm_scale(self):
        """Test symmetrical log-norm color scale."""
        fig, ax = _make_tri_mg().plot(
            np.array([1.0, 20.0]),
            color_scale="sym-lognorm",
        )
        assert fig is not None, "Should return a Figure"

    def test_boundary_norm_scale(self):
        """Test boundary-norm color scale with custom bounds."""
        fig, ax = _make_tri_mg().plot(
            np.array([1.0, 5.0]),
            color_scale="boundary-norm",
            bounds=[0, 2, 4, 6],
        )
        assert fig is not None, "Should return a Figure"

    def test_midpoint_scale(self):
        """Test midpoint color scale."""
        fig, ax = _make_tri_mg().plot(
            np.array([1.0, 5.0]),
            color_scale="midpoint",
            midpoint=3.0,
            cmap="coolwarm",
        )
        assert fig is not None, "Should return a Figure"

    def test_node_with_power_scale(self):
        """Test node-centered data with power color scale."""
        fig, ax = _make_tri_mg().plot(
            np.array([0.0, 1.0, 2.0, 3.0]),
            location="node",
            color_scale="power",
            gamma=0.5,
        )
        assert fig is not None, "Should return a Figure"

    def test_colorbar_customization(self):
        """Test colorbar label, orientation, and size."""
        fig, ax = _make_tri_mg().plot(
            np.array([1.0, 2.0]),
            cbar_label="Depth [m]",
            cbar_orientation="horizontal",
            cbar_length=0.5,
        )
        assert fig is not None, "Should return a Figure"


class TestPlotReuse:
    """Tests for calling plot() multiple times on the same instance."""

    def test_vmin_vmax_recomputed_on_different_data(self):
        """Test that vmin/vmax are recomputed when data range changes."""
        mg = _make_tri_mg()
        mg.plot(np.array([1.0, 2.0]))
        assert mg.vmin == 1.0, f"Expected vmin=1.0, got {mg.vmin}"
        mg.plot(np.array([10.0, 20.0]))
        assert mg.vmin == 10.0, f"Expected vmin=10.0 after replot, got {mg.vmin}"
        assert mg.vmax == 20.0, f"Expected vmax=20.0 after replot, got {mg.vmax}"

    def test_explicit_vmin_vmax_preserved(self):
        """Test that user-supplied vmin/vmax are not overwritten."""
        mg = _make_tri_mg()
        mg.plot(np.array([1.0, 2.0]), vmin=0.0, vmax=5.0)
        assert mg.vmin == 0.0, f"Expected vmin=0.0, got {mg.vmin}"
        assert mg.vmax == 5.0, f"Expected vmax=5.0, got {mg.vmax}"

    def test_no_colorbar_accumulation(self):
        """Test that repeated plot() calls don't stack colorbars."""
        mg = _make_tri_mg()
        mg.plot(np.array([1.0, 2.0]))
        mg.plot(np.array([3.0, 4.0]))
        # Count axes: main axes + 1 colorbar axes = 2
        n_axes = len(mg.fig.axes)
        assert n_axes == 2, f"Expected 2 axes (plot + 1 colorbar), got {n_axes}"

    def test_plot_outline_overlays_on_existing(self):
        """Test that plot_outline() after plot() uses the same axes."""
        mg = _make_tri_mg()
        fig1, ax1 = mg.plot(np.array([1.0, 2.0]))
        fig2, ax2 = mg.plot_outline()
        assert fig1 is fig2, "plot_outline should reuse the stored figure"
        assert ax1 is ax2, "plot_outline should reuse the stored axes"


class TestAnimate:
    """Tests for MeshGlyph.animate()."""

    def test_basic_animation(self):
        """Test basic face-centered animation."""
        mg = MeshGlyph(
            np.array([0.0, 1.0, 0.5, 1.5]),
            np.array([0.0, 0.0, 1.0, 1.0]),
            np.array([[0, 1, 2], [1, 3, 2]]),
        )
        frames = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        anim = mg.animate(frames, time=["t0", "t1", "t2"])
        assert anim is not None, "Should return a FuncAnimation"
        assert mg.anim is anim, "Should store animation on instance"

    def test_node_animation(self):
        """Test node-centered animation."""
        mg = MeshGlyph(
            np.array([0.0, 1.0, 0.5, 1.5]),
            np.array([0.0, 0.0, 1.0, 1.0]),
            np.array([[0, 1, 2], [1, 3, 2]]),
        )
        frames = [
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([3.0, 2.0, 1.0, 0.0]),
        ]
        anim = mg.animate(frames, time=["t0", "t1"], location="node")
        assert anim is not None, "Should return a FuncAnimation"

    def test_animation_with_color_scale(self):
        """Test animation with power color scale."""
        mg = MeshGlyph(
            np.array([0.0, 1.0, 0.5, 1.5]),
            np.array([0.0, 0.0, 1.0, 1.0]),
            np.array([[0, 1, 2], [1, 3, 2]]),
        )
        frames = np.array([[1.0, 2.0], [3.0, 4.0]])
        anim = mg.animate(
            frames,
            time=["t0", "t1"],
            color_scale="power",
            gamma=0.5,
            cmap="coolwarm",
        )
        assert anim is not None, "Should return a FuncAnimation"

    def test_animation_time_mismatch_raises(self):
        """Test that mismatched time length raises ValueError."""
        mg = MeshGlyph(
            np.array([0.0, 1.0, 0.5]),
            np.array([0.0, 0.0, 1.0]),
            np.array([[0, 1, 2]]),
        )
        frames = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError, match="time length"):
            mg.animate(frames, time=["t0", "t1"])

    def test_animation_inconsistent_frame_length_raises(self):
        """Test that frames with different lengths raise ValueError."""
        mg = _make_tri_mg()
        frames = [np.array([1.0, 2.0]), np.array([3.0])]  # second frame wrong length
        with pytest.raises(ValueError, match="Frame 1"):
            mg.animate(frames, time=["t0", "t1"])

    def test_save_animation_gif(self, tmp_path):
        """Test saving mesh animation as GIF."""
        mg = MeshGlyph(
            np.array([0.0, 1.0, 0.5, 1.5]),
            np.array([0.0, 0.0, 1.0, 1.0]),
            np.array([[0, 1, 2], [1, 3, 2]]),
        )
        frames = np.array([[1.0, 2.0], [3.0, 4.0]])
        mg.animate(frames, time=["t0", "t1"])
        path = str(tmp_path / "test.gif")
        mg.save_animation(path, fps=2)
        assert (tmp_path / "test.gif").exists(), "GIF file should be created"
