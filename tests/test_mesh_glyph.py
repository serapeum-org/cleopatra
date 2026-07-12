"""Tests for cleopatra.mesh_glyph.MeshGlyph.

Covers mesh data plotting (tripcolor, tricontourf), wireframe rendering,
fan triangulation, face-to-triangle value mapping, and edge cases.
"""

from __future__ import annotations

import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.text import Text

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

    def test_padded_all_triangle_strips_fill(self):
        """Test that triangles in a wider padded array exclude fill values.

        Every face is a triangle but the connectivity array has 4 columns
        padded with the fill value; the triangulation must not leak ``-1``.
        """
        mg = MeshGlyph(
            np.array([0.0, 1.0, 0.5, 1.5]),
            np.array([0.0, 0.0, 1.0, 1.0]),
            np.array([[0, 1, 2, -1], [1, 3, 2, -1]], dtype=np.intp),
            fill_value=-1,
        )
        tris = mg._fan_triangles()
        assert tris.shape == (2, 3), f"Expected (2, 3), got {tris.shape}"
        assert tris.min() >= 0, "Fill value leaked into triangulation"
        np.testing.assert_array_equal(tris, np.array([[0, 1, 2], [1, 3, 2]]))

    def test_mixed_mesh_indices_correct(self, mixed_glyph):
        """Test fan triangle indices for a known quad + triangle mesh."""
        expected = np.array(
            [
                [0, 1, 4],  # quad [0,1,4,3] -> fan
                [0, 4, 3],
                [1, 2, 5],  # triangle
                [1, 5, 4],  # triangle
            ]
        )
        np.testing.assert_array_equal(mixed_glyph._fan_triangles(), expected)

    def test_pentagon_fans_into_three_triangles(self):
        """Test a single 5-node face decomposes into 3 fan triangles.

        Test scenario:
            A pentagon [0,1,2,3,4] must fan from vertex 0 into
            (0,1,2), (0,2,3), (0,3,4) -- exactly N-2 = 3 triangles.
        """
        node_x = np.array([0.0, 1.0, 2.0, 1.5, 0.5])
        node_y = np.array([0.0, 0.0, 1.0, 2.0, 2.0])
        mg = MeshGlyph(node_x, node_y, np.array([[0, 1, 2, 3, 4]], dtype=np.intp))
        expected = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]])
        np.testing.assert_array_equal(mg._fan_triangles(), expected)

    def test_pure_quad_mesh_two_triangles_each(self):
        """Test a quad-only mesh (no triangles) fans each quad into 2 triangles.

        Test scenario:
            Two quads should bypass the (n, 3) fast path and produce
            2 * 2 = 4 triangles via the vectorized mixed-mesh path.
        """
        node_x = np.array([0.0, 1.0, 1.0, 0.0, 2.0, 2.0])
        node_y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        faces = np.array([[0, 1, 2, 3], [1, 4, 5, 2]], dtype=np.intp)
        mg = MeshGlyph(node_x, node_y, faces)
        tris = mg._fan_triangles()
        assert tris.shape == (4, 3), f"Expected (4, 3), got {tris.shape}"
        expected = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]])
        np.testing.assert_array_equal(tris, expected)


class TestGroupedArange:
    """Tests for MeshGlyph._grouped_arange static helper."""

    @pytest.mark.parametrize(
        "sizes, expected",
        [
            ([3], [0, 1, 2]),
            ([1, 1, 1], [0, 0, 0]),
            ([2, 3, 1], [0, 1, 0, 1, 2, 0]),
            ([4], [0, 1, 2, 3]),
        ],
    )
    def test_positive_groups(self, sizes, expected):
        """Test concatenated per-group ranges for positive group sizes.

        Args:
            sizes: Group sizes to expand.
            expected: Concatenated [0..s-1] ranges across groups.

        Test scenario:
            Each group i contributes range(sizes[i]); the result is their
            concatenation in order.
        """
        result = MeshGlyph._grouped_arange(np.array(sizes))
        np.testing.assert_array_equal(
            result, np.array(expected), err_msg=f"sizes={sizes}"
        )

    def test_empty_input_returns_empty(self):
        """Test empty sizes yields an empty intp array (total == 0 branch).

        Test scenario:
            No groups -> length-0 result, exercising the early return.
        """
        result = MeshGlyph._grouped_arange(np.array([], dtype=np.intp))
        assert result.shape == (0,), f"Expected empty, got shape {result.shape}"
        assert result.dtype == np.intp, f"Expected intp dtype, got {result.dtype}"

    def test_all_zero_sizes_returns_empty(self):
        """Test all-zero sizes yields an empty array (total == 0 branch).

        Test scenario:
            Groups that are all empty contribute nothing, so the total is 0.
        """
        result = MeshGlyph._grouped_arange(np.array([0, 0, 0]))
        assert result.shape == (0,), f"Expected empty, got shape {result.shape}"

    def test_interspersed_zero_size_groups(self):
        """Test zero-size groups are skipped without corrupting later groups.

        Test scenario:
            sizes [2, 0, 3] -> [0, 1] (group 0), nothing (group 1),
            [0, 1, 2] (group 2). This is the robustness case the helper must
            handle: an empty middle group must not shift the counter.
        """
        result = MeshGlyph._grouped_arange(np.array([2, 0, 3]))
        np.testing.assert_array_equal(result, np.array([0, 1, 0, 1, 2]))

    def test_matches_naive_concatenation(self):
        """Test the vectorized helper matches a naive arange concatenation.

        Test scenario:
            For randomized size vectors (including zeros), the output equals
            np.concatenate([np.arange(s) for s in sizes]).
        """
        rng = np.random.default_rng(7)
        for _ in range(50):
            sizes = rng.integers(0, 6, size=int(rng.integers(0, 8)))
            naive = (
                np.concatenate([np.arange(s) for s in sizes])
                if sizes.sum() > 0
                else np.empty(0, dtype=np.intp)
            )
            result = MeshGlyph._grouped_arange(sizes)
            np.testing.assert_array_equal(
                result, naive, err_msg=f"sizes={sizes.tolist()}"
            )

    def test_result_dtype_is_intp(self):
        """Test the returned array is intp for use as a fancy index.

        Test scenario:
            Index arrays must be intp so they can index numpy arrays directly.
        """
        result = MeshGlyph._grouped_arange(np.array([2, 3]))
        assert result.dtype == np.intp, f"Expected intp, got {result.dtype}"


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


class TestPlotLineContour:
    """Tests for the node line-contour path (`filled=False`, T4.1c)."""

    def _node_glyph(self):
        """Build a fresh 2-triangle MeshGlyph for node-data tests.

        Returns:
            MeshGlyph: A glyph over four nodes / two triangles.
        """
        node_x = np.array([0.0, 1.0, 0.5, 1.5])
        node_y = np.array([0.0, 0.0, 1.0, 1.0])
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        return MeshGlyph(node_x, node_y, faces)

    def test_render_mesh_filled_true_is_filled(self):
        """_render_mesh(filled=True) yields a filled TriContourSet.

        Test scenario:
            The default node path renders filled contours (`tricontourf`).
        """
        mg = self._node_glyph()
        mg.fig, mg.ax = mg.create_figure_axes()
        mg.default_options["vmin"], mg.default_options["vmax"] = 0.0, 3.0
        cs = mg._render_mesh(mg.ax, np.array([0.0, 1.0, 2.0, 3.0]), "node", filled=True)
        assert cs.filled is True, "filled=True should produce a filled contour set"
        plt.close(mg.fig)

    def test_render_mesh_filled_false_is_line(self):
        """_render_mesh(filled=False) yields a line TriContourSet.

        Test scenario:
            The opt-in node path renders line contours (`tricontour`).
        """
        mg = self._node_glyph()
        mg.fig, mg.ax = mg.create_figure_axes()
        mg.default_options["vmin"], mg.default_options["vmax"] = 0.0, 3.0
        cs = mg._render_mesh(
            mg.ax, np.array([0.0, 1.0, 2.0, 3.0]), "node", filled=False
        )
        assert cs.filled is False, "filled=False should produce a line contour set"
        plt.close(mg.fig)

    def test_plot_filled_false_returns_fig_ax(self):
        """plot(location='node', filled=False) renders without error.

        Test scenario:
            The public plot entry point accepts filled=False for node
            data and returns a Figure/Axes.
        """
        mg = self._node_glyph()
        fig, ax = mg.plot(np.array([0.0, 1.0, 2.0, 3.0]), location="node", filled=False)
        assert fig is not None and ax is not None, "Should return a Figure and Axes"
        plt.close(fig)

    def test_plot_filled_false_honours_levels(self):
        """Line contours honour an explicit `levels` count.

        Test scenario:
            Passing levels through kwargs reaches tricontour without
            error (the levels/cmap options are forwarded).
        """
        mg = self._node_glyph()
        fig, ax = mg.plot(
            np.array([0.0, 1.0, 2.0, 3.0]),
            location="node",
            filled=False,
            levels=4,
        )
        assert fig is not None, "Should render line contours with explicit levels"
        plt.close(fig)

    def test_face_data_ignores_filled_flag(self, triangle_glyph):
        """Face data always uses tripcolor regardless of `filled`.

        Test scenario:
            filled=False on face data still renders (tripcolor), proving
            the flag only affects the node path.
        """
        fig, ax = triangle_glyph.plot(
            np.array([1.0, 2.0]), location="face", filled=False
        )
        assert fig is not None, "Face data should render irrespective of filled"
        plt.close(fig)

    def test_render_mesh_norm_none_unset_vmin_vmax(self):
        """`_render_mesh` omits vmin/vmax when both are None and norm is None.

        Test scenario:
            Calling `_render_mesh` directly with `norm=None` while
            `default_options["vmin"]`/`["vmax"]` are still their unset
            `None` defaults must skip both colour-limit kwargs (the
            `else` branch where neither bound is forwarded) and still
            render a line contour set.
        """
        mg = self._node_glyph()
        mg.fig, mg.ax = mg.create_figure_axes()
        mg.default_options["vmin"] = None
        mg.default_options["vmax"] = None
        cs = mg._render_mesh(
            mg.ax, np.array([0.0, 1.0, 2.0, 3.0]), "node", filled=False
        )
        assert cs.filled is False, "Should render line contours with no vmin/vmax"
        plt.close(mg.fig)


class TestContourLabels:
    """Inline label support for line tricontours (issue #151).

    Mirrors `ArrayGlyph`'s `plot(kind="contour", labels=True)` (#148/#149)
    on `MeshGlyph`'s unstructured `tricontour` path.
    """

    @staticmethod
    def _grid_glyph(n: int = 12) -> MeshGlyph:
        """A quad grid mesh over [-3, 3]^2 with `n` nodes per side.

        Args:
            n: Number of nodes along each axis.

        Returns:
            MeshGlyph: A glyph whose node field can carry several
                labelled isolines.
        """
        xs = np.linspace(-3.0, 3.0, n)
        gx, gy = np.meshgrid(xs, xs)
        node_x = gx.ravel()
        node_y = gy.ravel()
        # Quad faces over the structured grid (fanned into triangles
        # internally by the glyph).
        i = np.arange(n - 1)
        j = np.arange(n - 1)
        jj, ii = np.meshgrid(j, i, indexing="ij")
        a = (jj * n + ii).ravel()
        faces = np.column_stack([a, a + 1, a + n + 1, a + n])
        return MeshGlyph(node_x, node_y, faces)

    @staticmethod
    def _smooth_field(mg: MeshGlyph) -> np.ndarray:
        """A smooth Gaussian node field that yields many isolines."""
        return np.exp(-(mg.node_x**2 + mg.node_y**2))

    def test_labels_true_draws_and_exposes_text(self):
        """`labels=True` populates `contour_labels` with `Text` artists."""
        mg = self._grid_glyph()
        fig, ax = mg.plot(
            self._smooth_field(mg), location="node", filled=False, labels=True
        )
        assert isinstance(mg.contour_labels, list)
        assert len(mg.contour_labels) > 0
        assert all(isinstance(t, Text) for t in mg.contour_labels)
        # The label artists are attached to the axes' text list.
        assert set(mg.contour_labels).issubset(set(ax.texts))
        plt.close(fig)

    def test_labels_default_is_noop(self):
        """Default (`labels=False`) draws no labels; `contour_labels` is None."""
        mg = self._grid_glyph()
        fig, ax = mg.plot(self._smooth_field(mg), location="node", filled=False)
        assert mg.contour_labels is None
        assert len(ax.texts) == 0
        plt.close(fig)

    def test_label_kw_forwarded_to_clabel(self):
        """`label_kw` reaches `ax.clabel` (custom `fmt`/`fontsize` applied)."""
        mg = self._grid_glyph()
        fig, ax = mg.plot(
            self._smooth_field(mg),
            location="node",
            filled=False,
            labels=True,
            label_kw={"fmt": "%.3f", "fontsize": 6},
        )
        assert len(mg.contour_labels) > 0
        # The custom fontsize on every label proves label_kw was forwarded.
        assert all(t.get_fontsize() == 6 for t in mg.contour_labels)
        # The "%.3f" format yields three decimal places in every label.
        assert all(re.search(r"\.\d{3}$", t.get_text()) for t in mg.contour_labels)
        plt.close(fig)

    def test_labels_on_tricontourf_is_noop(self):
        """`labels=True` is ignored for `filled=True` (no isolines to label)."""
        mg = self._grid_glyph()
        fig, ax = mg.plot(
            self._smooth_field(mg), location="node", filled=True, labels=True
        )
        assert mg.contour_labels is None
        plt.close(fig)

    def test_labels_on_face_data_is_noop(self, triangle_glyph):
        """`labels=True` is ignored for face data (`tripcolor`)."""
        fig, ax = triangle_glyph.plot(
            np.array([1.0, 2.0]), location="face", labels=True
        )
        assert triangle_glyph.contour_labels is None
        plt.close(fig)

    def test_replot_without_labels_resets_contour_labels(self):
        """Re-plotting with `labels=False` clears a prior render's labels."""
        mg = self._grid_glyph()
        data = self._smooth_field(mg)
        mg.plot(data, location="node", filled=False, labels=True)
        assert mg.contour_labels is not None
        fig, ax = mg.plot(data, location="node", filled=False)
        assert mg.contour_labels is None
        plt.close(fig)

    def test_switching_to_filled_resets_contour_labels(self):
        """A subsequent filled render clears stale label artists."""
        mg = self._grid_glyph()
        data = self._smooth_field(mg)
        mg.plot(data, location="node", filled=False, labels=True)
        assert mg.contour_labels is not None
        fig, ax = mg.plot(data, location="node", filled=True, labels=True)
        assert mg.contour_labels is None
        plt.close(fig)

    def test_labels_on_constant_field_is_empty_list(self):
        """A labelled contour with no isolines yields [] (not None), no raise."""
        mg = self._grid_glyph()
        constant = np.full(mg.n_nodes, 3.0)
        with warnings.catch_warnings():
            # A constant field has no contour lines; matplotlib's
            # "no contour levels" warning is expected and unrelated.
            # `colorbar=False` avoids matplotlib's colorbar choking on a
            # line set with zero isolines (an orthogonal limitation).
            warnings.simplefilter("ignore")
            fig, ax = mg.plot(
                constant,
                location="node",
                filled=False,
                labels=True,
                colorbar=False,
            )
        assert mg.contour_labels == []
        plt.close(fig)

    def test_label_kw_overrides_cleopatra_defaults(self):
        """User `label_kw` keys win over cleopatra's clabel defaults."""
        mg = self._grid_glyph()
        # Default fontsize is 8; the user value must take precedence.
        fig, ax = mg.plot(
            self._smooth_field(mg),
            location="node",
            filled=False,
            labels=True,
            label_kw={"fontsize": 14},
        )
        assert len(mg.contour_labels) > 0
        assert all(t.get_fontsize() == 14 for t in mg.contour_labels)
        plt.close(fig)

    def test_label_kw_forwards_arbitrary_clabel_kwarg(self):
        """A non-default `label_kw` key (`colors`) reaches `ax.clabel`."""
        mg = self._grid_glyph()
        fig, ax = mg.plot(
            self._smooth_field(mg),
            location="node",
            filled=False,
            labels=True,
            label_kw={"colors": "red"},
        )
        assert len(mg.contour_labels) > 0
        red = to_rgba("red")
        assert all(
            to_rgba(t.get_color()) == red for t in mg.contour_labels
        ), "every label should be red when label_kw={'colors': 'red'}"
        plt.close(fig)

    def test_contour_labels_none_before_any_render(self):
        """`contour_labels` is None on a freshly constructed glyph."""
        mg = self._grid_glyph()
        assert (
            mg.contour_labels is None
        ), f"Expected None before render, got {mg.contour_labels!r}"

    def test_animate_resets_stale_contour_labels(self):
        """`animate()` clears labels left by a prior `plot(labels=True)`."""
        mg = self._grid_glyph()
        field = self._smooth_field(mg)
        mg.plot(field, location="node", filled=False, labels=True)
        assert mg.contour_labels is not None
        # An animation draws no inline labels; the stale list must be cleared.
        anim = mg.animate([field, field * 2.0], time=["t0", "t1"], location="node")
        assert mg.contour_labels is None
        assert anim is not None
        plt.close(mg.fig)


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

    def test_single_triangle_three_edges(self):
        """Test a lone triangle yields exactly 3 undirected edges.

        Test scenario:
            Face [0,1,2] has boundary edges (0,1), (1,2), (0,2); none are
            shared, so all 3 survive deduplication.
        """
        node_x = np.array([0.0, 1.0, 0.5])
        node_y = np.array([0.0, 0.0, 1.0])
        mg = MeshGlyph(node_x, node_y, np.array([[0, 1, 2]], dtype=np.intp))
        segs = mg._build_edge_segments()
        assert segs.shape == (3, 2, 2), f"Expected (3, 2, 2), got {segs.shape}"

    def test_shared_edge_deduplicated_across_faces(self):
        """Test an edge shared by two faces appears only once.

        Test scenario:
            Two triangles [0,1,2] and [1,3,2] share edge (1,2). The unique
            edge set is {(0,1),(0,2),(1,2),(1,3),(2,3)} -> 5 segments, with
            the shared (1,2) collapsed to a single entry.
        """
        node_x = np.array([0.0, 1.0, 0.5, 1.5])
        node_y = np.array([0.0, 0.0, 1.0, 1.0])
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp)
        mg = MeshGlyph(node_x, node_y, faces)
        segs = mg._build_edge_segments()
        coord_to_idx = {
            (round(x, 9), round(y, 9)): i
            for i, (x, y) in enumerate(zip(node_x, node_y))
        }
        edges = {
            tuple(
                sorted(
                    (
                        coord_to_idx[(round(s[0, 0], 9), round(s[0, 1], 9))],
                        coord_to_idx[(round(s[1, 0], 9), round(s[1, 1], 9))],
                    )
                )
            )
            for s in segs
        }
        assert edges == {
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 3),
        }, f"Unexpected edge set: {edges}"
        assert segs.shape[0] == 5, f"Expected 5 unique edges, got {segs.shape[0]}"


class TestVectorizedEquivalenceAndPerformance:
    """Tests that the vectorized triangulation/edge derivation are correct
    and fast on large mixed-element meshes."""

    @staticmethod
    def _reference_triangles(face_nodes, fill):
        """Original Python-loop fan triangulation, used as ground truth."""
        out = []
        for row in face_nodes:
            nodes = row[row != fill]
            n = len(nodes)
            for j in range(1, n - 1):
                out.append((int(nodes[0]), int(nodes[j]), int(nodes[j + 1])))
        return set(out)

    @staticmethod
    def _reference_edges(face_nodes, fill):
        """Original Python-loop edge derivation, used as ground truth."""
        edges = set()
        for row in face_nodes:
            nodes = row[row != fill]
            n = len(nodes)
            for j in range(n):
                a, b = int(nodes[j]), int(nodes[(j + 1) % n])
                edges.add((min(a, b), max(a, b)))
        return edges

    def test_vectorized_matches_reference_on_random_meshes(self):
        """Test vectorized output matches the naive loop on random meshes."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            n_faces = int(rng.integers(1, 8))
            max_nodes = int(rng.integers(3, 6))
            n_nodes = int(rng.integers(4, 12))
            faces = np.full((n_faces, max_nodes), -1, dtype=np.intp)
            for i in range(n_faces):
                c = int(rng.integers(0, max_nodes + 1))
                if c > 0:
                    faces[i, :c] = rng.integers(0, n_nodes, size=c)
            # x-coordinate equals the node index (exactly representable in
            # float64), so a segment endpoint maps back to its node with no
            # rounding ambiguity; y is arbitrary and unused for recovery.
            node_x = np.arange(n_nodes, dtype=float)
            node_y = rng.random(n_nodes)
            counts = np.sum(faces != -1, axis=1)

            if np.any(counts >= 3):
                tris = MeshGlyph(node_x, node_y, faces)._fan_triangles()
                got_tris = set(map(tuple, tris.tolist()))
                assert got_tris == self._reference_triangles(faces, -1)

            segs = MeshGlyph(node_x, node_y, faces)._build_edge_segments()
            got_edges = set()
            for seg in segs:
                i = int(seg[0, 0])
                j = int(seg[1, 0])
                got_edges.add((min(i, j), max(i, j)))
            assert got_edges == self._reference_edges(faces, -1)

    def test_large_mixed_mesh_performance(self):
        """Test 100k mixed-element triangulation/edges stay far from O(n) loop times.

        Guards against accidental reintroduction of a Python-loop
        implementation rather than asserting a tight millisecond budget, so it
        stays stable on slow or loaded CI runners. Mesh construction is kept
        out of the timed region; the vectorized paths complete in tens of
        milliseconds locally, so the 2-second bound only trips if the per-face
        Python loop comes back.
        """
        n = 100_000
        rng = np.random.default_rng(0)
        node_x = rng.random(n * 2)
        node_y = rng.random(n * 2)
        faces = np.column_stack(
            [
                np.arange(0, n * 2, 2),
                np.arange(1, n * 2 + 1, 2),
                np.arange(1, n * 2 + 1, 2) + 1,
                np.full(n, -1),
            ]
        ).astype(np.intp)
        # Make alternate rows quads so the mesh is genuinely mixed-element.
        faces[::2, 3] = faces[::2, 2]
        faces[faces >= n * 2] = n * 2 - 1

        # Separate instances so neither method benefits from the other's cache.
        tri_glyph = MeshGlyph(node_x, node_y, faces, fill_value=-1)
        edge_glyph = MeshGlyph(node_x, node_y, faces, fill_value=-1)

        t0 = time.perf_counter()
        tri_glyph._fan_triangles()
        tri_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        edge_glyph._build_edge_segments()
        edge_s = time.perf_counter() - t1

        assert tri_s < 2.0, f"_fan_triangles too slow: {tri_s * 1000:.1f}ms"
        assert edge_s < 2.0, f"_build_edge_segments too slow: {edge_s * 1000:.1f}ms"


class TestMeshGlyphSubplots:
    """Tests for MeshGlyph.plot on shared subplot figures."""

    def test_two_mesh_plots_on_subplots(self, triangle_glyph):
        """Test two mesh plots on a 1x2 subplot layout.

        Test scenario:
            Both axes should have rendered content after plotting
            on separate subplots of the same figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        data = np.array([1.0, 2.0])

        mg1 = MeshGlyph(
            triangle_glyph.node_x,
            triangle_glyph.node_y,
            triangle_glyph._face_nodes,
        )
        mg1.plot(data, location="face", ax=axes[0], title="Left")

        mg2 = MeshGlyph(
            triangle_glyph.node_x,
            triangle_glyph.node_y,
            triangle_glyph._face_nodes,
        )
        mg2.plot(data * 2, location="face", ax=axes[1], title="Right")

        assert (
            axes[0].get_title() == "Left"
        ), f"Expected 'Left', got '{axes[0].get_title()}'"
        assert (
            axes[1].get_title() == "Right"
        ), f"Expected 'Right', got '{axes[1].get_title()}'"
        assert len(fig.axes) >= 2, f"Expected at least 2 axes, got {len(fig.axes)}"
        plt.close(fig)

    def test_mesh_plot_and_outline_on_subplots(self, triangle_glyph):
        """Test mesh data plot and wireframe on shared subplots.

        Test scenario:
            Left subplot gets face data, right gets wireframe.
            Both should render without interfering.
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        mg1 = MeshGlyph(
            triangle_glyph.node_x,
            triangle_glyph.node_y,
            triangle_glyph._face_nodes,
        )
        mg1.plot(np.array([1.0, 2.0]), location="face", ax=axes[0])
        mg1.plot_outline(ax=axes[1], color="blue")

        assert len(axes[1].collections) >= 1, "Right axes should have a LineCollection"
        plt.close(fig)


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

    def test_ticks_spacing_written_to_default_options(self):
        """Test that computed ticks_spacing is written to default_options.

        Regression test for C1: get_ticks() reads from default_options,
        so the computed spacing must be written there.
        """
        mg = _make_tri_mg()
        mg.plot(np.array([0.0, 0.1]))
        assert mg.default_options["ticks_spacing"] == pytest.approx(
            0.01
        ), f"Expected ticks_spacing=0.01, got {mg.default_options['ticks_spacing']}"

    def test_constant_data_does_not_crash(self):
        """Test that plotting constant data (vmin==vmax) does not crash.

        Regression test for H1: zero ticks_spacing must be guarded.
        """
        mg = _make_tri_mg()
        fig, ax = mg.plot(np.array([5.0, 5.0]))
        assert fig is not None, "Should return a Figure for constant data"

    def test_explicit_ticks_spacing_preserved(self):
        """User-supplied `ticks_spacing` skips auto-computation.

        Test scenario:
            Passing `ticks_spacing` to plot() must take the explicit
            value (the branch that does not recompute spacing from the
            data range) rather than `(vmax - vmin) / 10`.
        """
        mg = _make_tri_mg()
        fig, ax = mg.plot(np.array([0.0, 10.0]), ticks_spacing=2.5)
        assert (
            mg.ticks_spacing == 2.5
        ), f"Expected explicit ticks_spacing=2.5, got {mg.ticks_spacing}"
        plt.close(fig)

    def test_all_nan_data_raises(self):
        """Test that all-NaN data raises a clear ValueError.

        Regression test for H2: np.nanmin on all-NaN returns nan
        which crashes downstream.
        """
        mg = _make_tri_mg()
        with pytest.raises(ValueError, match="entirely NaN"):
            mg.plot(np.array([np.nan, np.nan]))

    def test_kwargs_not_persisted_across_plot_calls(self):
        """Test that kwargs from one plot() call don't leak to the next.

        Regression test for H3: default_options must be reset on each
        plot() call so module-scoped fixtures aren't polluted.
        """
        mg = _make_tri_mg()
        mg.plot(np.array([1.0, 2.0]), cmap="plasma")
        assert (
            mg.default_options["cmap"] == "plasma"
        ), "Should be plasma after first call"
        mg.plot(np.array([3.0, 4.0]))
        assert (
            mg.default_options["cmap"] == "coolwarm_r"
        ), f"Should reset to default coolwarm_r, got {mg.default_options['cmap']}"


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

    def test_animate_explicit_options_and_title(self):
        """animate honours explicit text_loc, vmin/vmax, ticks_spacing, title.

        Test scenario:
            Supplying all of `text_loc`, `vmin`, `vmax`, `ticks_spacing`
            and `title` exercises the non-default branches: the time
            label uses the given position, the colour limits and tick
            spacing are taken verbatim (not recomputed from frames), and
            the axes title is set.
        """
        mg = _make_tri_mg()
        frames = np.array([[1.0, 2.0], [3.0, 4.0]])
        anim = mg.animate(
            frames,
            time=["t0", "t1"],
            text_loc=[0.5, 0.5],
            vmin=0.0,
            vmax=10.0,
            ticks_spacing=2.0,
            title="My Animation",
        )
        assert anim is not None, "Should return a FuncAnimation"
        assert mg.vmin == 0.0, f"Expected explicit vmin=0.0, got {mg.vmin}"
        assert mg.vmax == 10.0, f"Expected explicit vmax=10.0, got {mg.vmax}"
        assert (
            mg.ticks_spacing == 2.0
        ), f"Expected explicit ticks_spacing=2.0, got {mg.ticks_spacing}"
        assert (
            mg.ax.get_title() == "My Animation"
        ), f"Expected title 'My Animation', got {mg.ax.get_title()!r}"
        plt.close(mg.fig)

    def test_animate_update_advances_frame(self):
        """Driving the frame-update callback re-renders and relabels.

        Test scenario:
            Invoking the `FuncAnimation` update function for frame 1
            removes the previous mappable and sets the time-label text,
            exercising the `_update` closure that a non-rendered
            animation otherwise never runs.
        """
        mg = _make_tri_mg()
        frames = np.array([[1.0, 2.0], [3.0, 4.0]])
        anim = mg.animate(frames, time=["t0", "t1"])
        anim._func(1)
        time_labels = [t.get_text() for t in mg.ax.texts if t.get_text() == "t1"]
        assert time_labels, "Frame update should set the time label to 't1'"
        plt.close(mg.fig)

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


class TestMeshGlyphMappable:
    """`MeshGlyph` exposes the colour-mapped artist as `self.im` (#2)."""

    @staticmethod
    def _mesh():
        from matplotlib.tri import Triangulation

        rng = np.random.default_rng(0)
        x = rng.random(20)
        y = rng.random(20)
        return MeshGlyph(x, y, Triangulation(x, y).triangles), x

    def test_im_none_before_plot(self):
        """`self.im` is None before the first render."""
        glyph, _ = self._mesh()
        assert glyph.im is None, "im should start as None"

    def test_plot_sets_im_to_live_artist(self):
        """After `plot`, `self.im` is the artist registered on the axes.

        Test scenario:
            The tricontour/tripcolor mappable is stored on `self.im` so a
            caller need not scrape `ax.collections[-1]`.
        """
        glyph, x = self._mesh()
        z = np.random.default_rng(1).random(x.size)
        fig, ax = glyph.plot(z, location="node", colorbar=False)
        try:
            assert glyph.im is not None, "im should be set after plot"
            assert glyph.im in ax.collections, "im must be the artist on the axes"
        finally:
            plt.close(fig)

    def test_plot_outline_clears_im(self):
        """`plot_outline` resets `self.im` (an outline has no scalar mapping).

        Test scenario:
            After a coloured `plot()` sets `self.im`, an outline-only render
            must clear it rather than leave a stale mappable.
        """
        glyph, x = self._mesh()
        z = np.random.default_rng(2).random(x.size)
        fig, ax = glyph.plot(z, location="node", colorbar=False)
        plt.close(fig)
        assert glyph.im is not None, "im should be set after a coloured plot"
        fig, ax = glyph.plot_outline()
        plt.close(fig)
        assert glyph.im is None, "plot_outline should clear im"


class TestMeshGlyphHillshade:
    """Tests for the `hillshade` relief-shading option (triangulated terrain)."""

    @staticmethod
    def _terrain_mesh(n=12):
        gx, gy = np.meshgrid(np.linspace(0, 10, n), np.linspace(0, 10, n))
        nx, ny = gx.ravel(), gy.ravel()
        z = 50.0 + 150.0 * np.exp(-(((nx - 7) / 2) ** 2 + ((ny - 5) / 3) ** 2)) + 6.0 * np.sin(nx)
        faces = np.array(
            [[j * n + i, j * n + i + 1, j * n + i + n + 1, j * n + i + n]
             for j in range(n - 1) for i in range(n - 1)]
        )
        return nx, ny, faces, z

    def test_node_hillshade_draws_shaded_polycollection(self):
        """Node-centered `hillshade` renders a per-face shaded `tripcolor` mesh."""
        nx, ny, faces, z = self._terrain_mesh()
        mg = MeshGlyph(nx, ny, faces)
        mg.plot(z, location="node", cmap="terrain", hillshade={"vert_exag": 3})
        facecolors = mg.im.get_facecolor()
        assert facecolors.shape[1] == 4
        assert len(np.unique(np.round(facecolors[:, 0], 3))) > 5, "faces should vary"
        assert mg._cbar is not None
        plt.close("all")

    def test_face_data_with_hillshade_raises(self):
        """Hillshade needs node elevation; face-centered data raises `ValueError`."""
        nx, ny, faces, _ = self._terrain_mesh()
        with pytest.raises(ValueError, match="node-centered"):
            MeshGlyph(nx, ny, faces).plot(
                np.ones(len(faces)), location="face", hillshade=True
            )
        plt.close("all")

    def test_without_hillshade_uses_contours(self):
        """Node data without hillshade still uses the contour path (not tripcolor)."""
        nx, ny, faces, z = self._terrain_mesh()
        mg = MeshGlyph(nx, ny, faces)
        mg.plot(z, location="node")
        assert "Contour" in type(mg.im).__name__
        plt.close("all")

    def test_constructor_time_hillshade_is_honoured(self):
        """`hillshade` set at construction shades the mesh, like ArrayGlyph/KDEGlyph.

        MeshGlyph.plot() resets default_options each call, so the option is
        restored from the constructor value when plot() does not override it.
        """
        nx, ny, faces, z = self._terrain_mesh()
        mg = MeshGlyph(nx, ny, faces, hillshade=True)
        mg.plot(z, location="node", cmap="terrain")
        assert type(mg.im).__name__ == "PolyCollection", "constructor hillshade should shade"
        assert mg.im.get_array() is None, "shaded relief uses explicit facecolors"
        plt.close("all")

    def test_hillshade_with_labels_and_lines_does_not_crash(self):
        """`labels=True`, `filled=False`, and `hillshade` together are a safe no-op.

        The shaded relief is a `PolyCollection` with no isolines, so `clabel`
        must be skipped rather than raising `AttributeError`.
        """
        nx, ny, faces, z = self._terrain_mesh()
        mg = MeshGlyph(nx, ny, faces)
        mg.plot(z, location="node", filled=False, labels=True, hillshade=True)
        assert mg.contour_labels is None, "labels are a no-op under hillshade"
        plt.close("all")

    def test_hillshade_honours_nonlinear_color_scale(self):
        """A non-linear `color_scale` feeds its `norm` into the shaded tripcolor.

        Exercises the `norm is not None` branch of `_render_shaded_relief`,
        which the default (linear) hillshade tests skip.
        """
        nx, ny, faces, z = self._terrain_mesh()
        mg = MeshGlyph(nx, ny, faces)
        mg.plot(z, location="node", cmap="terrain", color_scale="power",
                gamma=0.4, hillshade=True)
        assert type(mg.im).__name__ == "PolyCollection"
        assert type(mg.im.norm).__name__ == "PowerNorm", "the color_scale norm is applied"
        plt.close("all")

    def test_hillshade_makes_nodata_faces_transparent(self):
        """Faces touching a non-finite (nodata) node render fully transparent.

        Mirrors the raster hillshade, which drops NaN cells rather than
        colouring them; a triangle with a NaN node has no defined normal.
        """
        nx = np.array([0.0, 1.0, 0.0, 1.0])
        ny = np.array([0.0, 0.0, 1.0, 1.0])
        faces = np.array([[0, 1, 3], [1, 2, 3], [0, 1, 2]])
        z = np.array([10.0, 20.0, 30.0, np.nan])  # node 3 is nodata
        mg = MeshGlyph(nx, ny, faces)
        mg.plot(z, location="node", cmap="terrain", hillshade=True)
        alphas = mg.im.get_facecolor()[:, 3]
        assert np.allclose(alphas[[0, 1]], 0.0), "nodata-touching faces are transparent"
        assert alphas[2] > 0.0, "the fully-finite face stays opaque"
        plt.close("all")


class TestMeshGlyphDataStyle:
    """Tests for the `style` data-style preset option on MeshGlyph."""

    @staticmethod
    def _mesh(n=8):
        gx, gy = np.meshgrid(np.linspace(0, 10, n), np.linspace(0, 10, n))
        nx, ny = gx.ravel(), gy.ravel()
        faces = np.array(
            [
                [j * n + i, j * n + i + 1, j * n + i + n + 1, j * n + i + n]
                for j in range(n - 1)
                for i in range(n - 1)
            ]
        )
        return nx, ny, faces

    def test_continuous_preset_sets_cmap_and_norm(self):
        """A continuous preset drives the mesh through its cmap + norm."""
        nx, ny, faces = self._mesh()
        fvals = np.abs(np.random.default_rng(0).normal(size=len(faces))) * 100
        g = MeshGlyph(nx, ny, faces)
        g.plot(fvals, location="face", style="flow_accumulation")
        assert g.im.cmap.name == "Blues"
        assert type(g.im.norm).__name__ == "SymLogNorm"
        plt.close("all")

    def test_categorical_preset_draws_disjoint_legend(self):
        """A categorical preset renders a discrete legend and no colorbar."""
        nx, ny, faces = self._mesh()
        d8 = np.random.default_rng(1).choice(
            [1, 2, 4, 8, 16, 32, 64, 128], size=len(faces)
        ).astype(float)
        g = MeshGlyph(nx, ny, faces)
        _, ax = g.plot(d8, location="face", style="flow_direction_d8")
        assert ax.get_legend() is not None
        assert g._cbar is None
        plt.close("all")

    def test_unknown_style_raises(self):
        """An unknown style name raises a clear `ValueError`."""
        nx, ny, faces = self._mesh()
        with pytest.raises(ValueError, match="unknown data style"):
            MeshGlyph(nx, ny, faces).plot(
                np.ones(len(faces)), location="face", style="not_a_style"
            )
        plt.close("all")

    def test_continuous_style_composes_with_hillshade(self):
        """A continuous preset composes with node-elevation relief shading."""
        nx, ny, faces = self._mesh()
        z = 50 + 150 * np.exp(-(((nx - 7) / 2) ** 2 + ((ny - 5) / 3) ** 2))
        g = MeshGlyph(nx, ny, faces)
        g.plot(z, location="node", style="topography", hillshade=True)
        assert type(g.im).__name__ == "PolyCollection"
        plt.close("all")

    def test_categorical_preset_with_node_location_warns(self):
        """A categorical preset on node data warns (tricontourf interpolates class codes)."""
        nx, ny, faces = self._mesh()
        codes = np.random.default_rng(2).choice(
            [1, 2, 4, 8, 16, 32, 64, 128], size=len(nx)
        ).astype(float)
        with pytest.warns(UserWarning, match="interpolates discrete class codes"):
            MeshGlyph(nx, ny, faces).plot(
                codes, location="node", style="flow_direction_d8"
            )
        plt.close("all")

    def test_symlog_preset_colorbar_uses_a_log_locator(self):
        """A symlog preset's colorbar picks a log locator, not the linear ticks."""
        nx, ny, faces = self._mesh()
        fvals = np.abs(np.random.default_rng(0).normal(size=len(faces))) * 100
        g = MeshGlyph(nx, ny, faces)
        g.plot(fvals, location="face", style="flow_accumulation")
        assert type(g._cbar.locator).__name__ == "SymmetricalLogLocator"
        plt.close("all")


class TestMeshGlyphApplyStyle:
    """Tests for the public `apply_style` method and `style` read-back."""

    @staticmethod
    def _mesh(n=8):
        gx, gy = np.meshgrid(np.linspace(0, 10, n), np.linspace(0, 10, n))
        nx, ny = gx.ravel(), gy.ravel()
        faces = np.array(
            [
                [j * n + i, j * n + i + 1, j * n + i + n + 1, j * n + i + n]
                for j in range(n - 1)
                for i in range(n - 1)
            ]
        )
        return nx, ny, faces

    def test_apply_style_reuses_last_data_and_reads_back(self):
        """apply_style restyles using the last-plotted data; `style` reads back."""
        nx, ny, faces = self._mesh()
        fvals = np.abs(np.random.default_rng(0).normal(size=len(faces))) * 100
        g = MeshGlyph(nx, ny, faces)
        g.plot(fvals, location="face")
        g.apply_style("flow_accumulation")
        assert g.style == "flow_accumulation"
        assert g.im.cmap.name == "Blues"
        plt.close("all")

    def test_construction_time_style_is_honoured(self):
        """A `style` set at construction survives plot()'s options reset."""
        nx, ny, faces = self._mesh()
        fvals = np.abs(np.random.default_rng(1).normal(size=len(faces))) * 100
        g = MeshGlyph(nx, ny, faces, style="flow_accumulation")
        g.plot(fvals, location="face")
        assert g.style == "flow_accumulation" and g.im.cmap.name == "Blues"
        plt.close("all")

    def test_apply_style_without_data_or_prior_plot_raises(self):
        """apply_style before any plot and with no data raises a clear error."""
        nx, ny, faces = self._mesh()
        with pytest.raises(ValueError, match="apply_style needs mesh data"):
            MeshGlyph(nx, ny, faces).apply_style("flow_accumulation")
        plt.close("all")

    def test_apply_style_unknown_name_raises(self):
        """An unknown preset name raises a clear `ValueError`."""
        nx, ny, faces = self._mesh()
        fvals = np.ones(len(faces))
        g = MeshGlyph(nx, ny, faces)
        g.plot(fvals, location="face")
        with pytest.raises(ValueError, match="unknown data style"):
            g.apply_style("not_a_style")
        plt.close("all")

    def test_style_is_sticky_and_clearable(self):
        """A style survives a later plain plot(data) and is cleared by style=None."""
        nx, ny, faces = self._mesh()
        fvals = np.abs(np.random.default_rng(4).normal(size=len(faces))) * 100
        g = MeshGlyph(nx, ny, faces)
        g.plot(fvals, location="face", style="flow_accumulation")
        g.plot(fvals, location="face")
        assert g.style == "flow_accumulation"
        g.plot(fvals, location="face", style=None)
        assert g.style is None
        plt.close("all")

    def test_apply_style_repeated_does_not_stack_legends(self):
        """Restyling a mesh repeatedly replaces the render rather than stacking legends."""
        nx, ny, faces = self._mesh()
        fvals = np.abs(np.random.default_rng(5).normal(size=len(faces))) * 100
        g = MeshGlyph(nx, ny, faces)
        g.plot(fvals, location="face")
        g.apply_style("flow_accumulation")
        g.apply_style("topography")
        assert g.style == "topography"
        n_cbar_axes = sum(1 for a in g.fig.axes if a is not g.ax)
        assert n_cbar_axes == 1  # one colorbar, not stacked
        plt.close("all")

    def test_apply_style_data_override_and_copy(self):
        """apply_style uses an explicit data= override and the cached data is a copy."""
        nx, ny, faces = self._mesh()
        fvals = np.abs(np.random.default_rng(6).normal(size=len(faces))) * 100
        g = MeshGlyph(nx, ny, faces)
        g.plot(fvals, location="face")
        fvals[:] = 0.0  # mutate caller buffer after plot -> must not affect cache
        assert not np.allclose(g._last_data, 0.0)
        other = np.abs(np.random.default_rng(7).normal(size=len(faces))) * 50
        g.apply_style("flow_accumulation", data=other)
        assert g.style == "flow_accumulation"
        plt.close("all")
