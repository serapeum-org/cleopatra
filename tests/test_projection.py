"""Tests for ``cleopatra.projection``.

Covers the public helper ``apply_projection_frame`` and the private
``_as_xy`` coercion utility. The module is pure matplotlib with no PROJ
dependency, so all geometry is built inline as deterministic numpy arrays
(no reprojection, no network, no filesystem).

Tests are grouped one class per function. Randomised image data is seeded
so runs are deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.plot

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

from cleopatra.projection import (  # noqa: E402
    DEFAULT_BOUNDARY_KW,
    DEFAULT_GRATICULE_KW,
    _as_xy,
    apply_projection_frame,
)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded numpy generator for deterministic image data.

    Returns:
        numpy.random.Generator: A generator seeded with a fixed value so
        every test run produces identical pixel data.
    """
    return np.random.default_rng(1337)


@pytest.fixture
def globe_boundary() -> np.ndarray:
    """Closed unit-circle boundary -- the canonical globe outline.

    Returns:
        numpy.ndarray: A ``(200, 2)`` array of x/y vertices tracing the
        unit circle, with the final vertex repeating the first.
    """
    theta = np.linspace(0, 2 * np.pi, 200)
    return np.column_stack([np.cos(theta), np.sin(theta)])


@pytest.fixture
def meridian() -> np.ndarray:
    """A single vertical graticule polyline (the prime meridian).

    Returns:
        numpy.ndarray: A ``(50, 2)`` polyline from the south to the north
        pole along ``x == 0``.
    """
    return np.column_stack([np.zeros(50), np.linspace(-1, 1, 50)])


@pytest.fixture
def axes():
    """Create a fresh figure/axes pair and close it after the test.

    Yields:
        matplotlib.axes.Axes: A clean axes with no data plotted.
    """
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


class TestAsXy:
    """Tests for the private ``_as_xy`` coercion helper."""

    def test_coerces_nested_list_to_float_array(self):
        """Coerce a nested Python list into a float ``(N, 2)`` array.

        Test scenario:
            A list of integer x/y pairs is converted to a numpy array with
            float dtype and ``(N, 2)`` shape.
        """
        result = _as_xy([[1, 0], [0, 1], [-1, 0]], "boundary_xy")
        assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"
        assert result.shape == (3, 2), f"Expected shape (3, 2), got {result.shape}"
        assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"

    def test_passes_through_float_ndarray(self):
        """Return an equivalent array for an already-valid float ndarray.

        Test scenario:
            A ``(N, 2)`` float array is returned unchanged in value/shape.
        """
        source = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = _as_xy(source, "boundary_xy")
        assert result.shape == (2, 2), f"Expected shape (2, 2), got {result.shape}"
        assert np.array_equal(result, source), f"Values changed: {result}"

    def test_accepts_single_vertex(self):
        """Accept the minimal ``(1, 2)`` array without raising.

        Test scenario:
            A single x/y pair is a valid 2-D, two-column array.
        """
        result = _as_xy([[0.5, 0.5]], "graticule_lines[0]")
        assert result.shape == (1, 2), f"Expected shape (1, 2), got {result.shape}"

    @pytest.mark.parametrize(
        "bad_input, shape_desc",
        [
            (np.arange(10), "(10,)"),
            (np.zeros((4, 3)), "(4, 3)"),
            (np.zeros((2, 2, 2)), "(2, 2, 2)"),
            (5.0, "()"),
        ],
    )
    def test_rejects_non_n_by_2(self, bad_input, shape_desc):
        """Raise ``ValueError`` for inputs that are not ``(N, 2)``.

        Args:
            bad_input: Array-like with an invalid shape.
            shape_desc: Human-readable shape, for the assertion message.

        Test scenario:
            1-D, three-column, 3-D, and scalar inputs each fail the
            ``ndim == 2 and shape[1] == 2`` contract.
        """
        with pytest.raises(ValueError, match=r"must be an \(N, 2\) array") as exc:
            _as_xy(bad_input, "boundary_xy")
        assert "boundary_xy" in str(exc.value), (
            f"Error should name the argument for shape {shape_desc}, got: {exc.value}"
        )

    def test_error_message_includes_argument_name(self):
        """Include the caller-supplied name in the error message.

        Test scenario:
            ``_as_xy`` is told the argument name so the raised message
            points at the offending parameter (e.g. ``graticule_lines[2]``).
        """
        with pytest.raises(ValueError, match=r"graticule_lines\[2\]") as exc:
            _as_xy(np.arange(3), "graticule_lines[2]")
        assert "graticule_lines[2]" in str(exc.value), (
            f"Argument name missing from message: {exc.value}"
        )


class TestApplyProjectionFrame:
    """Tests for the public ``apply_projection_frame`` helper."""

    def test_returns_boundary_patch_added_to_axes(self, axes, globe_boundary):
        """Return the boundary patch and add it to the axes.

        Test scenario:
            The return value is a ``PathPatch`` that is also registered in
            ``ax.patches``.
        """
        patch = apply_projection_frame(
            axes, boundary_xy=globe_boundary, xlim=(-1, 1), ylim=(-1, 1)
        )
        assert isinstance(patch, PathPatch), f"Expected PathPatch, got {type(patch)}"
        assert patch in axes.patches, "Boundary patch was not added to ax.patches"

    def test_boundary_facecolor_is_transparent(self, axes, globe_boundary):
        """Default the boundary face to fully transparent.

        Test scenario:
            ``facecolor="none"`` means the patch never hides the data; the
            resolved RGBA alpha is 0.
        """
        patch = apply_projection_frame(
            axes, boundary_xy=globe_boundary, xlim=(-1, 1), ylim=(-1, 1)
        )
        facecolor = patch.get_facecolor()
        assert facecolor[3] == 0.0, f"Expected transparent face, got RGBA {facecolor}"

    def test_default_boundary_and_graticule_style(self, axes, globe_boundary, meridian):
        """Apply the module default styles when no overrides are passed.

        Test scenario:
            The boundary uses ``DEFAULT_BOUNDARY_KW`` (black, 0.8 lw) and the
            graticule uses ``DEFAULT_GRATICULE_KW`` (gray, 0.4 lw).
        """
        patch = apply_projection_frame(
            axes,
            boundary_xy=globe_boundary,
            xlim=(-1, 1),
            ylim=(-1, 1),
            graticule_lines=[meridian],
        )
        assert patch.get_edgecolor()[:3] == pytest.approx((0.0, 0.0, 0.0)), (
            f"Expected black boundary, got {patch.get_edgecolor()}"
        )
        assert patch.get_linewidth() == DEFAULT_BOUNDARY_KW["linewidth"], (
            f"Expected lw {DEFAULT_BOUNDARY_KW['linewidth']}, got {patch.get_linewidth()}"
        )
        line = axes.lines[0]
        assert line.get_linewidth() == DEFAULT_GRATICULE_KW["linewidth"], (
            f"Expected graticule lw {DEFAULT_GRATICULE_KW['linewidth']}, "
            f"got {line.get_linewidth()}"
        )

    def test_equal_aspect_and_limits(self, axes, globe_boundary):
        """Set equal aspect and the requested projected limits.

        Test scenario:
            ``get_aspect() == 1`` and the x/y limits match the inputs.
        """
        apply_projection_frame(
            axes, boundary_xy=globe_boundary, xlim=(-2, 2), ylim=(-3, 3)
        )
        assert axes.get_aspect() == 1, f"Expected aspect 1, got {axes.get_aspect()}"
        assert axes.get_xlim() == (-2, 2), f"Unexpected xlim: {axes.get_xlim()}"
        assert axes.get_ylim() == (-3, 3), f"Unexpected ylim: {axes.get_ylim()}"

    def test_limits_accept_numpy_array(self, axes, globe_boundary):
        """Accept numpy arrays of length 2 as ``xlim``/``ylim``.

        Test scenario:
            A ``(2,)`` numpy array passes the ``len(...) == 2`` check and is
            applied like a tuple.
        """
        apply_projection_frame(
            axes,
            boundary_xy=globe_boundary,
            xlim=np.array([-1.0, 1.0]),
            ylim=np.array([-1.0, 1.0]),
        )
        assert axes.get_xlim() == (-1.0, 1.0), f"Unexpected xlim: {axes.get_xlim()}"

    def test_axis_turned_off(self, axes, globe_boundary):
        """Turn the axis decorations off.

        Test scenario:
            ``ax.axison`` is ``False`` after framing.
        """
        apply_projection_frame(
            axes, boundary_xy=globe_boundary, xlim=(-1, 1), ylim=(-1, 1)
        )
        assert axes.axison is False, "Axis decorations should be turned off"

    def test_image_clipped_to_boundary(self, axes, globe_boundary, rng):
        """Clip a pre-existing image to the boundary path.

        Test scenario:
            An ``imshow`` image drawn before the call gets a clip path whose
            transformed vertex count matches the boundary patch -- i.e. it is
            clipped to the globe, not the default axes bbox.
        """
        image = axes.imshow(rng.random((8, 8)), extent=(-1, 1, -1, 1))
        patch = apply_projection_frame(
            axes, boundary_xy=globe_boundary, xlim=(-1, 1), ylim=(-1, 1)
        )
        clip_path = image.get_clip_path()
        assert clip_path is not None, "Image should have a clip path set"
        verts = clip_path.get_fully_transformed_path().vertices
        expected = len(patch.get_path().vertices)
        assert len(verts) == expected, (
            f"Clip path should follow the boundary ({expected} verts), got {len(verts)}"
        )

    def test_collection_clipped_to_boundary(self, axes, globe_boundary):
        """Clip a pre-existing collection (scatter) to the boundary.

        Test scenario:
            A ``scatter`` collection drawn before the call has a non-None
            clip path afterwards.
        """
        collection = axes.scatter([0.2, -0.3], [0.1, -0.4])
        apply_projection_frame(
            axes, boundary_xy=globe_boundary, xlim=(-1, 1), ylim=(-1, 1)
        )
        assert collection.get_clip_path() is not None, "Collection was not clipped"

    def test_line_clipped_to_boundary(self, axes, globe_boundary):
        """Clip a pre-existing data line to the boundary.

        Test scenario:
            A ``plot`` line drawn before the call has a non-None clip path
            afterwards (``ax.lines`` are clipped too).
        """
        (data_line,) = axes.plot([-0.5, 0.5], [0.0, 0.0])
        apply_projection_frame(
            axes, boundary_xy=globe_boundary, xlim=(-1, 1), ylim=(-1, 1)
        )
        assert data_line.get_clip_path() is not None, "Data line was not clipped"

    def test_graticule_clipped_when_clip_enabled(self, axes, globe_boundary, meridian):
        """Clip the drawn graticule lines to the boundary as well.

        Test scenario:
            With ``clip_artists=True`` the freshly drawn graticule line is
            itself clipped to the globe.
        """
        apply_projection_frame(
            axes,
            boundary_xy=globe_boundary,
            xlim=(-1, 1),
            ylim=(-1, 1),
            graticule_lines=[meridian],
        )
        assert axes.lines[0].get_clip_path() is not None, "Graticule was not clipped"

    def test_clip_artists_false_leaves_data_unclipped(self, axes, globe_boundary, rng):
        """Leave data layers unclipped when ``clip_artists=False``.

        Test scenario:
            The image's clip path is the same object it had before the call
            (the matplotlib default), i.e. untouched.
        """
        image = axes.imshow(rng.random((4, 4)), extent=(-1, 1, -1, 1))
        default_clip = image.get_clip_path()
        apply_projection_frame(
            axes,
            boundary_xy=globe_boundary,
            xlim=(-1, 1),
            ylim=(-1, 1),
            clip_artists=False,
        )
        assert image.get_clip_path() is default_clip, (
            "Image clip path should be untouched when clip_artists=False"
        )

    def test_graticule_lines_drawn(self, axes, globe_boundary):
        """Draw one Line2D per supplied graticule polyline.

        Test scenario:
            Two polylines produce exactly two entries in ``ax.lines``.
        """
        equator = np.column_stack([np.linspace(-1, 1, 50), np.zeros(50)])
        meridian = np.column_stack([np.zeros(50), np.linspace(-1, 1, 50)])
        apply_projection_frame(
            axes,
            boundary_xy=globe_boundary,
            xlim=(-1, 1),
            ylim=(-1, 1),
            graticule_lines=[meridian, equator],
        )
        assert len(axes.lines) == 2, f"Expected 2 graticule lines, got {len(axes.lines)}"

    def test_graticule_default_none_draws_nothing(self, axes, globe_boundary):
        """Draw no graticule when ``graticule_lines`` is omitted.

        Test scenario:
            The default ``None`` leaves ``ax.lines`` empty.
        """
        apply_projection_frame(
            axes, boundary_xy=globe_boundary, xlim=(-1, 1), ylim=(-1, 1)
        )
        assert len(axes.lines) == 0, f"Expected no lines, got {len(axes.lines)}"

    def test_empty_graticule_list_draws_nothing(self, axes, globe_boundary):
        """Treat an empty graticule list the same as ``None``.

        Test scenario:
            ``graticule_lines=[]`` iterates zero times and draws nothing.
        """
        apply_projection_frame(
            axes,
            boundary_xy=globe_boundary,
            xlim=(-1, 1),
            ylim=(-1, 1),
            graticule_lines=[],
        )
        assert len(axes.lines) == 0, f"Expected no lines, got {len(axes.lines)}"

    def test_style_overrides_applied(self, axes, globe_boundary, meridian):
        """Apply caller style overrides over the module defaults.

        Test scenario:
            ``boundary_kw`` and ``graticule_kw`` override edge color, line
            width, and graticule color.
        """
        patch = apply_projection_frame(
            axes,
            boundary_xy=globe_boundary,
            xlim=(-1, 1),
            ylim=(-1, 1),
            graticule_lines=[meridian],
            boundary_kw={"edgecolor": "red", "linewidth": 2.0},
            graticule_kw={"color": "blue"},
        )
        assert patch.get_edgecolor()[:3] == pytest.approx((1.0, 0.0, 0.0)), (
            f"Expected red boundary, got {patch.get_edgecolor()}"
        )
        assert patch.get_linewidth() == 2.0, (
            f"Expected lw 2.0, got {patch.get_linewidth()}"
        )
        assert axes.lines[0].get_color() == "blue", (
            f"Expected blue graticule, got {axes.lines[0].get_color()}"
        )

    def test_does_not_mutate_caller_style_dicts(self, axes, globe_boundary, meridian):
        """Leave the caller-supplied style dicts unmodified.

        Test scenario:
            Merging defaults must not mutate the passed ``boundary_kw`` /
            ``graticule_kw`` dicts (purity of inputs).
        """
        boundary_kw = {"edgecolor": "red"}
        graticule_kw = {"color": "blue"}
        apply_projection_frame(
            axes,
            boundary_xy=globe_boundary,
            xlim=(-1, 1),
            ylim=(-1, 1),
            graticule_lines=[meridian],
            boundary_kw=boundary_kw,
            graticule_kw=graticule_kw,
        )
        assert boundary_kw == {"edgecolor": "red"}, (
            f"boundary_kw was mutated: {boundary_kw}"
        )
        assert graticule_kw == {"color": "blue"}, (
            f"graticule_kw was mutated: {graticule_kw}"
        )

    def test_accepts_plain_list_geometry(self, axes):
        """Coerce plain-list boundary and graticule geometry.

        Test scenario:
            Nested Python lists (not numpy arrays) are accepted for both the
            boundary and the graticule polylines.
        """
        boundary = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        patch = apply_projection_frame(
            axes,
            boundary_xy=boundary,
            xlim=(-1, 1),
            ylim=(-1, 1),
            graticule_lines=[[[0.0, -1.0], [0.0, 1.0]]],
        )
        assert isinstance(patch, PathPatch), f"Expected PathPatch, got {type(patch)}"
        assert len(axes.lines) == 1, f"Expected 1 graticule line, got {len(axes.lines)}"

    def test_open_boundary_still_clips(self, axes, rng):
        """Clip correctly even when the boundary ring is not closed.

        Test scenario:
            A boundary whose last vertex does not repeat the first (an open
            ring) still produces a clip path -- matplotlib closes the polygon
            implicitly for containment.
        """
        open_square = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
        image = axes.imshow(rng.random((4, 4)), extent=(-1, 1, -1, 1))
        apply_projection_frame(
            axes, boundary_xy=open_square, xlim=(-1, 1), ylim=(-1, 1)
        )
        assert image.get_clip_path() is not None, "Open boundary should still clip"

    def test_bad_axes_raises_type_error(self, globe_boundary):
        """Raise ``TypeError`` when ``ax`` is not an Axes-like object.

        Test scenario:
            A bare ``object()`` lacks ``set_xlim``; the helper rejects it.
        """
        with pytest.raises(TypeError, match="matplotlib.axes.Axes") as exc:
            apply_projection_frame(
                object(), boundary_xy=globe_boundary, xlim=(-1, 1), ylim=(-1, 1)
            )
        assert "object" in str(exc.value), (
            f"Error should report the bad type name, got: {exc.value}"
        )

    def test_axes_missing_add_patch_raises_type_error(self, globe_boundary):
        """Raise ``TypeError`` when ``ax`` has ``set_xlim`` but not ``add_patch``.

        Test scenario:
            Exercises the second operand of the duck-type check: an object
            with ``set_xlim`` but no ``add_patch`` is still rejected.
        """

        class _HalfAxes:
            def set_xlim(self, *args, **kwargs):  # pragma: no cover - never called
                pass

        with pytest.raises(TypeError, match="matplotlib.axes.Axes"):
            apply_projection_frame(
                _HalfAxes(), boundary_xy=globe_boundary, xlim=(-1, 1), ylim=(-1, 1)
            )

    def test_bad_boundary_shape_raises_value_error(self, axes):
        """Raise ``ValueError`` for a non ``(N, 2)`` boundary.

        Test scenario:
            A 1-D boundary array fails the shape contract.
        """
        with pytest.raises(ValueError, match=r"boundary_xy must be an \(N, 2\)"):
            apply_projection_frame(
                axes, boundary_xy=np.arange(10), xlim=(-1, 1), ylim=(-1, 1)
            )

    def test_bad_xlim_raises_value_error(self, axes, globe_boundary):
        """Raise ``ValueError`` when ``xlim`` is not a 2-tuple.

        Test scenario:
            A 3-element ``xlim`` trips the first operand of the limit check.
        """
        with pytest.raises(ValueError, match="min, max"):
            apply_projection_frame(
                axes, boundary_xy=globe_boundary, xlim=(-1, 0, 1), ylim=(-1, 1)
            )

    def test_bad_ylim_raises_value_error(self, axes, globe_boundary):
        """Raise ``ValueError`` when ``ylim`` is not a 2-tuple.

        Test scenario:
            A valid ``xlim`` with a 1-element ``ylim`` trips the second
            operand of the limit check.
        """
        with pytest.raises(ValueError, match="min, max"):
            apply_projection_frame(
                axes, boundary_xy=globe_boundary, xlim=(-1, 1), ylim=(1,)
            )

    def test_bad_graticule_shape_raises_value_error(self, axes, globe_boundary):
        """Raise ``ValueError`` for a malformed graticule polyline.

        Test scenario:
            A 1-D graticule entry fails ``_as_xy`` and the error names the
            offending index.
        """
        with pytest.raises(ValueError, match=r"graticule_lines\[0\]"):
            apply_projection_frame(
                axes,
                boundary_xy=globe_boundary,
                xlim=(-1, 1),
                ylim=(-1, 1),
                graticule_lines=[np.arange(6)],
            )

    def test_second_graticule_index_reported_in_error(self, axes, globe_boundary, meridian):
        """Report the correct index when a later graticule line is bad.

        Test scenario:
            The first polyline is valid; the second is malformed, so the
            error names ``graticule_lines[1]``.
        """
        with pytest.raises(ValueError, match=r"graticule_lines\[1\]"):
            apply_projection_frame(
                axes,
                boundary_xy=globe_boundary,
                xlim=(-1, 1),
                ylim=(-1, 1),
                graticule_lines=[meridian, np.arange(6)],
            )
