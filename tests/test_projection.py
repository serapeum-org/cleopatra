"""Tests for cleopatra.projection.

Covers `apply_projection_frame`, the stateless helper that turns a plain
axes into a static projected ('globe') frame. Pure matplotlib -- no PROJ
dependency -- so the geometry is built inline as numpy arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.plot

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

from cleopatra.projection import apply_projection_frame  # noqa: E402


@pytest.fixture
def globe_boundary() -> np.ndarray:
    """Unit-circle boundary, the canonical orthographic-globe outline."""
    theta = np.linspace(0, 2 * np.pi, 200)
    return np.column_stack([np.cos(theta), np.sin(theta)])


@pytest.fixture
def axes():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_returns_boundary_patch_added_to_axes(axes, globe_boundary):
    patch = apply_projection_frame(
        axes, boundary_xy=globe_boundary, xlim=(-1, 1), ylim=(-1, 1)
    )
    assert isinstance(patch, PathPatch)
    assert patch in axes.patches


def test_equal_aspect_and_limits(axes, globe_boundary):
    apply_projection_frame(
        axes, boundary_xy=globe_boundary, xlim=(-2, 2), ylim=(-3, 3)
    )
    assert axes.get_aspect() == 1
    assert axes.get_xlim() == (-2, 2)
    assert axes.get_ylim() == (-3, 3)


def test_axis_turned_off(axes, globe_boundary):
    apply_projection_frame(
        axes, boundary_xy=globe_boundary, xlim=(-1, 1), ylim=(-1, 1)
    )
    assert axes.axison is False


def test_data_image_clipped_to_boundary(axes, globe_boundary):
    image = axes.imshow(np.random.rand(8, 8), extent=(-1, 1, -1, 1))
    patch = apply_projection_frame(
        axes, boundary_xy=globe_boundary, xlim=(-1, 1), ylim=(-1, 1)
    )
    clip_path = image.get_clip_path()
    assert clip_path is not None
    # The clip path is the boundary patch we returned (matplotlib wraps it
    # in a TransformedPatchPath); the vertex count proves it is the
    # boundary circle, not the default axes bbox.
    verts = clip_path.get_fully_transformed_path().vertices
    assert len(verts) == len(patch.get_path().vertices)


def test_collection_clipped_to_boundary(axes, globe_boundary):
    collection = axes.scatter([0.2, -0.3], [0.1, -0.4])
    apply_projection_frame(
        axes, boundary_xy=globe_boundary, xlim=(-1, 1), ylim=(-1, 1)
    )
    assert collection.get_clip_path() is not None


def test_clip_artists_false_leaves_data_unclipped(axes, globe_boundary):
    image = axes.imshow(np.random.rand(4, 4), extent=(-1, 1, -1, 1))
    default_clip = image.get_clip_path()
    apply_projection_frame(
        axes,
        boundary_xy=globe_boundary,
        xlim=(-1, 1),
        ylim=(-1, 1),
        clip_artists=False,
    )
    # Unchanged from the matplotlib default (the axes bbox / None).
    assert image.get_clip_path() is default_clip


def test_graticule_lines_drawn(axes, globe_boundary):
    meridian = np.column_stack([np.zeros(50), np.linspace(-1, 1, 50)])
    equator = np.column_stack([np.linspace(-1, 1, 50), np.zeros(50)])
    apply_projection_frame(
        axes,
        boundary_xy=globe_boundary,
        xlim=(-1, 1),
        ylim=(-1, 1),
        graticule_lines=[meridian, equator],
    )
    assert len(axes.lines) == 2


def test_graticule_default_none_draws_nothing(axes, globe_boundary):
    apply_projection_frame(
        axes, boundary_xy=globe_boundary, xlim=(-1, 1), ylim=(-1, 1)
    )
    assert len(axes.lines) == 0


def test_style_overrides_applied(axes, globe_boundary):
    meridian = np.column_stack([np.zeros(10), np.linspace(-1, 1, 10)])
    patch = apply_projection_frame(
        axes,
        boundary_xy=globe_boundary,
        xlim=(-1, 1),
        ylim=(-1, 1),
        graticule_lines=[meridian],
        boundary_kw={"edgecolor": "red", "linewidth": 2.0},
        graticule_kw={"color": "blue"},
    )
    assert patch.get_edgecolor()[:3] == pytest.approx((1.0, 0.0, 0.0))
    assert patch.get_linewidth() == 2.0
    assert axes.lines[0].get_color() == "blue"


def test_accepts_list_input(axes):
    # Plain Python lists, not numpy arrays, must be coerced.
    boundary = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    patch = apply_projection_frame(
        axes,
        boundary_xy=boundary,
        xlim=(-1, 1),
        ylim=(-1, 1),
        graticule_lines=[[[0.0, -1.0], [0.0, 1.0]]],
    )
    assert isinstance(patch, PathPatch)
    assert len(axes.lines) == 1


def test_bad_axes_raises_type_error(globe_boundary):
    with pytest.raises(TypeError, match="matplotlib.axes.Axes"):
        apply_projection_frame(
            object(), boundary_xy=globe_boundary, xlim=(-1, 1), ylim=(-1, 1)
        )


def test_bad_boundary_shape_raises_value_error(axes):
    with pytest.raises(ValueError, match=r"boundary_xy must be an \(N, 2\)"):
        apply_projection_frame(
            axes, boundary_xy=np.arange(10), xlim=(-1, 1), ylim=(-1, 1)
        )


def test_bad_limits_raise_value_error(axes, globe_boundary):
    with pytest.raises(ValueError, match="min, max"):
        apply_projection_frame(
            axes, boundary_xy=globe_boundary, xlim=(-1, 0, 1), ylim=(-1, 1)
        )


def test_bad_graticule_shape_raises_value_error(axes, globe_boundary):
    with pytest.raises(ValueError, match=r"graticule_lines\[0\]"):
        apply_projection_frame(
            axes,
            boundary_xy=globe_boundary,
            xlim=(-1, 1),
            ylim=(-1, 1),
            graticule_lines=[np.arange(6)],
        )
