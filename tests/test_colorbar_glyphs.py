"""Cross-glyph tests for the shared `ColorBar` spec — issue #239.

Every glyph type (not just `ArrayGlyph`) accepts `colorbar=ColorBar(...)` and
emits the loose-`cbar_*` `DeprecationWarning`. These tests parametrize over all
six non-array glyphs so the shared wiring is verified uniformly.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from cleopatra.colorbar import ColorBar
from cleopatra.flow_glyph import FlowGlyph
from cleopatra.kde_glyph import KDEGlyph
from cleopatra.mesh_glyph import MeshGlyph
from cleopatra.polygon_glyph import PolygonGlyph
from cleopatra.scatter_glyph import ScatterGlyph
from cleopatra.vector_glyph import VectorGlyph

_RNG = np.random.default_rng(1337)
_FLOW_PATHS = [np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[0.0, 1.0], [1.0, 0.0]])]
_SQUARES = [
    np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]),
]
_MESH_NX = np.array([0.0, 1.0, 1.0, 0.0])
_MESH_NY = np.array([0.0, 0.0, 1.0, 1.0])
_MESH_FACES = np.array([[0, 1, 2], [0, 2, 3]])
_MESH_DATA = np.array([1.0, 2.0])


def _flow():
    """A value-coloured FlowGlyph."""
    return FlowGlyph(_FLOW_PATHS, values=np.array([3.0, 7.0]))


def _vector():
    """A magnitude-coloured VectorGlyph on a small grid."""
    x, y = np.meshgrid(np.arange(4.0), np.arange(4.0))
    return VectorGlyph(x, y, _RNG.random((4, 4)), _RNG.random((4, 4)))


def _scatter():
    """A value-coloured ScatterGlyph."""
    return ScatterGlyph(list(_RNG.random(12)), list(_RNG.random(12)), values=np.array(_RNG.random(12)))


def _polygon():
    """A value-filled PolygonGlyph of two squares."""
    return PolygonGlyph(_SQUARES, values=np.array([1.0, 2.0]))


def _kde():
    """A KDEGlyph over a random point cloud."""
    return KDEGlyph(_RNG.random(60), _RNG.random(60))


#: name -> (make glyph, draw with a colorbar= spec, read the drawn colorbar,
#: trigger a loose cbar_* kwarg). MeshGlyph stores its bar on `_cbar` and takes
#: loose kwargs at plot time; the others store `cbar` and take them at construction.
GLYPHS = {
    "flow": (_flow, lambda g, cb: g.plot(colorbar=cb), lambda g: g.cbar,
             lambda: FlowGlyph(_FLOW_PATHS, values=np.array([3.0, 7.0]), cbar_label="X")),
    "vector": (_vector, lambda g, cb: g.plot(colorbar=cb), lambda g: g.cbar,
               lambda: VectorGlyph(*np.meshgrid(np.arange(4.0), np.arange(4.0)),
                                   _RNG.random((4, 4)), _RNG.random((4, 4)), cbar_label="X")),
    "scatter": (_scatter, lambda g, cb: g.plot(colorbar=cb), lambda g: g.cbar,
                lambda: ScatterGlyph(list(_RNG.random(6)), list(_RNG.random(6)),
                                     values=np.array(_RNG.random(6)), cbar_label="X")),
    "polygon": (_polygon, lambda g, cb: g.plot(colorbar=cb), lambda g: g.cbar,
                lambda: PolygonGlyph(_SQUARES, values=np.array([1.0, 2.0]), cbar_label="X")),
    "kde": (_kde, lambda g, cb: g.plot(colorbar=cb), lambda g: g.cbar,
            lambda: KDEGlyph(_RNG.random(60), _RNG.random(60), cbar_label="X")),
    "mesh": (lambda: MeshGlyph(_MESH_NX, _MESH_NY, _MESH_FACES),
             lambda g, cb: g.plot(_MESH_DATA, colorbar=cb), lambda g: g._cbar,
             lambda: MeshGlyph(_MESH_NX, _MESH_NY, _MESH_FACES).plot(_MESH_DATA, cbar_label="X")),
}


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    plt.close("all")


@pytest.mark.parametrize("name", list(GLYPHS))
def test_colorbar_spec_configures_and_draws(name):
    """`colorbar=ColorBar(orientation=...)` draws and reaches the render (#239).

    Args:
        name: The glyph type under test.

    Test scenario:
        Every glyph honours the typed spec: it draws a colorbar whose drawn
        orientation matches the spec, not just the loose kwargs.
    """
    make, draw, read_cbar, _ = GLYPHS[name]
    glyph = make()
    draw(glyph, ColorBar(orientation="horizontal", label="Value"))
    cbar = read_cbar(glyph)
    assert cbar is not None, f"{name}: colorbar=ColorBar should draw a colorbar"
    assert cbar.orientation == "horizontal", f"{name}: spec orientation not applied, got {cbar.orientation}"


@pytest.mark.parametrize("name", list(GLYPHS))
def test_colorbar_false_suppresses(name):
    """`colorbar=False` suppresses the colorbar on every glyph (#239).

    Args:
        name: The glyph type under test.

    Test scenario:
        The shared spec's `False` turns the bar off regardless of the glyph's
        own default.
    """
    make, draw, read_cbar, _ = GLYPHS[name]
    glyph = make()
    draw(glyph, False)
    assert read_cbar(glyph) is None, f"{name}: colorbar=False should suppress the colorbar"


@pytest.mark.parametrize("name", list(GLYPHS))
def test_loose_cbar_kwarg_is_deprecated(name):
    """A loose `cbar_label` warns on every glyph, steering to `ColorBar` (#239).

    Args:
        name: The glyph type under test.

    Test scenario:
        The deprecation fires wherever the loose kwarg lands (construction for
        the add_colorbar glyphs, plot for MeshGlyph).
    """
    _, _, _, loose = GLYPHS[name]
    with pytest.warns(DeprecationWarning, match=r"cbar_label.*ColorBar\(label="):
        loose()


@pytest.mark.parametrize("name", list(GLYPHS))
def test_colorbar_spec_ticks_spacing_reaches_render(name):
    """`colorbar=ColorBar(ticks_spacing=...)` is honored on every glyph (#239).

    Args:
        name: The glyph type under test.

    Test scenario:
        A spec-provided tick spacing survives to the resolved options rather
        than being overwritten by a glyph's auto-computed value -- the MeshGlyph
        regression (H1) this guards against.
    """
    make, draw, _, _ = GLYPHS[name]
    glyph = make()
    draw(glyph, ColorBar(ticks_spacing=2.5))
    got = glyph.default_options["ticks_spacing"]
    assert got == 2.5, f"{name}: spec ticks_spacing not honored, got {got}"


def test_mesh_animate_colorbar_spec_and_suppression():
    """`MeshGlyph.animate(colorbar=...)` honors the spec and suppresses on False (#239).

    Test scenario:
        The animate path -- covered only for plot elsewhere -- must apply a
        ColorBar spec (orientation + ticks_spacing) and drop the bar on False.
    """
    frames = [_MESH_DATA, _MESH_DATA * 2]
    g = MeshGlyph(_MESH_NX, _MESH_NY, _MESH_FACES)
    g.animate(frames, time=[0, 1], colorbar=ColorBar(orientation="horizontal", ticks_spacing=2.5))
    assert g._cbar is not None and g._cbar.orientation == "horizontal", "animate spec orientation not applied"
    assert g.default_options["ticks_spacing"] == 2.5, "animate spec ticks_spacing not honored"
    g2 = MeshGlyph(_MESH_NX, _MESH_NY, _MESH_FACES)
    g2.animate(frames, time=[0, 1], colorbar=False)
    assert getattr(g2, "_cbar", None) is None, "animate colorbar=False should suppress the bar"


def test_unvalidated_label_location_errors_clearly_at_render():
    """A `label_location` invalid for the default vertical bar errors clearly (#241/L1).

    Test scenario:
        `ColorBar(label_location="left")` pins no orientation (so construction
        validation is skipped), but the default bar is vertical -- the render
        raises a clear cleopatra error rather than a raw matplotlib one.
    """
    glyph = _scatter()
    with pytest.raises(ValueError, match="not valid for a vertical colorbar"):
        glyph.plot(colorbar=ColorBar(label_location="left"))


def test_colorbar_spec_is_sticky_like_arrayglyph():
    """The `colorbar=` spec is sticky on the add_colorbar glyphs, like ArrayGlyph (#239).

    Test scenario:
        A spec applied once persists into a later plain `plot()` (options are
        sticky by design), and `colorbar=True` resets it to a default bar --
        matching ArrayGlyph's contract, not per-call behaviour.
    """
    glyph = _scatter()
    glyph.plot(colorbar=ColorBar(orientation="horizontal"))
    assert glyph.cbar.orientation == "horizontal", "spec should apply"
    glyph.plot()
    assert glyph.cbar.orientation == "horizontal", "spec should persist across plain plot() calls"
    glyph.plot(colorbar=True)
    assert glyph.cbar.orientation == "vertical", "colorbar=True should reset to the default bar"
