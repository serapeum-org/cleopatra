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

from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
from cleopatra.glyphs.gridded.vector_glyph import VectorGlyph
from cleopatra.glyphs.primitives.flow_glyph import FlowGlyph
from cleopatra.glyphs.primitives.polygon_glyph import PolygonGlyph
from cleopatra.glyphs.primitives.scatter_glyph import ScatterGlyph
from cleopatra.glyphs.stats.kde_glyph import KDEGlyph
from cleopatra.styling.colorbar import ColorBar
from cleopatra.styling.params import Classify
from cleopatra.styling.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS

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
    assert g._cbar is not None, "animate spec should draw a colorbar"
    assert g._cbar.orientation == "horizontal", "animate spec orientation not applied"
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
    spec = ColorBar(label_location="left")  # constructs OK (pins no orientation)
    with pytest.raises(ValueError, match="not valid for a vertical colorbar"):
        glyph.plot(colorbar=spec)


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


def test_colorbar_placement_renders_on_non_array_glyph():
    """A ColorBar's location/inside/box reach the render on a non-array glyph (#239).

    Test scenario:
        Placement keys absent from the glyph's own DEFAULT_OPTIONS still flow
        through `_resolve_colorbar` into the shared `create_color_bar` -- an
        inset bottom bar draws horizontally.
    """
    glyph = _scatter()
    glyph.plot(colorbar=ColorBar(location="bottom", inside=True, box=True))
    assert glyph.cbar is not None, "placement spec should draw a colorbar"
    assert glyph.cbar.orientation == "horizontal", "location='bottom' should force horizontal"


@pytest.mark.parametrize("name", list(GLYPHS))
def test_colorbar_none_draws_default(name):
    """Explicit `colorbar=None` draws the glyph's default bar (#239).

    Args:
        name: The glyph type under test.

    Test scenario:
        `None` is the draw-default path (distinct from `False`); every glyph
        still renders a colorbar.
    """
    make, draw, read_cbar, _ = GLYPHS[name]
    glyph = make()
    draw(glyph, None)
    assert read_cbar(glyph) is not None, f"{name}: colorbar=None should draw the default bar"


def test_colorbar_false_suppresses_under_categorical_scheme():
    """`colorbar=False` suppresses the bar even under a categorical scheme (#239).

    Test scenario:
        A categorical ScatterGlyph draws a disjoint legend instead of a colorbar;
        `colorbar=False` must still leave `cbar` unset.
    """
    glyph = ScatterGlyph(
        list(_RNG.random(9)), list(_RNG.random(9)),
        values=np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
    )
    glyph.plot(colorbar=False, classify=Classify(scheme="categorical"))
    assert glyph.cbar is None, "colorbar=False should suppress the bar under a categorical scheme"


class TestColorBarMethods:
    """Direct unit tests for `ColorBar`'s own option-building methods."""

    def test_to_options_maps_placement_and_omits_unset(self):
        """`to_options` emits placement keys but omits unset caption fields.

        Test scenario:
            A spec with only placement set yields the `cbar_*` placement keys
            plus `add_colorbar=True`, and no `cbar_label` (it was never set).
        """
        opts = ColorBar(location="left", inside=True).to_options()
        assert opts["cbar_location"] == "left", f"location not mapped: {opts}"
        assert opts["cbar_inside"] is True, f"inside not mapped: {opts}"
        assert opts["add_colorbar"] is True, f"add_colorbar missing: {opts}"
        assert "cbar_label" not in opts, f"unset caption should be omitted: {opts}"

    def test_to_options_emits_set_caption_and_tick_fields(self):
        """Set caption / sizing / tick fields map onto their `cbar_*` keys.

        Test scenario:
            `label`, `length`, and `ticks_spacing` set on the spec appear in the
            emitted dict.
        """
        opts = ColorBar(label="Depth", length=0.8, ticks_spacing=2.0).to_options()
        assert opts["cbar_label"] == "Depth", f"label not mapped: {opts}"
        assert opts["cbar_length"] == 0.8, f"length not mapped: {opts}"
        assert opts["ticks_spacing"] == 2.0, f"ticks_spacing not mapped: {opts}"

    def test_resolve_none_returns_empty(self):
        """`resolve(None)` leaves options untouched (empty dict).

        Test scenario:
            `None` is the "keep current" case, so no keys are emitted.
        """
        assert ColorBar.resolve(None) == {}, "None should resolve to an empty dict"

    def test_resolve_false_suppresses(self):
        """`resolve(False)` emits only `add_colorbar=False`.

        Test scenario:
            `False` suppresses the colorbar and nothing else.
        """
        assert ColorBar.resolve(False) == {"add_colorbar": False}, "False should suppress"

    def test_resolve_true_matches_reset_options(self):
        """`resolve(True)` returns the `reset_options` default dict.

        Test scenario:
            `True` and `reset_options()` must agree (both draw the default bar).
        """
        assert ColorBar.resolve(True) == ColorBar.reset_options(), (
            "resolve(True) should equal reset_options()"
        )

    def test_resolve_instance_delegates_to_to_options(self):
        """`resolve(spec)` delegates to the instance's `to_options`.

        Test scenario:
            A `ColorBar` instance resolves to exactly its `to_options()` dict.
        """
        cb = ColorBar(location="bottom")
        assert ColorBar.resolve(cb) == cb.to_options(), "instance should delegate to to_options"

    def test_resolve_invalid_type_raises(self):
        """`resolve` rejects a non-bool / non-`ColorBar` / non-`None` argument.

        Test scenario:
            An int argument raises `TypeError` naming the accepted types.
        """
        with pytest.raises(TypeError, match="bool, ColorBar, or None"):
            ColorBar.resolve(123)

    def test_reset_options_resets_family_to_defaults(self):
        """`reset_options` clears the resettable `cbar_*` family to defaults.

        Test scenario:
            Placement keys go to `None`/`False`, and the caption/orientation keys
            take their `STYLE_DEFAULTS` values.
        """
        opts = ColorBar.reset_options()
        assert opts["add_colorbar"] is True, f"add_colorbar missing: {opts}"
        assert opts["cbar_location"] is None, f"location not reset: {opts}"
        assert opts["cbar_inside"] is False, f"inside not reset: {opts}"
        assert opts["cbar_orientation"] == STYLE_DEFAULTS["cbar_orientation"], (
            f"orientation not from defaults: {opts}"
        )
        assert opts["cbar_label"] == STYLE_DEFAULTS["cbar_label"], (
            f"label not from defaults: {opts}"
        )

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"location": "left"}, True),
            ({"inside": True}, True),
            ({"orientation": "horizontal"}, True),
            ({}, False),
            ({"label": "x"}, False),
        ],
    )
    def test_specifies_placement(self, kwargs, expected):
        """`specifies_placement` is true iff location/inside/orientation is set.

        Args:
            kwargs: `ColorBar` constructor arguments for the case.
            expected: Whether the spec should count as requesting placement.

        Test scenario:
            Any of `location`/`inside`/`orientation` yields True; a bare spec or
            a caption-only spec yields False.
        """
        got = ColorBar(**kwargs).specifies_placement()
        assert got is expected, f"specifies_placement({kwargs}) -> {got}, expected {expected}"
