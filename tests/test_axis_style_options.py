"""Tests for the shared axis-styling options -- issue #347.

`DEFAULT_OPTIONS` advertises `xlabel`, `ylabel`, their font sizes, the tick label
sizes and `grid_alpha`, and the option validator accepts them on every glyph.
These assert they reach the rendered axes, so an advertised option cannot go
dead again, and that glyphs which do not ask for them are left alone.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cleopatra.glyphs.base.glyph import apply_axis_style
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
from cleopatra.glyphs.primitives.flow_glyph import FlowGlyph
from cleopatra.glyphs.primitives.polygon_glyph import PolygonGlyph
from cleopatra.glyphs.primitives.line_glyph import LineGlyph
from cleopatra.glyphs.gridded.vector_glyph import VectorGlyph
from cleopatra.glyphs.primitives.scatter_glyph import ScatterGlyph
from cleopatra.glyphs.stats.histogram_glyph import HistogramGlyph
from cleopatra.glyphs.stats.kde_glyph import KDEGlyph
from cleopatra.styling.styles import DEFAULT_OPTIONS

#: The options this issue is about: advertised by the validator on every glyph.
AXIS_OPTIONS = (
    "xlabel",
    "ylabel",
    "xlabel_font_size",
    "ylabel_font_size",
    "xtick_font_size",
    "ytick_font_size",
    "grid_alpha",
)

#: Fixed arrays rather than draws from a shared generator, so a case gets the
#: same data whatever else ran first -- a `-k` selection or a different
#: collection order must not change what a test renders.
_RNG = np.random.default_rng(0)
_GRID_X, _GRID_Y = np.meshgrid(np.arange(20), np.arange(15))
_FIELD_U = _RNG.random((15, 20))
_FIELD_V = _RNG.random((15, 20))
_SCALAR = _RNG.random((15, 20))
_NORMAL = _RNG.normal(size=200)
_POINTS_X = _RNG.random(20)
_POINTS_Y = _RNG.random(20)
_KDE_X = _RNG.random(60)
_KDE_Y = _RNG.random(60)

#: A two-cell quad mesh: six nodes in a 3x2 lattice, two faces, one value each.
_MESH_NODE_X = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
_MESH_NODE_Y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
_MESH_FACES = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
_MESH_DATA = np.array([1.0, 2.0])

#: Two open paths for `FlowGlyph` and two closed rings for `PolygonGlyph`.
_PATHS = [
    np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.5]]),
    np.array([[0.0, 2.0], [1.0, 2.5], [2.0, 2.0]]),
]
_POLYGONS = [
    np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    np.array([[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]]),
]

#: One construct/render pair per glyph that advertises the axis options, so the
#: cross-glyph tests stay a table rather than a copy per class.
GLYPH_CASES = {
    "array": (
        lambda **kw: ArrayGlyph(_SCALAR, extent=[0, 0, 10, 10], **kw),
        lambda g: g.plot(),
    ),
    "line": (
        lambda **kw: LineGlyph(np.arange(5.0), np.arange(5.0), **kw),
        lambda g: g.line(),
    ),
    "histogram": (
        lambda **kw: HistogramGlyph(_NORMAL, **kw),
        lambda g: g.histogram(),
    ),
    "vector": (
        lambda **kw: VectorGlyph(
            _GRID_X,
            _GRID_Y,
            _FIELD_U,
            _FIELD_V,
            add_colorbar=False,
            **kw,
        ),
        lambda g: g.plot(kind="quiver"),
    ),
    "scatter": (
        lambda **kw: ScatterGlyph(_POINTS_X, _POINTS_Y, **kw),
        lambda g: g.plot(),
    ),
    "kde": (
        lambda **kw: KDEGlyph(_KDE_X, _KDE_Y, **kw),
        lambda g: g.plot(),
    ),
    "mesh": (
        lambda **kw: MeshGlyph(_MESH_NODE_X, _MESH_NODE_Y, _MESH_FACES, **kw),
        lambda g: g.plot(_MESH_DATA, colorbar=False),
    ),
    "flow": (
        lambda **kw: FlowGlyph(_PATHS, values=np.array([1.0, 2.0]), **kw),
        lambda g: g.plot(add_colorbar=False),
    ),
    "polygon": (
        lambda **kw: PolygonGlyph(_POLYGONS, np.array([1.0, 2.0]), **kw),
        lambda g: g.plot(add_colorbar=False),
    ),
}


@pytest.fixture
def xy():
    """Provide a small x/y pair for the line glyphs.

    Returns:
        tuple: Two 5-element float arrays.
    """
    return np.arange(5.0), np.array([1.0, 3.0, 2.0, 5.0, 4.0])


class TestAdvertisedOptionsAreApplied:
    """Every option the validator advertises changes the rendered axes."""

    @pytest.mark.parametrize("kind", ["line", "bar", "fill_between"])
    def test_line_glyph_render_methods_apply_labels_and_ticks(self, xy, kind):
        """`line`, `bar` and `fill_between` all honour the axis options.

        Args:
            xy: The x/y fixture.
            kind: The render method under test.

        Test scenario:
            Each of the three previously applied only the title, so `xlabel`
            and the tick sizes were accepted and silently dropped.
        """
        x, y = xy
        glyph = LineGlyph(
            x,
            y,
            title="T",
            xlabel="TIME",
            ylabel="VALUE",
            xtick_font_size=20,
            ytick_font_size=18,
        )
        fig, ax, _ = getattr(glyph, kind)()
        assert ax.get_xlabel() == "TIME", f"{kind}: xlabel not applied"
        assert ax.get_ylabel() == "VALUE", f"{kind}: ylabel not applied"
        assert ax.get_xticklabels()[0].get_fontsize() == 20, (
            f"{kind}: xtick size not applied"
        )
        assert ax.get_yticklabels()[0].get_fontsize() == 18, (
            f"{kind}: ytick size not applied"
        )
        plt.close(fig)

    def test_label_font_sizes_are_applied(self, xy):
        """`xlabel_font_size` / `ylabel_font_size` reach the label artists.

        Test scenario:
            The sizes are separate options from the label text, so a caller can
            set the size alone.
        """
        x, y = xy
        fig, ax, _ = LineGlyph(
            x,
            y,
            xlabel="TIME",
            ylabel="VALUE",
            xlabel_font_size=17,
            ylabel_font_size=19,
        ).line()
        assert ax.xaxis.label.get_fontsize() == 17, "xlabel_font_size not applied"
        assert ax.yaxis.label.get_fontsize() == 19, "ylabel_font_size not applied"
        plt.close(fig)

    def test_grid_alpha_draws_a_grid(self, xy):
        """`grid_alpha` draws gridlines at that alpha.

        Test scenario:
            Previously accepted and ignored, so a caller asking for a grid got
            none and no error.
        """
        x, y = xy
        fig, ax, _ = LineGlyph(x, y, grid_alpha=0.5).line()
        lines = ax.get_xgridlines()
        assert lines, "no gridlines drawn"
        assert lines[0].get_alpha() == 0.5, (
            f"grid alpha {lines[0].get_alpha()}, expected 0.5"
        )
        plt.close(fig)

    def test_array_glyph_applies_labels(self):
        """`ArrayGlyph.plot` honours `xlabel` / `ylabel`.

        Test scenario:
            The array path applied the title only, like the line glyphs.
        """
        glyph = ArrayGlyph(
            np.random.default_rng(0).random((10, 12)),
            xlabel="TIME",
            ylabel="VALUE",
            extent=[0, 0, 10, 10],
        )
        fig, ax = glyph.plot()
        assert ax.get_xlabel() == "TIME", "ArrayGlyph xlabel not applied"
        assert ax.get_ylabel() == "VALUE", "ArrayGlyph ylabel not applied"
        plt.close(fig)

    @pytest.mark.parametrize("option", AXIS_OPTIONS)
    def test_every_advertised_option_is_accepted(self, xy, option):
        """Each advertised option is accepted by the constructor.

        Args:
            xy: The x/y fixture.
            option: The option key under test.

        Test scenario:
            The validator names these as supported; passing one must not raise,
            which is the half of the contract that already held.
        """
        x, y = xy
        if option in ("xlabel", "ylabel"):
            value: str | float = "T"
        elif option == "grid_alpha":
            value = 0.5  # matplotlib rejects an alpha outside 0-1
        else:
            value = 12
        fig, ax, _ = LineGlyph(x, y, **{option: value}).line()
        plt.close(fig)


class TestEveryGlyphHonoursTheOptions:
    """The validator advertises these on every glyph, so every glyph applies them."""

    @pytest.mark.parametrize("name", sorted(GLYPH_CASES))
    def test_xlabel_reaches_the_axes(self, name):
        """Each glyph renders the `xlabel` it was constructed with.

        Args:
            name: The glyph under test.

        Test scenario:
            The first fix wired only the line, array and histogram glyphs, while
            the validator advertises the options on all of them -- so the rest
            still accepted `xlabel` and dropped it.
        """
        build, render = GLYPH_CASES[name]
        glyph = build(xlabel="TIME", ylabel="VALUE")
        render(glyph)
        ax = plt.gcf().axes[0]
        assert ax.get_xlabel() == "TIME", f"{name}: xlabel not applied"
        assert ax.get_ylabel() == "VALUE", f"{name}: ylabel not applied"
        plt.close("all")

    @pytest.mark.parametrize("name", sorted(GLYPH_CASES))
    def test_untouched_glyph_is_not_restyled(self, name):
        """A glyph asked for none of them keeps matplotlib's tick size.

        Args:
            name: The glyph under test.

        Test scenario:
            The no-silent-restyle guarantee has to hold for every glyph, not
            only the three the first fix reached.
        """
        build, render = GLYPH_CASES[name]
        render(build())
        ax = plt.gcf().axes[0]
        # HistogramGlyph is the documented exception: it has always rendered the
        # declared defaults (ticks at 11) and opts into them explicitly.
        expected = DEFAULT_OPTIONS["xtick_font_size"] if name == "histogram" else 10.0
        assert ax.get_xticklabels()[0].get_fontsize() == expected, (
            f"{name}: tick size changed without being asked"
        )
        plt.close("all")


class TestUnaskedGlyphsAreUnchanged:
    """A glyph that asks for none of these renders exactly as before."""

    def test_defaults_do_not_restyle_the_axes(self, xy):
        """Not passing the options leaves matplotlib's own defaults in place.

        Test scenario:
            The package declares tick labels at 11 where matplotlib uses 10, so
            applying the declared defaults unconditionally would silently
            restyle every existing figure. Only explicit options are applied.
        """
        x, y = xy
        fig, ax, _ = LineGlyph(x, y, title="T").line()
        assert ax.get_xticklabels()[0].get_fontsize() == 10.0, (
            "tick size changed without being asked"
        )
        assert ax.get_xlabel() == "", "a label appeared without being asked"
        assert not ax.get_xgridlines()[0].get_visible(), (
            "a grid appeared without being asked"
        )
        plt.close(fig)


class TestHistogramGlyphBehaviourPreserved:
    """`HistogramGlyph` already applied these; its output must not move."""

    def test_histogram_keeps_its_declared_defaults(self):
        """The histogram still renders ticks at 11 and a y grid at 0.75.

        Test scenario:
            It is the one glyph that always applied the declared defaults, and
            now shares the helper -- so it opts into `apply_defaults` and its
            rendering is unchanged.
        """
        fig, ax, _ = HistogramGlyph(
            np.random.default_rng(0).normal(size=200)
        ).histogram()
        assert ax.get_xticklabels()[0].get_fontsize() == 11.0, (
            "histogram tick size changed"
        )
        ygrid = ax.get_ygridlines()
        assert ygrid and ygrid[0].get_alpha() == DEFAULT_OPTIONS["grid_alpha"], (
            "histogram y grid changed"
        )
        plt.close(fig)

    def test_histogram_still_honours_explicit_options(self):
        """Explicit options continue to override the defaults there too.

        Test scenario:
            Sharing the helper must not cost the histogram its own overrides.
        """
        fig, ax, _ = HistogramGlyph(
            np.random.default_rng(0).normal(size=200),
            xlabel="VALUES",
            ylabel="COUNT",
            xtick_font_size=20,
        ).histogram()
        assert ax.get_xlabel() == "VALUES"
        assert ax.get_ylabel() == "COUNT"
        assert ax.get_xticklabels()[0].get_fontsize() == 20
        plt.close(fig)


class TestApplyAxisStyle:
    """Direct tests for the shared helper."""

    def test_applies_nothing_when_no_option_was_explicit(self):
        """With no explicit keys and no opt-in, the axes is untouched.

        Test scenario:
            This is what protects existing figures from a silent restyle.
        """
        fig, ax = plt.subplots()
        apply_axis_style(ax, DEFAULT_OPTIONS, set())
        assert ax.get_xlabel() == ""
        assert ax.get_xticklabels()[0].get_fontsize() == 10.0
        plt.close(fig)

    def test_apply_defaults_applies_everything(self):
        """`apply_defaults=True` applies the declared defaults.

        Test scenario:
            The opt-in `HistogramGlyph` uses to keep its historical output.
        """
        fig, ax = plt.subplots()
        apply_axis_style(ax, DEFAULT_OPTIONS, set(), apply_defaults=True)
        assert (
            ax.get_xticklabels()[0].get_fontsize() == DEFAULT_OPTIONS["xtick_font_size"]
        )
        plt.close(fig)

    @pytest.mark.parametrize(
        "grid_axis, getter", [("x", "get_xgridlines"), ("y", "get_ygridlines")]
    )
    def test_grid_axis_selects_which_gridlines(self, grid_axis, getter):
        """`grid_axis` limits the grid to one axis.

        Args:
            grid_axis: The axis to draw gridlines on.
            getter: The accessor for those gridlines.

        Test scenario:
            The histogram draws a y-only grid, so the helper has to express it.
        """
        fig, ax = plt.subplots()
        apply_axis_style(ax, DEFAULT_OPTIONS, {"grid_alpha"}, grid_axis=grid_axis)
        assert getattr(ax, getter)()[0].get_visible(), f"{grid_axis} grid not drawn"
        other = "get_ygridlines" if grid_axis == "x" else "get_xgridlines"
        assert not getattr(ax, other)()[0].get_visible(), (
            "the other axis got a grid too"
        )
        plt.close(fig)

    def test_none_explicit_is_treated_as_empty(self):
        """`explicit=None` is accepted and applies nothing.

        Test scenario:
            The parameter is optional, so a caller with no tracking can omit it.
        """
        fig, ax = plt.subplots()
        apply_axis_style(ax, DEFAULT_OPTIONS, None)
        assert ax.get_xlabel() == ""
        plt.close(fig)
