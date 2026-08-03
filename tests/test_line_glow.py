"""Tests for the line-glow primitive and the glyph ``glow`` options.

Covers ``cleopatra.colors.add_line_glow`` and ``resolve_glow_options`` plus the
opt-in ``glow`` option on ``LineGlyph`` and ``FlowGlyph``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection

from cleopatra.colors import GLOW_OPTION_KEYS, add_line_glow, resolve_glow_options
from cleopatra.flow_glyph import FlowGlyph
from cleopatra.line_glyph import LineGlyph


@pytest.fixture(scope="function")
def line_axes():
    """Provide an axes carrying a single red source line.

    Yields:
        tuple[matplotlib.axes.Axes, matplotlib.lines.Line2D]: The axes and the
        one source line drawn on it.
    """
    fig, ax = plt.subplots()
    (src,) = ax.plot([0, 1, 2], [0, 1, 0], color="red", linewidth=2.0)
    yield ax, src
    plt.close(fig)


class TestAddLineGlow:
    """Tests for ``cleopatra.colors.add_line_glow``."""

    def test_adds_n_glow_copies_per_line(self, line_axes):
        """``add_line_glow`` adds exactly ``n_glow`` copies per source line.

        Test scenario:
            One source line with ``n_glow=6`` yields six glow artists, and the
            axes then holds the source plus its six copies.
        """
        ax, _src = line_axes
        glow = add_line_glow(ax, n_glow=6)
        assert len(glow) == 6, f"expected 6 glow artists, got {len(glow)}"
        assert len(ax.get_lines()) == 7, f"expected 1 source + 6 glow = 7 lines, got {len(ax.get_lines())}"

    def test_copies_grow_in_width_and_sit_beneath(self, line_axes):
        """Glow copies grow by ``linewidth_step`` and sit under the source.

        Test scenario:
            With base width 2.0 and ``linewidth_step=1.5``, copies are at
            3.5/5.0/6.5 points and all have a lower zorder than the source.
        """
        ax, src = line_axes
        glow = add_line_glow(ax, [src], n_glow=3, linewidth_step=1.5)
        widths = [g.get_linewidth() for g in glow]
        assert widths == [3.5, 5.0, 6.5], f"unexpected glow widths: {widths}"
        assert all(g.get_zorder() < src.get_zorder() for g in glow), "glow must sit beneath the source line"

    def test_copies_are_low_alpha_and_excluded_from_legend(self, line_axes):
        """Glow copies use the given per-copy alpha and are legend-excluded.

        Test scenario:
            Each copy carries ``alpha=0.05`` (default) and the ``_nolegend_``
            label so it never appears in a legend.
        """
        ax, _src = line_axes
        glow = add_line_glow(ax)
        assert all(round(float(g.get_alpha()), 3) == 0.05 for g in glow), "every glow copy should use alpha=0.05"
        assert all(g.get_label() == "_nolegend_" for g in glow), "glow copies must be excluded from the legend"

    def test_inherits_each_source_line_colour(self):
        """Glow copies take the colour of their own source line.

        Test scenario:
            Two source lines of different colours each get copies matching their
            own colour (not a single shared colour).
        """
        fig, ax = plt.subplots()
        (red,) = ax.plot([0, 1], [0, 1], color="red")
        (blue,) = ax.plot([0, 1], [1, 0], color="blue")
        glow = add_line_glow(ax, [red, blue], n_glow=2)
        colours = [g.get_color() for g in glow]
        assert colours == ["red", "red", "blue", "blue"], f"glow colours should follow their source: {colours}"
        plt.close(fig)

    def test_defaults_to_all_axes_lines(self, line_axes):
        """With ``lines=None`` the primitive haloes every line on the axes.

        Test scenario:
            Passing no explicit line list defaults to ``ax.get_lines()``.
        """
        ax, _src = line_axes
        glow = add_line_glow(ax, n_glow=3)
        assert len(glow) == 3, f"expected 3 glow copies for the single axes line, got {len(glow)}"


class TestResolveGlowOptions:
    """Tests for ``cleopatra.colors.resolve_glow_options``."""

    def test_true_maps_to_defaults(self):
        """``True`` resolves to an empty kwargs dict (use all defaults).

        Test scenario:
            ``resolve_glow_options(True)`` returns ``{}``.
        """
        assert resolve_glow_options(True) == {}, "True should map to default (empty) kwargs"

    def test_dict_is_passed_through(self):
        """A dict of valid keys is returned as forwardable kwargs.

        Test scenario:
            Every key in ``GLOW_OPTION_KEYS`` is accepted and returned verbatim.
        """
        opts = {key: 1 for key in GLOW_OPTION_KEYS}
        assert resolve_glow_options(opts) == opts, f"valid dict should pass through unchanged: {opts}"

    def test_unknown_key_raises_valueerror(self):
        """An unknown dict key raises ``ValueError`` naming the bad key.

        Test scenario:
            ``{"bogus": 1}`` is rejected with a message mentioning the key.
        """
        with pytest.raises(ValueError, match=r"unknown glow option") as exc_info:
            resolve_glow_options({"bogus": 1})
        assert "bogus" in str(exc_info.value), f"message should name the bad key: {exc_info.value}"

    @pytest.mark.parametrize("bad", [1, "yes", (), None], ids=["int", "str", "tuple", "none"])
    def test_non_bool_non_dict_raises_typeerror(self, bad):
        """A non-bool, non-dict value raises ``TypeError``.

        Args:
            bad: A value that is neither ``True`` nor a dict.

        Test scenario:
            Only ``True`` or a dict are valid; anything else is a type error.
        """
        with pytest.raises(TypeError, match=r"glow must be True or a dict"):
            resolve_glow_options(bad)


class TestLineGlyphGlow:
    """Tests for the ``glow`` option on ``LineGlyph``."""

    def test_glow_true_adds_default_halo(self):
        """``glow=True`` adds six halo copies beneath the plotted line.

        Test scenario:
            A single-series line with ``glow=True`` yields one data line plus
            six glow copies on the axes.
        """
        x = np.linspace(0, 1, 10)
        fig, ax, lines = LineGlyph(x, np.sin(x), glow=True).line()
        extra = len(ax.get_lines()) - len(lines)
        assert extra == 6, f"expected 6 glow lines, got {extra}"
        plt.close(fig)

    def test_glow_dict_overrides_count(self):
        """``glow={'n_glow': 8}`` adds the requested number of copies.

        Test scenario:
            A dict override is forwarded to ``add_line_glow``.
        """
        x = np.linspace(0, 1, 10)
        fig, ax, lines = LineGlyph(x, np.sin(x), glow={"n_glow": 8, "alpha": 0.08}).line()
        extra = len(ax.get_lines()) - len(lines)
        assert extra == 8, f"expected 8 glow lines, got {extra}"
        plt.close(fig)

    def test_glow_off_is_unchanged(self):
        """The default ``glow=False`` adds no extra artists.

        Test scenario:
            Omitting ``glow`` draws only the data line(s).
        """
        x = np.linspace(0, 1, 10)
        fig, ax, lines = LineGlyph(x, np.sin(x)).line()
        assert len(ax.get_lines()) == len(lines), "glow=False must not add artists"
        plt.close(fig)

    def test_bad_glow_key_raises(self):
        """An unknown glow key raises ``ValueError`` at plot time.

        Test scenario:
            ``glow={'bad': 1}`` surfaces a clear error rather than a deep
            ``TypeError``.
        """
        x = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match=r"unknown glow option"):
            LineGlyph(x, np.sin(x), glow={"bad": 1}).line()
        plt.close("all")


class TestFlowGlyphGlow:
    """Tests for the ``glow`` option on ``FlowGlyph``."""

    @pytest.fixture(scope="function")
    def paths(self):
        """Provide two simple flow polylines.

        Returns:
            list[numpy.ndarray]: Two ``(N, 2)`` coordinate arrays.
        """
        return [np.array([[0, 0], [1, 1], [2, 0]], float), np.array([[0, 1], [1, 0]], float)]

    def _line_collections(self, ax):
        """Return the ``LineCollection`` artists on ``ax``."""
        return [c for c in ax.collections if isinstance(c, LineCollection)]

    def test_uncoloured_glow_adds_collections(self, paths):
        """``glow=True`` overlays ``n_glow`` collections on an uncoloured flow.

        Test scenario:
            An uncoloured flow with ``glow=True`` yields the main collection
            plus six glow collections (seven total).
        """
        fig, ax, _lc = FlowGlyph(paths, glow=True).plot()
        assert len(self._line_collections(ax)) == 7, "expected 1 main + 6 glow collections"
        plt.close(fig)

    def test_coloured_glow_keeps_single_mappable(self, paths):
        """The coloured glow keeps only the main collection as the mappable.

        Test scenario:
            With per-path values and ``glow={'n_glow': 4}`` the axes holds five
            collections, and the returned collection is the array-carrying
            mappable (glow copies are not colorbar-registered).
        """
        fig, ax, lc = FlowGlyph(paths, values=np.array([1.0, 2.0]), glow={"n_glow": 4}).plot()
        assert len(self._line_collections(ax)) == 5, "expected 1 main + 4 glow collections"
        assert lc.get_array() is not None, "the returned collection should be the scalar mappable"
        plt.close(fig)

    def test_glow_off_is_unchanged(self, paths):
        """The default ``glow=False`` leaves a single collection.

        Test scenario:
            Omitting ``glow`` draws only the main flow collection.
        """
        fig, ax, _lc = FlowGlyph(paths).plot()
        assert len(self._line_collections(ax)) == 1, "glow=False must not add collections"
        plt.close(fig)
