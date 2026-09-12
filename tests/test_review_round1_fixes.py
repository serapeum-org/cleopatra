"""Regression cover for the round-1 review findings on PR #350.

Each test here corresponds to a defect the first review round found in the three
issue fixes themselves -- the headline `#347` fix reaching only the constructor,
a crash on a non-numeric `title_size`, gridlines appearing where they never had,
`compose=` ignored on the styled render path, and live artists being evicted
from the ownership registry.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import copy

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cleopatra.glyphs.base.glyph import (
    _entry_is_detached,
    _mark_render_artists,
    _render_owner_token,
    apply_axis_style,
    multiline_title_pad,
)
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
from cleopatra.glyphs.gridded.vector_glyph import VectorGlyph
from cleopatra.glyphs.stats.histogram_glyph import HistogramGlyph
from cleopatra.styling.params import DataStyle
from cleopatra.styling.styles import DEFAULT_OPTIONS


@pytest.fixture
def arr():
    """Provide a small array to render.

    Returns:
        np.ndarray: A 20x30 array.
    """
    return np.random.default_rng(0).random((20, 30))


@pytest.fixture
def values():
    """Provide a sample for the histogram glyphs.

    Returns:
        np.ndarray: 200 normal deviates.
    """
    return np.random.default_rng(0).normal(size=200)


class TestAxisOptionsThroughPlot:
    """H1 -- the axis options must reach the axes via `plot(**kwargs)` too."""

    def test_plot_kwargs_are_treated_as_explicit(self, arr):
        """`plot(xlabel=...)` applies, like the constructor form.

        Test scenario:
            `_apply_axis_style` only applies options the caller asked for, and
            `plot`'s kwargs went into `default_options` without ever being
            recorded as explicit -- so the headline fix worked only through the
            constructor, which is not how most callers pass these.
        """
        fig, ax = ArrayGlyph(arr, extent=[0, 0, 10, 10]).plot(
            xlabel="TIME", ylabel="VALUE", xtick_font_size=22
        )
        assert ax.get_xlabel() == "TIME", "plot(xlabel=) was dropped"
        assert ax.get_ylabel() == "VALUE", "plot(ylabel=) was dropped"
        assert ax.get_xticklabels()[0].get_fontsize() == 22
        plt.close(fig)

    def test_plot_without_options_still_does_not_restyle(self, arr):
        """Recording plot kwargs must not make the defaults apply.

        Test scenario:
            The guard against a package-wide restyle has to survive the fix --
            an unasked-for render keeps matplotlib's own tick size.
        """
        fig, ax = ArrayGlyph(arr, extent=[0, 0, 10, 10]).plot()
        assert ax.get_xlabel() == ""
        assert ax.get_xticklabels()[0].get_fontsize() == 10.0
        plt.close(fig)


class TestNonNumericTitleSize:
    """H2 -- `title_size` accepts everything `set_title` does."""

    @pytest.mark.parametrize("size", ["large", None, 15, 26, "xx-small"])
    def test_multi_line_title_renders_for_any_size_form(self, arr, size):
        """A multi-line title renders whatever form `title_size` takes.

        Args:
            arr: The array fixture.
            size: The title size form under test.

        Test scenario:
            The pad arithmetic multiplied the size by a line count, so a
            relative name or `None` raised `TypeError` -- on a path that worked
            before this branch.
        """
        fig, ax = ArrayGlyph(
            arr,
            extent=[0, 0, 10, 10],
            title="first\nsecond",
            title_size=size,
            figsize=(8, 6),
        ).plot()
        assert ax.get_title() == "first\nsecond"
        plt.close(fig)

    @pytest.mark.parametrize("size", ["large", None])
    def test_pad_resolves_relative_sizes_to_points(self, size):
        """The pad is a number for a relative or absent size.

        Args:
            size: The title size form under test.

        Test scenario:
            `None` resolves through `axes.titlesize`, not the font manager's own
            smaller default.
        """
        fig, ax = plt.subplots()
        ax.matshow(np.zeros((4, 4)))
        pad = multiline_title_pad(ax, "a\nb", size)
        assert isinstance(pad, float) and pad > 0, f"pad was {pad!r}"
        plt.close(fig)


class TestHistogramGridUnchanged:
    """H3 -- the histogram family's gridlines must match what they always drew."""

    @pytest.mark.parametrize(
        "method, x_grid, y_grid",
        [
            ("histogram", False, True),
            ("boxplot", False, True),
            ("stripes", False, False),
        ],
    )
    def test_grid_matches_historical_output(self, values, method, x_grid, y_grid):
        """Each method keeps exactly the gridlines it drew before.

        Args:
            values: The sample fixture.
            method: The render method under test.
            x_grid: Whether x gridlines are expected.
            y_grid: Whether y gridlines are expected.

        Test scenario:
            Routing the shared helper through `_apply_axis_labels` brought its
            `grid_axis="both"` default with it, adding x gridlines to three
            methods that never had them -- and a y grid to `stripes`, which drew
            none at all.
        """
        glyph = HistogramGlyph(values)
        getattr(glyph, method)()
        ax = plt.gcf().axes[0]
        assert ax.xaxis._major_tick_kw.get("gridOn", False) is x_grid, (
            f"{method}: unexpected x grid"
        )
        assert ax.yaxis._major_tick_kw.get("gridOn", False) is y_grid, (
            f"{method}: unexpected y grid"
        )
        plt.close("all")


class TestComposeOnStyledRender:
    """H4 -- `compose=` must be honoured on the preset/style render path too."""

    def test_styled_overlay_composes(self, arr):
        """A `data_style=` overlay with `compose=True` keeps the host layer.

        Test scenario:
            `plot` forwarded `compose` to its own clear calls but not to
            `_plot_with_style`, so asking for composition on a styled render
            silently wiped the host.
        """
        fig, ax = plt.subplots()
        ArrayGlyph(arr).plot(ax=ax)
        ArrayGlyph(arr).plot(
            ax=ax, compose=True, data_style=DataStyle(style="topography")
        )
        assert len(ax.images) == 2, "the styled overlay cleared the host layer"
        plt.close(fig)

    def test_styled_overlay_still_replaces_by_default(self, arr):
        """Without `compose` the styled path replaces, as before.

        Test scenario:
            The issue #210 contract applies to this path as much as the others.
        """
        fig, ax = plt.subplots()
        ArrayGlyph(arr).plot(ax=ax)
        ArrayGlyph(arr).plot(ax=ax, data_style=DataStyle(style="topography"))
        assert len(ax.images) == 1, "the styled path stopped replacing"
        plt.close(fig)


class TestRegistryKeepsLiveEntries:
    """H5 -- an owner whose artists cannot report an axes is never evicted."""

    def test_container_owning_glyph_survives_another_render(self, values):
        """A histogram's entry survives another glyph marking on the axes.

        Test scenario:
            `ax.hist` returns a `BarContainer`, which exposes no `.axes`. The
            prune read a missing attribute as "detached" and dropped the live
            entry, orphaning its artists -- the exact defect the tracking
            prevents.
        """

        class _Other:
            pass

        HistogramGlyph(values).histogram()
        ax = plt.gcf().axes[0]
        assert len(ax._cleo_render_artists) == 1, "precondition: histogram recorded"

        _mark_render_artists(ax, _Other(), ax.plot([0, 1], [0, 1])[0])
        assert len(ax._cleo_render_artists) == 2, "the live histogram entry was evicted"
        plt.close("all")

    def test_detached_entry_is_still_pruned(self):
        """An entry whose artists really have left the axes is dropped.

        Test scenario:
            Pruning must still work for the throwaway-glyph case it exists for.
        """
        fig, ax = plt.subplots()
        line = ax.plot([0, 1], [0, 1])[0]
        line.remove()
        assert _entry_is_detached([line]) is True
        plt.close(fig)

    def test_entry_that_cannot_report_is_kept(self):
        """An entry of artists with no `.axes` is treated as live.

        Test scenario:
            Deadness must be proven, not inferred from a missing attribute.
        """

        class _NoAxes:
            pass

        assert _entry_is_detached([_NoAxes()]) is False
        assert _entry_is_detached([]) is False, "an empty entry is not proof of death"


class TestLabelTextPreserved:
    """M1 -- resizing a label must not blank one the caller set."""

    def test_font_size_alone_keeps_the_caller_label(self):
        """Passing only `xlabel_font_size` resizes without retitling.

        Test scenario:
            The helper re-set the label text alongside the size, so a caller who
            had labelled the axes themselves and only wanted a bigger font got a
            blank label instead.
        """
        fig, ax = plt.subplots()
        ax.set_xlabel("CALLER SET THIS")
        apply_axis_style(ax, DEFAULT_OPTIONS, {"xlabel_font_size"})
        assert ax.get_xlabel() == "CALLER SET THIS", (
            "the caller's label was overwritten"
        )
        assert ax.xaxis.label.get_fontsize() == DEFAULT_OPTIONS["xlabel_font_size"]
        plt.close(fig)

    def test_explicit_label_still_wins(self):
        """An explicitly passed label replaces whatever was there.

        Test scenario:
            Preserving the caller's text must not stop the glyph's own label
            from applying when it was actually asked for.
        """
        fig, ax = plt.subplots()
        ax.set_xlabel("CALLER SET THIS")
        apply_axis_style(ax, dict(DEFAULT_OPTIONS, xlabel="MINE"), {"xlabel"})
        assert ax.get_xlabel() == "MINE"
        plt.close(fig)


class TestThinValidationOrder:
    """M4/L2 -- `thin` is checked before drawing, and flagged where it is inert."""

    def test_invalid_thin_leaves_the_axes_untouched(self):
        """A bad `thin` raises without having cleared the host first.

        Test scenario:
            Validation sat inside the subsampling helper, which runs after the
            clear -- so an invalid value wiped the axes and then raised, leaving
            the caller with neither their old figure nor a new one.
        """
        x, y = np.meshgrid(np.arange(30), np.arange(20))
        rng = np.random.default_rng(0)
        fig, ax = plt.subplots()
        ArrayGlyph(rng.random((20, 30))).plot(ax=ax)
        with pytest.raises(ValueError, match="thin must be a positive integer"):
            VectorGlyph(
                x,
                y,
                rng.random((20, 30)),
                rng.random((20, 30)),
                thin=0,
                add_colorbar=False,
            ).plot(kind="quiver", ax=ax)
        assert len(ax.images) == 1, "the host layer was cleared before the error"
        plt.close(fig)

    def test_thin_on_streamplot_warns(self):
        """`thin` with `streamplot` says it does nothing rather than ignoring it.

        Test scenario:
            Streamplot seeds its own lines, so there is no per-grid-point arrow
            to drop. Silently accepting the option left a caller believing a
            dense figure had been thinned.
        """
        x, y = np.meshgrid(np.arange(30), np.arange(20))
        rng = np.random.default_rng(0)
        fig, ax = plt.subplots()
        with pytest.warns(UserWarning, match="no effect on kind='streamplot'"):
            VectorGlyph(
                x,
                y,
                rng.random((20, 30)),
                rng.random((20, 30)),
                thin=5,
                add_colorbar=False,
            ).plot(kind="streamplot", ax=ax)
        plt.close(fig)


class TestOwnerTokenIdentity:
    """M5 -- a copied glyph is a different owner."""

    def test_deepcopy_is_a_distinct_owner(self):
        """A `deepcopy` of a glyph does not inherit its ownership token.

        Test scenario:
            The token was stamped on the glyph as an attribute, so it travelled
            with a clone -- and the clone would then clear the original's
            artists, which is the collision the token exists to prevent.
        """
        glyph = ArrayGlyph(np.random.default_rng(0).random((20, 30)))
        original = _render_owner_token(glyph)
        clone = copy.deepcopy(glyph)
        assert _render_owner_token(clone) != original, "the clone reused the token"

    def test_same_glyph_keeps_its_token(self):
        """Repeated lookups for one glyph return the same token.

        Test scenario:
            Stability is what lets a glyph replace its own artists across calls.
        """
        glyph = ArrayGlyph(np.random.default_rng(0).random((20, 30)))
        assert _render_owner_token(glyph) == _render_owner_token(glyph)
