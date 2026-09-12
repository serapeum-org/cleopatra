"""Regression cover for the round-2 review findings on PR #350.

Each test here corresponds to a defect the second review round found in the
round-1 fixes themselves -- the `compose=` contract leaking on `animate`, on the
extent-less path and under a styled preset, render kwargs bleeding into the set
that decides figure sizing, and the ownership registry evicting an entry whose
only *reporting* artist had gone.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cleopatra.glyphs.base.glyph import (
    _entry_is_detached,
    _render_owner_token,
    _render_owner_tokens,
    apply_axis_style,
)
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
from cleopatra.styling.params import DataStyle


@pytest.fixture
def arr():
    """Provide a small deterministic array to render.

    Returns:
        np.ndarray: A 20x30 array of values in [0, 1).
    """
    return np.random.default_rng(0).random((20, 30))


@pytest.fixture
def frames():
    """Provide a small deterministic frame stack to animate.

    Returns:
        np.ndarray: Three 20x30 frames of values in [0, 1).
    """
    return np.random.default_rng(1).random((3, 20, 30))


@pytest.fixture
def host(arr):
    """Provide an axes with a titled, extent-bearing glyph already on it.

    Args:
        arr: The array fixture.

    Yields:
        matplotlib.axes.Axes: The host axes.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ArrayGlyph(arr, title="HOST", extent=[0, 0, 10, 10]).plot(ax=ax)
    yield ax
    plt.close("all")


class TestRenderKwargsDoNotResizeTheFigure:
    """`plot(figsize=...)` keeps the behaviour it had before the axis-option fix."""

    def test_plot_figsize_is_still_auto_computed(self, arr):
        """A `figsize` passed to `plot()` does not switch auto-sizing off.

        Args:
            arr: The array fixture.

        Test scenario:
            Recording render kwargs as explicit made `create_figure_axes` read
            them, which turned `plot(figsize=...)` from ignored into honoured --
            an unannounced rendering change for every existing caller. The two
            sets are now separate.
        """
        fig, _ = ArrayGlyph(arr).plot(figsize=(8, 3))
        assert fig.get_size_inches().tolist() != [8.0, 3.0], (
            "plot(figsize=) must stay auto-sized, as it was before the fix"
        )
        plt.close(fig)

    def test_constructor_figsize_is_still_honoured(self, arr):
        """A `figsize` passed to the constructor is still taken literally.

        Args:
            arr: The array fixture.

        Test scenario:
            The separation must not cost the constructor form, which is the one
            `create_figure_axes` was always meant to read.
        """
        fig, _ = ArrayGlyph(arr, figsize=(8, 3)).plot()
        assert fig.get_size_inches().tolist() == [8.0, 3.0], (
            f"constructor figsize ignored: {fig.get_size_inches()}"
        )
        plt.close(fig)

    def test_a_failed_plot_leaves_no_explicit_keys_behind(self, arr):
        """A `plot()` that raises does not change the next call's output.

        Args:
            arr: The array fixture.

        Test scenario:
            The keys were recorded before the validation loop, so
            `plot(figsize=(9, 2), nope=1)` raised and still left `figsize`
            marked explicit for every later call.
        """
        glyph = ArrayGlyph(arr)
        with pytest.raises(ValueError, match="nope"):
            glyph.plot(figsize=(9, 2), nope=1)
        assert "nope" not in getattr(glyph, "_render_explicit_options", set()), (
            "a rejected key was still recorded as explicit"
        )
        fig, _ = glyph.plot()
        assert fig.get_size_inches().tolist() != [9.0, 2.0], (
            "the failed call's figsize leaked into the next render"
        )
        plt.close(fig)


class TestDroppedOptionsStopBeingApplied:
    """The explicit set is the constructor's keys plus this call's, nothing more."""

    def test_a_later_call_does_not_re_apply_a_dropped_option(self, arr):
        """An option the caller stops passing stops being applied.

        Args:
            arr: The array fixture.

        Test scenario:
            The set only ever grew, so a second `plot(ax=ax)` that passed
            neither option still overwrote the label and grid the caller had
            since set on the axes themselves.
        """
        glyph = ArrayGlyph(arr, extent=[0, 0, 10, 10])
        fig, ax = plt.subplots()
        glyph.plot(ax=ax, xlabel="A", grid_alpha=0.3)
        ax.set_xlabel("USER SET")
        ax.grid(False)

        glyph.plot(ax=ax)
        assert ax.get_xlabel() == "USER SET", (
            f"a dropped option was re-applied: {ax.get_xlabel()!r}"
        )
        assert not any(line.get_visible() for line in ax.xaxis.get_gridlines()), (
            "a dropped grid_alpha re-drew the gridlines"
        )
        plt.close(fig)

    def test_a_constructor_option_survives_every_call(self, arr):
        """A constructor option keeps applying on calls that do not repeat it.

        Args:
            arr: The array fixture.

        Test scenario:
            Rebuilding the set per call must not lose the constructor's keys --
            those stay explicit for the life of the glyph.
        """
        glyph = ArrayGlyph(arr, xlabel="CTOR", extent=[0, 0, 10, 10])
        fig, ax = plt.subplots()
        glyph.plot(ax=ax)
        glyph.plot(ax=ax)
        assert ax.get_xlabel() == "CTOR", (
            f"constructor option lost on a later call: {ax.get_xlabel()!r}"
        )
        plt.close(fig)

    def test_a_render_kwarg_still_reaches_the_axes(self, arr):
        """The round-1 fix still holds: `plot(xlabel=...)` is not dropped.

        Args:
            arr: The array fixture.

        Test scenario:
            This is the defect the accumulation was introduced for; separating
            the two sets must not undo it.
        """
        glyph = ArrayGlyph(arr, extent=[0, 0, 10, 10])
        fig, ax = plt.subplots()
        glyph.plot(ax=ax, xlabel="PLOTX", xtick_font_size=20)
        assert ax.get_xlabel() == "PLOTX", f"render xlabel dropped: {ax.get_xlabel()!r}"
        assert ax.get_xticklabels()[0].get_fontsize() == 20.0, (
            "render xtick_font_size dropped"
        )
        plt.close(fig)


class TestComposeLeavesTheHostChromeAlone:
    """Every line that touches shared axes chrome honours `compose=`."""

    def test_animate_keeps_the_host_title(self, frames, host):
        """An untitled composed animation does not blank the host's title.

        Args:
            frames: The frame-stack fixture.
            host: The pre-rendered host axes.

        Test scenario:
            `animate` set the title unconditionally, so an overlay with no title
            of its own wrote its empty default over the host's caption.
        """
        ArrayGlyph(frames, ax=host, add_colorbar=False).animate(
            list(range(3)), compose=True
        )
        assert host.get_title() == "HOST", (
            f"composed animation blanked the host title: {host.get_title()!r}"
        )

    def test_animate_keeps_the_host_tick_labels(self, frames, host):
        """A composed animation does not strip the host's ticks.

        Args:
            frames: The frame-stack fixture.
            host: The pre-rendered host axes.

        Test scenario:
            `animate` called `set_xticklabels([])` and `set_xticks([])` on the
            host's axes regardless of `compose`.
        """
        ArrayGlyph(frames, ax=host, add_colorbar=False).animate(
            list(range(3)), compose=True
        )
        assert host.get_xticks().size, "composed animation removed the host's x ticks"
        assert host.get_yticks().size, "composed animation removed the host's y ticks"

    def test_animate_without_compose_still_replaces(self, frames, host):
        """The default `animate` still owns the axes outright.

        Args:
            frames: The frame-stack fixture.
            host: The pre-rendered host axes.

        Test scenario:
            Issue #210's replace-by-default contract must survive the guard.
        """
        ArrayGlyph(frames, ax=host, title="OVERLAY", add_colorbar=False).animate(
            list(range(3))
        )
        assert host.get_title() == "OVERLAY", (
            f"a plain animate no longer retitles: {host.get_title()!r}"
        )
        assert not host.get_xticks().size, "a plain animate no longer blanks the ticks"

    def test_an_extentless_overlay_keeps_the_host_ticks(self, arr, host):
        """An overlay built without an `extent` leaves the host's ticks alone.

        Args:
            arr: The array fixture.
            host: The pre-rendered host axes.

        Test scenario:
            The `extent is None and kind == "imshow"` branch blanked the ticks
            unconditionally, so the idiomatic bare `ArrayGlyph(arr)` overlay
            stripped the host's axes. The tick *values* legitimately move -- an
            extent-less overlay draws in pixel coordinates and widens the data
            limits -- so what is pinned here is that ticks and their labels
            still exist at all.
        """
        ArrayGlyph(arr, add_colorbar=False).plot(ax=host, compose=True)
        assert host.get_xticks().size, "extent-less overlay removed the host's x ticks"
        assert host.get_yticks().size, "extent-less overlay removed the host's y ticks"
        assert any(label.get_text() for label in host.get_xticklabels()), (
            "extent-less overlay blanked the host's tick labels"
        )

    def test_an_extentless_solo_render_still_blanks_its_ticks(self, arr):
        """Without `compose` a pixel-space render still hides its ticks.

        Args:
            arr: The array fixture.

        Test scenario:
            Row/column indices are meaningless axis labels; the guard must be
            compose-only, not a change to the solo path.
        """
        fig, ax = ArrayGlyph(arr).plot()
        assert not ax.get_xticks().size, "a solo extent-less render kept its ticks"
        plt.close(fig)

    def test_a_styled_overlay_keeps_the_host_background(self, arr, host):
        """A preset overlay does not repaint the host's axes facecolor.

        Args:
            arr: The array fixture.
            host: The pre-rendered host axes.

        Test scenario:
            `_apply_style_background` ran before the `compose` guard, so a
            dark-canvas preset blackened a light host.
        """
        before = host.get_facecolor()
        ArrayGlyph(arr, add_colorbar=False).plot(
            ax=host, compose=True, data_style=DataStyle(style="temperature_flame")
        )
        assert host.get_facecolor() == before, (
            f"composed preset repainted the host: {host.get_facecolor()}"
        )

    def test_a_solo_styled_render_still_paints_its_background(self, arr):
        """Without `compose` a dark preset still owns the canvas.

        Args:
            arr: The array fixture.

        Test scenario:
            The guard must not cost the preset its background on the path it
            does own.
        """
        fig, ax = plt.subplots()
        ArrayGlyph(arr, add_colorbar=False).plot(
            ax=ax, data_style=DataStyle(style="temperature_flame")
        )
        assert ax.get_facecolor()[:3] == (0.0, 0.0, 0.0), (
            f"the solo styled render lost its background: {ax.get_facecolor()}"
        )
        plt.close(fig)


class TestRegistryProvesDeathBeforeEvicting:
    """An entry is dropped only when every artist in it has actually gone."""

    def test_a_colorbar_outliving_its_image_keeps_the_entry(self, arr):
        """A live colorbar is not evicted because its image was removed.

        Args:
            arr: The array fixture.

        Test scenario:
            Deadness was read off the artists that expose `.axes` and the rest
            were ignored, so an entry of `[Colorbar, AxesImage]` reported
            detached the moment the image went -- dropping the still-live
            colorbar from the registry, where no later render could clear it.
        """
        fig, ax = plt.subplots()
        glyph = ArrayGlyph(arr, extent=[0, 0, 10, 10])
        glyph.plot(ax=ax)
        group = [glyph.cbar, glyph.im]
        group[1].remove()
        assert not _entry_is_detached(group), (
            "a live colorbar was read as a dead entry"
        )
        plt.close(fig)

    def test_an_entry_whose_artists_all_left_is_evicted(self, arr):
        """A genuinely dead entry is still pruned.

        Args:
            arr: The array fixture.

        Test scenario:
            Proving life must not become never proving death -- a throwaway
            glyph's entry has to go, or the registry only grows.
        """
        fig, ax = plt.subplots()
        glyph = ArrayGlyph(arr, extent=[0, 0, 10, 10])
        glyph.plot(ax=ax)
        group = [glyph.cbar, glyph.im]
        group[0].remove()
        group[1].remove()
        assert _entry_is_detached(group), "a dead entry was kept"
        plt.close(fig)

    def test_an_entry_nothing_can_be_asked_about_is_kept(self):
        """An entry of artists that cannot report is not assumed dead.

        Test scenario:
            An empty entry, or one holding only objects that answer neither
            `.axes` nor `.ax` nor indexing, is cheaper to keep than to guess
            about -- guessing wrong orphans live artists.
        """
        assert not _entry_is_detached([]), "an empty entry was read as dead"
        assert not _entry_is_detached([object()]), (
            "an unaskable entry was read as dead"
        )
