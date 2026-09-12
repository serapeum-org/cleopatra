"""Regression cover for the round-2 review findings on PR #350.

Each test here corresponds to a defect the second review round found in the
round-1 fixes themselves -- render kwargs bleeding into the set that decides
figure sizing, the `compose=` contract leaking on `animate`, on the extent-less
path and under a styled preset, and the ownership registry evicting an entry
whose only *reporting* artist had gone.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph


@pytest.fixture
def arr():
    """Provide a small deterministic array to render.

    Returns:
        np.ndarray: A 20x30 array of values in [0, 1).
    """
    return np.random.default_rng(0).random((20, 30))


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
