"""Tests for the default gap between a plot frame and its outside colorbar."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph

pytestmark = pytest.mark.plot


def _gap(glyph) -> float:
    """Horizontal gap (figure fraction) between the axes' right edge and the colorbar."""
    return glyph.cbar.ax.get_position().x0 - glyph.ax.get_position().x1


class TestColorbarPad:
    """The outside colorbar sits close to the frame by default, but stays tunable."""

    def test_default_gap_is_tight(self):
        """The default colorbar sits close to the plot frame.

        Test scenario:
            A wide field (whose equal-aspect axes is wide, so matplotlib's default
            pad would be a large absolute gap) keeps the colorbar within a small
            fraction of the figure width of the frame.
        """
        glyph = ArrayGlyph(np.random.default_rng(0).random((20, 100)))
        glyph.plot()
        assert _gap(glyph) < 0.03, f"default colorbar gap should be tight, got {_gap(glyph):.4f}"
        plt.close("all")

    def test_pad_override_widens_the_gap(self):
        """A `cbar_kwargs` pad override still controls the gap.

        Test scenario:
            Passing a larger `pad` moves the colorbar farther from the frame than
            the tight default.
        """
        tight = ArrayGlyph(np.random.default_rng(0).random((20, 100)))
        tight.plot()
        wide = ArrayGlyph(np.random.default_rng(0).random((20, 100)), cbar_kwargs={"pad": 0.15})
        wide.plot()
        assert _gap(wide) > _gap(tight), "a larger pad must widen the gap"
        plt.close("all")
