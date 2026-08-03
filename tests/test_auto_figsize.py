"""Tests for ArrayGlyph's data-aspect default figure sizing."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cleopatra.array_glyph import ArrayGlyph
from cleopatra.templates import publication_map

pytestmark = pytest.mark.plot


class TestAutoFigsize:
    """Tests for ``ArrayGlyph._auto_figsize`` and the ``create_figure_axes`` override."""

    def test_wide_field_gets_a_wide_figure(self):
        """A wide field (more columns than rows) yields a wider-than-tall figure.

        Test scenario:
            With no ``extent``/``coords`` the pixel shape drives the aspect, so a
            20×100 array produces a landscape figure.
        """
        w, h = ArrayGlyph(np.random.default_rng(0).random((20, 100)))._auto_figsize()
        assert w > h, f"a wide field should give a wide figure, got {(w, h)}"

    def test_tall_field_is_not_wide(self):
        """A tall field does not produce a landscape figure.

        Test scenario:
            A 100×20 array (portrait) gives a figure at least as tall as wide.
        """
        w, h = ArrayGlyph(np.random.default_rng(0).random((100, 20)))._auto_figsize()
        assert h >= w, f"a tall field should not give a wide figure, got {(w, h)}"

    def test_plot_applies_auto_figsize(self):
        """`plot()` renders at the auto figure size when `figsize` is omitted.

        Test scenario:
            A 20×100 field renders into a landscape figure without a manual
            `figsize`.
        """
        glyph = ArrayGlyph(np.random.default_rng(0).random((20, 100)))
        fig, _ax = glyph.plot()
        width, height = fig.get_size_inches()
        assert width > height, f"expected a landscape figure, got {(width, height)}"
        plt.close(fig)

    def test_explicit_figsize_is_respected(self):
        """An explicit `figsize` overrides the auto sizing.

        Test scenario:
            Passing `figsize=(5, 5)` renders exactly that, unchanged.
        """
        glyph = ArrayGlyph(np.random.default_rng(0).random((20, 100)), figsize=(5, 5))
        fig, _ax = glyph.plot()
        assert tuple(fig.get_size_inches()) == (5.0, 5.0), "explicit figsize must be honoured"
        plt.close(fig)

    def test_globe_projection_gets_squarish_figure(self):
        """A globe projection uses a near-square figure (the disc), not lon/lat aspect.

        Test scenario:
            The orthographic disc is square, so a wide lon/lat field still gets a
            roughly square figure under `projection="globe"`.
        """
        pytest.importorskip("pyproj", reason="globe needs the [tiles] extra")
        lon = np.linspace(-40.0, 40.0, 60)   # wide in lon/lat
        lat = np.linspace(60.0, 30.0, 30)
        glyph = ArrayGlyph(np.random.default_rng(0).random((30, 60)), coords=(lon, lat), projection="globe")
        w, h = glyph._auto_figsize()
        assert 0.8 < (w / h) < 1.4, f"globe figure should be near-square, got {(w, h)}"

    def test_publication_map_inherits_auto_figsize(self):
        """`publication_map` with no `figsize` inherits the auto sizing.

        Test scenario:
            A wide field through `publication_map` (no `figsize`) renders
            landscape rather than the old fixed square.
        """
        fig, _ax = publication_map(np.random.default_rng(0).random((20, 100)), cmap="viridis")
        width, height = fig.get_size_inches()
        assert width > height, f"publication_map should inherit a landscape figure, got {(width, height)}"
        plt.close(fig)
