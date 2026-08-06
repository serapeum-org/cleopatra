"""Tests for ArrayGlyph's data-aspect default figure sizing."""

from __future__ import annotations

from unittest.mock import MagicMock

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

    def test_auto_sized_plot_is_cropped_to_its_content(self):
        """An auto-sized plot is tightened so the figure *is* its content.

        Test scenario:
            After `plot`, the figure's tight bounding box fills the figure on
            every side (to within the small crop pad), so a plain `savefig` --
            not just Jupyter's `bbox_inches="tight"` inline preview -- has no
            surrounding whitespace.
        """
        glyph = ArrayGlyph(np.random.default_rng(0).random((40, 60)), extent=[-30.0, 30.0, 40.0, 65.0])
        fig, _ax = glyph.plot(title="A wide titled map", colorbar=True)
        fig.canvas.draw()
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        width, height = fig.get_size_inches()
        margins = (tb.x0, tb.y0, width - tb.x1, height - tb.y1)
        assert all(m < 0.1 for m in margins), f"figure should hug its content, margins(in)={margins}"
        plt.close(fig)

    def test_explicit_figsize_is_not_cropped(self):
        """An explicit `figsize=` is honoured verbatim -- the crop never fires.

        Test scenario:
            The caller sized the figure deliberately, so `_tighten_figure` must
            leave it at exactly that size rather than shrinking it to content.
        """
        glyph = ArrayGlyph(np.random.default_rng(0).random((40, 60)), extent=[-30.0, 30.0, 40.0, 65.0], figsize=(8, 8))
        fig, _ax = glyph.plot(title="fixed size")
        assert tuple(round(v, 3) for v in fig.get_size_inches()) == (8.0, 8.0)
        plt.close(fig)

    def test_external_axes_figure_is_not_resized(self):
        """Drawing into a caller-provided axes never resizes that figure.

        Test scenario:
            A subplot the caller manages must keep its size; the crop is limited
            to auto-sized, glyph-owned figures.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ArrayGlyph(np.random.default_rng(0).random((40, 60)), extent=[-30.0, 30.0, 40.0, 65.0]).plot(ax=ax)
        assert tuple(round(v, 3) for v in fig.get_size_inches()) == (6.0, 6.0)
        plt.close(fig)

    def test_tighten_figure_survives_a_backend_draw_error(self):
        """A backend draw/renderer error leaves the auto figsize intact, not a crash.

        Test scenario:
            Tightening is cosmetic and fully optional; if `canvas.draw()` raises a
            backend-specific error (not just the handled renderer `AttributeError`),
            `_tighten_figure` must swallow it and return, leaving the figure at its
            auto size rather than breaking an otherwise-valid render.
        """
        glyph = ArrayGlyph(np.random.default_rng(0).random((20, 30)), extent=[-10.0, 10.0, 40.0, 60.0])
        glyph.fig, glyph.ax = glyph.create_figure_axes()
        before = tuple(glyph.fig.get_size_inches())
        glyph.fig.canvas.draw = MagicMock(side_effect=RuntimeError("backend boom"))
        glyph._tighten_figure()
        assert tuple(glyph.fig.get_size_inches()) == before, "figure left at its auto size"
        plt.close(glyph.fig)
