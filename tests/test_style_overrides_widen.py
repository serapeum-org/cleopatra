"""Widened per-call preset overrides on a styled `ArrayGlyph` plot/animate.

Alongside `vmin`/`vmax`/`center`/`cmap` (loose) and `levels`
(`contour=Contour(...)`), a styled render also accepts `extend` (loose) and the
`bands` / `alpha` / `alpha_range` per-call overrides via
`data_style=DataStyle(...)`: each replaces just that field of the `DATA_STYLES`
preset while the rest of the preset is kept. These tests exercise every field in
both the `plot` and the `animate` code paths, plus the field-interaction rules
the shared `resolve_style_overrides` helper enforces, and that the moved-off
loose keywords now raise.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm

from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
from cleopatra.styling.colors import DATA_STYLES, resolve_style_overrides
from cleopatra.styling.params import Contour, DataStyle

# A styled continuous preset with a fixed `levels` scale + `extend='both'`.
_LEVELS_STYLE = "temperature_2m"
# A styled preset whose opacity is value-linked (`alpha_vmin`/`alpha_vmax`).
_ALPHA_STYLE = "temperature_flame"
# A styled preset whose scale is a pure discrete `bands` shading (no `levels`,
# and no fixed `vmin`/`vmax` -- so a `levels` override is not suppressed by the
# preset's own bounds counting as a caller override in `resolve_style_norm`).
_BANDS_STYLE = "carbon_monoxide"


def _data2d() -> np.ndarray:
    """Return a seeded 8x8 field spanning the temperature presets' scale."""
    return np.random.default_rng(0).uniform(-5.0, 35.0, size=(8, 8))


def _stack() -> np.ndarray:
    """Return a seeded 3-frame `(t, rows, cols)` stack for animate tests."""
    base = _data2d()
    return np.stack([base, base + 1.0, base + 2.0])


def _finite_alpha(image) -> np.ndarray:
    """Return the finite entries of a baked RGBA image's alpha channel."""
    rgba = np.asarray(image.get_array())
    alpha = rgba[..., 3]
    return alpha[np.isfinite(alpha)]


class TestResolveStyleOverrides:
    """Unit coverage of the shared override-resolution helper."""

    def test_none_passed_resolves_to_empty(self):
        """No override inputs yields an empty merge dict (preset untouched)."""
        assert resolve_style_overrides({k: None for k in ("vmin", "cmap", "alpha")}) == {}

    def test_bands_clears_levels(self):
        """An explicit `bands` also clears the preset's `levels`."""
        out = resolve_style_overrides({"bands": 6})
        assert out == {"bands": 6, "levels": None}

    def test_alpha_clears_value_linked_opacity(self):
        """A constant `alpha` clears `alpha_vmin`/`alpha_vmax`."""
        out = resolve_style_overrides({"alpha": 0.5})
        assert out == {"alpha": 0.5, "alpha_vmin": None, "alpha_vmax": None}

    def test_alpha_range_clears_constant_alpha(self):
        """An `alpha_range` maps to `alpha_vmin`/`alpha_vmax` and clears `alpha`."""
        out = resolve_style_overrides({"alpha_range": (0.1, 0.9)})
        assert out == {"alpha_vmin": 0.1, "alpha_vmax": 0.9, "alpha": None}

    def test_alpha_wins_over_alpha_range(self):
        """Given both, the constant `alpha` wins and value-linked is cleared."""
        out = resolve_style_overrides({"alpha": 0.4, "alpha_range": (0.1, 0.9)})
        assert out == {"alpha": 0.4, "alpha_vmin": None, "alpha_vmax": None}

    @pytest.mark.parametrize("bad", [0.5, (1.0,), (0.0, 1.0, 2.0), "0,1"])
    def test_malformed_alpha_range_raises_at_the_datastyle_boundary(self, bad):
        """A non-`(vmin, vmax)` `alpha_range` raises a clear TypeError from DataStyle."""
        with pytest.raises(TypeError, match="alpha_range"):
            DataStyle(alpha_range=bad)


class TestStyleOverridesPlot:
    """Per-call preset overrides on the styled `plot` path."""

    def test_extend_override_changes_colorbar_extend(self):
        """`extend` overrides the preset's colorbar/norm extension."""
        g = ArrayGlyph(_data2d(), extend="max")
        g.plot(data_style=DataStyle(style=_LEVELS_STYLE), colorbar=True)
        assert DATA_STYLES[_LEVELS_STYLE][_LEVELS_STYLE]["extend"] == "both"
        assert g.cbar.norm.extend == "max"

    def test_levels_override_via_contour(self):
        """A `contour=Contour(levels=...)` overrides the preset's own levels."""
        g = ArrayGlyph(_data2d())
        g.plot(
            data_style=DataStyle(style=_LEVELS_STYLE),
            contour=Contour(levels=[0.0, 10.0, 20.0, 30.0]),
            colorbar=True,
        )
        assert list(g.cbar.norm.boundaries) == [0.0, 10.0, 20.0, 30.0]

    def test_bands_override_clears_preset_levels(self):
        """A `bands` override rebands the scale, clearing the preset's 41 levels."""
        g = ArrayGlyph(_data2d())
        g.plot(data_style=DataStyle(style=_LEVELS_STYLE, bands=5), colorbar=True)
        assert isinstance(g.cbar.norm, BoundaryNorm)
        assert len(g.cbar.norm.boundaries) == 6  # 5 bands -> 6 edges, not 41

    def test_bands_on_a_diverging_preset_warns_and_is_ignored(self):
        """A `bands` override warns and is a no-op on a diverging (center) preset."""
        g = ArrayGlyph(np.random.default_rng(0).uniform(-5.0, 5.0, size=(8, 8)))
        with pytest.warns(UserWarning, match="bands"):
            g.plot(data_style=DataStyle(style="anomaly", bands=5), colorbar=True)
        assert isinstance(g.cbar.norm, TwoSlopeNorm)  # preset's diverging scale kept

    def test_levels_override_on_a_bands_preset_uses_the_levels(self):
        """A `levels` override wins over a preset defined with discrete `bands`."""
        g = ArrayGlyph(_data2d())
        g.plot(
            data_style=DataStyle(style=_BANDS_STYLE),
            contour=Contour(levels=[0.0, 5.0, 10.0, 15.0, 20.0]),
            colorbar=True,
        )
        assert list(g.cbar.norm.boundaries) == [0.0, 5.0, 10.0, 15.0, 20.0]

    def test_alpha_override_on_value_linked_preset_does_not_raise(self):
        """A constant `alpha` on a value-linked preset applies without raising."""
        g = ArrayGlyph(_data2d())
        g.plot(data_style=DataStyle(style=_ALPHA_STYLE, alpha=0.5))
        alpha = _finite_alpha(g.im)
        assert np.allclose(alpha, 0.5)

    def test_alpha_range_override_makes_constant_preset_value_linked(self):
        """An `alpha_range` on a constant-alpha preset makes opacity value-linked."""
        g = ArrayGlyph(_data2d())
        g.plot(data_style=DataStyle(style=_LEVELS_STYLE, alpha_range=(10.0, 30.0)))
        alpha = _finite_alpha(g.im)
        assert alpha.max() - alpha.min() > 0.1  # varies with data, not constant

    def test_alpha_wins_over_alpha_range(self):
        """Passing both `alpha` and `alpha_range`, the constant `alpha` wins."""
        g = ArrayGlyph(_data2d())
        g.plot(
            data_style=DataStyle(style=_LEVELS_STYLE, alpha=0.4, alpha_range=(10.0, 30.0))
        )
        alpha = _finite_alpha(g.im)
        assert np.allclose(alpha, 0.4)

    def test_no_override_leaves_preset_unchanged(self):
        """Passing none of the override fields keeps the preset as-is."""
        g = ArrayGlyph(_data2d())
        g.plot(data_style=DataStyle(style=_LEVELS_STYLE), colorbar=True)
        assert g._style_color_overrides == {}
        # The preset's own 41-edge level scale is used verbatim.
        assert len(g.cbar.norm.boundaries) == 41

    def test_overrides_do_not_leak_as_stray_imshow_kwargs(self):
        """Stacked overrides never reach `imshow` as unexpected keywords."""
        g = ArrayGlyph(_data2d(), extend="both")
        # Would raise "unexpected keyword" from AxesImage.set if any leaked.
        g.plot(data_style=DataStyle(style=_LEVELS_STYLE, bands=6, alpha=0.6))
        assert np.asarray(g.im.get_array()).shape[-1] == 4  # baked RGBA


class TestStyleOverridesCrossCall:
    """Overrides are sticky across calls but clear/switch correctly on a re-plot."""

    def test_bands_override_cleared_on_a_later_none(self):
        """`DataStyle(bands=None)` on a later call clears a prior `bands` override."""
        g = ArrayGlyph(_data2d())
        g.plot(data_style=DataStyle(style=_LEVELS_STYLE, bands=5), colorbar=True)
        assert len(g.cbar.norm.boundaries) == 6
        g.plot(data_style=DataStyle(bands=None), colorbar=True)
        assert "bands" not in g._style_color_overrides
        assert len(g.cbar.norm.boundaries) == 41  # the preset's own scale is restored

    def test_alpha_override_cleared_on_a_later_none(self):
        """`DataStyle(alpha=None)` on a later call clears a prior constant `alpha`."""
        g = ArrayGlyph(_data2d())
        g.plot(data_style=DataStyle(style=_ALPHA_STYLE, alpha=0.5))
        assert np.allclose(_finite_alpha(g.im), 0.5)
        g.plot(data_style=DataStyle(alpha=None))
        assert "alpha" not in g._style_color_overrides
        assert not np.allclose(_finite_alpha(g.im), 0.5)  # preset opacity restored

    def test_switch_from_constant_alpha_to_value_linked(self):
        """A later `alpha_range` replaces a previously-set constant `alpha`."""
        g = ArrayGlyph(_data2d())
        g.plot(data_style=DataStyle(style=_LEVELS_STYLE, alpha=0.3))
        assert np.allclose(_finite_alpha(g.im), 0.3)
        g.plot(data_style=DataStyle(alpha_range=(10.0, 30.0)))
        assert "alpha" not in g._style_color_overrides
        alpha = _finite_alpha(g.im)
        assert alpha.max() - alpha.min() > 0.1  # value-linked now, not the stale 0.3

    def test_switch_from_value_linked_to_constant_alpha(self):
        """A later constant `alpha` replaces a previously-set `alpha_range`."""
        g = ArrayGlyph(_data2d())
        g.plot(data_style=DataStyle(style=_LEVELS_STYLE, alpha_range=(10.0, 30.0)))
        assert _finite_alpha(g.im).max() - _finite_alpha(g.im).min() > 0.1
        g.plot(data_style=DataStyle(alpha=0.6))
        assert "alpha_range" not in g._style_color_overrides
        assert np.allclose(_finite_alpha(g.im), 0.6)


class TestStyleOverridesAnimate:
    """Per-call preset overrides on the styled `animate` path."""

    def test_extend_override(self):
        """`extend` overrides the animated preset's norm extension."""
        g = ArrayGlyph(_stack(), extend="max")
        g.animate(data_style=DataStyle(style=_LEVELS_STYLE), time=[0, 1, 2])
        assert isinstance(g.im.norm, BoundaryNorm)
        assert g.im.norm.extend == "max"

    def test_bands_override_clears_preset_levels(self):
        """A `bands` override rebands the animated scale, clearing `levels`."""
        g = ArrayGlyph(_stack())
        g.animate(data_style=DataStyle(style=_LEVELS_STYLE, bands=4), time=[0, 1, 2])
        assert isinstance(g.im.norm, BoundaryNorm)
        assert len(g.im.norm.boundaries) == 5  # 4 bands, not the preset's 41

    def test_levels_override_carries_into_animate(self):
        """A `levels` override captured on a styled plot carries into animate.

        `animate` has no `contour` parameter, so a `levels` override reaches it
        through the sticky `_style_color_overrides` captured by a prior styled
        `plot` on the same glyph (the public `arr` setter swaps in the stack).
        """
        g = ArrayGlyph(_data2d())
        g.plot(
            data_style=DataStyle(style=_LEVELS_STYLE),
            contour=Contour(levels=[0.0, 10.0, 20.0, 30.0]),
        )
        g.arr = _stack()
        g.animate(data_style=DataStyle(style=_LEVELS_STYLE), time=[0, 1, 2])
        assert list(g.im.norm.boundaries) == [0.0, 10.0, 20.0, 30.0]

    def test_alpha_override_on_value_linked_preset_does_not_raise(self):
        """A constant `alpha` animates a value-linked preset without raising."""
        g = ArrayGlyph(_stack())
        anim = g.animate(
            data_style=DataStyle(style=_ALPHA_STYLE, alpha=0.5), time=[0, 1, 2]
        )
        assert isinstance(anim, FuncAnimation)
        assert np.asarray(g.im.get_array()).shape[-1] == 4

    def test_alpha_range_override_animates_value_linked(self):
        """An `alpha_range` override animates a constant-alpha preset value-linked."""
        g = ArrayGlyph(_stack())
        anim = g.animate(
            data_style=DataStyle(style=_LEVELS_STYLE, alpha_range=(10.0, 30.0)),
            time=[0, 1, 2],
        )
        assert isinstance(anim, FuncAnimation)

    def test_no_override_leaves_preset_unchanged(self):
        """Passing none of the override fields keeps the animated preset as-is."""
        g = ArrayGlyph(_stack())
        g.animate(data_style=DataStyle(style=_LEVELS_STYLE), time=[0, 1, 2])
        assert g._style_color_overrides == {}
        assert len(g.im.norm.boundaries) == 41  # the preset's own levels


class TestLooseOverrideKwargsRejected:
    """The moved-off `bands`/`alpha`/`alpha_range` loose keywords now raise."""

    @pytest.mark.parametrize(
        "kwarg, value",
        [("bands", 6), ("alpha", 0.5), ("alpha_range", (10.0, 30.0))],
    )
    def test_loose_override_kwarg_on_plot_rejected(self, kwarg, value):
        """A loose `bands=`/`alpha=`/`alpha_range=` on `plot` points at `DataStyle`."""
        g = ArrayGlyph(_data2d())
        style = DataStyle(style=_LEVELS_STYLE)
        with pytest.raises(ValueError, match="moved onto a grouped parameter object"):
            g.plot(data_style=style, **{kwarg: value})

    @pytest.mark.parametrize(
        "kwarg, value",
        [("bands", 6), ("alpha", 0.5), ("alpha_range", (10.0, 30.0))],
    )
    def test_loose_override_kwarg_on_plot_points_at_data_style(self, kwarg, value):
        """The rejection message names the `data_style=DataStyle(...)` object."""
        g = ArrayGlyph(_data2d())
        with pytest.raises(ValueError, match=rf"data_style=DataStyle\({kwarg}="):
            g.plot(**{kwarg: value})

    @pytest.mark.parametrize(
        "kwarg, value",
        [("bands", 4), ("alpha", 0.5), ("alpha_range", (10.0, 30.0))],
    )
    def test_loose_override_kwarg_on_animate_rejected(self, kwarg, value):
        """A loose `bands=`/`alpha=`/`alpha_range=` on `animate` is rejected."""
        g = ArrayGlyph(_stack())
        style = DataStyle(style=_LEVELS_STYLE)
        with pytest.raises(ValueError, match="moved onto a grouped parameter object"):
            g.animate(data_style=style, time=[0, 1, 2], **{kwarg: value})

    @pytest.mark.parametrize(
        "kwarg, value",
        [("bands", 6), ("alpha", 0.5), ("alpha_range", (10.0, 30.0))],
    )
    def test_loose_override_kwarg_at_construction_rejected(self, kwarg, value):
        """A loose `bands=`/`alpha=`/`alpha_range=` at construction is rejected."""
        arr = _data2d()
        with pytest.raises(ValueError, match="moved onto a grouped parameter object"):
            ArrayGlyph(arr, **{kwarg: value})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
