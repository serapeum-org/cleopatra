"""Tests for cleopatra.glyph.Glyph base class.

Covers initialization, properties, kwargs merging, figure/axes creation,
tick computation, color scale normalization, colorbar creation,
tick adjustment, point overlay, and animation saving.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

from cleopatra.glyph import SUPPORTED_VIDEO_FORMAT, Glyph
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS
from cleopatra.styles import MidpointNormalize


def _make_options(**overrides) -> dict:
    """Build a default_options dict with optional overrides."""
    opts = STYLE_DEFAULTS.copy()
    opts["vmin"] = None
    opts["vmax"] = None
    opts.update(overrides)
    return opts


def _make_glyph(**overrides) -> Glyph:
    """Create a Glyph with standard defaults and optional overrides."""
    kwargs = {}
    opts = _make_options()
    for key in list(overrides):
        if key in ("fig", "ax"):
            kwargs[key] = overrides.pop(key)
        elif key in opts:
            kwargs[key] = overrides.pop(key)
    return Glyph(default_options=_make_options(), **kwargs)


class TestInit:
    """Tests for Glyph.__init__."""

    def test_default_state(self):
        """Test that a fresh Glyph has None fig/ax and vmin/vmax."""
        g = Glyph(default_options=_make_options())
        assert g.fig is None, "fig should be None"
        assert g.ax is None, "ax should be None"
        assert g.vmin is None, "vmin should be None"
        assert g.vmax is None, "vmax should be None"
        assert g.ticks_spacing is None, "ticks_spacing should be None"

    def test_with_fig_ax(self):
        """Test that passing fig/ax stores them on the instance."""
        fig, ax = plt.subplots()
        g = Glyph(default_options=_make_options(), fig=fig, ax=ax)
        assert g.fig is fig, "Should store the provided figure"
        assert g.ax is ax, "Should store the provided axes"

    def test_kwargs_override_defaults(self):
        """Test that kwargs override default_options values."""
        g = Glyph(default_options=_make_options(), cmap="plasma")
        assert g.default_options["cmap"] == "plasma", (
            f"Expected cmap='plasma', got '{g.default_options['cmap']}'"
        )

    def test_invalid_kwarg_raises(self):
        """Test that an unknown kwarg raises ValueError."""
        with pytest.raises(ValueError, match="not_a_real_option"):
            Glyph(default_options=_make_options(), not_a_real_option=42)

    def test_default_options_are_copied(self):
        """Test that modifying glyph options doesn't affect the original dict."""
        opts = _make_options()
        g = Glyph(default_options=opts)
        g.default_options["cmap"] = "viridis"
        assert opts["cmap"] == "coolwarm_r", "Original dict should not be mutated"


class TestProperties:
    """Tests for Glyph properties."""

    def test_vmin_vmax_readable(self):
        """Test vmin/vmax properties return set values."""
        g = Glyph(default_options=_make_options())
        g._vmin = 1.0
        g._vmax = 10.0
        assert g.vmin == 1.0, f"Expected vmin=1.0, got {g.vmin}"
        assert g.vmax == 10.0, f"Expected vmax=10.0, got {g.vmax}"

    def test_default_options_property(self):
        """Test default_options returns the internal dict."""
        g = Glyph(default_options=_make_options())
        assert isinstance(g.default_options, dict), "Should return a dict"
        assert "cmap" in g.default_options, "Should contain cmap key"

    def test_anim_before_animate_raises(self):
        """Test accessing anim before animate() raises ValueError."""
        g = Glyph(default_options=_make_options())
        with pytest.raises(ValueError, match="animate"):
            _ = g.anim

    def test_anim_after_setting(self):
        """Test anim property returns stored animation."""
        g = Glyph(default_options=_make_options())
        sentinel = object()
        g._anim = sentinel
        assert g.anim is sentinel, "Should return the stored _anim"


class TestMergeKwargs:
    """Tests for Glyph._merge_kwargs."""

    def test_valid_kwargs_applied(self):
        """Test that valid kwargs update default_options."""
        g = Glyph(default_options=_make_options())
        g._merge_kwargs({"gamma": 0.8, "midpoint": 5})
        assert g.default_options["gamma"] == 0.8, (
            f"Expected gamma=0.8, got {g.default_options['gamma']}"
        )
        assert g.default_options["midpoint"] == 5, (
            f"Expected midpoint=5, got {g.default_options['midpoint']}"
        )

    def test_invalid_key_raises(self):
        """Test that invalid key raises ValueError."""
        g = Glyph(default_options=_make_options())
        with pytest.raises(ValueError, match="bogus_key"):
            g._merge_kwargs({"bogus_key": 99})

    def test_empty_kwargs_noop(self):
        """Test that empty kwargs dict does nothing."""
        g = Glyph(default_options=_make_options())
        original_cmap = g.default_options["cmap"]
        g._merge_kwargs({})
        assert g.default_options["cmap"] == original_cmap, "Should be unchanged"


class TestCreateFigureAxes:
    """Tests for Glyph.create_figure_axes."""

    def test_returns_figure_and_axes(self):
        """Test that create_figure_axes returns (Figure, Axes)."""
        g = Glyph(default_options=_make_options())
        fig, ax = g.create_figure_axes()
        assert isinstance(fig, Figure), f"Expected Figure, got {type(fig)}"
        assert ax is not None, "Should return Axes"
        plt.close(fig)

    def test_respects_figsize(self):
        """Test that figsize from default_options is used."""
        g = Glyph(default_options=_make_options(figsize=(12, 4)))
        fig, ax = g.create_figure_axes()
        w, h = fig.get_size_inches()
        assert (w, h) == (12.0, 4.0), f"Expected (12,4), got ({w},{h})"
        plt.close(fig)


class TestGetTicks:
    """Tests for Glyph.get_ticks."""

    def test_evenly_divisible(self):
        """Test ticks when vmax is evenly divisible by spacing."""
        g = Glyph(default_options=_make_options())
        g._default_options["vmin"] = 0.0
        g._default_options["vmax"] = 10.0
        g._default_options["ticks_spacing"] = 2.0
        ticks = g.get_ticks()
        expected = np.arange(0, 12, 2)
        np.testing.assert_array_almost_equal(ticks, expected)

    def test_not_evenly_divisible(self):
        """Test ticks when vmax is not evenly divisible by spacing."""
        g = Glyph(default_options=_make_options())
        g._default_options["vmin"] = 0.0
        g._default_options["vmax"] = 7.0
        g._default_options["ticks_spacing"] = 3.0
        ticks = g.get_ticks()
        assert ticks[0] == 0.0, f"First tick should be 0.0, got {ticks[0]}"
        assert ticks[-1] >= 7.0, f"Last tick should be >= 7.0, got {ticks[-1]}"

    def test_single_tick_spacing_equals_range(self):
        """Test when ticks_spacing equals the full range."""
        g = Glyph(default_options=_make_options())
        g._default_options["vmin"] = 0.0
        g._default_options["vmax"] = 5.0
        g._default_options["ticks_spacing"] = 5.0
        ticks = g.get_ticks()
        assert len(ticks) >= 2, f"Should have at least 2 ticks, got {len(ticks)}"
        assert ticks[0] == 0.0, f"First tick should be 0.0, got {ticks[0]}"


class TestCreateNormAndCbarKw:
    """Tests for Glyph._create_norm_and_cbar_kw."""

    @pytest.fixture()
    def glyph(self):
        """Create a Glyph with all color scale options available."""
        return Glyph(default_options=_make_options())

    def test_linear_returns_none_norm(self, glyph):
        """Test that linear color scale returns None norm."""
        glyph._default_options["color_scale"] = "linear"
        ticks = np.array([0.0, 5.0, 10.0])
        norm, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert norm is None, f"Linear should return None norm, got {type(norm)}"
        assert "ticks" in cbar_kw, "cbar_kw should contain ticks"

    def test_power_returns_power_norm(self, glyph):
        """Test that power color scale returns PowerNorm."""
        glyph._default_options["color_scale"] = "power"
        glyph._default_options["gamma"] = 0.3
        ticks = np.array([0.0, 5.0, 10.0])
        norm, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert isinstance(norm, mcolors.PowerNorm), (
            f"Expected PowerNorm, got {type(norm)}"
        )

    def test_sym_lognorm(self, glyph):
        """Test that sym-lognorm returns SymLogNorm with formatter."""
        glyph._default_options["color_scale"] = "sym-lognorm"
        ticks = np.array([0.0, 5.0, 10.0])
        norm, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert isinstance(norm, mcolors.SymLogNorm), (
            f"Expected SymLogNorm, got {type(norm)}"
        )
        assert "format" in cbar_kw, "sym-lognorm should include a formatter"

    def test_boundary_norm_with_default_bounds(self, glyph):
        """Test boundary-norm with no explicit bounds uses ticks."""
        glyph._default_options["color_scale"] = "boundary-norm"
        glyph._default_options["bounds"] = None
        ticks = np.array([0.0, 5.0, 10.0])
        norm, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert isinstance(norm, mcolors.BoundaryNorm), (
            f"Expected BoundaryNorm, got {type(norm)}"
        )

    def test_boundary_norm_with_custom_bounds(self, glyph):
        """Test boundary-norm with user-provided bounds."""
        glyph._default_options["color_scale"] = "boundary-norm"
        glyph._default_options["bounds"] = [0, 3, 6, 9]
        ticks = np.array([0.0, 5.0, 10.0])
        norm, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert isinstance(norm, mcolors.BoundaryNorm), (
            f"Expected BoundaryNorm, got {type(norm)}"
        )
        np.testing.assert_array_equal(cbar_kw["ticks"], [0, 3, 6, 9])

    def test_midpoint_returns_midpoint_normalize(self, glyph):
        """Test midpoint scale returns MidpointNormalize."""
        glyph._default_options["color_scale"] = "midpoint"
        glyph._default_options["midpoint"] = 5.0
        ticks = np.array([0.0, 5.0, 10.0])
        norm, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert isinstance(norm, MidpointNormalize), (
            f"Expected MidpointNormalize, got {type(norm)}"
        )

    def test_invalid_color_scale_raises(self, glyph):
        """Test that unknown color_scale raises ValueError."""
        glyph._default_options["color_scale"] = "rainbow-magic"
        ticks = np.array([0.0, 5.0, 10.0])
        with pytest.raises(ValueError, match="Invalid color scale"):
            glyph._create_norm_and_cbar_kw(ticks)

    @pytest.mark.parametrize(
        "scale",
        ["Linear", "POWER", "Sym-Lognorm", "Boundary-Norm", "Midpoint"],
    )
    def test_case_insensitive(self, glyph, scale):
        """Test that color_scale matching is case-insensitive.

        Args:
            scale: Color scale name with mixed case.
        """
        glyph._default_options["color_scale"] = scale
        ticks = np.array([0.0, 5.0, 10.0])
        norm, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert "ticks" in cbar_kw, f"Should produce cbar_kw for scale '{scale}'"


class TestCreateColorBar:
    """Tests for Glyph.create_color_bar."""

    def test_creates_colorbar(self):
        """Test that create_color_bar returns a Colorbar."""
        g = Glyph(default_options=_make_options())
        fig, ax = plt.subplots()
        im = ax.imshow(np.arange(9).reshape(3, 3))
        ticks = np.array([0.0, 4.0, 8.0])
        cbar = g.create_color_bar(ax, im, {"ticks": ticks})
        assert isinstance(cbar, Colorbar), f"Expected Colorbar, got {type(cbar)}"
        plt.close(fig)

    def test_respects_orientation(self):
        """Test that horizontal orientation is applied."""
        g = Glyph(
            default_options=_make_options(cbar_orientation="horizontal")
        )
        fig, ax = plt.subplots()
        im = ax.imshow(np.arange(9).reshape(3, 3))
        cbar = g.create_color_bar(ax, im, {"ticks": np.array([0, 4, 8])})
        assert cbar.orientation == "horizontal", (
            f"Expected horizontal, got {cbar.orientation}"
        )
        plt.close(fig)


class TestAdjustTicks:
    """Tests for Glyph.adjust_ticks."""

    def test_adjust_x_ticks(self):
        """Test adjust_ticks on x axis applies formatter."""
        g = Glyph(default_options=_make_options())
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 2])
        g.ax = ax
        g.fig = fig
        g.adjust_ticks(axis="x", multiply_value=10, add_value=5, fmt="{0:.0f}")
        formatter = ax.xaxis.get_major_formatter()
        assert formatter is not None, "Formatter should be set"
        plt.close(fig)

    def test_adjust_y_ticks(self):
        """Test adjust_ticks on y axis applies formatter."""
        g = Glyph(default_options=_make_options())
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 2])
        g.ax = ax
        g.fig = fig
        g.adjust_ticks(axis="y", multiply_value=2)
        formatter = ax.yaxis.get_major_formatter()
        assert formatter is not None, "Formatter should be set"
        plt.close(fig)

    def test_hide_x_axis(self):
        """Test adjust_ticks with visible=False hides the axis."""
        g = Glyph(default_options=_make_options())
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        g.ax = ax
        g.fig = fig
        g.adjust_ticks(axis="x", visible=False)
        assert not ax.get_xaxis().get_visible(), "x-axis should be hidden"
        plt.close(fig)

    def test_hide_y_axis(self):
        """Test adjust_ticks with visible=False hides the y-axis."""
        g = Glyph(default_options=_make_options())
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        g.ax = ax
        g.fig = fig
        g.adjust_ticks(axis="y", visible=False)
        assert not ax.get_yaxis().get_visible(), "y-axis should be hidden"
        plt.close(fig)


class TestPlotPointValues:
    """Tests for Glyph._plot_point_values static method."""

    def test_creates_text_per_point(self):
        """Test that one text artist is created per point."""
        fig, ax = plt.subplots()
        points = np.array([[10.0, 0, 0], [20.0, 1, 1], [30.0, 2, 2]])
        texts = Glyph._plot_point_values(ax, points, "blue", 12)
        assert len(texts) == 3, f"Expected 3 text artists, got {len(texts)}"
        plt.close(fig)

    def test_text_positions(self):
        """Test that text is placed at (col, row) coordinates."""
        fig, ax = plt.subplots()
        points = np.array([[99.0, 3.0, 5.0]])
        texts = Glyph._plot_point_values(ax, points, "red", 10)
        pos = texts[0].get_position()
        assert pos == (5.0, 3.0), f"Expected position (5, 3), got {pos}"
        plt.close(fig)

    def test_empty_points_returns_empty_list(self):
        """Test that empty points array returns empty list."""
        fig, ax = plt.subplots()
        points = np.empty((0, 3))
        texts = Glyph._plot_point_values(ax, points, "red", 10)
        assert len(texts) == 0, f"Expected 0 text artists, got {len(texts)}"
        plt.close(fig)


class TestSaveAnimation:
    """Tests for Glyph.save_animation."""

    def test_unsupported_format_raises(self):
        """Test that unsupported format raises ValueError."""
        g = Glyph(default_options=_make_options())
        with pytest.raises(ValueError, match="not supported"):
            g.save_animation("output.webm")

    def test_no_anim_raises(self):
        """Test that saving without animate() raises ValueError."""
        g = Glyph(default_options=_make_options())
        with pytest.raises(ValueError, match="animate"):
            g.save_animation("output.gif")

    @pytest.mark.parametrize("ext", SUPPORTED_VIDEO_FORMAT)
    def test_supported_formats_accepted(self, ext):
        """Test that all supported formats don't raise ValueError on format check.

        Args:
            ext: File extension to test.

        Test scenario:
            We test that the format validation passes. The actual save
            will fail because there's no real animation, but the
            ValueError for unsupported format should NOT be raised.
        """
        g = Glyph(default_options=_make_options())
        g._anim = None
        with pytest.raises(ValueError, match="animate"):
            g.save_animation(f"output.{ext}")


class TestSupportedVideoFormat:
    """Tests for module-level SUPPORTED_VIDEO_FORMAT constant."""

    def test_contains_expected_formats(self):
        """Test that all expected formats are present."""
        expected = {"gif", "mov", "avi", "mp4"}
        assert set(SUPPORTED_VIDEO_FORMAT) == expected, (
            f"Expected {expected}, got {set(SUPPORTED_VIDEO_FORMAT)}"
        )
