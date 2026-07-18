"""Tests for cleopatra.glyph.Glyph base class.

Covers initialization, properties, kwargs merging, figure/axes creation,
tick computation, color scale normalization, colorbar creation,
tick adjustment, point overlay, and animation saving.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

import cleopatra.glyph as glyph_mod
from cleopatra.glyph import (
    MAX_DISCRETE_LEVELS,
    SUPPORTED_VIDEO_FORMAT,
    Glyph,
    _clear_prior_render_artists,
    _mark_render_artists,
)
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS
from cleopatra.styles import ColorScale, MidpointNormalize


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
        plt.close(fig)

    def test_ax_without_fig_keeps_axes_and_derives_fig(self):
        """Passing `ax` alone keeps the axes and derives the figure from it.

        Test scenario:
            Previously an `ax` given without a `fig` was dropped (both set
            to None). Now the axes is retained and `self.fig` is derived via
            `ax.get_figure()`, so a caller can bind a target with `ax=` only.
        """
        fig, ax = plt.subplots()
        g = Glyph(default_options=_make_options(), ax=ax)
        assert g.ax is ax, "ax passed alone must be kept"
        assert g.fig is ax.get_figure(), "fig must be derived from the axes"
        plt.close(fig)

    def test_fig_without_ax_keeps_fig_and_leaves_ax_none(self):
        """Passing `fig` alone keeps the figure and leaves `ax` None.

        Test scenario:
            A figure with no specific axes binds the figure handle; the axes
            stays None until render time.
        """
        fig = plt.figure()
        g = Glyph(default_options=_make_options(), fig=fig)
        assert g.fig is fig, "fig passed alone must be kept"
        assert g.ax is None, "ax should remain None when only fig is given"
        plt.close(fig)

    def test_explicit_fig_wins_over_axes_parent(self):
        """When both `fig` and `ax` are given, the explicit `fig` is stored.

        Test scenario:
            The explicit figure handle takes precedence for `self.fig` rather
            than being recomputed from the axes.
        """
        fig, ax = plt.subplots()
        g = Glyph(default_options=_make_options(), fig=fig, ax=ax)
        assert g.fig is fig, "explicit fig should be stored as-is"
        plt.close(fig)

    def test_mismatched_fig_ax_warns(self):
        """A `fig` that does not own `ax` warns the caller.

        Test scenario:
            Passing an `ax` together with an unrelated `fig` is almost always
            a mistake (the two figure handles disagree), so the constructor
            emits a warning while still honouring the explicit `fig`.
        """
        fig_a, ax_a = plt.subplots()
        fig_b = plt.figure()
        with pytest.warns(UserWarning, match="not the figure that owns"):
            g = Glyph(default_options=_make_options(), fig=fig_b, ax=ax_a)
        assert g.fig is fig_b, "explicit fig is still honoured despite the mismatch"
        plt.close(fig_a)
        plt.close(fig_b)

    def test_matched_fig_ax_does_not_warn(self):
        """Passing `ax` with its own parent `fig` does not warn.

        Test scenario:
            The consistent case (fig owns ax) must stay silent.
        """
        import warnings as _warnings

        fig, ax = plt.subplots()
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            g = Glyph(default_options=_make_options(), fig=fig, ax=ax)
        assert g.fig is fig, "matched fig should be stored without warning"
        plt.close(fig)

    def test_kwargs_override_defaults(self):
        """Test that kwargs override default_options values."""
        g = Glyph(default_options=_make_options(), cmap="plasma")
        assert (
            g.default_options["cmap"] == "plasma"
        ), f"Expected cmap='plasma', got '{g.default_options['cmap']}'"

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
        assert (
            g.default_options["gamma"] == 0.8
        ), f"Expected gamma=0.8, got {g.default_options['gamma']}"
        assert (
            g.default_options["midpoint"] == 5
        ), f"Expected midpoint=5, got {g.default_options['midpoint']}"

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
        assert isinstance(
            norm, mcolors.PowerNorm
        ), f"Expected PowerNorm, got {type(norm)}"

    def test_sym_lognorm(self, glyph):
        """Test that sym-lognorm returns SymLogNorm with formatter."""
        glyph._default_options["color_scale"] = "sym-lognorm"
        ticks = np.array([0.0, 5.0, 10.0])
        norm, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert isinstance(
            norm, mcolors.SymLogNorm
        ), f"Expected SymLogNorm, got {type(norm)}"
        assert "format" in cbar_kw, "sym-lognorm should include a formatter"

    def test_boundary_norm_with_default_bounds(self, glyph):
        """Test boundary-norm with no explicit bounds uses ticks."""
        glyph._default_options["color_scale"] = "boundary-norm"
        glyph._default_options["bounds"] = None
        ticks = np.array([0.0, 5.0, 10.0])
        norm, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert isinstance(
            norm, mcolors.BoundaryNorm
        ), f"Expected BoundaryNorm, got {type(norm)}"

    def test_boundary_norm_with_custom_bounds(self, glyph):
        """Test boundary-norm with user-provided bounds."""
        glyph._default_options["color_scale"] = "boundary-norm"
        glyph._default_options["bounds"] = [0, 3, 6, 9]
        ticks = np.array([0.0, 5.0, 10.0])
        norm, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert isinstance(
            norm, mcolors.BoundaryNorm
        ), f"Expected BoundaryNorm, got {type(norm)}"
        np.testing.assert_array_equal(cbar_kw["ticks"], [0, 3, 6, 9])

    def test_midpoint_returns_midpoint_normalize(self, glyph):
        """Test midpoint scale returns MidpointNormalize."""
        glyph._default_options["color_scale"] = "midpoint"
        glyph._default_options["midpoint"] = 5.0
        ticks = np.array([0.0, 5.0, 10.0])
        norm, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert isinstance(
            norm, MidpointNormalize
        ), f"Expected MidpointNormalize, got {type(norm)}"

    def test_invalid_color_scale_string_raises(self, glyph):
        """An unrecognised color_scale string raises `ValueError`."""
        glyph._default_options["color_scale"] = "rainbow-magic"
        ticks = np.array([0.0, 5.0, 10.0])
        with pytest.raises(ValueError, match="Invalid color_scale"):
            glyph._create_norm_and_cbar_kw(ticks)

    def test_non_string_color_scale_raises_valueerror_not_attributeerror(self, glyph):
        """A non-string color_scale (e.g. an int) raises `ValueError`, not `AttributeError`.

        Regression for the case where `color_scale.lower()` blew up with
        `AttributeError: 'int' object has no attribute 'lower'`.
        """
        glyph._default_options["color_scale"] = 1
        ticks = np.array([0.0, 5.0, 10.0])
        with pytest.raises(ValueError, match="Invalid color_scale"):
            glyph._create_norm_and_cbar_kw(ticks)

    def test_colorscale_member_accepted(self, glyph):
        """Passing a `ColorScale` member directly works."""
        glyph._default_options["color_scale"] = ColorScale.POWER
        ticks = np.array([0.0, 5.0, 10.0])
        norm, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert isinstance(norm, mcolors.PowerNorm)

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
        g = Glyph(default_options=_make_options(cbar_orientation="horizontal"))
        fig, ax = plt.subplots()
        im = ax.imshow(np.arange(9).reshape(3, 3))
        cbar = g.create_color_bar(ax, im, {"ticks": np.array([0, 4, 8])})
        assert (
            cbar.orientation == "horizontal"
        ), f"Expected horizontal, got {cbar.orientation}"
        plt.close(fig)

    def test_single_axes_uses_gridspec(self):
        """Test that single-axes figure uses gridspec for colorbar.

        Test scenario:
            A figure with one axes should use use_gridspec=True
            for optimal colorbar layout.
        """
        g = Glyph(default_options=_make_options())
        fig, ax = plt.subplots()
        im = ax.imshow(np.arange(9).reshape(3, 3))
        cbar = g.create_color_bar(ax, im, {"ticks": np.array([0, 4, 8])})
        assert cbar is not None, "Colorbar should be created"
        plt.close(fig)

    def test_subplot_colorbar_does_not_steal_space(self):
        """Test that colorbar on subplot doesn't break sibling axes.

        Test scenario:
            Create a 1x2 subplot, add an image and colorbar to each.
            Both axes should retain their images — the second colorbar
            should not steal space from the first axes via gridspec.
        """
        g = Glyph(default_options=_make_options())
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        im1 = axes[0].imshow(np.arange(9).reshape(3, 3))
        g.create_color_bar(axes[0], im1, {"ticks": np.array([0, 4, 8])})

        im2 = axes[1].imshow(np.arange(9, 0, -1).reshape(3, 3))
        g.create_color_bar(axes[1], im2, {"ticks": np.array([1, 5, 9])})

        assert (
            len(axes[0].images) == 1
        ), f"First axes should have 1 image, got {len(axes[0].images)}"
        assert (
            len(axes[1].images) == 1
        ), f"Second axes should have 1 image, got {len(axes[1].images)}"
        assert (
            len(fig.axes) == 4
        ), f"Expected 4 axes (2 plot + 2 colorbar), got {len(fig.axes)}"
        plt.close(fig)

    def test_subplot_colorbars_independent(self):
        """Test that each subplot's colorbar is independent.

        Test scenario:
            Two subplots with different data ranges should each get
            their own colorbar with correct tick values.
        """
        g1 = Glyph(default_options=_make_options())
        g2 = Glyph(default_options=_make_options())
        fig, axes = plt.subplots(1, 2)

        im1 = axes[0].imshow(np.zeros((3, 3)))
        ticks1 = np.array([0.0])
        cbar1 = g1.create_color_bar(axes[0], im1, {"ticks": ticks1})

        im2 = axes[1].imshow(np.ones((3, 3)) * 100)
        ticks2 = np.array([100.0])
        cbar2 = g2.create_color_bar(axes[1], im2, {"ticks": ticks2})

        assert cbar1 is not cbar2, "Colorbars should be different objects"
        plt.close(fig)

    def test_three_subplots_all_visible(self):
        """Test colorbar works correctly with 3+ subplots.

        Test scenario:
            1x3 subplot layout, each with an image and colorbar.
            All 3 axes should retain their images.
        """
        g = Glyph(default_options=_make_options())
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for i, ax in enumerate(axes):
            data = np.random.RandomState(i).rand(5, 5)
            im = ax.imshow(data)
            g.create_color_bar(ax, im, {"ticks": np.array([0.0, 0.5, 1.0])})

        for i, ax in enumerate(axes):
            assert (
                len(ax.images) == 1
            ), f"Axes {i} should have 1 image, got {len(ax.images)}"
        assert (
            len(fig.axes) == 6
        ), f"Expected 6 axes (3 plot + 3 colorbar), got {len(fig.axes)}"
        plt.close(fig)

    def test_2x2_grid_subplots(self):
        """Test colorbar on a 2x2 grid of subplots.

        Test scenario:
            4 subplots in a 2x2 grid, each with colorbar.
            All should render independently.
        """
        g = Glyph(default_options=_make_options())
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        for ax in axes.flat:
            im = ax.imshow(np.random.RandomState(42).rand(4, 4))
            g.create_color_bar(ax, im, {"ticks": np.array([0.0, 0.5, 1.0])})

        for i, ax in enumerate(axes.flat):
            assert (
                len(ax.images) == 1
            ), f"Axes {i} should have 1 image, got {len(ax.images)}"
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
        """Test that unsupported format raises ValueError.

        An animation is attached so the format check (not the missing-anim
        guard) is what fires.
        """
        g = Glyph(default_options=_make_options())
        g._anim = MagicMock(spec=FuncAnimation)
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

    def test_delegates_to_free_function(self, monkeypatch):
        """`Glyph.save_animation` forwards `self.anim` to the free function."""
        g = Glyph(default_options=_make_options())
        anim = MagicMock(spec=FuncAnimation)
        g._anim = anim

        spy = MagicMock()
        monkeypatch.setattr(glyph_mod, "_save_animation", spy)
        g.save_animation("movie.gif", fps=5)

        spy.assert_called_once_with(anim, "movie.gif", fps=5)

    def test_forwards_quality_kwargs(self, monkeypatch):
        """Extra quality kwargs are forwarded verbatim to the free function.

        Test scenario:
            ``crf``/``preset``/``dpi`` passed to the wrapper must reach
            ``cleopatra.animation.save_animation`` unchanged so callers get the
            full quality-control surface through the glyph.
        """
        g = Glyph(default_options=_make_options())
        anim = MagicMock(spec=FuncAnimation)
        g._anim = anim

        spy = MagicMock()
        monkeypatch.setattr(glyph_mod, "_save_animation", spy)
        g.save_animation("movie.mp4", fps=2, crf=24, preset="slow", dpi=150)

        spy.assert_called_once_with(
            anim, "movie.mp4", fps=2, crf=24, preset="slow", dpi=150
        )


class TestSupportedVideoFormat:
    """Tests for module-level SUPPORTED_VIDEO_FORMAT constant."""

    def test_contains_expected_formats(self):
        """Test that all expected formats are present."""
        expected = {"gif", "mov", "avi", "mp4", "webp"}
        assert (
            set(SUPPORTED_VIDEO_FORMAT) == expected
        ), f"Expected {expected}, got {set(SUPPORTED_VIDEO_FORMAT)}"


class TestLevelsToBounds:
    """Tests for `Glyph._levels_to_bounds` static method."""

    def test_none_returns_none(self):
        """`levels=None` produces `None`: continuous-norm path is selected."""
        result = Glyph._levels_to_bounds(None, 0.0, 1.0)
        assert result is None, f"Expected None for levels=None, got {result!r}"

    def test_int_creates_linspace(self):
        """`levels=int` produces `int` linearly-spaced edges between vmin/vmax."""
        result = Glyph._levels_to_bounds(5, 0.0, 1.0)
        assert result is not None, "int levels should produce an array"
        assert len(result) == 5, f"Expected 5 edges, got {len(result)}"
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)

    def test_list_sorted_ascending(self):
        """Unsorted explicit edges are returned sorted ascending."""
        result = Glyph._levels_to_bounds([5.0, 1.0, 3.0], 0.0, 10.0)
        np.testing.assert_array_equal(result, [1.0, 3.0, 5.0])

    def test_ndarray_passthrough(self):
        """A numpy array of edges is converted to float and sorted."""
        result = Glyph._levels_to_bounds(np.array([2, 1, 3]), 0.0, 10.0)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
        assert result.dtype == np.float64

    def test_int_with_negative_range(self):
        """`levels=int` works for negative-to-positive vmin/vmax."""
        result = Glyph._levels_to_bounds(3, -1.0, 1.0)
        np.testing.assert_array_almost_equal(result, [-1.0, 0.0, 1.0])

    def test_explicit_edges_with_negative_values(self):
        """Explicit edges spanning negative-to-positive values are sorted ascending."""
        result = Glyph._levels_to_bounds([3.0, -2.0, 1.0, -5.0], -10.0, 10.0)
        np.testing.assert_array_equal(result, [-5.0, -2.0, 1.0, 3.0])

    @pytest.mark.parametrize("bad", [0, 1, -3, MAX_DISCRETE_LEVELS + 1, 10**9])
    def test_int_levels_out_of_range_raises(self, bad):
        """An integer `levels` outside `[2, MAX_DISCRETE_LEVELS]` raises.

        Args:
            bad: An integer level count that should be rejected.
        """
        with pytest.raises(ValueError, match=r"`levels` as an integer must be between"):
            Glyph._levels_to_bounds(bad, 0.0, 1.0)

    def test_int_levels_at_bounds_ok(self):
        """The min (2) and max (`MAX_DISCRETE_LEVELS`) integer levels are accepted."""
        assert len(Glyph._levels_to_bounds(2, 0.0, 1.0)) == 2
        assert len(Glyph._levels_to_bounds(MAX_DISCRETE_LEVELS, 0.0, 1.0)) == (
            MAX_DISCRETE_LEVELS
        )


class TestCreateNormAndCbarKwLevelsExtend:
    """Tests for the levels/extend branches added in CLEO-3."""

    @pytest.fixture()
    def glyph(self):
        """Create a Glyph configured for linear color scale."""
        return Glyph(default_options=_make_options())

    def test_levels_int_under_linear_creates_boundary_norm(self, glyph):
        """`color_scale='linear', levels=4` switches to a `BoundaryNorm`."""
        glyph._default_options["color_scale"] = "linear"
        glyph._default_options["levels"] = 4
        ticks = np.array([0.0, 5.0, 10.0])
        norm, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert isinstance(
            norm, mcolors.BoundaryNorm
        ), f"Expected BoundaryNorm under linear+levels, got {type(norm)}"
        assert len(cbar_kw["ticks"]) == 4

    def test_levels_list_under_linear(self, glyph):
        """A list of edges under `linear` produces a `BoundaryNorm`."""
        glyph._default_options["color_scale"] = "linear"
        glyph._default_options["levels"] = [0.0, 0.5, 1.0]
        ticks = np.array([0.0, 0.5, 1.0])
        norm, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert isinstance(norm, mcolors.BoundaryNorm)
        np.testing.assert_array_almost_equal(cbar_kw["ticks"], [0.0, 0.5, 1.0])

    def test_levels_under_boundary_norm_with_explicit_bounds_wins(self, glyph):
        """`boundary-norm` with explicit `bounds` ignores `levels`."""
        glyph._default_options["color_scale"] = "boundary-norm"
        glyph._default_options["bounds"] = [0, 10, 20]
        glyph._default_options["levels"] = [0, 5, 10, 15]
        ticks = np.array([0.0, 10.0, 20.0])
        norm, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert isinstance(norm, mcolors.BoundaryNorm)
        np.testing.assert_array_equal(cbar_kw["ticks"], [0, 10, 20])

    def test_levels_under_boundary_norm_no_explicit_bounds(self, glyph):
        """`boundary-norm` with no `bounds` falls through to `levels`."""
        glyph._default_options["color_scale"] = "boundary-norm"
        glyph._default_options["bounds"] = None
        glyph._default_options["levels"] = [0, 1, 2]
        ticks = np.array([0.0, 1.0, 2.0])
        norm, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert isinstance(norm, mcolors.BoundaryNorm)
        np.testing.assert_array_equal(cbar_kw["ticks"], [0.0, 1.0, 2.0])

    def test_extend_auto_resolves_to_neither_when_no_levels(self, glyph):
        """`extend=None, levels=None` -> `cbar_kw['extend'] == 'neither'`."""
        glyph._default_options["color_scale"] = "linear"
        glyph._default_options["levels"] = None
        glyph._default_options["extend"] = None
        ticks = np.array([0.0, 1.0])
        _, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert cbar_kw["extend"] == "neither"

    def test_extend_auto_resolves_to_both_when_levels_set(self, glyph):
        """`extend=None, levels=[...]` -> `cbar_kw['extend'] == 'both'`."""
        glyph._default_options["color_scale"] = "linear"
        glyph._default_options["levels"] = [0.0, 0.5, 1.0]
        glyph._default_options["extend"] = None
        ticks = np.array([0.0, 0.5, 1.0])
        _, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert cbar_kw["extend"] == "both"

    @pytest.mark.parametrize("explicit", ["neither", "both", "min", "max"])
    def test_extend_explicit_value_passes_through(self, glyph, explicit):
        """Any allowed explicit `extend` value is forwarded as-is.

        Args:
            explicit: One of the allowed extend keywords.
        """
        glyph._default_options["color_scale"] = "linear"
        glyph._default_options["extend"] = explicit
        ticks = np.array([0.0, 1.0])
        _, cbar_kw = glyph._create_norm_and_cbar_kw(ticks)
        assert cbar_kw["extend"] == explicit


class TestCreateColorBarKwargsMerge:
    """Tests for `create_color_bar` merging of `cbar_kwargs`."""

    def test_user_label_replaces_default(self):
        """`cbar_kwargs={'label': ...}` flows through `cbar.set_label`."""
        g = Glyph(
            default_options=_make_options(
                cbar_label="Default", cbar_kwargs={"label": "Custom"}
            )
        )
        fig, ax = plt.subplots()
        im = ax.imshow(np.arange(9).reshape(3, 3))
        cbar = g.create_color_bar(ax, im, {"ticks": np.array([0, 4, 8])})
        assert (
            cbar.ax.get_ylabel() == "Custom"
        ), f"Expected user label 'Custom', got {cbar.ax.get_ylabel()!r}"
        plt.close(fig)

    def test_user_shrink_overrides_default(self):
        """User-supplied `shrink` survives the defaults merge."""
        g = Glyph(
            default_options=_make_options(cbar_length=0.9, cbar_kwargs={"shrink": 0.3})
        )
        fig, ax = plt.subplots()
        im = ax.imshow(np.arange(9).reshape(3, 3))
        cbar = g.create_color_bar(ax, im, {"ticks": np.array([0, 4, 8])})
        cbar_h = cbar.ax.get_position().height
        ax_h = ax.get_position().height
        assert cbar_h <= ax_h * 0.6, (
            f"User shrink=0.3 should produce a shorter colorbar, "
            f"got ratio {cbar_h / ax_h:.3f}"
        )
        plt.close(fig)

    def test_invalid_cbar_kwargs_type_raises(self):
        """Non-dict `cbar_kwargs` raises a clear `TypeError`."""
        g = Glyph(default_options=_make_options(cbar_kwargs="not-a-dict"))
        fig, ax = plt.subplots()
        im = ax.imshow(np.arange(9).reshape(3, 3))
        with pytest.raises(TypeError, match="cbar_kwargs must be a dict"):
            g.create_color_bar(ax, im, {"ticks": np.array([0, 4, 8])})
        plt.close(fig)

    def test_none_cbar_kwargs_is_noop(self):
        """`cbar_kwargs=None` falls back to the cleopatra defaults."""
        g = Glyph(default_options=_make_options(cbar_kwargs=None))
        fig, ax = plt.subplots()
        im = ax.imshow(np.arange(9).reshape(3, 3))
        cbar = g.create_color_bar(ax, im, {"ticks": np.array([0, 4, 8])})
        assert isinstance(cbar, Colorbar)
        plt.close(fig)


class TestSaveAnimationVideoBranch:
    """Tests covering the FFmpeg fallback path of `save_animation`."""

    def test_ffmpeg_missing_raises_friendly_error(self, monkeypatch, tmp_path):
        """A missing FFmpeg binary surfaces as `FileNotFoundError` with URL."""
        g = Glyph(default_options=_make_options())
        anim = MagicMock(spec=FuncAnimation)
        anim.save = MagicMock(side_effect=FileNotFoundError("ffmpeg not found"))
        g._anim = anim

        target = tmp_path / "movie.mp4"
        with pytest.raises(FileNotFoundError, match="ffmpeg.org"):
            g.save_animation(str(target))


class TestResolveLimits:
    """Tests for Glyph._resolve_limits (T0.0)."""

    def test_auto_resolves_both_from_data(self):
        """Both limits unset -> taken from the nan-aware data range.

        Test scenario:
            vmin/vmax are None, so the finite min/max of the data
            (1.0, 9.0) are returned as floats.
        """
        g = _make_glyph()
        vmin, vmax = g._resolve_limits(np.array([5.0, 1.0, 9.0]))
        assert (vmin, vmax) == (1.0, 9.0), f"Expected (1.0, 9.0), got {(vmin, vmax)}"
        assert isinstance(vmin, float) and isinstance(
            vmax, float
        ), "Limits must be returned as plain floats"

    def test_explicit_vmin_preserved(self):
        """An explicit vmin is kept; only the missing vmax comes from data.

        Test scenario:
            vmin=0.0 pinned, vmax=None -> returns (0.0, data_max).
        """
        g = _make_glyph()
        g._default_options["vmin"] = 0.0
        vmin, vmax = g._resolve_limits(np.array([1.0, 5.0, 9.0]))
        assert (vmin, vmax) == (0.0, 9.0), f"Expected (0.0, 9.0), got {(vmin, vmax)}"

    def test_explicit_vmax_preserved(self):
        """An explicit vmax is kept; only the missing vmin comes from data.

        Test scenario:
            vmax=20.0 pinned, vmin=None -> returns (data_min, 20.0).
        """
        g = _make_glyph()
        g._default_options["vmax"] = 20.0
        vmin, vmax = g._resolve_limits(np.array([3.0, 5.0, 9.0]))
        assert (vmin, vmax) == (3.0, 20.0), f"Expected (3.0, 20.0), got {(vmin, vmax)}"

    def test_both_explicit_ignores_data(self):
        """When both limits are pinned the data range is not consulted.

        Test scenario:
            vmin=-1.0, vmax=1.0 -> returned verbatim even though the data
            spans a wider range.
        """
        g = _make_glyph()
        g._default_options["vmin"] = -1.0
        g._default_options["vmax"] = 1.0
        vmin, vmax = g._resolve_limits(np.array([-100.0, 100.0]))
        assert (vmin, vmax) == (-1.0, 1.0), f"Expected (-1.0, 1.0), got {(vmin, vmax)}"

    def test_nan_aware_with_partial_nans(self):
        """NaNs are ignored when computing the data range.

        Test scenario:
            A mix of NaN and finite values resolves to the finite min/max.
        """
        g = _make_glyph()
        vmin, vmax = g._resolve_limits(np.array([np.nan, 2.0, np.nan, 8.0]))
        assert (vmin, vmax) == (2.0, 8.0), f"Expected (2.0, 8.0), got {(vmin, vmax)}"

    def test_all_nan_raises_valueerror(self):
        """An all-NaN array with unpinned limits raises ValueError.

        Test scenario:
            No finite values and no explicit limits -> a clear ValueError
            rather than a downstream NaN crash.
        """
        g = _make_glyph()
        with pytest.raises(ValueError, match="no finite values") as exc:
            g._resolve_limits(np.array([np.nan, np.nan]))
        assert "vmin/vmax" in str(exc.value), f"Unexpected message: {exc.value}"

    def test_all_nan_with_explicit_limits_ok(self):
        """All-NaN data is fine when both limits are pinned explicitly.

        Test scenario:
            vmin/vmax pinned -> nanmin/nanmax are never consulted, so
            all-NaN data does not raise.
        """
        g = _make_glyph()
        g._default_options["vmin"] = 0.0
        g._default_options["vmax"] = 1.0
        vmin, vmax = g._resolve_limits(np.array([np.nan, np.nan]))
        assert (vmin, vmax) == (0.0, 1.0), f"Expected (0.0, 1.0), got {(vmin, vmax)}"


class TestPrepareScalarMapping:
    """Tests for Glyph._prepare_scalar_mapping (T0.0)."""

    def test_auto_limits_set_options_and_ticks(self):
        """Auto limits populate default_options and return matching ticks.

        Test scenario:
            vmin/vmax/ticks_spacing all unset -> resolved from data,
            ticks_spacing derived as range/10, and ticks span 0..10.
        """
        g = _make_glyph()
        g._default_options["ticks_spacing"] = None
        norm, cbar_kw, ticks = g._prepare_scalar_mapping(np.array([0.0, 5.0, 10.0]))
        assert norm is None, "Linear scale with no levels should give norm=None"
        assert g.default_options["vmin"] == 0.0, "vmin should be written back"
        assert g.default_options["vmax"] == 10.0, "vmax should be written back"
        assert (
            g.default_options["ticks_spacing"] == 1.0
        ), f"ticks_spacing should be range/10=1.0, got {g.default_options['ticks_spacing']}"
        np.testing.assert_array_almost_equal(ticks, np.arange(0.0, 11.0, 1.0))

    def test_vmin_vmax_written_back_for_get_ticks(self):
        """Resolved limits are visible to get_ticks via default_options.

        Test scenario:
            After the call, get_ticks() reproduces the same tick array,
            proving the limits were written into default_options.
        """
        g = _make_glyph()
        g._default_options["ticks_spacing"] = None
        _, _, ticks = g._prepare_scalar_mapping(np.array([2.0, 12.0]))
        np.testing.assert_array_almost_equal(ticks, g.get_ticks())

    def test_explicit_ticks_spacing_preserved(self):
        """A pinned ticks_spacing is not overwritten by the range/10 rule.

        Test scenario:
            ticks_spacing=5 stays 5 even though range/10 would be 1.0.
        """
        g = _make_glyph()
        g._default_options["ticks_spacing"] = 5.0
        g._prepare_scalar_mapping(np.array([0.0, 10.0]))
        assert g.default_options["ticks_spacing"] == 5.0, (
            f"Explicit ticks_spacing should be preserved, got "
            f"{g.default_options['ticks_spacing']}"
        )

    def test_flat_data_ticks_spacing_guard(self):
        """Flat data does not yield a zero ticks_spacing.

        Test scenario:
            All values equal -> range is 0, but the `or 1.0` guard keeps
            ticks_spacing at 1.0 so get_ticks() does not return empty.
        """
        g = _make_glyph()
        g._default_options["ticks_spacing"] = None
        _, _, ticks = g._prepare_scalar_mapping(np.array([3.0, 3.0, 3.0]))
        assert g.default_options["ticks_spacing"] == 1.0, (
            f"Flat-data spacing should be guarded to 1.0, got "
            f"{g.default_options['ticks_spacing']}"
        )
        assert len(ticks) >= 1, "Ticks should not be empty for flat data"

    def test_all_nan_raises_valueerror(self):
        """All-NaN input propagates the _resolve_limits ValueError.

        Test scenario:
            No finite values and no explicit limits -> ValueError.
        """
        g = _make_glyph()
        with pytest.raises(ValueError, match="no finite values"):
            g._prepare_scalar_mapping(np.array([np.nan, np.nan]))

    def test_levels_forwarded_into_norm(self):
        """An integer `levels` produces a BoundaryNorm via the helper.

        Test scenario:
            levels=5 under the default linear scale -> BoundaryNorm and a
            colorbar tick set sized to the levels.
        """
        g = _make_glyph()
        g._default_options["ticks_spacing"] = None
        g._default_options["levels"] = 5
        norm, cbar_kw, _ = g._prepare_scalar_mapping(np.array([0.0, 10.0]))
        assert isinstance(
            norm, mcolors.BoundaryNorm
        ), f"levels should yield a BoundaryNorm, got {type(norm)}"
        assert (
            cbar_kw["extend"] == "both"
        ), f"levels should default extend to 'both', got {cbar_kw['extend']}"

    def test_color_scale_forwarded_into_norm(self):
        """A non-linear color_scale is honoured by the helper.

        Test scenario:
            color_scale='power' -> the returned norm is a PowerNorm built
            from the resolved limits.
        """
        g = _make_glyph()
        g._default_options["ticks_spacing"] = None
        g._default_options["color_scale"] = "power"
        norm, _, _ = g._prepare_scalar_mapping(np.array([0.0, 10.0]))
        assert isinstance(
            norm, mcolors.PowerNorm
        ), f"color_scale='power' should yield a PowerNorm, got {type(norm)}"


class TestArrayGlyphUnchangedByHelper:
    """Regression: ArrayGlyph's tick output matches the shared helper (T0.0)."""

    def test_arrayglyph_ticks_match_helper_path(self):
        """A bare Glyph using the helper reproduces ArrayGlyph's ticks.

        Test scenario:
            For the same data and auto limits, the helper-driven ticks
            equal those ArrayGlyph computes for its non-robust/non-center
            path, confirming the contract is identical.
        """
        from cleopatra.array_glyph import ArrayGlyph

        data = np.array([[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]])
        ag = ArrayGlyph(data)
        ag.plot()
        array_ticks = ag.get_ticks()
        plt.close(ag.fig)

        g = _make_glyph()
        g._default_options["ticks_spacing"] = None
        _, _, helper_ticks = g._prepare_scalar_mapping(data)
        np.testing.assert_array_almost_equal(
            helper_ticks,
            array_ticks,
            err_msg="Helper ticks should match ArrayGlyph's ticks for the same data",
        )


class TestOptionKeysAndFilterKwargs:
    """Tests for `Glyph.option_keys` / `Glyph.filter_kwargs` (issue #131).

    Pre-construction introspection of a glyph's accepted option keys, and a
    filter helper that drops unknown keys so a forwarded kwargs bag can be
    used without tripping the strict `_merge_kwargs` validation.
    """

    def test_option_keys_without_instance(self):
        """`option_keys()` works on the class, with no instance built.

        Test scenario:
            The base Glyph exposes the shared style-default keys via a
            classmethod, so no construction (and no `_merge_kwargs`) is needed.
        """
        keys = Glyph.option_keys()
        assert keys == set(STYLE_DEFAULTS), "base keys should equal STYLE_DEFAULTS"
        assert "cmap" in keys, "a known style key should be present"

    @pytest.mark.parametrize(
        "import_path, class_name, const_name",
        [
            ("cleopatra.array_glyph", "ArrayGlyph", "DEFAULT_OPTIONS"),
            ("cleopatra.scatter_glyph", "ScatterGlyph", "SCATTER_DEFAULT_OPTIONS"),
            ("cleopatra.polygon_glyph", "PolygonGlyph", "POLYGON_DEFAULT_OPTIONS"),
            ("cleopatra.vector_glyph", "VectorGlyph", "VECTOR_DEFAULT_OPTIONS"),
            ("cleopatra.line_glyph", "LineGlyph", "LINE_DEFAULT_OPTIONS"),
            ("cleopatra.mesh_glyph", "MeshGlyph", "MESH_DEFAULT_OPTIONS"),
        ],
    )
    def test_subclass_keys_match_their_option_dict(
        self, import_path, class_name, const_name
    ):
        """Each glyph's `option_keys()` equals its own option-dict keys.

        Args:
            import_path: Dotted module path of the glyph.
            class_name: The glyph class to introspect.
            const_name: Name of that glyph's module-level option dict.

        Test scenario:
            The class attribute is the single source of truth, so the keys
            reported per glyph match its `*_DEFAULT_OPTIONS` constant exactly.
        """
        import importlib

        module = importlib.import_module(import_path)
        glyph_cls = getattr(module, class_name)
        const = getattr(module, const_name)
        assert glyph_cls.option_keys() == set(
            const
        ), f"{glyph_cls.__name__}.option_keys() must match {const_name}"

    def test_keys_differ_per_glyph(self):
        """Different glyphs expose different option keys.

        Test scenario:
            A polygon-specific key (`edgecolor`) is accepted by PolygonGlyph
            but not by ScatterGlyph, proving the keys are resolved per class.
        """
        from cleopatra.polygon_glyph import PolygonGlyph
        from cleopatra.scatter_glyph import ScatterGlyph

        assert "edgecolor" in PolygonGlyph.option_keys(), "polygon accepts edgecolor"
        assert "edgecolor" not in ScatterGlyph.option_keys(), "scatter does not"

    def test_filter_kwargs_keeps_accepted_drops_unknown(self):
        """`filter_kwargs` returns only accepted keys, preserving values.

        Test scenario:
            A mixed bag of known and unknown keys is filtered to just the
            known ones, with their values intact.
        """
        from cleopatra.polygon_glyph import PolygonGlyph

        raw = {"cmap": "viridis", "edgecolor": "black", "bogus": 1}
        safe = PolygonGlyph.filter_kwargs(raw)
        assert sorted(safe) == ["cmap", "edgecolor"], f"unexpected keys: {sorted(safe)}"
        assert safe["cmap"] == "viridis", "values must be preserved"

    def test_filter_kwargs_empty_returns_empty(self):
        """Filtering an empty mapping yields an empty mapping.

        Test scenario:
            Boundary case — no keys in, no keys out.
        """
        from cleopatra.scatter_glyph import ScatterGlyph

        assert ScatterGlyph.filter_kwargs({}) == {}, "empty in -> empty out"

    def test_filter_then_construct_does_not_raise(self):
        """Pre-filtering resolves the catch-22: construction then succeeds.

        Test scenario:
            Forwarding a bag with an unknown key would raise on construction;
            filtering first lets the same bag build the glyph cleanly while
            keeping the accepted styling.
        """
        import numpy as np

        from cleopatra.array_glyph import ArrayGlyph

        raw = {"cmap": "viridis", "totally_unknown": 1}
        with pytest.raises(ValueError, match="totally_unknown"):
            ArrayGlyph(np.arange(9.0).reshape(3, 3), **raw)
        safe = ArrayGlyph.filter_kwargs(raw)
        glyph = ArrayGlyph(np.arange(9.0).reshape(3, 3), **safe)
        assert glyph.default_options["cmap"] == "viridis", "accepted key must survive"

    def test_option_keys_returns_independent_set(self):
        """`option_keys()` returns a fresh set each call; mutation can't leak.

        Test scenario:
            Mutating the returned set must not corrupt the class option keys
            seen by a subsequent call (the helper builds a new set from the
            class dict rather than aliasing it).
        """
        first = Glyph.option_keys()
        first.add("totally_unknown")
        second = Glyph.option_keys()
        assert "totally_unknown" not in second, "returned set must be independent"

    def test_option_keys_matches_instance_accepted_keys(self):
        """Class-level `option_keys()` equals a built instance's option keys.

        Test scenario:
            The class attribute is the real source of truth: the keys reported
            without an instance match the keys an actual instance ends up with.
        """
        import numpy as np

        from cleopatra.array_glyph import ArrayGlyph

        glyph = ArrayGlyph(np.arange(9.0).reshape(3, 3))
        assert ArrayGlyph.option_keys() == set(
            glyph.default_options
        ), "class option_keys must match the instance's accepted keys"

    def test_filter_kwargs_does_not_mutate_input(self):
        """`filter_kwargs` leaves the caller's dict untouched and returns a copy.

        Test scenario:
            Filtering is pure — the input mapping keeps all its original keys,
            and the returned dict is a distinct object.
        """
        from cleopatra.scatter_glyph import ScatterGlyph

        raw = {"cmap": "viridis", "bogus": 1}
        safe = ScatterGlyph.filter_kwargs(raw)
        assert raw == {"cmap": "viridis", "bogus": 1}, "input must not be mutated"
        assert safe is not raw, "a new dict should be returned"

    def test_filter_kwargs_preserves_insertion_order(self):
        """`filter_kwargs` preserves the order of the accepted keys.

        Test scenario:
            Two accepted keys given in a specific order come back in that same
            order (rejected keys are dropped without reordering the rest).
        """
        from cleopatra.array_glyph import ArrayGlyph

        raw = {"vmax": 5, "bogus": 1, "vmin": 0}
        safe = ArrayGlyph.filter_kwargs(raw)
        assert list(safe) == ["vmax", "vmin"], f"order not preserved: {list(safe)}"

    def test_filter_kwargs_all_unknown_returns_empty(self):
        """A mapping of only unknown keys filters down to nothing.

        Test scenario:
            Boundary case — when no key is accepted, the result is empty.
        """
        from cleopatra.scatter_glyph import ScatterGlyph

        assert (
            ScatterGlyph.filter_kwargs({"nope": 1, "nah": 2}) == {}
        ), "all-unknown input should yield an empty dict"


class TestRootFigure:
    """Tests for the module-level `_root_figure` helper."""

    def test_returns_owning_figure_for_normal_axes(self):
        """`_root_figure(ax)` returns the Figure that owns a normal axes.

        Test scenario:
            For an axes created by `plt.subplots`, the helper resolves to that
            same figure object.
        """
        from cleopatra.glyph import _root_figure

        fig, ax = plt.subplots()
        try:
            assert _root_figure(ax) is fig, "should return the axes' owning figure"
        finally:
            plt.close(fig)

    def test_uses_root_kwarg_when_supported(self):
        """`_root_figure` passes `root=True` when the axes supports it.

        Test scenario:
            On matplotlib >= 3.10 `get_figure(root=True)` returns the top-level
            Figure; the helper must use that path. Simulated with a fake axes
            whose `get_figure` accepts `root`.
        """
        from cleopatra.glyph import _root_figure

        root_fig = object()

        class _ModernAx:
            def get_figure(self, root=False):
                return root_fig if root else "non-root"

        assert _root_figure(_ModernAx()) is root_fig, "should request the root figure"

    def test_falls_back_when_root_kwarg_unsupported(self):
        """`_root_figure` falls back to bare `get_figure()` on older matplotlib.

        Test scenario:
            When `get_figure(root=True)` raises TypeError (no `root` kwarg, as
            on the 3.8.4 floor), the helper retries without it.
        """
        from cleopatra.glyph import _root_figure

        sentinel = object()

        class _LegacyAx:
            def get_figure(self):
                return sentinel

        assert _root_figure(_LegacyAx()) is sentinel, "should fall back to get_figure()"


class TestDefaultOptionsAlias:
    """The renamed option dicts keep a backwards-compatible `DEFAULT_OPTIONS` alias."""

    def test_array_alias_is_same_object(self):
        """`array_glyph.DEFAULT_OPTIONS` aliases `ARRAY_DEFAULT_OPTIONS`.

        Test scenario:
            The public `DEFAULT_OPTIONS` name still resolves to the renamed
            constant (same object), and the class attribute / `option_keys`
            agree with it.
        """
        import cleopatra.array_glyph as ag

        assert (
            ag.DEFAULT_OPTIONS is ag.ARRAY_DEFAULT_OPTIONS
        ), "alias must be the same object"
        assert (
            ag.ArrayGlyph.DEFAULT_OPTIONS is ag.ARRAY_DEFAULT_OPTIONS
        ), "class attr mismatch"
        assert ag.ArrayGlyph.option_keys() == set(
            ag.ARRAY_DEFAULT_OPTIONS
        ), "keys mismatch"

    def test_statistical_alias_is_same_object(self):
        """`statistical_glyph.DEFAULT_OPTIONS` aliases `STATISTICAL_DEFAULT_OPTIONS`.

        Test scenario:
            The public `DEFAULT_OPTIONS` name still resolves to the renamed
            constant (same object), and the class attribute / `option_keys`
            agree with it.
        """
        import cleopatra.statistical_glyph as sg

        assert (
            sg.DEFAULT_OPTIONS is sg.STATISTICAL_DEFAULT_OPTIONS
        ), "alias must be the same object"
        assert (
            sg.StatisticalGlyph.DEFAULT_OPTIONS is sg.STATISTICAL_DEFAULT_OPTIONS
        ), "class attr mismatch"
        assert sg.StatisticalGlyph.option_keys() == set(
            sg.STATISTICAL_DEFAULT_OPTIONS
        ), "keys mismatch"


class TestSubFigureFigureResolution:
    """`_root_figure` / the mismatch warning behave correctly with SubFigures.

    Regression coverage for the review's N1: an axes living on a SubFigure must
    resolve to the top-level Figure, and passing either the root figure or the
    immediate SubFigure as `fig` must not trip the mismatch warning.
    """

    def test_root_figure_resolves_to_top_level_for_subfigure_axes(self):
        """`_root_figure(ax)` returns the root Figure, not the SubFigure.

        Test scenario:
            An axes created on a SubFigure resolves up to the owning top-level
            Figure (so a host-owned colorbar/figure handle is the real one).
        """
        from cleopatra.glyph import _root_figure

        fig = plt.figure()
        sub = fig.subfigures(1, 1)
        ax = sub.subplots()
        try:
            assert _root_figure(ax) is fig, "should climb to the top-level figure"
        finally:
            plt.close(fig)

    def test_root_figure_passed_for_subfigure_axes_does_not_warn(self):
        """Passing the root Figure with a SubFigure axes does not warn.

        Test scenario:
            The N1 case — `fig` is the top-level figure that transitively owns
            an axes on a SubFigure; this is a valid pairing and must be silent.
        """
        import warnings as _warnings

        fig = plt.figure()
        sub = fig.subfigures(1, 1)
        ax = sub.subplots()
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("error")
                g = Glyph(default_options=_make_options(), fig=fig, ax=ax)
            assert g.fig is fig, "the explicit root fig should be stored"
        finally:
            plt.close(fig)

    def test_immediate_subfigure_passed_does_not_warn(self):
        """Passing the immediate SubFigure with its own axes does not warn.

        Test scenario:
            `fig` is the SubFigure the axes is directly attached to — also a
            valid pairing, so no warning.
        """
        import warnings as _warnings

        fig = plt.figure()
        sub = fig.subfigures(1, 1)
        ax = sub.subplots()
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("error")
                g = Glyph(default_options=_make_options(), fig=sub, ax=ax)
            assert g.fig is sub, "the explicit sub-figure should be stored"
        finally:
            plt.close(fig)

    def test_unrelated_figure_with_subfigure_axes_still_warns(self):
        """An unrelated `fig` with a SubFigure axes still warns.

        Test scenario:
            The warning must still fire for a genuinely unrelated figure, even
            when the axes lives on a SubFigure.
        """
        fig = plt.figure()
        sub = fig.subfigures(1, 1)
        ax = sub.subplots()
        other = plt.figure()
        try:
            with pytest.warns(UserWarning, match="not the figure that owns"):
                Glyph(default_options=_make_options(), fig=other, ax=ax)
        finally:
            plt.close(fig)
            plt.close(other)


class TestFigureResolutionInternals:
    """Branch coverage for the matplotlib-version fallback paths of the helpers.

    On matplotlib >= 3.10 the `root=` path is always taken for real axes, so
    the legacy (< 3.10) branches are exercised here with small fakes.
    """

    def test_supports_root_false_when_signature_uninspectable(self):
        """`_get_figure_supports_root` returns False if the signature can't be read.

        Test scenario:
            A non-callable (whose signature inspection raises) must yield False
            rather than propagating the error.
        """
        from cleopatra.glyph import _get_figure_supports_root

        assert _get_figure_supports_root(object()) is False, "uninspectable -> False"

    def test_root_figure_climbs_subfigure_on_legacy_path(self):
        """The legacy path climbs a `SubFigure` to the top-level `Figure`.

        Test scenario:
            A fake axes whose `get_figure` has no `root` kwarg returns a real
            SubFigure; `_root_figure` must climb to the owning Figure.
        """
        import warnings as _warnings

        from cleopatra.glyph import _root_figure

        fig = plt.figure()
        sub = fig.subfigures(1, 1)

        class _LegacyAxOnSub:
            def get_figure(self):
                return sub

        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                assert (
                    _root_figure(_LegacyAxOnSub()) is fig
                ), "should climb to the root figure"
        finally:
            plt.close(fig)

    def test_immediate_figure_legacy_path(self):
        """`_immediate_figure` uses the bare `get_figure()` on the legacy path.

        Test scenario:
            A fake axes whose `get_figure` has no `root` kwarg returns its
            figure directly.
        """
        from cleopatra.glyph import _immediate_figure

        sentinel = object()

        class _LegacyAx:
            def get_figure(self):
                return sentinel

        assert _immediate_figure(_LegacyAx()) is sentinel, "should return get_figure()"


class TestGetTicksDegenerateRange:
    """`get_ticks` must not divide by zero on a degenerate colour range (#4)."""

    def test_zero_spacing_returns_single_tick(self):
        """`ticks_spacing == 0` yields a single tick at `vmin`.

        Test scenario:
            A constant-value field gives `vmax == vmin` and a zero spacing;
            `get_ticks` returns `[vmin]` rather than calling `np.arange` with
            a zero step.
        """
        opts = _make_options()
        opts.update({"vmin": 5.0, "vmax": 5.0, "ticks_spacing": 0.0})
        g = Glyph(default_options=opts)
        ticks = g.get_ticks()
        assert ticks.tolist() == [5.0], f"expected a single tick, got {ticks.tolist()}"

    def test_normal_range_unaffected(self):
        """A normal range still produces evenly spaced ticks.

        Test scenario:
            Regression guard — the degenerate check does not alter the
            ordinary path.
        """
        opts = _make_options()
        opts.update({"vmin": 0.0, "vmax": 10.0, "ticks_spacing": 2.0})
        g = Glyph(default_options=opts)
        assert g.get_ticks().tolist() == [
            0.0,
            2.0,
            4.0,
            6.0,
            8.0,
            10.0,
        ], "ticks changed"


class TestClearAndMarkRenderArtists:
    """Direct unit tests for the shared render-artist cleanup helpers.

    Regression coverage for review finding L1 (issue #210): these helpers
    were previously only exercised indirectly through each glyph's
    `plot()`/`animate()` integration tests.
    """

    @staticmethod
    def _dummy_artist(remove_error: type[Exception] | None = None):
        """Build a bare object with a `.remove()` that succeeds or raises `remove_error`."""

        class _Artist:
            def __init__(self):
                self.removed = False

            def remove(self):
                if remove_error is not None:
                    raise remove_error("boom")
                self.removed = True

        return _Artist()

    def test_clear_on_axes_without_marker_is_a_no_op(self):
        """No `_cleo_render_artists` marker set means nothing happens, no raise."""
        fig, ax = plt.subplots()
        try:
            _clear_prior_render_artists(ax)
            assert getattr(ax, "_cleo_render_artists", None) is None
        finally:
            plt.close(fig)

    def test_mark_stores_artists_and_filters_none(self):
        """`_mark_render_artists` stores the given artists, dropping `None` entries."""
        fig, ax = plt.subplots()
        try:
            a1, a2 = self._dummy_artist(), self._dummy_artist()
            _mark_render_artists(ax, a1, None, a2)
            assert ax._cleo_render_artists == [a1, a2]
        finally:
            plt.close(fig)

    def test_clear_removes_marked_artists_and_resets_marker(self):
        """`_clear_prior_render_artists` removes every marked artist and clears the marker."""
        fig, ax = plt.subplots()
        try:
            a1, a2 = self._dummy_artist(), self._dummy_artist()
            _mark_render_artists(ax, a1, a2)
            _clear_prior_render_artists(ax)
            assert a1.removed and a2.removed, "both artists must be removed"
            assert ax._cleo_render_artists is None
        finally:
            plt.close(fig)

    @pytest.mark.parametrize("error", [KeyError, NotImplementedError, AttributeError])
    def test_clear_tolerates_already_removed_artist(self, error):
        """A `.remove()` failure from an already-partially-removed artist doesn't stop cleanup.

        Test scenario:
            Locks in H1's fix: matplotlib raises a different exception type
            depending on how much of the artist is already detached
            (`KeyError` for a colorbar axes off the figure's axes stack,
            `NotImplementedError` for any other artist already detached,
            `AttributeError` for a colorbar whose mappable was detached out
            from under it). All three must be swallowed, and the other
            marked artist must still be removed.
        """
        fig, ax = plt.subplots()
        try:
            bad = self._dummy_artist(remove_error=error)
            good = self._dummy_artist()
            _mark_render_artists(ax, bad, good)
            _clear_prior_render_artists(ax)
            assert good.removed, "the other artist must still be removed"
            assert ax._cleo_render_artists is None
        finally:
            plt.close(fig)

    def test_clear_propagates_unexpected_exception_types(self):
        """An exception outside the tolerated set is not swallowed.

        Test scenario:
            Guards the narrow scope of the tolerated exception set --
            `_clear_prior_render_artists` must not turn into a blanket
            `except Exception: pass` that would hide genuine bugs.
        """
        fig, ax = plt.subplots()
        try:
            _mark_render_artists(ax, self._dummy_artist(remove_error=RuntimeError))
            with pytest.raises(RuntimeError):
                _clear_prior_render_artists(ax)
        finally:
            plt.close(fig)
