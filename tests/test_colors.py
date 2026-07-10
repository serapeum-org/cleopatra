from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
from matplotlib.image import AxesImage

from cleopatra.colors import (
    CAMS_AOD_COLORMAPS,
    HAZE_COLORMAPS,
    DATA_STYLES,
    Colors,
    alpha_scaled_image,
    alpha_scaled_mesh,
    apply_data_style,
)


class TestHazeColormaps:
    """Tests for the `HAZE_COLORMAPS` preset constant."""

    def test_has_organic_matter_and_dust(self):
        """The two documented preset names are present, and only those two."""
        assert set(HAZE_COLORMAPS) == {"organic_matter", "dust"}, (
            f"unexpected preset names: {set(HAZE_COLORMAPS)}"
        )

    @pytest.mark.parametrize("name", ["organic_matter", "dust"])
    def test_entries_are_colormaps(self, name):
        """Each entry is a ready `Colormap`, not a name string or dict."""
        assert isinstance(HAZE_COLORMAPS[name], Colormap), (
            f"{name} is not a Colormap: {type(HAZE_COLORMAPS[name])}"
        )

    @pytest.mark.parametrize("name", ["organic_matter", "dust"])
    def test_starts_white_at_zero(self, name):
        """Every haze colormap starts at opaque white for value 0.0."""
        assert HAZE_COLORMAPS[name](0.0) == (1.0, 1.0, 1.0, 1.0), (
            f"{name}(0.0) should be white, got {HAZE_COLORMAPS[name](0.0)}"
        )

    def test_dust_ends_dark_brown(self):
        """The dust colormap saturates to a dark brown at value 1.0."""
        r, g, b, a = HAZE_COLORMAPS["dust"](1.0)
        assert a == 1.0, "alpha should be opaque"
        assert r > g > b, f"dust top stop should be brown-toned, got rgb=({r}, {g}, {b})"

    def test_organic_matter_ends_purple(self):
        """The organic_matter colormap saturates to a deep purple at value 1.0."""
        r, g, b, a = HAZE_COLORMAPS["organic_matter"](1.0)
        assert a == 1.0, "alpha should be opaque"
        assert r > g and b > g, (
            f"organic_matter top stop should be purple-toned, got rgb=({r}, {g}, {b})"
        )


class TestCamsAodColormaps:
    """Tests for the `CAMS_AOD_COLORMAPS` preset constant (official CAMS AOD scales)."""

    NAMES = ["blue_yellow_red", "blue_yellow_red_brown", "blue_red", "oranges"]

    def test_has_the_four_documented_palettes(self):
        """The four documented preset names are present, and only those four."""
        assert set(CAMS_AOD_COLORMAPS) == set(self.NAMES), (
            f"unexpected preset names: {set(CAMS_AOD_COLORMAPS)}"
        )

    @pytest.mark.parametrize("name", NAMES)
    def test_entries_are_colormaps(self, name):
        """Each entry is a ready `Colormap`, not a name string or dict."""
        assert isinstance(CAMS_AOD_COLORMAPS[name], Colormap), (
            f"{name} is not a Colormap: {type(CAMS_AOD_COLORMAPS[name])}"
        )

    @pytest.mark.parametrize("name", NAMES)
    def test_are_fully_opaque(self, name):
        """The vendored colormaps are pure colour -- opaque at both ends.

        Magics' `sh_Oranges_aod` ramps opacity with value, but that alpha is
        intentionally handled by cleopatra's separate opacity axis, not baked
        into the colormap (see the `CAMS_AOD_COLORMAPS` docstring).
        """
        assert CAMS_AOD_COLORMAPS[name](0.0)[3] == 1.0, f"{name}(0.0) should be opaque"
        assert CAMS_AOD_COLORMAPS[name](1.0)[3] == 1.0, f"{name}(1.0) should be opaque"

    @pytest.mark.parametrize("name", ["blue_yellow_red", "blue_yellow_red_brown", "blue_red"])
    def test_blue_low_end(self, name):
        """The blue-to-red AOD scales start blue-dominant at value 0.0."""
        r, g, b, _ = CAMS_AOD_COLORMAPS[name](0.0)
        assert b >= r, f"{name}(0.0) should be blue-toned, got rgb=({r}, {g}, {b})"

    @pytest.mark.parametrize("name", ["blue_yellow_red", "blue_red"])
    def test_red_high_end(self, name):
        """The red-topped AOD scales saturate red-dominant at value 1.0."""
        r, g, b, _ = CAMS_AOD_COLORMAPS[name](1.0)
        assert r > g and r > b, f"{name}(1.0) should be red-toned, got rgb=({r}, {g}, {b})"

    def test_oranges_low_end_is_near_white(self):
        """The `oranges` scale starts near white (its Magics form fades in via alpha)."""
        r, g, b, _ = CAMS_AOD_COLORMAPS["oranges"](0.0)
        assert min(r, g, b) > 0.85, f"oranges(0.0) should be near white, got rgb=({r}, {g}, {b})"


class TestAlphaScaledImage:
    """Tests for `alpha_scaled_image`."""

    @pytest.fixture
    def ax(self):
        """A fresh Axes on the Agg backend, closed after the test."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_returns_axes_image(self, ax):
        """The call returns an `AxesImage` attached to the given axes."""
        img = alpha_scaled_image(ax, np.array([[0.0, 1.0]]), "viridis")
        assert isinstance(img, AxesImage), f"expected AxesImage, got {type(img)}"
        assert img in ax.images, "image should be attached to the given axes"

    def test_alpha_matches_normalised_value(self, ax):
        """Alpha equals the (default-normalised) data value, not the colour."""
        data = np.array([[0.0, 0.25, 1.0]])
        img = alpha_scaled_image(ax, data, "viridis")
        alpha = img.get_array()[..., 3]
        np.testing.assert_allclose(
            alpha, [[0.0, 0.25, 1.0]], err_msg=f"unexpected alpha channel: {alpha}"
        )

    def test_nan_is_always_fully_transparent(self, ax):
        """A NaN cell is alpha=0 even under an alpha_norm that would not zero it."""
        data = np.array([[np.nan, 5.0]])
        img = alpha_scaled_image(
            ax, data, "viridis", alpha_norm=Normalize(vmin=0.0, vmax=5.0, clip=False)
        )
        alpha = img.get_array()[..., 3]
        assert alpha[0, 0] == 0.0, f"NaN pixel should be transparent, got alpha={alpha[0, 0]}"
        assert alpha[0, 1] == 1.0, f"finite max value should be opaque, got {alpha[0, 1]}"

    def test_decoupled_alpha_norm(self, ax):
        """A separate `alpha_norm` drives opacity independently of `norm`."""
        data = np.array([[0.0, 10.0]])
        img = alpha_scaled_image(
            ax,
            data,
            "viridis",
            norm=Normalize(vmin=0.0, vmax=10.0),
            alpha_norm=Normalize(vmin=0.0, vmax=20.0),
        )
        alpha = img.get_array()[..., 3]
        np.testing.assert_allclose(
            alpha, [[0.0, 0.5]], err_msg=f"alpha_norm override not applied: {alpha}"
        )

    def test_cmap_accepts_colormap_object(self, ax):
        """A `Colormap` instance (not just a name string) is accepted directly."""
        img = alpha_scaled_image(ax, np.array([[0.0, 1.0]]), HAZE_COLORMAPS["dust"])
        rgb = img.get_array()[0, 1, :3]
        np.testing.assert_allclose(
            rgb, [0.165, 0.031, 0.0], atol=0.01, err_msg=f"unexpected top-stop colour: {rgb}"
        )

    def test_constant_alpha_makes_field_opaque(self, ax):
        """`constant_alpha=1.0` draws every finite cell opaque, ignoring value."""
        data = np.array([[0.0, 0.5, 1.0]])
        img = alpha_scaled_image(ax, data, "viridis", constant_alpha=1.0)
        alpha = img.get_array()[..., 3]
        np.testing.assert_allclose(
            alpha, [[1.0, 1.0, 1.0]], err_msg=f"expected all-opaque, got {alpha}"
        )

    def test_constant_alpha_keeps_nan_transparent(self, ax):
        """`constant_alpha` still leaves NaN cells fully transparent."""
        data = np.array([[np.nan, 1.0]])
        img = alpha_scaled_image(ax, data, "viridis", constant_alpha=1.0)
        alpha = img.get_array()[..., 3]
        assert alpha[0, 0] == 0.0, f"NaN should stay transparent, got {alpha[0, 0]}"
        assert alpha[0, 1] == 1.0, f"finite cell should be opaque, got {alpha[0, 1]}"

    def test_constant_alpha_is_clipped(self, ax):
        """An out-of-range `constant_alpha` is clipped into [0, 1]."""
        img = alpha_scaled_image(ax, np.array([[0.0, 1.0]]), "viridis", constant_alpha=2.5)
        assert img.get_array()[0, 1, 3] == 1.0, "constant_alpha > 1 should clip to 1.0"

    def test_non_2d_data_raises(self, ax):
        """A 1D (or higher-dimensional) `data` array raises `ValueError`."""
        with pytest.raises(ValueError, match="2-dimensional"):
            alpha_scaled_image(ax, np.array([0.0, 1.0]), "viridis")

    def test_forwards_imshow_kwargs(self, ax):
        """Extra keyword arguments (e.g. `zorder`) reach the underlying `imshow`."""
        img = alpha_scaled_image(ax, np.array([[0.0, 1.0]]), "viridis", zorder=7)
        assert img.get_zorder() == 7, f"zorder not forwarded, got {img.get_zorder()}"


class TestAlphaScaledMesh:
    """Tests for `alpha_scaled_mesh` (the curvilinear-grid counterpart)."""

    @pytest.fixture
    def ax(self):
        """A fresh Axes on the Agg backend, closed after the test."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    @pytest.fixture
    def xy(self):
        """A 3x3 corner grid for a 2x2 quad mesh (shading='flat' convention)."""
        return np.meshgrid(np.arange(3), np.arange(3))

    def test_returns_quadmesh_attached_to_ax(self, ax, xy):
        """The call returns a `QuadMesh` registered as a collection on `ax`."""
        x, y = xy
        data = np.array([[0.0, 1.0], [0.5, 0.25]])
        mesh = alpha_scaled_mesh(ax, x, y, data, "viridis", shading="flat")
        assert isinstance(mesh, QuadMesh), f"expected QuadMesh, got {type(mesh)}"
        assert mesh in ax.collections, "mesh should be attached to ax"

    def test_facecolor_alpha_matches_normalised_value(self, ax, xy):
        """Per-quad alpha in `facecolor` equals the (default-normalised) value."""
        x, y = xy
        data = np.array([[0.0, 1.0], [0.5, 0.25]])
        mesh = alpha_scaled_mesh(ax, x, y, data, "viridis", shading="flat")
        alpha = mesh.get_facecolor()[:, 3]
        np.testing.assert_allclose(
            alpha, [0.0, 1.0, 0.5, 0.25], err_msg=f"unexpected facecolor alpha: {alpha}"
        )

    def test_nan_cell_is_fully_transparent(self, ax, xy):
        """A NaN cell renders with alpha=0 in the facecolor array."""
        x, y = xy
        data = np.array([[np.nan, 1.0], [0.5, 0.25]])
        mesh = alpha_scaled_mesh(ax, x, y, data, "viridis", shading="flat")
        alpha = mesh.get_facecolor()[:, 3]
        assert alpha[0] == 0.0, f"NaN cell should be transparent, got alpha={alpha[0]}"

    def test_array_cleared_so_cmap_norm_do_not_override_facecolor(self, ax, xy):
        """`set_array(None)` is applied so the mesh renders the explicit facecolor."""
        x, y = xy
        data = np.array([[0.0, 1.0], [0.5, 0.25]])
        mesh = alpha_scaled_mesh(ax, x, y, data, "viridis", shading="flat")
        assert mesh.get_array() is None, "mesh array should be cleared after colouring"

    def test_non_2d_data_raises(self, ax, xy):
        """A 1D `data` array raises `ValueError`."""
        x, y = xy
        with pytest.raises(ValueError, match="2-dimensional"):
            alpha_scaled_mesh(ax, x, y, np.array([0.0, 1.0]), "viridis")

    def test_default_shading_is_auto(self, ax):
        """With no explicit `shading`, same-shape x/y/data does not raise."""
        x, y = np.meshgrid(np.arange(2), np.arange(2))
        data = np.array([[0.0, 1.0], [0.5, 0.25]])
        mesh = alpha_scaled_mesh(ax, x, y, data, "viridis")
        assert mesh in ax.collections, "default shading should still produce a mesh"


class TestApplyDataStyle:
    """Tests for `apply_data_style` and the `DATA_STYLES` registry."""

    @pytest.fixture
    def ax(self):
        """A fresh Axes on the Agg backend, closed after the test."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_haze_preset_has_both_layers(self):
        """The registered 'haze' preset defines exactly organic_matter and dust."""
        assert set(DATA_STYLES["haze"]) == {"organic_matter", "dust"}, (
            f"unexpected haze layers: {set(DATA_STYLES['haze'])}"
        )

    @pytest.mark.parametrize("layer", ["organic_matter", "dust"])
    def test_haze_layers_declare_decoupled_alpha(self, layer):
        """Every 'haze' layer sets a narrower alpha_vmin/alpha_vmax than its colour vmin/vmax."""
        cfg = DATA_STYLES["haze"][layer]
        assert cfg["alpha_vmin"] > cfg["vmin"], f"{layer}: alpha_vmin should be > vmin"
        assert cfg["alpha_vmax"] < cfg["vmax"], f"{layer}: alpha_vmax should be < vmax"

    def test_haze_alpha_saturates_before_color_range_ends(self, ax):
        """A mid-range 'haze' value is already fully opaque, unlike a shared-curve style.

        Test scenario:
            With alpha_vmin=0.1/alpha_vmax=0.5 (the 'haze' dust preset), a data
            value of 0.5 should be fully opaque (alpha=1.0) even though it is
            only the midpoint of the 0.0-1.0 *colour* range -- this decoupling
            is what produces the bright, opaque "flame" rim at moderate density
            instead of a value that would still be half-transparent under a
            single shared norm.
        """
        images = apply_data_style(ax, {"dust": np.array([[0.5, 1.0]])})
        alpha = images["dust"].get_array()[..., 3]
        assert alpha[0, 0] == 1.0, f"expected fully opaque at data=0.5, got alpha={alpha[0, 0]}"

    def test_cams_aod_preset_has_single_aod_layer(self):
        """The registered 'cams_aod' preset defines exactly one 'aod' layer."""
        assert set(DATA_STYLES["cams_aod"]) == {"aod"}, (
            f"unexpected cams_aod layers: {set(DATA_STYLES['cams_aod'])}"
        )

    def test_cams_aod_uses_official_palette(self):
        """The 'cams_aod' layer uses the canonical CAMS_AOD_COLORMAPS scale, not a haze map."""
        assert (
            DATA_STYLES["cams_aod"]["aod"]["cmap"]
            is CAMS_AOD_COLORMAPS["blue_yellow_red"]
        ), "cams_aod should reuse the official CAMS AOD colormap object"

    def test_cams_aod_declares_no_decoupled_alpha(self):
        """Unlike 'haze', 'cams_aod' sets no alpha_vmin/alpha_vmax (opacity tracks colour)."""
        cfg = DATA_STYLES["cams_aod"]["aod"]
        assert "alpha_vmin" not in cfg and "alpha_vmax" not in cfg, (
            f"cams_aod should not decouple alpha, got {cfg}"
        )

    def test_cams_aod_alpha_tracks_value_linearly(self, ax):
        """'cams_aod' opacity fades in with AOD: transparent at ~0, opaque at the top.

        Test scenario:
            With no alpha_vmin/alpha_vmax, alpha follows the same 0.0-1.0 norm
            as colour, so an AOD field renders transparent where it is ~0 and
            opaque red where it is high -- the natural overlay behaviour, and
            the deliberate contrast with 'haze''s decoupled glowing rim.
        """
        images = apply_data_style(ax, {"aod": np.array([[0.0, 1.0]])}, style="cams_aod")
        alpha = images["aod"].get_array()[..., 3]
        assert alpha[0, 0] == 0.0, f"AOD 0.0 should be transparent, got {alpha[0, 0]}"
        assert alpha[0, 1] == 1.0, f"AOD 1.0 should be opaque, got {alpha[0, 1]}"

    CLIMATE_PRESETS = [
        "temperature", "elevation", "vegetation", "wind_speed", "anomaly", "precipitation",
    ]

    @pytest.mark.parametrize("style", CLIMATE_PRESETS)
    def test_climate_preset_is_registered_single_layer(self, style):
        """Each climate/GIS preset is registered with one same-named layer."""
        assert style in DATA_STYLES, f"{style} missing from DATA_STYLES"
        assert set(DATA_STYLES[style]) == {style}, (
            f"{style} should have one '{style}' layer, got {set(DATA_STYLES[style])}"
        )

    @pytest.mark.parametrize(
        "style", ["temperature", "elevation", "vegetation", "wind_speed", "anomaly"]
    )
    def test_opaque_presets_fill_the_field(self, ax, style):
        """The opaque presets draw every finite cell at full opacity, NaN transparent."""
        images = apply_data_style(
            ax, {style: np.array([[0.0, 1.0], [np.nan, 0.5]])}, style=style
        )
        alpha = images[style].get_array()[..., 3]
        assert alpha[0, 0] == alpha[0, 1] == alpha[1, 1] == 1.0, (
            f"{style} finite cells should be opaque, got {alpha}"
        )
        assert alpha[1, 0] == 0.0, f"{style} NaN cell should be transparent, got {alpha[1, 0]}"

    def test_auto_range_uses_data_min_max(self, ax):
        """A preset without vmin/vmax auto-ranges the colour norm to the data.

        Test scenario:
            The lowest data value maps to the colormap's start and the highest
            to its end, proving the norm resolved to the field's own [min, max]
            rather than a hard-coded 0-1.
        """
        cmap = plt.get_cmap("RdYlBu_r")
        images = apply_data_style(
            ax, {"temperature": np.array([[10.0, 30.0]])}, style="temperature"
        )
        rgba = images["temperature"].get_array()
        np.testing.assert_allclose(
            rgba[0, 0, :3], cmap(0.0)[:3], atol=1e-6,
            err_msg="min value should map to the colormap start",
        )
        np.testing.assert_allclose(
            rgba[0, 1, :3], cmap(1.0)[:3], atol=1e-6,
            err_msg="max value should map to the colormap end",
        )

    def test_flat_field_avoids_degenerate_norm(self, ax):
        """A constant field (min == max) renders without a zero-width norm error."""
        images = apply_data_style(
            ax, {"temperature": np.full((2, 2), 15.0)}, style="temperature"
        )
        assert images["temperature"].get_array()[..., 3].min() == 1.0, (
            "a flat opaque field should still draw fully opaque"
        )

    def test_diverging_center_puts_zero_at_midpoint(self, ax):
        """'anomaly' centres 0 on the colormap midpoint, even for asymmetric data.

        Test scenario:
            With center=0 and data spanning -1..4, the symmetric range is
            [-4, 4], so the value 0.0 lands on the colormap's exact midpoint
            (near-white for RdBu_r) regardless of the data being lopsided.
        """
        cmap = plt.get_cmap("RdBu_r")
        images = apply_data_style(
            ax, {"anomaly": np.array([[-1.0, 4.0, 0.0]])}, style="anomaly"
        )
        rgba = images["anomaly"].get_array()
        np.testing.assert_allclose(
            rgba[0, 2, :3], cmap(0.5)[:3], atol=0.02,
            err_msg="the 0.0 cell should map to the colormap midpoint",
        )

    def test_precipitation_overlay_is_transparent_when_dry(self, ax):
        """'precipitation' fades to transparent where the value is ~0 (overlay behaviour)."""
        images = apply_data_style(
            ax, {"precipitation": np.array([[0.0, 50.0]])}, style="precipitation"
        )
        alpha = images["precipitation"].get_array()[..., 3]
        assert alpha[0, 0] == 0.0, f"dry cell should be transparent, got {alpha[0, 0]}"
        assert alpha[0, 1] == 1.0, f"wettest cell should be opaque, got {alpha[0, 1]}"

    def test_constant_alpha_and_decoupled_alpha_are_mutually_exclusive(self, ax, monkeypatch):
        """A preset combining a constant 'alpha' with alpha_vmin/vmax raises ValueError."""
        import cleopatra.colors as colors_mod

        bad = {
            "bad": {
                "x": {
                    "cmap": "viridis", "label": "X",
                    "alpha": 1.0, "alpha_vmin": 0.1, "alpha_vmax": 0.5,
                }
            }
        }
        monkeypatch.setattr(colors_mod, "DATA_STYLES", bad)
        with pytest.raises(ValueError, match="mutually exclusive"):
            apply_data_style(ax, {"x": np.array([[0.0, 1.0]])}, style="bad")

    def test_custom_style_without_alpha_keys_uses_shared_norm(self, ax, monkeypatch):
        """A custom style lacking alpha_vmin/alpha_vmax falls back to sharing the colour norm.

        Test scenario:
            Backward compatibility: a caller-registered style dict with only
            cmap/label/vmin/vmax (no alpha_vmin/alpha_vmax) must behave exactly
            as before -- alpha tracks the same norm as colour.
        """
        import cleopatra.colors as colors_mod

        custom_styles = {
            "plain": {"dust": {"cmap": "viridis", "label": "Plain", "vmin": 0.0, "vmax": 1.0}}
        }
        monkeypatch.setattr(colors_mod, "DATA_STYLES", custom_styles)
        images = apply_data_style(ax, {"dust": np.array([[0.5, 1.0]])}, style="plain")
        alpha = images["dust"].get_array()[..., 3]
        assert alpha[0, 0] == 0.5, (
            f"without alpha_vmin/vmax, alpha should equal the colour norm (0.5), got {alpha[0, 0]}"
        )

    def test_draws_one_image_per_layer(self, ax):
        """Each key in `layers` produces one returned `AxesImage`, drawn on `ax`."""
        layers = {"dust": np.array([[0.0, 1.0]]), "organic_matter": np.array([[0.2, 0.8]])}
        images = apply_data_style(ax, layers)
        assert set(images) == {"dust", "organic_matter"}, f"unexpected keys: {set(images)}"
        for img in images.values():
            assert isinstance(img, AxesImage), f"expected AxesImage, got {type(img)}"
            assert img in ax.images, "image should be drawn on ax"

    def test_uses_the_layer_specific_colormap(self, ax):
        """Each layer is drawn with its own DATA_STYLES colormap, not a shared one."""
        images = apply_data_style(ax, {"dust": np.array([[0.0, 1.0]])})
        top_rgb = images["dust"].get_array()[0, 1, :3]
        expected = HAZE_COLORMAPS["dust"](1.0)[:3]
        np.testing.assert_allclose(
            top_rgb, expected, atol=1e-6, err_msg="dust layer used the wrong colormap"
        )

    def test_legend_true_attaches_one_swatch_per_layer(self, ax):
        """`legend=True` (the default) attaches one swatch legend per layer."""
        apply_data_style(ax, {"dust": np.array([[0.0, 1.0]]), "organic_matter": np.array([[0.0, 1.0]])})
        assert len(ax.child_axes) == 2, f"expected 2 swatch legends, got {len(ax.child_axes)}"

    def test_legend_false_attaches_no_swatch(self, ax):
        """`legend=False` draws the layers without any swatch legend."""
        apply_data_style(ax, {"dust": np.array([[0.0, 1.0]])}, legend=False)
        assert ax.child_axes == [], f"expected no swatch legends, got {ax.child_axes}"

    def test_partial_layer_subset_is_allowed(self, ax):
        """Passing only one of the preset's layers draws just that one."""
        images = apply_data_style(ax, {"dust": np.array([[0.0, 1.0]])})
        assert list(images) == ["dust"], f"expected only 'dust', got {list(images)}"

    def test_unknown_style_raises_key_error(self, ax):
        """An unregistered `style` name raises `KeyError` before drawing anything."""
        with pytest.raises(KeyError, match="Unknown data style"):
            apply_data_style(ax, {"dust": np.array([[0.0, 1.0]])}, style="not-a-style")
        assert len(ax.images) == 0, "nothing should be drawn when style is invalid"

    def test_unknown_layer_name_raises_key_error(self, ax):
        """A layer name the style doesn't define raises `KeyError`, nothing drawn."""
        with pytest.raises(KeyError, match="smoke"):
            apply_data_style(ax, {"smoke": np.array([[0.0, 1.0]])})
        assert len(ax.images) == 0, "nothing should be drawn when a layer is unknown"

    def test_explicit_legend_bounds_are_used(self, ax):
        """Explicit `legend_bounds` override the auto-stacked default position."""
        apply_data_style(
            ax,
            {"dust": np.array([[0.0, 1.0]])},
            legend_bounds=[(0.5, 0.5, 0.2, 0.05)],
        )
        assert ax.child_axes[0].get_position().bounds is not None, (
            "swatch should have a position derived from the explicit bounds"
        )

    def test_forwards_alpha_scaled_image_kwargs(self, ax):
        """Extra kwargs (e.g. `zorder`) reach the underlying `alpha_scaled_image`."""
        images = apply_data_style(ax, {"dust": np.array([[0.0, 1.0]])}, zorder=5)
        assert images["dust"].get_zorder() == 5, "zorder not forwarded to alpha_scaled_image"

    def test_x_y_dispatches_to_alpha_scaled_mesh(self, ax):
        """Passing `x`/`y` renders every layer as a `QuadMesh`, not an `AxesImage`."""
        x, y = np.meshgrid(np.arange(3), np.arange(3))
        images = apply_data_style(
            ax,
            {"dust": np.array([[0.0, 1.0], [0.5, 0.25]])},
            x=x,
            y=y,
            shading="flat",
        )
        assert isinstance(images["dust"], QuadMesh), (
            f"expected QuadMesh with x/y given, got {type(images['dust'])}"
        )

    def test_without_x_y_uses_alpha_scaled_image(self, ax):
        """With no `x`/`y`, layers render as `AxesImage` (the default path)."""
        images = apply_data_style(ax, {"dust": np.array([[0.0, 1.0]])})
        assert isinstance(images["dust"], AxesImage), (
            f"expected AxesImage without x/y, got {type(images['dust'])}"
        )

    @pytest.mark.parametrize("kwargs", [{"x": [[0.0, 1.0]]}, {"y": [[0.0, 1.0]]}])
    def test_only_one_of_x_y_raises(self, ax, kwargs):
        """Passing only `x` or only `y` raises `ValueError`, not a silent fallback.

        Args:
            kwargs: Either `{"x": ...}` or `{"y": ...}` alone.

        Test scenario:
            A caller who mis-destructures apply_projection_style's 3-tuple
            (e.g. passing only `x`) must get a clear error instead of
            silently falling back to the flat imshow path.
        """
        with pytest.raises(ValueError, match="x and y must be given together"):
            apply_data_style(ax, {"dust": np.array([[0.0, 1.0]])}, **kwargs)


class TestCreateColors:
    def test_create_from_hex(self):
        """test_create_colors_object."""
        hex_number = "ff0000"
        color = Colors(hex_number)
        assert color._color_value == [hex_number]

    def test_create_from_rgb(self):
        """test_create_colors_object."""
        rgb_color = (128, 51, 204)
        color = Colors(rgb_color)
        assert color._color_value == [rgb_color]

    def test_create_from_image(self, color_ramp_image: str):
        colors = Colors.create_from_image(color_ramp_image)
        assert isinstance(colors.color_value, list)
        assert len(colors.color_value) == 2713
        with pytest.raises(FileNotFoundError):
            Colors.create_from_image("color_ramp_image")

    def test_create_from_image_accepts_pathlib_path(self, color_ramp_image: str):
        """`create_from_image` accepts a `pathlib.Path`, not just `str` (issue #180)."""
        colors = Colors.create_from_image(Path(color_ramp_image))
        assert isinstance(colors.color_value, list)
        assert len(colors.color_value) == 2713

    def test_create_from_image_missing_pathlib_path_raises(self, tmp_path):
        """A missing `pathlib.Path` raises `FileNotFoundError` (widened type, error branch)."""
        with pytest.raises(FileNotFoundError):
            Colors.create_from_image(tmp_path / "does-not-exist.png")

    def test_raise_error(self, color_ramp_image: str):
        with pytest.raises(ValueError):
            Colors(11)


class TestColorRamp:
    def test_create_color_ramp(self, color_ramp_image: str):
        colors = Colors.create_from_image(color_ramp_image)
        color_ramp = colors.get_color_map()
        assert isinstance(color_ramp, LinearSegmentedColormap)


def test_get_type():
    """test_create_colors_object."""
    mixed_color = [(128, 51, 204), "#23a9dd", (0.5, 0.2, 0.8)]
    color = Colors(mixed_color)
    color_types = color.get_type()
    assert color_types == ["rgb", "hex", "rgb-normalized"]


def test_is_valid_rgb_norm_255():
    """test_create_colors_object."""
    rgb_color = (128, 51, 204)
    color = Colors(rgb_color)
    assert color._is_valid_rgb_255(rgb_color) is True
    rgb_color = (0.5, 0.2, 0.8)
    color = Colors(rgb_color)
    assert color._is_valid_rgb_norm(rgb_color) is True


def test_is_valid_rgb():
    """test_create_colors_object."""
    rgb_color = [(128, 51, 204), (0.5, 0.2, 0.8)]
    color = Colors(rgb_color)
    assert all(color.is_valid_rgb())


def test_is_valid_hex():
    """test_create_colors_object."""
    hex_number = ["ff0000", "#23a9dd", (128, 51, 204), (0.5, 0.2, 0.8)]
    color = Colors(hex_number)
    valid = color.is_valid_hex()
    assert valid == [False, True, False, False]


def test_to_rgb():
    """test_create_colors_object."""
    hex_number = ["#ff0000", "#23a9dd", (0.5, 0.2, 0.8), (35, 169, 221)]
    color = Colors(hex_number)
    rgb_scale_1 = color.to_rgb(normalized=True)
    assert rgb_scale_1 == [
        (1.0, 0.0, 0.0),
        (0.13725490196078433, 0.6627450980392157, 0.8666666666666667),
        (0.5, 0.2, 0.8),
        (0.13725490196078433, 0.6627450980392157, 0.8666666666666667),
    ]
    rgb_scale_255 = color.to_rgb(normalized=False)
    assert rgb_scale_255 == [
        (255, 0, 0),
        (35, 169, 221),
        (127, 51, 204),
        (35, 169, 221),
    ]


def test_to_hex():
    """test_create_colors_object."""
    mixed_color = [(128, 51, 204), "#23a9dd", (0.5, 0.2, 0.8)]
    color = Colors(mixed_color)
    hex_colors = color.to_hex()
    assert hex_colors == ["#8033cc", "#23a9dd", "#8033cc"]
