import importlib.resources
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.colors import (
    BoundaryNorm,
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    LogNorm,
    Normalize,
    SymLogNorm,
    to_hex,
    to_rgb,
)
from matplotlib.image import AxesImage

from cleopatra.styling.colors import (
    CAMS_AOD_COLORMAPS,
    DATA_STYLES,
    FLAME_COLORMAPS,
    HAZE_COLORMAPS,
    Colors,
    _category_boundaries,
    _load_presets,
    _resolve_style_norm,
    alpha_scaled_image,
    alpha_scaled_mesh,
    apply_data_style,
)
from cleopatra.styling.perceptual import perceptual_uniformity


class TestHazeColormaps:
    """Tests for the `HAZE_COLORMAPS` preset constant."""

    def test_has_organic_matter_and_dust(self):
        """The two documented preset names are present, and only those two."""
        assert set(HAZE_COLORMAPS) == {
            "organic_matter",
            "dust",
        }, f"unexpected preset names: {set(HAZE_COLORMAPS)}"

    @pytest.mark.parametrize("name", ["organic_matter", "dust"])
    def test_entries_are_colormaps(self, name):
        """Each entry is a ready `Colormap`, not a name string or dict."""
        assert isinstance(HAZE_COLORMAPS[name], Colormap), (
            f"{name} is not a Colormap: {type(HAZE_COLORMAPS[name])}"
        )

    @pytest.mark.parametrize("name", ["organic_matter", "dust"])
    def test_starts_white_at_zero(self, name):
        """Every haze colormap starts at opaque white for value 0.0."""
        assert HAZE_COLORMAPS[name](0.0) == (
            1.0,
            1.0,
            1.0,
            1.0,
        ), f"{name}(0.0) should be white, got {HAZE_COLORMAPS[name](0.0)}"

    def test_dust_ends_dark_brown(self):
        """The dust colormap saturates to a dark brown at value 1.0."""
        r, g, b, a = HAZE_COLORMAPS["dust"](1.0)
        assert a == 1.0, "alpha should be opaque"
        assert r > g > b, (
            f"dust top stop should be brown-toned, got rgb=({r}, {g}, {b})"
        )

    def test_organic_matter_ends_purple(self):
        """The organic_matter colormap saturates to a deep purple at value 1.0."""
        r, g, b, a = HAZE_COLORMAPS["organic_matter"](1.0)
        assert a == 1.0, "alpha should be opaque"
        assert r > g, f'organic_matter top stop should be purple-toned, got rgb=({r}, {g}, {b})'
        assert b > g, f'organic_matter top stop should be purple-toned, got rgb=({r}, {g}, {b})'


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

    @pytest.mark.parametrize(
        "name", ["blue_yellow_red", "blue_yellow_red_brown", "blue_red"]
    )
    def test_blue_low_end(self, name):
        """The blue-to-red AOD scales start blue-dominant at value 0.0."""
        r, g, b, _ = CAMS_AOD_COLORMAPS[name](0.0)
        assert b >= r, f"{name}(0.0) should be blue-toned, got rgb=({r}, {g}, {b})"

    @pytest.mark.parametrize("name", ["blue_yellow_red", "blue_red"])
    def test_red_high_end(self, name):
        """The red-topped AOD scales saturate red-dominant at value 1.0."""
        r, g, b, _ = CAMS_AOD_COLORMAPS[name](1.0)
        assert r > g, f'{name}(1.0) should be red-toned, got rgb=({r}, {g}, {b})'
        assert r > b, f'{name}(1.0) should be red-toned, got rgb=({r}, {g}, {b})'

    def test_oranges_low_end_is_near_white(self):
        """The `oranges` scale starts near white (its Magics form fades in via alpha)."""
        r, g, b, _ = CAMS_AOD_COLORMAPS["oranges"](0.0)
        assert min(r, g, b) > 0.85, (
            f"oranges(0.0) should be near white, got rgb=({r}, {g}, {b})"
        )


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
        assert alpha[0, 0] == 0.0, (
            f"NaN pixel should be transparent, got alpha={alpha[0, 0]}"
        )
        assert alpha[0, 1] == 1.0, (
            f"finite max value should be opaque, got {alpha[0, 1]}"
        )

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
            rgb,
            [0.165, 0.031, 0.0],
            atol=0.01,
            err_msg=f"unexpected top-stop colour: {rgb}",
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
        img = alpha_scaled_image(
            ax, np.array([[0.0, 1.0]]), "viridis", constant_alpha=2.5
        )
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
        assert set(DATA_STYLES["haze"]) == {
            "organic_matter",
            "dust",
        }, f"unexpected haze layers: {set(DATA_STYLES['haze'])}"

    def test_normalize_object_norm_does_not_raise(self, ax):
        """A Normalize-instance `norm=` is left to imshow (no-op on RGBA), not mis-read as a kind."""
        data = np.array([[0.0, 10.0], [5.0, 8.0]])
        # Must not raise "data style 'norm' must be 'linear'/'log'/'symlog'".
        apply_data_style(
            ax,
            {"wind_speed": data},
            style="wind_speed",
            norm=Normalize(0, 10),
            legend=False,
        )

    def test_string_norm_override_selects_the_norm_kind(self, ax):
        """A string `norm=` overrides the preset's norm kind (e.g. 'log') via cfg and renders."""
        data = np.array(
            [[1.0, 10.0], [100.0, 1000.0]]
        )  # positive -> valid LogNorm window
        img = apply_data_style(
            ax, {"wind_speed": data}, style="wind_speed", norm="log", legend=False
        )
        assert (
            np.asarray(img["wind_speed"].get_array()).shape[-1] == 4
        )  # rendered RGBA, no raise

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
        assert alpha[0, 0] == 1.0, (
            f"expected fully opaque at data=0.5, got alpha={alpha[0, 0]}"
        )

    def test_cams_aod_preset_has_single_aod_layer(self):
        """The registered 'cams_aod' preset defines exactly one 'aod' layer."""
        assert set(DATA_STYLES["cams_aod"]) == {"aod"}, (
            f"unexpected cams_aod layers: {set(DATA_STYLES['cams_aod'])}"
        )

    def test_cams_aod_uses_official_palette(self):
        """The 'cams_aod' layer uses the canonical CAMS_AOD_COLORMAPS scale, not a haze map."""
        ref, xs = CAMS_AOD_COLORMAPS["blue_yellow_red"], np.linspace(0.0, 1.0, 16)
        got = DATA_STYLES["cams_aod"]["aod"]["cmap"]
        assert np.allclose([got(x) for x in xs], [ref(x) for x in xs]), (
            "cams_aod should reproduce the official CAMS AOD colormap"
        )

    def test_cams_aod_declares_no_decoupled_alpha(self):
        """Unlike 'haze', 'cams_aod' sets no alpha_vmin/alpha_vmax (opacity tracks colour)."""
        cfg = DATA_STYLES["cams_aod"]["aod"]
        assert 'alpha_vmin' not in cfg, f'cams_aod should not decouple alpha, got {cfg}'
        assert 'alpha_vmax' not in cfg, f'cams_aod should not decouple alpha, got {cfg}'

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
        "temperature",
        "elevation",
        "vegetation",
        "wind_speed",
        "anomaly",
        "precipitation",
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
        assert alpha[1, 0] == 0.0, (
            f"{style} NaN cell should be transparent, got {alpha[1, 0]}"
        )

    def test_auto_range_uses_data_min_max(self, ax):
        """A preset without vmin/vmax auto-ranges the colour norm to the data.

        Test scenario:
            The lowest data value maps to the colormap's start and the highest
            to its end, proving the norm resolved to the field's own [min, max]
            rather than a hard-coded 0-1.
        """
        # `wind_speed` (viridis) auto-ranges -- no vmin/vmax/levels -- so it is
        # the right probe for the data-min/max behaviour. (The fixed ECMWF
        # contour scale lives under the `temperature_2m` preset, not `temperature`.)
        cmap = plt.get_cmap("viridis")
        images = apply_data_style(
            ax, {"wind_speed": np.array([[10.0, 30.0]])}, style="wind_speed"
        )
        rgba = images["wind_speed"].get_array()
        np.testing.assert_allclose(
            rgba[0, 0, :3],
            cmap(0.0)[:3],
            atol=1e-6,
            err_msg="min value should map to the colormap start",
        )
        np.testing.assert_allclose(
            rgba[0, 1, :3],
            cmap(1.0)[:3],
            atol=1e-6,
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
            rgba[0, 2, :3],
            cmap(0.5)[:3],
            atol=0.02,
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

    def test_constant_alpha_and_decoupled_alpha_are_mutually_exclusive(
        self, ax, monkeypatch
    ):
        """A preset combining a constant 'alpha' with alpha_vmin/vmax raises ValueError."""
        import cleopatra.styling.colors as colors_mod

        bad = {
            "bad": {
                "x": {
                    "cmap": "viridis",
                    "label": "X",
                    "alpha": 1.0,
                    "alpha_vmin": 0.1,
                    "alpha_vmax": 0.5,
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
        import cleopatra.styling.colors as colors_mod

        custom_styles = {
            "plain": {
                "dust": {"cmap": "viridis", "label": "Plain", "vmin": 0.0, "vmax": 1.0}
            }
        }
        monkeypatch.setattr(colors_mod, "DATA_STYLES", custom_styles)
        images = apply_data_style(ax, {"dust": np.array([[0.5, 1.0]])}, style="plain")
        alpha = images["dust"].get_array()[..., 3]
        assert alpha[0, 0] == 0.5, (
            f"without alpha_vmin/vmax, alpha should equal the colour norm (0.5), got {alpha[0, 0]}"
        )

    def test_draws_one_image_per_layer(self, ax):
        """Each key in `layers` produces one returned `AxesImage`, drawn on `ax`."""
        layers = {
            "dust": np.array([[0.0, 1.0]]),
            "organic_matter": np.array([[0.2, 0.8]]),
        }
        images = apply_data_style(ax, layers)
        assert set(images) == {
            "dust",
            "organic_matter",
        }, f"unexpected keys: {set(images)}"
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
        apply_data_style(
            ax,
            {"dust": np.array([[0.0, 1.0]]), "organic_matter": np.array([[0.0, 1.0]])},
        )
        assert len(ax.child_axes) == 2, (
            f"expected 2 swatch legends, got {len(ax.child_axes)}"
        )

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
        assert images["dust"].get_zorder() == 5, (
            "zorder not forwarded to alpha_scaled_image"
        )

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


class TestReferencePresets:
    """Tests for the reference styles half of the merged `weather_presets.json` library."""

    @pytest.fixture
    def ax(self):
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_target_parameters_registered(self):
        """The curated ECMWF parameter set is registered by GRIB shortName."""
        for key in [
            "temperature_2m",
            "dewpoint_temperature_2m",
            "aerosol_optical_depth_550nm",
            "dust_aerosol_optical_depth_550nm",
            "wind_u_10m",
            "wind_v_10m",
            "wind_speed_10m",
            "total_precipitation",
            "convective_available_potential_energy",
        ]:
            assert key in DATA_STYLES, f"missing reference preset {key}"

    def test_reference_overrides_magics_neon_2t(self):
        """The reference temperature_2m (Spectral_r, banded) overrides the Magics rainbow ListedColormap."""
        layer = DATA_STYLES["temperature_2m"]["temperature_2m"]
        assert layer["cmap"] == "Spectral_r"
        assert layer["extend"] == "both"
        assert layer['levels'][0] == -40
        assert layer['levels'][-1] == 40
        assert "bands" not in layer  # not the Magics discrete-band path

    def test_cmap_name_style_stays_a_name(self):
        """A style whose reference `colors` is a matplotlib name keeps it as a string cmap."""
        assert DATA_STYLES["dewpoint_temperature_2m"]["dewpoint_temperature_2m"]["cmap"] == "BrBG_r"
        assert DATA_STYLES["wind_u_10m"]["wind_u_10m"]["cmap"] == "PiYG"

    def test_colour_list_with_levels_is_discrete_listed_colormap(self):
        """A colour-list reference preset (with levels) keeps the exact ECMWF colours (ListedColormap)."""
        layer = DATA_STYLES["aerosol_optical_depth_550nm"]["aerosol_optical_depth_550nm"]
        assert isinstance(layer["cmap"], ListedColormap)
        assert (
            layer["cmap"].N == 9
        )  # the 9 exact ECMWF colours, not a 256-entry resample
        assert layer["extend"] == "max"
        assert layer['levels'][0] == 0.1
        assert layer['levels'][-1] == 1.0

    def test_colour_list_without_levels_stays_continuous(self):
        """A colour-list reference preset with no levels (total_precipitation gradient) is a continuous ramp."""
        layer = DATA_STYLES["total_precipitation"]["total_precipitation"]
        assert isinstance(layer["cmap"], LinearSegmentedColormap)
        assert "levels" not in layer

    def test_colour_rich_list_preset_honours_extend(self):
        """convective_available_potential_energy (255 colours over 16 bands) keeps extend='max' -- it has room for an over colour."""
        layer = DATA_STYLES["convective_available_potential_energy"]["convective_available_potential_energy"]
        norm, _, _ = _resolve_style_norm(
            np.linspace(0.0, 5000.0, 400).reshape(20, 20), layer
        )
        assert isinstance(norm, BoundaryNorm)
        assert norm.extend == 'max'

    def test_one_colour_per_band_preset_drops_extend(self):
        """aerosol_optical_depth_550nm (9 colours, 9 bands) drops extend -- no spare colour for the over slot -- and clamps."""
        layer = DATA_STYLES["aerosol_optical_depth_550nm"]["aerosol_optical_depth_550nm"]
        norm, _, _ = _resolve_style_norm(
            np.linspace(0.0, 1.5, 400).reshape(20, 20), layer
        )
        assert isinstance(norm, BoundaryNorm)
        assert norm.extend == 'neither'

    def test_colour_list_preset_renders_exact_discrete_colours(self, ax):
        """A ListedColormap reference preset paints each band with its exact palette colour."""
        cmap = DATA_STYLES["aerosol_optical_depth_550nm"]["aerosol_optical_depth_550nm"]["cmap"]
        levels = DATA_STYLES["aerosol_optical_depth_550nm"]["aerosol_optical_depth_550nm"]["levels"]
        data = np.array(
            [[levels[0] + 1e-3, levels[4] + 1e-3]]
        )  # falls in band 0 and band 4
        img = apply_data_style(ax, {"aerosol_optical_depth_550nm": data}, style="aerosol_optical_depth_550nm", legend=False)[
            "aerosol_optical_depth_550nm"
        ]
        rgb = np.asarray(img.get_array())[..., :3]
        assert np.allclose(rgb[0, 0], to_rgb(cmap.colors[0]), atol=1 / 255)
        assert np.allclose(rgb[0, 1], to_rgb(cmap.colors[4]), atol=1 / 255)

    def test_reference_style_renders_banded(self, ax):
        """A vendored reference style renders discrete level bands end-to-end."""
        data = np.linspace(-10.0, 38.0, 60 * 60).reshape(60, 60)
        img = apply_data_style(ax, {"temperature_2m": data}, style="temperature_2m", legend=False)
        rgb = np.asarray(img["temperature_2m"].get_array())[..., :3].reshape(-1, 3)
        assert len(np.unique(np.round(rgb, 3), axis=0)) <= 43  # ~41 bands + extend

    def test_loader_degrades_without_asset(self, monkeypatch):
        """A missing weather asset degrades to no presets rather than raising."""
        import cleopatra.styling.colors as colors_mod

        def boom(_pkg):
            raise FileNotFoundError("no data package")

        monkeypatch.setattr(colors_mod.importlib.resources, "files", boom)
        assert _load_presets("weather_presets.json") == {}


class TestContourLevelsStyle:
    """Tests for the explicit `levels`/`extend` contour-band styling (ECMWF look)."""

    @pytest.fixture
    def ax(self):
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_2t_uses_ecmwf_spectral_bands(self):
        """The reference `temperature_2m` preset is ECMWF's default: Spectral_r banded at 2 degC over -40..40."""
        layer = DATA_STYLES["temperature_2m"]["temperature_2m"]
        assert layer["cmap"] == "Spectral_r"
        assert layer["extend"] == "both"
        assert layer['levels'][0] == -40
        assert layer['levels'][-1] == 40
        assert layer["levels"][1] - layer["levels"][0] == 2  # 2 degC interval

    def test_temperature_is_a_generic_auto_ranging_ramp(self):
        """The `temperature` preset is a generic Spectral_r ramp with no fixed levels, so it
        auto-ranges to the data (the fixed ECMWF scale lives under `temperature_2m`)."""
        layer = DATA_STYLES["temperature"]["temperature"]
        assert layer["cmap"] == "Spectral_r"
        assert 'levels' not in layer
        assert 'extend' not in layer
        norm, vmin, vmax = _resolve_style_norm(np.array([[5.0, 25.0]]), layer)
        assert not isinstance(norm, BoundaryNorm)
        assert (vmin, vmax) == (5.0, 25.0)  # fit to the data, not a fixed -40..40

    def test_levels_resolve_to_boundary_norm_with_extend(self):
        """A preset carrying `levels`/`extend` resolves to a BoundaryNorm honouring both."""
        layer = DATA_STYLES["temperature_2m"]["temperature_2m"]
        norm, vmin, vmax = _resolve_style_norm(np.array([[0.0, 25.0]]), layer)
        assert isinstance(norm, BoundaryNorm)
        assert norm.extend == "both"
        assert (vmin, vmax) == (-40.0, 40.0)
        assert list(norm.boundaries[:2]) == [-40.0, -38.0]

    def test_levels_render_discrete_bands(self, ax):
        """A continuous field styled with `levels` paints in a small set of banded colours."""
        data = np.linspace(-30.0, 38.0, 60 * 60).reshape(60, 60)
        img = apply_data_style(ax, {"temperature_2m": data}, style="temperature_2m", legend=False)
        rgb = np.asarray(img["temperature_2m"].get_array())[..., :3].reshape(-1, 3)
        distinct = len(np.unique(np.round(rgb, 3), axis=0))
        assert distinct <= 41, f"expected discrete level bands, got {distinct} colours"

    def test_caller_override_rescales_a_levels_preset(self, ax):
        """An explicit caller vmin/vmax overrides a levels preset's fixed scale (not a silent no-op)."""
        data = np.linspace(-40.0, 100.0, 400).reshape(20, 20)
        fig2, ax2 = plt.subplots()
        base = apply_data_style(ax, {"temperature_2m": data}, style="temperature_2m", legend=False)
        over = apply_data_style(
            ax2, {"temperature_2m": data}, style="temperature_2m", vmin=-40.0, vmax=100.0, legend=False
        )
        assert not np.allclose(base["temperature_2m"].get_array(), over["temperature_2m"].get_array())
        plt.close(fig2)

    def test_string_norm_kind_overrides_a_levels_preset(self):
        """A caller string `norm='log'`/`'symlog'` rescales a levels preset (matching a
        Normalize instance), while `'linear'` keeps the discrete bands."""
        data = np.array([[0.5, 1.0, 2.0, 3.5]])  # strictly positive, valid for LogNorm
        cfg = {"cmap": "viridis", "levels": [0, 1, 2, 3, 4], "extend": "both"}
        assert isinstance(_resolve_style_norm(data, cfg)[0], BoundaryNorm)
        assert isinstance(_resolve_style_norm(data, {**cfg, "norm": "log"})[0], LogNorm)
        assert isinstance(
            _resolve_style_norm(data, {**cfg, "norm": "symlog"})[0], SymLogNorm
        )
        # 'linear' is the implicit default and must not abandon the banding.
        assert isinstance(
            _resolve_style_norm(data, {**cfg, "norm": "linear"})[0], BoundaryNorm
        )

    def test_string_norm_log_is_not_a_silent_noop_on_a_levels_preset(self, ax):
        """Through `apply_data_style`, a string `norm='log'` on a levels preset changes the
        rendered pixels instead of being silently dropped (the L1 inconsistency)."""
        data = np.linspace(0.05, 4.5, 400).reshape(
            20, 20
        )  # positive, spans the aerosol_optical_depth_550nm levels
        fig2, ax2 = plt.subplots()
        base = apply_data_style(ax, {"aerosol_optical_depth_550nm": data}, style="aerosol_optical_depth_550nm", legend=False)
        logged = apply_data_style(
            ax2, {"aerosol_optical_depth_550nm": data}, style="aerosol_optical_depth_550nm", norm="log", legend=False
        )
        assert not np.allclose(base["aerosol_optical_depth_550nm"].get_array(), logged["aerosol_optical_depth_550nm"].get_array())
        plt.close(fig2)

    def test_instance_norm_labels_legend_with_its_own_range_on_a_levels_preset(
        self, ax
    ):
        """An instance `norm=` on a levels preset labels the swatch with the instance's range,
        not the preset's fixed level endpoints (L1)."""
        data = np.linspace(1.0, 90.0, 400).reshape(
            20, 20
        )  # positive, valid for LogNorm
        apply_data_style(
            ax, {"temperature_2m": data}, style="temperature_2m", norm=LogNorm(vmin=1, vmax=100), legend=True
        )
        swatch_texts = [t.get_text() for c in ax.child_axes for t in c.texts]
        assert "1" in swatch_texts, (
            f"low endpoint should be the instance vmin (1), got {swatch_texts}"
        )
        assert "≥100" in swatch_texts, (
            f"high endpoint should be the instance vmax (100), got {swatch_texts}"
        )
        assert '-40' not in swatch_texts, f"must not label with the preset's fixed level endpoints, got {swatch_texts}"
        assert '≤-40' not in swatch_texts, f"must not label with the preset's fixed level endpoints, got {swatch_texts}"

    def test_data_outside_fixed_levels_warns(self, ax):
        """Data entirely outside a levels preset's scale warns (Celsius levels, Kelvin data footgun)."""
        kelvin = np.full(
            (4, 4), 290.0
        )  # ~17 degC in K, far above the -40..40 degC temperature_2m levels
        with pytest.warns(UserWarning, match="expected units"):
            apply_data_style(ax, {"temperature_2m": kelvin}, style="temperature_2m", legend=False)

    def test_two_sided_extend_legend_caps_the_low_endpoint(self, ax):
        """A two-sided `extend='both'` levels preset marks both endpoints as capped ('≤'/'≥')."""
        data = np.linspace(-30.0, 38.0, 400).reshape(20, 20)
        apply_data_style(ax, {"temperature_2m": data}, style="temperature_2m", legend=True)
        swatch_texts = [t.get_text() for c in ax.child_axes for t in c.texts]
        assert "≤-40" in swatch_texts, (
            f"expected a capped '≤-40' low endpoint, got {swatch_texts}"
        )
        assert "≥40" in swatch_texts, (
            f"expected a capped '≥40' high endpoint, got {swatch_texts}"
        )

    def test_downgraded_extend_legend_caps_neither_endpoint(self, ax):
        """`aerosol_optical_depth_550nm`'s extend='max' is downgraded to 'neither' (its 9-colour ListedColormap
        has no spare over-slot), so the legend caps NEITHER end -- not a spurious '≥'.
        """
        data = np.linspace(0.05, 4.5, 400).reshape(20, 20)
        apply_data_style(ax, {"aerosol_optical_depth_550nm": data}, style="aerosol_optical_depth_550nm", legend=True)
        swatch_texts = [t.get_text() for c in ax.child_axes for t in c.texts]
        assert not any(t.startswith("≤") for t in swatch_texts), (
            f"a downgraded extend must not cap the low endpoint, got {swatch_texts}"
        )
        assert not any(t.startswith("≥") for t in swatch_texts), (
            f"a downgraded extend must not cap the high endpoint either, got {swatch_texts}"
        )

    def test_max_extend_with_room_caps_only_the_high_endpoint(self, ax):
        """A colour-rich `extend='max'` preset (convective_available_potential_energy keeps its over-slot) caps only the high
        end ('≥'), leaving the low endpoint plain."""
        data = np.linspace(100.0, 4500.0, 400).reshape(20, 20)
        apply_data_style(ax, {"convective_available_potential_energy": data}, style="convective_available_potential_energy", legend=True)
        swatch_texts = [t.get_text() for c in ax.child_axes for t in c.texts]
        assert any(t.startswith("≥") for t in swatch_texts), (
            f"a kept extend='max' should cap the high endpoint, got {swatch_texts}"
        )
        assert not any(t.startswith("≤") for t in swatch_texts), (
            f"extend='max' must not cap the low endpoint, got {swatch_texts}"
        )


class TestFlameColormapsAndPresets:
    """Tests for the flame/heat colormaps and the temperature_flame presets."""

    @pytest.fixture
    def ax(self):
        """A fresh Axes on the Agg backend, closed after the test."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_flame_colormaps_are_registered(self):
        """Both flame flavours are present as ready Colormaps, and only those two."""
        assert set(FLAME_COLORMAPS) == {"white_hot", "amber"}
        for name in FLAME_COLORMAPS:
            assert isinstance(FLAME_COLORMAPS[name], Colormap), name

    def test_white_hot_runs_dark_to_bright(self):
        """`white_hot` starts near-black (cool) and ends near-white (hot), like a flame."""
        r0, g0, b0, _ = FLAME_COLORMAPS["white_hot"](0.0)
        r1, g1, b1, _ = FLAME_COLORMAPS["white_hot"](1.0)
        assert max(r0, g0, b0) < 0.1, "cool end should be near-black"
        assert min(r1, g1, b1) > 0.9, "hot end should be near-white"

    @pytest.mark.parametrize(
        "style, cmap_name",
        [("temperature_flame", "white_hot"), ("temperature_flame_amber", "amber")],
    )
    def test_flame_presets_carry_glow_ramp(self, style, cmap_name):
        """Each flame preset is a single layer with a colour range and a value-linked opacity ramp."""
        assert set(DATA_STYLES[style]) == {style}
        layer = DATA_STYLES[style][style]
        ref, xs = FLAME_COLORMAPS[cmap_name], np.linspace(0.0, 1.0, 16)
        assert np.allclose([layer["cmap"](x) for x in xs], [ref(x) for x in xs]), (
            "flame preset should reproduce the FLAME_COLORMAPS ramp"
        )
        assert (layer["vmin"], layer["vmax"]) == (0.0, 40.0)
        # alpha decoupled from colour -> the glow (transparent when cool, opaque when hot)
        assert layer["alpha_vmin"] < layer["alpha_vmax"]
        assert "alpha" not in layer

    def test_flame_preset_render_ties_opacity_to_value(self, ax):
        """A flame preset renders RGBA whose alpha rises with the value (cool fades, hot glows)."""
        data = np.linspace(0.0, 40.0, 400).reshape(20, 20)
        img = apply_data_style(
            ax, {"temperature_flame": data}, style="temperature_flame", legend=False
        )
        alpha = np.asarray(img["temperature_flame"].get_array())[..., 3]
        assert alpha.flat[0] < 0.1, "coolest cell should be nearly transparent"
        assert alpha.flat[-1] == 1.0, "hottest cell should be fully opaque"


class TestMagicsPresets:
    """Tests for the ECMWF/Magics half of the merged `weather_presets.json` library."""

    @pytest.fixture
    def ax(self):
        """A fresh Axes on the Agg backend, closed after the test."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    HAND_AUTHORED = {
        "haze",
        "cams_aod",
        "temperature",
        "elevation",
        "vegetation",
        "wind_speed",
        "anomaly",
        "precipitation",
    }

    def test_known_parameters_are_registered(self):
        """Well-known GRIB parameters resolve to Magics presets carrying their real labels.

        (Uses `min_temperature_2m`/`max_temperature_2m`: the `temperature_2m`/`total_precipitation`/`aerosol_optical_depth_550nm` shortNames are now the reference
        default styles -- see `TestReferencePresets`.)
        """
        assert DATA_STYLES["min_temperature_2m"]["min_temperature_2m"]["label"].startswith(
            "Minimum temperature at 2 metres"
        )
        assert DATA_STYLES["max_temperature_2m"]["max_temperature_2m"]["label"].startswith(
            "Maximum temperature at 2 metres"
        )

    def test_a_substantial_library_was_loaded(self):
        """The vendored asset registers a large batch of parameter presets."""
        magics = set(DATA_STYLES) - self.HAND_AUTHORED
        assert len(magics) >= 50, f"expected many Magics presets, got {len(magics)}"

    def test_preset_layer_structure(self):
        """Each Magics preset is a single layer keyed by its own name, with a Colormap."""
        entry = DATA_STYLES["min_temperature_2m"]
        assert set(entry) == {"min_temperature_2m"}, f"unexpected layers: {set(entry)}"
        layer = entry["min_temperature_2m"]
        assert isinstance(layer["cmap"], Colormap), f"cmap is {type(layer['cmap'])}"
        assert isinstance(layer['label'], str)
        assert layer['label']

    def test_opaque_preset_carries_constant_alpha(self):
        """An opaque Magics field (min 2m temperature) sets alpha=1.0 -- a full opaque field."""
        assert DATA_STYLES["min_temperature_2m"]["min_temperature_2m"]["alpha"] == 1.0

    def test_overlay_preset_has_no_constant_alpha(self):
        """An alpha-ramped Magics field (high cloud cover) is a value-linked overlay."""
        assert "alpha" not in DATA_STYLES["high_cloud_cover"]["high_cloud_cover"], (
            "an alpha-ramped Magics palette should map to the overlay policy"
        )

    def test_magics_preset_renders_opaque_field(self, ax):
        """A Magics preset draws end-to-end; an opaque one fills the field, NaN transparent."""
        images = apply_data_style(
            ax, {"min_temperature_2m": np.array([[-20.0, 30.0], [np.nan, 5.0]])}, style="min_temperature_2m"
        )
        alpha = images["min_temperature_2m"].get_array()[..., 3]
        assert alpha[0, 0] == alpha[0, 1] == alpha[1, 1] == 1.0, f"not opaque: {alpha}"
        assert alpha[1, 0] == 0.0, f"NaN cell should be transparent, got {alpha[1, 0]}"

    def test_loader_degrades_gracefully_without_asset(self, monkeypatch):
        """If the vendored asset is unreadable, the loader returns {} instead of raising."""
        import cleopatra.styling.colors as colors_mod

        def boom(_pkg):
            raise FileNotFoundError("no data package")

        monkeypatch.setattr(colors_mod.importlib.resources, "files", boom)
        assert _load_presets("weather_presets.json") == {}, (
            "missing asset should degrade to no presets"
        )

    def test_preset_carries_decoded_fixed_range(self):
        """A Magics preset whose style name encodes a range ships that vmin/vmax."""
        layer = DATA_STYLES["min_temperature_2m"]["min_temperature_2m"]
        assert layer['vmin'] == -48.0
        assert layer['vmax'] == 56.0

    def test_magics_preset_is_discrete_banded(self):
        """A Magics preset renders as flat discrete bands (ListedColormap + band count)."""
        layer = DATA_STYLES["min_temperature_2m"]["min_temperature_2m"]
        assert isinstance(layer["cmap"], ListedColormap)
        assert layer["bands"] == layer["cmap"].N == 27

    @pytest.mark.parametrize("key", ["min_temperature_2m", "total_precipitation_index"])
    def test_banded_preset_edges_stay_within_the_range(self, key):
        """Band edges partition [vmin, vmax] exactly -- no overshoot beyond vmax (every colour reachable)."""
        layer = DATA_STYLES[key][key]
        data = np.linspace(layer["vmin"], layer["vmax"], 400).reshape(20, 20)
        norm, vmin, vmax = _resolve_style_norm(data, layer)
        assert isinstance(norm, BoundaryNorm)
        edges = np.asarray(norm.boundaries)
        assert edges[0] == vmin == layer["vmin"]
        assert edges[-1] == vmax == layer["vmax"]  # no phantom band beyond vmax
        assert len(edges) == layer["bands"] + 1

    def test_banded_render_produces_few_distinct_colours(self, ax):
        """A banded preset paints flat bands, so a smooth field renders in few colours."""
        data = np.linspace(-30.0, 45.0, 60 * 60).reshape(60, 60)
        img = apply_data_style(ax, {"min_temperature_2m": data}, style="min_temperature_2m", legend=False)["min_temperature_2m"]
        rgb = np.asarray(img.get_array())[..., :3].reshape(-1, 3)
        distinct = np.unique(np.round(rgb, 3), axis=0)
        assert len(distinct) <= 27, (
            f"expected discrete bands, got {len(distinct)} colours"
        )

    def test_cmocean_preset_stays_continuous(self):
        """A non-Magics (cmocean) preset is a genuine continuous ramp, not banded."""
        layer = DATA_STYLES["bathymetry"]["bathymetry"]
        assert isinstance(layer["cmap"], LinearSegmentedColormap)
        assert "bands" not in layer

    @pytest.mark.parametrize("style", ["bathymetry", "salinity", "total_precipitation"])
    def test_continuous_presets_are_perceptually_even(self, style):
        """Continuous ocean/weather ramps are interpolated in CIELAB, so their per-step
        colour change is even (a plain RGB `from_list` of the same palette scores far higher)."""
        assert perceptual_uniformity(DATA_STYLES[style][style]["cmap"]) < 0.05

    def test_temperature_preset_keeps_full_colour_ramp(self):
        """The vendored min_temperature_2m palette keeps its full blue->green->yellow->red->magenta ramp.

        Magics palettes name intermediate colours (`greenish_blue`, `yellow_green`, ...)
        that are not matplotlib colours; dropping the unrecognised names truncates the
        ramp and over-weights the magenta cap (whole summers rendered magenta). Guard
        the shipped asset: the ramp is long and the green mid-band survives.

        (Uses `min_temperature_2m`, not `temperature_2m`: `temperature_2m` is now the reference default -- see
        `TestReferencePresets` -- but `min_temperature_2m` is from the same Magics temperature
        family and keeps the same long named-colour ramp.)
        """
        rec = json.loads(
            importlib.resources.files("cleopatra.styling.data")
            .joinpath("weather_presets.json")
            .read_text()
        )["presets"]["min_temperature_2m"]["layers"]["min_temperature_2m"]
        palette = rec["colors"]
        assert len(palette) >= 27, f"min_temperature_2m ramp truncated to {len(palette)} colours"
        assert any(g > r and g > b and g > 0.5 for r, g, b in map(to_rgb, palette)), (
            "the green transition band (Magics named colours) must be preserved"
        )

    def test_temperature_family_shares_the_style_range(self):
        """The Magics -48..56 temperature family carries the same decoded range.

        (`temperature_2m`/`dewpoint_temperature_2m` are now the reference default; `min_temperature_2m`/`max_temperature_2m` remain Magics.)
        """
        for key in ("min_temperature_2m", "max_temperature_2m"):
            layer = DATA_STYLES[key][key]
            assert (layer["vmin"], layer["vmax"]) == (-48.0, 56.0), key

    def test_explicit_none_vmin_does_not_wipe_fixed_range(self, ax):
        """Passing vmin=None to apply_data_style keeps the preset's fixed range (not auto-range)."""
        data = np.linspace(-10.0, 50.0, 400).reshape(20, 20)
        fig2, ax2 = plt.subplots()
        base = apply_data_style(ax, {"min_temperature_2m": data}, style="min_temperature_2m", legend=False)
        none = apply_data_style(
            ax2, {"min_temperature_2m": data}, style="min_temperature_2m", vmin=None, legend=False
        )
        assert np.allclose(base["min_temperature_2m"].get_array(), none["min_temperature_2m"].get_array())
        plt.close(fig2)

    def test_data_outside_fixed_range_warns(self, ax):
        """A fixed-range Magics preset warns when the data is entirely outside its scale."""
        kelvin = np.full((4, 4), 290.0)  # far above min_temperature_2m's decoded -48..56 degC scale
        with pytest.warns(UserWarning, match="expected units"):
            apply_data_style(ax, {"min_temperature_2m": kelvin}, style="min_temperature_2m", legend=False)

    def test_range_without_interval_is_decoded(self):
        """A style name with a range but no interval (wind f0t80) still gets vmin/vmax."""
        layer = DATA_STYLES["wind_gust_10m"]["wind_gust_10m"]
        assert (layer["vmin"], layer["vmax"]) == (0.0, 80.0)

    def test_named_palette_preset_has_no_range(self):
        """A Magics preset whose style name carries no range (carbon_monoxide) still auto-ranges."""
        assert "vmin" not in DATA_STYLES["carbon_monoxide"]["carbon_monoxide"]

    def test_caller_vmin_vmax_overrides_preset_range(self, ax):
        """An explicit vmin/vmax at draw time overrides the preset's fixed range."""
        data = np.linspace(-10.0, 50.0, 400).reshape(20, 20)
        fig2, ax2 = plt.subplots()
        base = apply_data_style(ax, {"min_temperature_2m": data}, style="min_temperature_2m", legend=False)
        over = apply_data_style(
            ax2, {"min_temperature_2m": data}, style="min_temperature_2m", vmin=-10.0, vmax=50.0, legend=False
        )
        assert not np.allclose(base["min_temperature_2m"].get_array(), over["min_temperature_2m"].get_array())
        plt.close(fig2)


class TestCategoricalPresets:
    """Tests for categorical (class-code) presets -- the `flood_status` preset."""

    @pytest.fixture
    def ax(self):
        """A fresh Axes on the Agg backend, closed after the test."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_flood_status_declares_categories(self):
        """The 'flood_status' preset carries a 5-class category list, not a cmap."""
        cfg = DATA_STYLES["flood_status"]["flood_status"]
        assert 'categories' in cfg
        assert 'cmap' not in cfg
        assert [c[2] for c in cfg["categories"]] == [
            "Normal",
            "Action",
            "Minor",
            "Moderate",
            "Major",
        ]

    def test_category_boundaries_are_midpoints_and_half_gaps(self):
        """Integer class codes 0..4 produce the expected +/-0.5 bin edges."""
        assert _category_boundaries([0, 1, 2, 3, 4]) == [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]

    def test_single_category_boundaries(self):
        """A lone class value still yields a valid two-edge bin."""
        assert _category_boundaries([3]) == [2.5, 3.5]

    def test_each_class_maps_to_its_colour(self, ax):
        """Every class code renders as exactly its declared category colour."""
        data = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 4.0]])
        img = apply_data_style(ax, {"flood_status": data}, style="flood_status")[
            "flood_status"
        ]
        rgba = img.get_array()
        expected = ["#2c7fb8", "#31a354", "#ffeb3b", "#ff7f00", "#e31a1c"]
        got = [
            to_hex(rgba[0, 0, :3]),
            to_hex(rgba[0, 1, :3]),
            to_hex(rgba[0, 2, :3]),
            to_hex(rgba[1, 0, :3]),
            to_hex(rgba[1, 1, :3]),
        ]
        assert got == expected, f"class colours wrong: {got}"

    def test_categorical_is_opaque_with_nan_transparent(self, ax):
        """Classes are drawn fully opaque; NaN (no-data) is transparent."""
        data = np.array([[0.0, 4.0], [np.nan, 2.0]])
        img = apply_data_style(ax, {"flood_status": data}, style="flood_status")[
            "flood_status"
        ]
        alpha = img.get_array()[..., 3]
        assert alpha[0, 0] == alpha[0, 1] == alpha[1, 1] == 1.0, f"not opaque: {alpha}"
        assert alpha[1, 0] == 0.0, f"NaN should be transparent, got {alpha[1, 0]}"

    def test_categorical_attaches_a_disjoint_legend(self, ax):
        """A categorical preset gets a discrete matplotlib legend, titled by `label`."""
        apply_data_style(
            ax, {"flood_status": np.array([[0.0, 4.0]])}, style="flood_status"
        )
        legend = ax.get_legend()
        assert legend is not None, "categorical preset should attach a legend"
        assert [t.get_text() for t in legend.get_texts()] == [
            "Normal",
            "Action",
            "Minor",
            "Moderate",
            "Major",
        ]
        assert legend.get_title().get_text() == "Flood status"

    def test_categorical_legend_can_be_suppressed(self, ax):
        """`legend=False` draws the classes without a legend."""
        apply_data_style(
            ax,
            {"flood_status": np.array([[0.0, 4.0]])},
            style="flood_status",
            legend=False,
        )
        assert ax.get_legend() is None, "no legend expected when legend=False"

    def test_categorical_legend_honours_bounds_and_stacks(self, ax, monkeypatch):
        """Two categorical layers keep both legends, anchored by `legend_bounds`."""
        import cleopatra.styling.colors as colors_mod

        styles = dict(
            colors_mod.DATA_STYLES,
            two={
                "a": {
                    "categories": [(0, "#111111", "A"), (1, "#eeeeee", "B")],
                    "label": "A",
                },
                "b": {
                    "categories": [(0, "#ff0000", "X"), (1, "#00ff00", "Y")],
                    "label": "B",
                },
            },
        )
        monkeypatch.setattr(colors_mod, "DATA_STYLES", styles)
        apply_data_style(
            ax,
            {"a": np.array([[0.0, 1.0]]), "b": np.array([[0.0, 1.0]])},
            style="two",
            legend_bounds=[(0.02, 0.9, 0.3, 0.05), (0.6, 0.9, 0.3, 0.05)],
        )
        legends = [c for c in ax.get_children() if type(c).__name__ == "Legend"]
        assert len(legends) == 2, "both categorical legends should coexist, not clobber"

    def test_out_of_range_and_nodata_codes_are_transparent(self, ax):
        """Codes outside the declared set (sinks, nodata) render transparent, not clamped.

        Test scenario:
            A categorical field carrying values that are not declared class
            codes -- a D8 sink (0), a nodata sentinel (255), or an out-of-range
            flood code -- must be masked to transparent rather than clamped to
            the first/last category at full opacity.
        """
        img = apply_data_style(
            ax,
            {"flood_status": np.array([[-3.0, 0.0, 4.0, 7.0]])},
            style="flood_status",
        )["flood_status"]
        alpha = img.get_array()[..., 3]
        assert list(alpha[0]) == [
            0.0,
            1.0,
            1.0,
            0.0,
        ], f"only declared codes should be opaque, got {list(alpha[0])}"


class TestFlowRasterPresets:
    """Tests for the DEM-derived hydrology presets (flow direction / accumulation)."""

    @pytest.fixture
    def ax(self):
        """A fresh Axes on the Agg backend, closed after the test."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_flow_direction_d8_has_eight_compass_classes(self):
        """`flow_direction_d8` maps the 8 ESRI D8 codes to compass-labelled classes."""
        cats = DATA_STYLES["flow_direction_d8"]["flow_direction_d8"]["categories"]
        assert [c[0] for c in cats] == [1, 2, 4, 8, 16, 32, 64, 128]
        assert [c[2] for c in cats] == ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]

    def test_flow_direction_d8_maps_codes_to_cyclic_colours(self, ax):
        """Each D8 code renders as its declared cyclic colour."""
        img = apply_data_style(
            ax,
            {"flow_direction_d8": np.array([[1.0, 128.0]])},
            style="flow_direction_d8",
        )["flow_direction_d8"]
        rgba = img.get_array()
        assert to_hex(rgba[0, 0, :3]) == "#e2d9e2"
        assert to_hex(rgba[0, 1, :3]) == "#cca389"

    def test_flow_accumulation_declares_symlog(self):
        """`flow_accumulation` uses a symmetric-log norm for its skewed range."""
        assert DATA_STYLES["flow_accumulation"]["flow_accumulation"]["norm"] == "symlog"

    def test_flow_accumulation_fades_zeros_shows_channels(self, ax):
        """Zero-accumulation cells fade out; high-accumulation channels are opaque."""
        img = apply_data_style(
            ax,
            {"flow_accumulation": np.array([[0.0, 9000.0]])},
            style="flow_accumulation",
        )["flow_accumulation"]
        alpha = img.get_array()[..., 3]
        assert alpha[0, 0] == 0.0, f"zero cell should be transparent, got {alpha[0, 0]}"
        assert alpha[0, 1] == 1.0, f"channel cell should be opaque, got {alpha[0, 1]}"


class TestStyleNormKinds:
    """Tests for the `norm` key in `_resolve_style_norm` (linear / log / symlog)."""

    def test_default_is_linear(self):
        """Omitting `norm` yields a plain `Normalize`."""
        norm, _, _ = _resolve_style_norm(np.array([1.0, 10.0]), {})
        assert type(norm).__name__ == "Normalize"

    def test_symlog_norm(self):
        """`norm='symlog'` yields a `SymLogNorm` (handles zeros)."""
        norm, _, _ = _resolve_style_norm(np.array([0.0, 100.0]), {"norm": "symlog"})
        assert type(norm).__name__ == "SymLogNorm"

    def test_log_norm_clamps_vmin_to_positive(self):
        """`norm='log'` yields a `LogNorm` with a positive vmin even if data hits 0."""
        norm, _, _ = _resolve_style_norm(np.array([0.0, 5.0, 100.0]), {"norm": "log"})
        assert type(norm).__name__ == "LogNorm"
        assert norm.vmin > 0, "LogNorm needs a positive lower bound"

    def test_unknown_norm_raises(self):
        """An unrecognised `norm` value raises `ValueError`."""
        with pytest.raises(ValueError, match="'norm' must be"):
            _resolve_style_norm(np.array([1.0, 2.0]), {"norm": "bogus"})

    def test_center_is_honoured_with_explicit_bounds(self):
        """`center` puts zero on the midpoint even when explicit vmin/vmax are set.

        With asymmetric explicit bounds a plain `Normalize` would land `center`
        off-midpoint; a `TwoSlopeNorm` keeps it at 0.5.
        """
        data = np.array([-2.0, 4.0, 10.0])
        norm, _, _ = _resolve_style_norm(data, {"center": 0, "vmin": -2, "vmax": 10})
        assert type(norm).__name__ == "TwoSlopeNorm"
        assert float(norm(0.0)) == pytest.approx(0.5)

    def test_center_outside_bounds_raises(self):
        """A `center` not strictly inside [vmin, vmax] raises a clear `ValueError`."""
        with pytest.raises(ValueError, match="must lie strictly between"):
            _resolve_style_norm(
                np.array([1.0, 10.0]), {"center": 0, "vmin": 1, "vmax": 10}
            )

    def test_log_reports_clamped_positive_vmin(self):
        """The `log` branch reports the clamped positive lower bound (matches the LogNorm).

        The returned vmin must equal the `LogNorm`'s vmin so the swatch legend
        does not label a 0/negative bound the colours never actually use.
        """
        norm, vmin, _ = _resolve_style_norm(
            np.array([0.0, 5.0, 100.0]), {"norm": "log"}
        )
        assert vmin > 0
        assert vmin == norm.vmin

    def test_log_on_all_negative_data_raises_clearly(self):
        """`norm='log'` on all-negative data raises a clear cleopatra error.

        There is no positive value to anchor a log scale, so the branch must
        fail with an actionable message rather than building an invalid
        `LogNorm(vmin>vmax)` that crashes deep in matplotlib at draw time.
        """
        with pytest.raises(ValueError, match="norm='log' needs positive data"):
            _resolve_style_norm(np.array([-5.0, -3.0, -1.0]), {"norm": "log"})

    def test_log_on_all_zero_data_raises_clearly(self):
        """`norm='log'` on all-zero data raises rather than a degenerate `LogNorm(1,1)`."""
        with pytest.raises(ValueError, match="norm='log' needs positive data"):
            _resolve_style_norm(np.array([0.0, 0.0, 0.0]), {"norm": "log"})

    def test_log_on_single_positive_value_raises_clearly(self):
        """`norm='log'` where the only positive value equals vmax raises (no flat LogNorm).

        Data `[0, 5]` resolves lower bound == upper bound (5); a strict `>` guard
        would let it build a degenerate `LogNorm(vmin==vmax)`, so the guard uses
        `>=` and fails clearly instead.
        """
        with pytest.raises(ValueError, match="norm='log' needs positive data"):
            _resolve_style_norm(np.array([0.0, 5.0]), {"norm": "log"})


class TestCmoceanPresets:
    """Tests for the cmocean ocean/hydrology/DEM preset library in `DATA_STYLES`."""

    @pytest.fixture
    def ax(self):
        """A fresh Axes on the Agg backend, closed after the test."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_known_variables_are_registered(self):
        """Ocean/hydrology variables resolve to presets carrying cmocean labels."""
        assert DATA_STYLES["salinity"]["salinity"]["label"] == "Salinity"
        assert DATA_STYLES["bathymetry"]["bathymetry"]["label"] == "Ocean depth"
        assert DATA_STYLES["turbidity"]["turbidity"]["label"] == "Turbidity / sediment"

    def test_the_batch_was_loaded(self):
        """The full curated cmocean batch is registered as presets."""
        expected = {
            "salinity",
            "bathymetry",
            "topography",
            "turbidity",
            "current_speed",
            "chlorophyll",
            "dissolved_oxygen",
            "sea_surface_temperature",
            "sea_ice",
            "solar_radiation",
            "rainfall",
            "phase",
            "sea_level_anomaly",
            "vorticity",
            "water_density",
        }
        assert expected <= set(DATA_STYLES), f"missing: {expected - set(DATA_STYLES)}"

    def test_diverging_and_land_sea_presets_center_on_zero(self):
        """The diverging and land+sea presets render symmetric about zero."""
        for key in ("sea_level_anomaly", "vorticity", "topography"):
            assert DATA_STYLES[key][key].get("center") == 0.0, (
                f"{key} should center on 0"
            )

    def test_preset_is_an_opaque_single_layer(self):
        """Each cmocean preset is one opaque layer with a ready Colormap."""
        layer = DATA_STYLES["salinity"]["salinity"]
        assert isinstance(layer["cmap"], Colormap), f"cmap is {type(layer['cmap'])}"
        assert layer["alpha"] == 1.0, "cmocean presets are opaque full fields"

    def test_cmocean_preset_renders_opaque_field(self, ax):
        """A cmocean preset draws end-to-end: opaque field, NaN transparent."""
        images = apply_data_style(
            ax,
            {"bathymetry": np.array([[0.0, 5000.0], [np.nan, 2000.0]])},
            style="bathymetry",
        )
        alpha = images["bathymetry"].get_array()[..., 3]
        assert alpha[0, 0] == alpha[0, 1] == alpha[1, 1] == 1.0, f"not opaque: {alpha}"
        assert alpha[1, 0] == 0.0, f"NaN cell should be transparent, got {alpha[1, 0]}"

    def test_missing_asset_degrades_to_empty(self):
        """The shared asset loader returns {} for an absent resource, never raising."""
        assert _load_presets("does_not_exist.json") == {}

    @staticmethod
    def _patch_asset_text(monkeypatch, text):
        """Make the loader read `text` as the asset content."""
        import cleopatra.styling.colors as colors_mod

        class _File:
            def read_text(self, encoding=None):
                return text

        class _Dir:
            def joinpath(self, name):
                return _File()

        monkeypatch.setattr(colors_mod.importlib.resources, "files", lambda pkg: _Dir())

    def test_malformed_json_degrades_to_empty(self, monkeypatch):
        """A corrupt (invalid JSON) asset returns {} instead of crashing the import."""
        self._patch_asset_text(monkeypatch, "{not valid json")
        assert _load_presets("ocean_presets.json") == {}

    def test_non_mapping_json_degrades_to_empty(self, monkeypatch):
        """A structurally-wrong (non-object) asset returns {}, never raises."""
        self._patch_asset_text(monkeypatch, "[1, 2, 3]")
        assert _load_presets("ocean_presets.json") == {}

    def test_one_bad_record_is_skipped_others_survive(self, monkeypatch):
        """A single malformed record is skipped; the sibling well-formed presets load."""
        import json

        asset = json.dumps(
            {
                "version": 1,
                "presets": {
                    "good1": {"layers": {"good1": {
                        "label": "Good 1", "colors": ["#000000", "#ffffff"], "colormap": "perceptual"}}},
                    "bad": {"label": "no layers"},
                    "good2": {"layers": {"good2": {
                        "label": "Good 2", "colors": ["#ff0000", "#00ff00"], "colormap": "perceptual"}}},
                },
            }
        )
        self._patch_asset_text(monkeypatch, asset)
        loaded = _load_presets("ocean_presets.json")
        assert set(loaded) == {
            "good1",
            "good2",
        }, f"bad record should not drop siblings: {loaded}"


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
