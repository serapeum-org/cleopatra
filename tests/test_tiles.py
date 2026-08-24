"""Tests for cleopatra.basemap.tiles.

Covers `add_tiles` and helper functions in the ported web-tile
basemap module. HTTP fetching is mocked at the `urllib.request`
layer so the suite never hits the public internet -- the same strategy
that pyramids uses for its basemap tests.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.plot

pytest.importorskip("PIL", reason="Pillow not installed (tiles extra)")
pytest.importorskip("xyzservices", reason="xyzservices not installed (tiles extra)")
pytest.importorskip("pyproj", reason="pyproj not installed (tiles extra)")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from cleopatra.basemap import tiles as tiles_mod  # noqa: E402
from cleopatra.basemap.tiles import (  # noqa: E402
    MAX_TILES,
    Tile,
    _densify_and_reproject_bounds,
    _lonlat_to_tile_xy,
    _looks_like_image,
    _require_tiles_extra,
    _tile_xy_bounds,
    _tiles_for_bbox,
    add_tiles,
    auto_zoom,
    fetch_single_tile,
    fetch_tiles,
    get_provider,
    mercator_to_equirectangular,
    stitch_tiles,
    world_texture,
)


def _make_tile_png(size: int = 256) -> bytes:
    """Encode a solid-color RGBA PNG tile.

    Args:
        size: Square tile side in pixels.

    Returns:
        bytes: PNG-encoded image bytes.
    """
    img = Image.new("RGBA", (size, size), (128, 128, 128, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestGetProvider:
    """Tests for `cleopatra.basemap.tiles.get_provider`."""

    def test_default_provider_is_openstreetmap(self):
        """Calling `get_provider(None)` returns OpenStreetMap.Mapnik."""
        provider = get_provider(None)
        assert "openstreetmap" in provider.name.lower() or "OpenStreetMap" in str(
            provider
        ), f"Default provider should be OpenStreetMap, got {provider}"

    def test_resolve_cartodb_positron(self):
        """A dotted string resolves to a provider with `build_url`."""
        provider = get_provider("CartoDB.Positron")
        assert hasattr(provider, "build_url"), (
            f"Provider should have build_url method: {provider}"
        )

    def test_invalid_provider_raises_value_error(self):
        """An unknown provider name raises `ValueError`."""
        with pytest.raises(ValueError, match="Unknown tile provider"):
            get_provider("NonExistent.FakeProvider")

    def test_partial_invalid_name_raises_value_error(self):
        """A partially-valid name reports which segment failed."""
        with pytest.raises(ValueError, match="Failed at"):
            get_provider("OpenStreetMap.NonExistent")


class TestAutoZoom:
    """Tests for `cleopatra.basemap.tiles.auto_zoom`."""

    def test_global_extent_is_zoom_1(self):
        """A global extent spans two tiles across at the default floor."""
        assert auto_zoom((-180.0, -85.0, 180.0, 85.0)) == 1

    def test_city_extent_yields_zoom_11(self):
        """A ~0.6-degree city extent yields zoom 11 at the default floor."""
        assert auto_zoom((13.0, 52.4, 13.6, 52.6)) == 11

    def test_tiny_extent_clamps_to_19(self):
        """An infinitesimal extent clamps to the max zoom 19."""
        assert auto_zoom((0.0, 0.0, 1e-8, 1e-8)) == 19

    def test_min_tiles_across_one_restores_coarse_heuristic(self):
        """`min_tiles_across=1` reproduces the older one-tile-across zoom."""
        assert auto_zoom((-180.0, -85.0, 180.0, 85.0), min_tiles_across=1) == 0
        assert auto_zoom((13.0, 52.4, 13.6, 52.6), min_tiles_across=1) == 10

    def test_regional_extent_does_not_collapse_to_two_tiles(self):
        """A 6-11 degree extent (issue #176 Gulf) zooms past the coarse z6."""
        gulf = (-94.314, 27.439, -87.735, 30.867)
        assert auto_zoom(gulf) == 7  # default floor (2) -> sharper basemap
        assert auto_zoom(gulf, min_tiles_across=1) == 6  # old coarse value

    def test_non_positive_min_tiles_across_is_treated_as_one(self):
        """`min_tiles_across` below 1 is clamped to the one-tile heuristic."""
        assert auto_zoom((-180.0, -85.0, 180.0, 85.0), min_tiles_across=0) == 0


class TestDensifyAndReprojectBounds:
    """Tests for `cleopatra.basemap.tiles._densify_and_reproject_bounds`."""

    def test_4326_to_3857_produces_meters(self):
        """4326 -> 3857 produces bounds with absolute values in meters."""
        west, south, east, north = _densify_and_reproject_bounds(
            10.0, 50.0, 11.0, 51.0, "EPSG:4326", "EPSG:3857"
        )
        assert abs(west) > 100000, f"West should be in meters (large value), got {west}"
        assert west < east
        assert south < north

    def test_identity_transform_preserves_bounds(self):
        """4326 -> 4326 returns approximately the same bounds."""
        bounds = (10.0, 50.0, 11.0, 51.0)
        result = _densify_and_reproject_bounds(*bounds, "EPSG:4326", "EPSG:4326")
        for orig, reprojected in zip(bounds, result):
            assert abs(orig - reprojected) < 0.001


class TestAddTilesValidation:
    """Validation / error-path tests for `add_tiles`."""

    def test_raises_on_empty_axes(self):
        """Axes with default 0-1 limits raise `ValueError`."""
        ax = MagicMock()
        ax.get_xlim.return_value = (0.0, 1.0)
        ax.get_ylim.return_value = (0.0, 1.0)
        with pytest.raises(ValueError, match="no data extent"):
            add_tiles(ax)

    @pytest.mark.parametrize(
        "bad_ax",
        [None, "not_an_axes", 42, {}],
        ids=["none", "string", "int", "dict"],
    )
    def test_raises_on_invalid_ax_type(self, bad_ax):
        """Non-axes objects raise `TypeError`."""
        with pytest.raises(TypeError, match="matplotlib.axes.Axes"):
            add_tiles(bad_ax)

    @pytest.mark.parametrize(
        "bad_zoom",
        [-1, 20, 100, "invalid"],
        ids=["negative", "too_high", "way_too_high", "string"],
    )
    def test_raises_on_invalid_zoom(self, bad_zoom):
        """Invalid zoom values raise `ValueError`."""
        ax = MagicMock()
        ax.get_xlim.return_value = (1000000.0, 1200000.0)
        ax.get_ylim.return_value = (6000000.0, 6200000.0)
        ax.get_aspect.return_value = "auto"
        with pytest.raises(ValueError, match="zoom"):
            add_tiles(ax, crs=3857, zoom=bad_zoom)

    def test_invalid_source_string_raises_value_error(self):
        """A bogus source string raises `ValueError`."""
        ax = MagicMock()
        ax.get_xlim.return_value = (1000000.0, 1200000.0)
        ax.get_ylim.return_value = (6000000.0, 6200000.0)
        with pytest.raises(ValueError, match="Unknown tile provider"):
            add_tiles(ax, crs=3857, source="Bogus.NotARealProvider")

    def test_missing_extras_raises_import_error(self, monkeypatch):
        """If the tiles extra is unavailable, `ImportError` is raised."""
        monkeypatch.setattr(tiles_mod, "_TILES_AVAILABLE", False)
        ax = MagicMock()
        ax.get_xlim.return_value = (1000000.0, 1200000.0)
        ax.get_ylim.return_value = (6000000.0, 6200000.0)
        with pytest.raises(ImportError, match=r"cleopatra\[tiles\]"):
            add_tiles(ax)


@pytest.fixture
def mock_ax():
    """Return a `MagicMock` axes with a realistic Web Mercator extent."""
    ax = MagicMock()
    ax.get_xlim.return_value = (1000000.0, 1200000.0)
    ax.get_ylim.return_value = (6000000.0, 6200000.0)
    ax.get_aspect.return_value = "auto"

    mock_transform = MagicMock()
    mock_transform.inverted.return_value = mock_transform
    mock_fig = MagicMock()
    mock_fig.dpi = 100.0
    type(mock_fig).dpi_scale_trans = PropertyMock(return_value=mock_transform)

    mock_bbox = MagicMock()
    mock_bbox.width = 6.0
    mock_bbox.height = 4.0
    mock_bbox.transformed.return_value = mock_bbox

    ax.get_figure.return_value = mock_fig
    ax.get_window_extent.return_value = mock_bbox
    return ax


@pytest.fixture
def _patch_tiles():
    """Patch `auto_zoom`, `fetch_tiles`, `stitch_tiles`."""
    fake_image = np.zeros((256, 256, 4), dtype=np.uint8)
    with (
        patch.object(tiles_mod, "auto_zoom", return_value=10) as mock_zoom,
        patch.object(
            tiles_mod,
            "fetch_tiles",
            return_value={Tile(0, 0, 10): _make_tile_png()},
        ) as mock_fetch,
        patch.object(
            tiles_mod,
            "stitch_tiles",
            return_value=(
                fake_image,
                (1000000.0, 6000000.0, 1200000.0, 6200000.0),
            ),
        ) as mock_stitch,
    ):
        yield mock_zoom, mock_fetch, mock_stitch


class TestAddTilesBehaviour:
    """Behavioural tests for `add_tiles` (mocked HTTP layer)."""

    def test_default_source_renders_image(self, mock_ax, _patch_tiles):
        """`source=None` -> default provider; imshow is called once."""
        result = add_tiles(mock_ax)
        mock_ax.imshow.assert_called_once()
        assert result is mock_ax

    def test_default_crs_is_3857(self, mock_ax, _patch_tiles):
        """`crs=None` treats the data as Web Mercator (no error)."""
        result = add_tiles(mock_ax, crs=None)
        mock_ax.imshow.assert_called_once()
        assert result is mock_ax

    def test_string_source_renders(self, mock_ax, _patch_tiles):
        """`source="CartoDB.Positron"` resolves and renders."""
        result = add_tiles(mock_ax, source="CartoDB.Positron")
        mock_ax.imshow.assert_called_once()
        assert result is mock_ax

    def test_explicit_crs_4326_renders(self, mock_ax, _patch_tiles):
        """`crs=4326` works end-to-end (no GDAL warping needed)."""
        mock_ax.get_xlim.return_value = (10.0, 11.0)
        mock_ax.get_ylim.return_value = (50.0, 51.0)
        add_tiles(mock_ax, crs=4326)
        mock_ax.imshow.assert_called_once()

    def test_nonmercator_imshow_uses_mosaic_reprojected_bounds(
        self, mock_ax, _patch_tiles
    ):
        """For a lon/lat axis the imshow extent is the mosaic's reprojected 3857
        coverage, not the raw data bounds (issue #176)."""
        mock_ax.get_xlim.return_value = (10.0, 11.0)
        mock_ax.get_ylim.return_value = (50.0, 51.0)
        add_tiles(mock_ax, crs=4326)
        # Independent expected value: the mocked mosaic 3857 bounds
        # (1e6, 6e6, 1.2e6, 6.2e6) reproject to these EPSG:4326 [w, e, s, n]
        # degrees (precomputed, not re-derived from the production helper).
        got = mock_ax.imshow.call_args.kwargs["extent"]
        assert got == pytest.approx([8.9832, 10.7798, 47.3537, 48.5569], abs=1e-4), (
            f"imshow extent should be the mosaic's reprojected bounds, got {got}"
        )
        assert got != [10.0, 11.0, 50.0, 51.0], "extent must not be the raw data bounds"

    def test_nonmercator_falls_back_to_data_bounds_when_mosaic_overflows(self, mock_ax):
        """A mosaic overflowing a singular projection falls back to the data bounds,
        rendering instead of raising (issue #176 regression guard)."""
        pytest.importorskip("pyproj", reason="pyproj not installed (tiles extra)")
        ortho = "+proj=ortho +lat_0=0 +lon_0=0"  # undefined on the far hemisphere
        mock_ax.get_xlim.return_value = (0.0, 100000.0)  # ortho metres near centre
        mock_ax.get_ylim.return_value = (0.0, 100000.0)
        world = 20037508.342789244  # half the Web Mercator world extent (metres)
        with (
            patch.object(tiles_mod, "auto_zoom", return_value=0),
            patch.object(
                tiles_mod, "fetch_tiles", return_value={Tile(0, 0, 0): _make_tile_png()}
            ),
            patch.object(
                tiles_mod,
                "stitch_tiles",
                return_value=(
                    np.zeros((256, 256, 4), dtype=np.uint8),
                    (-world, -world, world, world),  # whole-world mosaic
                ),
            ),
        ):
            add_tiles(mock_ax, crs=ortho)
        mock_ax.imshow.assert_called_once()
        got = mock_ax.imshow.call_args.kwargs["extent"]
        assert got == [
            0.0,
            100000.0,
            0.0,
            100000.0,
        ], f"overflow should fall back to the data bounds, got {got}"

    def test_nonmercator_mosaic_extent_envelops_data(self, mock_ax):
        """A mosaic larger than the data is placed enveloping the data bounds (issue #176)."""
        pytest.importorskip("pyproj", reason="pyproj not installed (tiles extra)")
        mock_ax.get_xlim.return_value = (10.0, 11.0)  # lon/lat data extent
        mock_ax.get_ylim.return_value = (50.0, 51.0)
        # A mosaic whose Web-Mercator coverage is wider than the data's:
        mosaic_3857 = (1.0e6, 6.3e6, 1.35e6, 6.75e6)
        with (
            patch.object(tiles_mod, "auto_zoom", return_value=6),
            patch.object(
                tiles_mod, "fetch_tiles", return_value={Tile(0, 0, 6): _make_tile_png()}
            ),
            patch.object(
                tiles_mod,
                "stitch_tiles",
                return_value=(np.zeros((256, 256, 4), dtype=np.uint8), mosaic_3857),
            ),
        ):
            add_tiles(mock_ax, crs=4326)
        west, east, south, north = mock_ax.imshow.call_args.kwargs["extent"]
        assert west <= 10.0 and east >= 11.0 and south <= 50.0 and north >= 51.0, (
            f"mosaic extent {(west, east, south, north)} should envelop the data bounds"
        )

    def test_min_tiles_across_forwarded_to_auto_zoom(self, mock_ax, _patch_tiles):
        """`add_tiles(min_tiles_across=...)` is forwarded to `auto_zoom` for zoom='auto'."""
        mock_zoom, _fetch, _stitch = _patch_tiles
        add_tiles(mock_ax, crs=3857, min_tiles_across=6)
        mock_zoom.assert_called_once()
        assert mock_zoom.call_args.kwargs.get("min_tiles_across") == 6

    def test_explicit_zoom_ignores_min_tiles_across(self, mock_ax, _patch_tiles):
        """An explicit `zoom=` bypasses `auto_zoom` (and thus `min_tiles_across`)."""
        mock_zoom, _fetch, _stitch = _patch_tiles
        add_tiles(mock_ax, crs=3857, zoom=5, min_tiles_across=6)
        mock_zoom.assert_not_called()

    def test_axes_limits_are_restored(self, mock_ax, _patch_tiles):
        """`set_xlim` / `set_ylim` are called with the original limits."""
        add_tiles(mock_ax, crs=3857)
        mock_ax.set_xlim.assert_called_once_with((1000000.0, 1200000.0))
        mock_ax.set_ylim.assert_called_once_with((6000000.0, 6200000.0))

    def test_attribution_false_skips_text(self, mock_ax, _patch_tiles):
        """`attribution=False` -> `ax.text` is never called."""
        add_tiles(mock_ax, crs=3857, attribution=False)
        mock_ax.text.assert_not_called()

    def test_custom_attribution_string(self, mock_ax, _patch_tiles):
        """`attribution="Custom"` -> exact string is written to axes."""
        add_tiles(mock_ax, crs=3857, attribution="Custom Attribution")
        mock_ax.text.assert_called_once()
        call_args = mock_ax.text.call_args
        assert call_args[0][2] == "Custom Attribution"

    def test_imshow_receives_alpha_and_zorder(self, mock_ax, _patch_tiles):
        """Custom `alpha` and `zorder` are forwarded to `imshow`."""
        add_tiles(mock_ax, crs=3857, alpha=0.5, zorder=-2)
        call_kwargs = mock_ax.imshow.call_args[1]
        assert call_kwargs["alpha"] == 0.5
        assert call_kwargs["zorder"] == -2

    def test_default_user_agent_propagates_to_fetch_tiles(self, mock_ax, _patch_tiles):
        """With no `user_agent` the module default is forwarded to `fetch_tiles`."""
        from cleopatra.basemap.tiles import USER_AGENT

        _, mock_fetch, _ = _patch_tiles
        add_tiles(mock_ax, crs=3857)
        assert mock_fetch.call_args.kwargs.get("user_agent") == USER_AGENT

    def test_custom_user_agent_propagates_to_fetch_tiles(self, mock_ax, _patch_tiles):
        """`add_tiles(user_agent=...)` is forwarded verbatim to `fetch_tiles`."""
        _, mock_fetch, _ = _patch_tiles
        add_tiles(mock_ax, crs=3857, user_agent="myapp/1.0 (+https://example.test)")
        assert (
            mock_fetch.call_args.kwargs.get("user_agent")
            == "myapp/1.0 (+https://example.test)"
        )


class TestAddTilesIntegration:
    """End-to-end integration tests against a real matplotlib axes."""

    def test_savefig_roundtrip(self, tmp_path):
        """Render tiles on a real axes, save to PNG, verify non-empty file."""
        fake_image = np.zeros((256, 256, 4), dtype=np.uint8)
        fake_image[..., :3] = 200
        fake_image[..., 3] = 255

        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.plot([1000000.0, 1200000.0], [6000000.0, 6200000.0])
        try:
            with (
                patch.object(tiles_mod, "auto_zoom", return_value=10),
                patch.object(
                    tiles_mod,
                    "fetch_tiles",
                    return_value={Tile(0, 0, 10): _make_tile_png()},
                ),
                patch.object(
                    tiles_mod,
                    "stitch_tiles",
                    return_value=(
                        fake_image,
                        (1000000.0, 6000000.0, 1200000.0, 6200000.0),
                    ),
                ),
            ):
                add_tiles(ax, crs=3857)

            out = tmp_path / "tiles.png"
            fig.savefig(out)
            assert out.exists(), f"Expected {out} to exist after savefig"
            assert out.stat().st_size > 0, "PNG file should be non-empty"
            images = [
                child
                for child in ax.get_children()
                if isinstance(child, matplotlib.image.AxesImage)
            ]
            assert images, "Expected at least one AxesImage on the axes"
        finally:
            plt.close(fig)

    def test_max_tiles_reduces_zoom(self, mock_ax):
        """Zoom is decreased when the requested level needs > MAX_TILES tiles."""
        fake_image = np.zeros((256, 256, 4), dtype=np.uint8)
        many_tiles = [Tile(x=i, y=j, z=10) for i in range(20) for j in range(20)]
        few_tiles = [Tile(x=i, y=j, z=9) for i in range(10) for j in range(10)]

        with (
            patch.object(tiles_mod, "auto_zoom", return_value=10),
            patch.object(
                tiles_mod,
                "fetch_tiles",
                return_value={Tile(0, 0, 9): _make_tile_png()},
            ),
            patch.object(
                tiles_mod,
                "stitch_tiles",
                return_value=(
                    fake_image,
                    (1000000.0, 6000000.0, 1200000.0, 6200000.0),
                ),
            ),
            patch.object(
                tiles_mod,
                "_tiles_for_bbox",
                side_effect=[many_tiles, few_tiles],
            ) as mock_tiles,
        ):
            add_tiles(mock_ax, crs=3857)

        calls = mock_tiles.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["zoom"] == 10
        assert calls[1][1]["zoom"] == 9

    def test_custom_max_tiles_relaxes_reduction(self, mock_ax):
        """A higher `max_tiles=` avoids the zoom reduction (N2)."""
        fake_image = np.zeros((256, 256, 4), dtype=np.uint8)
        # 400 tiles at the requested zoom — over the default 256, under 500.
        many_tiles = [Tile(x=i, y=j, z=10) for i in range(20) for j in range(20)]

        with (
            patch.object(tiles_mod, "auto_zoom", return_value=10),
            patch.object(
                tiles_mod,
                "fetch_tiles",
                return_value={Tile(0, 0, 10): _make_tile_png()},
            ),
            patch.object(
                tiles_mod,
                "stitch_tiles",
                return_value=(fake_image, (1e6, 6e6, 1.2e6, 6.2e6)),
            ),
            patch.object(
                tiles_mod, "_tiles_for_bbox", side_effect=[many_tiles]
            ) as mock_tiles,
        ):
            add_tiles(mock_ax, crs=3857, max_tiles=500)

        # Only one call: 400 <= max_tiles=500, so no reduction.
        assert len(mock_tiles.call_args_list) == 1
        assert mock_tiles.call_args_list[0][1]["zoom"] == 10

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True, "8"])
    def test_invalid_max_tiles_raises(self, mock_ax, bad):
        """`max_tiles` must be a positive int (N2).

        Args:
            bad: A value that should be rejected.
        """
        with pytest.raises(ValueError, match="max_tiles must be a positive int"):
            add_tiles(mock_ax, crs=3857, max_tiles=bad)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True, "8", None])
    def test_invalid_min_tiles_across_raises(self, mock_ax, bad):
        """`min_tiles_across` must be a positive int, rejected at the boundary.

        Args:
            bad: A value that should be rejected.
        """
        with pytest.raises(ValueError, match="min_tiles_across must be a positive int"):
            add_tiles(mock_ax, crs=3857, min_tiles_across=bad)


class TestRequireTilesExtra:
    """Tests for `cleopatra.basemap.tiles._require_tiles_extra` guard."""

    def test_available_returns_silently(self):
        """When deps are present, the helper is a no-op and returns `None`."""
        result = _require_tiles_extra()
        assert result is None, (
            f"_require_tiles_extra should return None on success, got {result!r}"
        )

    def test_missing_raises_with_install_hint(self, monkeypatch):
        """When `_TILES_AVAILABLE` is False, raise `ImportError` with the hint."""
        monkeypatch.setattr(tiles_mod, "_TILES_AVAILABLE", False)
        with pytest.raises(ImportError, match=r"cleopatra\[tiles\]"):
            _require_tiles_extra()


class TestAutoZoomEdgeCases:
    """Boundary tests for `cleopatra.basemap.tiles.auto_zoom`."""

    def test_zero_extent_clamps_to_max(self):
        """A zero-area extent (west == east, south == north) clamps to zoom 19."""
        result = auto_zoom((0.0, 0.0, 0.0, 0.0))
        assert result == 19, f"Zero extent should clamp to 19, got {result}"

    def test_negative_extent_uses_absolute_value(self):
        """Reversed bounds (`east < west`) still produce a non-negative zoom."""
        result = auto_zoom((10.0, 10.0, 5.0, 5.0))
        assert 0 <= result <= 19, f"Zoom must be in [0, 19], got {result}"

    @pytest.mark.parametrize(
        "bounds, expected_min",
        [
            ((-180.0, -85.0, 180.0, 85.0), 0),
            ((0.0, 0.0, 180.0, 90.0), 1),
        ],
        ids=["global", "hemisphere"],
    )
    def test_known_extents(self, bounds, expected_min):
        """Manual sanity for hand-computed zoom values.

        Args:
            bounds: `(west, south, east, north)` in degrees.
            expected_min: Lower bound on the expected zoom value.
        """
        result = auto_zoom(bounds)
        assert result >= expected_min, (
            f"auto_zoom{bounds} should be >= {expected_min}, got {result}"
        )


class TestDensifyAndReprojectEdgeCases:
    """Edge-case tests for `_densify_and_reproject_bounds`."""

    def test_n_points_default_runs(self):
        """`n_points` default of 21 produces sensible output bounds."""
        west, south, east, north = _densify_and_reproject_bounds(
            -10.0, -5.0, 10.0, 5.0, "EPSG:4326", "EPSG:3857"
        )
        assert west < east, f"west {west} should be < east {east}"
        assert south < north, f"south {south} should be < north {north}"

    def test_invalid_reprojection_raises_value_error(self):
        """Reprojecting through an invalid CRS pair surfaces as `ValueError`.

        Test scenario:
            Patch `pyproj.Transformer.from_crs` so the transform
            yields infinite coordinates; the helper must raise
            `ValueError` with a clear message rather than silently
            returning garbage bounds.
        """
        with patch("pyproj.Transformer.from_crs") as mock_from_crs:
            mock_transformer = MagicMock()
            mock_transformer.transform.return_value = (
                np.array([np.inf, np.inf]),
                np.array([np.inf, np.inf]),
            )
            mock_from_crs.return_value = mock_transformer
            with pytest.raises(ValueError, match="infinite or NaN"):
                _densify_and_reproject_bounds(
                    10.0,
                    50.0,
                    11.0,
                    51.0,
                    "EPSG:4326",
                    "EPSG:3857",
                    n_points=2,
                )

    def test_n_points_low_value_runs(self):
        """`n_points=2` (only corners) still produces finite bounds."""
        west, south, east, north = _densify_and_reproject_bounds(
            10.0,
            50.0,
            11.0,
            51.0,
            "EPSG:4326",
            "EPSG:3857",
            n_points=2,
        )
        assert all(np.isfinite([west, south, east, north])), (
            f"Bounds should all be finite, got ({west}, {south}, {east}, {north})"
        )


class TestTile:
    """Tests for `cleopatra.basemap.tiles.Tile`."""

    def test_fields_are_positional_and_named(self):
        """`Tile(x, y, z)` stores each field, accessible both by name and by index."""
        tile = Tile(1, 2, 3)
        assert (tile.x, tile.y, tile.z) == (1, 2, 3), f"unexpected fields: {tile}"
        assert (tile[0], tile[1], tile[2]) == (1, 2, 3), f"unexpected indexing: {tile}"

    def test_equality_is_by_value(self):
        """Two independently-constructed `Tile`s with the same fields compare equal."""
        assert Tile(1, 2, 3) == Tile(x=1, y=2, z=3)
        assert Tile(1, 2, 3) != Tile(1, 2, 4)

    def test_hashable_as_dict_key(self):
        """`Tile` is hashable, so it can key a `{tile: data}` mapping (as `fetch_tiles` does)."""
        mapping = {Tile(0, 0, 0): b"a", Tile(1, 0, 1): b"b"}
        assert mapping[Tile(0, 0, 0)] == b"a"
        assert mapping[Tile(1, 0, 1)] == b"b"

    def test_repr_is_readable(self):
        """`repr(Tile(...))` names the fields, matching the module's own doctest."""
        assert repr(Tile(0, 0, 0)) == "Tile(x=0, y=0, z=0)"


class TestLonLatToTileXY:
    """Tests for `cleopatra.basemap.tiles._lonlat_to_tile_xy`.

    Expected `(x, y)` pairs are cross-checked against the real `mercantile`
    package's `mercantile.tile()` (see the removal PR's commit messages for
    the full equivalence-fuzzing methodology); the literal values here just
    lock those results in as a regression test that doesn't need
    `mercantile` installed to run.
    """

    @pytest.mark.parametrize(
        "lon, lat, zoom, expected",
        [
            (0.0, 0.0, 0, (0, 0)),
            (13.4, 52.5, 10, (550, 335)),
            (-179.9, 89.9, 3, (0, 0)),
            (179.9, -89.9, 3, (7, 7)),
            (0.0, 0.0, 1, (1, 1)),
        ],
        ids=["world-origin-z0", "berlin-z10", "nw-corner", "se-corner", "equator-z1"],
    )
    def test_known_points(self, lon, lat, zoom, expected):
        """Known (lon, lat, zoom) triples resolve to their mercantile-verified tile index."""
        assert _lonlat_to_tile_xy(lon, lat, zoom) == expected

    @pytest.mark.parametrize(
        "lon, expected_x",
        [(200.0, 31), (-200.0, 0), (180.0, 31), (-180.0, 0)],
        ids=["past-east-edge", "past-west-edge", "east-edge", "west-edge"],
    )
    def test_out_of_range_longitude_clamps_to_edge_column(self, lon, expected_x):
        """A longitude outside +/-180 (or exactly on it) clamps to the edge column, not raise."""
        x, _y = _lonlat_to_tile_xy(lon, 0.0, zoom=5)
        assert x == expected_x

    @pytest.mark.parametrize(
        "lat, expected_y",
        [(90.0, 0), (-90.0, 31), (89.99999999, 0), (-89.99999999, 31)],
        ids=["exact-north-pole", "exact-south-pole", "near-north-pole", "near-south-pole"],
    )
    def test_pole_latitude_clamps_instead_of_crashing(self, lat, expected_y):
        """A latitude at (or within float precision of) +/-90 clamps to the edge row.

        Regression test for a `ZeroDivisionError`/`ValueError: math domain
        error` that used to be raised here: `sin(radians(lat))` rounds to
        exactly +/-1.0 within ~6e-7 degrees of either pole, dividing by zero
        in the Mercator `y` formula.
        """
        _x, y = _lonlat_to_tile_xy(0.0, lat, zoom=5)
        assert y == expected_y


class TestTilesForBbox:
    """Tests for `cleopatra.basemap.tiles._tiles_for_bbox`.

    Expected tile lists are cross-checked against `mercantile.tiles()` (see
    the removal PR's commit messages); the literal values here lock those
    results in without needing `mercantile` installed to run.
    """

    def test_whole_world_at_zoom_0_is_one_tile(self):
        """The whole world at zoom 0 is exactly the single root tile."""
        assert _tiles_for_bbox(-180.0, -85.0, 180.0, 85.0, zoom=0) == [Tile(0, 0, 0)]

    def test_known_city_bbox(self):
        """A real-world (Berlin) bbox resolves to its mercantile-verified 2x3 tile grid."""
        tiles = sorted(_tiles_for_bbox(13.0, 52.4, 13.6, 52.6, zoom=10))
        expected = sorted(
            Tile(x, y, 10)
            for x in (548, 549, 550)
            for y in (335, 336)
        )
        assert tiles == expected

    def test_bbox_exactly_matching_one_tile_resolves_to_that_tile(self):
        """A bbox exactly on one tile's own lon/lat bounds resolves to just that tile."""
        tiles = _tiles_for_bbox(
            13.359375, 52.48278022207821, 13.7109375, 52.69636107827448, zoom=10
        )
        assert tiles == [Tile(550, 335, 10)]

    def test_degenerate_point_bbox_resolves_to_one_tile(self):
        """A zero-area (single-point) bbox still resolves to the one tile containing it."""
        tiles = _tiles_for_bbox(13.4, 52.5, 13.4, 52.5, zoom=10)
        assert tiles == [Tile(550, 335, 10)]

    def test_zero_width_bbox_resolves_to_a_single_column(self):
        """A `west == east` bbox (zero width, non-zero height) still resolves sanely.

        Test scenario:
            A degenerate box that collapses only one axis (unlike the
            single-point case above, which collapses both) should still
            return every tile row the non-degenerate `south`/`north` span
            covers, all in the same column.
        """
        tiles = sorted(_tiles_for_bbox(13.4, 52.4, 13.4, 52.6, zoom=10))
        assert tiles == [Tile(550, 335, 10), Tile(550, 336, 10)]

    def test_high_zoom_small_bbox(self):
        """A small bbox at zoom 18 resolves to the mercantile-verified 8x13 tile grid."""
        tiles = _tiles_for_bbox(13.4, 52.5, 13.41, 52.51, zoom=18)
        xs = sorted({t.x for t in tiles})
        ys = sorted({t.y for t in tiles})
        assert len(tiles) == 104
        assert (xs[0], xs[-1]) == (140829, 140836)
        assert (ys[0], ys[-1]) == (85983, 85995)

    def test_bbox_touching_north_pole_clamps_instead_of_crashing(self):
        """A bbox whose `north` is exactly 90 clamps to the top row rather than crashing."""
        tiles = sorted(_tiles_for_bbox(-1.0, 89.0, 1.0, 90.0, zoom=5))
        assert tiles == [Tile(15, 0, 5), Tile(16, 0, 5)]

    def test_bbox_touching_south_pole_clamps_instead_of_crashing(self):
        """A bbox whose `south` is exactly -90 clamps to the bottom row rather than crashing."""
        tiles = sorted(_tiles_for_bbox(-1.0, -90.0, 1.0, -89.0, zoom=5))
        assert tiles == [Tile(15, 31, 5), Tile(16, 31, 5)]

    def test_antimeridian_crossing_bbox_splits_like_mercantile(self):
        """A `west > east` bbox (antimeridian-crossing) splits into both dateline halves.

        Regression test for a `ValueError` this used to raise instead: a
        `west > east` bbox is not just a hand-crafted edge case, it is a
        real, reachable input through `add_tiles` -- reprojecting a
        near-global Web Mercator extent to EPSG:4326 wraps longitude at
        the +/-180 seam, producing exactly this shape (see
        `TestAddTilesNearGlobalExtent`). `mercantile.tiles()` handles it by
        splitting into `[-180, east]` and `[west, 180]`; this must match.
        """
        tiles = sorted(_tiles_for_bbox(170.0, -10.0, -170.0, 10.0, zoom=4))
        expected = sorted([Tile(0, 7, 4), Tile(0, 8, 4), Tile(15, 7, 4), Tile(15, 8, 4)])
        assert tiles == expected

    def test_antimeridian_crossing_bbox_also_touching_a_pole(self):
        """An antimeridian-crossing bbox that also clips the north pole splits and clamps.

        Both the west>east split and the latitude clamp apply at once;
        verified to match `mercantile.tiles()` exactly for this combination.
        """
        tiles = sorted(_tiles_for_bbox(170.0, 80.0, -170.0, 85.0, zoom=4))
        expected = sorted([Tile(0, 0, 4), Tile(0, 1, 4), Tile(15, 0, 4), Tile(15, 1, 4)])
        assert tiles == expected

    def test_antimeridian_crossing_bbox_at_zoom_0(self):
        """An antimeridian-crossing bbox at zoom 0 matches mercantile's own duplicate-tile quirk.

        At zoom 0 there is only one tile in the whole world, so both
        dateline-side sub-boxes resolve to it; `mercantile.tiles()` does
        not deduplicate across the split and returns it twice, and this
        implementation matches that exactly rather than silently
        "improving" on it.
        """
        tiles = _tiles_for_bbox(170.0, -10.0, -170.0, 10.0, zoom=0)
        assert tiles == [Tile(0, 0, 0), Tile(0, 0, 0)]

    def test_antimeridian_crossing_bbox_with_east_exactly_on_seam(self):
        """An antimeridian-crossing bbox whose `east` sits exactly on `-180` still splits correctly."""
        tiles = sorted(_tiles_for_bbox(170.0, -10.0, -180.0, 10.0, zoom=4))
        expected = sorted([Tile(0, 7, 4), Tile(0, 8, 4), Tile(15, 7, 4), Tile(15, 8, 4)])
        assert tiles == expected


class TestTileXYBounds:
    """Tests for `cleopatra.basemap.tiles._tile_xy_bounds`.

    Expected bounds are cross-checked against `mercantile.xy_bounds()` (see
    the removal PR's commit messages); the literal values here lock those
    results in without needing `mercantile` installed to run.
    """

    def test_whole_world_tile(self):
        """The root tile's bounds span the full EPSG:3857 world extent."""
        left, bottom, right, top = _tile_xy_bounds(Tile(0, 0, 0))
        assert left == pytest.approx(-20037508.342789244)
        assert bottom == pytest.approx(-20037508.342789244)
        assert right == pytest.approx(20037508.342789244)
        assert top == pytest.approx(20037508.342789244)

    def test_quadrant_tile(self):
        """Tile (1, 1, 1) is the southeast quadrant of the world."""
        left, bottom, right, top = _tile_xy_bounds(Tile(1, 1, 1))
        assert left == pytest.approx(0.0)
        assert bottom == pytest.approx(-20037508.342789244)
        assert right == pytest.approx(20037508.342789244)
        assert top == pytest.approx(0.0)

    def test_known_tile(self):
        """A real-world tile's bounds match mercantile's `xy_bounds` exactly."""
        left, bottom, right, top = _tile_xy_bounds(Tile(550, 335, 10))
        assert left == pytest.approx(1487158.8223163895)
        assert bottom == pytest.approx(6887893.492833803)
        assert right == pytest.approx(1526294.5807983999)
        assert top == pytest.approx(6927029.2513158135)


class TestFetchSingleTile:
    """Tests for `cleopatra.basemap.tiles.fetch_single_tile`."""

    def _make_provider(self) -> MagicMock:
        """Build a mock provider that returns a stable tile URL."""
        provider = MagicMock()
        provider.build_url = MagicMock(return_value="http://example.test/0/0/0.png")
        return provider

    def test_succeeds_on_valid_png(self):
        """A valid PNG response returns `(tile, bytes)` on the first try."""
        png = _make_tile_png(size=64)
        tile = Tile(0, 0, 0)
        provider = self._make_provider()

        with patch("cleopatra.basemap.tiles.urlopen_http") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = png
            mock_urlopen.return_value = mock_response

            returned_tile, returned_bytes = fetch_single_tile(
                tile, provider, timeout=5, retries=0
            )

        assert returned_tile is tile, "Should return the original tile"
        assert returned_bytes == png, "Should return the PNG payload unchanged"

    def test_invalid_image_bytes_treated_as_failure(self):
        """A non-image response triggers retries and ultimately raises."""
        tile = Tile(1, 2, 3)
        provider = self._make_provider()

        with patch("cleopatra.basemap.tiles.urlopen_http") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"<html>not-an-image</html>"
            mock_urlopen.return_value = mock_response

            with pytest.raises(ConnectionError, match="Failed to fetch tile"):
                fetch_single_tile(tile, provider, timeout=1, retries=1)
            assert mock_urlopen.call_count == 2, (
                f"Expected 2 attempts (retries=1), got {mock_urlopen.call_count}"
            )

    def test_retries_and_succeeds(self):
        """A transient `URLError` is retried and a later success is returned."""
        import urllib.error

        tile = Tile(0, 0, 0)
        provider = self._make_provider()
        png = _make_tile_png()

        successful_response = MagicMock()
        successful_response.read.return_value = png

        with patch("cleopatra.basemap.tiles.urlopen_http") as mock_urlopen:
            mock_urlopen.side_effect = [
                urllib.error.URLError("transient"),
                successful_response,
            ]
            _, returned_bytes = fetch_single_tile(tile, provider, timeout=1, retries=2)
        assert returned_bytes == png, (
            f"Expected png bytes after retry, got {len(returned_bytes)} bytes"
        )

    def test_raises_after_all_retries_exhausted(self):
        """All retries failing raises `ConnectionError` referencing the tile."""
        import urllib.error

        tile = Tile(5, 6, 7)
        provider = self._make_provider()

        with patch("cleopatra.basemap.tiles.urlopen_http") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("permanent")
            with pytest.raises(ConnectionError, match="z=7/x=5/y=6"):
                fetch_single_tile(tile, provider, timeout=1, retries=2)
            assert mock_urlopen.call_count == 3, (
                f"Expected 3 attempts, got {mock_urlopen.call_count}"
            )

    @pytest.mark.parametrize(
        "header",
        [
            b"\xff\xd8\xff\xe0",  # JFIF
            b"\xff\xd8\xff\xe1",  # EXIF
            b"\xff\xd8\xff\xe2",  # ICC / SPIFF APP2
            b"\xff\xd8\xff\xe8",  # SPIFF APP8
            b"\xff\xd8\xff\xef",  # APP15
            b"\xff\xd8\xff\xdb",  # bare SOI + DQT (some progressive JPEGs)
            b"\xff\xd8\xff\xc0",  # bare SOI + SOF0
            b"GIF89a",  # GIF
            b"RIFF\x00\x00\x00\x00WEBP",  # WebP
        ],
        ids=[
            "jpeg-app0",
            "jpeg-app1",
            "jpeg-app2",
            "jpeg-app8",
            "jpeg-app15",
            "jpeg-dqt",
            "jpeg-sof0",
            "gif",
            "webp",
        ],
    )
    def test_non_png_image_headers_accepted(self, header):
        """Tile bodies starting with any common image signature pass through.

        Regression for the bug where only `\\xff\\xd8\\xff\\xe0`/`\\xe1` JPEG
        markers were accepted, so progressive/EXIF/ICC JPEGs (and GIF/WebP)
        were treated as fetch failures and retried into a `ConnectionError`.

        Args:
            header: A leading byte sequence for a valid image format.
        """
        body = header + b"\x00" * 64
        tile = Tile(0, 0, 0)
        provider = self._make_provider()
        with patch("cleopatra.basemap.tiles.urlopen_http") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = body
            mock_urlopen.return_value = mock_response
            _, returned_bytes = fetch_single_tile(tile, provider, timeout=1, retries=0)
        assert returned_bytes == body, "image bytes should pass through unchanged"

    def _captured_user_agent(self, mock_urlopen) -> str:
        """Extract the User-Agent header from the Request passed to urlopen."""
        request = mock_urlopen.call_args[0][0]
        # exactly one header is set on the request
        return next(iter(request.headers.values()))

    def test_default_user_agent_is_versioned(self):
        """The default User-Agent identifies cleopatra with a version and URL."""
        from cleopatra.basemap.tiles import USER_AGENT

        png = _make_tile_png(size=32)
        provider = self._make_provider()
        with patch("cleopatra.basemap.tiles.urlopen_http") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = png
            mock_urlopen.return_value = mock_response
            fetch_single_tile(Tile(0, 0, 0), provider, timeout=1, retries=0)
        ua = self._captured_user_agent(mock_urlopen)
        assert ua == USER_AGENT
        assert ua.startswith("cleopatra/"), f"UA should start with 'cleopatra/': {ua!r}"
        assert "github.com/serapeum-org/cleopatra" in ua, (
            f"UA should carry a contact URL: {ua!r}"
        )
        assert ua != "cleopatra/Python", "the old placeholder UA must be gone"

    def test_custom_user_agent_is_sent_verbatim(self):
        """A `user_agent=` override is sent on the request unchanged."""
        png = _make_tile_png(size=32)
        provider = self._make_provider()
        custom = "myapp/2.0 (+https://example.test)"
        with patch("cleopatra.basemap.tiles.urlopen_http") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = png
            mock_urlopen.return_value = mock_response
            fetch_single_tile(
                Tile(0, 0, 0), provider, timeout=1, retries=0, user_agent=custom
            )
        assert self._captured_user_agent(mock_urlopen) == custom


class TestLooksLikeImage:
    """Unit tests for `cleopatra.basemap.tiles._looks_like_image`."""

    @pytest.mark.parametrize(
        "data",
        [
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 8,
            b"\xff\xd8\xff\xe0" + b"\x00" * 8,
            b"\xff\xd8\xff\xe2" + b"\x00" * 8,
            b"\xff\xd8\xff\xdb" + b"\x00" * 8,
            b"\xff\xd8\xff\xc0" + b"\x00" * 8,
            b"GIF87a" + b"\x00" * 8,
            b"GIF89a" + b"\x00" * 8,
            b"RIFF\x00\x00\x00\x00WEBP\x00\x00\x00\x00",
        ],
        ids=[
            "png",
            "jpeg-app0",
            "jpeg-app2",
            "jpeg-dqt",
            "jpeg-sof0",
            "gif87a",
            "gif89a",
            "webp",
        ],
    )
    def test_accepts_known_signatures(self, data):
        """Every recognised raster signature returns True.

        Args:
            data: A byte string that begins with a known image signature.
        """
        assert _looks_like_image(data) is True

    @pytest.mark.parametrize(
        "data",
        [
            b"",
            b"<html><body>404 Not Found</body></html>",
            b"{\"error\": \"forbidden\"}",
            b"\x00\x00\x00\x00",
            b"RIFF\x00\x00\x00\x00WAVE",  # RIFF but not WebP
            b"\xff\xd8",  # truncated SOI, not enough bytes
        ],
        ids=["empty", "html", "json", "zeros", "riff-not-webp", "truncated-soi"],
    )
    def test_rejects_non_images(self, data):
        """Empty bodies, error pages and non-image payloads return False.

        Args:
            data: A byte string that is not a recognised image.
        """
        assert _looks_like_image(data) is False


class TestFetchTiles:
    """Tests for `cleopatra.basemap.tiles.fetch_tiles`."""

    def test_returns_dict_keyed_by_tile(self):
        """Successful fetch produces a `{tile: bytes}` mapping."""
        tiles = [Tile(0, 0, 0), Tile(1, 0, 0)]
        png = _make_tile_png()
        provider = MagicMock()
        provider.build_url = MagicMock(return_value="http://example.test/")

        def fake_single(tile, _provider, _timeout, _retries, _user_agent=None):
            return tile, png

        with patch.object(tiles_mod, "fetch_single_tile", side_effect=fake_single):
            result = fetch_tiles(tiles, provider, max_workers=2, timeout=1, retries=0)

        assert set(result.keys()) == set(tiles), (
            f"Result should be keyed by all input tiles, got {set(result.keys())}"
        )
        for v in result.values():
            assert v == png, "All tile values should be the mocked PNG bytes"

    def test_propagates_connection_error(self):
        """If any tile fails permanently, `ConnectionError` propagates."""
        tiles = [Tile(0, 0, 0)]
        provider = MagicMock()

        with patch.object(
            tiles_mod,
            "fetch_single_tile",
            side_effect=ConnectionError("kaboom"),
        ):
            with pytest.raises(ConnectionError, match="kaboom"):
                fetch_tiles(tiles, provider, max_workers=1, timeout=1, retries=0)

    def test_unexpected_exception_re_raises(self):
        """Non-`ConnectionError` exceptions surface to the caller."""
        tiles = [Tile(0, 0, 0)]
        provider = MagicMock()

        with patch.object(
            tiles_mod,
            "fetch_single_tile",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                fetch_tiles(tiles, provider, max_workers=1, timeout=1, retries=0)


class TestStitchTiles:
    """Tests for `cleopatra.basemap.tiles.stitch_tiles`."""

    def test_single_tile_returns_correct_shape(self):
        """One 256-px tile yields a `(256, 256, 4)` uint8 array."""
        tiles = [Tile(0, 0, 1)]
        png = _make_tile_png(size=256)
        image, extent = stitch_tiles({tiles[0]: png}, tiles, zoom=1)

        assert image.shape == (
            256,
            256,
            4,
        ), f"Expected (256, 256, 4), got {image.shape}"
        assert image.dtype.name == "uint8", f"Expected uint8, got {image.dtype}"
        assert len(extent) == 4, f"Expected 4-tuple extent, got {extent}"

    def test_two_tiles_horizontal_doubles_width(self):
        """Two horizontally-adjacent tiles produce a `(256, 512, 4)` image."""
        tiles = [Tile(0, 0, 1), Tile(1, 0, 1)]
        png = _make_tile_png(size=256)
        image, _ = stitch_tiles({tiles[0]: png, tiles[1]: png}, tiles, zoom=1)
        assert image.shape == (
            256,
            512,
            4,
        ), f"Expected (256, 512, 4) for two horizontal tiles, got {image.shape}"

    def test_invalid_first_image_raises(self):
        """A corrupt first PNG raises `ValueError` with a decode hint."""
        tiles = [Tile(0, 0, 1)]
        with pytest.raises(ValueError, match="Failed to decode tile image"):
            stitch_tiles({tiles[0]: b"not a png"}, tiles, zoom=1)

    def test_invalid_second_image_raises_with_tile_coords(self):
        """A corrupt non-first PNG identifies the offending tile in the message."""
        good_png = _make_tile_png()
        good_tile = Tile(0, 0, 1)
        bad_tile = Tile(1, 0, 1)
        tiles = [good_tile, bad_tile]
        with pytest.raises(ValueError, match="z=1/x=1/y=0"):
            stitch_tiles({good_tile: good_png, bad_tile: b"junk"}, tiles, zoom=1)

    def test_extent_is_4_floats_in_3857(self):
        """The returned extent is four floats in EPSG:3857 meters."""
        tiles = [Tile(0, 0, 0)]
        png = _make_tile_png()
        _, extent = stitch_tiles({tiles[0]: png}, tiles, zoom=0)

        west, south, east, north = extent
        assert west < east, f"west {west} < east {east} should hold"
        assert south < north, f"south {south} < north {north} should hold"
        for v in extent:
            assert isinstance(v, float), (
                f"Extent component should be float, got {type(v)}"
            )


class TestAddTilesAdditionalValidation:
    """Additional validation paths for `add_tiles`."""

    def test_zero_area_extent_raises(self):
        """An axes with `west == east` raises `ValueError` about zero area."""
        ax = MagicMock()
        ax.get_xlim.return_value = (1000000.0, 1000000.0)
        ax.get_ylim.return_value = (6000000.0, 6200000.0)
        with pytest.raises(ValueError, match="zero-area"):
            add_tiles(ax, crs=3857)

    def test_invalid_crs_string_raises_value_error(self):
        """A bogus CRS string surfaces as `ValueError` from add_tiles."""
        ax = MagicMock()
        ax.get_xlim.return_value = (10.0, 11.0)
        ax.get_ylim.return_value = (50.0, 51.0)
        with pytest.raises((ValueError, Exception)):
            add_tiles(ax, crs="EPSG:NOT-A-REAL-CRS")

    def test_zoom_integer_acceptable(self, mock_ax, _patch_tiles):
        """An explicit `zoom=5` is accepted and used downstream."""
        result = add_tiles(mock_ax, crs=3857, zoom=5)
        assert result is mock_ax

    def test_attribution_true_strips_html_tags(self, mock_ax, _patch_tiles):
        """`attribution=True` strips HTML tags before placing the text."""
        add_tiles(mock_ax, crs=3857, attribution=True)
        if mock_ax.text.called:
            placed_text = mock_ax.text.call_args[0][2]
            assert "<" not in placed_text, (
                f"Attribution text should be HTML-stripped, got: {placed_text!r}"
            )

    def test_attribution_unescapes_html_entities(self, mock_ax, _patch_tiles):
        """`attribution=True` strips tags *and* unescapes HTML entities."""
        from types import SimpleNamespace

        provider = SimpleNamespace(
            attribution="&copy; <a href='x'>OpenStreetMap</a> &amp; contributors"
        )
        add_tiles(mock_ax, source=provider, crs=3857, attribution=True)
        mock_ax.text.assert_called_once()
        placed = mock_ax.text.call_args[0][2]
        assert placed == "© OpenStreetMap & contributors", (
            f"expected entities unescaped, got {placed!r}"
        )


class TestStitchTilesPerformance:
    """Performance micro-test: stitching a handful of tiles is fast."""

    def test_stitch_completes_quickly_for_small_grid(self):
        """A 2x2 grid of 64-px tiles stitches in well under 100 ms."""
        import time

        tiles = [Tile(x, y, 1) for x in (0, 1) for y in (0, 1)]
        png = _make_tile_png(size=64)
        tile_data = {t: png for t in tiles}

        start = time.perf_counter()
        image, _ = stitch_tiles(tile_data, tiles, zoom=1)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"Stitch took {elapsed:.3f}s, expected < 1.0s"
        assert image.shape == (128, 128, 4)


class TestMaxTilesConstant:
    """Module-level `MAX_TILES` constant invariants."""

    def test_max_tiles_is_positive_int(self):
        """`MAX_TILES` is a positive integer."""
        assert isinstance(MAX_TILES, int), f"Expected int, got {type(MAX_TILES)}"
        assert MAX_TILES > 0, f"MAX_TILES should be positive, got {MAX_TILES}"


class TestAddTilesProviderObject:
    """Cover the branch where `source` is a provider object, not a string."""

    def test_provider_object_passed_directly(self, mock_ax, _patch_tiles):
        """Passing an `xyzservices.TileProvider` instance bypasses lookup."""
        import xyzservices.providers as xyz

        provider = xyz.OpenStreetMap.Mapnik
        result = add_tiles(mock_ax, source=provider, crs=3857)
        mock_ax.imshow.assert_called_once()
        assert result is mock_ax


class TestAddTilesCRSReprojectionFailure:
    """Cover the non-CRS reraise + EPSG:4326 NaN guard branches."""

    def test_non_crs_exception_reraises(self, monkeypatch):
        """A non-CRS exception from reprojection re-raises unchanged."""
        ax = MagicMock()
        ax.get_xlim.return_value = (10.0, 11.0)
        ax.get_ylim.return_value = (50.0, 51.0)

        def fake_reproject(*args, **kwargs):
            raise RuntimeError("Some other failure")

        monkeypatch.setattr(tiles_mod, "_densify_and_reproject_bounds", fake_reproject)
        with pytest.raises(RuntimeError, match="Some other failure"):
            add_tiles(ax, crs=4326)

    def test_4326_inf_after_back_transform_raises(self, monkeypatch):
        """A back-transform to EPSG:4326 returning Inf raises `ValueError`."""
        import pyproj

        ax = MagicMock()
        ax.get_xlim.return_value = (1000000.0, 1200000.0)
        ax.get_ylim.return_value = (6000000.0, 6200000.0)

        original = pyproj.Transformer.from_crs

        def fake_from_crs(src, dst, always_xy=True):
            if src == "EPSG:3857" and dst == "EPSG:4326":
                t = MagicMock()
                t.transform.return_value = (np.inf, np.inf)
                return t
            return original(src, dst, always_xy=always_xy)

        # `add_tiles` does `from pyproj import Transformer` internally, so
        # patch the class itself rather than a name on `cleopatra.basemap.tiles`.
        monkeypatch.setattr(pyproj.Transformer, "from_crs", staticmethod(fake_from_crs))
        with pytest.raises(ValueError, match="Web Mercator"):
            add_tiles(ax, crs=3857)


class TestAddTilesEmptyTiles:
    """Cover the branch where the bbox has no covering tiles."""

    def test_empty_tiles_raises_value_error(self, mock_ax):
        """An empty tile list at the resolved zoom raises `ValueError`."""
        with (
            patch.object(tiles_mod, "auto_zoom", return_value=10),
            patch.object(tiles_mod, "_tiles_for_bbox", return_value=[]),
        ):
            with pytest.raises(ValueError, match="No tiles found"):
                add_tiles(mock_ax, crs=3857)


class TestAddTilesNearGlobalExtent:
    """End-to-end regression test for a near-global Web Mercator extent.

    Drives `add_tiles` through its real reprojection and `_tiles_for_bbox`
    call (only `fetch_tiles`/`stitch_tiles` are mocked, unlike the other
    `add_tiles` tests, which mock `_tiles_for_bbox` itself). This is
    deliberate: reprojecting a near-global EPSG:3857 extent to EPSG:4326
    wraps longitude at the +/-180 seam (`pyproj`'s inverse Web Mercator
    transform), producing a `west > east` bbox -- exactly the
    antimeridian-crossing shape `_tiles_for_bbox` must split rather than
    crash or silently drop tiles on. A test that mocks `_tiles_for_bbox`
    itself (as the other `add_tiles` tests do) cannot exercise this path.
    """

    @pytest.fixture
    def near_global_ax(self):
        """A mock axes whose Web Mercator x-limits extend past the world bounds by 5%."""
        extended = 20037508.342789244 * 1.05
        ax = MagicMock()
        ax.get_xlim.return_value = (-extended, extended)
        ax.get_ylim.return_value = (-8_000_000.0, 8_000_000.0)
        ax.get_aspect.return_value = "auto"

        mock_transform = MagicMock()
        mock_transform.inverted.return_value = mock_transform
        mock_fig = MagicMock()
        mock_fig.dpi = 100.0
        type(mock_fig).dpi_scale_trans = PropertyMock(return_value=mock_transform)

        mock_bbox = MagicMock()
        mock_bbox.width = 6.0
        mock_bbox.height = 4.0
        mock_bbox.transformed.return_value = mock_bbox

        ax.get_figure.return_value = mock_fig
        ax.get_window_extent.return_value = mock_bbox
        return ax

    def test_near_global_extent_does_not_raise(self, near_global_ax):
        """`add_tiles` succeeds on a near-global extent instead of raising.

        Test scenario:
            A 5%-margin whole-world Web Mercator extent (matplotlib's own
            default autoscale margin on full-world data would produce
            exactly this) reprojects to a `west > east` EPSG:4326 bbox.
            Before this fix, that raised `ValueError` from `_tiles_for_bbox`
            (or, before the round-1 fix, silently produced no tiles); now it
            must resolve successfully via the antimeridian split.
        """
        fake_image = np.zeros((256, 256, 4), dtype=np.uint8)
        with (
            patch.object(
                tiles_mod,
                "fetch_tiles",
                return_value={Tile(0, 0, 3): _make_tile_png()},
            ),
            patch.object(
                tiles_mod,
                "stitch_tiles",
                return_value=(fake_image, (-20037508.34, -8000000.0, 20037508.34, 8000000.0)),
            ) as mock_stitch,
        ):
            add_tiles(near_global_ax, crs=3857, zoom=3)

        tiles_passed = mock_stitch.call_args[0][1]
        xs = {t.x for t in tiles_passed}
        assert len(tiles_passed) > 0, "the antimeridian split should still produce tiles"
        assert 0 in xs, "should include tiles from the western half of the split (x near 0)"
        assert max(xs) >= 2**3 - 1, (
            "should include tiles from the eastern half of the split (x near the grid edge)"
        )


def _world_bounds() -> tuple[float, float, float, float]:
    """The full-world EPSG:3857 extent `(west, south, east, north)` in metres."""
    m = tiles_mod._MERC_MAX
    return (-m, -m, m, m)


def _fetch_split_by_x():
    """A `fetch_tiles` stand-in: west tiles solid blue, east tiles solid red."""

    def fake(tile_list, provider, **kwargs):
        out = {}
        for tile in tile_list:
            half = 2**tile.z // 2
            colour = (0, 0, 255, 255) if tile.x < half else (255, 0, 0, 255)
            buf = io.BytesIO()
            Image.new("RGBA", (256, 256), colour).save(buf, "PNG")
            out[tile] = buf.getvalue()
        return out

    return fake


class TestMercatorToEquirectangular:
    """Tests for `cleopatra.basemap.tiles.mercator_to_equirectangular`."""

    def test_returns_equirectangular_shape_and_float32(self):
        """A 3-band mosaic resamples to an `(n_lat, n_lon, 3)` float32 grid."""
        tex = mercator_to_equirectangular(
            np.zeros((16, 16, 3), "uint8"), _world_bounds(), n_lon=8, n_lat=6
        )
        assert tex.shape == (6, 8, 3), f"unexpected shape {tex.shape}"
        assert tex.dtype == np.float32, f"expected float32, got {tex.dtype}"

    def test_two_dimensional_mosaic_stays_2d(self):
        """A single-band (2-D) mosaic returns a 2-D `(n_lat, n_lon)` grid."""
        tex = mercator_to_equirectangular(
            np.zeros((16, 16), "uint8"), _world_bounds(), n_lon=8, n_lat=6
        )
        assert tex.shape == (6, 8), f"unexpected shape {tex.shape}"

    def test_preserves_west_east_longitude_order(self):
        """West-blue / east-red survives the resample (x is linear in longitude)."""
        m = np.zeros((8, 8, 3), "uint8")
        m[:, :4] = (0, 0, 255)
        m[:, 4:] = (255, 0, 0)
        tex = mercator_to_equirectangular(m, _world_bounds(), n_lon=8, n_lat=6)
        assert (tex[:, 0, 2] > tex[:, 0, 0]).all(), "west column should stay blue"
        assert (tex[:, -1, 0] > tex[:, -1, 2]).all(), "east column should stay red"

    def test_preserves_north_south_latitude_order(self):
        """North-red / south-blue keeps red at the top (north-up, origin upper)."""
        m = np.zeros((8, 8, 3), "uint8")
        m[:4] = (255, 0, 0)
        m[4:] = (0, 0, 255)
        tex = mercator_to_equirectangular(m, _world_bounds(), n_lon=8, n_lat=8)
        assert tex[0, 0, 0] > tex[0, 0, 2], "north row should stay red"
        assert tex[-1, 0, 2] > tex[-1, 0, 0], "south row should stay blue"

    def test_value_scale_preserved_and_averaged(self):
        """A constant uint8 mosaic yields that same value as float32 (0..255)."""
        tex = mercator_to_equirectangular(
            np.full((8, 8, 1), 200, "uint8"), _world_bounds(), n_lon=4, n_lat=4
        )
        assert tex.dtype == np.float32, f"expected float32, got {tex.dtype}"
        assert np.allclose(tex, 200.0), "a constant mosaic stays at its value"

    def test_no_south_seam_at_realistic_height(self):
        """A tall constant mosaic stays flat at the default `n_lat`, guarding the
        south-clamp divisor bug that darkened the output row near -85 deg S."""
        m = np.full((8192, 8, 3), 255, "uint8")
        tex = mercator_to_equirectangular(m, _world_bounds(), n_lon=8, n_lat=1440)
        assert np.allclose(tex, 255.0), f"seam near -85S: min {float(tex.min())}"

    @pytest.mark.parametrize("bad", [np.zeros((2,)), np.zeros((2, 2, 2, 2))])
    def test_rejects_non_2d_3d_mosaic(self, bad):
        """A 1-D or 4-D mosaic raises `ValueError`."""
        with pytest.raises(ValueError, match="2-D or 3-D"):
            mercator_to_equirectangular(bad, _world_bounds())

    @pytest.mark.parametrize("n_lon,n_lat", [(0, 4), (4, 0), (-1, 4)])
    def test_rejects_non_positive_grid(self, n_lon, n_lat):
        """A non-positive `n_lon`/`n_lat` raises `ValueError`."""
        with pytest.raises(ValueError, match="must be positive"):
            mercator_to_equirectangular(
                np.zeros((4, 4, 3)), _world_bounds(), n_lon=n_lon, n_lat=n_lat
            )

    @pytest.mark.parametrize("bounds", [(1.0, 1.0, 0.0, 2.0), (0.0, 2.0, 2.0, 1.0)])
    def test_rejects_degenerate_bounds(self, bounds):
        """Bounds with `east <= west` or `north <= south` raise `ValueError`."""
        with pytest.raises(ValueError, match="east > west"):
            mercator_to_equirectangular(np.zeros((4, 4, 3)), bounds)


class TestWorldTexture:
    """Tests for `cleopatra.basemap.tiles.world_texture`."""

    def test_returns_float32_texture_in_unit_range(self, tmp_path, monkeypatch):
        """A whole-world fetch resamples to an `(n_lat, n_lon, 3)` float32 in [0, 1]."""
        monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(tiles_mod, "fetch_tiles", _fetch_split_by_x())
        tex = world_texture(zoom=1, n_lon=16, n_lat=8)
        assert tex.shape == (8, 16, 3), f"unexpected shape {tex.shape}"
        assert tex.dtype == np.float32, f"expected float32, got {tex.dtype}"
        assert 0.0 <= float(tex.min()) and float(tex.max()) <= 1.0, "not in [0, 1]"

    def test_preserves_longitude_order(self, tmp_path, monkeypatch):
        """West tiles (blue) stay west end-to-end through fetch/stitch/reproject."""
        monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(tiles_mod, "fetch_tiles", _fetch_split_by_x())
        tex = world_texture(zoom=1, n_lon=16, n_lat=8)
        assert tex[4, 0, 2] > tex[4, 0, 0], "west should be blue"
        assert tex[4, -1, 0] > tex[4, -1, 2], "east should be red"

    def test_accepts_a_tileprovider_object(self, tmp_path, monkeypatch):
        """A resolved `TileProvider` object is accepted, not only a name string."""
        monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(tiles_mod, "fetch_tiles", _fetch_split_by_x())
        provider = get_provider("OpenStreetMap.Mapnik")
        tex = world_texture(provider, zoom=1, n_lon=8, n_lat=4)
        assert tex.shape == (4, 8, 3), f"unexpected shape {tex.shape}"

    def test_caches_texture_and_skips_refetch(self, tmp_path, monkeypatch):
        """A second call reads the on-disk cache instead of refetching tiles."""
        monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path))
        fetch = MagicMock(side_effect=_fetch_split_by_x())
        monkeypatch.setattr(tiles_mod, "fetch_tiles", fetch)
        first = world_texture(zoom=1, n_lon=8, n_lat=4)
        second = world_texture(zoom=1, n_lon=8, n_lat=4)
        assert fetch.call_count == 1, "the second call should hit the cache"
        assert np.array_equal(first, second), "cached texture should match"

    def test_cache_false_always_refetches(self, tmp_path, monkeypatch):
        """`cache=False` refetches on every call."""
        monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path))
        fetch = MagicMock(side_effect=_fetch_split_by_x())
        monkeypatch.setattr(tiles_mod, "fetch_tiles", fetch)
        world_texture(zoom=1, n_lon=8, n_lat=4, cache=False)
        world_texture(zoom=1, n_lon=8, n_lat=4, cache=False)
        assert fetch.call_count == 2, "cache=False must not reuse a cached texture"

    def test_missing_extra_raises_import_error(self, monkeypatch):
        """Without the `[tiles]` extra, `world_texture` raises `ImportError`."""
        monkeypatch.setattr(tiles_mod, "_TILES_AVAILABLE", False)
        with pytest.raises(ImportError, match=r"cleopatra\[tiles\]"):
            world_texture(zoom=0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"zoom": -1},
            {"zoom": 7},
            {"zoom": 1, "n_lon": 0},
            {"zoom": 1, "n_lat": 0},
        ],
    )
    def test_rejects_invalid_params(self, kwargs):
        """A zoom outside 0..6 or a non-positive grid dimension raises `ValueError`."""
        with pytest.raises(ValueError):
            world_texture(**kwargs)
