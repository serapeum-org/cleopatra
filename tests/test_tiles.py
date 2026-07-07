"""Tests for cleopatra.tiles.

Covers `add_tiles` and helper functions in the ported web-tile
basemap module. HTTP fetching is mocked at the `urllib.request`
layer so the suite never hits the public internet -- the same strategy
that pyramids uses for its basemap tests.
"""

from __future__ import annotations

import io
from collections import namedtuple
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.plot

pytest.importorskip("PIL", reason="Pillow not installed (tiles extra)")
pytest.importorskip("mercantile", reason="mercantile not installed (tiles extra)")
pytest.importorskip("xyzservices", reason="xyzservices not installed (tiles extra)")
pytest.importorskip("pyproj", reason="pyproj not installed (tiles extra)")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from cleopatra import tiles as tiles_mod  # noqa: E402
from cleopatra.tiles import (  # noqa: E402
    MAX_TILES,
    _densify_and_reproject_bounds,
    _looks_like_image,
    _require_tiles_extra,
    add_tiles,
    auto_zoom,
    fetch_single_tile,
    fetch_tiles,
    get_provider,
    stitch_tiles,
)

Tile = namedtuple("Tile", ["x", "y", "z"])


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
    """Tests for `cleopatra.tiles.get_provider`."""

    def test_default_provider_is_openstreetmap(self):
        """Calling `get_provider(None)` returns OpenStreetMap.Mapnik."""
        provider = get_provider(None)
        assert (
            "openstreetmap" in provider.name.lower()
            or "OpenStreetMap" in str(provider)
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
    """Tests for `cleopatra.tiles.auto_zoom`."""

    def test_global_extent_is_zoom_2(self):
        """A global extent spans four tiles across at the default floor."""
        assert auto_zoom((-180.0, -85.0, 180.0, 85.0)) == 2

    def test_city_extent_yields_zoom_12(self):
        """A ~0.6-degree city extent yields zoom 12 at the default floor."""
        assert auto_zoom((13.0, 52.4, 13.6, 52.6)) == 12

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
        assert auto_zoom(gulf) == 8  # default floor -> sharper basemap
        assert auto_zoom(gulf, min_tiles_across=1) == 6  # old coarse value

    def test_non_positive_min_tiles_across_is_treated_as_one(self):
        """`min_tiles_across` below 1 is clamped to the one-tile heuristic."""
        assert auto_zoom((-180.0, -85.0, 180.0, 85.0), min_tiles_across=0) == 0


class TestDensifyAndReprojectBounds:
    """Tests for `cleopatra.tiles._densify_and_reproject_bounds`."""

    def test_4326_to_3857_produces_meters(self):
        """4326 -> 3857 produces bounds with absolute values in meters."""
        west, south, east, north = _densify_and_reproject_bounds(
            10.0, 50.0, 11.0, 51.0, "EPSG:4326", "EPSG:3857"
        )
        assert abs(west) > 100000, (
            f"West should be in meters (large value), got {west}"
        )
        assert west < east
        assert south < north

    def test_identity_transform_preserves_bounds(self):
        """4326 -> 4326 returns approximately the same bounds."""
        bounds = (10.0, 50.0, 11.0, 51.0)
        result = _densify_and_reproject_bounds(
            *bounds, "EPSG:4326", "EPSG:4326"
        )
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
        # The mocked stitch_tiles reports the mosaic covering these 3857 metres:
        w, s, e, n = _densify_and_reproject_bounds(
            1000000.0, 6000000.0, 1200000.0, 6200000.0, "EPSG:3857", "EPSG:4326"
        )
        got = mock_ax.imshow.call_args.kwargs["extent"]
        assert got == pytest.approx([w, e, s, n]), (
            f"imshow extent should be the mosaic's reprojected bounds, got {got}"
        )
        assert got != [10.0, 11.0, 50.0, 51.0], "extent must not be the raw data bounds"

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
        from cleopatra.tiles import USER_AGENT

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
        import mercantile as merc_mod

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
                merc_mod,
                "tiles",
                side_effect=[many_tiles, few_tiles],
            ) as mock_tiles,
        ):
            add_tiles(mock_ax, crs=3857)

        calls = mock_tiles.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["zooms"] == 10
        assert calls[1][1]["zooms"] == 9

    def test_custom_max_tiles_relaxes_reduction(self, mock_ax):
        """A higher `max_tiles=` avoids the zoom reduction (N2)."""
        import mercantile as merc_mod

        fake_image = np.zeros((256, 256, 4), dtype=np.uint8)
        # 400 tiles at the requested zoom — over the default 256, under 500.
        many_tiles = [Tile(x=i, y=j, z=10) for i in range(20) for j in range(20)]

        with (
            patch.object(tiles_mod, "auto_zoom", return_value=10),
            patch.object(
                tiles_mod, "fetch_tiles", return_value={Tile(0, 0, 10): _make_tile_png()}
            ),
            patch.object(
                tiles_mod,
                "stitch_tiles",
                return_value=(fake_image, (1e6, 6e6, 1.2e6, 6.2e6)),
            ),
            patch.object(merc_mod, "tiles", side_effect=[many_tiles]) as mock_tiles,
        ):
            add_tiles(mock_ax, crs=3857, max_tiles=500)

        # Only one call: 400 <= max_tiles=500, so no reduction.
        assert len(mock_tiles.call_args_list) == 1
        assert mock_tiles.call_args_list[0][1]["zooms"] == 10

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True, "8"])
    def test_invalid_max_tiles_raises(self, mock_ax, bad):
        """`max_tiles` must be a positive int (N2).

        Args:
            bad: A value that should be rejected.
        """
        with pytest.raises(ValueError, match="max_tiles must be a positive int"):
            add_tiles(mock_ax, crs=3857, max_tiles=bad)


class TestRequireTilesExtra:
    """Tests for `cleopatra.tiles._require_tiles_extra` guard."""

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
    """Boundary tests for `cleopatra.tiles.auto_zoom`."""

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
                    10.0, 50.0, 11.0, 51.0,
                    "EPSG:4326", "EPSG:3857",
                    n_points=2,
                )

    def test_n_points_low_value_runs(self):
        """`n_points=2` (only corners) still produces finite bounds."""
        west, south, east, north = _densify_and_reproject_bounds(
            10.0, 50.0, 11.0, 51.0,
            "EPSG:4326", "EPSG:3857",
            n_points=2,
        )
        assert all(np.isfinite([west, south, east, north])), (
            f"Bounds should all be finite, got ({west}, {south}, {east}, {north})"
        )


class TestFetchSingleTile:
    """Tests for `cleopatra.tiles.fetch_single_tile`."""

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

        with patch("urllib.request.urlopen") as mock_urlopen:
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

        with patch("urllib.request.urlopen") as mock_urlopen:
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

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = [
                urllib.error.URLError("transient"),
                successful_response,
            ]
            _, returned_bytes = fetch_single_tile(
                tile, provider, timeout=1, retries=2
            )
        assert returned_bytes == png, (
            f"Expected png bytes after retry, got {len(returned_bytes)} bytes"
        )

    def test_raises_after_all_retries_exhausted(self):
        """All retries failing raises `ConnectionError` referencing the tile."""
        import urllib.error

        tile = Tile(5, 6, 7)
        provider = self._make_provider()

        with patch("urllib.request.urlopen") as mock_urlopen:
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
            "jpeg-app0", "jpeg-app1", "jpeg-app2", "jpeg-app8", "jpeg-app15",
            "jpeg-dqt", "jpeg-sof0", "gif", "webp",
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
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = body
            mock_urlopen.return_value = mock_response
            _, returned_bytes = fetch_single_tile(
                tile, provider, timeout=1, retries=0
            )
        assert returned_bytes == body, "image bytes should pass through unchanged"

    def _captured_user_agent(self, mock_urlopen) -> str:
        """Extract the User-Agent header from the Request passed to urlopen."""
        request = mock_urlopen.call_args[0][0]
        # exactly one header is set on the request
        return next(iter(request.headers.values()))

    def test_default_user_agent_is_versioned(self):
        """The default User-Agent identifies cleopatra with a version and URL."""
        from cleopatra.tiles import USER_AGENT

        png = _make_tile_png(size=32)
        provider = self._make_provider()
        with patch("urllib.request.urlopen") as mock_urlopen:
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
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = png
            mock_urlopen.return_value = mock_response
            fetch_single_tile(
                Tile(0, 0, 0), provider, timeout=1, retries=0, user_agent=custom
            )
        assert self._captured_user_agent(mock_urlopen) == custom


class TestLooksLikeImage:
    """Unit tests for `cleopatra.tiles._looks_like_image`."""

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
        ids=["png", "jpeg-app0", "jpeg-app2", "jpeg-dqt", "jpeg-sof0",
             "gif87a", "gif89a", "webp"],
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
    """Tests for `cleopatra.tiles.fetch_tiles`."""

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
    """Tests for `cleopatra.tiles.stitch_tiles`."""

    def test_single_tile_returns_correct_shape(self):
        """One 256-px tile yields a `(256, 256, 4)` uint8 array."""
        tiles = [Tile(0, 0, 1)]
        png = _make_tile_png(size=256)
        image, extent = stitch_tiles({tiles[0]: png}, tiles, zoom=1)

        assert image.shape == (256, 256, 4), (
            f"Expected (256, 256, 4), got {image.shape}"
        )
        assert image.dtype.name == "uint8", f"Expected uint8, got {image.dtype}"
        assert len(extent) == 4, f"Expected 4-tuple extent, got {extent}"

    def test_two_tiles_horizontal_doubles_width(self):
        """Two horizontally-adjacent tiles produce a `(256, 512, 4)` image."""
        tiles = [Tile(0, 0, 1), Tile(1, 0, 1)]
        png = _make_tile_png(size=256)
        image, _ = stitch_tiles(
            {tiles[0]: png, tiles[1]: png}, tiles, zoom=1
        )
        assert image.shape == (256, 512, 4), (
            f"Expected (256, 512, 4) for two horizontal tiles, got {image.shape}"
        )

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
            assert isinstance(v, float), f"Extent component should be float, got {type(v)}"


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

        monkeypatch.setattr(
            tiles_mod, "_densify_and_reproject_bounds", fake_reproject
        )
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
        # patch the class itself rather than a name on `cleopatra.tiles`.
        monkeypatch.setattr(
            pyproj.Transformer, "from_crs", staticmethod(fake_from_crs)
        )
        with pytest.raises(ValueError, match="Web Mercator"):
            add_tiles(ax, crs=3857)


class TestAddTilesEmptyTiles:
    """Cover the branch where mercantile returns no tiles."""

    def test_empty_tiles_raises_value_error(self, mock_ax):
        """An empty tile list at the resolved zoom raises `ValueError`."""
        import mercantile as merc_mod

        with (
            patch.object(tiles_mod, "auto_zoom", return_value=10),
            patch.object(merc_mod, "tiles", return_value=[]),
        ):
            with pytest.raises(ValueError, match="No tiles found"):
                add_tiles(mock_ax, crs=3857)
