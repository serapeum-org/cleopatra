"""Tests for cleopatra.tiles.

Covers :func:`add_tiles` and helper functions in the ported web-tile
basemap module. HTTP fetching is mocked at the :mod:`urllib.request`
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
    _densify_and_reproject_bounds,
    add_tiles,
    auto_zoom,
    get_provider,
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
    """Tests for :func:`cleopatra.tiles.get_provider`."""

    def test_default_provider_is_openstreetmap(self):
        """Calling ``get_provider(None)`` returns OpenStreetMap.Mapnik."""
        provider = get_provider(None)
        assert (
            "openstreetmap" in provider.name.lower()
            or "OpenStreetMap" in str(provider)
        ), f"Default provider should be OpenStreetMap, got {provider}"

    def test_resolve_cartodb_positron(self):
        """A dotted string resolves to a provider with ``build_url``."""
        provider = get_provider("CartoDB.Positron")
        assert hasattr(provider, "build_url"), (
            f"Provider should have build_url method: {provider}"
        )

    def test_invalid_provider_raises_value_error(self):
        """An unknown provider name raises :class:`ValueError`."""
        with pytest.raises(ValueError, match="Unknown tile provider"):
            get_provider("NonExistent.FakeProvider")

    def test_partial_invalid_name_raises_value_error(self):
        """A partially-valid name reports which segment failed."""
        with pytest.raises(ValueError, match="Failed at"):
            get_provider("OpenStreetMap.NonExistent")


class TestAutoZoom:
    """Tests for :func:`cleopatra.tiles.auto_zoom`."""

    def test_global_extent_is_zoom_0(self):
        """A global extent collapses to zoom 0."""
        assert auto_zoom((-180.0, -85.0, 180.0, 85.0)) == 0

    def test_city_extent_yields_zoom_10(self):
        """A ~0.6-degree city extent yields zoom 10."""
        assert auto_zoom((13.0, 52.4, 13.6, 52.6)) == 10

    def test_tiny_extent_clamps_to_19(self):
        """An infinitesimal extent clamps to the max zoom 19."""
        assert auto_zoom((0.0, 0.0, 1e-8, 1e-8)) == 19


class TestDensifyAndReprojectBounds:
    """Tests for :func:`cleopatra.tiles._densify_and_reproject_bounds`."""

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
    """Validation / error-path tests for :func:`add_tiles`."""

    def test_raises_on_empty_axes(self):
        """Axes with default 0-1 limits raise :class:`ValueError`."""
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
        """Non-axes objects raise :class:`TypeError`."""
        with pytest.raises(TypeError, match="matplotlib.axes.Axes"):
            add_tiles(bad_ax)

    @pytest.mark.parametrize(
        "bad_zoom",
        [-1, 20, 100, "invalid"],
        ids=["negative", "too_high", "way_too_high", "string"],
    )
    def test_raises_on_invalid_zoom(self, bad_zoom):
        """Invalid zoom values raise :class:`ValueError`."""
        ax = MagicMock()
        ax.get_xlim.return_value = (1000000.0, 1200000.0)
        ax.get_ylim.return_value = (6000000.0, 6200000.0)
        ax.get_aspect.return_value = "auto"
        with pytest.raises(ValueError, match="zoom"):
            add_tiles(ax, crs=3857, zoom=bad_zoom)

    def test_invalid_source_string_raises_value_error(self):
        """A bogus source string raises :class:`ValueError`."""
        ax = MagicMock()
        ax.get_xlim.return_value = (1000000.0, 1200000.0)
        ax.get_ylim.return_value = (6000000.0, 6200000.0)
        with pytest.raises(ValueError, match="Unknown tile provider"):
            add_tiles(ax, crs=3857, source="Bogus.NotARealProvider")

    def test_missing_extras_raises_import_error(self, monkeypatch):
        """If the tiles extra is unavailable, :class:`ImportError` is raised."""
        monkeypatch.setattr(tiles_mod, "_TILES_AVAILABLE", False)
        ax = MagicMock()
        ax.get_xlim.return_value = (1000000.0, 1200000.0)
        ax.get_ylim.return_value = (6000000.0, 6200000.0)
        with pytest.raises(ImportError, match=r"cleopatra\[tiles\]"):
            add_tiles(ax)


@pytest.fixture
def mock_ax():
    """Return a :class:`MagicMock` axes with a realistic Web Mercator extent."""
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
    """Patch :func:`auto_zoom`, :func:`fetch_tiles`, :func:`stitch_tiles`."""
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
    """Behavioural tests for :func:`add_tiles` (mocked HTTP layer)."""

    def test_default_source_renders_image(self, mock_ax, _patch_tiles):
        """``source=None`` -> default provider; imshow is called once."""
        result = add_tiles(mock_ax)
        mock_ax.imshow.assert_called_once()
        assert result is mock_ax

    def test_default_crs_is_3857(self, mock_ax, _patch_tiles):
        """``crs=None`` treats the data as Web Mercator (no error)."""
        result = add_tiles(mock_ax, crs=None)
        mock_ax.imshow.assert_called_once()
        assert result is mock_ax

    def test_string_source_renders(self, mock_ax, _patch_tiles):
        """``source="CartoDB.Positron"`` resolves and renders."""
        result = add_tiles(mock_ax, source="CartoDB.Positron")
        mock_ax.imshow.assert_called_once()
        assert result is mock_ax

    def test_explicit_crs_4326_renders(self, mock_ax, _patch_tiles):
        """``crs=4326`` works end-to-end (no GDAL warping needed)."""
        mock_ax.get_xlim.return_value = (10.0, 11.0)
        mock_ax.get_ylim.return_value = (50.0, 51.0)
        add_tiles(mock_ax, crs=4326)
        mock_ax.imshow.assert_called_once()

    def test_axes_limits_are_restored(self, mock_ax, _patch_tiles):
        """``set_xlim`` / ``set_ylim`` are called with the original limits."""
        add_tiles(mock_ax, crs=3857)
        mock_ax.set_xlim.assert_called_once_with((1000000.0, 1200000.0))
        mock_ax.set_ylim.assert_called_once_with((6000000.0, 6200000.0))

    def test_attribution_false_skips_text(self, mock_ax, _patch_tiles):
        """``attribution=False`` -> ``ax.text`` is never called."""
        add_tiles(mock_ax, crs=3857, attribution=False)
        mock_ax.text.assert_not_called()

    def test_custom_attribution_string(self, mock_ax, _patch_tiles):
        """``attribution="Custom"`` -> exact string is written to axes."""
        add_tiles(mock_ax, crs=3857, attribution="Custom Attribution")
        mock_ax.text.assert_called_once()
        call_args = mock_ax.text.call_args
        assert call_args[0][2] == "Custom Attribution"

    def test_imshow_receives_alpha_and_zorder(self, mock_ax, _patch_tiles):
        """Custom ``alpha`` and ``zorder`` are forwarded to ``imshow``."""
        add_tiles(mock_ax, crs=3857, alpha=0.5, zorder=-2)
        call_kwargs = mock_ax.imshow.call_args[1]
        assert call_kwargs["alpha"] == 0.5
        assert call_kwargs["zorder"] == -2


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
