"""Tests for cleopatra.reference.

Covers `add_relief` / `add_features` and their helpers. No test hits the
network: assets are pre-written into a temporary cache directory
(`CLEOPATRA_CACHE_DIR`) so `_download` short-circuits to the cached file
and the real read/parse/draw path is exercised offline.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.plot

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

from cleopatra import reference  # noqa: E402
from cleopatra.reference import (  # noqa: E402
    _is_4326,
    _paths,
    add_features,
    add_relief,
    available_layers,
    available_relief_resolutions,
    available_resolutions,
    natural_earth,
    relief,
)


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the reference cache at a temp dir for the duration of a test."""
    monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path))
    return tmp_path


def _write_layer(cache: Path, stem: str, resolution: str, geometry: dict) -> None:
    """Write a one-feature gzipped GeoJSON asset into the cache."""
    collection = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": geometry}],
    }
    path = cache / f"ne_{resolution}_{stem}.geojson.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(collection, fh)


# --- discovery helpers ------------------------------------------------------


def test_available_layers():
    assert available_layers() == [
        "coastline",
        "land",
        "ocean",
        "rivers",
        "lakes",
        "borders",
    ]


def test_available_resolutions():
    assert available_resolutions() == ["110m", "50m", "10m"]


def test_available_relief_resolutions():
    assert available_relief_resolutions() == ["low", "medium"]


# --- geometry flattening ----------------------------------------------------


def test_paths_polygon_drops_holes():
    geom = {
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [2, 0], [2, 2], [0, 0]],
            [[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 0.5]],
        ],
    }
    parts = _paths(geom)
    assert len(parts) == 1
    assert parts[0].shape == (4, 2)


def test_paths_multipolygon_one_per_part():
    geom = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[0, 0], [1, 0], [1, 1], [0, 0]]],
            [[[5, 5], [6, 5], [6, 6], [5, 5]]],
        ],
    }
    assert [p.shape for p in _paths(geom)] == [(4, 2), (4, 2)]


def test_paths_multilinestring():
    geom = {
        "type": "MultiLineString",
        "coordinates": [[[0, 0], [1, 1]], [[2, 2], [3, 3], [4, 4]]],
    }
    assert [p.shape for p in _paths(geom)] == [(2, 2), (3, 2)]


def test_paths_unsupported_type_raises():
    with pytest.raises(ValueError, match="Unsupported geometry type"):
        _paths({"type": "Point", "coordinates": [0, 0]})


def test_is_4326():
    assert _is_4326(None)
    assert _is_4326(4326)
    assert _is_4326("EPSG:4326")
    assert _is_4326("epsg:4326")
    assert not _is_4326(3857)
    assert not _is_4326("EPSG:3857")


# --- natural_earth ----------------------------------------------------------


def test_natural_earth_reads_cached_lines(cache: Path):
    _write_layer(
        cache,
        "coastline",
        "110m",
        {"type": "MultiLineString", "coordinates": [[[0, 0], [10, 10]], [[5, 5], [6, 7]]]},
    )
    parts = natural_earth("coastline", "110m")
    assert len(parts) == 2
    assert all(p.ndim == 2 and p.shape[1] == 2 for p in parts)


def test_natural_earth_skips_null_geometry(cache: Path):
    collection = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": None},
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
            },
        ],
    }
    with gzip.open(cache / "ne_110m_coastline.geojson.gz", "wt", encoding="utf-8") as fh:
        json.dump(collection, fh)
    parts = natural_earth("coastline", "110m")
    assert len(parts) == 1


def test_natural_earth_unknown_layer():
    with pytest.raises(ValueError, match="Unknown layer"):
        natural_earth("rivers_and_roads", "110m")


def test_natural_earth_unknown_resolution():
    with pytest.raises(ValueError, match="Unknown resolution"):
        natural_earth("coastline", "1m")


# --- add_features -----------------------------------------------------------


def test_add_features_line_layer(cache: Path):
    _write_layer(
        cache,
        "coastline",
        "110m",
        {"type": "MultiLineString", "coordinates": [[[0, 0], [10, 10], [20, 5]]]},
    )
    fig, ax = plt.subplots()
    ax.plot([0, 20], [0, 10])
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    out = add_features(ax, "coastline", "110m", colors="navy")
    assert out is ax
    lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(lcs) == 1
    assert ax.get_xlim() == xlim and ax.get_ylim() == ylim
    plt.close(fig)


def test_add_features_polygon_layer(cache: Path):
    _write_layer(
        cache,
        "land",
        "110m",
        {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]},
    )
    fig, ax = plt.subplots()
    ax.plot([0, 10], [0, 10])
    add_features(ax, "land", "110m", facecolors="0.7")
    assert any(isinstance(c, PolyCollection) for c in ax.collections)
    plt.close(fig)


def test_add_features_bad_axes():
    with pytest.raises(TypeError, match="matplotlib.axes.Axes"):
        add_features(object(), "coastline", "110m")


def test_add_features_unknown_layer():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Unknown layer"):
        add_features(ax, "nope", "110m")
    plt.close(fig)


def test_add_features_reproject(cache: Path):
    pytest.importorskip("pyproj", reason="pyproj not installed (tiles extra)")
    _write_layer(
        cache,
        "coastline",
        "110m",
        {"type": "LineString", "coordinates": [[0, 0], [10, 0]]},
    )
    fig, ax = plt.subplots()
    ax.plot([0, 2e6], [0, 1e6])
    add_features(ax, "coastline", "110m", crs=3857)
    verts = ax.collections[0].get_paths()[0].vertices
    # lon=0 maps to x=0 in EPSG:3857; lon=10 maps to ~1.11e6 m.
    assert abs(verts[0][0]) < 1.0
    assert verts[1][0] > 1.0e6
    plt.close(fig)


# --- relief -----------------------------------------------------------------


def test_relief_unknown_resolution():
    with pytest.raises(ValueError, match="Unknown relief resolution"):
        relief("ultra")


def test_add_relief(cache: Path):
    Image = pytest.importorskip(
        "PIL.Image", reason="Pillow not installed (tiles extra)"
    )
    arr = (np.random.default_rng(0).random((10, 20, 3)) * 255).astype("uint8")
    Image.fromarray(arr).save(cache / "ne_hypso_rgb_720x360.png")

    fig, ax = plt.subplots()
    ax.set_xlim(-20, 40)
    ax.set_ylim(0, 60)
    out = add_relief(ax, "low")
    assert out is ax
    assert len(ax.images) == 1
    assert ax.get_xlim() == (-20.0, 40.0)
    img = ax.images[0]
    assert tuple(img.get_extent()) == (-180.0, 180.0, -90.0, 90.0)
    plt.close(fig)


def test_add_relief_custom_extent(cache: Path):
    Image = pytest.importorskip(
        "PIL.Image", reason="Pillow not installed (tiles extra)"
    )
    arr = (np.random.default_rng(1).random((4, 8, 3)) * 255).astype("uint8")
    Image.fromarray(arr).save(cache / "ne_hypso_rgb_720x360.png")

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    add_relief(ax, "low", extent=(0, 0, 10, 10))
    assert tuple(ax.images[0].get_extent()) == (0.0, 10.0, 0.0, 10.0)
    plt.close(fig)


def test_add_relief_bad_axes():
    with pytest.raises(TypeError, match="matplotlib.axes.Axes"):
        add_relief(object())


# --- download guard ---------------------------------------------------------


def test_download_rejects_non_http(tmp_path: Path):
    with pytest.raises(ValueError, match="non-http"):
        reference._download("ftp://example.com/x.gz", tmp_path / "x.gz")


def test_download_returns_cached(tmp_path: Path):
    dest = tmp_path / "cached.bin"
    dest.write_bytes(b"already here")
    # No network: a cached file is returned untouched even for a bogus URL.
    assert reference._download("https://nope.invalid/cached.bin", dest) == dest
    assert dest.read_bytes() == b"already here"
