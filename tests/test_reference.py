"""Tests for cleopatra.reference.

Covers `add_relief` / `add_features` and their helpers. No test hits the
network: assets are pre-written into a temporary cache directory
(`CLEOPATRA_CACHE_DIR`) so `_download` short-circuits to the cached file
and the real read/parse/draw path is exercised offline.
"""

from __future__ import annotations

import gzip
import io
import json
import urllib.error
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.plot

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PathCollection  # noqa: E402

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
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    add_features(ax, "land", "110m", facecolors="0.7")
    assert any(isinstance(c, PathCollection) for c in ax.collections)
    assert ax.get_xlim() == xlim and ax.get_ylim() == ylim
    plt.close(fig)


def test_add_features_polygon_hole_renders_as_cutout(cache: Path):
    """A polygon with a hole must leave the hole unfilled (H1 regression)."""
    _write_layer(
        cache,
        "ocean",
        "110m",
        {
            "type": "Polygon",
            "coordinates": [
                [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],  # exterior
                [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]],  # hole
            ],
        },
    )
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")
    add_features(ax, "ocean", "110m", facecolors="black", edgecolors="none")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    height, width, _ = buf.shape

    def pixel(x_frac: float, y_frac: float) -> np.ndarray:
        # y is flipped: buffer row 0 is the top of the figure.
        return buf[int((1 - y_frac) * height) - 1, int(x_frac * width) - 1, :3]

    # The exterior ring is filled (dark); the hole's centre is the white
    # background — i.e. the hole is cut out, not painted over (H1).
    assert pixel(0.1, 0.1).sum() < 60
    assert pixel(0.5, 0.5).sum() > 600
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


def test_add_features_reproject_polygon_drops_nonfinite_at_poles(cache: Path):
    """Reprojecting a pole-touching polygon to 3857 must not emit inf (M1)."""
    pytest.importorskip("pyproj", reason="pyproj not installed (tiles extra)")
    _write_layer(
        cache,
        "land",
        "110m",
        {
            "type": "Polygon",
            "coordinates": [[[-10, -90], [10, -90], [10, -80], [-10, -80], [-10, -90]]],
        },
    )
    fig, ax = plt.subplots()
    ax.plot([0, 1e6], [0, 1e6])
    add_features(ax, "land", "110m", crs=3857)
    pc = next(c for c in ax.collections if isinstance(c, PathCollection))
    for path in pc.get_paths():
        assert np.isfinite(path.vertices).all()
    plt.close(fig)


def test_add_features_reproject_line_drops_nonfinite_at_poles(cache: Path):
    """A meridian line through the pole keeps only its finite span (M1)."""
    pytest.importorskip("pyproj", reason="pyproj not installed (tiles extra)")
    _write_layer(
        cache,
        "coastline",
        "110m",
        {"type": "LineString", "coordinates": [[0, -90], [0, -80], [0, 0], [0, 80]]},
    )
    fig, ax = plt.subplots()
    ax.plot([0, 1e6], [0, 1e6])
    add_features(ax, "coastline", "110m", crs=3857)
    lc = next(c for c in ax.collections if isinstance(c, LineCollection))
    segments = lc.get_segments()
    assert segments  # the finite span survived
    for seg in segments:
        assert np.isfinite(seg).all()
    plt.close(fig)


def test_natural_earth_corrupt_cache_self_heals(cache: Path):
    """A poisoned (non-gzip) cache file is removed so a retry re-fetches (L1)."""
    bad = cache / "ne_110m_coastline.geojson.gz"
    bad.write_bytes(b"this is not gzip data")
    with pytest.raises(OSError, match="removed"):
        natural_earth("coastline", "110m")
    assert not bad.exists()


def test_natural_earth_non_featurecollection_cache_self_heals(cache: Path):
    """Valid gzip+JSON that is not a FeatureCollection is also removed (L1)."""
    bad = cache / "ne_110m_coastline.geojson.gz"
    with gzip.open(bad, "wt", encoding="utf-8") as fh:
        json.dump({"message": "Not Found"}, fh)
    with pytest.raises(OSError, match="removed"):
        natural_earth("coastline", "110m")
    assert not bad.exists()


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


def test_relief_corrupt_cache_self_heals(cache: Path):
    """A poisoned (non-image) relief cache file is removed on read (L1)."""
    pytest.importorskip("PIL.Image", reason="Pillow not installed (tiles extra)")
    bad = cache / "ne_hypso_rgb_720x360.png"
    bad.write_bytes(b"this is not a png")
    with pytest.raises(OSError, match="removed"):
        relief("low")
    assert not bad.exists()


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


def test_download_streams_to_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A successful download is streamed to `dest`, leaving no `.part` file."""
    payload = b"reference-asset-bytes"
    monkeypatch.setattr(
        reference.urllib.request,
        "urlopen",
        lambda request, timeout=None: io.BytesIO(payload),
    )
    dest = tmp_path / "asset.bin"
    out = reference._download("https://example.com/asset.bin", dest)
    assert out == dest
    assert dest.read_bytes() == payload
    assert not (tmp_path / "asset.bin.part").exists()


def test_download_failure_raises_and_cleans_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A failed fetch raises ConnectionError and removes the partial file."""

    def boom(request, timeout=None):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(reference.urllib.request, "urlopen", boom)
    dest = tmp_path / "asset.bin"
    with pytest.raises(ConnectionError, match="Failed to download"):
        reference._download("https://example.com/asset.bin", dest)
    assert not dest.exists()
    assert not (tmp_path / "asset.bin.part").exists()


# --- helpers: geometry, orientation, finite runs ----------------------------


def test_polygons_multipolygon_keeps_rings():
    geom = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[0, 0], [1, 0], [1, 1], [0, 0]], [[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.2]]],
            [[[5, 5], [6, 5], [6, 6], [5, 5]]],
        ],
    }
    polys = reference._polygons(geom)
    assert [len(p) for p in polys] == [2, 1]


def test_polygons_non_polygon_returns_empty():
    assert reference._polygons({"type": "LineString", "coordinates": [[0, 0], [1, 1]]}) == []


def test_signed_area_sign():
    ccw = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    assert reference._signed_area(ccw) > 0
    assert reference._signed_area(ccw[::-1]) < 0


def test_orient_forces_requested_winding():
    ccw = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    assert reference._signed_area(reference._orient(ccw, ccw=True)) > 0
    assert reference._signed_area(reference._orient(ccw, ccw=False)) < 0
    # Already-correct winding is returned unchanged (no needless copy/flip).
    assert np.array_equal(reference._orient(ccw, ccw=True), ccw)


@pytest.mark.parametrize(
    "rows, expected_lengths",
    [
        ([], []),
        ([[0, 0]], []),
        ([[0, 0], [1, 1]], [2]),
        ([[0, 0], [1, 1], [np.inf, 0], [2, 2], [3, 3]], [2, 2]),
        ([[np.inf, np.inf], [1, 1], [2, 2]], [2]),
        ([[0, 0], [1, 1], [np.inf, 0], [2, 2]], [2]),
        ([[0, 0], [np.inf, 0], [2, 2], [3, 3]], [2]),
    ],
)
def test_finite_runs(rows, expected_lengths):
    """`_finite_runs` keeps maximal finite spans of length >= 2."""
    arr = np.array(rows, dtype=float).reshape(-1, 2) if rows else np.empty((0, 2))
    runs = reference._finite_runs(arr)
    assert [len(r) for r in runs] == expected_lengths
    for run in runs:
        assert np.isfinite(run).all()


def test_compound_path_empty_returns_none():
    assert reference._compound_path([]) is None


def test_polygon_paths_skips_degenerate_rings():
    """A polygon whose only ring has < 3 points yields no drawable path."""
    geom = {"type": "Polygon", "coordinates": [[[0, 0], [1, 1]]]}
    assert reference._polygon_paths([geom], None) == []


# --- dependency guards ------------------------------------------------------


def test_relief_requires_pillow(monkeypatch: pytest.MonkeyPatch):
    """`relief` raises a helpful ImportError when Pillow is unavailable."""
    monkeypatch.setattr(reference, "_PILLOW_AVAILABLE", False)
    with pytest.raises(ImportError, match="Pillow"):
        relief("low")


def test_make_transformer_requires_pyproj(monkeypatch: pytest.MonkeyPatch):
    """`_make_transformer` raises ImportError when pyproj is unavailable."""
    real_find_spec = reference.importlib.util.find_spec
    monkeypatch.setattr(
        reference.importlib.util,
        "find_spec",
        lambda name: None if name == "pyproj" else real_find_spec(name),
    )
    with pytest.raises(ImportError, match="pyproj"):
        reference._make_transformer(3857)


def test_make_transformer_invalid_crs_raises_valueerror():
    """An unparseable CRS surfaces a wrapped ValueError, not a pyproj error (L2)."""
    pytest.importorskip("pyproj", reason="pyproj not installed (tiles extra)")
    with pytest.raises(ValueError, match="Invalid CRS"):
        reference._make_transformer("not-a-real-crs")


def test_reproject_arr_roundtrip():
    """`_reproject_arr` maps EPSG:4326 lon/lat into the target CRS."""
    pytest.importorskip("pyproj", reason="pyproj not installed (tiles extra)")
    transformer = reference._make_transformer(3857)
    out = reference._reproject_arr(np.array([[0.0, 0.0], [10.0, 0.0]]), transformer)
    assert out.shape == (2, 2)
    assert abs(out[0, 0]) < 1.0
    assert out[1, 0] > 1.0e6
