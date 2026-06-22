"""Tests for cleopatra.geo.GeoMixin.

Covers which glyphs inherit the basemap convenience methods, that the
methods delegate to the standalone `cleopatra.tiles` / `cleopatra.reference`
functions with the glyph's axes, the `ax=` override, and the
no-axes-yet error. Delegation is checked with spies so no test needs the
network or the `[tiles]` extra; one integration test exercises a real
glyph against a synthetic on-disk cache.
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
from matplotlib.collections import LineCollection  # noqa: E402

import cleopatra.reference as refmod  # noqa: E402
import cleopatra.tiles as tilesmod  # noqa: E402
from cleopatra.array_glyph import ArrayGlyph  # noqa: E402
from cleopatra.flow_glyph import FlowGlyph  # noqa: E402
from cleopatra.geo import GeoMixin  # noqa: E402
from cleopatra.kde_glyph import KDEGlyph  # noqa: E402
from cleopatra.line_glyph import LineGlyph  # noqa: E402
from cleopatra.mesh_glyph import MeshGlyph  # noqa: E402
from cleopatra.polygon_glyph import PolygonGlyph  # noqa: E402
from cleopatra.scatter_glyph import ScatterGlyph  # noqa: E402
from cleopatra.statistical_glyph import StatisticalGlyph  # noqa: E402
from cleopatra.vector_glyph import VectorGlyph  # noqa: E402

GEO_GLYPHS = [ArrayGlyph, MeshGlyph, VectorGlyph, FlowGlyph, PolygonGlyph, ScatterGlyph]
NON_GEO_GLYPHS = [LineGlyph, StatisticalGlyph, KDEGlyph]
METHODS = ("add_tiles", "add_features", "add_relief")


class _Dummy(GeoMixin):
    """Minimal GeoMixin host exposing an `ax` attribute, like a glyph."""

    def __init__(self, ax=None):
        self.ax = ax


@pytest.mark.parametrize("cls", GEO_GLYPHS)
def test_geographic_glyphs_inherit_basemap_methods(cls):
    """Each geographic glyph subclasses GeoMixin and exposes all three methods."""
    assert issubclass(cls, GeoMixin), f"{cls.__name__} should inherit GeoMixin"
    for name in METHODS:
        assert callable(getattr(cls, name, None)), f"{cls.__name__}.{name} missing"


@pytest.mark.parametrize("cls", NON_GEO_GLYPHS)
def test_nongeographic_glyphs_lack_basemap_methods(cls):
    """Chart/statistical glyphs do not inherit the geo-only methods."""
    assert not issubclass(cls, GeoMixin), f"{cls.__name__} should not inherit GeoMixin"
    for name in METHODS:
        assert not hasattr(cls, name), f"{cls.__name__} unexpectedly has {name}"


def test_add_features_delegates_with_axes_and_args(monkeypatch):
    """add_features forwards self.ax plus positional/keyword args to the function."""
    seen = {}

    def spy(ax, *args, **kwargs):
        seen.update(ax=ax, args=args, kwargs=kwargs)
        return ax

    monkeypatch.setattr(refmod, "add_features", spy)
    fig, ax = plt.subplots()
    result = _Dummy(ax).add_features("coastline", "50m", colors="navy")
    assert result is ax, "method should return the function's result"
    assert seen["ax"] is ax, f"expected self.ax forwarded, got {seen['ax']}"
    assert seen["args"] == ("coastline", "50m"), f"args not forwarded: {seen['args']}"
    assert seen["kwargs"] == {"colors": "navy"}, f"kwargs not forwarded: {seen['kwargs']}"
    plt.close(fig)


def test_add_relief_delegates(monkeypatch):
    """add_relief forwards self.ax and arguments to reference.add_relief."""
    seen = {}
    monkeypatch.setattr(
        refmod, "add_relief", lambda ax, *a, **k: seen.update(ax=ax, a=a, k=k) or ax
    )
    fig, ax = plt.subplots()
    _Dummy(ax).add_relief("low", alpha=0.5)
    assert seen["ax"] is ax and seen["a"] == ("low",) and seen["k"] == {"alpha": 0.5}
    plt.close(fig)


def test_add_tiles_delegates(monkeypatch):
    """add_tiles forwards self.ax and arguments to tiles.add_tiles."""
    seen = {}
    monkeypatch.setattr(
        tilesmod, "add_tiles", lambda ax, *a, **k: seen.update(ax=ax, a=a, k=k) or ax
    )
    fig, ax = plt.subplots()
    _Dummy(ax).add_tiles(crs=3857)
    assert seen["ax"] is ax and seen["k"] == {"crs": 3857}
    plt.close(fig)


def test_ax_override_takes_precedence(monkeypatch):
    """An explicit ax= overrides the glyph's own axes."""
    seen = {}
    monkeypatch.setattr(refmod, "add_features", lambda ax, *a, **k: seen.update(ax=ax) or ax)
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    _Dummy(ax1).add_features("coastline", ax=ax2)
    assert seen["ax"] is ax2, "ax= should override self.ax"
    plt.close(fig1)
    plt.close(fig2)


def test_no_axes_raises():
    """Calling a basemap method before plotting (no axes) raises RuntimeError."""
    with pytest.raises(RuntimeError, match="Plot the glyph first"):
        _Dummy(None).add_features("coastline")


def test_real_glyph_integration(tmp_path: Path, monkeypatch):
    """A real glyph draws a cached layer on its own axes via the mixin method."""
    monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path))
    collection = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[0, 0], [10, 10]]}}
        ],
    }
    with gzip.open(tmp_path / "ne_110m_coastline.geojson.gz", "wt", encoding="utf-8") as fh:
        json.dump(collection, fh)

    fig, ax = plt.subplots()
    glyph = PolygonGlyph([np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])], ax=ax)
    glyph.add_features("coastline", "110m", colors="navy")
    assert any(isinstance(c, LineCollection) for c in ax.collections)
    plt.close(fig)
