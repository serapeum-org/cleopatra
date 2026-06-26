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
import inspect
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


def test_crs_defaults_to_self_crs_when_omitted(monkeypatch):
    """add_features/add_tiles fall back to self.crs when crs= is omitted."""
    seen = {}
    monkeypatch.setattr(refmod, "add_features", lambda ax, *a, **k: seen.update(k) or ax)
    fig, ax = plt.subplots()
    glyph = _Dummy(ax)
    glyph.crs = 4326
    glyph.add_features("coastline", "50m")
    assert seen.get("crs") == 4326, f"expected crs defaulted to 4326, got {seen.get('crs')}"
    plt.close(fig)

    seen.clear()
    monkeypatch.setattr(tilesmod, "add_tiles", lambda ax, *a, **k: seen.update(k) or ax)
    fig, ax = plt.subplots()
    glyph = _Dummy(ax)
    glyph.crs = "EPSG:3857"
    glyph.add_tiles()
    assert seen.get("crs") == "EPSG:3857", f"expected crs defaulted, got {seen.get('crs')}"
    plt.close(fig)


def test_explicit_crs_overrides_self_crs(monkeypatch):
    """An explicit crs= wins over self.crs."""
    seen = {}
    monkeypatch.setattr(refmod, "add_features", lambda ax, *a, **k: seen.update(k) or ax)
    fig, ax = plt.subplots()
    glyph = _Dummy(ax)
    glyph.crs = 4326
    glyph.add_features("coastline", "50m", crs=3857)
    assert seen.get("crs") == 3857, f"explicit crs should win, got {seen.get('crs')}"
    plt.close(fig)


def test_unset_crs_is_passthrough(monkeypatch):
    """With self.crs unset (None), no crs is injected (helper default preserved)."""
    seen = {}
    monkeypatch.setattr(refmod, "add_features", lambda ax, *a, **k: seen.update(kwargs=k) or ax)
    fig, ax = plt.subplots()
    _Dummy(ax).add_features("coastline", "50m")  # crs left at class default None
    assert "crs" not in seen["kwargs"], f"crs should not be injected, got {seen['kwargs']}"
    plt.close(fig)


def test_add_relief_ignores_self_crs(monkeypatch):
    """add_relief never receives crs, even when self.crs is set."""
    seen = {}
    monkeypatch.setattr(refmod, "add_relief", lambda ax, *a, **k: seen.update(kwargs=k) or ax)
    fig, ax = plt.subplots()
    glyph = _Dummy(ax)
    glyph.crs = 4326
    glyph.add_relief("low")
    assert "crs" not in seen["kwargs"], f"add_relief must not get crs, got {seen['kwargs']}"
    plt.close(fig)


def test_basemap_kwargs_helper():
    """_basemap_kwargs injects only when self.crs is set and crs is absent."""
    d = _Dummy(None)
    assert d._basemap_kwargs({}) == {}                      # crs unset -> passthrough
    assert d._basemap_kwargs({"crs": 3857}) == {"crs": 3857}
    d.crs = 4326
    assert d._basemap_kwargs({}) == {"crs": 4326}           # injected
    assert d._basemap_kwargs({"crs": 3857}) == {"crs": 3857}  # explicit wins
    assert d._basemap_kwargs({"crs": None}) == {"crs": 4326}  # None treated as unset


@pytest.mark.parametrize("fn", [tilesmod.add_tiles, refmod.add_features])
def test_crs_is_keyword_only_in_helpers(fn):
    """crs is keyword-only in add_tiles/add_features, so it cannot be positional."""
    kind = inspect.signature(fn).parameters["crs"].kind
    assert kind is inspect.Parameter.KEYWORD_ONLY, f"{fn.__name__}.crs is {kind}"


def test_default_crs_is_none_on_geomixin():
    """A GeoMixin host's crs defaults to None and is exposed as a property."""
    assert _Dummy(None).crs is None
    assert isinstance(type(_Dummy(None)).crs, property)


def test_crs_accepts_valid_values():
    """int EPSG codes, CRS strings, and None are accepted and round-trip."""
    g = _Dummy(None)
    g.crs = 4326
    assert g.crs == 4326
    g.crs = "EPSG:3857"
    assert g.crs == "EPSG:3857"
    g.crs = None
    assert g.crs is None


def test_crs_normalizes_bare_numeric_string():
    """A digits-only CRS string is normalised to an int EPSG code on assignment."""
    g = _Dummy(None)
    g.crs = "4326"
    assert g.crs == 4326
    g.crs = " 3857 "  # stripped, then normalised
    assert g.crs == 3857


def test_crs_rejects_bad_type():
    """Non int/str/None (including bool) is rejected at assignment with TypeError."""
    g = _Dummy(None)
    with pytest.raises(TypeError, match="crs must be"):
        g.crs = [4326]
    with pytest.raises(TypeError, match="crs must be"):
        g.crs = True  # bool is not a valid EPSG code


def test_crs_rejects_nonpositive_or_empty():
    """A non-positive EPSG code or a blank string is rejected with ValueError."""
    g = _Dummy(None)
    with pytest.raises(ValueError, match="positive int"):
        g.crs = 0
    with pytest.raises(ValueError, match="non-empty"):
        g.crs = "   "


def test_crs_rejects_unresolvable_when_pyproj_available():
    """An unresolvable CRS is caught at assignment when pyproj is installed."""
    pytest.importorskip("pyproj", reason="pyproj not installed (tiles extra)")
    g = _Dummy(None)
    with pytest.raises(ValueError, match="Invalid CRS"):
        g.crs = "definitely-not-a-crs"


def test_crs_skips_deep_validation_without_pyproj(monkeypatch):
    """Without pyproj, a well-typed-but-unresolvable CRS is accepted (deferred)."""
    import importlib.util as ilu

    real_find_spec = ilu.find_spec
    monkeypatch.setattr(
        ilu, "find_spec", lambda name: None if name == "pyproj" else real_find_spec(name)
    )
    g = _Dummy(None)
    g.crs = "deferred-to-draw-time"  # no deep check -> accepted
    assert g.crs == "deferred-to-draw-time"


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


def test_glyph_crs_drives_reprojected_placement(tmp_path: Path, monkeypatch):
    """End-to-end: glyph.crs alone reprojects a drawn layer to that CRS."""
    pytest.importorskip("pyproj", reason="pyproj not installed (tiles extra)")
    monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path))
    collection = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[0, 0], [10, 0]]}}
        ],
    }
    with gzip.open(tmp_path / "ne_110m_coastline.geojson.gz", "wt", encoding="utf-8") as fh:
        json.dump(collection, fh)

    fig, ax = plt.subplots()
    glyph = PolygonGlyph([np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])], ax=ax)
    glyph.crs = 3857  # axis CRS recorded once; no crs= on the draw call
    glyph.add_features("coastline", "110m")
    lc = next(c for c in ax.collections if isinstance(c, LineCollection))
    verts = lc.get_paths()[0].vertices
    # In EPSG:3857, lon=0 -> x~=0 m and lon=10 -> x~=1.11e6 m (not lon/lat degrees).
    assert abs(verts[0][0]) < 1.0, f"first vertex not at x~=0: {verts[0]}"
    assert verts[1][0] > 1.0e6, f"second vertex not reprojected to metres: {verts[1]}"
    plt.close(fig)
