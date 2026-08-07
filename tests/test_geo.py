"""Tests for cleopatra.basemap.geo.GeoMixin.

Covers which glyphs inherit the basemap convenience methods, that the
methods delegate to the standalone `cleopatra.basemap.tiles` / `cleopatra.basemap.reference`
functions with the glyph's axes, the `ax=` override, and the
no-axes-yet error. Delegation is checked with spies so no test needs the
network or the `[tiles]` extra; one integration test exercises a real
glyph against a synthetic on-disk cache.
"""

from __future__ import annotations

import gzip
import inspect
import json
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.plot

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

import cleopatra.basemap.reference as refmod  # noqa: E402
import cleopatra.basemap.tiles as tilesmod  # noqa: E402
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph  # noqa: E402
from cleopatra.glyphs.primitives.flow_glyph import FlowGlyph  # noqa: E402
from cleopatra.basemap.geo import (  # noqa: E402
    REFERENCE_MAP_STYLES,
    Basemap,
    Feature,
    GeoMixin,
    _lat_formatter,
    _lon_formatter,
    _nice_step,
    add_point_labels,
    available_map_styles,
)
from cleopatra.glyphs.stats.kde_glyph import KDEGlyph  # noqa: E402
from cleopatra.glyphs.primitives.line_glyph import LineGlyph  # noqa: E402
from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph  # noqa: E402
from cleopatra.glyphs.primitives.polygon_glyph import PolygonGlyph  # noqa: E402
from cleopatra.glyphs.primitives.scatter_glyph import ScatterGlyph  # noqa: E402
from cleopatra.glyphs.stats.histogram_glyph import HistogramGlyph  # noqa: E402
from cleopatra.glyphs.gridded.vector_glyph import VectorGlyph  # noqa: E402

GEO_GLYPHS = [ArrayGlyph, MeshGlyph, VectorGlyph, FlowGlyph, PolygonGlyph, ScatterGlyph]
NON_GEO_GLYPHS = [LineGlyph, HistogramGlyph, KDEGlyph]
METHODS = ("add_tiles", "add_features", "add_relief", "add_reference_map", "add_labels")


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
    assert seen["kwargs"] == {"colors": "navy"}, (
        f"kwargs not forwarded: {seen['kwargs']}"
    )
    plt.close(fig)


def test_add_relief_delegates(monkeypatch):
    """add_relief forwards self.ax and arguments to reference.add_relief."""
    seen = {}
    monkeypatch.setattr(
        refmod, "add_relief", lambda ax, *a, **k: seen.update(ax=ax, a=a, k=k) or ax
    )
    fig, ax = plt.subplots()
    _Dummy(ax).add_relief("low", alpha=0.5)
    assert seen['ax'] is ax
    assert seen['a'] == ('low',)
    assert seen['k'] == {'alpha': 0.5}
    plt.close(fig)


def test_add_labels_delegates(monkeypatch):
    """add_labels forwards self.ax, points, and kwargs to geo.add_point_labels."""
    import cleopatra.basemap.geo as geomod

    seen = {}
    monkeypatch.setattr(
        geomod,
        "add_point_labels",
        lambda ax, points, **k: seen.update(ax=ax, points=points, k=k) or ax,
    )
    fig, ax = plt.subplots()
    points = {"London": (-0.1, 51.5)}
    _Dummy(ax).add_labels(points, color="yellow")
    assert seen["ax"] is ax
    assert seen["points"] == points
    assert seen["k"] == {"color": "yellow"}
    plt.close(fig)


def test_add_labels_no_axes_raises():
    """Calling add_labels before plotting (no axes) raises RuntimeError."""
    with pytest.raises(RuntimeError, match="Plot the glyph first"):
        _Dummy(None).add_labels({"London": (-0.1, 51.5)})


class TestAddPointLabels:
    """Tests for the standalone `cleopatra.basemap.geo.add_point_labels` function."""

    @pytest.fixture
    def ax(self):
        """A fresh Axes on the Agg backend, closed after the test."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_draws_one_marker_and_label_per_point(self, ax):
        """Each point in the mapping gets exactly one marker line and one text."""
        add_point_labels(ax, {"London": (-0.1, 51.5), "Moscow": (37.6, 55.8)})
        assert len(ax.lines) == 2, f"expected 2 markers, got {len(ax.lines)}"
        assert len(ax.texts) == 2, f"expected 2 labels, got {len(ax.texts)}"

    def test_label_text_matches_the_key(self, ax):
        """The drawn text matches the point's label (dict key)."""
        add_point_labels(ax, {"Reykjavik": (-21.9, 64.1)})
        assert ax.texts[0].get_text() == "Reykjavik"

    def test_marker_placed_at_the_given_coordinates(self, ax):
        """The marker is drawn at exactly the given (x, y)."""
        add_point_labels(ax, {"Nuuk": (-51.7, 64.2)})
        line = ax.lines[0]
        assert line.get_xdata() == [-51.7]
        assert line.get_ydata() == [64.2]

    def test_empty_points_draws_nothing_but_returns_ax(self, ax):
        """An empty mapping draws no artists and still returns `ax`."""
        result = add_point_labels(ax, {})
        assert result is ax, "should return ax even with no points"
        assert len(ax.lines) == 0
        assert len(ax.texts) == 0

    def test_returns_the_same_axes(self, ax):
        """The function returns `ax` itself, enabling call chaining."""
        result = add_point_labels(ax, {"London": (-0.1, 51.5)})
        assert result is ax

    def test_custom_color_and_fontsize_applied(self, ax):
        """`color`/`fontsize` are applied to both the marker and the label."""
        add_point_labels(ax, {"London": (-0.1, 51.5)}, color="red", fontsize=14)
        assert ax.lines[0].get_color() == "red"
        assert ax.texts[0].get_color() == "red"
        assert ax.texts[0].get_fontsize() == 14

    def test_composes_with_existing_plot(self, ax):
        """Labelling does not disturb artists already drawn on `ax`."""
        img = ax.imshow([[0, 1], [1, 0]], cmap="gray")
        add_point_labels(ax, {"London": (-0.1, 51.5)})
        assert img in ax.images, "pre-existing plot should be untouched"


def test_add_tiles_delegates(monkeypatch):
    """add_tiles forwards self.ax and arguments to tiles.add_tiles."""
    seen = {}
    monkeypatch.setattr(
        tilesmod, "add_tiles", lambda ax, *a, **k: seen.update(ax=ax, a=a, k=k) or ax
    )
    fig, ax = plt.subplots()
    _Dummy(ax).add_tiles(crs=3857)
    assert seen['ax'] is ax
    assert seen['k'] == {'crs': 3857}
    plt.close(fig)


def test_crs_defaults_to_self_crs_when_omitted(monkeypatch):
    """add_features/add_tiles fall back to self.crs when crs= is omitted."""
    seen = {}
    monkeypatch.setattr(
        refmod, "add_features", lambda ax, *a, **k: seen.update(k) or ax
    )
    fig, ax = plt.subplots()
    glyph = _Dummy(ax)
    glyph.crs = 4326
    glyph.add_features("coastline", "50m")
    assert seen.get("crs") == 4326, (
        f"expected crs defaulted to 4326, got {seen.get('crs')}"
    )
    plt.close(fig)

    seen.clear()
    monkeypatch.setattr(tilesmod, "add_tiles", lambda ax, *a, **k: seen.update(k) or ax)
    fig, ax = plt.subplots()
    glyph = _Dummy(ax)
    glyph.crs = "EPSG:3857"
    glyph.add_tiles()
    assert seen.get("crs") == "EPSG:3857", (
        f"expected crs defaulted, got {seen.get('crs')}"
    )
    plt.close(fig)


def test_explicit_crs_overrides_self_crs(monkeypatch):
    """An explicit crs= wins over self.crs."""
    seen = {}
    monkeypatch.setattr(
        refmod, "add_features", lambda ax, *a, **k: seen.update(k) or ax
    )
    fig, ax = plt.subplots()
    glyph = _Dummy(ax)
    glyph.crs = 4326
    glyph.add_features("coastline", "50m", crs=3857)
    assert seen.get("crs") == 3857, f"explicit crs should win, got {seen.get('crs')}"
    plt.close(fig)


def test_unset_crs_is_passthrough(monkeypatch):
    """With self.crs unset (None), no crs is injected (helper default preserved)."""
    seen = {}
    monkeypatch.setattr(
        refmod, "add_features", lambda ax, *a, **k: seen.update(kwargs=k) or ax
    )
    fig, ax = plt.subplots()
    _Dummy(ax).add_features("coastline", "50m")  # crs left at class default None
    assert "crs" not in seen["kwargs"], (
        f"crs should not be injected, got {seen['kwargs']}"
    )
    plt.close(fig)


def test_add_relief_defaults_to_self_crs(monkeypatch):
    """add_relief defaults crs to self.crs, like add_features/add_tiles."""
    seen = {}
    monkeypatch.setattr(
        refmod, "add_relief", lambda ax, *a, **k: seen.update(kwargs=k) or ax
    )
    fig, ax = plt.subplots()
    glyph = _Dummy(ax)
    glyph.crs = 3857
    glyph.add_relief("low")
    assert seen["kwargs"].get("crs") == 3857, (
        f"add_relief should default crs to self.crs, got {seen['kwargs']}"
    )
    plt.close(fig)


def test_basemap_kwargs_helper():
    """_basemap_kwargs injects only when self.crs is set and crs is absent."""
    d = _Dummy(None)
    assert d._basemap_kwargs({}) == {}  # crs unset -> passthrough
    assert d._basemap_kwargs({"crs": 3857}) == {"crs": 3857}
    d.crs = 4326
    assert d._basemap_kwargs({}) == {"crs": 4326}  # injected
    assert d._basemap_kwargs({"crs": 3857}) == {"crs": 3857}  # explicit wins
    assert d._basemap_kwargs({"crs": None}) == {"crs": 4326}  # None treated as unset


@pytest.mark.parametrize(
    "fn", [tilesmod.add_tiles, refmod.add_features, refmod.add_relief]
)
def test_crs_is_keyword_only_in_helpers(fn):
    """crs is keyword-only in add_tiles/add_features/add_relief, not positional."""
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
        ilu,
        "find_spec",
        lambda name: None if name == "pyproj" else real_find_spec(name),
    )
    g = _Dummy(None)
    g.crs = "deferred-to-draw-time"  # no deep check -> accepted
    assert g.crs == "deferred-to-draw-time"


def test_ax_override_takes_precedence(monkeypatch):
    """An explicit ax= overrides the glyph's own axes."""
    seen = {}
    monkeypatch.setattr(
        refmod, "add_features", lambda ax, *a, **k: seen.update(ax=ax) or ax
    )
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


def test_real_glyph_basemap_axes_lazily_creates_and_seeds_bounds():
    """A real glyph creates its axes on demand for a basemap layer added before
    plotting, seeded with the data bounds so the later draw keeps the view.

    Test scenario:
        `ArrayGlyph` has no axes until it draws; `_basemap_axes` (the resolver
        every add_* method uses) now creates one when the glyph can
        (`create_figure_axes`) and seeds it with the glyph's `_flat_axis_bounds`,
        so the builder flow `glyph.add_features(...)` then `glyph.plot()`/
        `.animate()` works (the latter reuse the same axes).
    """
    glyph = ArrayGlyph(np.zeros((6, 8)), extent=[-12.0, 32.0, 34.0, 64.0])
    assert glyph.ax is None, "a fresh ArrayGlyph has no axes yet"
    ax = glyph._basemap_axes()
    assert glyph.ax is not None, "should create and store the axes"
    assert ax is glyph.ax
    x_min, x_max, y_min, y_max = glyph._flat_axis_bounds()
    assert tuple(round(v) for v in ax.get_xlim()) == (round(x_min), round(x_max)), "x seeded to data bounds"
    assert tuple(round(v) for v in ax.get_ylim()) == (round(y_min), round(y_max)), "y seeded to data bounds"
    plt.close("all")


def test_non_seeding_glyph_basemap_axes_raises_clearly():
    """A glyph that cannot seed its own bounds refuses to lazily create an axes.

    Test scenario:
        Only `ArrayGlyph` exposes `_flat_axis_bounds`. A `FlowGlyph` (which has
        no such bounds) that added a basemap layer before plotting would otherwise
        get an unseeded `(0, 1)` axes that pins the view and silently breaks the
        later plot, so `_basemap_axes` keeps raising the clear "plot first" error
        rather than returning an unusable axes.
    """
    glyph = FlowGlyph([np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])])
    assert glyph.ax is None, "a fresh FlowGlyph has no axes yet"
    assert not hasattr(glyph, "_flat_axis_bounds"), "FlowGlyph cannot seed its own bounds"
    with pytest.raises(RuntimeError, match="Plot the glyph first"):
        glyph._basemap_axes()
    plt.close("all")


def test_pixel_index_basemap_axes_seeds_matshow_orientation():
    """The pre-plot builder flow seeds a pixel-index array in matshow orientation.

    Test scenario:
        A non-georeferenced pixel array plots via `matshow(origin="upper")` (row 0
        at the top, inverted y, half-pixel edges). `_basemap_axes` must seed that
        same view, or a reference/label layer added before `plot()` locks a
        non-inverted box (autoscale off) and the raster renders upside-down.
    """
    glyph = ArrayGlyph(np.arange(24, dtype=float).reshape(4, 6))  # 4 rows x 6 cols, no extent/coords
    ax = glyph._basemap_axes()
    assert ax.get_ylim() == (3.5, -0.5), "y inverted (row 0 at top), matching matshow"
    assert ax.get_xlim() == (-0.5, 5.5), "x half-pixel edges, matching matshow"
    plt.close("all")


def test_real_glyph_integration(tmp_path: Path, monkeypatch):
    """A real glyph draws a cached layer on its own axes via the mixin method."""
    monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path))
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [10, 10]]},
            }
        ],
    }
    with gzip.open(
        tmp_path / "ne_110m_coastline.geojson.gz", "wt", encoding="utf-8"
    ) as fh:
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
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [10, 0]]},
            }
        ],
    }
    with gzip.open(
        tmp_path / "ne_110m_coastline.geojson.gz", "wt", encoding="utf-8"
    ) as fh:
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


def test_glyph_crs_drives_relief_warp(tmp_path: Path, monkeypatch):
    """End-to-end: glyph.crs alone warps the relief backdrop to that CRS."""
    pytest.importorskip("pyproj", reason="pyproj not installed (tiles extra)")
    Image = pytest.importorskip(
        "PIL.Image", reason="Pillow not installed (tiles extra)"
    )
    monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path))
    arr = np.zeros((4, 8, 3), dtype="uint8")
    arr[:, :4] = (0, 0, 255)
    arr[:, 4:] = (255, 0, 0)
    Image.fromarray(arr).save(tmp_path / "ne_hypso_rgb_720x360.png")

    fig, ax = plt.subplots()
    glyph = PolygonGlyph([np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])], ax=ax)
    glyph.crs = 3857  # axis CRS recorded once; no crs= on the draw call
    ax.set_xlim(-2e7, 2e7)
    ax.set_ylim(-1e7, 1e7)
    glyph.add_relief("low")
    placed = np.asarray(ax.images[0].get_array())
    assert placed.shape[2] == 4, f"relief should warp to RGBA, got {placed.shape}"
    w = placed.shape[1]
    west = placed[:, : w // 4].reshape(-1, 4)
    east = placed[:, -(w // 4) :].reshape(-1, 4)
    west = west[west[:, 3] > 0]
    east = east[east[:, 3] > 0]
    assert west.size, 'expected opaque cells on both strips'
    assert east.size, 'expected opaque cells on both strips'
    assert (west[:, 2] > west[:, 0]).all(), "west should stay blue after the warp"
    assert (east[:, 0] > east[:, 2]).all(), "east should stay red after the warp"
    plt.close(fig)


class TestAddReferenceMap:
    """`GeoMixin.add_reference_map` reference-map style preset (issue #184)."""

    @staticmethod
    def _host(extent=None, im=None):
        """A GeoMixin host with a real axes and mocked `add_features`/`add_relief`."""
        fig, ax = plt.subplots()
        host = _Dummy(ax=ax)
        host.extent = extent
        host.im = im
        host.crs = None
        host.add_features = MagicMock(return_value=ax)
        host.add_relief = MagicMock(return_value=ax)
        return host, fig, ax

    def test_available_map_styles(self):
        """The built-in preset names are exposed and stable."""
        assert available_map_styles() == ["ecmwf", "ecmwf-dark"]

    @pytest.mark.parametrize(
        "value,expected",
        [
            (-75, "75°W"),
            (10, "10°E"),
            (0, "0°"),
            (180, "180°"),
            (-180, "180°"),
            (200, "160°W"),
        ],
    )
    def test_lon_formatter(self, value, expected):
        """Longitude ticks label W/E, 0, and the ±180° antimeridian (L1)."""
        assert _lon_formatter(value) == expected

    @pytest.mark.parametrize("value,expected", [(-20, "20°S"), (45, "45°N"), (0, "0°")])
    def test_lat_formatter(self, value, expected):
        """Latitude ticks label S/N and the equator."""
        assert _lat_formatter(value) == expected

    def test_composes_features_graticule_and_frame(self):
        """The preset draws coastline+borders and styles graticule/labels/frame."""
        host, fig, ax = self._host(extent=[-100, 20, -80, 40])
        ret = host.add_reference_map("ecmwf")

        assert ret is ax, "should return the axes for chaining"
        layers = [c.args[0] for c in host.add_features.call_args_list]
        assert layers == ["coastline", "borders"], layers
        coast = host.add_features.call_args_list[0]
        assert coast.kwargs["colors"] == "0.45"
        assert coast.kwargs["linewidths"] == 0.8
        assert ax.xaxis.get_major_formatter()(-75) == "75°W"
        assert ax.yaxis.get_major_formatter()(40) == "40°N"
        assert ax.spines["bottom"].get_edgecolor() == (0.6, 0.6, 0.6, 1.0)
        plt.close(fig)

    def test_dark_style_uses_lighter_greys(self):
        """`ecmwf-dark` uses light-grey coastlines for dark backgrounds."""
        host, fig, ax = self._host(extent=[-100, 20, -80, 40])
        host.add_reference_map("ecmwf-dark")
        assert host.add_features.call_args_list[0].kwargs["colors"] == "0.85"
        plt.close(fig)

    def test_dark_style_draws_relief_backdrop(self):
        """`ecmwf-dark` draws a dimmed relief backdrop under the chrome."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        host.add_reference_map("ecmwf-dark")
        host.add_relief.assert_called_once()
        call = host.add_relief.call_args
        assert call.args[0] == "low", "relief resolution"
        assert call.kwargs["alpha"] == 0.5
        assert call.kwargs["zorder"] == -2
        assert call.kwargs["ax"] is ax
        plt.close(fig)

    def test_light_style_draws_no_relief(self):
        """Plain `ecmwf` stays chrome-only -- no relief backdrop."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        host.add_reference_map("ecmwf")
        host.add_relief.assert_not_called()
        plt.close(fig)

    def test_relief_missing_pillow_degrades_with_warning(self):
        """Without Pillow the relief is skipped with a warning; chrome still drawn."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        host.add_relief = MagicMock(side_effect=ImportError("no Pillow"))
        with pytest.warns(UserWarning, match="relief backdrop skipped"):
            host.add_reference_map("ecmwf-dark")
        host.add_features.assert_called()  # coastline/borders still drawn
        plt.close(fig)

    def test_relief_fetch_failure_degrades_with_warning(self):
        """A relief fetch/decode failure (ConnectionError/OSError) is skipped with
        a warning; the coastline/border chrome is still drawn."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        host.add_relief = MagicMock(side_effect=ConnectionError("offline"))
        with pytest.warns(UserWarning, match="relief backdrop skipped"):
            host.add_reference_map("ecmwf-dark")
        host.add_features.assert_called()  # chrome unaffected by the relief failure
        plt.close(fig)

    def test_custom_relief_bad_resolution_raises(self, monkeypatch):
        """A custom preset's bad relief resolution raises loudly (a config error),
        it is not swallowed by the environmental degrade path."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        host.add_relief = MagicMock(
            side_effect=ValueError("Unknown relief resolution 'ultra'")
        )
        monkeypatch.setitem(
            REFERENCE_MAP_STYLES["ecmwf-dark"], "relief", {"resolution": "ultra"}
        )
        with pytest.raises(ValueError, match="Unknown relief resolution"):
            host.add_reference_map("ecmwf-dark")
        plt.close(fig)

    def test_no_extent_skips_relief(self):
        """With no geographic extent, the relief backdrop is skipped entirely."""
        host, fig, ax = self._host(extent=None)
        with pytest.warns(UserWarning, match="no geographic extent"):
            host.add_reference_map("ecmwf-dark")
        host.add_relief.assert_not_called()
        plt.close(fig)

    @pytest.mark.parametrize(
        "relief_value,expected_resolution",
        [("medium", "medium"), (True, "low")],
    )
    def test_relief_config_forms(self, monkeypatch, relief_value, expected_resolution):
        """A preset `relief` as a resolution string or `True` selects the relief
        resolution (a string overrides it; `True` keeps the `"low"` default)."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        monkeypatch.setitem(REFERENCE_MAP_STYLES["ecmwf-dark"], "relief", relief_value)
        host.add_reference_map("ecmwf-dark")
        assert host.add_relief.call_args.args[0] == expected_resolution, (
            f"expected relief resolution {expected_resolution!r}"
        )
        plt.close(fig)

    def test_auto_picks_dark_on_dark_background(self):
        """`style="auto"` selects `ecmwf-dark` for a dark rendered image."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        host.im = ax.imshow(np.zeros((4, 4, 3)))  # black RGB
        host.add_reference_map("auto")
        assert host.add_features.call_args_list[0].kwargs["colors"] == "0.85"
        plt.close(fig)

    def test_auto_picks_light_on_light_background(self):
        """`style="auto"` selects `ecmwf` for a light rendered image."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        host.im = ax.imshow(np.ones((4, 4, 3)))  # white RGB
        host.add_reference_map("auto")
        assert host.add_features.call_args_list[0].kwargs["colors"] == "0.45"
        plt.close(fig)

    def test_auto_uses_rendered_colours_not_data_magnitude(self):
        """`auto` judges a colormapped field by its rendered colour (M1)."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        # data == 0 (would read "dark" by raw magnitude) but `gray_r` renders
        # 0 as white -> the rendered image is light, so not dark.
        host.im = ax.imshow(np.zeros((4, 4)), cmap="gray_r", vmin=0, vmax=1)
        assert host._background_is_dark(ax) is False
        plt.close(fig)

    def test_background_dark_masked_field_no_warning(self):
        """A fully-masked field yields a plain bool with no NaN warning (L2)."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        masked = np.ma.masked_all((4, 4))
        host.im = ax.imshow(masked, cmap="viridis")
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any RuntimeWarning would fail here
            result = host._background_is_dark(ax)
        assert isinstance(result, bool)
        plt.close(fig)

    def test_background_is_dark_no_image_returns_false(self):
        """With no plotted image, the background reads as not-dark."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        host.im = None
        assert host._background_is_dark(ax) is False
        plt.close(fig)

    def test_auto_ignores_masked_no_data_cells(self):
        """A light field that is mostly no-data is not misread as dark (M1)."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        field = np.ma.masked_array(np.ones((10, 10)))
        field[:6] = np.ma.masked  # 60% no-data; the unmasked cells are bright
        host.im = ax.imshow(field, cmap="viridis", vmin=0, vmax=1)
        assert host._background_is_dark(ax) is False
        plt.close(fig)

    def test_auto_samples_target_axes_image(self):
        """`auto` decides from an image on the target `ax`, not `self.im` (L1)."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        host.im = ax.imshow(np.ones((4, 4, 3)))  # glyph's own axes: white/light
        fig2, other = plt.subplots()
        other.imshow(np.zeros((4, 4, 3)))  # target axes: black/dark
        host.add_reference_map("auto", ax=other)
        assert host.add_features.call_args_list[0].kwargs["colors"] == "0.85"
        plt.close(fig)
        plt.close(fig2)

    def test_extent_sets_image_and_axis_limits(self):
        """`extent=[xmin, ymin, xmax, ymax]` (ArrayGlyph order) sets image + limits."""
        im = MagicMock()
        host, fig, ax = self._host(im=im)
        # [west, south, east, north] == [xmin, ymin, xmax, ymax], like ArrayGlyph
        host.add_reference_map("ecmwf", extent=[-100, 15, -40, 55])
        im.set_extent.assert_called_once_with((-100, -40, 15, 55))  # matplotlib order
        assert ax.get_xlim() == (-100, -40)
        assert ax.get_ylim() == (15, 55)
        plt.close(fig)

    def test_no_extent_warns(self):
        """With no extent, a warning flags that coastlines may not align."""
        host, fig, ax = self._host(extent=None)
        with pytest.warns(UserWarning, match="no geographic extent"):
            host.add_reference_map("ecmwf")
        plt.close(fig)

    def test_unknown_style_raises(self):
        """An unknown style name raises `ValueError` listing the options."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        with pytest.raises(ValueError, match="Unknown map style"):
            host.add_reference_map("bogus")
        plt.close(fig)

    @pytest.mark.parametrize("bad", [0, -5, float("nan"), float("inf")])
    def test_invalid_graticule_step_raises(self, bad):
        """A non-positive or non-finite graticule_step raises before drawing (L3)."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        with pytest.raises(ValueError, match="positive, finite"):
            host.add_reference_map("ecmwf", graticule_step=bad)
        host.add_features.assert_not_called()  # failed fast, no layers drawn
        plt.close(fig)

    def test_wrong_length_extent_raises(self):
        """A non-4-element extent raises a clear ValueError naming the order (N1)."""
        host, fig, ax = self._host()
        with pytest.raises(ValueError, match=r"\[xmin, ymin, xmax, ymax\]"):
            host.add_reference_map("ecmwf", extent=[-100, 15, 55])
        plt.close(fig)

    def test_graticule_step_override(self):
        """An explicit `graticule_step` sets the locator base."""
        host, fig, ax = self._host(extent=[-100, 20, -80, 40])
        host.add_reference_map("ecmwf", graticule_step=10)
        # base is 10 -> ticks land on multiples of 10 within the view
        ticks = ax.xaxis.get_major_locator().tick_values(-100, 20)
        assert all(abs(t % 10) < 1e-9 for t in ticks), ticks
        plt.close(fig)

    @pytest.mark.parametrize(
        "span,expected",
        [(0, 1.0), (-5, 1.0), (1.2, 0.2), (4, 1.0), (30, 5.0), (12, 2.0), (1000, 90.0)],
    )
    def test_nice_step(self, span, expected):
        """`_nice_step` returns round steps (incl. sub-degree) and the 90 fallback."""
        assert _nice_step(span) == expected

    def test_resolution_and_zorder_override(self):
        """`resolution` and `zorder` reach both underlying add_features calls."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        host.add_reference_map("ecmwf", resolution="10m", zorder=9)
        for call in host.add_features.call_args_list:
            assert call.args[1] == "10m"
            assert call.kwargs["zorder"] == 9
        plt.close(fig)

    def test_ax_parameter_decorates_given_axes(self):
        """An explicit `ax=` is decorated instead of `self.ax`."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        fig2, other = plt.subplots()
        host.add_reference_map("ecmwf", ax=other)
        assert host.add_features.call_args_list[0].kwargs["ax"] is other
        assert other.spines["bottom"].get_edgecolor() == (0.6, 0.6, 0.6, 1.0)
        plt.close(fig)
        plt.close(fig2)


def test_add_reference_map_integration(tmp_path: Path, monkeypatch):
    """Non-mocked: add_reference_map draws real coastline + border collections."""
    monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path))
    line = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-90, 20], [-50, 50]],
                },
            }
        ],
    }
    for fname in (
        "ne_110m_coastline.geojson.gz",
        "ne_110m_admin_0_boundary_lines_land.geojson.gz",
    ):
        with gzip.open(tmp_path / fname, "wt", encoding="utf-8") as fh:
            json.dump(line, fh)

    glyph = ArrayGlyph(np.random.rand(20, 30), extent=[-100, 15, -40, 55])
    fig, ax = glyph.plot()
    xlim0, ylim0 = ax.get_xlim(), ax.get_ylim()
    glyph.add_reference_map("ecmwf", resolution="110m")

    lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(lcs) >= 2, "coastline + borders should both draw"
    assert all(c.get_zorder() == 5 for c in lcs)
    assert ax.xaxis.get_major_formatter()(-75) == "75°W"
    # the reference layers must not perturb the data extent
    assert ax.get_xlim() == xlim0
    assert ax.get_ylim() == ylim0
    # the preset styling reaches the real axes (frame + visible graticule)
    assert ax.spines["bottom"].get_edgecolor() == (0.6, 0.6, 0.6, 1.0)
    assert ax.xaxis.get_gridlines()[0].get_visible(), "graticule not drawn"
    plt.close(fig)


def test_add_reference_map_dark_draws_real_relief(tmp_path: Path, monkeypatch):
    """Non-mocked: `ecmwf-dark` places a real dimmed relief image beneath data."""
    Image = pytest.importorskip(
        "PIL.Image", reason="Pillow not installed (tiles extra)"
    )
    monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path))
    arr = (np.random.default_rng(0).random((4, 8, 3)) * 255).astype("uint8")
    Image.fromarray(arr).save(tmp_path / "ne_hypso_rgb_720x360.png")
    line = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-90, 20], [-50, 50]],
                },
            }
        ],
    }
    for fname in (
        "ne_110m_coastline.geojson.gz",
        "ne_110m_admin_0_boundary_lines_land.geojson.gz",
    ):
        with gzip.open(tmp_path / fname, "wt", encoding="utf-8") as fh:
            json.dump(line, fh)

    glyph = ArrayGlyph(np.random.rand(20, 30), extent=[-100, 15, -40, 55])
    fig, ax = glyph.plot()
    n_before = len(ax.images)
    glyph.add_reference_map("ecmwf-dark", resolution="110m")
    assert len(ax.images) == n_before + 1, "ecmwf-dark should add a relief image"
    relief_img = ax.images[-1]
    assert relief_img.get_zorder() == -2
    assert relief_img.get_alpha() == 0.5
    plt.close(fig)


def test_add_reference_map_relief_warps_non_4326(tmp_path: Path, monkeypatch):
    """ecmwf-dark forwards self.crs to the relief, so on a non-4326 axis the
    backdrop is warped (RGBA) rather than placed in lon/lat."""
    pytest.importorskip("pyproj", reason="pyproj not installed (tiles extra)")
    Image = pytest.importorskip(
        "PIL.Image", reason="Pillow not installed (tiles extra)"
    )
    monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path))
    arr = (np.random.default_rng(0).random((4, 8, 3)) * 255).astype("uint8")
    Image.fromarray(arr).save(tmp_path / "ne_hypso_rgb_720x360.png")
    line = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-90, 20], [-50, 50]],
                },
            }
        ],
    }
    for fname in (
        "ne_110m_coastline.geojson.gz",
        "ne_110m_admin_0_boundary_lines_land.geojson.gz",
    ):
        with gzip.open(tmp_path / fname, "wt", encoding="utf-8") as fh:
            json.dump(line, fh)

    glyph = ArrayGlyph(np.random.rand(20, 30), extent=[-2e7, -1e7, 2e7, 1e7])
    glyph.crs = 3857  # axis CRS recorded once; add_reference_map defaults to it
    fig, ax = glyph.plot()
    glyph.add_reference_map("ecmwf-dark", resolution="110m")
    placed = np.asarray(ax.images[-1].get_array())
    assert placed.shape[2] == 4, f"relief should warp to RGBA, got {placed.shape}"
    plt.close(fig)


class TestBasemapAlignmentCheck:
    """`_check_basemap_alignment`: the opt-in mis-georeferencing warning."""

    @staticmethod
    def _relief(ref_land: np.ndarray) -> np.ndarray:
        """RGB where ocean is blue-dominant and land is green."""
        rgb = np.zeros((*ref_land.shape, 3), dtype=np.uint8)
        rgb[..., 2] = 255  # blue everywhere == ocean by default
        rgb[ref_land] = (0, 255, 0)  # land: green (not blue-dominant)
        return rgb

    def test_no_extent_skips_before_fetching_relief(self, monkeypatch):
        fetched = []
        monkeypatch.setattr(
            refmod,
            "relief",
            lambda res="low": fetched.append(1) or np.zeros((4, 4, 3), np.uint8),
        )
        glyph = ArrayGlyph(np.arange(3 * 10 * 10, dtype=float).reshape(3, 10, 10))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            glyph._check_basemap_alignment()
        assert caught == []
        assert fetched == []

    def test_no_land_sea_boundary_skips(self, monkeypatch):
        monkeypatch.setattr(
            refmod, "relief", lambda res="low": np.zeros((4, 4, 3), np.uint8)
        )
        glyph = ArrayGlyph(np.ones((3, 10, 10)), extent=[0, 0, 10, 10])  # all finite
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            glyph._check_basemap_alignment()
        assert caught == []

    def test_relief_unavailable_never_fails(self, monkeypatch):
        def offline(res="low"):
            raise ConnectionError("no network")

        monkeypatch.setattr(refmod, "relief", offline)
        arr = np.ones((3, 10, 20))
        arr[:, :, 10:] = np.nan  # half land, half sea
        glyph = ArrayGlyph(arr, extent=[-20, -10, 20, 10])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            glyph._check_basemap_alignment()
        assert caught == []

    def test_opt_in_wiring(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            ArrayGlyph, "_check_basemap_alignment", lambda self, *a, **k: seen.append(1)
        )
        glyph = ArrayGlyph(np.ones((3, 10, 10)), extent=[0, 0, 10, 10])
        glyph._draw_basemap({"relief": False, "features": [], "check_alignment": True})
        assert seen == [1]
        glyph._draw_basemap({"relief": False, "features": []})  # opt-in absent
        assert seen == [1]  # not called again

    def test_warns_on_misregistration_but_not_when_aligned(self, monkeypatch):
        rows, cols = np.mgrid[0:180, 0:360]
        ref_land = (rows + cols) % 2 == 0  # global 1-cell checkerboard
        monkeypatch.setattr(refmod, "relief", lambda res="low": self._relief(ref_land))

        def sample(extent, n=20):
            xmin, ymin, xmax, ymax = extent
            lons = np.linspace(xmin, xmax, n)
            lats = np.linspace(ymax, ymin, n)  # origin="upper"
            ci = np.clip(((lons + 180) / 360 * 360).astype(int), 0, 359)
            ri = np.clip(((90 - lats) / 180 * 180).astype(int), 0, 179)
            return ref_land[np.ix_(ri, ci)]

        e0 = [0.0, 0.0, 20.0, 20.0]
        land = sample(e0)
        data = np.broadcast_to(np.where(land, 1.0, np.nan), (3, 20, 20)).copy()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ArrayGlyph(data, extent=e0)._check_basemap_alignment()
        assert not any("alignment" in str(x.message) for x in caught)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ArrayGlyph(data, extent=[1.0, 0.0, 21.0, 20.0])._check_basemap_alignment()
        assert any("alignment" in str(x.message) for x in caught)


class TestFeature:
    """`Feature`: a typed Natural Earth layer for a `Basemap`."""

    def test_layer_and_style_stored(self):
        """The layer name and style keywords are captured verbatim."""
        f = Feature("coastline", colors="0.55", linewidths=0.5)
        assert f.layer == "coastline", f"Got layer {f.layer!r}"
        assert f.style == {"colors": "0.55", "linewidths": 0.5}, f"Got style {f.style!r}"

    def test_no_style_is_empty_dict(self):
        """With no style keywords the style is an empty dict."""
        assert Feature("borders").style == {}, "Expected empty style"

    @pytest.mark.parametrize(
        "layer", ["coastline", "land", "ocean", "rivers", "lakes", "borders"]
    )
    def test_all_known_layers_accepted(self, layer):
        """Every layer `reference.available_layers()` reports is accepted."""
        assert Feature(layer).layer == layer, f"{layer!r} should be accepted"

    def test_unknown_layer_raises(self):
        """An unknown layer is rejected at construction with a helpful message."""
        with pytest.raises(ValueError, match=r"Unknown basemap feature layer 'countries'"):
            Feature("countries")


class TestBasemap:
    """`Basemap`: the typed, validated form of the `basemap=` dict."""

    def test_defaults(self):
        """Defaults mirror `basemap=True`: relief on, default features, 50m."""
        bm = Basemap()
        assert bm.relief is True, f"Got relief {bm.relief!r}"
        assert bm.features is None, f"Got features {bm.features!r}"
        assert bm.resolution == "50m", f"Got resolution {bm.resolution!r}"
        assert bm.check_alignment is False, f"Got check_alignment {bm.check_alignment!r}"

    def test_explicit_values_stored(self):
        """All keywords are stored; `features` is materialised into a list copy."""
        feats = (Feature("coastline"), Feature("borders"))
        bm = Basemap(relief="medium", features=feats, resolution="10m", check_alignment=True)
        assert bm.relief == "medium", f"Got relief {bm.relief!r}"
        assert bm.features == list(feats), "features should be a list copy of the iterable"
        assert bm.resolution == "10m", f"Got resolution {bm.resolution!r}"
        assert bm.check_alignment is True, f"Got check_alignment {bm.check_alignment!r}"

    def test_unknown_relief_resolution_raises(self):
        """A string `relief` that is not a known resolution is rejected."""
        with pytest.raises(ValueError, match=r"Unknown relief resolution 'ultra'"):
            Basemap(relief="ultra")

    def test_bool_and_dict_relief_pass_through(self):
        """`relief` as a bool or dict is stored without a resolution check."""
        assert Basemap(relief=False).relief is False, "False relief must pass through"
        cfg = {"resolution": "low", "alpha": 0.3}
        assert Basemap(relief=cfg).relief == cfg, "dict relief must pass through"

    def test_as_config_omits_features_when_unset(self):
        """`_as_config` drops `features` when None so the default still applies."""
        cfg = Basemap()._as_config()
        assert "features" not in cfg, f"features should be omitted, got {cfg}"
        assert cfg == {"relief": True, "resolution": "50m", "check_alignment": False}, cfg

    def test_as_config_includes_features_when_set(self):
        """`_as_config` includes the features list when provided."""
        feats = [Feature("ocean")]
        assert Basemap(relief=False, features=feats)._as_config()["features"] == feats

    def test_reexported_from_array_glyph(self):
        """`Basemap`/`Feature` are the same objects re-exported from array_glyph."""
        from cleopatra.glyphs.gridded.array_glyph import Basemap as B2
        from cleopatra.glyphs.gridded.array_glyph import Feature as F2

        assert B2 is Basemap, 're-exports must be the same classes'
        assert F2 is Feature, 're-exports must be the same classes'


class TestDrawBasemapRouting:
    """`_draw_basemap` treats a `Basemap`/`Feature` like the equivalent dict."""

    @staticmethod
    def _record(monkeypatch):
        """Patch add_relief/add_features to record calls; return the two logs."""
        relief_calls: list = []
        feature_calls: list = []
        monkeypatch.setattr(
            ArrayGlyph, "add_relief", lambda self, *a, **k: relief_calls.append((a, k))
        )
        monkeypatch.setattr(
            ArrayGlyph, "add_features", lambda self, *a, **k: feature_calls.append((a, k))
        )
        return relief_calls, feature_calls

    def test_basemap_matches_equivalent_dict(self, monkeypatch):
        """A `Basemap` yields the same add_relief/add_features calls as its dict."""
        glyph = ArrayGlyph(np.ones((3, 5, 5)), extent=[0, 0, 5, 5])
        r1, f1 = self._record(monkeypatch)
        glyph._draw_basemap(
            Basemap(
                relief=False,
                features=[
                    Feature("coastline", colors="0.55"),
                    Feature("borders", colors="0.45"),
                ],
            )
        )
        r2, f2 = self._record(monkeypatch)
        glyph._draw_basemap(
            {
                "relief": False,
                "features": [
                    ("coastline", {"colors": "0.55"}),
                    ("borders", {"colors": "0.45"}),
                ],
            }
        )
        assert (r1, f1) == (r2, f2), "Basemap should route identically to the dict form"

    def test_feature_zorder_overrides_default(self, monkeypatch):
        """A `Feature`'s own `zorder` overrides the default feature zorder (3)."""
        glyph = ArrayGlyph(np.ones((3, 5, 5)), extent=[0, 0, 5, 5])
        _, feats = self._record(monkeypatch)
        glyph._draw_basemap(Basemap(relief=False, features=[Feature("ocean", zorder=-2)]))
        (args, kwargs), = feats
        assert args[0] == "ocean", f"Got layer {args[0]!r}"
        assert kwargs["zorder"] == -2, f"Feature zorder should win, got {kwargs.get('zorder')}"
