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

import cleopatra.reference as refmod  # noqa: E402
import cleopatra.tiles as tilesmod  # noqa: E402
from cleopatra.array_glyph import ArrayGlyph  # noqa: E402
from cleopatra.flow_glyph import FlowGlyph  # noqa: E402
from cleopatra.geo import (  # noqa: E402
    GeoMixin,
    _lat_formatter,
    _lon_formatter,
    _nice_step,
    available_map_styles,
)
from cleopatra.kde_glyph import KDEGlyph  # noqa: E402
from cleopatra.line_glyph import LineGlyph  # noqa: E402
from cleopatra.mesh_glyph import MeshGlyph  # noqa: E402
from cleopatra.polygon_glyph import PolygonGlyph  # noqa: E402
from cleopatra.scatter_glyph import ScatterGlyph  # noqa: E402
from cleopatra.statistical_glyph import StatisticalGlyph  # noqa: E402
from cleopatra.vector_glyph import VectorGlyph  # noqa: E402

GEO_GLYPHS = [ArrayGlyph, MeshGlyph, VectorGlyph, FlowGlyph, PolygonGlyph, ScatterGlyph]
NON_GEO_GLYPHS = [LineGlyph, StatisticalGlyph, KDEGlyph]
METHODS = ("add_tiles", "add_features", "add_relief", "add_reference_map")


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


class TestAddReferenceMap:
    """`GeoMixin.add_reference_map` reference-map style preset (issue #184)."""

    @staticmethod
    def _host(extent=None, im=None):
        """A GeoMixin host with a real axes and a mocked `add_features`."""
        fig, ax = plt.subplots()
        host = _Dummy(ax=ax)
        host.extent = extent
        host.im = im
        host.crs = None
        host.add_features = MagicMock(return_value=ax)
        return host, fig, ax

    def test_available_map_styles(self):
        """The built-in preset names are exposed and stable."""
        assert available_map_styles() == ["ecmwf", "ecmwf-dark"]

    @pytest.mark.parametrize(
        "value,expected",
        [(-75, "75°W"), (10, "10°E"), (0, "0°"), (180, "180°"), (-180, "180°"), (200, "160°W")],
    )
    def test_lon_formatter(self, value, expected):
        """Longitude ticks label W/E, 0, and the ±180° antimeridian (L1)."""
        assert _lon_formatter(value) == expected

    @pytest.mark.parametrize(
        "value,expected", [(-20, "20°S"), (45, "45°N"), (0, "0°")]
    )
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

    @pytest.mark.parametrize("bad", [0, -5])
    def test_nonpositive_graticule_step_raises(self, bad):
        """A zero/negative graticule_step raises before anything is drawn (L3)."""
        host, fig, ax = self._host(extent=[-100, 15, -40, 55])
        with pytest.raises(ValueError, match="graticule_step must be a positive"):
            host.add_reference_map("ecmwf", graticule_step=bad)
        host.add_features.assert_not_called()  # failed fast, no layers drawn
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
            {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[-90, 20], [-50, 50]]}}
        ],
    }
    for fname in ("ne_110m_coastline.geojson.gz", "ne_110m_admin_0_boundary_lines_land.geojson.gz"):
        with gzip.open(tmp_path / fname, "wt", encoding="utf-8") as fh:
            json.dump(line, fh)

    glyph = ArrayGlyph(np.random.rand(20, 30), extent=[-100, 15, -40, 55])
    fig, ax = glyph.plot()
    glyph.add_reference_map("ecmwf", resolution="110m")

    lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(lcs) >= 2, "coastline + borders should both draw"
    assert all(c.get_zorder() == 5 for c in lcs)
    assert ax.xaxis.get_major_formatter()(-75) == "75°W"
    plt.close(fig)
