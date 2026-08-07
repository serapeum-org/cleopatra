"""Geographic basemap convenience methods for glyphs.

`GeoMixin` adds three convenience methods -- `add_tiles`, `add_features`,
and `add_relief` -- to the glyph classes that plot geographic data, so a
basemap can be dropped under a plot without importing the standalone
helpers and without repeating the axes:

    >>> glyph.plot()                 # doctest: +SKIP
    >>> glyph.add_relief("low")      # doctest: +SKIP
    >>> glyph.add_features("coastline", "50m")  # doctest: +SKIP

Each method is a thin wrapper that draws on the glyph's own axes
(`self.ax`) and delegates to the single implementation in
`cleopatra.basemap.tiles` / `cleopatra.basemap.reference`. The standalone functions remain
the source of truth; this mixin only removes the import + explicit-axes
boilerplate for the geographic glyphs (`ArrayGlyph`, `MeshGlyph`,
`VectorGlyph`, `FlowGlyph`, `PolygonGlyph`, `ScatterGlyph`). Non-geographic
glyphs (line/bar charts, statistical plots) deliberately do not inherit
it.

Importing this module (and the `cleopatra.basemap.tiles` / `cleopatra.basemap.reference`
modules it calls) does not require the optional `cleopatra[tiles]` extra:
those modules gate their `[tiles]` dependencies (`pyproj`, `Pillow`, ...)
behind their own internal lazy imports, so the extra is only needed when
a basemap is actually drawn.
"""

from __future__ import annotations

import importlib.util
import math
import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

from cleopatra.basemap import reference, tiles

#: Built-in reference-map style presets for `GeoMixin.add_reference_map`.
#: `"ecmwf"` is tuned for light backgrounds; `"ecmwf-dark"` uses lighter
#: greys so coastlines stay visible over a dark field (e.g. a satellite
#: true-colour RGB) and adds a dimmed hypsometric relief backdrop under the
#: data. Each entry is a plain dict of the layer styles, graticule,
#: tick-label, and frame (spine) parameters, plus an optional `"relief"`
#: backdrop (a dict of `add_relief` kwargs, a resolution string, or `True`)
#: -- read or copy it to build a custom preset; `add_reference_map` itself
#: exposes only the `resolution` and `graticule_step` knobs per call.
REFERENCE_MAP_STYLES: dict[str, dict[str, Any]] = {
    "ecmwf": {
        "resolution": "50m",
        "coastline": {"colors": "0.45", "linewidths": 0.8},
        "borders": {"colors": "0.55", "linewidths": 0.5},
        "graticule": {"color": "0.7", "linestyle": (0, (4, 4)), "linewidth": 0.5},
        "labels": {"colors": "0.35", "labelsize": 8},
        "spines": {"edgecolor": "0.6", "linewidth": 0.8},
    },
    "ecmwf-dark": {
        "resolution": "50m",
        "relief": {"resolution": "low", "alpha": 0.5, "zorder": -2},
        "coastline": {"colors": "0.85", "linewidths": 0.8},
        "borders": {"colors": "0.85", "linewidths": 0.5},
        "graticule": {"color": "0.75", "linestyle": (0, (4, 4)), "linewidth": 0.5},
        "labels": {"colors": "0.8", "labelsize": 8},
        "spines": {"edgecolor": "0.7", "linewidth": 0.8},
    },
}


def available_map_styles() -> list[str]:
    """Return the built-in `add_reference_map` style names.

    Returns:
        list[str]: The preset names accepted by
        `GeoMixin.add_reference_map` (excluding the special `"auto"`).

    Examples:
        ```python
        >>> from cleopatra.basemap.geo import available_map_styles
        >>> available_map_styles()
        ['ecmwf', 'ecmwf-dark']

        ```
    """
    return list(REFERENCE_MAP_STYLES)


def _nice_step(span: float, target_divisions: int = 6) -> float:
    """Pick a human-friendly graticule step for `span` over ~N divisions.

    Args:
        span: The axis span in degrees (`max(width, height)`).
        target_divisions: Rough number of gridlines to aim for.

    Returns:
        float: A "nice" step (..., 0.25, 0.5, 1, 2, 2.5, 5, 10, ...) so the
        graticule lands on round degree values, including sub-degree steps
        for zoomed-in (city/basin-scale) maps.

    Examples:
        ```python
        >>> from cleopatra.basemap.geo import _nice_step
        >>> _nice_step(30)
        5.0
        >>> _nice_step(4)
        1.0
        >>> _nice_step(1.2)
        0.2

        ```
    """
    if span <= 0:
        return 1.0
    raw = span / max(target_divisions, 1)
    for candidate in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 15, 20, 30, 45, 60):
        if raw <= candidate:
            return float(candidate)
    return 90.0


def _lon_formatter(value: float, _pos: Any = None) -> str:
    """Format a longitude tick as `°W`/`°E` (0 at the meridian).

    Examples:
        ```python
        >>> from cleopatra.basemap.geo import _lon_formatter
        >>> _lon_formatter(-75), _lon_formatter(10), _lon_formatter(0)
        ('75°W', '10°E', '0°')
        >>> _lon_formatter(180), _lon_formatter(-180)
        ('180°', '180°')

        ```
    """
    lon = ((value + 180) % 360) - 180
    if abs(lon) == 180:  # the antimeridian is neither W nor E
        return "180°"
    if lon < 0:
        return f"{abs(lon):g}°W"
    if lon > 0:
        return f"{lon:g}°E"
    return "0°"


def _lat_formatter(value: float, _pos: Any = None) -> str:
    """Format a latitude tick as `°S`/`°N` (0 at the equator).

    Examples:
        ```python
        >>> from cleopatra.basemap.geo import _lat_formatter
        >>> _lat_formatter(-20), _lat_formatter(45), _lat_formatter(0)
        ('20°S', '45°N', '0°')

        ```
    """
    if value < 0:
        return f"{abs(value):g}°S"
    if value > 0:
        return f"{value:g}°N"
    return "0°"


def _validate_crs(crs: int | str | None) -> int | str | None:
    """Validate (and lightly normalise) a value assigned to `GeoMixin.crs`.

    Cheap type/shape checks always run (and need no third-party package);
    full CRS-resolvability is additionally checked with `pyproj` **only when
    it is installed**, so setting `crs` never requires the optional
    `cleopatra[tiles]` extra. `None` is always accepted. When `pyproj` is
    absent, an unresolvable-but-well-typed CRS is still caught later, at
    draw time, by `add_features` / `add_tiles`.

    Strings are stripped, and a bare numeric EPSG string (e.g. `"4326"`) is
    normalised to the int `4326` so it is treated identically to the int
    form and to the draw path across `pyproj` versions (some reject the
    digits-only string).

    Args:
        crs: An int EPSG code, a CRS string, or `None`.

    Returns:
        The validated `crs` -- whitespace-stripped, and with a bare numeric
        string converted to an int EPSG code.

    Raises:
        TypeError: If `crs` is not an int, str, or `None` (`bool` is
            rejected).
        ValueError: If `crs` is a non-positive EPSG code, an empty string,
            or (when `pyproj` is installed) an unresolvable CRS.

    Examples:
        - `None` and well-formed values pass through; a bare numeric string
            is normalised to an int:
            ```python
            >>> from cleopatra.basemap.geo import _validate_crs
            >>> _validate_crs(None) is None
            True
            >>> _validate_crs(4326)
            4326
            >>> _validate_crs("4326")
            4326

            ```
        - Wrong types are rejected immediately:
            ```python
            >>> from cleopatra.basemap.geo import _validate_crs
            >>> _validate_crs([4326])
            Traceback (most recent call last):
                ...
            TypeError: crs must be an int EPSG code, a CRS string, or None, got list

            ```
    """
    if crs is None:
        return None
    if isinstance(crs, bool) or not isinstance(crs, (int, str)):
        raise TypeError(
            "crs must be an int EPSG code, a CRS string, or None, got "
            f"{type(crs).__name__}"
        )
    if isinstance(crs, str):
        crs = crs.strip()
        if not crs:
            raise ValueError("crs string must be a non-empty CRS identifier")
        if crs.isdigit():
            crs = int(crs)  # bare EPSG code as a string -> int
    if isinstance(crs, int) and crs <= 0:
        raise ValueError(f"crs EPSG code must be a positive int, got {crs}")
    if importlib.util.find_spec("pyproj") is not None:
        from pyproj import CRS
        from pyproj.exceptions import CRSError

        try:
            CRS.from_user_input(crs)
        except CRSError as e:
            raise ValueError(f"Invalid CRS {crs!r}: {e}") from e
    return crs


def add_point_labels(
    ax: Any,
    points: dict[str, tuple[float, float]],
    *,
    color: str = "white",
    marker_size: float = 5.0,
    fontsize: float = 9.0,
    offset: tuple[float, float] = (4.0, 0.0),
    zorder: int = 6,
) -> Any:
    """Annotate named points with a plain dot marker + text label.

    Draws a small circular marker at each point and a plain text label
    beside it -- no halo, no bounding box -- matching the minimalist look
    ECMWF/CAMS maps use for city labels. `points` are plotted at whatever
    coordinates `ax` is already using (plain lon/lat on a flat axes, or
    projected x/y on an orthographic globe -- reproject the points yourself,
    e.g. with the same transformer `cleopatra.basemap.projection.orthographic_grid`
    builds, before calling this on a globe view), so this composes with any
    projection or colour styling; it makes no assumption about either.

    Args:
        ax: Axes to draw on.
        points: Mapping of label text to `(x, y)` coordinates.
        color: Colour for both the marker and the label text.
        marker_size: Marker size in points.
        fontsize: Label font size in points.
        offset: `(dx, dy)` label offset from the marker, in points (applied
            via `textcoords="offset points"`), so it scales with font size
            rather than the data coordinates.
        zorder: Draw order for both the marker and the label.

    Returns:
        Axes: The same `ax`, for chaining.

    Examples:
        - Label two points and read back the drawn markers/labels:
            ```python
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.basemap.geo import add_point_labels
            >>> fig, ax = plt.subplots()
            >>> _ = add_point_labels(ax, {"London": (-0.1, 51.5), "Moscow": (37.6, 55.8)})
            >>> len(ax.lines)  # one marker per point
            2
            >>> [t.get_text() for t in ax.texts]
            ['London', 'Moscow']
            >>> plt.close(fig)

            ```
        - An empty mapping draws nothing but still returns `ax`, for chaining:
            ```python
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.basemap.geo import add_point_labels
            >>> fig, ax = plt.subplots()
            >>> add_point_labels(ax, {}) is ax
            True
            >>> plt.close(fig)

            ```

    See Also:
        GeoMixin.add_labels: The glyph convenience wrapper for this function.
    """
    for label, (x, y) in points.items():
        ax.plot(
            x,
            y,
            marker="o",
            markersize=marker_size,
            color=color,
            linestyle="none",
            zorder=zorder,
        )
        ax.annotate(
            label,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            color=color,
            fontsize=fontsize,
            zorder=zorder,
        )
    return ax


class Feature:
    """A Natural Earth reference layer for a `Basemap`.

    Pairs a layer name with its matplotlib style keywords, so a `Basemap`'s
    `features` list reads as typed values instead of raw `(name, dict)`
    tuples. The style keywords are forwarded verbatim to
    `GeoMixin.add_features` (e.g. `colors`, `linewidths`, `facecolor`,
    `alpha`, `zorder`).

    Attributes:
        layer: The Natural Earth layer to draw -- one of
            `cleopatra.basemap.reference.available_layers()` (`"coastline"`,
            `"borders"`, `"land"`, `"ocean"`, `"rivers"`, `"lakes"`).
        style: The style keywords forwarded to `add_features`.

    Examples:
        - A thin grey coastline:
            ```python
            >>> from cleopatra.basemap.geo import Feature
            >>> f = Feature("coastline", colors="0.55", linewidths=0.5)
            >>> f.layer, f.style
            ('coastline', {'colors': '0.55', 'linewidths': 0.5})

            ```
        - An unknown layer is rejected at construction:
            ```python
            >>> Feature("countries")
            Traceback (most recent call last):
            ValueError: Unknown basemap feature layer 'countries'. Choose from ['coastline', 'land', 'ocean', 'rivers', 'lakes', 'borders'].

            ```
    """

    def __init__(self, layer: str, **style: Any) -> None:
        """Initialise a `Feature`.

        Args:
            layer: Natural Earth layer name (see
                `cleopatra.basemap.reference.available_layers`).
            **style: Style keywords forwarded to `add_features`
                (`colors`, `linewidths`, `facecolor`, `alpha`, `zorder`, ...).

        Raises:
            ValueError: If `layer` is not a known Natural Earth layer.
        """
        valid = reference.available_layers()
        if layer not in valid:
            raise ValueError(
                f"Unknown basemap feature layer {layer!r}. Choose from {valid}."
            )
        self.layer = layer
        self.style = style


class Basemap:
    """Structured spec for `ArrayGlyph.plot` / `animate`'s `basemap=`.

    Bundles the reference-backdrop choices -- the hypsometric relief drawn
    under the data, the Natural Earth feature layers drawn over it, and the
    opt-in alignment check -- into one value, mirroring `ColorBar` /
    `FrameLabel`. `plot`/`animate` still accept `basemap=True` (the default
    backdrop), a plain `dict`, or a `callable f(glyph)`; a `Basemap` is the
    typed, validated form, and `_draw_basemap` consumes any of them.

    Attributes:
        relief: The hypsometric relief drawn under the data (`zorder=-2`).
            `True` (default) draws the default low-resolution relief;
            `False` skips it; a resolution string (`"low"` / `"medium"`, see
            `cleopatra.basemap.reference.available_relief_resolutions`) picks the
            product; a dict of `add_relief` keyword arguments overrides
            resolution / alpha / zorder in full.
        features: The Natural Earth layers drawn over the relief
            (`zorder=3`). `None` (default) draws grey `coastline` +
            `borders`. Otherwise an iterable of `Feature` (preferred), bare
            layer-name strings, or `(layer, style_dict)` tuples.
        resolution: Natural Earth resolution for the features (`"10m"` /
            `"50m"` / `"110m"`), by default `"50m"`.
        check_alignment: Run the opt-in mis-georeferencing check
            (`_check_basemap_alignment`), by default `False`.

    Examples:
        - The default backdrop, spelled out:
            ```python
            >>> from cleopatra.basemap.geo import Basemap
            >>> bm = Basemap()
            >>> bm.relief, bm.features, bm.resolution
            (True, None, '50m')

            ```
        - A "dark ocean" basemap -- no relief, a thin coastline + borders:
            ```python
            >>> from cleopatra.basemap.geo import Basemap, Feature
            >>> bm = Basemap(relief=False,
            ...              features=[Feature("coastline", colors="0.55"),
            ...                        Feature("borders", colors="0.45")])
            >>> bm.relief, [f.layer for f in bm.features]
            (False, ['coastline', 'borders'])

            ```
    """

    def __init__(
        self,
        *,
        relief: bool | str | dict = True,
        features: Iterable[Feature | str | tuple] | None = None,
        resolution: str = "50m",
        check_alignment: bool = False,
    ) -> None:
        """Initialise a `Basemap`.

        Args:
            relief: Relief backdrop -- `True` / `False`, a resolution string
                (`"low"` / `"medium"`), or a dict of `add_relief` kwargs.
            features: Layers over the relief -- an iterable of `Feature`,
                layer-name strings, or `(layer, style_dict)` tuples; `None`
                keeps the default `coastline` + `borders`.
            resolution: Natural Earth resolution for the features, by default
                `"50m"`.
            check_alignment: Opt-in mis-georeferencing check, by default
                `False`.

        Raises:
            ValueError: If `relief` is a string that is not a known relief
                resolution.
        """
        if isinstance(relief, str) and (
            relief not in reference.available_relief_resolutions()
        ):
            raise ValueError(
                f"Unknown relief resolution {relief!r}. "
                f"Choose from {reference.available_relief_resolutions()}."
            )
        self.relief = relief
        self.features = list(features) if features is not None else None
        self.resolution = resolution
        self.check_alignment = check_alignment

    def _as_config(self) -> dict:
        """Normalise to the dict form `_draw_basemap` consumes.

        Omits `features` when unset so `_draw_basemap`'s default
        (`coastline` + `borders`) still applies.

        Returns:
            dict: The `{relief, resolution, check_alignment[, features]}`
                config `_draw_basemap` understands.
        """
        cfg: dict = {
            "relief": self.relief,
            "resolution": self.resolution,
            "check_alignment": self.check_alignment,
        }
        if self.features is not None:
            cfg["features"] = self.features
        return cfg


class GeoMixin:
    """Mixin giving geographic glyphs `add_tiles` / `add_features` / `add_relief`.

    The host class is expected to expose the plotted axes as `self.ax`
    (every `cleopatra.glyphs.base.glyph.Glyph` subclass does). Call these after
    plotting, or pass `ax=` explicitly.

    Set `self.crs` to the CRS of the data plotted on the axes (an EPSG code
    or CRS string) and `add_features` / `add_tiles` / `add_relief` default
    their `crs=` argument to it, so the reference layer is placed in matching
    coordinates without restating it on every call. An explicit `crs=` still
    wins; leaving `self.crs` as `None` preserves each helper's own default.
    For `add_relief`, a non-EPSG:4326 `crs` warps the relief raster into the
    axis CRS (a 4326 axis places it in lon/lat, unchanged).
    """

    #: Set by `cleopatra.glyphs.base.glyph.Glyph`; the axes the basemap is drawn on.
    ax: Any

    #: Backing store for the validated `crs` property; `None` means unset.
    _crs: int | str | None = None

    @property
    def crs(self) -> int | str | None:
        """CRS of the data plotted on `self.ax` (EPSG code or CRS string).

        When set, `add_features` / `add_tiles` / `add_relief` default `crs=`
        to it; `None` keeps each helper's own default. The value is **validated on
        assignment** (see `cleopatra.basemap.geo._validate_crs`) so mistakes surface
        at `glyph.crs = ...` rather than later, when a basemap is drawn.

        Raises:
            TypeError: If assigned something other than an int, str, or
                `None`.
            ValueError: If assigned a non-positive EPSG code, an empty
                string, or (when `pyproj` is installed) an unresolvable CRS.
        """
        return self._crs

    @crs.setter
    def crs(self, value: int | str | None) -> None:
        self._crs = _validate_crs(value)

    def _basemap_axes(self, ax: Any = None) -> Any:
        """Return the axes to draw a basemap on, creating one if needed.

        Resolves the target as `ax`, else the glyph's own `self.ax`. When the
        glyph has not drawn yet (`self.ax` is `None`) but can make its own figure
        (`create_figure_axes` -- i.e. a real glyph), the axes is created eagerly
        and stored on the glyph, so a reference layer can be added **before**
        `plot`/`animate`. Those then reuse the same axes (both guard on
        `self.fig is None`), so the builder flow

            glyph = ArrayGlyph(data, ...)
            glyph.add_features("coastline")   # creates the axes here
            glyph.animate(...)                # reuses it

        works directly, with no basemap callback.

        Args:
            ax: An explicit axes to use instead of the glyph's own.

        Returns:
            The resolved matplotlib axes.

        Raises:
            RuntimeError: If there is no axes and the host cannot create *and*
                seed one on demand. Only a glyph that exposes its own data bounds
                (`_flat_axis_bounds`, i.e. `ArrayGlyph`) supports the pre-plot
                builder flow; for the others, plot the glyph first (or pass `ax=`).
        """
        target = ax if ax is not None else getattr(self, "ax", None)
        if target is not None:
            return target
        if hasattr(self, "create_figure_axes") and hasattr(self, "_flat_axis_bounds"):
            self.fig, self.ax = self.create_figure_axes()
            x_min, x_max, y_min, y_max = self._flat_axis_bounds()
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            return self.ax
        raise RuntimeError(
            "No axes to draw on. Plot the glyph first (or pass ax=) "
            "before adding a basemap layer."
        )

    def _basemap_kwargs(self, kwargs: dict) -> dict:
        """Default `crs` to `self.crs` when the caller did not set it.

        Only injects when `self.crs` is set and `crs` is absent (or `None`)
        in `kwargs`, so the default `self.crs is None` is a pure pass-through
        and an explicit `crs=` always wins. `crs` is keyword-only in
        `add_features`, `add_tiles`, and `add_relief`, so it always arrives
        via `kwargs`.

        Args:
            kwargs: The keyword arguments destined for the basemap helper.

        Returns:
            dict: `kwargs`, with `crs` filled in from `self.crs` when needed.
        """
        if self.crs is not None and kwargs.get("crs") is None:
            return {**kwargs, "crs": self.crs}
        return kwargs

    def add_tiles(self, *args: Any, ax: Any = None, **kwargs: Any) -> Any:
        """Overlay a web-tile basemap on the glyph's axes.

        Thin wrapper over `cleopatra.basemap.tiles.add_tiles`; positional and
        keyword arguments are forwarded unchanged (e.g. `source`, `crs`,
        `zoom`, `alpha`). When `crs` is omitted it defaults to `self.crs`.
        Requires the `cleopatra[tiles]` extra.

        Args:
            *args: Positional arguments for `cleopatra.basemap.tiles.add_tiles`
                (after the axes).
            ax: Axes to draw on. Defaults to the glyph's `self.ax`.
            **kwargs: Keyword arguments for `cleopatra.basemap.tiles.add_tiles`. A
                `crs` keyword is defaulted to `self.crs` when omitted; an
                explicit `crs=` overrides it.

        Returns:
            matplotlib.axes.Axes: The axes, for chaining.

        Raises:
            RuntimeError: Only if there is no axes, no `ax` is given, and the
                glyph cannot create one (a bare `GeoMixin`); only `ArrayGlyph` creates and seeds
                its axes on demand, so the pre-plot builder flow is ArrayGlyph-only
                -- plot the other glyphs first (or pass `ax=`).

        See Also:
            cleopatra.basemap.tiles.add_tiles: The underlying implementation and its
                full parameter list.
        """
        return tiles.add_tiles(
            self._basemap_axes(ax), *args, **self._basemap_kwargs(kwargs)
        )

    def add_features(self, *args: Any, ax: Any = None, **kwargs: Any) -> Any:
        """Draw a Natural Earth reference layer on the glyph's axes.

        Thin wrapper over `cleopatra.basemap.reference.add_features`; arguments are
        forwarded unchanged (e.g. `layer`, `resolution`, `crs`, and style
        keywords). When `crs` is omitted it defaults to `self.crs`.

        Args:
            *args: Positional arguments for
                `cleopatra.basemap.reference.add_features` (after the axes), such as
                `layer` and `resolution`.
            ax: Axes to draw on. Defaults to the glyph's `self.ax`.
            **kwargs: Keyword arguments for
                `cleopatra.basemap.reference.add_features`. A `crs` keyword is
                defaulted to `self.crs` when omitted; an explicit `crs=`
                overrides it.

        Returns:
            matplotlib.axes.Axes: The axes, for chaining.

        Raises:
            RuntimeError: Only if there is no axes, no `ax` is given, and the
                glyph cannot create one (a bare `GeoMixin`); only `ArrayGlyph` creates and seeds
                its axes on demand, so the pre-plot builder flow is ArrayGlyph-only
                -- plot the other glyphs first (or pass `ax=`).

        See Also:
            cleopatra.basemap.reference.add_features: The underlying implementation
                and its full parameter list.
        """
        return reference.add_features(
            self._basemap_axes(ax), *args, **self._basemap_kwargs(kwargs)
        )

    def add_relief(self, *args: Any, ax: Any = None, **kwargs: Any) -> Any:
        """Draw a hypsometric relief backdrop under the glyph's data.

        Thin wrapper over `cleopatra.basemap.reference.add_relief`; arguments are
        forwarded unchanged (e.g. `resolution`, `extent`, `alpha`). When
        `crs` is omitted it defaults to `self.crs`, so on a non-EPSG:4326
        axis the relief is warped to match the data (an explicit `crs=` still
        wins). Requires the `cleopatra[tiles]` extra (Pillow, and pyproj for a
        non-4326 `crs`).

        Args:
            *args: Positional arguments for
                `cleopatra.basemap.reference.add_relief` (after the axes), such as
                `resolution`.
            ax: Axes to draw on. Defaults to the glyph's `self.ax`.
            **kwargs: Keyword arguments for
                `cleopatra.basemap.reference.add_relief`. A `crs` keyword is
                defaulted to `self.crs` when omitted; an explicit `crs=`
                overrides it.

        Returns:
            matplotlib.axes.Axes: The axes, for chaining.

        Raises:
            RuntimeError: Only if there is no axes, no `ax` is given, and the
                glyph cannot create one (a bare `GeoMixin`); only `ArrayGlyph` creates and seeds
                its axes on demand, so the pre-plot builder flow is ArrayGlyph-only
                -- plot the other glyphs first (or pass `ax=`).

        See Also:
            cleopatra.basemap.reference.add_relief: The underlying implementation and
                its full parameter list.
        """
        return reference.add_relief(
            self._basemap_axes(ax), *args, **self._basemap_kwargs(kwargs)
        )

    def add_labels(
        self, points: dict[str, tuple[float, float]], *, ax: Any = None, **kwargs: Any
    ) -> Any:
        """Annotate named points on the glyph's axes with a dot + label.

        Thin wrapper over `cleopatra.basemap.geo.add_point_labels`; draws a plain
        dot marker and text label per point, matching the minimalist
        city-label look ECMWF/CAMS maps use. Points are plotted at whatever
        coordinates the axes is already using -- plain lon/lat for a flat
        map, or reprojected x/y for an orthographic globe.

        Args:
            points: Mapping of label text to `(x, y)` coordinates, in the
                same coordinate space as whatever is already plotted on the
                axes.
            ax: Axes to draw on. Defaults to the glyph's `self.ax`.
            **kwargs: Forwarded to `cleopatra.basemap.geo.add_point_labels` (e.g.
                `color`, `marker_size`, `fontsize`, `offset`, `zorder`).

        Returns:
            matplotlib.axes.Axes: The axes, for chaining.

        Raises:
            RuntimeError: Only if there is no axes, no `ax` is given, and the
                glyph cannot create one (a bare `GeoMixin`); only `ArrayGlyph` creates and seeds
                its axes on demand, so the pre-plot builder flow is ArrayGlyph-only
                -- plot the other glyphs first (or pass `ax=`).

        Examples:
            - Label a city on a plotted glyph:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> data = np.random.rand(20, 30)
                >>> glyph = ArrayGlyph(data, extent=[-100, 15, -40, 55])
                >>> fig, ax = glyph.plot()  # doctest: +SKIP
                >>> glyph.add_labels({"London": (-0.1, 51.5)})  # doctest: +SKIP

                ```

        See Also:
            cleopatra.basemap.geo.add_point_labels: The underlying implementation.
            add_reference_map: The full basemap-chrome preset this pairs with.
        """
        return add_point_labels(self._basemap_axes(ax), points, **kwargs)

    def _background_is_dark(self, ax: Any) -> bool:
        """Whether the field displayed on `ax` reads as a dark background.

        Used only by `add_reference_map(style="auto")`. Samples an image on
        the target `ax` (falling back to the glyph's own `self.im`) and runs
        it through `im.to_rgba(...)`, which applies the colormap and `norm`
        for a colormapped scalar field and passes an RGB(A) frame through, so
        the decision reflects the *displayed* colours (mean Rec. 709
        luminance) rather than raw data magnitude. Only **opaque** cells are
        counted: masked / no-data cells render to the colormap's transparent
        "bad" colour and are excluded, so a light field that merely has a lot
        of no-data is not misread as dark. Large fields are decimated to keep
        the check O(1) in memory. Returns `False` when there is nothing
        opaque to sample (a neutral default).

        Args:
            ax: The axes being decorated. An image drawn on it is preferred
                as the sample source; otherwise `self.im` is used.

        Returns:
            bool: `True` when the mean displayed luminance is below 0.5.
        """
        images = ax.get_images() if ax is not None and hasattr(ax, "get_images") else []
        im = images[-1] if images else getattr(self, "im", None)
        arr = im.get_array() if im is not None and hasattr(im, "get_array") else None
        if arr is None:
            return False
        # `arr` is only non-None when `im` is too (see above).
        assert im is not None
        # Decimate so the decision costs O(1) memory on large rasters.
        if getattr(arr, "ndim", 0) >= 2:
            sy = max(1, arr.shape[0] // 256)
            sx = max(1, arr.shape[1] // 256)
            arr = arr[::sy, ::sx]
        rgba = np.asarray(im.to_rgba(arr), dtype=float)
        if rgba.size == 0:
            return False
        rgb, alpha = rgba[..., :3], rgba[..., 3]
        opaque = alpha > 0
        if not opaque.any():
            return False
        lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        return bool(np.mean(lum[opaque]) < 0.5)

    def _draw_basemap(self, spec: Any) -> None:
        """Compose a reference basemap under/over the glyph's data.

        Drives the glyph's own `add_relief` / `add_features` from a compact
        `spec` -- the engine behind `ArrayGlyph.animate(basemap=...)`. The
        relief is drawn first (below the data by default, `zorder=-2`) and the
        vector features after (above, `zorder=3`), so on a value-linked-opacity
        style the cool areas reveal the terrain while the data glows on top.

        Args:
            spec: The basemap specification. One of:

                * a **callable** ``f(glyph)`` -- invoked with the glyph for
                  full control (draw whatever you like on `self.ax`);
                * ``True`` -- the default backdrop: a `"low"` relief under the
                  data plus grey `coastline` and `borders` (`"50m"`) over it;
                * a **`Basemap`** -- the typed, validated form of the dict
                  below (its `relief` / `features` / `resolution` /
                  `check_alignment` fields, `features` taking `Feature`
                  objects); normalised via `Basemap._as_config`;
                * a **dict** with optional keys:
                    * ``relief`` -- ``False`` to skip it, ``True`` for the
                      default, a resolution string (e.g. ``"medium"``), or a
                      dict of `add_relief` keyword arguments;
                    * ``resolution`` -- Natural Earth resolution for the
                      features (default ``"50m"``);
                    * ``features`` -- an iterable of layer names (``str``) or
                      ``(layer, style_dict)`` pairs (default ``coastline`` and
                      ``borders``).
                    * ``check_alignment`` -- ``True`` to run an opt-in
                      mis-georeferencing check (off by default): warns when the
                      data's land/sea mask matches the relief better at a shifted
                      position, i.e. the extent looks wrong. See
                      `_check_basemap_alignment`.
        """
        if callable(spec):
            spec(self)
            return
        if isinstance(spec, Basemap):
            spec = spec._as_config()
        cfg: dict = {} if spec is True else dict(spec)

        relief = cfg.get("relief", True)
        if relief:
            relief_kwargs: dict = {"resolution": "low", "alpha": 0.55, "zorder": -2}
            if isinstance(relief, str):
                relief_kwargs["resolution"] = relief
            elif isinstance(relief, dict):
                relief_kwargs.update(relief)
            self.add_relief(relief_kwargs.pop("resolution"), **relief_kwargs)

        resolution = cfg.get("resolution", "50m")
        features = cfg.get(
            "features",
            (
                ("coastline", {"colors": "0.6", "linewidths": 0.5}),
                ("borders", {"colors": "0.4", "linewidths": 0.4}),
            ),
        )
        for feature in features:
            if isinstance(feature, Feature):
                layer, style = feature.layer, feature.style
            elif isinstance(feature, str):
                layer, style = feature, {}
            else:
                layer, style = feature
            self.add_features(layer, resolution, **{"zorder": 3, **style})

        if cfg.get("check_alignment"):
            self._check_basemap_alignment()

    def _check_basemap_alignment(
        self, resolution: str = "low", *, margin: float = 0.06
    ) -> None:
        """Warn (opt-in) when the data looks mis-georeferenced against the relief.

        Compares the data's own land/sea mask (finite = land, masked/NaN = sea)
        with the relief's ocean at the data's `extent`, and at a few one/two-cell
        shifts. If some shift matches the relief markedly better than the given
        extent does, the extent is likely wrong -- a bad offset, or a scale error
        (non-square / wrong pixel size) that drifts the field off the coastline
        progressively toward the edges. Because a shift can only *partly*
        compensate a scale error, even the residual improvement is a reliable
        tell. Testing "does a shift help?" rather than an absolute agreement
        threshold self-calibrates for coastline complexity and relief coarseness.

        This is a **heuristic diagnostic**, so it only *warns* (never corrects,
        and never raises). It no-ops when it cannot judge: no `extent`, no
        land/sea boundary in the data (all land or all sea), or the relief cannot
        be fetched (offline / no Pillow). It keys off the finite/NaN mask, so it
        is meaningful only for land-masked fields; a field defined over sea too
        has no boundary to check and is skipped.

        Args:
            resolution: Relief product used as the land/sea reference
                (`"low"`/`"medium"`). Low is enough for a coarse check.
            margin: Minimum agreement gain from shifting that triggers the
                warning. `0.06` sits between an aligned field (a shift barely
                helps) and a misregistered one (a shift helps clearly).
        """
        extent = getattr(self, "extent", None)
        arr = getattr(self, "arr", None)
        if extent is None or arr is None:
            return
        frame = arr[0] if getattr(arr, "ndim", 2) >= 3 else arr
        data = np.ma.filled(np.ma.asarray(frame).astype(float), np.nan)
        if data.ndim != 2:
            return
        land = np.isfinite(data)
        frac = float(land.mean())
        if not 0.05 < frac < 0.95:
            return  # no usable land/sea boundary to compare against

        try:
            rgb = reference.relief(resolution)
        except Exception:  # noqa: BLE001
            # offline / no Pillow: skip the check, never fail a plot
            return
        red, green, blue = (rgb[:, :, i].astype(int) for i in range(3))
        ref_land = ~((blue > red + 8) & (blue > green + 8))  # ocean == blue-dominant
        rel_h, rel_w = ref_land.shape

        xmin, xmax, ymin, ymax = extent
        rows, cols = land.shape
        dx, dy = (xmax - xmin) / cols, (ymax - ymin) / rows

        def agreement(shift_x: float, shift_y: float) -> float:
            lons = np.linspace(xmin + shift_x, xmax + shift_x, cols)
            lats = np.linspace(ymax + shift_y, ymin + shift_y, rows)  # high to low: upper image origin
            col = np.clip(((lons + 180.0) / 360.0 * rel_w).astype(int), 0, rel_w - 1)
            row = np.clip(((90.0 - lats) / 180.0 * rel_h).astype(int), 0, rel_h - 1)
            return float((ref_land[np.ix_(row, col)] == land).mean())

        here = agreement(0.0, 0.0)
        steps = (-2, -1, 0, 1, 2)
        best = max(agreement(i * dx, j * dy) for i in steps for j in steps)
        if best - here > margin:
            warnings.warn(
                "basemap alignment: the data's land/sea mask matches the "
                f"reference relief better when shifted ({here:.2f} -> {best:.2f} "
                "agreement). The extent may be mis-georeferenced (wrong pixel "
                "size or offset); verify it -- a scale error misaligns "
                "progressively toward the edges.",
                stacklevel=3,
            )

    def add_reference_map(
        self,
        style: str = "ecmwf",
        *,
        ax: Any = None,
        extent: Any = None,
        resolution: str | None = None,
        graticule_step: float | None = None,
        zorder: int = 5,
    ) -> Any:
        """Dress the glyph's axes in a weather-centre reference-map style.

        One call composes the recipe that otherwise takes ~15 lines of
        matplotlib after `plot`/`animate`: grey Natural Earth `coastline`
        + `borders`, a dashed lon/lat graticule, `°W`/`°N` degree labels,
        and a subtle frame, plus -- for `"ecmwf-dark"` -- a dimmed relief
        backdrop beneath the data. The chrome layers on top of the existing
        data, so call it after plotting.

        The map is drawn in the axes' current geographic coordinates. Pass
        `extent` (or construct the glyph with `extent=`) so the axes are
        georeferenced — otherwise the coastlines cannot align with the data
        and a warning is emitted. Deriving that extent from a source dataset
        is the caller's job (cleopatra renders supplied coordinates; it does
        not read geotransforms).

        Args:
            style: A name from `available_map_styles()` (`"ecmwf"`,
                `"ecmwf-dark"`), or `"auto"` to pick between them from the
                background luminance (dark backgrounds get the lighter
                `"ecmwf-dark"` greys so coastlines stay visible). Default
                `"ecmwf"`. `"ecmwf-dark"` also draws a dimmed relief backdrop
                under the data (needs the `[tiles]` extra; if Pillow is
                missing the backdrop is skipped with a warning and the
                coastline/border chrome is still drawn).
            ax: Axes to draw on. Defaults to the glyph's `self.ax`.
            extent: Optional `[xmin, ymin, xmax, ymax]` (i.e.
                `[west, south, east, north]`) in the axes' CRS -- the same
                order as `ArrayGlyph(extent=...)`. When given, the image and
                axis limits are set to it (handling the pixel-coordinate
                RGB/animate case); when omitted the current axis limits are
                used.
            resolution: Natural Earth resolution for the coastline/borders
                (`"110m"`/`"50m"`/`"10m"`). Defaults to the style's value.
            graticule_step: Degree spacing for the graticule. Defaults to a
                "nice" step giving ~6 divisions across the wider span.
            zorder: Draw order for the reference layers (drawn above the
                data; the graticule sits just below the coastlines).

        Returns:
            matplotlib.axes.Axes: The decorated axes, for chaining.

        Raises:
            RuntimeError: Only if there is no axes, no `ax` is given, and the
                glyph cannot create one (a bare `GeoMixin`); only `ArrayGlyph` creates and seeds
                its axes on demand, so the pre-plot builder flow is ArrayGlyph-only
                -- plot the other glyphs first (or pass `ax=`).
            ValueError: If `style` is not a known preset or `"auto"`, or if
                `graticule_step` is given and is not a positive number.

        Examples:
            - Dress a georeferenced field in the ECMWF look:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> data = np.random.rand(20, 30)
                >>> glyph = ArrayGlyph(data, extent=[-100, 15, -40, 55])
                >>> fig, ax = glyph.plot()  # doctest: +SKIP
                >>> glyph.add_reference_map("ecmwf")  # doctest: +SKIP

                ```

        See Also:
            add_features: The Natural Earth layer helper this composes.
            available_map_styles: The built-in preset names.
        """
        if graticule_step is not None and (
            not math.isfinite(graticule_step) or graticule_step <= 0
        ):
            raise ValueError(
                "graticule_step must be a positive, finite number, got "
                f"{graticule_step}"
            )
        target = self._basemap_axes(ax)

        resolved = style
        if style == "auto":
            resolved = "ecmwf-dark" if self._background_is_dark(target) else "ecmwf"
        if resolved not in REFERENCE_MAP_STYLES:
            raise ValueError(
                f"Unknown map style {style!r}; available: "
                f"{available_map_styles()} (or 'auto')."
            )
        preset = REFERENCE_MAP_STYLES[resolved]

        if extent is not None:
            if len(extent) != 4:
                raise ValueError(
                    "extent must be [xmin, ymin, xmax, ymax] "
                    f"(4 values), got {len(extent)}"
                )
            west, south, east, north = extent
            im = getattr(self, "im", None)
            if im is not None and hasattr(im, "set_extent"):
                im.set_extent((west, east, south, north))
            target.set_xlim(west, east)
            target.set_ylim(south, north)
        elif getattr(self, "extent", None) is None:
            warnings.warn(
                "add_reference_map: the glyph has no geographic extent, so "
                "coastlines/borders may not align with the data. Pass "
                "extent=[west, south, east, north] or construct the glyph "
                "with extent=.",
                stacklevel=2,
            )

        # A preset may bundle a dimmed relief backdrop (ecmwf-dark). Draw it
        # under the data, defaulting crs to self.crs like the other helpers.
        # Skip it when the axes are not georeferenced (already warned above),
        # and -- so the coastline chrome never hard-depends on Pillow --
        # degrade with a warning when the [tiles] extra is missing.
        relief = preset.get("relief")
        if relief and (extent is not None or getattr(self, "extent", None) is not None):
            relief_kwargs: dict = {"resolution": "low", "alpha": 0.5, "zorder": -2}
            if isinstance(relief, str):
                relief_kwargs["resolution"] = relief
            elif isinstance(relief, dict):
                relief_kwargs.update(relief)
            try:
                self.add_relief(
                    relief_kwargs.pop("resolution"), ax=target, **relief_kwargs
                )
            except (ImportError, OSError) as exc:
                # ImportError -> Pillow (the [tiles] extra) is missing; OSError
                # (incl. ConnectionError) -> the relief asset could not be
                # fetched or decoded. Either way, skip the backdrop but still
                # draw the independently cached coastline/border chrome; a bad
                # relief resolution in a custom preset (ValueError) still raises.
                warnings.warn(
                    "add_reference_map: relief backdrop skipped "
                    f"({exc}); the coastline/border chrome is still drawn.",
                    stacklevel=2,
                )

        res = resolution or preset["resolution"]
        self.add_features(
            "coastline", res, ax=target, zorder=zorder, **preset["coastline"]
        )
        self.add_features("borders", res, ax=target, zorder=zorder, **preset["borders"])

        xmin, xmax = target.get_xlim()
        ymin, ymax = target.get_ylim()
        step = (
            graticule_step
            if graticule_step is not None
            else _nice_step(max(abs(xmax - xmin), abs(ymax - ymin)))
        )
        target.xaxis.set_major_locator(MultipleLocator(step))
        target.yaxis.set_major_locator(MultipleLocator(step))
        target.xaxis.set_major_formatter(FuncFormatter(_lon_formatter))
        target.yaxis.set_major_formatter(FuncFormatter(_lat_formatter))
        target.grid(True, zorder=zorder - 1, **preset["graticule"])
        target.tick_params(length=0, **preset["labels"])
        for spine in target.spines.values():
            spine.set(**preset["spines"])
        return target
