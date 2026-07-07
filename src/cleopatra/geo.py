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
`cleopatra.tiles` / `cleopatra.reference`. The standalone functions remain
the source of truth; this mixin only removes the import + explicit-axes
boilerplate for the geographic glyphs (`ArrayGlyph`, `MeshGlyph`,
`VectorGlyph`, `FlowGlyph`, `PolygonGlyph`, `ScatterGlyph`). Non-geographic
glyphs (line/bar charts, statistical plots) deliberately do not inherit
it.

Importing this module (and the `cleopatra.tiles` / `cleopatra.reference`
modules it calls) does not require the optional `cleopatra[tiles]` extra:
those modules gate their `[tiles]` dependencies (`mercantile`, `pyproj`,
`Pillow`, ...) behind their own internal lazy imports, so the extra is
only needed when a basemap is actually drawn.
"""

from __future__ import annotations

import importlib.util
import warnings
from typing import Any

import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

from cleopatra import reference, tiles

#: Built-in reference-map style presets for `GeoMixin.add_reference_map`.
#: `"ecmwf"` is tuned for light backgrounds; `"ecmwf-dark"` uses lighter
#: greys so coastlines stay visible over a dark field (e.g. a satellite
#: true-colour RGB). Each entry is a plain, overridable dict of the layer
#: styles, graticule, tick-label, and frame (spine) parameters.
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
        >>> from cleopatra.geo import available_map_styles
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
        float: A "nice" step (1, 2, 2.5, 5, 10, ...) so the graticule lands
        on round degree values.

    Examples:
        ```python
        >>> from cleopatra.geo import _nice_step
        >>> _nice_step(30)
        5.0
        >>> _nice_step(4)
        1.0

        ```
    """
    if span <= 0:
        return 1.0
    raw = span / max(target_divisions, 1)
    for candidate in (1, 2, 2.5, 5, 10, 15, 20, 30, 45, 60):
        if raw <= candidate:
            return float(candidate)
    return 90.0


def _lon_formatter(value: float, _pos: Any = None) -> str:
    """Format a longitude tick as `°W`/`°E` (0 at the meridian).

    Examples:
        ```python
        >>> from cleopatra.geo import _lon_formatter
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
        >>> from cleopatra.geo import _lat_formatter
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
            >>> from cleopatra.geo import _validate_crs
            >>> _validate_crs(None) is None
            True
            >>> _validate_crs(4326)
            4326
            >>> _validate_crs("4326")
            4326

            ```
        - Wrong types are rejected immediately:
            ```python
            >>> from cleopatra.geo import _validate_crs
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


class GeoMixin:
    """Mixin giving geographic glyphs `add_tiles` / `add_features` / `add_relief`.

    The host class is expected to expose the plotted axes as `self.ax`
    (every `cleopatra.glyph.Glyph` subclass does). Call these after
    plotting, or pass `ax=` explicitly.

    Set `self.crs` to the CRS of the data plotted on the axes (an EPSG code
    or CRS string) and `add_features` / `add_tiles` default their `crs=`
    argument to it, so the reference layer is placed in matching
    coordinates without restating it on every call. An explicit `crs=`
    still wins; leaving `self.crs` as `None` preserves each helper's own
    default. `add_relief` ignores `crs` -- relief is a fixed EPSG:4326
    raster placed by `extent`.
    """

    #: Set by `cleopatra.glyph.Glyph`; the axes the basemap is drawn on.
    ax: Any

    #: Backing store for the validated `crs` property; `None` means unset.
    _crs: int | str | None = None

    @property
    def crs(self) -> int | str | None:
        """CRS of the data plotted on `self.ax` (EPSG code or CRS string).

        When set, `add_features` / `add_tiles` default `crs=` to it; `None`
        keeps each helper's own default. The value is **validated on
        assignment** (see `cleopatra.geo._validate_crs`) so mistakes surface
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
        """Return the axes to draw a basemap on (`ax` or `self.ax`).

        Args:
            ax: An explicit axes to use instead of the glyph's own.

        Returns:
            The resolved matplotlib axes.

        Raises:
            RuntimeError: If neither `ax` nor `self.ax` is available -- the
                glyph has not been plotted yet.
        """
        target = ax if ax is not None else getattr(self, "ax", None)
        if target is None:
            raise RuntimeError(
                "No axes to draw on. Plot the glyph first (or pass ax=) "
                "before adding a basemap layer."
            )
        return target

    def _basemap_kwargs(self, kwargs: dict) -> dict:
        """Default `crs` to `self.crs` when the caller did not set it.

        Only injects when `self.crs` is set and `crs` is absent (or `None`)
        in `kwargs`, so the default `self.crs is None` is a pure pass-through
        and an explicit `crs=` always wins. `crs` is keyword-only in both
        `add_features` and `add_tiles`, so it always arrives via `kwargs`.

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

        Thin wrapper over `cleopatra.tiles.add_tiles`; positional and
        keyword arguments are forwarded unchanged (e.g. `source`, `crs`,
        `zoom`, `alpha`). When `crs` is omitted it defaults to `self.crs`.
        Requires the `cleopatra[tiles]` extra.

        Args:
            *args: Positional arguments for `cleopatra.tiles.add_tiles`
                (after the axes).
            ax: Axes to draw on. Defaults to the glyph's `self.ax`.
            **kwargs: Keyword arguments for `cleopatra.tiles.add_tiles`. A
                `crs` keyword is defaulted to `self.crs` when omitted; an
                explicit `crs=` overrides it.

        Returns:
            matplotlib.axes.Axes: The axes, for chaining.

        Raises:
            RuntimeError: If the glyph has no axes yet and `ax` is not given.

        See Also:
            cleopatra.tiles.add_tiles: The underlying implementation and its
                full parameter list.
        """
        return tiles.add_tiles(
            self._basemap_axes(ax), *args, **self._basemap_kwargs(kwargs)
        )

    def add_features(self, *args: Any, ax: Any = None, **kwargs: Any) -> Any:
        """Draw a Natural Earth reference layer on the glyph's axes.

        Thin wrapper over `cleopatra.reference.add_features`; arguments are
        forwarded unchanged (e.g. `layer`, `resolution`, `crs`, and style
        keywords). When `crs` is omitted it defaults to `self.crs`.

        Args:
            *args: Positional arguments for
                `cleopatra.reference.add_features` (after the axes), such as
                `layer` and `resolution`.
            ax: Axes to draw on. Defaults to the glyph's `self.ax`.
            **kwargs: Keyword arguments for
                `cleopatra.reference.add_features`. A `crs` keyword is
                defaulted to `self.crs` when omitted; an explicit `crs=`
                overrides it.

        Returns:
            matplotlib.axes.Axes: The axes, for chaining.

        Raises:
            RuntimeError: If the glyph has no axes yet and `ax` is not given.

        See Also:
            cleopatra.reference.add_features: The underlying implementation
                and its full parameter list.
        """
        return reference.add_features(
            self._basemap_axes(ax), *args, **self._basemap_kwargs(kwargs)
        )

    def add_relief(self, *args: Any, ax: Any = None, **kwargs: Any) -> Any:
        """Draw a hypsometric relief backdrop under the glyph's data.

        Thin wrapper over `cleopatra.reference.add_relief`; arguments are
        forwarded unchanged (e.g. `resolution`, `extent`, `alpha`).
        Requires the `cleopatra[tiles]` extra (Pillow).

        Args:
            *args: Positional arguments for
                `cleopatra.reference.add_relief` (after the axes), such as
                `resolution`.
            ax: Axes to draw on. Defaults to the glyph's `self.ax`.
            **kwargs: Keyword arguments for
                `cleopatra.reference.add_relief`.

        Returns:
            matplotlib.axes.Axes: The axes, for chaining.

        Raises:
            RuntimeError: If the glyph has no axes yet and `ax` is not given.

        See Also:
            cleopatra.reference.add_relief: The underlying implementation and
                its full parameter list.
        """
        return reference.add_relief(self._basemap_axes(ax), *args, **kwargs)

    def _background_is_dark(self, ax: Any) -> bool:
        """Whether the plotted field on `ax` reads as a dark background.

        Used only by `add_reference_map(style="auto")`. Samples the glyph's
        rendered image (`self.im`): `im.to_rgba(...)` applies the colormap
        and `norm` for a colormapped scalar field and passes an RGB(A) frame
        through, so the decision reflects the *displayed* colours (mean
        Rec. 709 luminance) rather than raw data magnitude. Masked / no-data
        cells contribute their rendered "bad" colour, and there is no NaN in
        the RGBA result, so the reduction is warning-free. Returns `False`
        when there is no image to sample (a neutral default).

        Args:
            ax: The axes being decorated (unused directly; the sample comes
                from `self.im`, kept for signature symmetry).

        Returns:
            bool: `True` when the mean displayed luminance is below 0.5.
        """
        im = getattr(self, "im", None)
        arr = im.get_array() if im is not None and hasattr(im, "get_array") else None
        if arr is None:
            return False
        # Render through the image's norm+colormap so a colormapped field is
        # judged by what is shown, not by its data units; RGB(A) passes through.
        rgb = np.asarray(im.to_rgba(arr), dtype=float)[..., :3]
        if rgb.size == 0:
            return False
        lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        return bool(np.mean(lum) < 0.5)

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
        and a subtle frame. It layers on top of the existing data, so call
        it after plotting.

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
                `"ecmwf"`.
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
            RuntimeError: If the glyph has no axes yet and `ax` is not given.
            ValueError: If `style` is not a known preset or `"auto"`, or if
                `graticule_step` is given and is not a positive number.

        Examples:
            - Dress a georeferenced field in the ECMWF look:
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> data = np.random.rand(20, 30)
                >>> glyph = ArrayGlyph(data, extent=[-100, 15, -40, 55])
                >>> fig, ax = glyph.plot()  # doctest: +SKIP
                >>> glyph.add_reference_map("ecmwf")  # doctest: +SKIP

                ```

        See Also:
            add_features: The Natural Earth layer helper this composes.
            available_map_styles: The built-in preset names.
        """
        if graticule_step is not None and graticule_step <= 0:
            raise ValueError(
                f"graticule_step must be a positive number, got {graticule_step}"
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
            # `[xmin, ymin, xmax, ymax]` == `[west, south, east, north]`,
            # the same order as ArrayGlyph(extent=...); matplotlib wants
            # `(xmin, xmax, ymin, ymax)` for `set_extent`.
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

        res = resolution or preset["resolution"]
        self.add_features(
            "coastline", res, ax=target, zorder=zorder, **preset["coastline"]
        )
        self.add_features(
            "borders", res, ax=target, zorder=zorder, **preset["borders"]
        )

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
