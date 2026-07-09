"""Static projected ('globe') map frame for matplotlib axes.

Provides `apply_projection_frame` -- a single stateless helper that turns
a plain `matplotlib.axes.Axes` into a static projected map frame: it sets
the projected limits and equal aspect, draws the projection *boundary*
(the globe's circle, Robinson's rounded rectangle, ...), optionally
*clips* the existing data layers to that boundary, and draws the
*graticule* polylines.

The module is **pure matplotlib with no PROJ/CRS dependency**. It only
*receives* already-computed geometry -- boundary vertices, graticule
polylines, and projected limits -- as plain arrays. Whatever produces the
projection (reprojecting data and deriving the boundary/graticule) lives
upstream; cleopatra only renders the result. This keeps the engine split
clean: the upstream owns CRS/PROJ, cleopatra owns matplotlib.

Examples:
    Frame a plain axes as an orthographic globe and clip an image to the
    boundary circle:

    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> theta = np.linspace(0, 2 * np.pi, 200)
    >>> boundary = np.column_stack([np.cos(theta), np.sin(theta)])
    >>> fig, ax = plt.subplots()
    >>> _ = ax.imshow(np.random.rand(8, 8), extent=(-1, 1, -1, 1))
    >>> patch = apply_projection_frame(
    ...     ax, boundary_xy=boundary, xlim=(-1, 1), ylim=(-1, 1)
    ... )
    >>> ax.get_aspect()
    1.0
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path

#: Default style for the projection boundary patch. Merged with (and
#: overridden by) the caller's ``boundary_kw``.
DEFAULT_BOUNDARY_KW: dict[str, Any] = {"edgecolor": "black", "linewidth": 0.8}

#: Default style for the graticule polylines. Merged with (and overridden
#: by) the caller's ``graticule_kw``.
DEFAULT_GRATICULE_KW: dict[str, Any] = {"color": "gray", "linewidth": 0.4}


def _as_xy(array: Any, name: str) -> np.ndarray:
    """Coerce input to a float ``(N, 2)`` array of x/y vertices.

    Args:
        array: Anything array-like holding vertex coordinates.
        name: Argument name, used in error messages.

    Returns:
        numpy.ndarray: A ``(N, 2)`` float array.

    Raises:
        ValueError: If the input is not 2-D with two columns.

    Examples:
        - Coerce a nested list of integer pairs to a float array:
            ```python
            >>> from cleopatra.projection import _as_xy
            >>> _as_xy([[1, 0], [0, 1], [-1, 0]], "boundary_xy").tolist()
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]

            ```
        - A single vertex is a valid ``(1, 2)`` array:
            ```python
            >>> from cleopatra.projection import _as_xy
            >>> _as_xy([[0.5, 0.5]], "graticule_lines[0]").shape
            (1, 2)

            ```
        - A 1-D input raises ``ValueError`` naming the argument:
            ```python
            >>> from cleopatra.projection import _as_xy
            >>> _as_xy([1, 2, 3], "boundary_xy")
            Traceback (most recent call last):
                ...
            ValueError: boundary_xy must be an (N, 2) array of x/y coordinates, got shape (3,).

            ```
    """
    xy = np.asarray(array, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(
            f"{name} must be an (N, 2) array of x/y coordinates, "
            f"got shape {xy.shape}."
        )
    return xy


def apply_projection_frame(
    ax: Any,
    *,
    boundary_xy: Any,
    xlim: Sequence[float],
    ylim: Sequence[float],
    graticule_lines: Sequence[Any] | None = None,
    clip_artists: bool = True,
    boundary_kw: dict[str, Any] | None = None,
    graticule_kw: dict[str, Any] | None = None,
) -> PathPatch:
    """Turn a plain axes into a static projected ('globe') frame.

    Sets equal aspect and the projected x/y limits, draws the projection
    boundary as a `matplotlib.patches.PathPatch`, draws the graticule
    polylines, optionally clips the existing data layers to the boundary,
    and turns the axis decorations off. The boundary geometry, graticule
    polylines, and limits are supplied as plain arrays -- this function
    performs no reprojection and has no PROJ/CRS dependency.

    This is a one-shot helper: each call appends a fresh boundary patch and
    a fresh set of graticule lines, so calling it twice on the same axes
    stacks duplicate artists. Apply it once per axes (create a new axes to
    re-frame).

    Args:
        ax: Matplotlib `matplotlib.axes.Axes` to frame. Any data
            layers to be clipped should already be plotted.
        boundary_xy: ``(N, 2)`` projection-boundary vertices in projected
            coordinates (e.g. the globe's circle). Array-like.
        xlim: Projected-coordinate x-limits ``(xmin, xmax)`` -- the CRS
            domain in the x direction.
        ylim: Projected-coordinate y-limits ``(ymin, ymax)``.
        graticule_lines: Optional list of ``(M, 2)`` polylines (already
            densified and projected), each drawn as a graticule line.
            `None` (default) draws no graticule.
        clip_artists: If `True` (default), clip the existing data layers
            (`ax.images`, `ax.collections`, `ax.lines`) and the drawn
            graticule to the boundary path.
        boundary_kw: Style overrides for the boundary patch, merged over
            `DEFAULT_BOUNDARY_KW`. ``facecolor`` defaults to
            ``"none"`` so the patch never hides the data.
        graticule_kw: Style overrides for the graticule lines, merged
            over `DEFAULT_GRATICULE_KW`.

    Returns:
        matplotlib.patches.PathPatch: The boundary patch added to the
        axes (also used as the clip path).

    Raises:
        TypeError: If `ax` is not a matplotlib Axes.
        ValueError: If `boundary_xy` or a graticule line is not an
            ``(N, 2)`` array, or if `xlim`/`ylim` are not 2-tuples.

    Examples:
        - Frame a plain axes as a globe and read back the result: the
            returned patch is registered on the axes and the aspect is equal:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.projection import apply_projection_frame
            >>> theta = np.linspace(0, 2 * np.pi, 200)
            >>> boundary = np.column_stack([np.cos(theta), np.sin(theta)])
            >>> fig, ax = plt.subplots()
            >>> patch = apply_projection_frame(
            ...     ax, boundary_xy=boundary, xlim=(-1, 1), ylim=(-1, 1)
            ... )
            >>> ax.get_aspect()
            1.0
            >>> patch in ax.patches
            True

            ```
        - Clip a data image and draw one graticule line (plain lists are
            accepted): the image gains a clip path and one line is added:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.projection import apply_projection_frame
            >>> boundary = [[1, 0], [0, 1], [-1, 0], [0, -1]]
            >>> meridian = [[0, -1], [0, 1]]
            >>> fig, ax = plt.subplots()
            >>> im = ax.imshow(np.zeros((4, 4)), extent=(-1, 1, -1, 1))
            >>> patch = apply_projection_frame(
            ...     ax,
            ...     boundary_xy=boundary,
            ...     xlim=(-1, 1),
            ...     ylim=(-1, 1),
            ...     graticule_lines=[meridian],
            ... )
            >>> im.get_clip_path() is not None
            True
            >>> len(ax.lines)
            1

            ```
        - Passing a non-Axes object raises ``TypeError``:
            ```python
            >>> from cleopatra.projection import apply_projection_frame
            >>> apply_projection_frame(
            ...     object(), boundary_xy=[[0, 0], [1, 1]], xlim=(-1, 1), ylim=(-1, 1)
            ... )
            Traceback (most recent call last):
                ...
            TypeError: ax must be a matplotlib.axes.Axes instance, got object

            ```
    """
    if not hasattr(ax, "set_xlim") or not hasattr(ax, "add_patch"):
        raise TypeError(
            "ax must be a matplotlib.axes.Axes instance, "
            f"got {type(ax).__name__}"
        )

    if len(xlim) != 2 or len(ylim) != 2:
        raise ValueError(
            f"xlim and ylim must each be a (min, max) pair, "
            f"got xlim={xlim!r}, ylim={ylim!r}."
        )

    boundary = _as_xy(boundary_xy, "boundary_xy")

    ax.set_aspect("equal")
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    # The boundary is treated as a closed ring for clipping/fill regardless
    # of whether its final vertex repeats the first: matplotlib's
    # point-in-path test implicitly connects the last vertex back to the
    # first, so an open ring still clips correctly.
    patch = PathPatch(
        Path(boundary),
        facecolor="none",
        **{**DEFAULT_BOUNDARY_KW, **(boundary_kw or {})},
    )
    ax.add_patch(patch)

    graticule_style = {**DEFAULT_GRATICULE_KW, **(graticule_kw or {})}
    graticule_artists = []
    for i, line in enumerate(graticule_lines or []):
        xy = _as_xy(line, f"graticule_lines[{i}]")
        (artist,) = ax.plot(xy[:, 0], xy[:, 1], **graticule_style)
        graticule_artists.append(artist)

    if clip_artists:
        for art in (*ax.images, *ax.collections, *ax.lines):
            art.set_clip_path(patch)

    ax.set_axis_off()
    return patch


#: Mean Earth radius (metres), IUGG 1980 value. Used as the orthographic
#: projection's spherical datum radius (`+R=`), which keeps the visible
#: hemisphere's boundary an exact circle -- see `_ortho_proj4`.
ORTHOGRAPHIC_RADIUS_M = 6371008.7714


def _ortho_proj4(center_lat: float, center_lon: float) -> str:
    """Build a spherical-datum orthographic proj4 string.

    A spherical (not ellipsoidal) datum is used deliberately: with `+R=`
    fixed, the orthographic projection's visible-hemisphere boundary is an
    exact circle of radius `ORTHOGRAPHIC_RADIUS_M` regardless of the centre
    point, so `orthographic_boundary` can return it in closed form -- no
    pyproj query needed. This is standard practice for an illustrative globe
    view (not for geodetically precise reprojection).
    """
    return (
        f"+proj=ortho +R={ORTHOGRAPHIC_RADIUS_M} +lat_0={center_lat} "
        f"+lon_0={center_lon} +x_0=0 +y_0=0 +units=m +no_defs"
    )


def _require_pyproj(action: str) -> None:
    """Raise an actionable `ImportError` if pyproj is not installed.

    Args:
        action: Short description of what needed pyproj, used in the
            message (e.g. `"Orthographic ('globe') projection"`).
    """
    if importlib.util.find_spec("pyproj") is None:
        raise ImportError(
            f"{action} requires pyproj, provided by the [tiles] extra. "
            "Install with `pip install cleopatra[tiles]`."
        )


def _visible_hemisphere(
    lon: np.ndarray, lat: np.ndarray, center_lat: float, center_lon: float
) -> np.ndarray:
    """Boolean mask: `True` where `(lon, lat)` is on the near (visible) side.

    Uses the spherical law of cosines for the great-circle central angle `c`
    between `(lon, lat)` and the centre point; the near hemisphere is exactly
    `cos(c) >= 0`.
    """
    lat0_rad, lon0_rad = math.radians(center_lat), math.radians(center_lon)
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    cos_c = np.sin(lat0_rad) * np.sin(lat_rad) + np.cos(lat0_rad) * np.cos(
        lat_rad
    ) * np.cos(lon_rad - lon0_rad)
    return cos_c >= 0


def _split_visible_runs(xy: np.ndarray, visible: np.ndarray) -> list[np.ndarray]:
    """Split an `(N, 2)` polyline into contiguous runs where `visible` is `True`.

    Runs shorter than 2 points are dropped (not a drawable line segment).
    """
    runs: list[np.ndarray] = []
    start: int | None = None
    for i, ok in enumerate(visible):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= 2:
                runs.append(xy[start:i])
            start = None
    if start is not None and len(visible) - start >= 2:
        runs.append(xy[start:])
    return runs


def orthographic_grid(
    lon: Any,
    lat: Any,
    data: Any,
    center_lat: float = 90.0,
    center_lon: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproject a lon/lat grid onto an orthographic ('globe') view.

    Projects every grid point via pyproj onto a sphere viewed from directly
    above `(center_lat, center_lon)`, and masks out the far (non-visible)
    hemisphere -- the orthographic projection formula is defined everywhere
    but is only meaningful for the visible half, so points beyond it are set
    to NaN in the returned data rather than silently folded onto the visible
    disk. Pair the result with `alpha_scaled_mesh` (`cleopatra.colors`),
    which -- unlike `alpha_scaled_image` -- can render this kind of
    curvilinear (non-rectangular) grid.

    Args:
        lon: Longitudes in degrees, either a 1D vector (paired with a 1D
            `lat` to build a regular grid via `np.meshgrid`) or a 2D array
            already matching `data`'s shape.
        lat: Latitudes in degrees, same convention as `lon`.
        data: 2D array of values to reproject alongside the grid.
        center_lat: Latitude the globe is centred on (the "camera" points at
            this point). Defaults to `90.0` (the North Pole), matching the
            ECMWF/CAMS Arctic-centred style.
        center_lon: Longitude the globe is centred on. Defaults to `0.0`.

    Returns:
        tuple: `(x, y, masked_data)` -- projected x/y coordinates (metres,
        same shape as `data`) and a copy of `data` with the far hemisphere
        set to NaN.

    Raises:
        ImportError: If pyproj (the `[tiles]` extra) is not installed.
        ValueError: If `lon`/`lat`/`data` shapes are incompatible.

    Examples:
        - Reproject a tiny 2x2 grid centred on the North Pole; the row at
          latitude -90 (South Pole) is masked out as not visible:
            ```python
            >>> import numpy as np
            >>> from cleopatra.projection import orthographic_grid
            >>> lon = np.array([0.0, 90.0])
            >>> lat = np.array([90.0, -90.0])
            >>> data = np.array([[1.0, 2.0], [3.0, 4.0]])
            >>> x, y, masked = orthographic_grid(lon, lat, data)
            >>> masked[0]  # latitude 90: visible, values kept
            array([1., 2.])
            >>> np.all(np.isnan(masked[1]))  # latitude -90: not visible
            np.True_

            ```

    See Also:
        orthographic_boundary: The matching globe outline for this centre.
        orthographic_graticule: Matching meridian/parallel lines.
        cleopatra.colors.alpha_scaled_mesh: Renders the returned grid.
    """
    _require_pyproj("Orthographic ('globe') projection")
    from pyproj import Transformer

    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    data_arr = np.asarray(data, dtype=float)
    if lon_arr.ndim == 1 and lat_arr.ndim == 1 and data_arr.ndim == 2:
        lon_arr, lat_arr = np.meshgrid(lon_arr, lat_arr)
    if lon_arr.shape != data_arr.shape or lat_arr.shape != data_arr.shape:
        raise ValueError(
            "lon/lat/data shapes must match (after meshgrid for 1D lon/lat), "
            f"got lon={lon_arr.shape}, lat={lat_arr.shape}, data={data_arr.shape}"
        )

    transformer = Transformer.from_crs(
        "EPSG:4326", _ortho_proj4(center_lat, center_lon), always_xy=True
    )
    x, y = transformer.transform(lon_arr, lat_arr)
    visible = _visible_hemisphere(lon_arr, lat_arr, center_lat, center_lon)
    masked_data = np.where(visible, data_arr, np.nan)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), masked_data


def orthographic_boundary(
    n: int = 200, radius: float = ORTHOGRAPHIC_RADIUS_M
) -> np.ndarray:
    """Return the orthographic globe's boundary circle.

    The orthographic projection's visible-hemisphere boundary is always a
    circle of the projection's radius centred at the origin, independent of
    which point the globe is centred on -- no pyproj call is needed. Pass
    the result as `apply_projection_frame`'s `boundary_xy`.

    Args:
        n: Number of vertices around the circle.
        radius: Circle radius in projected-CRS units (metres). Defaults to
            `ORTHOGRAPHIC_RADIUS_M`, the same radius `orthographic_grid`
            projects with -- pass a matching value to both if overridden.

    Returns:
        np.ndarray: `(n, 2)` array of boundary vertices.

    Examples:
        - The boundary is a circle of the given radius, centred at the origin:
            ```python
            >>> import numpy as np
            >>> from cleopatra.projection import orthographic_boundary
            >>> boundary = orthographic_boundary(n=100, radius=1.0)
            >>> boundary.shape
            (100, 2)
            >>> np.allclose(np.hypot(boundary[:, 0], boundary[:, 1]), 1.0)
            True

            ```

    See Also:
        orthographic_grid: The projected data this boundary frames.
        apply_projection_frame: Renders the boundary onto an axes.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def orthographic_graticule(
    center_lat: float = 90.0,
    center_lon: float = 0.0,
    step: float = 30.0,
    densify: int = 200,
) -> list[np.ndarray]:
    """Build graticule (meridian/parallel) polylines for an orthographic view.

    Generates meridian (constant-longitude) and parallel (constant-latitude)
    lines spaced `step` degrees apart, reprojects them with the same centre
    as `orthographic_grid`, and splits each at the visible-hemisphere edge so
    only the visible portion of each line is returned. Pass the result as
    `apply_projection_frame`'s `graticule_lines`.

    Args:
        center_lat: Latitude the globe is centred on. Must match the value
            passed to `orthographic_grid` for the graticule to align with
            the data.
        center_lon: Longitude the globe is centred on. Must match
            `orthographic_grid`.
        step: Degree spacing between meridians/parallels. Must be positive.
        densify: Number of points each line is sampled at before
            reprojecting -- higher values give smoother curves near the
            visible-hemisphere edge.

    Returns:
        list[np.ndarray]: One `(m, 2)` array per visible line segment (lines
        crossing the hemisphere edge are split into separate segments).

    Raises:
        ImportError: If pyproj (the `[tiles]` extra) is not installed.
        ValueError: If `step` is not a positive, finite number.

    Examples:
        - A 90-degree step gives a small set of meridians/parallels, each a
          `(densify, 2)` or shorter (edge-clipped) segment:
            ```python
            >>> from cleopatra.projection import orthographic_graticule
            >>> lines = orthographic_graticule(step=90.0, densify=50)
            >>> len(lines) > 0
            True
            >>> all(line.shape[1] == 2 for line in lines)
            True

            ```
        - A non-positive step raises `ValueError`:
            ```python
            >>> from cleopatra.projection import orthographic_graticule
            >>> orthographic_graticule(step=0)
            Traceback (most recent call last):
                ...
            ValueError: step must be a positive, finite number, got 0

            ```

    See Also:
        orthographic_grid: The projected data this graticule frames.
        orthographic_boundary: The matching globe outline.
    """
    if not math.isfinite(step) or step <= 0:
        raise ValueError(f"step must be a positive, finite number, got {step}")
    _require_pyproj("Orthographic ('globe') graticule")
    from pyproj import Transformer

    transformer = Transformer.from_crs(
        "EPSG:4326", _ortho_proj4(center_lat, center_lon), always_xy=True
    )

    lines_lonlat = []
    lat_dense = np.linspace(-90.0, 90.0, densify)
    for lon0 in np.arange(-180.0, 180.0, step):
        lines_lonlat.append(np.column_stack([np.full_like(lat_dense, lon0), lat_dense]))
    lon_dense = np.linspace(-180.0, 180.0, densify)
    for lat0 in np.arange(-90.0 + step, 90.0, step):
        lines_lonlat.append(np.column_stack([lon_dense, np.full_like(lon_dense, lat0)]))

    segments: list[np.ndarray] = []
    for line in lines_lonlat:
        lon_l, lat_l = line[:, 0], line[:, 1]
        x_l, y_l = transformer.transform(lon_l, lat_l)
        visible = _visible_hemisphere(lon_l, lat_l, center_lat, center_lon)
        xy = np.column_stack([x_l, y_l])
        segments.extend(_split_visible_runs(xy, visible))
    return segments
