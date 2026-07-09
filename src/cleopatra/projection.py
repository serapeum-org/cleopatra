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
        set to NaN. `x`/`y` are always finite: the orthographic formula
        diverges at the antipodal point, so any non-visible or non-finite
        coordinate is replaced with a `0.0` placeholder (safe because those
        cells carry no data and are never rendered), keeping the grid a
        valid `pcolormesh`/`imshow` input.

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
    x_raw, y_raw = transformer.transform(lon_arr, lat_arr)
    x = np.asarray(x_raw, dtype=float)
    y = np.asarray(y_raw, dtype=float)
    visible = _visible_hemisphere(lon_arr, lat_arr, center_lat, center_lon)
    masked_data = np.where(visible, data_arr, np.nan)
    # The orthographic formula diverges (inf) exactly at the antipodal point,
    # and pcolormesh/imshow require finite coordinates for every cell even
    # when its data is masked out. Those cells carry no data (masked_data is
    # NaN there), so their exact position is irrelevant -- replace any
    # non-visible or non-finite coordinate with a finite placeholder.
    coord_ok = visible & np.isfinite(x) & np.isfinite(y)
    x = np.where(coord_ok, x, 0.0)
    y = np.where(coord_ok, y, 0.0)
    return x, y, masked_data


def _bin_edges(centers: np.ndarray) -> np.ndarray:
    """Compute `N + 1` bin edges from `N` monotonic bin centres.

    Interior edges are midpoints between consecutive centres; the two outer
    edges are extrapolated by the same half-step as their neighbouring
    interior edge.

    Raises:
        ValueError: If fewer than 2 centres are given (at least one interior
            gap is required to infer an edge spacing).
    """
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        raise ValueError(
            "at least 2 centres are required to infer bin edges, got "
            f"{centers.size}"
        )
    mid = (centers[:-1] + centers[1:]) / 2.0
    first = centers[0] - (mid[0] - centers[0])
    last = centers[-1] + (centers[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def orthographic_grid_edges(
    lon: Any,
    lat: Any,
    center_lat: float = 90.0,
    center_lon: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject 1D lon/lat cell-centre vectors to orthographic cell-*edge* coordinates.

    Companion to `orthographic_grid`: that function reprojects data at cell
    *centres*, matching `pcolormesh`'s `shading="auto"`/`"nearest"`
    convention -- but the orthographic projection's extreme local distortion
    (longitude lines compress drastically near the projection centre) makes
    matplotlib's automatic centre-to-edge inference unreliable there: most
    cells can render as degenerate slivers, silently dropping most of the
    grid. This function instead reprojects the cell **edges**, for use with
    `shading="flat"` (matplotlib draws exactly the given quads, inferring
    nothing) -- the reliable choice for this projection.

    Args:
        lon: 1D vector of cell-centre longitudes, degrees.
        lat: 1D vector of cell-centre latitudes, degrees.
        center_lat: Latitude the globe is centred on. Must match the value
            passed to `orthographic_grid` for the data these edges frame.
        center_lon: Longitude the globe is centred on. Must match
            `orthographic_grid`.

    Returns:
        tuple: `(x_edges, y_edges)`, each shaped `(len(lat) + 1, len(lon) + 1)`
        -- one more per axis than a `(len(lat), len(lon))` data array, ready
        for `ax.pcolormesh(x_edges, y_edges, data, shading="flat")`.

    Raises:
        ImportError: If pyproj (the `[tiles]` extra) is not installed.

    Warning:
        The orthographic projection is only finite on the visible (near)
        hemisphere -- an edge point beyond it has no real coordinate, so it
        is placed at a finite placeholder (not a meaningful position). A data
        cell whose *centre* is visible can still have a corner past the
        horizon; drawing it anyway pulls that corner toward the placeholder,
        producing a wrongly-shaped quad. Before drawing, drop (`NaN`) any
        cell for which not all four corners are visible -- `apply_projection_style`
        does this automatically; call it instead of this function directly
        unless you are prepared to replicate that check.

    Examples:
        - Edge arrays are one larger per axis than the centre vectors:
            ```python
            >>> import numpy as np
            >>> from cleopatra.projection import orthographic_grid_edges
            >>> lon = np.linspace(-180, 180, 8)
            >>> lat = np.linspace(-90, 90, 4)
            >>> x_edges, y_edges = orthographic_grid_edges(lon, lat)
            >>> x_edges.shape, y_edges.shape
            ((5, 9), (5, 9))

            ```

    See Also:
        orthographic_grid: The matching cell-centre reprojection for `data`.
        cleopatra.colors.alpha_scaled_mesh: Renders with `shading="flat"`.
    """
    lon_edges_1d = _bin_edges(np.asarray(lon, dtype=float))
    lat_edges_1d = _bin_edges(np.asarray(lat, dtype=float))
    lon_e, lat_e = np.meshgrid(lon_edges_1d, lat_edges_1d)
    dummy = np.zeros_like(lon_e)
    x_edges, y_edges, _ = orthographic_grid(
        lon_e, lat_e, dummy, center_lat=center_lat, center_lon=center_lon
    )
    return x_edges, y_edges


def orthographic_points(
    lon: Any,
    lat: Any,
    center_lat: float = 90.0,
    center_lon: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject scattered lon/lat points onto an orthographic ('globe') view.

    The point counterpart to `orthographic_grid`: use this for discrete
    locations -- e.g. city markers for `cleopatra.geo.add_point_labels` --
    rather than a raster grid. A globe's axes are scaled in projected
    metres (`ORTHOGRAPHIC_RADIUS_M`), not degrees, so plotting raw lon/lat
    values directly on a globe axes collapses every point toward the origin;
    reproject with this function first.

    Args:
        lon: Longitudes in degrees, scalar or 1D array.
        lat: Latitudes in degrees, scalar or 1D array, paired with `lon`.
        center_lat: Latitude the globe is centred on. Must match the value
            passed to `orthographic_grid`/`apply_projection_style` for the
            points to align with the data.
        center_lon: Longitude the globe is centred on. Must match
            `orthographic_grid`/`apply_projection_style`.

    Returns:
        tuple: `(x, y)`, the same shape as `lon`/`lat` (broadcast to at
        least 1D). A point on the far (non-visible) hemisphere is `NaN` in
        both `x` and `y` -- filter those out before plotting.

    Raises:
        ImportError: If pyproj (the `[tiles]` extra) is not installed.

    Examples:
        - Reproject two cities visible from a North-Pole-centred view; a
          point on the far side comes back `NaN`:
            ```python
            >>> import numpy as np
            >>> from cleopatra.projection import orthographic_points
            >>> lon = np.array([-21.9, 0.0])
            >>> lat = np.array([64.1, -80.0])
            >>> x, y = orthographic_points(lon, lat, center_lat=90.0, center_lon=0.0)
            >>> np.isnan(x[0])  # Reykjavik: visible
            np.False_
            >>> np.isnan(x[1])  # near the South Pole: not visible
            np.True_

            ```
        - A single scalar point round-trips as a 1-element array:
            ```python
            >>> from cleopatra.projection import orthographic_points
            >>> x, y = orthographic_points(0.0, 90.0)
            >>> round(float(x[0]), 6), round(float(y[0]), 6)
            (0.0, -0.0)

            ```

    See Also:
        orthographic_grid: The raster-grid counterpart.
        cleopatra.geo.add_point_labels: Renders the reprojected points.
    """
    _require_pyproj("Orthographic ('globe') point reprojection")
    from pyproj import Transformer

    lon_arr = np.atleast_1d(np.asarray(lon, dtype=float))
    lat_arr = np.atleast_1d(np.asarray(lat, dtype=float))
    transformer = Transformer.from_crs(
        "EPSG:4326", _ortho_proj4(center_lat, center_lon), always_xy=True
    )
    x_raw, y_raw = transformer.transform(lon_arr, lat_arr)
    x = np.asarray(x_raw, dtype=float)
    y = np.asarray(y_raw, dtype=float)
    visible = _visible_hemisphere(lon_arr, lat_arr, center_lat, center_lon)
    x = np.where(visible & np.isfinite(x), x, np.nan)
    y = np.where(visible & np.isfinite(y), y, np.nan)
    return x, y


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


#: Named "projection style" presets for `apply_projection_style` -- the
#: coordinate-frame half of the "haze" look (the glowing, ECMWF/CAMS-style
#: aerosol-animation aesthetic), independent of the
#: `cleopatra.colors.DATA_STYLES` colour/legend half. `"globe"` reprojects
#: onto an orthographic view and frames it with a boundary + graticule;
#: `"flat"` leaves the data in plain lon/lat and touches nothing on `ax`.
PROJECTION_STYLES: dict[str, dict[str, Any]] = {
    "globe": {"center_lat": 90.0, "center_lon": 0.0, "graticule_step": 30.0},
    "flat": {},
}


def apply_projection_style(
    ax: Axes,
    lon: Any,
    lat: Any,
    data: Any,
    style: str = "globe",
    *,
    draw_frame: bool = True,
    **overrides: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproject `(lon, lat, data)` and frame `ax` per a `PROJECTION_STYLES` preset.

    For `"globe"`, reprojects via `orthographic_grid` and draws the globe
    boundary + graticule on `ax` via `apply_projection_frame`. For `"flat"`,
    returns `(lon, lat, data)` unchanged and does not touch `ax` at all --
    the data is meant to be plotted directly in lon/lat coordinates.

    This is the projection half of composing the haze look; the colour half
    is `cleopatra.colors.apply_data_style`, called with the `(x, y, data)`
    this function returns as its `x`/`y`/layer arguments. Both styles return
    cell-**edge** coordinates (one larger per axis than `data`) for use with
    `shading="flat"`: the orthographic projection's extreme distortion near
    its centre makes matplotlib's automatic centre-to-edge inference
    (`shading="auto"`/`"nearest"`) unreliable -- most cells can render as
    degenerate slivers -- so edges are computed explicitly and reliably
    instead. The *same* two-line pattern composes both layers regardless of
    which style was chosen:

    ```python
    x, y, masked = apply_projection_style(ax, lon, lat, data, style=chosen)
    apply_data_style(ax, {"dust": masked}, x=x, y=y, shading="flat")
    ```

    Neither function requires the other: use `"globe"` with a single plain
    colormap instead of `apply_data_style`, or `"flat"` with the `"haze"`
    data style and no globe at all.

    This function takes a single `data` array, not a `layers` dict like
    `apply_data_style` -- drawing several layers on the *same* grid/axes
    means calling it once per layer. `apply_projection_frame` (which draws
    the boundary/graticule) is one-shot per axes: a second unguarded call
    stacks a duplicate boundary patch and graticule. Pass `draw_frame=False`
    on every call after the first to reproject/mask that layer's data
    without redrawing the chrome:

    ```python
    x, y, om = apply_projection_style(ax, lon, lat, organic_matter, style="globe")
    _, _, du = apply_projection_style(ax, lon, lat, dust, style="globe", draw_frame=False)
    ```

    Args:
        ax: Axes to draw the boundary/graticule on. Ignored for `"flat"`
            (which never draws) and when `draw_frame=False`.
        lon: 1D vector of cell-centre longitudes, degrees. Both styles
            require 1D `lon`/`lat` so cell edges can be computed reliably;
            for an already-2D curvilinear grid, call `orthographic_grid`
            directly and build your own edge coordinates.
        lat: 1D vector of cell-centre latitudes, degrees, paired with `lon`.
        data: 2D array of values, shape `(len(lat), len(lon))`. Reprojected
            and hemisphere-masked for `"globe"`; returned unchanged for
            `"flat"`.
        style: A name from `PROJECTION_STYLES` (`"globe"` or `"flat"`).
        draw_frame: If `True` (default), draw the boundary/graticule via
            `apply_projection_frame` (`"globe"` only -- `"flat"` never draws
            regardless). Pass `False` for every call after the first when
            drawing multiple layers on one already-framed axes.
        **overrides: Override any of the style's parameters -- for
            `"globe"`: `center_lat`, `center_lon`, `graticule_step`,
            `boundary_kw`, `graticule_kw` (the last two forwarded to
            `apply_projection_frame`).

    Returns:
        tuple: `(x_edges, y_edges, data)` -- cell-edge coordinates, each
        shaped `(len(lat) + 1, len(lon) + 1)`, and `data` unchanged (for
        `"flat"`) or hemisphere-masked (for `"globe"`).

    Raises:
        KeyError: If `style` is not registered.
        ValueError: If `lon`/`lat` are not 1D.
        ImportError: If `style="globe"` and pyproj (the `[tiles]` extra) is
            not installed.

    Examples:
        - `"globe"` reprojects the data, draws a boundary patch on `ax`, and
          returns edge coordinates one larger per axis than `data`:
            ```python
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.projection import apply_projection_style
            >>> fig, ax = plt.subplots()
            >>> lon = np.array([0.0, 90.0])
            >>> lat = np.array([90.0, -90.0])
            >>> data = np.array([[1.0, 2.0], [3.0, 4.0]])
            >>> x, y, masked = apply_projection_style(ax, lon, lat, data, style="globe")
            >>> len(ax.patches)  # the boundary circle
            1
            >>> x.shape  # one larger per axis than data's (2, 2)
            (3, 3)
            >>> np.all(np.isnan(masked[1]))  # far hemisphere still masked
            np.True_
            >>> plt.close(fig)

            ```
        - `"flat"` draws nothing but still returns matching edge coordinates,
          so the same `shading="flat"` call works for either style:
            ```python
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.projection import apply_projection_style
            >>> fig, ax = plt.subplots()
            >>> lon = np.array([0.0, 90.0])
            >>> lat = np.array([90.0, -90.0])
            >>> data = np.array([[1.0, 2.0], [3.0, 4.0]])
            >>> x, y, out = apply_projection_style(ax, lon, lat, data, style="flat")
            >>> len(ax.patches)
            0
            >>> x.shape
            (3, 3)
            >>> np.array_equal(out, data)
            True
            >>> plt.close(fig)

            ```
        - A second layer on the same globe with `draw_frame=False` reuses the
          already-drawn chrome instead of stacking a duplicate boundary:
            ```python
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.projection import apply_projection_style
            >>> fig, ax = plt.subplots()
            >>> lon = np.array([0.0, 90.0])
            >>> lat = np.array([90.0, -90.0])
            >>> first = np.array([[1.0, 2.0], [3.0, 4.0]])
            >>> second = np.array([[5.0, 6.0], [7.0, 8.0]])
            >>> _ = apply_projection_style(ax, lon, lat, first, style="globe")
            >>> _ = apply_projection_style(ax, lon, lat, second, style="globe", draw_frame=False)
            >>> len(ax.patches)  # still one boundary, not two
            1
            >>> plt.close(fig)

            ```
        - An unknown style raises `KeyError`:
            ```python
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.projection import apply_projection_style
            >>> fig, ax = plt.subplots()
            >>> apply_projection_style(
            ...     ax, np.array([0.0]), np.array([0.0]), np.array([[1.0]]),
            ...     style="not-a-style",
            ... )
            Traceback (most recent call last):
                ...
            KeyError: "Unknown projection style 'not-a-style'; available: ['flat', 'globe']"
            >>> plt.close(fig)

            ```

    See Also:
        orthographic_grid: The `"globe"` cell-centre reprojection primitive.
        orthographic_grid_edges: The `"globe"` cell-edge reprojection primitive.
        apply_projection_frame: Draws the boundary/graticule this composes.
        cleopatra.colors.apply_data_style: The companion data-style axis.
    """
    if style not in PROJECTION_STYLES:
        raise KeyError(
            f"Unknown projection style {style!r}; available: "
            f"{sorted(PROJECTION_STYLES)}"
        )
    cfg = {**PROJECTION_STYLES[style], **overrides}

    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    if lon_arr.ndim != 1 or lat_arr.ndim != 1:
        raise ValueError(
            "apply_projection_style requires 1D lon/lat vectors so cell "
            "edges can be computed reliably; for an already-2D curvilinear "
            "grid, call orthographic_grid directly and build your own edge "
            "coordinates."
        )

    if style == "flat":
        lon_edges_1d = _bin_edges(lon_arr)
        lat_edges_1d = _bin_edges(lat_arr)
        x_edges, y_edges = np.meshgrid(lon_edges_1d, lat_edges_1d)
        return x_edges, y_edges, np.asarray(data, dtype=float)

    center_lat = cfg.get("center_lat", 90.0)
    center_lon = cfg.get("center_lon", 0.0)
    graticule_step = cfg.get("graticule_step", 30.0)
    boundary_kw = cfg.get("boundary_kw")
    graticule_kw = cfg.get("graticule_kw")

    _, _, masked = orthographic_grid(
        lon_arr, lat_arr, data, center_lat=center_lat, center_lon=center_lon
    )
    x_edges, y_edges = orthographic_grid_edges(
        lon_arr, lat_arr, center_lat=center_lat, center_lon=center_lon
    )
    # A cell is only safe to draw if ALL FOUR of its corners are on the
    # visible hemisphere, not just its centre. The orthographic projection
    # is only defined (finite) on the near hemisphere -- an invisible corner
    # has no real coordinate, so orthographic_grid_edges places a finite
    # placeholder there. Without this extra check, a cell whose centre is
    # visible (so it *would* be drawn with real data) but whose corner
    # straddles the horizon renders as a wrongly-shaped quad reaching toward
    # that placeholder instead of tapering correctly at the boundary.
    lon_edges_1d = _bin_edges(lon_arr)
    lat_edges_1d = _bin_edges(lat_arr)
    lon_e, lat_e = np.meshgrid(lon_edges_1d, lat_edges_1d)
    edge_visible = _visible_hemisphere(lon_e, lat_e, center_lat, center_lon)
    corner_ok = (
        edge_visible[:-1, :-1]
        & edge_visible[:-1, 1:]
        & edge_visible[1:, :-1]
        & edge_visible[1:, 1:]
    )
    masked = np.where(corner_ok, masked, np.nan)
    if draw_frame:
        boundary = orthographic_boundary()
        graticule = orthographic_graticule(
            center_lat=center_lat, center_lon=center_lon, step=graticule_step
        )
        apply_projection_frame(
            ax,
            boundary_xy=boundary,
            xlim=(-ORTHOGRAPHIC_RADIUS_M, ORTHOGRAPHIC_RADIUS_M),
            ylim=(-ORTHOGRAPHIC_RADIUS_M, ORTHOGRAPHIC_RADIUS_M),
            graticule_lines=graticule,
            boundary_kw=boundary_kw,
            graticule_kw=graticule_kw,
        )
    return x_edges, y_edges, masked
