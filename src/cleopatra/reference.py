"""Reference-basemap backdrops for matplotlib axes.

Two axes-level helpers that draw public cartographic reference data
*underneath* your own plotted data -- the `cartopy`
`GeoAxes.stock_img()` / `ax.coastlines()` niche, and the vector/raster
sibling of `cleopatra.tiles.add_tiles`:

* `add_relief` -- a global hypsometric relief image (the
    `stock_img()` analogue).
* `add_features` -- a Natural Earth vector layer: coastline, borders,
    land, ocean, rivers, or lakes (the `coastlines()` analogue).

Both fetch a small, **fixed public dataset** that cleopatra re-hosts as a
dependency-light artifact, cache it on disk, and render it with
matplotlib. They acquire reference data only; they never read user files
and never touch GDAL/geopandas:

* Relief is re-hosted as a plain **PNG**, so decoding needs only Pillow
    (already part of the `cleopatra[tiles]` extra). Every relief product
    is a global EPSG:4326 raster, so its extent is hardcoded rather than
    read from a geotransform.
* Natural Earth layers are pre-converted (offline, maintainer-side) to
    gzipped **GeoJSON**, so reading them needs only the standard library
    (`json` + `gzip`) plus numpy. Drawing in EPSG:4326 needs nothing
    beyond matplotlib; reprojecting to another CRS lazily uses `pyproj`
    (also in the `[tiles]` extra).

The cache directory defaults to `~/.cleopatra/naturalearth` and can be
overridden with the `CLEOPATRA_CACHE_DIR` environment variable. Downloads
are restricted to `http(s)` URLs.

Examples:
    Draw a relief backdrop and a coastline over data plotted in
    lon/lat (EPSG:4326):

    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> from cleopatra.reference import add_relief, add_features
    >>> fig, ax = plt.subplots()
    >>> ax.set_xlim(-20, 40); ax.set_ylim(0, 60)  # doctest: +SKIP
    >>> _ = add_relief(ax, resolution="low")            # doctest: +SKIP
    >>> _ = add_features(ax, "coastline", "50m")        # doctest: +SKIP
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.path import Path as MplPath

from cleopatra import __version__

logger = logging.getLogger(__name__)

#: `User-Agent` sent on every reference-data download, identifying
#: cleopatra and its version so asset hosts can attribute the traffic.
USER_AGENT = f"cleopatra/{__version__} (+https://github.com/serapeum-org/cleopatra)"

_PILLOW_HINT = (
    "Relief backdrops require Pillow, provided by the [tiles] extra. "
    "Install with `pip install cleopatra[tiles]`."
)
#: True when Pillow is importable. Probed with `find_spec` so importing
#: this module never pulls Pillow in for callers who only use
#: `add_features` (which needs no third-party dependency).
_PILLOW_AVAILABLE = importlib.util.find_spec("PIL") is not None


# --------------------------------------------------------------------------
# Shared download / cache helpers
# --------------------------------------------------------------------------


def _cache_dir() -> Path:
    """Resolve (and create) the on-disk cache directory.

    Honours the `CLEOPATRA_CACHE_DIR` environment variable; defaults to
    `~/.cleopatra/naturalearth`.

    Returns:
        pathlib.Path: The existing cache directory.
    """
    root = os.environ.get("CLEOPATRA_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".cleopatra" / "naturalearth"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _download(
    url: str,
    dest: Path,
    *,
    timeout: int = 60,
    retries: int = 3,
    backoff: float = 0.5,
) -> Path:
    """Download `url` to `dest`, skipping the fetch if already cached.

    The body is streamed to a `.part` sibling and renamed onto `dest`
    only after a complete download, so an interrupted fetch never leaves
    a half-written cache file and the response is not buffered whole in
    memory. Transient failures (`URLError`/`OSError`, e.g. a dropped
    connection or a DNS hiccup) are retried with exponential backoff
    before giving up.

    Args:
        url: Source URL. Must use the `http` or `https` scheme.
        dest: Destination path in the cache.
        timeout: Per-request timeout in seconds.
        retries: Maximum number of attempts before raising, by default 3.
        backoff: Base delay in seconds between attempts, doubled after
            each failure (0.5, 1.0, 2.0, ...), by default 0.5.

    Returns:
        pathlib.Path: `dest`, now present on disk.

    Raises:
        ValueError: If `url` is not an `http(s)` URL.
        ConnectionError: If every attempt fails.
    """
    if dest.exists():
        return dest
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"Refusing to fetch non-http(s) URL: {url!r}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        logger.debug("Fetching reference asset %s (attempt %d/%d)", url, attempt + 1, retries)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with open(tmp, "wb") as handle:
                    shutil.copyfileobj(response, handle)
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, OSError) as e:
            last_error = e
            tmp.unlink(missing_ok=True)
            if attempt < retries - 1:
                time.sleep(backoff * (2**attempt))
    raise ConnectionError(
        f"Failed to download reference asset {url!r} after {retries} attempts: {last_error}"
    ) from last_error


def _validate_axes(ax: Any) -> None:
    """Raise `TypeError` if `ax` is not a matplotlib Axes.

    Args:
        ax: Candidate axes object.

    Raises:
        TypeError: If `ax` lacks the `get_xlim`/`get_ylim` interface.
    """
    if not hasattr(ax, "get_xlim") or not hasattr(ax, "get_ylim"):
        raise TypeError(
            f"ax must be a matplotlib.axes.Axes instance, got {type(ax).__name__}"
        )


# --------------------------------------------------------------------------
# Relief (raster backdrop)
# --------------------------------------------------------------------------

#: Global EPSG:4326 extent shared by every relief product
#: (west, south, east, north).
_RELIEF_EXTENT_4326 = (-180.0, -90.0, 180.0, 90.0)

#: Re-hosted on a cleopatra release as PNG (no GeoTIFF -> no GDAL).
_RELIEF_BASE_URL = (
    "https://github.com/serapeum-org/cleopatra/releases/download/basemap-data-v1/"
)

#: Relief resolution -> PNG asset name.
_RELIEF_PRODUCTS = {
    "low": "ne_hypso_rgb_720x360.png",  # ~0.5 deg
    "medium": "ne_hypso_rgb_1440x720.png",  # ~0.25 deg
}


def available_relief_resolutions() -> list[str]:
    """Return the relief resolutions that can be requested.

    Returns:
        list[str]: The valid `resolution` values for `relief` /
            `add_relief` (`"low"`, `"medium"`).

    Examples:
        ```python
        >>> from cleopatra.reference import available_relief_resolutions
        >>> available_relief_resolutions()
        ['low', 'medium']

        ```
    """
    return list(_RELIEF_PRODUCTS)


def relief(resolution: str = "low") -> np.ndarray:
    """Fetch (and cache) a hypsometric relief product as an RGB array.

    Args:
        resolution: `"low"` (720x360) or `"medium"` (1440x720).

    Returns:
        numpy.ndarray: An `(H, W, 3)` uint8 RGB array in EPSG:4326,
            north-up (row 0 is the northern edge).

    Raises:
        ImportError: If Pillow (the `[tiles]` extra) is not installed.
        ValueError: If `resolution` is not a known product.
        ConnectionError: If the asset must be downloaded and the fetch
            fails.
        OSError: If the cached file cannot be decoded as an image (the
            poisoned file is removed first so a retry re-downloads).

    Examples:
        - Unknown resolutions raise `ValueError` before any download:
            ```python
            >>> from cleopatra.reference import relief
            >>> relief("ultra")
            Traceback (most recent call last):
                ...
            ValueError: Unknown relief resolution 'ultra'. Choose from ['low', 'medium'].

            ```
    """
    if resolution not in _RELIEF_PRODUCTS:
        raise ValueError(
            f"Unknown relief resolution {resolution!r}. "
            f"Choose from {available_relief_resolutions()}."
        )
    if not _PILLOW_AVAILABLE:
        raise ImportError(_PILLOW_HINT)
    from PIL import Image, UnidentifiedImageError

    name = _RELIEF_PRODUCTS[resolution]
    path = _download(_RELIEF_BASE_URL + name, _cache_dir() / name)
    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError) as e:
        # A complete-but-unreadable cache file (e.g. an HTML error page
        # served with HTTP 200) would fail forever because `_download`
        # returns the cached file unconditionally. Drop it so the next
        # call re-fetches instead of staying poisoned.
        path.unlink(missing_ok=True)
        raise OSError(
            f"Cached relief asset {path} could not be decoded ({e}); removed "
            "it -- retry to re-download."
        ) from e


def add_relief(
    ax: Any,
    resolution: str = "low",
    *,
    extent: tuple[float, float, float, float] | None = None,
    alpha: float = 1.0,
    zorder: int = -1,
    interpolation: str = "bilinear",
) -> Any:
    """Draw a global hypsometric relief backdrop under existing data.

    The `cartopy` `GeoAxes.stock_img()` analogue. Assumes the axes are
    in EPSG:4326 (lon/lat); for data in another CRS pass `extent` in the
    axes' own units. The current axis limits are preserved so adding the
    backdrop never changes the view.

    Note:
        The relief image is equirectangular (EPSG:4326). When placed with
        a non-EPSG:4326 `extent` it is simply stretched by `imshow` to fit
        those bounds -- visually acceptable for small extents but not a
        true reprojection. For a pixel-accurate backdrop in another
        projection, reproject the source before drawing.

    Args:
        ax: A matplotlib `Axes` with data already plotted (so its limits
            define the view).
        resolution: `"low"` or `"medium"` (see
            `available_relief_resolutions`).
        extent: `(west, south, east, north)` placement in axis units.
            Defaults to the global EPSG:4326 extent
            `(-180, -90, 180, 90)`.
        alpha: Backdrop opacity in `[0, 1]`.
        zorder: Matplotlib draw order (`-1` puts it behind all data).
        interpolation: Interpolation passed to `ax.imshow`.

    Returns:
        matplotlib.axes.Axes: The same axes, for chaining.

    Raises:
        ImportError: If Pillow (the `[tiles]` extra) is not installed.
        TypeError: If `ax` is not a matplotlib Axes.
        ValueError: If `resolution` is unknown.
        ConnectionError: If the asset must be downloaded and the fetch
            fails.

    Examples:
        - Draw a relief backdrop under data already plotted in lon/lat
            (downloads the asset on first use, then caches it):
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.reference import add_relief
            >>> fig, ax = plt.subplots()
            >>> ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)  # doctest: +SKIP
            >>> ax = add_relief(ax, "low")  # doctest: +SKIP
            >>> len(ax.images)  # doctest: +SKIP
            1

            ```
        - Unknown resolutions are rejected before any download:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.reference import add_relief
            >>> fig, ax = plt.subplots()
            >>> add_relief(ax, "high")
            Traceback (most recent call last):
                ...
            ValueError: Unknown relief resolution 'high'. Choose from ['low', 'medium'].

            ```
    """
    _validate_axes(ax)
    rgb = relief(resolution)
    west, south, east, north = extent if extent is not None else _RELIEF_EXTENT_4326

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.imshow(
        rgb,
        extent=[west, east, south, north],
        origin="upper",
        alpha=alpha,
        zorder=zorder,
        interpolation=interpolation,
        aspect=ax.get_aspect(),
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


# --------------------------------------------------------------------------
# Features (Natural Earth vectors)
# --------------------------------------------------------------------------

#: layer -> (Natural Earth dataset stem, geometry kind). The stem is the
#: artifact filename component; the kind selects the matplotlib collection
#: and default styling.
_LAYERS = {
    "coastline": ("coastline", "line"),
    "land": ("land", "polygon"),
    "ocean": ("ocean", "polygon"),
    "rivers": ("rivers_lake_centerlines", "line"),
    "lakes": ("lakes", "polygon"),
    "borders": ("admin_0_boundary_lines_land", "line"),
}
_RESOLUTIONS = ("110m", "50m", "10m")

#: Preprocessed (gzipped GeoJSON) layers re-hosted on a cleopatra release.
_FEATURE_BASE_URL = _RELIEF_BASE_URL

#: Per-kind default styling, merged under caller `**style` overrides. Keys
#: match the target collection's constructor (`PathCollection` /
#: `LineCollection`).
_FEATURE_STYLE = {
    "line": {"colors": "0.2", "linewidths": 0.6},
    "polygon": {"facecolors": "0.85", "edgecolors": "0.4", "linewidths": 0.4},
}


def available_layers() -> list[str]:
    """Return the Natural Earth layers that can be requested.

    Returns:
        list[str]: Valid `layer` names for `natural_earth` /
            `add_features`.

    Examples:
        ```python
        >>> from cleopatra.reference import available_layers
        >>> available_layers()
        ['coastline', 'land', 'ocean', 'rivers', 'lakes', 'borders']

        ```
    """
    return list(_LAYERS)


def available_resolutions() -> list[str]:
    """Return the supported Natural Earth resolutions.

    Returns:
        list[str]: `["110m", "50m", "10m"]`.

    Examples:
        ```python
        >>> from cleopatra.reference import available_resolutions
        >>> available_resolutions()
        ['110m', '50m', '10m']

        ```
    """
    return list(_RESOLUTIONS)


def _paths(geometry: dict) -> list[np.ndarray]:
    """Flatten a GeoJSON geometry into a list of `(N, 2)` float arrays.

    Polygons contribute their exterior ring only (interior holes are
    dropped). This is the coordinate view used by `natural_earth` and the
    line-layer renderer; the filled-polygon renderer uses `_polygons`,
    which keeps holes. `Multi*` geometries expand to one array per part.

    Args:
        geometry: A GeoJSON geometry object with `type` and
            `coordinates`.

    Returns:
        list[numpy.ndarray]: One `(N, 2)` coordinate array per part.

    Raises:
        ValueError: If the geometry type is not a (Multi)LineString or
            (Multi)Polygon.

    Examples:
        - A polygon yields its exterior ring; holes are ignored:
            ```python
            >>> from cleopatra.reference import _paths
            >>> geom = {
            ...     "type": "Polygon",
            ...     "coordinates": [
            ...         [[0, 0], [2, 0], [2, 2], [0, 0]],
            ...         [[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 0.5]],
            ...     ],
            ... }
            >>> [p.shape for p in _paths(geom)]
            [(4, 2)]

            ```
        - A MultiLineString expands to one array per line:
            ```python
            >>> from cleopatra.reference import _paths
            >>> geom = {
            ...     "type": "MultiLineString",
            ...     "coordinates": [[[0, 0], [1, 1]], [[2, 2], [3, 3], [4, 4]]],
            ... }
            >>> [p.shape for p in _paths(geom)]
            [(2, 2), (3, 2)]

            ```
    """
    gtype = geometry["type"]
    coords = geometry["coordinates"]
    if gtype == "LineString":
        return [np.asarray(coords, dtype=float)]
    if gtype == "MultiLineString":
        return [np.asarray(line, dtype=float) for line in coords]
    if gtype == "Polygon":
        return [np.asarray(coords[0], dtype=float)]
    if gtype == "MultiPolygon":
        return [np.asarray(poly[0], dtype=float) for poly in coords]
    raise ValueError(f"Unsupported geometry type: {gtype!r}")


def _polygons(geometry: dict) -> list[list[np.ndarray]]:
    """Return a GeoJSON geometry's polygons, each as `[exterior, *holes]`.

    Unlike `_paths`, interior rings (holes) are preserved so a filled
    polygon can be cut out correctly. Non-polygon geometries return an
    empty list.

    Args:
        geometry: A GeoJSON geometry object.

    Returns:
        list[list[numpy.ndarray]]: One entry per polygon part; each entry
            is `[exterior_ring, hole_ring, ...]` of `(N, 2)` arrays.

    Examples:
        - A polygon with one hole keeps both rings:
            ```python
            >>> from cleopatra.reference import _polygons
            >>> geom = {
            ...     "type": "Polygon",
            ...     "coordinates": [
            ...         [[0, 0], [2, 0], [2, 2], [0, 0]],
            ...         [[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 0.5]],
            ...     ],
            ... }
            >>> [len(p) for p in _polygons(geom)]
            [2]

            ```
    """
    gtype = geometry["type"]
    coords = geometry["coordinates"]
    if gtype == "Polygon":
        return [[np.asarray(ring, dtype=float) for ring in coords]]
    if gtype == "MultiPolygon":
        return [[np.asarray(ring, dtype=float) for ring in poly] for poly in coords]
    return []


def _signed_area(ring: np.ndarray) -> float:
    """Return the signed area of a ring (positive when counter-clockwise)."""
    x = ring[:, 0]
    y = ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _orient(ring: np.ndarray, ccw: bool) -> np.ndarray:
    """Return `ring` wound counter-clockwise when `ccw`, else clockwise.

    Source GeoJSON winding is not guaranteed (GDAL does not enforce RFC
    7946), so exterior rings are forced CCW and holes CW. With opposite
    windings the nonzero fill rule cuts the holes out of the exterior.
    """
    if (_signed_area(ring) > 0) != ccw:
        return ring[::-1]
    return ring


def _finite_runs(arr: np.ndarray) -> list[np.ndarray]:
    """Split `arr` into maximal runs of finite `(x, y)` rows (length >= 2).

    Reprojection (`crs=`) can map points at a projection's singularity
    (e.g. the poles in Web Mercator) to non-finite values; emitting those
    to matplotlib produces broken paths and warnings. Splitting keeps the
    finite spans and drops the non-finite breaks.
    """
    if arr.size == 0:
        return []
    mask = np.isfinite(arr).all(axis=1)
    if mask.all():
        return [arr] if len(arr) >= 2 else []
    runs: list[np.ndarray] = []
    start: int | None = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= 2:
                runs.append(arr[start:i])
            start = None
    if start is not None and len(arr) - start >= 2:
        runs.append(arr[start:])
    return runs


def _load_features(layer: str, resolution: str) -> list[dict]:
    """Download/cache a layer and return its non-null GeoJSON geometries.

    Args:
        layer: One of `available_layers()`.
        resolution: One of `available_resolutions()`.

    Returns:
        list[dict]: The GeoJSON geometry objects of every feature that has
            one.

    Raises:
        ValueError: If `layer` or `resolution` is unknown.
        ConnectionError: If the asset must be downloaded and the fetch
            fails.
        OSError: If the cached file cannot be parsed (the poisoned file is
            removed first so a retry re-downloads).
    """
    if layer not in _LAYERS:
        raise ValueError(f"Unknown layer {layer!r}. Choose from {available_layers()}.")
    if resolution not in _RESOLUTIONS:
        raise ValueError(
            f"Unknown resolution {resolution!r}. "
            f"Choose from {available_resolutions()}."
        )
    stem = _LAYERS[layer][0]
    name = f"ne_{resolution}_{stem}.geojson.gz"
    path = _download(_FEATURE_BASE_URL + name, _cache_dir() / name)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            collection = json.load(fh)
        features = collection["features"]
        return [
            feature["geometry"]
            for feature in features
            if feature.get("geometry") is not None
        ]
    except (OSError, EOFError, ValueError, KeyError, TypeError) as e:
        # Corrupt/poisoned cache: truncated gzip, HTML error page, bad JSON,
        # or valid JSON that is not a GeoJSON FeatureCollection (missing
        # "features", wrong shape). Drop it so the next call re-fetches
        # rather than failing forever on the cached bytes.
        path.unlink(missing_ok=True)
        raise OSError(
            f"Cached layer asset {path} could not be parsed ({e}); removed "
            "it -- retry to re-download."
        ) from e


def natural_earth(
    layer: str = "coastline", resolution: str = "110m"
) -> list[np.ndarray]:
    """Fetch (and cache) a Natural Earth layer as coordinate arrays.

    The layer is downloaded as preprocessed gzipped GeoJSON and parsed
    with the standard library only -- no GDAL/geopandas. Coordinates are
    EPSG:4326 lon/lat. Polygon layers return exterior rings only; use
    `add_features` for hole-aware filled rendering.

    Args:
        layer: One of `available_layers()`.
        resolution: One of `available_resolutions()`
            (`"110m"`/`"50m"`/`"10m"`).

    Returns:
        list[numpy.ndarray]: One `(N, 2)` lon/lat array per geometry part
            (exterior rings for polygon layers).

    Raises:
        ValueError: If `layer` or `resolution` is unknown.
        ConnectionError: If the asset must be downloaded and the fetch
            fails.

    Examples:
        - Fetch coastlines and inspect the parts (downloads on first use,
            then reads from the cache):
            ```python
            >>> from cleopatra.reference import natural_earth
            >>> parts = natural_earth("coastline", "110m")  # doctest: +SKIP
            >>> parts[0].shape[1]  # each part is an (N, 2) lon/lat array  # doctest: +SKIP
            2

            ```
        - Unknown layers are rejected before any download:
            ```python
            >>> from cleopatra.reference import natural_earth
            >>> natural_earth("countries")
            Traceback (most recent call last):
                ...
            ValueError: Unknown layer 'countries'. Choose from ['coastline', 'land', 'ocean', 'rivers', 'lakes', 'borders'].

            ```
    """
    parts: list[np.ndarray] = []
    for geometry in _load_features(layer, resolution):
        parts.extend(_paths(geometry))
    return parts


def _is_4326(crs: int | str | None) -> bool:
    """Return True when `crs` denotes EPSG:4326 (or is unspecified)."""
    if crs is None:
        return True
    return str(crs).upper().replace("EPSG:", "") == "4326"


def _make_transformer(crs: int | str) -> Any:
    """Build a pyproj transformer from EPSG:4326 to `crs`.

    Args:
        crs: Target CRS -- an int EPSG code or a CRS string/WKT.

    Returns:
        pyproj.Transformer: An `always_xy` transformer.

    Raises:
        ImportError: If `pyproj` (the `[tiles]` extra) is not installed.
        ValueError: If `crs` cannot be interpreted as a CRS.
    """
    if importlib.util.find_spec("pyproj") is None:
        raise ImportError(
            "Reprojecting reference features to a non-EPSG:4326 CRS requires "
            "pyproj, provided by the [tiles] extra. Install with "
            "`pip install cleopatra[tiles]`, or plot your data in EPSG:4326."
        )
    from pyproj import Transformer
    from pyproj.exceptions import CRSError

    dst = f"EPSG:{crs}" if isinstance(crs, int) else str(crs)
    try:
        return Transformer.from_crs("EPSG:4326", dst, always_xy=True)
    except CRSError as e:
        raise ValueError(
            f"Invalid CRS {crs!r}: {e}. Provide a valid EPSG code or CRS " "string."
        ) from e


def _reproject_arr(arr: np.ndarray, transformer: Any) -> np.ndarray:
    """Reproject an `(N, 2)` lon/lat array with `transformer`."""
    x, y = transformer.transform(arr[:, 0], arr[:, 1])
    return np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])


def _line_segments(geoms: list[dict], transformer: Any) -> list[np.ndarray]:
    """Build LineCollection segments from line geometries.

    Each part is reprojected (when `transformer` is set) and split into
    finite runs so projection singularities do not introduce broken
    segments.
    """
    segments: list[np.ndarray] = []
    for geometry in geoms:
        for part in _paths(geometry):
            if transformer is not None:
                part = _reproject_arr(part, transformer)
            segments.extend(_finite_runs(part))
    return segments


def _polygon_paths(geoms: list[dict], transformer: Any) -> list[MplPath]:
    """Build hole-aware compound `Path`s from polygon geometries.

    Each polygon becomes one compound path: the exterior ring forced
    counter-clockwise and every hole clockwise, so the nonzero fill rule
    renders the holes as cut-outs. Rings are reprojected (when
    `transformer` is set) and stripped of non-finite vertices first.
    """
    paths: list[MplPath] = []
    for geometry in geoms:
        for polygon in _polygons(geometry):
            oriented: list[np.ndarray] = []
            for index, ring in enumerate(polygon):
                if transformer is not None:
                    ring = _reproject_arr(ring, transformer)
                ring = ring[np.isfinite(ring).all(axis=1)]
                if len(ring) >= 3:
                    oriented.append(_orient(ring, ccw=(index == 0)))
            path = _compound_path(oriented)
            if path is not None:
                paths.append(path)
    return paths


def _compound_path(rings: list[np.ndarray]) -> MplPath | None:
    """Assemble `[exterior, *holes]` rings into one compound `Path`.

    Returns None when there is no drawable exterior ring.
    """
    if not rings:
        return None
    verts: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    for ring in rings:
        # MOVETO, LINETO..., CLOSEPOLY. The explicit CLOSEPOLY (with a
        # placeholder vertex) is what makes the nonzero fill rule treat the
        # ring as a closed contour, so an oppositely-wound hole is cut out
        # rather than filled solid.
        n = len(ring)
        ring_codes = np.full(n + 1, MplPath.LINETO, dtype=np.uint8)
        ring_codes[0] = MplPath.MOVETO
        ring_codes[-1] = MplPath.CLOSEPOLY
        verts.append(np.vstack([ring, ring[0]]))
        codes.append(ring_codes)
    return MplPath(np.concatenate(verts), np.concatenate(codes))


def add_features(
    ax: Any,
    layer: str = "coastline",
    resolution: str = "110m",
    *,
    crs: int | str | None = None,
    zorder: int = 0,
    **style: Any,
) -> Any:
    """Draw a Natural Earth reference layer on an Axes.

    The `cartopy` `ax.coastlines()` analogue. Polygon layers
    (`land`/`ocean`/`lakes`) are drawn hole-aware as a filled
    `PathCollection` (so `ocean`'s continent cut-outs and islands-in-lakes
    render correctly); line layers (`coastline`/`rivers`/`borders`) as a
    `LineCollection`. The source data is EPSG:4326; pass `crs` to reproject
    the geometry into the axes' CRS (requires `pyproj`). The current axis
    limits are preserved.

    Args:
        ax: A matplotlib `Axes` with data already plotted.
        layer: One of `available_layers()`.
        resolution: One of `available_resolutions()`.
        crs: CRS of the data on `ax`. `None` or `4326`/`"EPSG:4326"`
            draws the lon/lat coordinates directly; any other value
            reprojects from EPSG:4326 first.
        zorder: Matplotlib draw order for the layer.
        **style: Overrides merged over the per-kind defaults and
            forwarded to the underlying collection. Use polygon keys for
            polygon layers (`facecolors`, `edgecolors`, `linewidths`) and
            line keys for line layers (`colors`, `linewidths`).

    Returns:
        matplotlib.axes.Axes: The same axes, for chaining.

    Raises:
        TypeError: If `ax` is not a matplotlib Axes.
        ValueError: If `layer` or `resolution` is unknown, or `crs` is not
            a valid CRS.
        ImportError: If `crs` requires reprojection but `pyproj` is not
            installed.
        ConnectionError: If the asset must be downloaded and the fetch
            fails.

    Examples:
        - Overlay a coastline and country borders on a lon/lat map
            (downloads each layer on first use, then caches it):
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.reference import add_features
            >>> fig, ax = plt.subplots()
            >>> ax.set_xlim(-20, 40); ax.set_ylim(0, 60)  # doctest: +SKIP
            >>> ax = add_features(ax, "coastline", "50m", colors="navy")  # doctest: +SKIP
            >>> ax = add_features(ax, "borders", "50m")  # doctest: +SKIP

            ```
        - Unknown layers are rejected before any download:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.reference import add_features
            >>> fig, ax = plt.subplots()
            >>> add_features(ax, "countries")
            Traceback (most recent call last):
                ...
            ValueError: Unknown layer 'countries'. Choose from ['coastline', 'land', 'ocean', 'rivers', 'lakes', 'borders'].

            ```
    """
    _validate_axes(ax)
    geoms = _load_features(layer, resolution)
    transformer = None if _is_4326(crs) else _make_transformer(crs)  # type: ignore[arg-type]

    kind = _LAYERS[layer][1]
    opts = {**_FEATURE_STYLE[kind], **style}

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    collection: Any
    if kind == "polygon":
        collection = PathCollection(
            _polygon_paths(geoms, transformer), zorder=zorder, **opts
        )
        # PathCollection paths default to a non-data transform (its scatter
        # heritage); place them in data coordinates explicitly.
        collection.set_transform(ax.transData)
    else:
        collection = LineCollection(
            _line_segments(geoms, transformer), zorder=zorder, **opts
        )
    ax.add_collection(collection)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax
