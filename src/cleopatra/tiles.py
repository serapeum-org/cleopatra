"""Web-tile basemap helper for matplotlib axes.

Provides `add_tiles` -- a single entry point that fetches XYZ web
tiles for the current axes extent, stitches them into a composite image
with Pillow, and renders the image underneath the existing data layer.

The implementation is a pure-Python port of the `pyramids.basemap`
module (basemap.py + tiles.py). It supports any XYZ provider listed in
`xyzservices`. CRS handling is done with `pyproj` -- there is
no GDAL dependency, so the module is safe to use in environments that
only have matplotlib + numpy installed.

Notes:
    For data in CRSes other than Web Mercator (EPSG:3857) the stitched tile
    image is placed at the mosaic's own coverage: its Web-Mercator bounds
    are reprojected (with edge densification) into the target CRS and used
    as the `imshow` extent, while the axis limits stay at the data bounds.
    This aligns the basemap with the data even when the fetched tiles cover
    a tile-snapped area larger than the data. A residual Mercator-vs-linear
    nonlinearity remains for very large extents (the Mercator pixels are
    placed on a linear axis); if pixel-accurate warping is required,
    reproject the source data to Web Mercator (EPSG:3857) before plotting.

Examples:
    Add a default OpenStreetMap basemap to an axes that already has
    data plotted in Web Mercator coordinates:

    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> _ = ax.plot([1000000.0, 1200000.0], [6000000.0, 6200000.0])
    >>> _ = add_tiles(ax, source=None, crs=3857)  # doctest: +SKIP
"""

from __future__ import annotations

import html
import importlib.util
import io
import logging
import math
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from cleopatra import __version__

logger = logging.getLogger(__name__)


#: Default `User-Agent` sent on every tile request. Includes the cleopatra
#: version and a contact URL so tile providers (OpenStreetMap in particular,
#: whose usage policy requires an identifiable agent) can attribute and, if
#: necessary, throttle or reach out about traffic. Override per call via the
#: `user_agent` argument of `add_tiles`.
USER_AGENT = f"cleopatra/{__version__} (+https://github.com/serapeum-org/cleopatra)"

MAX_TILES = 256

_TILES_EXTRA_HINT = (
    "Web-tile basemap support requires the [tiles] extra. "
    "Install with `pip install cleopatra[tiles]`."
)

#: Import names of the packages the `[tiles]` extra provides ("PIL" is
#: Pillow's import name).
_TILES_EXTRA_MODULES = ("mercantile", "xyzservices", "PIL", "pyproj")

#: True when every `[tiles]`-extra dependency is importable. Probed with
#: `importlib.util.find_spec` so importing this module does not pull
#: those packages in when nobody calls `add_tiles`; the modules are
#: imported lazily inside the functions that use them.
_TILES_AVAILABLE = all(
    importlib.util.find_spec(_m) is not None for _m in _TILES_EXTRA_MODULES
)


def _require_tiles_extra() -> None:
    """Raise `ImportError` if the tiles extra is not installed.

    Returns:
        None

    Raises:
        ImportError: If any of mercantile, xyzservices, Pillow, or
            pyproj is missing from the active environment.
    """
    if not _TILES_AVAILABLE:
        raise ImportError(_TILES_EXTRA_HINT)


def get_provider(name: str | None = None) -> Any:
    """Resolve an XYZ tile provider by name.

    Args:
        name: Dot-separated provider name (e.g. `"OpenStreetMap.Mapnik"`,
            `"CartoDB.Positron"`, `"Esri.WorldImagery"`). `None`
            returns the default (`OpenStreetMap.Mapnik`).

    Returns:
        xyzservices.TileProvider: The resolved tile provider with
        `url` template and `attribution` metadata.

    Raises:
        ImportError: If the `[tiles]` extra is not installed.
        ValueError: If the provider name cannot be resolved.

    Examples:
        - Resolve the default OpenStreetMap provider and inspect its
            URL template:
            ```python
            >>> from cleopatra.tiles import get_provider
            >>> provider = get_provider()
            >>> provider.name
            'OpenStreetMap.Mapnik'
            >>> "{z}" in provider.url and "{x}" in provider.url
            True

            ```
        - Resolve a named provider via dot-path syntax:
            ```python
            >>> from cleopatra.tiles import get_provider
            >>> provider = get_provider("CartoDB.Positron")
            >>> provider.name
            'CartoDB.Positron'

            ```
        - Invalid provider names raise `ValueError`:
            ```python
            >>> from cleopatra.tiles import get_provider
            >>> get_provider("NonExistent.Provider")
            Traceback (most recent call last):
                ...
            ValueError: Unknown tile provider: 'NonExistent.Provider'. Failed at 'NonExistent'. Use xyzservices.providers to list available providers.

            ```
    """
    _require_tiles_extra()
    import xyzservices.providers as xyz

    provider: Any
    if name is None:
        provider = xyz.OpenStreetMap.Mapnik
    else:
        parts = name.split(".")
        provider = xyz
        for part in parts:
            try:
                provider = provider[part]
            except (KeyError, TypeError) as e:
                raise ValueError(
                    f"Unknown tile provider: '{name}'. "
                    f"Failed at '{part}'. Use "
                    f"xyzservices.providers to list "
                    f"available providers."
                ) from e
    return provider


def auto_zoom(
    bounds_4326: tuple[float, float, float, float],
    min_tiles_across: int = 2,
) -> int:
    """Compute a default zoom level for the given bounds in EPSG:4326.

    Picks the smallest zoom at which the larger of the two extents spans at
    least `min_tiles_across` tiles, i.e.
    `zoom = ceil(log2(min_tiles_across * 360 / max(lon_extent, lat_extent)))`,
    clamped to 0--19. The `min_tiles_across` floor (default 2) stops a
    mid-range regional extent from collapsing onto a single coarse tile
    stretched over the whole area (a 6--11 degree window would otherwise
    fetch just 2 tiles); `min_tiles_across=1` reproduces the older
    one-tile-across heuristic.

    This is a coarse heuristic that treats degrees of longitude and
    latitude as interchangeable; it does **not** account for Web
    Mercator's latitude distortion, so the result tends to be
    conservative (under-zoomed) for extents far from the equator. For
    high-latitude data, pass an explicit `zoom=` to `add_tiles`
    rather than relying on the auto value. The `MAX_TILES` cap in
    `add_tiles` will still step the zoom back down if the chosen
    level would require too many tiles.

    Args:
        bounds_4326: `(west, south, east, north)` in EPSG:4326 degrees.
        min_tiles_across: Minimum number of tiles the larger extent should
            span; higher values pick a sharper (higher) zoom. Values below 1
            are clamped to 1 (the older one-tile-across heuristic). Defaults
            to 2.

    Returns:
        int: Zoom level between 0 and 19.

    Examples:
        - Worldwide extent maps to zoom 1 (two tiles across the globe):
            ```python
            >>> from cleopatra.tiles import auto_zoom
            >>> auto_zoom((-180, -85, 180, 85))
            1

            ```
        - A 0.6 by 0.2 degree window over Berlin yields zoom 11:
            ```python
            >>> from cleopatra.tiles import auto_zoom
            >>> auto_zoom((13.0, 52.4, 13.6, 52.6))
            11

            ```
        - `min_tiles_across=1` restores the older, coarser one-tile
            heuristic (worldwide -> zoom 0):
            ```python
            >>> from cleopatra.tiles import auto_zoom
            >>> auto_zoom((-180, -85, 180, 85), min_tiles_across=1)
            0

            ```
        - Tiny extents are clamped to the maximum zoom (19):
            ```python
            >>> from cleopatra.tiles import auto_zoom
            >>> auto_zoom((0.0, 0.0, 1e-9, 1e-9))
            19

            ```
    """
    west, south, east, north = bounds_4326
    lon_extent = abs(east - west)
    lat_extent = abs(north - south)
    max_extent = max(lon_extent, lat_extent, 1e-10)
    across = max(1, min_tiles_across)
    zoom = math.ceil(math.log2(across * 360.0 / max_extent))
    result = max(0, min(zoom, 19))
    return result


def _densify_and_reproject_bounds(
    west: float,
    south: float,
    east: float,
    north: float,
    src_crs: str,
    dst_crs: str,
    n_points: int = 21,
) -> tuple[float, float, float, float]:
    """Reproject a bounding box with edge densification.

    Samples points along all four edges of the box before reprojecting,
    then takes the min/max of the reprojected points. This handles
    non-conformal projections where corners alone would underestimate
    the true extent.

    Args:
        west: Western bound in the source CRS.
        south: Southern bound in the source CRS.
        east: Eastern bound in the source CRS.
        north: Northern bound in the source CRS.
        src_crs: Source CRS identifier (e.g. `"EPSG:4326"`).
        dst_crs: Target CRS identifier (e.g. `"EPSG:3857"`).
        n_points: Number of sample points per edge. 21 balances
            accuracy and performance for typical warps.

    Returns:
        tuple[float, float, float, float]: `(west, south, east, north)`
        in the target CRS.

    Raises:
        ValueError: If the reprojection produces infinite or NaN
            coordinates.

    Examples:
        - Reproject a small bounding box from EPSG:4326 to EPSG:3857
            (Web Mercator) and verify the bounds are sane:
            ```python
            >>> from cleopatra.tiles import _densify_and_reproject_bounds
            >>> bounds = _densify_and_reproject_bounds(
            ...     13.0, 52.4, 13.6, 52.6, "EPSG:4326", "EPSG:3857"
            ... )
            >>> w, s, e, n = bounds
            >>> w < e and s < n
            True
            >>> int(round(w)), int(round(s))
            (1447153, 6872776)

            ```
        - The same box round-tripped back to EPSG:4326 recovers the
            input bounds:
            ```python
            >>> from cleopatra.tiles import _densify_and_reproject_bounds
            >>> w, s, e, n = _densify_and_reproject_bounds(
            ...     13.0, 52.4, 13.6, 52.6, "EPSG:4326", "EPSG:3857"
            ... )
            >>> w2, s2, e2, n2 = _densify_and_reproject_bounds(
            ...     w, s, e, n, "EPSG:3857", "EPSG:4326"
            ... )
            >>> round(w2, 4), round(s2, 4), round(e2, 4), round(n2, 4)
            (13.0, 52.4, 13.6, 52.6)

            ```
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    t = np.linspace(0, 1, n_points)
    xs = np.concatenate(
        [
            west + t * (east - west),
            np.full_like(t, east),
            east - t * (east - west),
            np.full_like(t, west),
        ]
    )
    ys = np.concatenate(
        [
            np.full_like(t, south),
            south + t * (north - south),
            np.full_like(t, north),
            north - t * (north - south),
        ]
    )

    tx, ty = transformer.transform(xs, ys)

    tx_arr = np.asarray(tx)
    ty_arr = np.asarray(ty)
    if not (np.all(np.isfinite(tx_arr)) and np.all(np.isfinite(ty_arr))):
        raise ValueError(
            f"CRS reprojection from {src_crs} to {dst_crs} produced "
            f"infinite or NaN coordinates. The data extent may be "
            f"outside the valid domain for this CRS transformation."
        )

    result = (
        float(tx_arr.min()),
        float(ty_arr.min()),
        float(tx_arr.max()),
        float(ty_arr.max()),
    )
    return result


def _looks_like_image(data: bytes) -> bool:
    """Return True if `data` begins with a known raster-image signature.

    Recognises PNG, JPEG (any APPn / SOFn / DQT variant via the 3-byte
    SOI marker `\\xff\\xd8\\xff`), GIF, and WebP — the formats XYZ tile
    servers serve. This is only a cheap "is this an image at all?" gate
    before Pillow does the real decoding in `stitch_tiles`; it is
    not a full validation.

    Args:
        data: The raw HTTP response body.

    Returns:
        bool: True if the leading bytes match a known image format.

    Examples:
        - A PNG header passes; an HTML error page does not:
            ```python
            >>> from cleopatra.tiles import _looks_like_image
            >>> _looks_like_image(b"\\x89PNG\\r\\n\\x1a\\n" + b"\\x00" * 8)
            True
            >>> _looks_like_image(b"<html>error</html>")
            False
            >>> _looks_like_image(b"")
            False

            ```
        - JPEGs with any APPn marker (`\\xe0`..`\\xef`) or a bare SOI
            followed by a DQT/SOFn byte pass:
            ```python
            >>> from cleopatra.tiles import _looks_like_image
            >>> all(
            ...     _looks_like_image(b"\\xff\\xd8\\xff" + bytes([m]) + b"\\x00" * 8)
            ...     for m in (0xE0, 0xE1, 0xE2, 0xE8, 0xEF, 0xDB, 0xC0)
            ... )
            True

            ```
    """
    if not data:
        return False
    return (
        data[:4] == b"\x89PNG"  # PNG
        or data[:3] == b"\xff\xd8\xff"  # JPEG (baseline, progressive, EXIF, ...)
        or data[:6] in (b"GIF87a", b"GIF89a")  # GIF
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")  # WebP
    )


def fetch_single_tile(
    tile: Any,
    provider: Any,
    timeout: int,
    retries: int,
    user_agent: str = USER_AGENT,
) -> tuple[Any, bytes]:
    """Fetch a single tile, retrying on transient failures.

    Args:
        tile: Tile to fetch (has `x`, `y`, `z` attributes).
        provider: `xyzservices.TileProvider` with a URL template.
        timeout: HTTP request timeout in seconds.
        retries: Number of retry attempts on failure.
        user_agent: `User-Agent` header sent on every request. Defaults
            to `USER_AGENT` (`cleopatra/<version> (+repo-url)`).

    Returns:
        tuple[Any, bytes]: The original tile and its PNG/JPEG bytes.

    Raises:
        ConnectionError: If the tile cannot be fetched after all
            retries are exhausted.

    Examples:
        - Fetch a single OpenStreetMap tile (network-dependent, hence
            skipped under doctest):
            ```python
            >>> import mercantile
            >>> from cleopatra.tiles import fetch_single_tile, get_provider
            >>> tile = mercantile.Tile(0, 0, 0)
            >>> provider = get_provider("OpenStreetMap.Mapnik")
            >>> tile_obj, data = fetch_single_tile(  # doctest: +SKIP
            ...     tile, provider, timeout=10, retries=2
            ... )
            >>> from cleopatra.tiles import _looks_like_image
            >>> _looks_like_image(data)  # doctest: +SKIP
            True

            ```
        - Tile failures raise `ConnectionError` after retries
            are exhausted:
            ```python
            >>> import mercantile
            >>> from cleopatra.tiles import fetch_single_tile
            >>> from xyzservices import TileProvider
            >>> bad = TileProvider(
            ...     name="bad",
            ...     url="http://127.0.0.1:1/{z}/{x}/{y}.png",
            ...     attribution="",
            ... )
            >>> fetch_single_tile(  # doctest: +SKIP
            ...     mercantile.Tile(0, 0, 0), bad, timeout=1, retries=0
            ... )
            Traceback (most recent call last):
                ...
            ConnectionError: Failed to fetch tile z=0/x=0/y=0 ...

            ```
    """
    url = provider.build_url(x=tile.x, y=tile.y, z=tile.z)
    last_error: Exception | None = None
    result_bytes: bytes | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": user_agent},
            )
            response = urllib.request.urlopen(request, timeout=timeout)
            data = response.read()
            if not _looks_like_image(data):
                raise OSError(
                    f"Tile response is not a valid image "
                    f"({len(data)} bytes, starts with "
                    f"{data[:8]!r})"
                )
            result_bytes = data
            break
        except (OSError, urllib.error.URLError, ConnectionError) as e:
            last_error = e
            logger.debug(
                "Tile fetch attempt %d/%d failed for %s: %s",
                attempt + 1,
                retries + 1,
                url,
                e,
            )
    if result_bytes is None:
        raise ConnectionError(
            f"Failed to fetch tile z={tile.z}/x={tile.x}/y={tile.y} "
            f"after {retries + 1} attempts: {last_error}"
        )
    return tile, result_bytes


def fetch_tiles(
    tiles: list,
    provider: Any,
    max_workers: int = 8,
    timeout: int = 10,
    retries: int = 2,
    user_agent: str = USER_AGENT,
) -> dict:
    """Fetch tile images in parallel over HTTP.

    Uses `concurrent.futures.ThreadPoolExecutor` for parallel
    downloads. Each tile URL is constructed via the provider's
    `build_url()`. A `User-Agent` header (`cleopatra/<version> (+repo-url)`
    by default) is sent on every request so tile providers can attribute
    the traffic — OpenStreetMap's usage policy requires an identifiable
    agent.

    Args:
        tiles: Tiles to fetch (each has `x`, `y`, `z` attributes).
        provider: `xyzservices.TileProvider` with a URL template.
        max_workers: Maximum concurrent HTTP connections.
        timeout: Per-tile HTTP request timeout in seconds.
        retries: Per-tile retry count on failure.
        user_agent: `User-Agent` header sent on every request. Defaults
            to `USER_AGENT`.

    Returns:
        dict: Mapping of Tile to PNG/JPEG bytes.

    Raises:
        ConnectionError: If any tile cannot be fetched after all
            retries.

    Examples:
        - Fetch a small tile grid in parallel (network-dependent, hence
            skipped under doctest):
            ```python
            >>> import mercantile
            >>> from cleopatra.tiles import fetch_tiles, get_provider
            >>> tiles = list(mercantile.tiles(13.0, 52.4, 13.6, 52.6, zooms=10))
            >>> provider = get_provider("OpenStreetMap.Mapnik")
            >>> data = fetch_tiles(tiles, provider, max_workers=4)  # doctest: +SKIP
            >>> len(data) == len(tiles)  # doctest: +SKIP
            True

            ```
        - Pass an empty list to short-circuit and get an empty dict:
            ```python
            >>> from cleopatra.tiles import fetch_tiles, get_provider
            >>> provider = get_provider("OpenStreetMap.Mapnik")
            >>> fetch_tiles([], provider)
            {}

            ```
    """
    tile_data: dict[Any, bytes] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                fetch_single_tile, tile, provider, timeout, retries, user_agent
            ): tile
            for tile in tiles
        }
        try:
            for future in as_completed(futures):
                tile_obj, png_bytes = future.result()
                tile_data[tile_obj] = png_bytes
        except ConnectionError:
            for f in futures:
                f.cancel()
            raise
        except Exception as e:
            for f in futures:
                f.cancel()
            logger.error("Unexpected error during tile fetching: %s", e)
            raise
    return tile_data


def stitch_tiles(
    tile_data: dict,
    tiles: list,
    zoom: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Stitch tile images into a single RGBA array.

    Arranges tiles in a grid based on their `x`, `y` positions. The
    tile size is read from the first decoded image (typically 256 or
    512 px). Computes the geographic extent of the stitched image in
    EPSG:3857 using `mercantile.xy_bounds` on the corner tiles.

    Args:
        tile_data: Mapping of Tile to PNG bytes (from
            `fetch_tiles`).
        tiles: All tiles in the grid, defining grid dimensions.
        zoom: Zoom level of the tiles.

    Returns:
        tuple[numpy.ndarray, tuple[float, float, float, float]]: The
        stitched RGBA image with shape `(H, W, 4)` and dtype
        `uint8`, plus `(west, south, east, north)` in EPSG:3857
        meters.

    Raises:
        ValueError: If any tile bytes cannot be decoded as an image.

    Examples:
        - Stitch a single synthetic tile into a 256x256 RGBA image:
            ```python
            >>> import io
            >>> import mercantile
            >>> from PIL import Image
            >>> from cleopatra.tiles import stitch_tiles
            >>> buf = io.BytesIO()
            >>> Image.new("RGBA", (256, 256), (255, 0, 0, 255)).save(buf, "PNG")
            >>> tile = mercantile.Tile(0, 0, 0)
            >>> image, extent = stitch_tiles({tile: buf.getvalue()}, [tile], 0)
            >>> image.shape
            (256, 256, 4)
            >>> image.dtype.name
            'uint8'

            ```
        - The returned EPSG:3857 extent comes from
            `mercantile.xy_bounds` on the corner tiles:
            ```python
            >>> import io
            >>> import mercantile
            >>> from PIL import Image
            >>> from cleopatra.tiles import stitch_tiles
            >>> buf = io.BytesIO()
            >>> Image.new("RGBA", (256, 256), (0, 255, 0, 255)).save(buf, "PNG")
            >>> tile = mercantile.Tile(0, 0, 0)
            >>> _, (w, s, e, n) = stitch_tiles({tile: buf.getvalue()}, [tile], 0)
            >>> w < e and s < n
            True

            ```
        - Invalid tile bytes raise `ValueError`:
            ```python
            >>> import mercantile
            >>> from cleopatra.tiles import stitch_tiles
            >>> tile = mercantile.Tile(0, 0, 0)
            >>> try:
            ...     stitch_tiles({tile: b"not an image"}, [tile], 0)
            ... except ValueError as exc:
            ...     str(exc).startswith("Failed to decode tile image:")
            True

            ```
    """
    _require_tiles_extra()
    import mercantile
    from PIL import Image

    try:
        first_img = Image.open(io.BytesIO(next(iter(tile_data.values()))))
    except Exception as e:
        raise ValueError(
            f"Failed to decode tile image: {e}. The tile server "
            f"may have returned invalid data."
        ) from e
    tile_size = first_img.width

    x_indices = sorted({t.x for t in tiles})
    y_indices = sorted({t.y for t in tiles})
    width = len(x_indices) * tile_size
    height = len(y_indices) * tile_size

    merged = Image.new("RGBA", (width, height))
    for tile, png_bytes in tile_data.items():
        try:
            img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        except Exception as e:
            raise ValueError(
                f"Failed to decode tile z={tile.z}/x={tile.x}/y={tile.y}: {e}"
            ) from e
        x_offset = (tile.x - x_indices[0]) * tile_size
        y_offset = (tile.y - y_indices[0]) * tile_size
        merged.paste(img, (x_offset, y_offset))

    image = np.array(merged)

    tl = mercantile.xy_bounds(mercantile.Tile(x_indices[0], y_indices[0], zoom))
    br = mercantile.xy_bounds(mercantile.Tile(x_indices[-1], y_indices[-1], zoom))
    extent_3857 = (tl.left, br.bottom, br.right, tl.top)

    return image, extent_3857


def add_tiles(
    ax: Any,
    source: Any | None = None,
    *,
    crs: int | str | None = None,
    zoom: int | str = "auto",
    alpha: float = 1.0,
    attribution: str | bool = True,
    zorder: int = -1,
    interpolation: str = "bilinear",
    timeout: int = 10,
    retries: int = 2,
    user_agent: str | None = None,
    max_tiles: int = MAX_TILES,
    min_tiles_across: int = 2,
) -> Any:
    """Overlay a web-tile basemap on a matplotlib axes.

    Fetches XYZ web tiles that cover the axes' current extent, stitches
    them into a single composite image, and renders the image below the
    existing data layer. When the data is already in Web Mercator
    (EPSG:3857) the tiles are placed in-place; for any other CRS the
    mosaic's own Web-Mercator coverage is reprojected into the target CRS
    and used as the image extent (the axis limits stay at the data
    bounds), so the basemap aligns with the data even though the fetched
    tiles cover a tile-snapped area larger than it.

    Args:
        ax: Matplotlib `matplotlib.axes.Axes` to add the
            basemap to. Data must already be plotted so the axis
            limits define the geographic extent.
        source: Tile provider. `None` defaults to
            `OpenStreetMap.Mapnik`. A dot-separated string such as
            `"CartoDB.Positron"` is resolved via
            `get_provider`. An `xyzservices.TileProvider` is
            used directly.
        crs: CRS of the data on `ax`. An integer is interpreted as an
            EPSG code; a string is passed through (`"EPSG:XXXX"` or
            WKT). `None` is treated as EPSG:3857.
        zoom: Tile zoom level. `"auto"` derives a level from the
            axes extent. Integers must be in `[0, 19]`.
        alpha: Opacity of the basemap (`0.0`--`1.0`).
        attribution: `True` adds the provider's attribution text, a
            string overrides it, `False` skips it.
        zorder: Matplotlib zorder for the basemap (`-1` puts it
            behind all data).
        interpolation: Interpolation method passed to `ax.imshow`.
        timeout: Per-tile HTTP timeout in seconds.
        retries: Per-tile retry count.
        user_agent: `User-Agent` header to send on tile requests.
            `None` (default) uses `USER_AGENT`
            (`cleopatra/<version> (+repo-url)`). Pass your own string
            when embedding cleopatra in an application so the traffic is
            attributed to *that* app (recommended for production use, and
            required by some providers' usage policies).
        max_tiles: Cap on how many tiles to fetch. If the chosen `zoom`
            would need more than this, the zoom is stepped down until the
            count fits (or reaches 0). Defaults to `MAX_TILES`
            (`256`). Must be a positive int.
        min_tiles_across: Floor for the automatic zoom, forwarded to
            `auto_zoom` when `zoom="auto"` (ignored for an explicit `zoom=`).
            Higher values give a sharper basemap at the cost of more tiles.
            Defaults to 2. See `auto_zoom`.

    Returns:
        matplotlib.axes.Axes: The same axes, for chaining.

    Raises:
        ImportError: If the `[tiles]` extra is not installed.
        TypeError: If `ax` is not a matplotlib Axes.
        ValueError: If the axes have no data extent or `zoom` is
            invalid.
        ConnectionError: If tiles cannot be fetched from the provider.

    Examples:
        Add a default OpenStreetMap basemap to an existing plot:

        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> _ = ax.plot([1000000.0, 1200000.0], [6000000.0, 6200000.0])
        >>> _ = add_tiles(ax, crs=3857)  # doctest: +SKIP
    """
    _require_tiles_extra()
    import mercantile
    from pyproj import Transformer

    if not hasattr(ax, "get_xlim") or not hasattr(ax, "get_ylim"):
        raise TypeError(
            f"ax must be a matplotlib.axes.Axes instance, got {type(ax).__name__}"
        )

    if not isinstance(max_tiles, int) or isinstance(max_tiles, bool) or max_tiles < 1:
        raise ValueError(f"max_tiles must be a positive int, got {max_tiles!r}.")

    if (
        not isinstance(min_tiles_across, int)
        or isinstance(min_tiles_across, bool)
        or min_tiles_across < 1
    ):
        raise ValueError(
            f"min_tiles_across must be a positive int, got {min_tiles_across!r}."
        )

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    west, east = min(x0, x1), max(x0, x1)
    south, north = min(y0, y1), max(y0, y1)

    if (west, east) == (0.0, 1.0) and (south, north) == (0.0, 1.0):
        raise ValueError("Axes have no data extent. Plot data before adding a basemap.")

    if west == east or south == north:
        raise ValueError(
            f"Axes have zero-area extent (west={west}, east={east}, "
            f"south={south}, north={north}). A basemap requires a "
            f"non-degenerate geographic extent."
        )

    if isinstance(source, str) or source is None:
        provider = get_provider(source)
    else:
        provider = source

    crs_value: int | str = 3857 if crs is None else crs
    crs_str = f"EPSG:{crs_value}" if isinstance(crs_value, int) else str(crs_value)
    is_3857 = crs_str == "EPSG:3857"

    if is_3857:
        w3857, s3857, e3857, n3857 = west, south, east, north
    else:
        try:
            w3857, s3857, e3857, n3857 = _densify_and_reproject_bounds(
                west, south, east, north, crs_str, "EPSG:3857"
            )
        except Exception as e:
            if "CRS" in type(e).__name__ or "Invalid" in str(e):
                raise ValueError(
                    f"Invalid CRS: {crs_str!r}. Provide a valid "
                    f"EPSG code or WKT string."
                ) from e
            raise

    transformer_to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    w4326, s4326 = transformer_to_4326.transform(w3857, s3857)
    e4326, n4326 = transformer_to_4326.transform(e3857, n3857)

    if not all(np.isfinite(v) for v in (w4326, s4326, e4326, n4326)):
        raise ValueError(
            "Reprojection to EPSG:4326 produced infinite or NaN "
            "coordinates. The data extent may be outside the valid "
            "Web Mercator domain."
        )

    s4326 = max(s4326, -85.06)
    n4326 = min(n4326, 85.06)

    bounds_4326 = (w4326, s4326, e4326, n4326)

    if zoom == "auto":
        tile_zoom = auto_zoom(bounds_4326, min_tiles_across=min_tiles_across)
    else:
        try:
            tile_zoom = int(zoom)
        except (ValueError, TypeError) as e:
            raise ValueError(f"zoom must be 'auto' or int 0-19, got {zoom!r}") from e
        if not 0 <= tile_zoom <= 19:
            raise ValueError(f"zoom must be 0-19, got {tile_zoom}")

    original_zoom = tile_zoom
    tiles = list(mercantile.tiles(w4326, s4326, e4326, n4326, zooms=tile_zoom))

    while len(tiles) > max_tiles and tile_zoom > 0:
        tile_zoom -= 1
        tiles = list(mercantile.tiles(w4326, s4326, e4326, n4326, zooms=tile_zoom))

    if tile_zoom != original_zoom:
        logger.warning(
            "Zoom reduced from %d to %d (extent requires > %d tiles).",
            original_zoom,
            tile_zoom,
            max_tiles,
        )

    if not tiles:
        raise ValueError(
            f"No tiles found for bounds {bounds_4326} at zoom "
            f"{tile_zoom}. The extent may be outside valid tile "
            f"coverage."
        )

    tile_data = fetch_tiles(
        tiles,
        provider,
        timeout=timeout,
        retries=retries,
        user_agent=user_agent if user_agent is not None else USER_AGENT,
    )
    image, extent_3857 = stitch_tiles(tile_data, tiles, tile_zoom)

    if is_3857:
        extent = extent_3857
    else:
        # Place the mosaic at its OWN geographic coverage, not the data
        # bounds. The stitched tiles span `extent_3857` (Web-Mercator
        # metres, tile-snapped and generally larger than the data), so
        # reproject those corners into the target CRS. Using the data
        # bounds instead stretched the mosaic onto the smaller extent and
        # offset the basemap by up to hundreds of km at coarse zooms (see
        # issue #176). A residual Mercator-vs-linear-axis nonlinearity
        # remains for large extents; reproject the data to EPSG:3857 before
        # plotting for pixel-accurate tiles.
        try:
            extent = _densify_and_reproject_bounds(
                extent_3857[0],
                extent_3857[1],
                extent_3857[2],
                extent_3857[3],
                "EPSG:3857",
                crs_str,
            )
        except ValueError:
            # `ValueError` is the complete failure surface here: `crs_str` was
            # already validated by the forward transform above (a bad CRS would
            # have raised there), and `_densify_and_reproject_bounds` reports an
            # out-of-domain mosaic as `ValueError` (non-finite corners) rather
            # than letting pyproj's `inf`/NaN through -- so no other exception
            # type is expected from the reverse transform.
            # A coarse tile-snapped mosaic can be far larger than the data
            # and overflow a limited-domain target CRS (a UTM zone, national
            # grid, ...) when reprojected, yielding non-finite corners. Fall
            # back to the data bounds (the mosaic is then stretched onto them,
            # slightly misaligned) so a figure is still produced rather than
            # raising -- use a higher zoom or reproject the data to EPSG:3857
            # for accurate placement.
            logger.warning(
                "Tile mosaic bounds could not be reprojected from EPSG:3857 "
                "to %s (they overflow the target CRS domain); falling back to "
                "the data bounds. Use a higher zoom or reproject the data to "
                "EPSG:3857 for accurate tile placement.",
                crs_str,
            )
            extent = (west, south, east, north)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.imshow(
        image,
        extent=[extent[0], extent[2], extent[1], extent[3]],
        interpolation=interpolation,
        alpha=alpha,
        zorder=zorder,
        aspect=ax.get_aspect(),
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if attribution is True:
        raw = getattr(provider, "attribution", None) or getattr(
            provider, "html_attribution", ""
        )
        if raw:
            # Strip HTML tags, then unescape entities (`&copy;` -> `©`,
            # `&amp;` -> `&`, ...) so the placed text reads cleanly.
            attr_text = html.unescape(re.sub(r"<[^>]+>", "", raw)).strip() or None
        else:
            attr_text = None
    elif isinstance(attribution, str):
        attr_text = attribution
    else:
        attr_text = None

    if attr_text:
        ax.text(
            0.99,
            0.01,
            attr_text,
            transform=ax.transAxes,
            fontsize=6,
            ha="right",
            va="bottom",
            alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.5),
        )

    return ax
