"""Web-tile basemap helper for matplotlib axes.

Provides :func:`add_tiles` -- a single entry point that fetches XYZ web
tiles for the current axes extent, stitches them into a composite image
with Pillow, and renders the image underneath the existing data layer.

The implementation is a pure-Python port of the ``pyramids.basemap``
module (basemap.py + tiles.py). It supports any XYZ provider listed in
:mod:`xyzservices`. CRS handling is done with :mod:`pyproj` -- there is
no GDAL dependency, so the module is safe to use in environments that
only have matplotlib + numpy installed.

Notes:
    For data in projected CRSes other than Web Mercator (EPSG:3857) the
    stitched tile image is placed in the target CRS using the data's
    densified bounds. Matplotlib stretches the image to fit, which is
    visually acceptable for small extents (e.g. local maps in EPSG:4326)
    but may show projection distortion over very large areas. If
    pixel-accurate warping is required, reproject the source data to
    Web Mercator (EPSG:3857) before plotting.

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

import io
import logging
import math
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

USER_AGENT = "cleopatra/Python"

MAX_TILES = 256

_TILES_EXTRA_HINT = (
    "Web-tile basemap support requires the [tiles] extra. "
    "Install with `pip install cleopatra[tiles]`."
)

try:
    import mercantile  # noqa: F401
    import xyzservices  # noqa: F401
    from PIL import Image  # noqa: F401
    from pyproj import Transformer  # noqa: F401

    _TILES_AVAILABLE = True
except ImportError:
    _TILES_AVAILABLE = False


def _require_tiles_extra() -> None:
    """Raise :class:`ImportError` if the tiles extra is not installed.

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
        name: Dot-separated provider name (e.g. ``"OpenStreetMap.Mapnik"``,
            ``"CartoDB.Positron"``, ``"Esri.WorldImagery"``). ``None``
            returns the default (``OpenStreetMap.Mapnik``).

    Returns:
        xyzservices.TileProvider: The resolved tile provider with
        ``url`` template and ``attribution`` metadata.

    Raises:
        ImportError: If the ``[tiles]`` extra is not installed.
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
        - Invalid provider names raise :class:`ValueError`:
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


def auto_zoom(bounds_4326: tuple[float, float, float, float]) -> int:
    """Compute a default zoom level for the given bounds in EPSG:4326.

    Uses the formula
    ``zoom = ceil(log2(360 / max(lon_extent, lat_extent)))`` clamped to
    the range 0--19.

    Args:
        bounds_4326: ``(west, south, east, north)`` in EPSG:4326 degrees.

    Returns:
        int: Zoom level between 0 and 19.

    Examples:
        - Worldwide extent maps to zoom 0:
            ```python
            >>> from cleopatra.tiles import auto_zoom
            >>> auto_zoom((-180, -85, 180, 85))
            0

            ```
        - A 0.6 by 0.2 degree window over Berlin yields zoom 10:
            ```python
            >>> from cleopatra.tiles import auto_zoom
            >>> auto_zoom((13.0, 52.4, 13.6, 52.6))
            10

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
    zoom = math.ceil(math.log2(360.0 / max_extent))
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
        src_crs: Source CRS identifier (e.g. ``"EPSG:4326"``).
        dst_crs: Target CRS identifier (e.g. ``"EPSG:3857"``).
        n_points: Number of sample points per edge. 21 balances
            accuracy and performance for typical warps.

    Returns:
        tuple[float, float, float, float]: ``(west, south, east, north)``
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


def fetch_single_tile(
    tile: Any,
    provider: Any,
    timeout: int,
    retries: int,
) -> tuple[Any, bytes]:
    """Fetch a single tile, retrying on transient failures.

    Args:
        tile: Tile to fetch (has ``x``, ``y``, ``z`` attributes).
        provider: ``xyzservices.TileProvider`` with a URL template.
        timeout: HTTP request timeout in seconds.
        retries: Number of retry attempts on failure.

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
            >>> data[:4] in (b"\\x89PNG", b"\\xff\\xd8\\xff\\xe0", b"\\xff\\xd8\\xff\\xe1")  # doctest: +SKIP
            True

            ```
        - Tile failures raise :class:`ConnectionError` after retries
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
                headers={"User-Agent": USER_AGENT},
            )
            response = urllib.request.urlopen(request, timeout=timeout)
            data = response.read()
            if not data or data[:4] not in (
                b"\x89PNG",
                b"\xff\xd8\xff\xe0",
                b"\xff\xd8\xff\xe1",
            ):
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
) -> dict:
    """Fetch tile images in parallel over HTTP.

    Uses :class:`concurrent.futures.ThreadPoolExecutor` for parallel
    downloads. Each tile URL is constructed via the provider's
    ``build_url()``. A ``User-Agent`` header (``cleopatra/Python``) is
    sent on every request to comply with tile provider policy.

    Args:
        tiles: Tiles to fetch (each has ``x``, ``y``, ``z`` attributes).
        provider: ``xyzservices.TileProvider`` with a URL template.
        max_workers: Maximum concurrent HTTP connections.
        timeout: Per-tile HTTP request timeout in seconds.
        retries: Per-tile retry count on failure.

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
            executor.submit(fetch_single_tile, tile, provider, timeout, retries): tile
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

    Arranges tiles in a grid based on their ``x``, ``y`` positions. The
    tile size is read from the first decoded image (typically 256 or
    512 px). Computes the geographic extent of the stitched image in
    EPSG:3857 using :func:`mercantile.xy_bounds` on the corner tiles.

    Args:
        tile_data: Mapping of Tile to PNG bytes (from
            :func:`fetch_tiles`).
        tiles: All tiles in the grid, defining grid dimensions.
        zoom: Zoom level of the tiles.

    Returns:
        tuple[numpy.ndarray, tuple[float, float, float, float]]: The
        stitched RGBA image with shape ``(H, W, 4)`` and dtype
        ``uint8``, plus ``(west, south, east, north)`` in EPSG:3857
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
            :func:`mercantile.xy_bounds` on the corner tiles:
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
        - Invalid tile bytes raise :class:`ValueError`:
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
    crs: int | str | None = None,
    zoom: int | str = "auto",
    alpha: float = 1.0,
    attribution: str | bool = True,
    zorder: int = -1,
    interpolation: str = "bilinear",
    timeout: int = 10,
    retries: int = 2,
) -> Any:
    """Overlay a web-tile basemap on a matplotlib axes.

    Fetches XYZ web tiles that cover the axes' current extent, stitches
    them into a single composite image, and renders the image below the
    existing data layer. When the data is already in Web Mercator
    (EPSG:3857) the tiles are placed in-place; for any other CRS the
    image is placed in the target CRS using the data's densified bounds
    -- matplotlib stretches the bitmap to fit, which is visually
    acceptable for small extents.

    Args:
        ax: Matplotlib :class:`~matplotlib.axes.Axes` to add the
            basemap to. Data must already be plotted so the axis
            limits define the geographic extent.
        source: Tile provider. ``None`` defaults to
            ``OpenStreetMap.Mapnik``. A dot-separated string such as
            ``"CartoDB.Positron"`` is resolved via
            :func:`get_provider`. An ``xyzservices.TileProvider`` is
            used directly.
        crs: CRS of the data on ``ax``. An integer is interpreted as an
            EPSG code; a string is passed through (``"EPSG:XXXX"`` or
            WKT). ``None`` is treated as EPSG:3857.
        zoom: Tile zoom level. ``"auto"`` derives a level from the
            axes extent. Integers must be in ``[0, 19]``.
        alpha: Opacity of the basemap (``0.0``--``1.0``).
        attribution: ``True`` adds the provider's attribution text, a
            string overrides it, ``False`` skips it.
        zorder: Matplotlib zorder for the basemap (``-1`` puts it
            behind all data).
        interpolation: Interpolation method passed to ``ax.imshow``.
        timeout: Per-tile HTTP timeout in seconds.
        retries: Per-tile retry count.

    Returns:
        matplotlib.axes.Axes: The same axes, for chaining.

    Raises:
        ImportError: If the ``[tiles]`` extra is not installed.
        TypeError: If ``ax`` is not a matplotlib Axes.
        ValueError: If the axes have no data extent or ``zoom`` is
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
            "ax must be a matplotlib.axes.Axes instance, "
            f"got {type(ax).__name__}"
        )

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    west, east = min(x0, x1), max(x0, x1)
    south, north = min(y0, y1), max(y0, y1)

    if (west, east) == (0.0, 1.0) and (south, north) == (0.0, 1.0):
        raise ValueError(
            "Axes have no data extent. Plot data before adding a basemap."
        )

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

    transformer_to_4326 = Transformer.from_crs(
        "EPSG:3857", "EPSG:4326", always_xy=True
    )
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
        tile_zoom = auto_zoom(bounds_4326)
    else:
        try:
            tile_zoom = int(zoom)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"zoom must be 'auto' or int 0-19, got {zoom!r}"
            ) from e
        if not 0 <= tile_zoom <= 19:
            raise ValueError(f"zoom must be 0-19, got {tile_zoom}")

    original_zoom = tile_zoom
    tiles = list(mercantile.tiles(w4326, s4326, e4326, n4326, zooms=tile_zoom))

    while len(tiles) > MAX_TILES and tile_zoom > 0:
        tile_zoom -= 1
        tiles = list(mercantile.tiles(w4326, s4326, e4326, n4326, zooms=tile_zoom))

    if tile_zoom != original_zoom:
        logger.warning(
            "Zoom reduced from %d to %d (extent requires > %d tiles).",
            original_zoom,
            tile_zoom,
            MAX_TILES,
        )

    if not tiles:
        raise ValueError(
            f"No tiles found for bounds {bounds_4326} at zoom "
            f"{tile_zoom}. The extent may be outside valid tile "
            f"coverage."
        )

    tile_data = fetch_tiles(tiles, provider, timeout=timeout, retries=retries)
    image, extent_3857 = stitch_tiles(tile_data, tiles, tile_zoom)

    if is_3857:
        extent = extent_3857
    else:
        # No GDAL: place the tile bitmap in the target CRS using the
        # data's bounds. Matplotlib stretches the image -- accurate
        # enough for small extents (see module docstring).
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
        attr_text = re.sub(r"<[^>]+>", "", raw) if raw else None
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
