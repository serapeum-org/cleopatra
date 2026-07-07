# Tiles Module — Web-tile Basemaps

The `cleopatra.tiles` module adds an optional, pure-Python web-tile basemap helper:
`add_tiles` fetches XYZ map tiles covering an axes' current extent, stitches them with
Pillow, and renders the composite underneath your data. No GDAL is required.

It is gated behind the `cleopatra[tiles]` optional extra (`mercantile`, `pillow`,
`pyproj`, `xyzservices`):

=== "pip"

    ```bash
    pip install "cleopatra[tiles]"
    ```

=== "conda"

    ```bash
    conda install -c conda-forge cleopatra-tiles
    ```

If the extra is not installed, the functions raise a clear `ImportError` with the install
hint.

## Usage

```python
import matplotlib
matplotlib.use("Agg")  # any backend; Agg shown for headless rendering
import matplotlib.pyplot as plt

from cleopatra.tiles import add_tiles

fig, ax = plt.subplots()
# plot something in Web Mercator (EPSG:3857) coordinates ...
ax.plot([1_000_000.0, 1_200_000.0], [6_000_000.0, 6_200_000.0])

# ... and drop an OpenStreetMap basemap underneath it
add_tiles(ax, crs=3857)

# a different provider, a fixed zoom, a custom User-Agent (recommended in production):
add_tiles(
    ax,
    source="CartoDB.Positron",
    crs="EPSG:4326",
    zoom=8,
    user_agent="my-app/1.0 (+https://example.org)",
)

fig.savefig("map.png")
```

!!! note
    `add_tiles` reads the axes' current `xlim`/`ylim`, so plot your data first. When the
    data CRS is Web Mercator the tiles are placed in-place; for any other `crs=` the
    mosaic's own Web-Mercator coverage is reprojected into the target CRS and used as the
    image extent (the axis limits stay at the data bounds), so the basemap aligns with the
    data even though the fetched tiles cover a tile-snapped area larger than it. A residual
    Mercator-vs-linear nonlinearity remains for very large extents — for pixel-accurate
    results reproject the source data to EPSG:3857 before plotting. If a coarse mosaic
    overflows a limited-domain target CRS (e.g. a whole-world mosaic into a UTM zone), the
    reprojection is skipped with a warning and the basemap falls back to the data bounds
    (slightly misaligned); use a higher zoom or reproject the data to EPSG:3857 to avoid it.
    The automatic zoom uses
    a `min_tiles_across` floor (default 2) so a mid-range extent is not rendered from one or
    two coarse tiles; the number of tiles is capped by `max_tiles` (default
    `MAX_TILES = 256`), and the zoom is stepped down if a level would need more.

## Module Documentation

::: cleopatra.tiles
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
