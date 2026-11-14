# Tiles Module — Web-tile Basemaps

The `cleopatra.basemap.tiles` module adds an optional, pure-Python web-tile basemap helper:
`add_tiles` fetches XYZ map tiles covering an axes' current extent, stitches them with
Pillow, and renders the composite underneath your data. No GDAL is required.

For a service published as OGC **WMS** or **WMTS** rather than an XYZ template, pass one of
the provider objects from [`cleopatra.basemap.ogc`](ogc.md) as the `source` — the fetch,
stitch and compose path below is the same one they use.

It is gated behind the `cleopatra[tiles]` optional extra (`pillow`, `pyproj`,
`xyzservices`):

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

Web-tile basemaps are added through the `add_tiles` glyph method
(`cleopatra.basemap.geo.GeoMixin`, inherited by every geographic glyph), which
forwards to this module. For the headless case (no axes), use the public
`stitch_basemap` to get the stitched image and its extent as data.

```python
import numpy as np
import matplotlib

matplotlib.use("Agg")  # any backend; Agg shown for headless rendering

from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph

# a 2-D array positioned in Web Mercator (EPSG:3857) metres
glyph = ArrayGlyph(
    np.random.random((20, 20)),
    extent=[1_000_000.0, 1_200_000.0, 6_000_000.0, 6_200_000.0],
)
glyph.plot()

# drop an OpenStreetMap basemap underneath the data
glyph.add_tiles(crs=3857)

# a different provider, a fixed zoom, a custom User-Agent (recommended in production):
glyph.add_tiles(
    source="CartoDB.Positron",
    crs="EPSG:4326",
    zoom=8,
    user_agent="my-app/1.0 (+https://example.org)",
)

glyph.fig.savefig("map.png")
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

::: cleopatra.basemap.tiles
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
