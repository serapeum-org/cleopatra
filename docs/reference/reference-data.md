# Reference Data — Coastlines, Borders & Relief

The `cleopatra.reference` module draws public cartographic reference data
*underneath* your own plot — the `cartopy` `ax.coastlines()` /
`GeoAxes.stock_img()` niche, and the vector/raster sibling of
[`add_tiles`](tiles.md):

- **`add_features`** — a Natural Earth vector layer: `coastline`, `borders`,
  `land`, `ocean`, `rivers`, or `lakes`.
- **`add_relief`** — a global hypsometric relief backdrop.

Both fetch a small, **fixed public dataset** that cleopatra re-hosts as a
dependency-light artifact (gzipped GeoJSON / PNG), cache it on disk, and render
it with matplotlib. They acquire reference data only — they never read your
files and **never import GDAL or geopandas**.

## Dependencies

`add_features` in EPSG:4326 needs nothing beyond `numpy` + `matplotlib`: the
layers are pre-converted to gzipped GeoJSON and read with the standard library.
Two paths use the optional `cleopatra[tiles]` extra:

- `add_relief` decodes a PNG with **Pillow**.
- `add_features(..., crs=...)` reprojection uses **pyproj**.

=== "pip"

    ```bash
    pip install "cleopatra[tiles]"
    ```

=== "conda"

    ```bash
    conda install -c conda-forge cleopatra-tiles
    ```

If a required package is missing, the function raises a clear `ImportError` with
the install hint.

## Usage

Draw a relief backdrop with coastlines and country borders on a global
lon/lat (EPSG:4326) map:

```python
import matplotlib
matplotlib.use("Agg")  # any backend; Agg shown for headless rendering
import matplotlib.pyplot as plt

from cleopatra.reference import add_relief, add_features

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)

add_relief(ax, resolution="low")                 # hypsometric backdrop
add_features(ax, "coastline", "110m", colors="black")
add_features(ax, "borders", "110m", colors="0.4")

fig.savefig("world.png", dpi=100)
```

A regional map with a filled land layer and higher-resolution coastline:

```python
fig, ax = plt.subplots()
ax.set_xlim(-20, 40)   # Europe / N. Africa, in lon/lat
ax.set_ylim(0, 60)

add_features(ax, "ocean", "50m", facecolors="#bdd7e7")
add_features(ax, "land", "50m", facecolors="0.9", edgecolors="0.5")
add_features(ax, "coastline", "50m", colors="navy", linewidths=0.8)

fig.savefig("europe.png")
```

If your data is in a projected CRS, pass `crs=` so the vectors are reprojected
to match (requires `pyproj`):

```python
add_features(ax, "coastline", "50m", crs=3857)   # Web Mercator axes
```

The raw data is also available without drawing:

```python
from cleopatra.reference import natural_earth, relief

parts = natural_earth("coastline", "110m")   # list of (N, 2) lon/lat arrays
rgb = relief("low")                          # (H, W, 3) uint8 RGB array
```

!!! note
    `add_features` / `add_relief` read the axes' current `xlim`/`ylim` and
    preserve them, so **plot your data first**. Polygon layers
    (`land`/`ocean`/`lakes`) are drawn as a filled `PolyCollection` (style with
    `facecolors` / `edgecolors` / `linewidths`); line layers
    (`coastline`/`rivers`/`borders`) as a `LineCollection` (style with `colors`
    / `linewidths`). Coordinates are EPSG:4326 unless you pass `crs=`.

!!! tip "Caching"
    The first call downloads the asset from the cleopatra
    [`basemap-data-v1`](https://github.com/serapeum-org/cleopatra/releases/tag/basemap-data-v1)
    release and caches it under `~/.cleopatra/naturalearth`; subsequent calls
    work offline. Override the location with the `CLEOPATRA_CACHE_DIR`
    environment variable. Downloads are restricted to `http(s)` URLs.

## Migrating from `pyramids.basemap`

This data used to live in `pyramids.basemap` (`natural_earth` / `relief`). It
has moved to `cleopatra.reference`, which is the matplotlib map-decoration layer
— the same boundary the web-tile basemaps already follow (`pyramids.basemap`
forwarded to [`cleopatra.tiles`](tiles.md)). cleopatra hosts its own copy of the
assets and has **no dependency on pyramids**.

| Old (`pyramids.basemap`) | New (`cleopatra.reference`) | Notes |
| --- | --- | --- |
| `natural_earth(layer, resolution)` → `FeatureCollection` | `natural_earth(layer, resolution)` → `list[np.ndarray]` | Now returns plain `(N, 2)` lon/lat arrays (exterior rings for polygons), not a GIS feature object. |
| `relief(resolution)` → GDAL `Dataset` | `relief(resolution)` → `(H, W, 3)` uint8 array | Now a NumPy RGB array; the asset is a PNG (no GeoTIFF / GDAL). |
| *(draw it yourself)* | `add_features(ax, layer, resolution)` | New axes helper — the `ax.coastlines()` analogue. |
| *(draw it yourself)* | `add_relief(ax, resolution)` | New axes helper — the `stock_img()` analogue. |
| `PYRAMIDS_CACHE_DIR`, `~/.pyramids/naturalearth` | `CLEOPATRA_CACHE_DIR`, `~/.cleopatra/naturalearth` | Cache env var and default directory renamed. |

The `pyramids.basemap.natural_earth` / `relief` entry points are deprecated and
emit a `DeprecationWarning`; update imports to `cleopatra.reference`. Resolutions
are unchanged (`110m` / `50m` / `10m` for vectors; `low` / `medium` for relief),
as are the six layer names.

## Module Documentation

::: cleopatra.reference
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
