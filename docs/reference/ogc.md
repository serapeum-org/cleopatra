# OGC Module — WMS and WMTS Basemaps

The `cleopatra.basemap.ogc` module lets an OGC **WMS** or **WMTS** service be drawn by the
same helper that draws XYZ tiles. `WMTSProvider` and `WMSProvider` are small frozen
dataclasses that satisfy the one-method contract
[`add_tiles`](tiles.md) already calls — `build_url(x=, y=, z=)` returning an http(s) URL —
so they are passed straight in as the `source`. Nothing about the fetch, the mosaic or the
composite changes.

Importing this module is free; the `cleopatra[tiles]` extra is needed only to render (see
[installation](../installation.md)).

## How each kind maps onto a tile

| | Request | Mapping |
|---|---|---|
| **WMTS** | `GetTile` | The `(TileMatrix, TileRow, TileCol)` triple *is* `(z, y, x)` — an identity mapping on the `GoogleMapsCompatible` grid. |
| **WMS** | `GetMap` | No tile index exists, so each tile is converted to its Web Mercator bounds and requested as its own image: one `GetMap` per tile. |

## Usage

```python
import matplotlib
matplotlib.use("Agg")  # any backend; Agg shown for headless rendering
import matplotlib.pyplot as plt

from cleopatra.basemap.ogc import WMSProvider, WMTSProvider
from cleopatra.basemap.tiles import add_tiles

fig, ax = plt.subplots()
ax.plot([1_000_000.0, 1_200_000.0], [6_000_000.0, 6_200_000.0])

# a WMTS service, KVP endpoint
add_tiles(ax, WMTSProvider(
    url="https://example.org/wmts",
    layer="TrueColor",
    tile_matrix_set="GoogleMapsCompatible",
    image_format="image/jpeg",
    attribution="Imagery provider",
), crs=3857)

# a WMS service: one GetMap per tile
add_tiles(ax, WMSProvider(
    url="https://example.org/wms",
    layers="ortho",
    attribution="Ortho provider",
), crs=3857)
```

A RESTful WMTS template works too — if the `url` contains `{TileMatrix}` it is substituted
rather than turned into a query:

```python
WMTSProvider(
    url="https://example.org/wmts/{Layer}/{TileMatrix}/{TileRow}/{TileCol}.png",
    layer="TrueColor",
)
```

## Credentials

A keyed service is just a query parameter, so `extra_params` covers it. It is merged last,
so it can also override a parameter the provider would otherwise generate:

```python
WMSProvider(url="https://example.org/wms", layers="ortho", extra_params={"token": "..."})
```

!!! note "Only the GoogleMapsCompatible grid"
    The surrounding tile geometry implements one tile-matrix set: Web Mercator, `2**z`
    columns, top-left origin, square tiles. A WMTS published on any other matrix set (NASA
    GIBS' `EPSG4326_250m`, a national grid such as EPSG:28992) will return tiles that do not
    line up. WMS is unaffected — it is matrix-set-free by construction, since every request
    names its own `BBOX`.

!!! note "WMS efficiency"
    A WMS answers any `BBOX`, so covering the viewport in one request is cheaper than a
    mosaic of many. `add_tiles(..., min_tiles_across=1)` lowers the tile count towards a
    single `GetMap` without a separate code path.

!!! note "Image formats and service exceptions"
    Only PNG, JPEG, GIF and WebP are recognised, so `image/tiff` and `image/svg+xml` are not
    supported. A service that answers with an XML `ServiceExceptionReport` — the usual reply
    to a bad layer name — is treated as an unreadable tile and surfaces as a
    `ConnectionError` after the retries, which reads as a network failure rather than the
    service error it is.

## Module Documentation

::: cleopatra.basemap.ogc
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
