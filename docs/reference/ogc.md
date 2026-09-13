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

A RESTful WMTS template works too — if the `url` carries any of the placeholders this module
owns (`{TileMatrix}`, `{TileRow}`, `{TileCol}`, `{TileMatrixSet}`, `{Layer}`, `{Style}`, in any
casing) it is substituted rather than turned into a query:

```python
WMTSProvider(
    url="https://example.org/wmts/{Layer}/{TileMatrix}/{TileRow}/{TileCol}.png",
    layer="TrueColor",
)
```

Substituted values are percent-encoded, the pass is single so a value that happens to look like a
placeholder is left as data, and a placeholder this module does not own is passed through
untouched.

## What is refused, and when

Both providers validate at construction rather than per tile mid-render, so a mistake names the
field it came from instead of arriving much later as an unreadable tile:

- a `url` that is not `http(s)`, or that contains any whitespace;
- an empty `layer` / `layers` / `image_format` / `version` (and `styles` / `attribution` must be
  strings, though empty is fine);
- a WMS `version` outside `1.1.0` / `1.1.1` / `1.3.0`, or a WMTS `version` other than `1.0.0`;
- a `transparent` that is not a real `bool`, or a `tile_size` that is not a positive `int`;
- a RESTful template missing `{TileMatrix}` / `{TileRow}` / `{TileCol}` — without them every tile
  resolves to the same URL, so the mosaic would silently repeat one image;
- a template whose placeholders are *none* of the ones above — an XYZ `{z}/{x}/{y}` template
  belongs in `add_tiles(source=...)` as an `xyzservices` provider, not here;
- `extra_params` that sets a parameter identifying the tile (`BBOX`, `WIDTH`, `HEIGHT`, `REQUEST`,
  `SERVICE`, `TILEMATRIX`, `TILEROW`, `TILECOL`), or that names one parameter twice in different
  casings.

Both providers are frozen, hashable, picklable and deep-copyable, so one can key a dict or an
`lru_cache`; a copy made with `dataclasses.replace` re-runs every check above.

## Credentials

A keyed service is just a query parameter, so `extra_params` covers it. It is merged last, so it
can also override a parameter the provider would otherwise generate — matched case-insensitively,
as OGC parameter names are, so `{"format": ...}` replaces the generated `FORMAT` rather than
joining it:

```python
WMSProvider(url="https://example.org/wms", layers="ortho", extra_params={"token": "..."})
```

Two limits are worth knowing. A parameter baked into the endpoint's own query (`.../wms?FORMAT=…`)
is kept verbatim and is *not* overridden this way — put it in `extra_params` instead of the `url`
if you need to control it. And the parameters that identify the tile cannot be overridden at all
(see above).

Credentials are kept out of the two places this package would otherwise publish them: the
provider's `repr()` masks every `extra_params` value, and a failed tile fetch logs a URL whose
credential-shaped query values are masked while the OGC parameter names that say *which* tile
failed are kept. A key embedded in the `url` **path** cannot be told from an ordinary path segment
and is not masked.

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

!!! note "`world_texture` is XYZ-only"
    `cleopatra.basemap.tiles.world_texture` keys its disk cache on
    `provider.get("name", ...)`, so it needs a Mapping-like provider and raises
    `AttributeError: 'WMTSProvider' object has no attribute 'get'` on these dataclasses. Use `add_tiles` for an
    OGC service, and an `xyzservices` provider when you want a cached world texture.

!!! note "RESTful templates fix their own format and version"
    On the RESTful WMTS branch the service encodes the format and version in the template path, so
    `image_format` and `version` are validated but never sent. They apply to the KVP branch only.

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
