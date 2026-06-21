# Geographic basemap methods (glyphs)

The glyphs that plot geographic data — `ArrayGlyph`, `MeshGlyph`, `VectorGlyph`,
`FlowGlyph`, `PolygonGlyph`, and `ScatterGlyph` — inherit
`cleopatra.geo.GeoMixin`, which adds three convenience methods that drop a
basemap onto the glyph's **own axes** without importing the standalone helpers:

- `add_tiles` → [`cleopatra.tiles.add_tiles`](tiles.md)
- `add_features` → [`cleopatra.reference.add_features`](reference-data.md)
- `add_relief` → [`cleopatra.reference.add_relief`](reference-data.md)

Each method is a thin wrapper: it draws on `self.ax` (the axes produced when you
plot the glyph) and forwards every argument to the matching standalone function,
which remains the single source of truth. Chart and statistical glyphs
(`LineGlyph`, `StatisticalGlyph`, `KDEGlyph`) deliberately do **not** inherit
these geo-only methods.

## Usage

```python
import matplotlib
matplotlib.use("Agg")  # any backend
import numpy as np

from cleopatra.array_glyph import ArrayGlyph

glyph = ArrayGlyph(np.random.default_rng(0).random((50, 50)))
fig, ax, *_ = glyph.plot()        # plot your data first

glyph.add_relief("low")            # hypsometric backdrop, on glyph.ax
glyph.add_features("coastline", "50m", colors="black")
glyph.add_features("borders", "50m", colors="0.4")
```

The standalone functions still work for plain matplotlib axes or non-geographic
glyphs:

```python
from cleopatra.reference import add_features

add_features(ax, "coastline", "110m")   # any matplotlib Axes
```

!!! note
    Call these **after** plotting (so the glyph has an axes), or pass an explicit
    `ax=`. `add_relief` and `crs=` reprojection require the `cleopatra[tiles]`
    extra; drawing vector layers in EPSG:4326 needs only numpy + matplotlib.

## Module Documentation

::: cleopatra.geo
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
