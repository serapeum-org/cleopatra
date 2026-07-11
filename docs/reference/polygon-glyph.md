# PolygonGlyph Class

The `PolygonGlyph` class wraps `matplotlib.collections.PolyCollection`. With a
per-polygon `values` array the polygons are filled and colour-mapped through the
shared scalar-mapping pipeline and a colorbar is attached. With no values (or
`outline_only=True`) only the polygon outlines are drawn.

The `edgecolor` option defaults to `"none"` so a value-filled choropleth renders
borderless. Outline mode substitutes `cleopatra.polygon_glyph.OUTLINE_EDGECOLOR`
(black) for that default, since an unfilled polygon with a transparent edge would
be invisible; pass an explicit `edgecolor` to override it.

## Class Documentation

::: cleopatra.polygon_glyph.PolygonGlyph
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Examples

### Value-filled polygons

```python
import numpy as np
from cleopatra.polygon_glyph import PolygonGlyph

polygons = [
    np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]),
    np.array([[0.0, 1.0], [1.0, 1.0], [0.5, 2.0]]),
]
values = np.array([10.0, 20.0, 30.0])

pg = PolygonGlyph(polygons, values=values)
fig, ax, pc = pg.plot(title="Polygons by value")
```

![Polygons filled by value](../images/polygon_glyph/polygons-filled.png)

### Outlines only

```python
pg = PolygonGlyph(polygons)
fig, ax, pc = pg.plot(outline_only=True)

# ... or pick the outline colour and width
pg = PolygonGlyph(polygons, edgecolor="navy", linewidth=1.5)
fig, ax, pc = pg.plot(outline_only=True)
```

![Polygon outlines](../images/polygon_glyph/polygons-outline.png)
