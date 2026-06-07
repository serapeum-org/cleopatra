# PolygonGlyph Class

The `PolygonGlyph` class wraps `matplotlib.collections.PolyCollection`. With a
per-polygon `values` array the polygons are filled and colour-mapped through the
shared scalar-mapping pipeline and a colorbar is attached. With no values (or
`outline_only=True`) only the polygon outlines are drawn.

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

### Outlines only

```python
pg = PolygonGlyph(polygons)
fig, ax, pc = pg.plot(outline_only=True)
```
