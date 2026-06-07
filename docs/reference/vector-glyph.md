# VectorGlyph Class

The `VectorGlyph` class renders a 2-D `(u, v)` vector field over `(x, y)`
positions as **arrows** (`quiver`), **wind barbs** (`barbs`), or **streamlines**
(`streamplot`). The artist is coloured by the per-vector magnitude
`hypot(u, v)` through the shared scalar-mapping pipeline, with a matching
colorbar.

## Class Documentation

::: cleopatra.vector_glyph.VectorGlyph
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Examples

### Arrows (quiver)

```python
import numpy as np
from cleopatra.vector_glyph import VectorGlyph

x, y = np.meshgrid(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
u, v = np.cos(x * np.pi), np.sin(y * np.pi)

vg = VectorGlyph(x, y, u, v)
fig, ax, artist = vg.plot(kind="quiver", title="Vector field")
```

### Wind barbs and streamlines

```python
fig, ax, barbs = vg.plot(kind="barbs")
fig, ax, stream = vg.plot(kind="streamplot")
```

### Adding a reference key

```python
fig, ax, quiv = vg.plot(kind="quiver")
vg.add_key(quiv, 0.9, 0.95, 1.0, "1 m/s")
```
