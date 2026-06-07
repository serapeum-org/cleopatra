# ScatterGlyph Class

The `ScatterGlyph` class visualizes 2-D point clouds. With a per-point `values`
array the points are colour-mapped through the shared scalar-mapping pipeline
(so `vmin` / `vmax` / `levels` / `color_scale` behave as for the other glyphs)
and a matching colorbar is attached. An independent per-point `sizes` array
scales the marker area — with an optional size legend — so colour and size can
encode two different quantities at once.

## Class Documentation

::: cleopatra.scatter_glyph.ScatterGlyph
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Examples

### Value-coloured points

```python
import numpy as np
from cleopatra.scatter_glyph import ScatterGlyph

rng = np.random.default_rng(0)
x, y = rng.random(100), rng.random(100)
values = rng.random(100)

sg = ScatterGlyph(x, y, values=values)
fig, ax, paths = sg.plot(title="Coloured points")
```

### Colour *and* size encoding two quantities

```python
import numpy as np
from cleopatra.scatter_glyph import ScatterGlyph

rng = np.random.default_rng(0)
x, y = rng.random(50), rng.random(50)
values = rng.random(50)          # drives colour
sizes = rng.random(50) * 10      # drives marker area (independent of colour)

sg = ScatterGlyph(x, y, values=values, sizes=sizes, size_legend=True)
fig, ax, paths = sg.plot(title="Two encodings at once")
```

### Composing onto shared axes (no per-glyph colorbar)

```python
fig, ax, paths = sg.plot(ax=ax, add_colorbar=False)
```
