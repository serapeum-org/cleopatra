# FlowGlyph Class

The `FlowGlyph` class renders a sequence of polylines as a `LineCollection`.
With a per-path `values` array the lines are colour-mapped through the shared
scalar-mapping pipeline and a colorbar is attached; a per-path `widths` array
scales the line widths by magnitude (with an optional width legend). It suits
flow / flux paths such as river reaches or transport links.

## Class Documentation

::: cleopatra.flow_glyph.FlowGlyph
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Examples

### Magnitude-coloured, width-scaled flows

```python
import numpy as np
from cleopatra.flow_glyph import FlowGlyph

paths = [
    np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 0.0]]),
    np.array([[0.0, 1.0], [1.0, 1.5], [2.0, 1.0]]),
    np.array([[0.0, 2.0], [1.0, 1.8], [2.0, 2.2]]),
]
values = np.array([10.0, 25.0, 40.0])   # colour
widths = np.array([1.0, 3.0, 6.0])      # line width

fg = FlowGlyph(paths, values=values, widths=widths, size_legend=True)
fig, ax, lc = fg.plot(title="Flow paths")
```

### Uncoloured flows (no colorbar)

```python
fg = FlowGlyph(paths)
fig, ax, lc = fg.plot()
```
