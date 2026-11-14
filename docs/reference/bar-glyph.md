# BarGlyph Class

The `bar_glyph` module provides the `BarGlyph` class for **categorical bar
charts** — one bar per category from a `(categories, values)` pair, coloured
from the shared palette system and styled through the shared `Glyph` pipeline
(title, axis labels, tick sizes, optional legend). A 2-D `values` block with
`mode="stacked"` or `mode="grouped"` draws a multi-series chart, and
`horizontal=True` draws a `barh`.

(For a numeric bar of a single series on an x/y axis, see
[`LineGlyph`](line-glyph.md)'s `bar` method; `BarGlyph` is the category-indexed
chart.)

## Class Documentation

::: cleopatra.glyphs.stats.bar_glyph.BarGlyph
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Examples

### Single-series bar chart

```python
from cleopatra.glyphs.stats.bar_glyph import BarGlyph

glyph = BarGlyph(["a", "b", "c"], [3.0, 1.0, 2.0])
fig, ax, bars = glyph.plot(title="Counts by category")
```

### Stacked multi-series chart

```python
import numpy as np
from cleopatra.glyphs.stats.bar_glyph import BarGlyph

values = np.array([[3.0, 1.0], [1.0, 2.0], [2.0, 2.0]])  # (n_categories, n_series)
glyph = BarGlyph(
    ["a", "b", "c"], values, mode="stacked", series_labels=["x", "y"], add_legend=True
)
fig, ax, bars = glyph.plot(title="Stacked by series")
```
