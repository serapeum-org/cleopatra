# PieGlyph Class

The `pie_glyph` module provides the `PieGlyph` class for **categorical pie
charts** — one wedge per category from a `(categories, values)` pair, sized by
each value's share of the total and coloured from the shared palette system.
Negative or all-zero values have no meaningful pie representation and are
rejected.

## Class Documentation

::: cleopatra.glyphs.stats.pie_glyph.PieGlyph
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Examples

### Pie with per-wedge percentage labels

```python
from cleopatra.glyphs.stats.pie_glyph import PieGlyph

glyph = PieGlyph(["a", "b", "c"], [3.0, 1.0, 2.0])
fig, ax, wedges = glyph.plot(title="Share by category")
```

### Pie with a category legend instead of wedge labels

```python
from cleopatra.glyphs.stats.pie_glyph import PieGlyph

glyph = PieGlyph(["a", "b"], [1.0, 3.0], add_legend=True)
fig, ax, wedges = glyph.plot()
```
