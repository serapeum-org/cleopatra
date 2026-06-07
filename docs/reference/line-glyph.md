# LineGlyph Class

The `LineGlyph` class wraps `Axes.plot`, `Axes.bar`, and `Axes.fill_between` for
**line**, **bar**, and **band** plots. `y` may be 1-D (a single series) or 2-D
`(n_points, n_series)` — `line` and `bar` draw one series per column. Styling
comes from the shared options (`color_1`, `line_width`, `marker`, `linestyle`,
`alpha`).

## Class Documentation

::: cleopatra.line_glyph.LineGlyph
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Examples

### Line plot (single and multi-series)

```python
import numpy as np
from cleopatra.line_glyph import LineGlyph

x = np.linspace(0, 2 * np.pi, 100)

# single series
fig, ax, lines = LineGlyph(x, np.sin(x)).line(title="sin(x)")

# multiple series — one column per line
y = np.column_stack([np.sin(x), np.cos(x)])
fig, ax, lines = LineGlyph(x, y).line(label=["sin", "cos"])
ax.legend()
```

### Bar chart

```python
import numpy as np
from cleopatra.line_glyph import LineGlyph

x = np.arange(5)
fig, ax, bars = LineGlyph(x, np.array([3.0, 5.0, 2.0, 8.0, 4.0])).bar(title="Counts")
```

### Band / envelope (fill_between)

```python
import numpy as np
from cleopatra.line_glyph import LineGlyph

x = np.linspace(0, 10, 50)
upper = np.sin(x) + 0.3
lower = np.sin(x) - 0.3
fig, ax, band = LineGlyph(x, upper).fill_between(y2=lower, alpha=0.3)
```
