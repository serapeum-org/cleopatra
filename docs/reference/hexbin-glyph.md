# HexbinGlyph Class

The `HexbinGlyph` class bins an `(x, y)` point cloud onto a hexagonal lattice and
colours each cell by a per-bin aggregate — a **count** by default, or the
`reduce` of a per-point `values` array (mean, sum, min, max, std, or a callable).
It is the discrete counterpart of [`KDEGlyph`](kde-glyph.md): where the KDE smooths
the cloud into a continuous *density* with a bandwidth, hexbin answers "how many
observations fell here" (or "what is their mean/…") — a value you can read straight
off the colorbar, without the over-plotting an alpha scatter suffers from.

It wraps `matplotlib.axes.Axes.hexbin` and routes the per-bin aggregate through the
shared scalar-mapping pipeline, so `vmin` / `vmax`, `color_scale`, `levels`,
`ticks_spacing`, the `ColorBar` spec and `classify=Classify(...)` behave exactly as
for the other colour-mapped glyphs. The glyph is geometry- and CRS-agnostic: it
takes plain `x` / `y` arrays in the axes' own coordinates.

## Class Documentation

::: cleopatra.glyphs.stats.hexbin_glyph.HexbinGlyph
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Examples

### Per-bin counts

```python
import numpy as np
from cleopatra.glyphs.stats.hexbin_glyph import HexbinGlyph

rng = np.random.default_rng(0)
x = rng.normal(0, 1, 5000)
y = rng.normal(0, 1, 5000)

fig, ax, pc = HexbinGlyph(x, y, gridsize=40).plot(title="Point density")
```

### Per-bin mean of a third variable, dropping sparse bins

```python
depth = x + y
glyph = HexbinGlyph(x, y, depth, reduce="mean", min_count=5)
fig, ax, pc = glyph.plot()
```

### Classified and log-scaled density

```python
from cleopatra.styling.params import Classify
from cleopatra.styling.scaling import ColorScaling

# stepped colorbar by quantile classes
HexbinGlyph(x, y).plot(classify=Classify(scheme="quantiles", k=5))

# log-scaled counts (min_count=1 drops the empty bins that a log scale cannot map)
HexbinGlyph(x, y, min_count=1).plot(color=ColorScaling.log())
```

### Reading the binned result without drawing

```python
# bin centres and the per-bin aggregate, on a throwaway figure
cx, cy, aggregate = HexbinGlyph(x, y, gridsize=30).evaluate()
```
