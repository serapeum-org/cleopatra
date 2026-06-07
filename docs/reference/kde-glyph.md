# KDEGlyph Class

The `KDEGlyph` class evaluates an isotropic Gaussian kernel-density estimate of
an `(x, y)` point cloud on a regular grid — **numpy only, no scipy** — and draws
it as filled (`shade=True`, the default) or line density contours, coloured
through the shared scalar-mapping pipeline. An optional `clip_path` restricts the
drawn contours.

## Class Documentation

::: cleopatra.kde_glyph.KDEGlyph
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Examples

### Filled density contours

```python
import numpy as np
from cleopatra.kde_glyph import KDEGlyph

rng = np.random.default_rng(0)
x = rng.normal(0, 1, 500)
y = rng.normal(0, 1, 500)

kde = KDEGlyph(x, y)
fig, ax, cs = kde.plot(title="Density")
```

### Line contours and a wider kernel

```python
# shade=False -> line contours; bw_method > 1 widens the kernel
kde = KDEGlyph(x, y, shade=False, bw_method=1.5, levels=12)
fig, ax, cs = kde.plot()
```
