# Perceptual colour toolkit

The `cleopatra.styling.perceptual` module builds good colormaps the way scientific palette
libraries do — by working in a **perceptually-uniform colour space** rather than
RGB — using only `numpy` and `matplotlib` (no extra dependency, and nothing
imported from or copied out of cmocean / cmcrameri / colorcet).

matplotlib interpolates colormaps in RGB, which is perceptually non-uniform: equal
data steps map to visually uneven steps, so hand-authored ramps band and have dead
zones. The one primitive here — a closed-form `sRGB ↔ CIELAB` transform — fixes
that, and everything else is built on top of it.

For scientific-grade **sequential** and **cyclic** maps, prefer matplotlib's own
`viridis` family and `twilight` (already optimised in CAM02-UCS); this toolkit
earns its keep on **diverging**, **categorical**, and smoothing bespoke domain
ramps.

## Colour-space transform

The pure-numpy `sRGB ↔ CIELAB` conversion that underlies everything else.

::: cleopatra.styling.perceptual.srgb_to_lab
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: cleopatra.styling.perceptual.lab_to_srgb
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Perceptual interpolation

Interpolate colour anchors in CIELAB (at uniform perceptual arc-length), so a ramp
progresses evenly to the eye. `perceptual_colormap` is a drop-in, perceptually-even
replacement for `matplotlib.colors.LinearSegmentedColormap.from_list`.

::: cleopatra.styling.perceptual.interp_perceptual
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: cleopatra.styling.perceptual.perceptual_colormap
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Generators

Build a perceptually-uniform **diverging** map from two endpoint colours, or a set
of maximally-distinguishable **categorical** colours (the glasbey max-min method) —
both from scratch, no colour data required.

::: cleopatra.styling.perceptual.make_diverging
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: cleopatra.styling.perceptual.make_categorical
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Diagnostic

Score how perceptually even a colormap's steps are — useful for comparing an
RGB-interpolated ramp against its `interp_perceptual` counterpart.

::: cleopatra.styling.perceptual.perceptual_uniformity
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Examples

### A smoother domain ramp

```python
import numpy as np
import matplotlib.pyplot as plt

from cleopatra.styling.perceptual import perceptual_colormap, perceptual_uniformity

anchors = ["#ffffff", "#ff6a00", "#7a1500", "#2a0800"]  # a "dust" ramp
cmap = perceptual_colormap("dust", anchors)

# far more perceptually even than an RGB LinearSegmentedColormap.from_list
print(perceptual_uniformity(cmap))  # ~0.02 (RGB build scores ~0.18)

fig, ax = plt.subplots(figsize=(6, 1))
ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), aspect="auto", cmap=cmap)
ax.set_axis_off()
```

### A diverging map and a categorical palette

```python
from cleopatra.styling.perceptual import make_diverging, make_categorical

diverging = make_diverging("#762a83", "#1b7837")  # purple ↔ green, Lab-balanced
classes = make_categorical(12)                     # 12 distinguishable class colours
```
