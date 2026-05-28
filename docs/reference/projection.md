# Projection Module — Static Projected ("Globe") Map Frames

The `cleopatra.projection` module adds a single, stateless helper:
`apply_projection_frame` turns a plain matplotlib `Axes` into a static projected
("globe") frame. It sets the projected limits and equal aspect, draws the projection
**boundary** (the globe's circle, Robinson's rounded rectangle, ...), optionally **clips**
the existing data layers to that boundary, and draws the **graticule** polylines.

The module is **pure matplotlib with no PROJ/CRS dependency**. It only *receives*
already-computed geometry — boundary vertices, graticule polylines, and projected limits —
as plain arrays. Whatever produces the projection (reprojecting data and deriving the
boundary/graticule) lives upstream; cleopatra only renders the result. This keeps the
engine split clean: the upstream owns CRS/PROJ, cleopatra owns matplotlib.

## Usage

```python
import matplotlib
matplotlib.use("Agg")  # any backend; Agg shown for headless rendering
import numpy as np
import matplotlib.pyplot as plt

from cleopatra.projection import apply_projection_frame

# Geometry comes in as plain arrays (here a unit-circle globe outline and a
# meridian); upstream produces the real projected boundary/graticule.
theta = np.linspace(0, 2 * np.pi, 200)
boundary = np.column_stack([np.cos(theta), np.sin(theta)])
meridian = np.column_stack([np.zeros(50), np.linspace(-1, 1, 50)])

fig, ax = plt.subplots()
# plot your (already reprojected) data first so it can be clipped ...
ax.imshow(np.zeros((8, 8)), extent=(-1, 1, -1, 1))

# ... then frame the axes as a globe and clip the data to the boundary
patch = apply_projection_frame(
    ax,
    boundary_xy=boundary,
    xlim=(-1, 1),
    ylim=(-1, 1),
    graticule_lines=[meridian],
)

fig.savefig("globe.png")
```

!!! note
    `apply_projection_frame` performs **no reprojection** — pass already-densified,
    already-projected geometry as `(N, 2)` arrays. It is a **one-shot** helper: each call
    appends a fresh boundary patch and graticule lines, so apply it once per axes (create a
    new axes to re-frame). With `clip_artists=True` (default) every existing
    `ax.images`/`ax.collections`/`ax.lines` artist — and the graticule — is clipped to the
    boundary; pass `clip_artists=False` to leave the data layers unclipped. Style the
    boundary and graticule via `boundary_kw` / `graticule_kw`, which override
    `DEFAULT_BOUNDARY_KW` / `DEFAULT_GRATICULE_KW`.

## Module Documentation

::: cleopatra.projection
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
