# Furniture — Scale Bar & North Arrow

The `cleopatra.styling.furniture` module draws the two remaining pieces of standard chart
furniture cleopatra did not have: a **scale bar** and a **north arrow**. Both are free
functions that decorate an existing `matplotlib.axes.Axes` and return the frameless inset axes
they drew on, exactly like `stamp_mark` — so they read alike and anchor identically (they share
`stamp_mark`'s corner-placement plumbing) and use the same `box` / `label_location` /
`label_size` vocabulary as `ColorBar`.

```python
from cleopatra.styling.furniture import add_scale_bar, add_north_arrow
```

The key design point is the **package boundary**: these are plain matplotlib artistry and know
nothing about geography. A scale bar is `length` axes **data units** wide with a caller-supplied
`label` string; a north arrow is rotated by a caller-supplied `rotation` in degrees. The
ellipsoidal questions — *how long is 100 km in axis units at this latitude in this projection*,
*what is the grid convergence here* — belong to whoever owns the CRS (the consumer), never to
this generic layer. So a micrograph with a µm bar, a floor plan, an engineering section and a map
all use the identical artist. There is no `pyproj` import here and no CRS logic of any kind.

Both draw the bar / arrow on a frameless inset axes in **axes-fraction** coordinates, so the
furniture stays anchored in its corner across a dpi or limits change rather than drifting like a
data-coordinate `Rectangle`, and sits at a high zorder above the data.

## Scale bar

```python
import matplotlib.pyplot as plt
import numpy as np
from cleopatra.styling.furniture import add_scale_bar, ScaleBar

fig, ax = plt.subplots()
ax.imshow(np.random.default_rng(0).random((100, 100)), extent=[0, 500_000, 0, 500_000])

# the consumer computes the ground distance in axis units; cleopatra draws it
add_scale_bar(ax, 100_000, ScaleBar(label="100 km", location="lower left", segments=4, box=True))
```

`length` is in the axes' **x data units**; all the presentation options are grouped into a
`ScaleBar` object (mirroring `FacetLayout` / `ColorBar`), so the call stays small. `label` is the
caption you supply (defaulting to `f"{length:g}"`); `segments` sets the number of alternating
blocks (`1` draws a plain bar); `ticks=True` (default) numbers the block boundaries `0 .. length`,
a sequence numbers those data positions, and `False` draws no numbers. `location` is one of the
four corners; `pad`, `height` and the two colours are axes-fraction / matplotlib values;
`label_location` (`"bottom"` / `"top"` / `None` for the interior side) picks the caption side;
`box` adds a backing panel (`True`, a colour, or a dict of `Rectangle` kwargs).

## North arrow

```python
from cleopatra.styling.furniture import add_north_arrow, NorthArrow

# rotation (grid convergence) is the caller's to supply — cleopatra never derives it
add_north_arrow(ax, 0.0, NorthArrow(location="upper right", style="arrow"))
```

`rotation` is degrees clockwise from up (e.g. the grid convergence at the map centre) and stays a
direct argument (like `length` on the scale bar); the arrow and its `"N"` label rotate together.
The presentation options are grouped into a `NorthArrow` object: `style` is `"arrow"` (a single
filled arrow), `"needle"` (a two-tone compass needle) or `"rose"` (a four-point compass star);
`size`, `location`, `pad`, `label`, the colours and `box` mirror `ScaleBar`.

## GeoMixin sugar

The six geographic glyphs (`ArrayGlyph`, `MeshGlyph`, `VectorGlyph`, `FlowGlyph`, `PolygonGlyph`,
`ScatterGlyph`) expose thin methods next to `add_tiles` / `add_features` / `add_labels`, so you
can decorate a glyph without importing the free functions or repeating the axes:

```python
glyph.plot()
glyph.add_scale_bar(100_000, ScaleBar(label="100 km", location="lower left"))
glyph.add_north_arrow(grid_convergence_deg, NorthArrow(style="needle"))
```

The free functions stay the API; the methods only supply the glyph's axes.

::: cleopatra.styling.furniture.ScaleBar

::: cleopatra.styling.furniture.add_scale_bar

::: cleopatra.styling.furniture.NorthArrow

::: cleopatra.styling.furniture.add_north_arrow
