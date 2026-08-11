# Migration guide

This page collects cleopatra's breaking changes and how to update your code. Two migrations are covered:

1. **[Grouped render-parameter objects](#grouped-render-parameter-objects)** — the loose styling keywords on
   `plot` / `animate` / `facet` were replaced by typed *group objects*, and the temporary deprecation shims that
   kept the old keywords working have now been **removed**. Read this section if you hit a
   `ValueError: The given keyword argument:... is not correct`, an `AttributeError` from passing a bare array
   where an object is expected, or a `ValueError` from `facet` saying `figsize` was renamed to `figure_size`.
2. **[Subpackage restructure](#subpackage-restructure)** — an earlier release moved the flat `cleopatra.*`
   modules into `glyphs/` / `styling/` / `basemap/` subpackages (import paths only).

---

## Grouped render-parameter objects

Related styling keywords that `plot` / `animate` (and the other glyphs) used to accept as long lists of loose
arguments are now bundled into small typed objects. During one release the old keywords kept working behind a
`DeprecationWarning`; **those shims are now gone**. Passing a removed keyword no longer warns — it funnels through
cleopatra's strict option validation and raises
`ValueError: The given keyword argument:<name> is not correct, possible parameters are, [...]`. Passing a bare
`(N, 3)` array as `points` raises `AttributeError` instead of being auto-wrapped on the overlay-drawing kinds
(`imshow` / `pcolormesh`); on `contour` / `contourf` the overlay is skipped, so a bare array is ignored rather
than raising — either way, pass a `PointOverlay`. The one exception is
`facet(figsize=...)`: because `figsize` is still a valid glyph option it would otherwise be absorbed silently, so
`facet` raises a targeted `ValueError` telling you to use `figure_size` (see the table below).

There is no automated rewrite for this one: the changes are semantic (loose keywords → object fields), so update
each call site by hand using the tables below.

### Point overlays → `PointOverlay`

`points` now takes a `PointOverlay` (or `None`); the marker/label styling lives on the object.

```python
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph, PointOverlay

# before
glyph.plot(points=arr, point_color="red", point_size=80,
           point_label_color="blue", point_label_size=10)
# after
glyph.plot(points=PointOverlay(arr, color="red", size=80,
                               label_color="blue", label_size=10))
```

| Removed keyword | New `PointOverlay` field |
| --- | --- |
| `points=<array>` (bare) | `points=PointOverlay(<array>)` |
| `point_color` | `color` |
| `point_size` | `size` |
| `point_label_color` (or oldest `pid_color`) | `label_color` |
| `point_label_size` (or oldest `pid_size`) | `label_size` |

### Frame labels → `FrameLabel`

`animate`'s per-frame time-label styling now lives on a `FrameLabel` passed as `frame_label=` (a bare `[x, y]`
list passed positionally is no longer accepted).

```python
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph, FrameLabel

# before
glyph.animate(time, label_location=[0.1, 0.1], label_color="yellow")
# after
glyph.animate(time, frame_label=FrameLabel(location=[0.1, 0.1], color="yellow"))
```

| Removed keyword | New `FrameLabel` field |
| --- | --- |
| `label_location` (or oldest `text_loc`) | `location` |
| `label_color` | `color` |

### Renamed / restructured keywords

| Old | New |
| --- | --- |
| `animate(text_colors=...)` | `animate(cell_value_text_colors=...)` |
| `facet(col_coords=..., row_coords=...)` | `facet(labels=PanelLabels(col=..., row=...))` |
| `facet(figsize=...)` | `facet(figure_size=...)` |
| `ArrayGlyph.no_elem` | `ArrayGlyph.num_domain_cells` |

`PanelLabels` is importable from `cleopatra.glyphs.gridded.array_glyph`. Note that `facet`'s `labels=` names the
per-panel *title* labels (a `PanelLabels`); it is unrelated to the loose `labels` contour-line keyword that now
lives on `Contour` (see the colour/scale/cell groups table below).

### Colour / scale / cell-value groups

The colour-scale, discretisation, cell-value, and preset/relief keywords were already folded into typed group
objects in a prior release; passing them as loose keywords **raises** with a pointer to the object:

| Loose keywords | Group object |
| --- | --- |
| `color_scale`, `gamma`, `line_threshold`, `line_scale`, `bounds`, `midpoint` | `cleopatra.styling.scaling.ColorScaling` |
| `levels`, `labels`, `label_kw` | `cleopatra.styling.params.Contour` |
| `display_cell_value`, `num_size`, `background_color_threshold` | `cleopatra.styling.params.CellValues` |
| `style`, `hillshade` | `cleopatra.styling.params.DataStyle` |

### Colour bars — `cbar_*` still work

The loose `cbar_*` / `ticks_spacing` keywords are **not** removed — they remain valid options and keep working.
Only the `DeprecationWarning` that steered you toward `ColorBar` is gone. The typed
`colorbar=ColorBar(...)` form (`cleopatra.styling.colorbar.ColorBar`) is still preferred and wins when both are
given.

---

## Subpackage restructure

This release reorganises cleopatra's previously flat `cleopatra.*` module layout into three subpackages
(`glyphs/`, `styling/`, `basemap/`) and renames the histogram glyph. It is a **breaking change to import paths
only**.

**Nothing else changed:** every class, function, method, argument, and return value keeps the same name and
behaviour. You only need to update **where you import from** (and rename one class). The package root still
re-exports nothing — you always import from a submodule.

## TL;DR

1. Repoint each `from cleopatra.<module> import ...` to its new subpackage path (table below).
2. Rename the one renamed symbol: `StatisticalGlyph` → `HistogramGlyph` (its module also moved).
3. `cleopatra.config` and `cleopatra.templates` are unchanged.

```python
# before
from cleopatra.array_glyph import ArrayGlyph
from cleopatra.colors import DATA_STYLES
from cleopatra.tiles import add_tiles
from cleopatra.statistical_glyph import StatisticalGlyph

# after
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
from cleopatra.styling.colors import DATA_STYLES
from cleopatra.basemap.tiles import add_tiles
from cleopatra.glyphs.stats.histogram_glyph import HistogramGlyph   # note the class rename
```

## Import path map

Symbol **names are unchanged** — only the module path moves (except the one rename called out below).

### `glyphs/` — the chart-type building blocks

| Old module | New module | Key symbols |
| --- | --- | --- |
| `cleopatra.glyph` | `cleopatra.glyphs.base.glyph` | `Glyph` |
| `cleopatra.animation` | `cleopatra.glyphs.base.animation` | `save_animation`, `SUPPORTED_VIDEO_FORMAT` |
| `cleopatra.hillshade` | `cleopatra.glyphs.base.hillshade` | `shade_grid`, `shade_rgb` |
| `cleopatra.array_glyph` | `cleopatra.glyphs.gridded.array_glyph` | `ArrayGlyph`, `FacetGrid` |
| `cleopatra.mesh_glyph` | `cleopatra.glyphs.gridded.mesh_glyph` | `MeshGlyph` |
| `cleopatra.vector_glyph` | `cleopatra.glyphs.gridded.vector_glyph` | `VectorGlyph` |
| `cleopatra.scatter_glyph` | `cleopatra.glyphs.primitives.scatter_glyph` | `ScatterGlyph` |
| `cleopatra.line_glyph` | `cleopatra.glyphs.primitives.line_glyph` | `LineGlyph` |
| `cleopatra.polygon_glyph` | `cleopatra.glyphs.primitives.polygon_glyph` | `PolygonGlyph` |
| `cleopatra.flow_glyph` | `cleopatra.glyphs.primitives.flow_glyph` | `FlowGlyph` |
| `cleopatra.kde_glyph` | `cleopatra.glyphs.stats.kde_glyph` | `KDEGlyph` |
| `cleopatra.statistical_glyph` | `cleopatra.glyphs.stats.histogram_glyph` | `StatisticalGlyph` → **`HistogramGlyph`** |

### `styling/` — colour, legends, presentation

| Old module | New module | Key symbols |
| --- | --- | --- |
| `cleopatra.styles` | `cleopatra.styling.styles` | `Styles`, `Scale`, `ColorScale`, `MidpointNormalize`, `classify` |
| `cleopatra.colors` | `cleopatra.styling.colors` | `Colors`, `DATA_STYLES`, `resolve_colormap`, `convert_units`, `style_for_parameter` |
| `cleopatra.colorbar` | `cleopatra.styling.colorbar` | `ColorBar` |
| `cleopatra.palettes` | `cleopatra.styling.palettes` | `Palette`, `PaletteKind`, `get_palette` |
| `cleopatra.perceptual` | `cleopatra.styling.perceptual` | `perceptual_colormap`, `make_diverging`, `make_categorical` |
| `cleopatra.data` | `cleopatra.styling.data` | bundled preset JSON assets |

### `basemap/` — networked basemap / CRS helpers

| Old module | New module | Key symbols |
| --- | --- | --- |
| `cleopatra.geo` | `cleopatra.basemap.geo` | `GeoMixin`, `Feature`, `Basemap` |
| `cleopatra.tiles` | `cleopatra.basemap.tiles` | `add_tiles`, `Tile` |
| `cleopatra.reference` | `cleopatra.basemap.reference` | `add_relief`, `add_features`, `natural_earth` |
| `cleopatra.projection` | `cleopatra.basemap.projection` | `apply_projection_frame` |

### Unchanged (still top-level)

| Module | Key symbols |
| --- | --- |
| `cleopatra.config` | `Config`, `is_notebook` |
| `cleopatra.templates` | `publication_map` |

## The one renamed symbol

`StatisticalGlyph` only ever drew histograms (plus boxplots/strip plots), so it was renamed for clarity. Both the
**module** and the **class** changed:

| Old | New |
| --- | --- |
| `from cleopatra.statistical_glyph import StatisticalGlyph` | `from cleopatra.glyphs.stats.histogram_glyph import HistogramGlyph` |

Its constructor, `histogram()` / `boxplot()` / `multiboxplot()` / `stripes()` methods, and all arguments are
unchanged — rename the import and the class references and you are done.

## Automated migration

The change is a pure find-and-replace of module paths. Because some module names are prefixes/substrings of
others (`colors` vs `colorbar`), replace on **word boundaries** and apply the longest keys first. This Python
script rewrites a downstream tree in place:

```python
import re
from pathlib import Path

# old flat module -> new dotted path (relative to `cleopatra.`)
MODULE_MAP = {
    "glyph": "glyphs.base.glyph",
    "animation": "glyphs.base.animation",
    "hillshade": "glyphs.base.hillshade",
    "array_glyph": "glyphs.gridded.array_glyph",
    "mesh_glyph": "glyphs.gridded.mesh_glyph",
    "vector_glyph": "glyphs.gridded.vector_glyph",
    "scatter_glyph": "glyphs.primitives.scatter_glyph",
    "line_glyph": "glyphs.primitives.line_glyph",
    "polygon_glyph": "glyphs.primitives.polygon_glyph",
    "flow_glyph": "glyphs.primitives.flow_glyph",
    "kde_glyph": "glyphs.stats.kde_glyph",
    "statistical_glyph": "glyphs.stats.histogram_glyph",
    "styles": "styling.styles",
    "colors": "styling.colors",
    "colorbar": "styling.colorbar",
    "palettes": "styling.palettes",
    "perceptual": "styling.perceptual",
    "data": "styling.data",
    "geo": "basemap.geo",
    "tiles": "basemap.tiles",
    "reference": "basemap.reference",
    "projection": "basemap.projection",
}
# longest-first so no key shadows another; word-boundary anchored both ends
ALT = "|".join(sorted(map(re.escape, MODULE_MAP), key=len, reverse=True))
DOTTED = re.compile(r"\bcleopatra\.(" + ALT + r")\b")

for f in Path("your_package").rglob("*.py"):
    text = f.read_text(encoding="utf-8")
    new = DOTTED.sub(lambda m: "cleopatra." + MODULE_MAP[m.group(1)], text)
    new = new.replace("StatisticalGlyph", "HistogramGlyph")  # the one class rename
    if new != text:
        f.write_text(new, encoding="utf-8")
```

Notes:

- Add `*.ipynb` to the glob if your notebooks import cleopatra.
- The script only rewrites the dotted `cleopatra.<module>` form. If you used
  `from cleopatra import tiles, reference`, rewrite those by hand to
  `from cleopatra.basemap import tiles, reference` (`__version__` stays `from cleopatra import __version__`).
- After running it, search for any leftover `cleopatra.statistical_glyph` or `StatisticalGlyph`.

## Packaging note (no code change)

`pillow` moved from the optional `cleopatra[tiles]` extra to a core dependency (the animation writer now imports
it at module load). It was already installed transitively via matplotlib, so nothing changes for you — but if you
were installing `cleopatra[tiles]` *only* to get Pillow, plain `cleopatra` now suffices.

## What did **not** change

- No function/method signatures, argument names, defaults, or return types changed.
- No rendering/behaviour changed (the restructure was verified to preserve every module's behaviour).
- `cleopatra.config` and `cleopatra.templates` keep their paths.
- The package root (`import cleopatra`) still deliberately re-exports nothing — always import from a submodule.
