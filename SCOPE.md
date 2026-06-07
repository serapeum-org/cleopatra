# Cleopatra — Package Scope

This document defines **what Cleopatra is and is not**, so that any contributor
(human or LLM) can decide whether a proposed feature belongs here *before*
writing it. If a request falls outside this scope, it should be declined or
redirected, not implemented.

## One-sentence scope

Cleopatra is a **high-level, matplotlib-only convenience layer** for visualizing
**in-memory NumPy data** — 2D/3D raster arrays, unstructured meshes, point
clouds, vector fields, polygons, lines, and statistical distributions — with
sensible scientific defaults, a shared colour-mapping/colorbar pipeline, and
animation export.

## Audience and design center

- **Users:** scientific / research users working with geospatial and raster
  data who want a one-call plot with good defaults over raw matplotlib.
- **Inputs:** plain `numpy` arrays (and array-likes). The user brings the data
  already loaded into memory.
- **Outputs:** matplotlib `Figure` / `Axes` / artist objects (returned, not
  hidden), plus animations exported to GIF/MP4/MOV/AVI.
- **Backend:** matplotlib only. Cleopatra never changes the active backend on
  import; `config.Config.set_matplotlib_backend()` is opt-in.

## In scope

### The glyph family (data → matplotlib artist + colorbar)

All glyphs share a `Glyph` base class providing the figure/axes lifecycle,
colour norms (`linear`, `power`, `sym-lognorm`, `boundary-norm`, `midpoint`),
colorbars, ticks, classification, and animation.

| Module | Class | Visualizes |
| --- | --- | --- |
| `array_glyph` | `ArrayGlyph`, `FacetGrid` | 2D/3D NumPy raster arrays; cell-value display, point overlays, RGB, faceting, animation |
| `mesh_glyph` | `MeshGlyph` | UGRID-style unstructured meshes via triangulation (`tripcolor`/`tricontourf`), wireframe outlines, contour labels, animation |
| `statistical_glyph` | `StatisticalGlyph` | Histograms (1D/2D), boxplots, multiboxplots, strip plots |
| `scatter_glyph` | `ScatterGlyph` | 2D point clouds; colour + independent size encoding, size legend |
| `vector_glyph` | `VectorGlyph` | 2D vector fields as arrows, wind barbs, or streamlines; magnitude colouring |
| `flow_glyph` | `FlowGlyph` | Magnitude-coloured, width-scaled flow polylines |
| `line_glyph` | `LineGlyph` | Line, bar, and fill-between (band) plots |
| `polygon_glyph` | `PolygonGlyph` | Filled / outlined polygon collections, value-coloured |
| `kde_glyph` | `KDEGlyph` | 2D Gaussian kernel-density contours (NumPy-only, no scipy) |

### Supporting utilities

- `colors` (`Colors`): convert hex ↔ RGB-255 ↔ RGB-normalized, extract colour
  ramps from images, build matplotlib colormaps.
- `styles`: predefined line/marker styles, `Scale` transforms, `ColorScale`
  enum, `MidpointNormalize`, value→size mapping, classification (`classify`:
  quantiles, equal-interval, percentiles, std-mean, Fisher-Jenks/natural-breaks
  — all NumPy-native), and reusable legend builders (disjoint/size/width/
  histogram/colorbar).
- `tiles` (optional `cleopatra[tiles]` extra): fetch + stitch XYZ web-tile
  basemaps; reprojection helpers. The **only** networked feature.
- `projection`: lightweight axes-frame / coordinate helpers.
- `animation`: turn a matplotlib `FuncAnimation` into a saved file, GIF bytes,
  or an embeddable IPython image (via ffmpeg).
- `config` (`Config`): opt-in matplotlib-backend selection; notebook detection.

### What new work generally belongs here

- New glyph types or plot kinds that follow the **"NumPy in → matplotlib
  Figure/Axes out"** contract and reuse the shared `Glyph` pipeline.
- New colour scales, classification schemes, legends, or styling that plug into
  the existing scalar-mapping pipeline.
- New animation export targets or colour-conversion helpers.
- Better defaults, customization knobs, and matplotlib-composition support
  (shared axes, `add_colorbar=False`, passing in existing `ax`/`fig`).

## Out of scope (decline or redirect)

- **Non-matplotlib backends / engines:** Plotly, Bokeh, Altair, pyvista,
  datashader, OpenGL, web/JS rendering. Cleopatra is matplotlib-only.
- **Data I/O and formats:** reading/writing GeoTIFF, NetCDF, shapefiles,
  GeoJSON, CSV, databases. Users bring NumPy arrays already; file/raster I/O
  belongs in sibling packages (e.g. `pyramids`), not here.
- **GIS / geoprocessing:** reprojection of user data, clipping, resampling,
  zonal stats, CRS management beyond what the optional `tiles` basemap needs.
- **Interactive / GUI apps:** dashboards, widget servers, event callbacks,
  click-to-edit tooling, real-time streaming.
- **Numerical / statistical modelling:** curve fitting, regression, ML,
  forecasting, hypothesis tests — Cleopatra *displays* results, it doesn't
  compute models. (KDE is the deliberate, self-contained exception.)
- **Heavy new hard dependencies:** keep the core to `numpy` + `matplotlib`
  (plus `ffmpeg-python`, `hpc-utils`). Anything bigger must be an optional
  extra like `tiles`, and only with strong justification.
- **3D rendering** (mplot3d surfaces/volumes), networked data sources other
  than `tiles`, and general-purpose plotting that matplotlib already does well
  without added value.

## Boundary heuristic for a feature request

Ask, in order:

1. **Input** — does it start from in-memory NumPy data (not a file/CRS/URL)?
2. **Output** — does it produce a matplotlib `Figure`/`Axes`/artist (or an
   animation of one)?
3. **Reuse** — can it build on `Glyph` and the shared colour/colorbar/legend
   pipeline rather than a parallel stack?
4. **Dependencies** — does it stay within numpy + matplotlib, or fit cleanly as
   an optional extra?

If all four are "yes," it likely belongs in Cleopatra. If any is "no," it
probably belongs in a different package (data I/O → `pyramids`; modelling →
elsewhere) and should be declined here.
