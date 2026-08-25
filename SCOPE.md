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
| `histogram_glyph` | `HistogramGlyph` | Histograms (1D/2D), boxplots, multiboxplots, strip plots |
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
  basemaps; reprojection helpers.
- `reference` (uses the `cleopatra[tiles]` extra for relief decoding /
  reprojection): fetch + draw fixed public reference-basemap data *under* your
  plot — Natural Earth vector layers (`add_features`) and a global hypsometric
  relief raster (`add_relief`), the `cartopy` `ax.coastlines()` /
  `GeoAxes.stock_img()` niche. It acquires only **fixed public datasets** that
  cleopatra re-hosts as dependency-light artifacts (gzipped GeoJSON / PNG);
  it reads no user files and never imports GDAL/geopandas.
- `tiles` and `reference` are cleopatra's **only** networked features.
- `projection`: lightweight axes-frame / coordinate helpers.
- `animation`: turn a matplotlib `FuncAnimation` into a saved file, GIF bytes,
  or an embeddable IPython image (via ffmpeg), and derive one output format from
  another (`gif_from_video`) — see "Animation output".
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
- **Data I/O and formats:** reading/writing *user* GeoTIFF, NetCDF,
  shapefiles, GeoJSON, CSV, databases. Users bring NumPy arrays already;
  file/raster I/O of user data belongs in sibling packages (e.g. `pyramids`),
  not here. Three deliberate exceptions: the `tiles` / `reference` basemap
  helpers, which fetch a handful of *fixed public* reference datasets (never
  user data) that cleopatra re-hosts as dependency-light artifacts — see
  "Supporting utilities"; reading a **presentation asset** — a logo / watermark
  image for `styling.watermark.stamp_mark` — which is decoration on the rendered
  figure, not user data, and loads via Pillow (an existing dependency), never
  GDAL/geopandas; and re-encoding cleopatra's **own animation output** between
  formats — see "Animation output" below.
- **GIS / geoprocessing:** reprojection of user data, clipping, resampling,
  zonal stats, CRS management beyond what the optional `tiles` basemap needs.
- **Interactive / GUI apps:** dashboards, widget servers, event callbacks,
  click-to-edit tooling, real-time streaming.
- **Numerical / statistical modelling:** curve fitting, regression, ML,
  forecasting, hypothesis tests — Cleopatra *displays* results, it doesn't
  compute models. (KDE is the deliberate, self-contained exception.)
- **Heavy new hard dependencies:** keep the core to `numpy` + `matplotlib`
  (plus `imageio-ffmpeg`, `hpc-utils`). Anything bigger must be an optional
  extra like `tiles`, and only with strong justification. `imageio-ffmpeg` is
  the deliberate exception: it replaces the pure-Python-but-unused
  `ffmpeg-python` and bundles a static FFmpeg binary so `save_animation` can
  export MP4/MOV/AVI out of the box (issue #185), which pure-Python packaging
  cannot provide.
- **3D rendering** (mplot3d surfaces/volumes), networked data sources other
  than the `tiles` / `reference` basemap helpers, and general-purpose plotting
  that matplotlib already does well without added value.

## Animation output

Rendering frames is by far the most expensive part of an animation — hours, for a
long scientific clip — while encoding them is cheap. Forcing every output format
to be produced from a live `FuncAnimation` therefore means re-rendering the same
frames once per format, which is the wrong trade at any real size.

So a helper may **read back a rendered video** and re-encode it to another
supported format, as `gif_from_video` does. The intended input is cleopatra's own
output, produced by `save_animation` moments earlier. Nothing in the code
enforces that — it decodes whatever FFmpeg can read, and a check would buy
nothing but a worse error message — so this is a statement of *purpose*, not a
guarantee about the argument.

What keeps it inside the line is what it does not do: it reads no user dataset in
an analytical format, opens no GIS format, exposes no transcoding matrix, and
adds no dependency — the FFmpeg it decodes with is the one `save_animation`
already encodes with. What it returns is an animation, not data.

The boundary that still holds: cleopatra does not become a general
media-conversion tool. A helper whose purpose was ingesting arbitrary user video,
or that grew a codec/container matrix, would be out of scope.

## Boundary heuristic for a feature request

Ask, in order:

1. **Input** — does it start from in-memory NumPy data (not a file/CRS/URL)?
   (Three deliberate exceptions: the `tiles` / `reference` basemap helpers,
   which acquire fixed *public* reference data, never user files; a presentation
   asset such as a logo for `stamp_mark`; and the animation re-encoders, whose
   input is cleopatra's own output — see "Animation output".)
2. **Output** — does it produce a matplotlib `Figure`/`Axes`/artist (or an
   animation of one)?
3. **Reuse** — can it build on `Glyph` and the shared colour/colorbar/legend
   pipeline rather than a parallel stack?
4. **Dependencies** — does it stay within numpy + matplotlib, or fit cleanly as
   an optional extra?

If all four are "yes," it likely belongs in Cleopatra. If any is "no," it
probably belongs in a different package (data I/O → `pyramids`; modelling →
elsewhere) and should be declined here.
