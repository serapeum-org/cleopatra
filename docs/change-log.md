# Changelog

## 0.21.0 (2026-07-06)


- fix(animation): accept os.PathLike in save_animation and create_from_image (#181)
- save_animation derived the output format with str.rsplit, so passing a                                              
  pathlib.Path (idiomatic from pyramids' Path-everywhere Dataset API)                                                 
  raised AttributeError: 'WindowsPath' object has no attribute 'rsplit'.                                              
                                                                                                                      
  - Normalise paths with os.fspath and derive the extension via                                                       
    os.path.splitext, so both str and os.PathLike work.                                                               
  - Widen the type hints to str | os.PathLike on animation.save_animation,                                            
    ArrayGlyph.save_animation, and Colors.create_from_image (the only two                                             
    public path-taking APIs; reference/tiles handle internal paths only).                                             
  - Give a clear error when the path has no file extension, and lock the                                              
    tightened dotfile / extension-less rejection (.gif, dir/.gif, bare                                                
    gif) with a regression test.                                                                                      
  - Cover PathLike on the happy and error branches of every widened API.                                              
                                                                                                                      
  Closes #180

## 0.20.0 (2026-06-26)


- feat(geo)!: glyph axis CRS with defaulting and assignment-time validation (#178)
- Let the geographic glyphs carry the CRS of their plotted data so reference                      
  layers default to it, and make bad values fail fast.                                            
                                                                                                  
  - `GeoMixin` gains a `crs` property (default `None`). `add_features` and                        
    `add_tiles` default their `crs=` to `self.crs` when omitted, so a caller                      
    that records the axis CRS once (`glyph.crs = 4326`) gets correctly placed                     
    layers without restating it; an explicit `crs=` still wins and                                
    `self.crs is None` is a pure pass-through.                                                    
  - `crs` lives on `GeoMixin`, not the base `Glyph`, so non-geographic glyphs                     
    are unaffected. `add_relief` is excluded -- it has no `crs` parameter                         
    (relief is a fixed EPSG:4326 raster placed by `extent`).                                      
  - The `crs` setter validates on assignment: `TypeError` for non                                 
    int/str/None (bool rejected), `ValueError` for a non-positive EPSG code,                      
    an empty string, or -- when `pyproj` is installed -- an unresolvable CRS.                     
    Strings are stripped and a bare numeric string (`"4326"`) is normalised                       
    to the int `4326`. Setting `crs` never requires the `[tiles]` extra; the                      
    deep check is skipped (deferred to draw time) without `pyproj`.                               
  - Make `add_tiles` options keyword-only after `source` (matching                                
    `add_features`), so `crs` can no longer be passed positionally and the                        
    default injection is unconditionally safe.                                                    
  - Hoist the `cleopatra.tiles` / `cleopatra.reference` imports to module top                     
    and call via the module, removing the inline imports.                                         
                                                                                                  
  BREAKING CHANGE: `cleopatra.tiles.add_tiles` now accepts `crs`, `zoom`,                         
  `alpha`, `attribution`, `zorder`, `interpolation`, `timeout`, `retries`,                        
  `user_agent` and `max_tiles` as keyword-only arguments (everything after                        
  `source`). Callers that passed any of these positionally must switch to                         
  keyword arguments; `ax` and `source` remain positional.                                         
                                                                                                  
  Closes #177

## 0.19.0 (2026-06-22)


- feat(geo): glyph convenience methods for basemaps (GeoMixin) (#172)
- Add cleopatra.geo.GeoMixin so the glyphs that plot geographic data can
  drop a basemap straight from the glyph, plus the supporting reference
  docs.
-   - GeoMixin exposes add_tiles / add_features / add_relief; each draws on
    the glyph's own axes (or an explicit ax=) and forwards all arguments
    to the standalone cleopatra.tiles / cleopatra.reference functions,
    which stay the single source of truth.
  - Basemap modules are imported lazily, so importing a glyph never pulls
    in the optional [tiles] extra.
  - ArrayGlyph, MeshGlyph, VectorGlyph, FlowGlyph, PolygonGlyph and
    ScatterGlyph inherit the mixin; the chart/statistical glyphs
    (LineGlyph, StatisticalGlyph, KDEGlyph) deliberately do not.
  - docs: add a reference example notebook (network/[tiles] cells tagged
    nbval-skip), a geo.md page, embedded equal-aspect screenshots, and a
    discovery-helper note; correct the polygon note to PathCollection.
  - Add tests covering inheritance gating, delegation, the ax= override,
    and the no-axes error.
-   - Closes #173
  - Closes #174

## 0.18.0 (2026-06-14)


- fix(changelog): generate on bump and backfill 0.8.0-0.17.0
- The release flow runs cz bump, which only writes the changelog when
update_changelog_on_bump is set. That key was missing from the
[tool.commitizen] config, so docs/change-log.md froze at 0.7.1 while
0.8.0-0.17.0 shipped (their notes only reached the GitHub Releases page).
- - Add update_changelog_on_bump = true so cz bump regenerates and commits
  the changelog on every future release.
- Backfill the missing 0.8.0-0.17.0 sections (regenerated with commitizen
  from the release commits on main; the 0.1.0-0.7.1 history is unchanged).
- feat(array_glyph): animate RGB / true-colour stacks (#169)
- ArrayGlyph.animate previously handled only 2-D single-band frames, so
  producing an RGB time-lapse meant abandoning cleopatra and hand-rolling
  matplotlib's FuncAnimation. It now also accepts a 4-D
  (time, rows, cols, 3|4) RGB/RGBA stack:
-   - 4-D stacks render each frame through imshow as true colour, with no
    norm/colormap/colorbar (self.cbar stays None), mirroring plot()'s RGB
    branch.
  - The lazy data_getter path accepts RGB frames whose spatial dims (first
    two axes) match the template's last two axes.
  - display_cell_value and background_color_threshold are skipped for RGB
    frames (per-cell annotation needs a scalar field).
  - The 3-D single-band path is unchanged; the no-time-axis error now
    names both the 3-D and 4-D accepted shapes.
-   Add TestAnimateRGB covering 4-D RGB/RGBA stacks, GIF rendering, lazy RGB
  frames, bad-shape errors, display_cell_value being ignored, and the
  unchanged single-band path. Update the animate docstring/doctest and add
  a docs/notebooks/array_glyph/rgb_animation.ipynb example wired into the
  mkdocs nav.
-   Closes #168

## 0.17.0 (2026-06-07)


- feat(reference): add Natural Earth and relief basemap helpers (#166)
- Add `cleopatra.reference`, a matplotlib map-decoration layer that draws
  fixed public reference data under a plot — the cartopy
  `ax.coastlines()` / `GeoAxes.stock_img()` niche and the vector/raster
  sibling of `cleopatra.tiles`.
-   - `add_features` (Natural Earth coastline/borders/land/ocean/rivers/
    lakes across 110m/50m/10m) and `add_relief` (global hypsometric
    backdrop, low/medium) axes helpers, plus raw `natural_earth` /
    `relief` accessors and discovery helpers.
  - Assets are fixed public datasets re-hosted on a cleopatra-owned
    `basemap-data-v1` release (gzipped GeoJSON + PNG); no GDAL/geopandas
    and no dependency on pyramids. Features in EPSG:4326 need only
    numpy+matplotlib; relief decode and `crs=` reprojection use the
    existing `[tiles]` extra.
  - Downloads are http(s)-only, streamed atomically, cached under
    `~/.cleopatra/naturalearth` (override `CLEOPATRA_CACHE_DIR`), and
    self-heal a corrupt/poisoned cache.
  - Polygon layers render hole-aware via compound paths (nonzero fill), so
    `ocean` continent cut-outs are correct; reprojection drops non-finite
    vertices at projection singularities; invalid `crs` raises a clear
    `ValueError`.
  - Add `tools/build_basemap_assets.py`, the offline maintainer script that
    builds the artifacts from upstream Natural Earth shapefiles and relief
    GeoTIFFs.
  - Document usage and migration from `pyramids.basemap.natural_earth` /
    `relief`; amend `SCOPE.md` to record the `tiles`/`reference` basemap
    helpers as the deliberate networked exception.
  - Tests reach 100% line and branch coverage of `cleopatra.reference`
    (50 tests) with 47 passing doctests.
-   ref: #165

## 0.16.0 (2026-06-06)


- feat(glyphs ): add classification scheme, value-to-size, KDEGlyph, FlowGlyph (#162)
- Consolidates the geoplot/Digital-Earth upstream tasks (#154–#157) into the
  shared glyph subsystem, all pure numpy + matplotlib with no new dependency.
-   - Categorical colour `scheme` (quantiles, equal_interval, percentiles,
    std_mean, explicit edges, and a native Fisher-Jenks natural-breaks
    optimisation) via `styles.classify` wired into the shared
    `Glyph._prepare_scalar_mapping`, so every norm-driven glyph
    (Scatter/Polygon/Vector/Flow) gains discrete classes and a stepped
    colorbar. Array/Mesh/KDE reject `scheme` rather than ignore it.
  - `ScatterGlyph` value-to-size scaling with an optional size legend,
    factored into the reusable `styles.resolve_sizes` helper.
  - `FlowGlyph`: magnitude-coloured, width-scaled `LineCollection` for
    flow/Sankey maps, reusing `resolve_sizes` for line width.
  - `KDEGlyph`: numpy-only 2-D Gaussian KDE drawn as filled/line contours
    with an optional clip path and memory-chunked evaluation.
  - Fisher-Jenks runs natively (exact O(k·n^2) DP, mean-centred for numeric
    stability) and falls back to a quantile sample above `MAX_JENKS_N`.
-   ref: #154, #155, #156, #157

## 0.15.0 (2026-06-02)


- feat(mesh_glyph): inline labels for line tricontours via labels argument (#152)
- Add `labels` (bool) and `label_kw` (dict) options to `MeshGlyph.plot`
  so node line tricontours can carry inline numeric labels through
  `ax.clabel`, the unstructured mirror of `ArrayGlyph`'s contour labels
  (#148/#149). The `TriContourSet` mappable was already returned, but
  every caller had to re-roll the same `ax.clabel` glue.
-   - labels=True (location="node", filled=False) draws inline labels
    (defaults inline=True, fontsize=8, fmt="%g") and stores the Text
    artists on self.contour_labels
  - label_kw is merged over those defaults and forwarded to ax.clabel,
    so user keys win on collision
  - documented no-op for tripcolor (face data) and tricontourf
    (filled=True); a labelled line set with no isolines yields an empty
    list
  - contour_labels resets to None on every plot() and animate() render,
    so re-plotting without labels (or switching to filled/animation)
    clears stale label artists
  - complete the node_x/node_y/n_faces/n_nodes/n_edges property
    docstrings and add TestContourLabels plus coverage for the
    render/animate option branches (mesh_glyph coverage 96% -> 98%)
-   Closes #151

## 0.14.0 (2026-06-02)


- feat(array_glyph): inline contour labels via plot(kind="contour", labels=True) (#149)
- feat(array_glyph): inline contour labels via plot(kind="contour", labels=True)
  
  Add `labels` (bool) and `label_kw` (dict) options to ArrayGlyph.plot so
  line contours can carry inline numeric labels through ax.clabel, the way
  ECMWF Magics / cartopy / earthkit-plots label isolines. Previously the
  QuadContourSet was returned as the mappable but every caller had to
  re-roll the same ax.clabel glue.
-   - labels=True draws inline labels (defaults inline=True, fontsize=8,
    fmt="%g") and keeps the Text artists on self.contour_labels.
  - label_kw is merged over those defaults and forwarded to ax.clabel,
    so user keys win on collision.
  - labels is a documented no-op for contourf and every non-contour kind;
    a labelled contour with no isolines yields an empty list.
  - self.contour_labels resets to None on each render, so re-plotting
    without labels (or switching kind) clears stale label artists.
-   Docstring section + two doctests on plot(); TestContourLabels adds 8
  cases covering draw/expose, default no-op, contourf no-op, both reset
  paths, the degenerate empty-list case, and label_kw forwarding and
  precedence.
-  ref: #148.

## 0.13.0 (2026-06-01)


- feat(animation): glyph-independent save/embed helpers for any FuncAnimation (#145)
- Expose cleopatra's animation save/embed machinery as glyph-independent
  free functions in a new `cleopatra.animation` module, so downstream
  packages and notebooks can reuse the writer/format handling on any
  matplotlib `FuncAnimation` instead of re-rolling temp-file + writer +
  `IPython.display` glue.
-   - add `save_animation(anim, path, fps=2)`, `to_gif(anim, fps=2)`, and
    `embed_gif(anim, fps=2)`; the writer is chosen from the file extension
    (gif via PillowWriter, mov/avi/mp4 via FFMpegWriter)
  - `Glyph.save_animation` now delegates to the free function, removing the
    duplicated writer logic; `SUPPORTED_VIDEO_FORMAT` moves to the new
    module and is re-exported from `cleopatra.glyph` for back-compat
  - match the file extension case-insensitively (`out.GIF` works)
  - raise actionable errors: a missing FFmpeg points at ffmpeg.org, and a
    missing IPython points at `pip install ipython` (and `to_gif`), while a
    missing IPython sub-dependency is surfaced unchanged
  - import IPython lazily so importing cleopatra never requires it
  - add an API reference page and register the `animation` submodule in the
    package surface
  - cover the module with unit tests (100% line + branch) and executable
    doctests
-   ref: #144

## 0.12.0 (2026-05-29)


- feat(projection): add apply_projection_frame for static projected map frames (#142)
- Add a stateless, PROJ-free helper that turns a plain matplotlib Axes into
  a static projected ("globe") frame: it sets equal aspect and projected
  limits, draws the projection boundary as a PathPatch, draws graticule
  polylines, and optionally clips existing data layers to the boundary.
-   All geometry (boundary vertices, graticule polylines, limits) is supplied
  as plain (N, 2) arrays, so the module has no PROJ/CRS dependency -- the
  upstream engine owns reprojection, cleopatra owns matplotlib.
-   - add cleopatra/projection.py with apply_projection_frame and the _as_xy
    input-coercion helper, validating axes, limits, and array shapes
  - register the projection submodule in the package docstring and the
    package-surface allowlist test
  - add a full test suite (100% line + branch coverage) plus an in-band
    doctest runner so the module examples are exercised by the default run
  - add the projection reference page and wire it into the mkdocs nav
-   Closes #141

## 0.11.0 (2026-05-28)


- feat(glyph): colorbar toggles, mesh mappable, per-band RGB stretch, flat-data guard (#136)
- Close the remaining cleopatra composition gaps used by the Digital-Earth port.
-   - Add an `add_colorbar` option (default True) to ScatterGlyph/PolygonGlyph/
    VectorGlyph and a plot-time override, so a shared-axes host can own one
    aggregated colorbar instead of one per layer. (MeshGlyph already exposes a
    `colorbar=` toggle; LineGlyph draws no colorbar.)
  - Expose the tripcolor/tricontour(f) artist as `MeshGlyph.im` (cleared by
    `plot_outline`), mirroring `ArrayGlyph.im` and the other glyphs.
  - Add a per-band percentile stretch to `ArrayGlyph.scale_to_rgb`
    (`per_band=True`, default cut `(2, 98)`); the default global-max path is
    unchanged, guards an all-zero array, and maps NaN/flat bands to a flat zero
    band without warning.
  - Fix constant-value arrays raising `ZeroDivisionError`: `Glyph.get_ticks`
    returns a single tick for a degenerate range, `ArrayGlyph` falls back to a
    unit `ticks_spacing`, and a constant-field line `contour` skips its empty
    colorbar with a warning.
-   ref: #61, #137, #138, #139

## 0.10.1 (2026-05-27)


- perf(mesh): vectorize fan triangulation and edge derivation (#134)
- Replace the Python loops in MeshGlyph._fan_triangles and
  MeshGlyph._build_edge_segments with vectorized numpy index
  manipulation, removing the performance bottleneck on large
  mixed-element meshes (>100k faces).
-   - _fan_triangles: compact valid nodes in face order and build fan
    triangles with np.repeat plus fancy indexing, via a new
    _grouped_arange helper (handles zero-size groups); no per-face
    Python loop.
  - _build_edge_segments: derive polygon edges with wrap-around
    indexing and deduplicate undirected edges via an int64 key encoding
    plus sort/diff instead of a Python set.
  - Fix a latent bug where a pure-triangle mesh stored in a wider padded
    connectivity array leaked fill values into the triangulation; the
    fast path now only applies to a clean (n, 3) array.
  - Add Google-style docstring examples for the vectorized helpers and
    expand the test suite (randomized equivalence vs the original loops,
    pentagon/pure-quad fans, shared-edge dedup, grouped-arange edge
    cases, and a non-flaky performance guard).
-   A 100k mixed-element mesh now triangulates and derives edges in well
  under 100ms; focus methods reach 100% line and branch coverage.
- ref: #102

## 0.10.0 (2026-05-27)


- feat(glyph)!: standardize axes/figure API and expand glyph controls (#132)
- Unify how the cleopatra glyphs bind to matplotlib axes/figures and round
  out ArrayGlyph's rendering surface, with a small pre-construction
  introspection API shared across all glyphs.
  
  - ArrayGlyph: store the colour-mapped artist on `self.im` for every kind
    (imshow/pcolormesh/contour/contourf/RGB) and add an `add_colorbar`
    option (default True) honoured by both `plot` and `animate`.
  - ArrayGlyph.plot: accept `ax` and `title`; resolve axes as
    `plot(ax=)` > constructor ax > fresh figure. `fig` stays a
    construction-time binding (derived from the axes, never a plot arg).
  - Glyph: keep a constructor `ax` given without `fig` (derive the figure
    from it) and warn on a mismatched `fig`/`ax` pair; resolve the
    top-level figure across matplotlib versions (SubFigure-safe).
  - Glyph: add `option_keys()` / `filter_kwargs()` classmethods plus a
    per-glyph `DEFAULT_OPTIONS` class attribute so accepted option keys can
    be inspected and filtered before construction; StatisticalGlyph gains
    matching helpers.
  - Rename the array/statistical option dicts to `ARRAY_DEFAULT_OPTIONS` /
    `STATISTICAL_DEFAULT_OPTIONS` (aligned with the other glyphs) with a
    backwards-compatible `DEFAULT_OPTIONS` alias.
-   BREAKING CHANGE: StatisticalGlyph.boxplot/multiboxplot/stripes no longer
  accept a `fig` argument; bind the figure at construction instead, e.g.
  `StatisticalGlyph(values, fig=fig).boxplot(ax=ax)`.
-   Closes #128, #129, #130, #131

## 0.9.0 (2026-05-26)


- feat: add matplotlib glyph primitives for new plot types (#117)
- Add generic matplotlib building blocks for point, vector, polygon, line,
  and statistical plots, each with tests and Google-style docstrings.
-   - glyph: add shared Glyph._prepare_scalar_mapping (+ _resolve_limits) so
    every colour-by-value glyph reuses one resolve-limits -> ticks_spacing
    -> norm/colorbar pipeline instead of re-deriving it; ArrayGlyph output
    is unchanged
  - scatter_glyph: new ScatterGlyph for coloured/uncoloured point clouds
  - vector_glyph: new VectorGlyph for quiver/barbs/streamplot coloured by
    magnitude, plus add_key (quiverkey)
  - polygon_glyph: new PolygonGlyph for filled choropleths / outlines,
    geometry-agnostic (plain vertex arrays, no geopandas)
  - line_glyph: new LineGlyph for line/bar/fill_between, and extend
    StatisticalGlyph with boxplot, multiboxplot, and stripes
  - mesh_glyph: add filled=False to render node data as line tricontour
  - styles: add disjoint_legend, colorbar_legend, and histogram_legend to
    complete the three reusable legend styles
  - build: raise the matplotlib floor to >=3.9 to match the APIs used
- ref: #118, #119, #120, #121, #122, #123, #124, #125
- feat(statistical_glyph)!: compose histograms into caller fig/ax; drop implicit plt.show() (#111)
- - add optional `fig`/`ax` parameters to `StatisticalGlyph`; `histogram()` resolves three
    composition modes: draw into a supplied `ax` (inferring its figure), add an axes onto an
    empty supplied `fig` (raising `ValueError` if that figure already has axes), or create a
    new figure/axes when neither is given
  - switch histogram styling from pyplot (`plt.grid/xlabel/ylabel/xticks/yticks`) to axes-level
    (`ax.grid/set_xlabel/set_ylabel/tick_params`) so labels land on the intended axes
  - remove internal `plt.show()` from `StatisticalGlyph.histogram()`, `ArrayGlyph.plot()`, and
    `ArrayGlyph.animate()`; these now return their `Figure`/`Axes`/`FuncAnimation` for the caller
    to display or save
  - document the composition modes and no-show behavior in docstrings; fix stale `Statistic`
    naming and the module usage example
  - add tests for fig/ax injection, the populated-figure `ValueError`, the no-show contract, a
    self-checking doctest runner, and validation guards (statistical_glyph at 100% coverage)
-   BREAKING CHANGE: `StatisticalGlyph.histogram()`, `ArrayGlyph.plot()`, and `ArrayGlyph.animate()`
  no longer call `plt.show()`. Code relying on the implicit display must call `plt.show()` itself
  (or save the returned figure/animation).
-   Closes #116

## 0.8.0 (2026-05-11)


- feat!: Expand `ArrayGlyph` plotting and add the `cleopatra.tiles` module (#112)
- xarray-aligned plotting features, a new web-tile basemap module, plus
  bug fixes and packaging cleanups from PR review.
-   - ArrayGlyph: plot(kind=imshow/pcolormesh/contour/contourf), colour
    kwargs (robust/center/levels/extend/cbar_kwargs), coords=(x, y)
    curvilinear grids, facet(col=, row=, col_wrap=, extents=) ->
    FacetGrid, animate(data_getter=) lazy frames.
  - New cleopatra.tiles module (add_tiles + helpers) behind the
    cleopatra[tiles] extra (mercantile, pillow, pyproj, xyzservices);
    cleopatra.styles.ColorScale enum.
  - Fixes: facet mask preservation, animate(display_cell_value=True)
    IndexError, import-time matplotlib backend no longer touched,
    all-NaN colour-limit guard, broader tile image-signature acceptance,
    color_scale validation.
  - Internal: single-backtick docstrings, CLAUDE.md untracked/gitignored,
    tests expanded.
-   Refs: #113, #114
-   BREAKING CHANGE: ArrayGlyph.no_elem -> num_domain_cells (deprecated
  alias kept); cleopatra.add_tiles / cleopatra.Config top-level re-exports
  removed (use the submodules); import cleopatra no longer sets the
  matplotlib backend; ArrayGlyph(all_nan_arr) and bad color_scale now
  raise ValueError.

## 0.7.1 (2026-04-09)

### fix

- **glyph**: set `use_gridspec=False` for colorbar on subplot figures
  so colorbars allocate space locally instead of stealing from sibling
  axes (#109)

### chore

- remove dead boilerplate from `__init__.py` (unused author metadata,
  empty hard-dependency checker, redundant module docstring)
- replace deprecated `typing.List` with builtin `list` in
  `array_glyph.py` parameter annotations and docstrings

### test

- add subplot colorbar tests for single-axes, 1x2, 1x3, and 2x2
  grid layouts in `test_glyph.py`
- add MeshGlyph subplot rendering tests in `test_mesh_glyph.py`

### docs

- update README

## 0.7.0 (2026-04-08)

### feat

- **mesh**: add MeshGlyph class for UGRID unstructured mesh
  visualization with tripcolor/tricontourf rendering, wireframe
  outlines via LineCollection, and mixed-element fan triangulation
  (#99, #100)
- **mesh**: add face-centered and node-centered data plotting with
  all 5 color scale types (linear, power, sym-lognorm, boundary-norm,
  midpoint) and full colorbar customization
- **mesh**: add `animate()` for time-varying mesh data with
  frame-by-frame rendering and gif/mp4/mov/avi export
- **mesh**: add `plot_outline()` for wireframe rendering with
  optional explicit edge connectivity
- extract `Glyph` base class from `ArrayGlyph` with shared
  infrastructure: fig/ax lifecycle, color scale normalization,
  colorbar creation, tick management, point overlays, and animation
  saving (#101)
- add input validation for constructor arrays, data length, all-NaN
  data, constant data, and animation frame consistency

### fix

- **ci**: correct PyPI release workflow (#94)
- fix ticks_spacing not propagated to `default_options`, causing
  wrong colorbar ticks for non-default data ranges
- fix colorbar accumulation on repeated `plot()` calls
- fix `default_options` mutation leaking across `plot()` calls on
  the same instance
- fix `save_animation` silently swallowing `FileNotFoundError` when
  FFmpeg is missing
- fix `anim` property returning `None` instead of raising when
  `_anim` is not set
- remove `plt.show()` from `adjust_ticks()` and `animate()` to
  prevent blocking in interactive backends
- fix broken image paths in reference documentation

### build

- **packaging**: migrate from setuptools/pip to uv and hatchling
  (#96, #97)
- replace setuptools build backend with hatchling
- convert `[project.optional-dependencies]` to PEP 735
  `[dependency-groups]` for dev and docs groups
- migrate all GitHub Actions workflows to composite actions with uv
- generate `uv.lock` for reproducible dependency resolution
- update classifiers to Python 3.11/3.12/3.13, drop 3.10

### docs

- convert all package docstrings from NumPy to Google style (#103)
- add API reference pages for `Glyph` and `MeshGlyph`
- add unstructured mesh visualization guide
- add MeshGlyph example Jupyter notebook
- update `mkdocs.yml` to `docstring_style: google`
- add missing type annotations to fix griffe/mkdocstrings warnings

### test

- add 105 new tests (45 Glyph, 60 MeshGlyph) with 100% line and
  branch coverage (#104)
- vectorize `_build_edge_segments` and `_map_face_to_triangle_values`
  with numpy; add fast path for pure-triangle meshes
- skip FFmpeg-dependent tests when FFmpeg is unavailable
- replace deprecated `typing.List`/`Tuple`/`Union`/`Dict` with
  Python 3.11+ builtins

### chore

- rename organisation from Serapieum-of-alex to serapeum-org (#98)

## 0.6.0 (2025-06-25)

### Dev

- replace the setup.py with pyproject.toml
- convert the documentation to use mkdocs instead of sphinx.
- remove the CI test workflow based on conda.
- test the jupyter notebook in ci.

### config

- add a config file to the package to handle the configuration of the matplotlib backend.
- in the __init__.py file, load the config file and set the matplotlib backend to `Agg`.

### ArrayGlyph

- rename the statistics module to statistical_glyph.
- move creating the ax, and fig from the constructor to the `plot`/`animate` methods .
- create `arr` property to access the array data.
- create `apply_colormap` method to apply a colormap to the array.
- create `to_image` method to convert the array to an RGB image.
- create `scale_to_rgb` method to scale the array to RGB values.
- create `adjust_ticks` method to adjust the plot ticks.

### colors

- add `get_color_map` function to create a color map from a list of colors.
- make the `_is_valid_rgb_norm`, and `_is_valid_rgb_255` protected and the public method is only `is_valid_rgb`.
- make the `_is_valid_hex_i` protected and the public method is only `is_valid_hex` to process single value and lists.
- create a `create_from_image` function to create a color map from an image.

## 0.5.1 (2024-07-24)

### ArrayGlyph

- the ArrayGlyph constructor uses a masked array instead of a numpy array.

## 0.5.0 (2024-07-22)

### ArrayGlyph

- rename the `Array` class to `ArrayGlyph`.
- add `scale_percentile` method to the `Array` class to scale the array using the percentile values.
- the `statistic.histogram` can plot multiple column array.
- change the `color_scale` values to be string (`linear`, "power", ...)
- the `kwargs` can be provided to the constructor or the `plot` method to plot the array.

### Colors

- rename the `get_rgb` to `to_rgb`
- add `get_type` to get the type of the color.
- add `to_hex` to convert the color to hex.
- add `to_rgb` to convert the color to rgb.

## 0.4.3 (2024-07-13)

- Add extent to the array plot when plotting an rgb array.
- Add `ax`, and `fig` parameters to the `Array` constructor method to take an Axes and plot the array on it.
- Add `__str__` to the `Array` class.

## 0.4.2 (2024-06-30)

- Update dependencies

## 0.4.1 (2024-1-11)

- add extent to the array plot.

## 0.4.0 (2023-9-24)

- Add a colors module to handle issues related to
- Converting colors from one format to another
- Creating colormaps

## 0.3.5 (2023-8-31)

- Update dependencies

## 0.3.4 (2023-04-26)

- pass the plot kwargs to the init of the array to scale the color bar using the vmin and vmax.

## 0.3.3 (2023-04-25)

- change the default value for the color bar label.

## 0.3.2 (2023-04-23)

- bump up hpc version

## 0.3.1 (2023-04-17)

- plot RGB plots

## 0.3.0 (2023-04-11)

- change API to work completly with numpy array inputs
- chenge to conda config
- add hpc-utils to filter and access arrays
- restructure the whole modules to array, statistics, and styles modules.
- all modules has classes.
- save animation function using ffmpeg.

## 0.2.7 (2023-01-31)

- bump up numpy to version 1.24.1

## 0.2.6 (2023-01-31)

- bump up versions
- add serapeum_utils as a dependency

## 0.2.5 (2022-12-26)

- plot array with discrete bounds takes the bounds as a parameter

## 0.2.4 (2022-12-26)

- bump up numpy versions to 1.23.5, add pandas

## 0.1.0 (2022-05-24)

- First release on PyPI.
