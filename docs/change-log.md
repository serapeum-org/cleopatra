# Changelog

## 0.26.1 (2026-07-18)


- fix(glyphs): remove orphaned render artists on repeated plot()/animate() calls (#211)
- - Track each Axes' prior render artists via a shared marker so a second                                                        
    plot()/animate() call — same glyph instance, or a different glyph                                                            
    sharing the Axes via `ax=`/`fig=` — removes them instead of leaving                                                          
    them attached and undriven.                                                                                                  
  - Extend cleanup to ArrayGlyph, MeshGlyph, StatisticalGlyph, and                                                               
    VectorGlyph: colorbars, frame-label text, point/cell-value overlays,                                                         
    and streamplot arrowheads (never actually attached via the returned                                                          
    collection, so removed by diffing ax.patches instead).                                                                       
  - Defer cleanup until after each call's own input validation succeeds,                                                         
    so a failed call (e.g. an invalid color_scale, or a mismatched color                                                         
    list) no longer destroys a valid prior render before propagating its                                                         
    exception.                                                                                                                   
  - Tolerate an artist already partially removed by ax.clear() or a                                                              
    prior apply_style() call instead of crashing on the second removal                                                           
    attempt.                                                                                                                     
  - Add direct unit tests for the shared cleanup helpers and regression                                                          
    tests covering same-instance repeats, cross-glyph shared axes, and                                                           
    validation-failure paths across all four glyph classes.                                                                      
                                                                                                                                 
  Closes #210

## 0.26.0 (2026-07-16)


- refactor(array_glyph)!: clean up plot()/animate()'s kwarg API (#207)
- - Rename pid_color/pid_size to point_label_color/point_label_size,                        
    animate()'s text_colors to cell_value_text_colors, and text_loc to                      
    label_location                                                                          
  - Bundle the five point-overlay parameters into a new PointOverlay                        
    class and animate()'s two frame-label parameters into a new                             
    FrameLabel class                                                                        
  - Type the remaining **kwargs on both methods via TypedDict + Unpack                      
    (PEP 692), inert at runtime                                                             
  - Keep every removed name/shape working via **kwargs behind a                             
    DeprecationWarning, resolved before the strict kwargs validation                        
  - Fix bugs found during two rounds of adversarial review: a silent                        
    positional-arg drop, wrong warning stacklevels, a both-given                            
    conflict false-negative, a missing precision field, and a crash on                      
    deprecated point kwargs passed without points                                           
  - Update the example notebooks to the new PointOverlay/FrameLabel API                     
                                                                                            
  BREAKING CHANGE: point_color, point_size, pid_color, pid_size, and                        
  animate()'s label_color are no longer explicit parameters. Keyword                        
  calls still work via a deprecated alias with a warning; positional                        
  calls to these slots now bind to the wrong parameter or raise                             
  TypeError.                                                                                
  Closes #208
- feat(styles,glyphs): distinct-value categorical colouring for PolygonGlyph/ScatterGlyph (#206)
- Add styles.categorize(values, cmap="tab10") -> (categories, colors), the                        
  distinct-value counterpart to classify(): one colour per unique value,                          
  sorted when sortable, cycling past the cmap's size, nulls dropped.                              
                                                                                                  
  Wire a "categorical" scheme into the shared Glyph scalar-mapping                                
  pipeline: _prepare_categorical_mapping builds a ListedColormap +                                
  BoundaryNorm over per-element integer class codes, and                                          
  create_categorical_legend draws a disjoint_legend in place of a                                 
  colorbar. PolygonGlyph and ScatterGlyph opt in via                                              
  _SUPPORTS_CATEGORICAL_SCHEME; VectorGlyph/FlowGlyph still accept scheme                         
  for continuous classification but reject "categorical" with a clear                             
  error.                                                                                          
                                                                                                  
  - Validate the full values shape (not just its first dimension) in                              
    PolygonGlyph, closing a silent colour-array mis-sizing bug                                    
  - Reset cbar/category_legend unconditionally on every plot() call so a                          
    scheme-switching re-plot never leaves a stale reference                                       
  - Fall back to a qualitative cmap when the caller left cmap at the                              
    glyph's own continuous default, matched by resolved name so a                                 
    Colormap object is caught the same as the equivalent string                                   
  - Re-attach the categorical legend via ax.add_artist() before drawing                           
    ScatterGlyph's size legend, since Axes.legend() is single-slot per                            
    axes and would otherwise silently evict it                                                    
  - Add category_legend_kwargs (mirroring size_legend_kwargs) to                                  
    reposition/restyle the disjoint legend                                                        
  - categorize() raises the documented TypeError for non-hashable                                 
    entries and documents the int/bool/float dedup collision                                      
                                                                                                  
  Closes #204

## 0.25.0 (2026-07-12)


- feat(glyphs): public apply_style() to (re)apply a data-style preset on an existing glyph (#203)
- Add a discoverable way to (re)apply a DATA_STYLES preset on a glyph                             
  instance that already exists, so a caller holding a rendered glyph can                          
  restyle it by name without rebuilding it.                                                       
                                                                                                  
  - Add apply_style(name, **kwargs) and a style read-back property to                             
    ArrayGlyph, MeshGlyph, and KDEGlyph. apply_style re-renders the glyph                         
    in place on its own axes (or a fresh figure if never plotted or the                           
    figure was closed) and forwards extra kwargs (hillshade, ...) to plot.                        
  - Unify style persistence across the three glyphs: an applied style is                          
    sticky (survives a plain plot()) and clearable (plot(style=None)).                            
    MeshGlyph tracks the current style and restores it after its options                          
    reset; KDEGlyph uses a typed _Unset sentinel so style=None can clear.                         
  - Validate a preset name before persisting or clearing, and roll back on                        
    a bad/categorical name, so a typo can't wipe the render or brick the                          
    glyph for later plain plots.                                                                  
  - Add a shared Glyph._reset_axes_for_restyle helper (root-figure aware,                         
    reuses an existing axes) and copy MeshGlyph's cached data (preserving a                       
    masked array's mask). Honour a construction-time MeshGlyph style too.                         
  - Document the option and add an apply_style example to the ArrayGlyph,                         
    MeshGlyph, and KDEGlyph notebooks.                                                            
                                                                                                  
  Closes #202

## 0.24.0 (2026-07-12)


- feat(glyphs): add a data-style preset option across the surface glyphs (#200)
- Add a `style` option that renders a named cleopatra DATA_STYLES preset                          
  (colormap, norm, transparent nodata, value-linked opacity, and a                                
  categorical legend) so a field can be visualised by name rather than by                         
  hand-built colour settings.                                                                     
                                                                                                  
  - ArrayGlyph: plot() delegates to apply_data_style; animate() reproduces                        
    the preset per frame for both continuous and categorical presets. It                          
    supports curvilinear coords, composes with hillshade (via the new                             
    hillshade.shade_rgb), and presents a swatch / discrete legend                                 
    consistently across plot and animate. Unknown or multi-layer names,                           
    RGB arrays, and overlay kwargs are guarded with clear errors or                               
    warnings; integer-masked rasters are handled.                                                 
  - MeshGlyph: continuous presets override the tripcolor/tricontour                               
    cmap+norm (composing with node-elevation hillshade); categorical                              
    presets draw a discrete legend and mask out-of-range codes.                                   
  - KDEGlyph: continuous presets colour the density; a categorical preset                         
    raises a clear error.                                                                         
  - Add hillshade.shade_rgb (light an already-coloured image) and promote                         
    colors.resolve_style_norm, alpha_rgba, category_boundaries, and                               
    resolve_single_layer_style to public API (private aliases kept).                              
  - Document the option in the glyph docstrings and add a style section to                        
    the ArrayGlyph example notebook.                                                              
                                                                                                  
  Closes #199
- feat(colors,glyphs): add reusable hillshade, style-preset libraries, and flow-raster palettes (#195)
- - Add a reusable cleopatra.hillshade module (shade_grid, shade_faces,                           
    resolve_hillshade) and wire relief shading into ArrayGlyph, MeshGlyph,                        
    and KDEGlyph via a hillshade option at construction and plot() time.                          
  - Extend the DATA_STYLES preset system with vendored ECMWF/Magics (69)                          
    and cmocean (15) libraries, CAMS AOD colormaps, and log, symlog,                              
    diverging, and categorical norm handling in apply_data_style.                                 
  - Add flow_direction_d8 and flow_accumulation presets and a                                     
    draw_order="width" stream-order mode for FlowGlyph.                                           
  - Bundle the vendored preset JSON assets with upstream NOTICE/LICENSE                           
    text, regenerated by tools/build_magics_presets.py and                                        
    tools/build_cmocean_presets.py.                                                               
  - Give swatch_legend a norm parameter so log/symlog legends sample the                          
    gradient honestly.                                                                            
  - Cover the new behaviour with tests (incl. tests/test_hillshade.py) and                        
    document it in the example notebooks and the mkdocs nav.                                      
                                                                                                  
  Closes #198
- fix(polygon_glyph): render visible outlines in outline_only mode (#196)
- PolygonGlyph.plot(outline_only=True) drew nothing under default                                 
  options: the "none" borderless-fill edgecolor was passed straight to                            
  the outline PolyCollection, leaving an unfilled polygon with a                                  
  transparent edge. Substitute a new OUTLINE_EDGECOLOR ("black") when                             
  edgecolor is left at its "none" default; an explicit edgecolor is                               
  still honoured and the filled choropleth branch keeps its borderless                            
  default. Add regression tests for the fallback, the no-values path, an                          
  explicit colour, and the still-borderless filled branch.                                        
                                                                                                  
  Also refresh the documentation to the 0.23.0 API and fix stale or                               
  incorrect content found in a full docs audit:                                                   
                                                                                                  
  - document the geo/GeoMixin basemap methods, the reference module, the                          
    haze data-styles in colors, and the orthographic globe presets in                             
    projection across the README, docs/index.md, and reference pages                              
  - add swatch_legend and apply_blank_canvas (styles), and WebP / to_mp4                          
    / quality kwargs / bundled ffmpeg (animation), correcting the                                 
    projection "single stateless helper" and "no PROJ dependency" claims                          
  - correct the bar() docs (bar takes a single 1D series, not one series                          
    per column) in the reference page and the LineGlyph docstring                                 
  - note the [tiles] extra also powers reference and the projection                               
    presets; fix the ArrayGlyph render kinds and netCDF4 note in the mesh                         
    guide; add per-glyph example figures to the README                                            
  - clean up mkdocs.yml (duplicate features/extensions, stray theme keys,                         
    unused table-reader/tags plugins) and un-orphan the glyph-architecture                        
    diagram, wiring it into the nav                                                               
  - fix the CODE_OF_CONDUCT project name and empty contact fields, and                            
    align CONTRIBUTING with the commitizen workflow                                               
                                                                                                  
  Closes #197

## 0.23.0 (2026-07-09)


- feat(colors,projection): add composable haze-style data and globe projection presets (#192)
- - Add HAZE_COLORMAPS, DATA_STYLES["haze"], and apply_data_style() for                          
    value-modulated-alpha colour layers with decoupled alpha/colour                              
    ranges (a glowing "flame" rim effect) plus swatch legends.                                   
  - Add alpha_scaled_image()/alpha_scaled_mesh() low-level rendering                             
    primitives that apply_data_style is built on.                                                
  - Add PROJECTION_STYLES and apply_projection_style() to reproject                              
    gridded data onto an orthographic globe or leave it flat behind one                          
    call, backed by orthographic_grid(), orthographic_grid_edges(),                              
    orthographic_points(), orthographic_boundary(), and                                          
    orthographic_graticule().                                                                    
  - Add apply_blank_canvas() and swatch_legend() to cleopatra.styles.                            
  - Add add_point_labels() to cleopatra.geo for plain dot-and-text                               
    point/city labels, reprojectable onto a globe via                                            
    orthographic_points().                                                                       
  - Add a tutorial notebook demonstrating every primitive independently                          
    and composed, including an animated multi-day globe view.                                    
  - Cover every new primitive with tests, guarding the pyproj-backed                             
    globe helpers with pytest.importorskip so the suite still passes                             
    without the [tiles] extra installed.                                                         
                                                                                                 
  Closes #193
- fix: stop animate() label clipping and execute docs notebooks in CI (#190)
- - Wire notebook execution into all three mkdocs-deploy jobs
  (deploy-pr/main/release) so docs/notebooks/** render fresh, real
  output during the build instead of whatever was last committed by
  hand.
- Add an nbstripout pre-commit hook so notebooks never carry baked-in
  outputs in git history again.
- Fix ArrayGlyph.animate()'s default text_loc: it clipped the frame
  label under the array's inverted Y-axis for any shape. The default
  now anchors via axes-fraction coordinates with top alignment; an
  explicit text_loc keeps the prior data-coordinate behavior
  unchanged.
- Add a keyword-only label_color parameter to animate() for the frame
  label's color.
- Retry transient network failures in reference._download with
  exponential backoff, so the two live-data example notebooks no
  longer hard-fail the docs build on a flaky fetch.
- Reformat most of src/cleopatra/*.py and tests/*.py (line-wrap and
  isort); no behavior change.
- Closes #191

## 0.22.0 (2026-07-07)


- feat(animation): add save_animation quality controls, WebP, and bundled ffmpeg (#188)
- Rework save_animation (and the Glyph.save_animation wrapper) from a                            
  minimal two-line writer into a configurable, robust exporter, resolving                        
  the rough edges surfaced in issue #185.                                                        
                                                                                                 
  - Replace the declared-but-unused ffmpeg-python dependency with                                
    imageio-ffmpeg, and fall back to its bundled static binary when no                           
    system ffmpeg is on PATH, so MP4/MOV/AVI export works out of the box.                        
  - Auto-pad odd frame dimensions (-vf pad) so libx264 no longer crashes                         
    on odd-sized figures, and set pix_fmt=yuv420p for universal playback.                        
  - Add keyword-only crf, bitrate, codec, preset, pix_fmt, dpi, optimize,                        
    loop, and extra_args controls, with crf and bitrate mutually                                 
    exclusive and a caller -vf/-pix_fmt merged into the built arguments.                         
  - Drop the hardcoded bitrate=1800 default in favour of libx264's                               
    constant-quality default; existing 3-arg calls stay valid but encode                         
    smaller, better files.                                                                       
  - Add animated WebP output and optimise/loop-configurable GIF output                           
    via an _OptimizedPillowWriter subclass.                                                      
  - Add to_bytes and to_mp4 in-memory render helpers alongside the                               
    existing to_gif/embed_gif; to_gif now delegates to to_bytes.                                 
  - Forward the new controls through Glyph.save_animation so ArrayGlyph                          
    and MeshGlyph inherit them.                                                                  
                                                                                                 
  Closes #185
- feat(geo): add ECMWF reference-map style preset for georeferenced glyphs (#187)
- Add GeoMixin.add_reference_map(style=...) so the ~15-line weather-centre                       
  map recipe (grey Natural Earth coastline + borders, a dashed lon/lat                           
  graticule, degree labels, a subtle frame) is a single call on top of a                         
  plotted, georeferenced glyph.                                                                  
                                                                                                 
  - Presets "ecmwf" (light backgrounds) and "ecmwf-dark" (lighter greys so                       
    coastlines stay visible over a dark field), plus style="auto" that                           
    picks between them from the displayed luminance (im.to_rgba through the                      
    colormap/norm; only opaque cells count, so light no-data fields are not                      
    misread as dark; the target axes' image is preferred over self.im).                          
  - extent=[xmin, ymin, xmax, ymax] (the ArrayGlyph order) georeferences                         
    the image and axis limits, handling the pixel-coordinate RGB/animate                         
    case; a warning fires when the glyph has no geographic extent.                               
  - Degree formatters label the +/-180 antimeridian as "180"; _nice_step                         
    covers sub-degree to 90-degree spacing; graticule_step is validated as                       
    positive and finite; extent length is validated with a clear message.                        
  - available_map_styles() and the REFERENCE_MAP_STYLES table expose and                         
    allow copying the presets; the geographic knowledge (deriving extent                         
    from a dataset geotransform) stays upstream.                                                 
                                                                                                 
  Add a runnable docs notebook (docs/notebooks/array_glyph/reference_map)                        
  wired into the mkdocs nav, and TestAddReferenceMap plus a non-mocked                           
  integration test (geo.py coverage 98%).                                                        
                                                                                                 
  Closes #184
- fix(tiles): align web-tile basemap for non-EPSG:3857 data (#186)
- For a non-EPSG:3857 axis, add_tiles stretched the stitched Mercator
mosaic onto the data bounds, discarding the mosaic's tile-snapped (larger)
coverage and offsetting the basemap by up to hundreds of km at coarse
zoom (e.g. a curvilinear ROMS field over the Gulf of Mexico).
- - Place the mosaic at its own geographic footprint: reproject its
  Web-Mercator bounds (extent_3857) into the target CRS with edge
  densification and use that as the imshow extent, keeping set_xlim/ylim
  at the data bounds. If a coarse mosaic overflows a limited-domain CRS
  (e.g. a whole-world mosaic into a UTM zone or a singular projection),
  the reprojection is caught and the basemap falls back to the data
  bounds with a warning, so a figure is still produced.
- Add an auto_zoom floor via a new min_tiles_across parameter (default 2,
  also exposed and validated on add_tiles): pick the smallest zoom at
  which the larger extent spans at least that many tiles, so a mid-range
  region is no longer rendered from one or two coarse tiles (global
  z0->z1, Berlin z10->z11, Gulf z6->z7). min_tiles_across=1 restores the
  old heuristic and MAX_TILES still caps the tile count.
- Update the module and reference-doc notes; add tests for the
  reprojected/enveloping extent, the overflow fallback, and the
  auto_zoom floor and its validation.
- The auto_zoom floor applies to every zoom="auto" call (all CRS, including
EPSG:3857), so the default basemap fetches a slightly higher zoom and a
few more tiles (bounded by MAX_TILES); a residual Mercator-vs-linear-axis
nonlinearity remains for very large extents.
- Closes #176

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
