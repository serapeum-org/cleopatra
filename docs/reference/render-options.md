# Render options (grouped parameters)

Glyph `plot()` / `animate()` calls take these typed objects in place of loose keyword arguments.
Each bundles a family of related options and exposes `to_options()`, which the glyph flattens into
its render settings — only the fields you set are applied, so a group never clobbers a glyph's own
defaults. (The `ArrayGlyph`-specific input objects — `RgbBands`, `PointOverlay`, `FrameLabel`,
`PanelLabels` — are documented on the [ArrayGlyph page](array-glyph.md).)

## ColorScaling

The colour-scale (norm) selector: `plot(color=ColorScaling.power(gamma=0.5))`,
`ColorScaling.sym_log(...)`, `ColorScaling.boundary(bounds=[...])`, `ColorScaling.midpoint(at=0)`,
`ColorScaling.linear()`.

::: cleopatra.styling.scaling.ColorScaling
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Contour

Discrete colour levels and inline contour labels: `plot(contour=Contour(levels=6, labels=True))`.

::: cleopatra.styling.params.Contour
    options:
      show_root_heading: true
      heading_level: 3

## CellValues

Per-cell value-text overlay (`ArrayGlyph`): `plot(cells=CellValues(show=True, size=10))`.

::: cleopatra.styling.params.CellValues
    options:
      show_root_heading: true
      heading_level: 3

## DataStyle

Named preset, relief shading, and per-call preset overrides:
`plot(data_style=DataStyle(style="topography", hillshade=True))`.

::: cleopatra.styling.params.DataStyle
    options:
      show_root_heading: true
      heading_level: 3

## Classify

Categorical / classed colour schemes on the scatter / vector / flow / polygon glyphs:
`plot(classify=Classify(scheme="categorical", k=5))`.

::: cleopatra.styling.params.Classify
    options:
      show_root_heading: true
      heading_level: 3

## ColorBar

Colorbar placement, caption, and sizing: `plot(colorbar=ColorBar(location="bottom", label="mm/day"))`.
Pass `colorbar=True`/`False` for the simple cases.

::: cleopatra.styling.colorbar.ColorBar
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
