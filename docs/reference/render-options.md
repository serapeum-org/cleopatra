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

## Composing onto one axes (`compose=`)

By default a glyph drawn onto an axes **replaces** whatever a glyph put there
before it. That is deliberate: a second glyph bound to an existing axes would
otherwise leave the first one's artists attached and driven by nothing, which is
what once froze an animation at its first frame.

Pass `compose=True` to `plot()` or `animate()` to draw *over* what is already
there instead — the classic scalar field with wind arrows on top:

```python
fig, ax = plt.subplots()
temperature.plot(ax=ax, colorbar=ColorBar(label="500 hPa T [C]"))
VectorGlyph(xx, yy, u, v, ax=ax, add_colorbar=False).plot(
    kind="quiver", ax=ax, compose=True, thin=4
)
```

A composing render clears only the artists it put there itself, so the host's
layers, colorbar, title and projection frame all survive. A glyph replotting
onto its own axes still replaces its own artists either way, so nothing is
orphaned.

Pair it with `add_colorbar=False` on the overlay when the host owns the single
shared colorbar.

## Thinning a vector field (`thin=`)

`quiver` and `barbs` draw one arrow per grid point, which on a real grid is both
unreadable and slow — a 141x321 window is 45,261 arrows. `thin=n` draws every
nth point along each axis:

```python
VectorGlyph(xx, yy, u, v, thin=4).plot(kind="quiver")
```

It applies to `quiver` and `barbs`. `streamplot` seeds its own lines and has no
per-point arrow to drop, so `thin` warns there — use `density=` instead.
