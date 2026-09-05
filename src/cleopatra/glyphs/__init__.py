"""cleopatra's glyph classes -- the chart-type building blocks.

Deliberately re-exports nothing, matching the package root: import each
glyph from its own submodule, e.g.
`from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph`,
`from cleopatra.glyphs.primitives.scatter_glyph import ScatterGlyph`.

Subpackages:

- `base` -- shared foundation (`glyph` base class, `animation`, `hillshade`,
  `compositing`).
- `gridded` -- values sampled over a 2-D domain: `array_glyph`, `mesh_glyph`,
  `vector_glyph`.
- `primitives` -- explicit geometry you pass in: `scatter_glyph`, `line_glyph`,
  `polygon_glyph`, `flow_glyph`.
- `stats` -- distribution summaries: `histogram_glyph`, `kde_glyph`.
"""
