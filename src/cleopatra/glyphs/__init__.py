"""cleopatra's glyph classes -- the chart-type building blocks.

Deliberately re-exports nothing, matching the package root: import each
glyph from its own submodule, e.g. `from cleopatra.glyphs.array_glyph import
ArrayGlyph`, `from cleopatra.glyphs.mesh_glyph import MeshGlyph`.

Submodules: `glyph` (shared base class), `animation`, `hillshade`,
`array_glyph`, `mesh_glyph`, `line_glyph`, `polygon_glyph`, `scatter_glyph`,
`vector_glyph`, `flow_glyph`, `kde_glyph`, `statistical_glyph`.
"""
