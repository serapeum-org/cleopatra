# Glyph Base Class

The `Glyph` class is the base class for all cleopatra visualization glyphs.
It provides shared infrastructure for figure/axes management, color scale
normalization (including value classification), colorbar creation, tick
control, point overlays, and animation saving.

`ArrayGlyph`, `MeshGlyph`, `ScatterGlyph`, `VectorGlyph`, `FlowGlyph`,
`LineGlyph`, `PolygonGlyph`, and `KDEGlyph` all inherit from `Glyph` and share
its colour-mapping / colorbar pipeline. `StatisticalGlyph` stands alone.

## Class Documentation

::: cleopatra.glyph.Glyph
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
