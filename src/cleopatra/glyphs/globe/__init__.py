"""Glyphs that render data on a 3-D globe.

Deliberately re-exports nothing, matching the package root: import from the
submodule, e.g. `from cleopatra.glyphs.globe.textured_globe_glyph import
TexturedGlobeGlyph`.

Submodules: `textured_globe_glyph` (wrap an equirectangular lon/lat texture
onto a sphere drawn on a matplotlib `Axes3D`).

This is cleopatra's one deliberate 3-D renderer. Every other glyph targets a
2-D axes; the globe is a single, self-contained exception. It adds no
dependency -- `mpl_toolkits.mplot3d` ships with matplotlib -- and keeps the
"NumPy in -> matplotlib artist out" contract.
"""
