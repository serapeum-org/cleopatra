"""Shared foundation for the glyph classes.

Deliberately re-exports nothing, matching the package root: import from the
submodule, e.g. `from cleopatra.glyphs.base.glyph import Glyph`.

Submodules: `glyph` (the `Glyph` base class), `animation` (generic
`FuncAnimation` save/embed helpers), `hillshade` (shaded-relief math), and
`compositing` (the Porter-Duff "over" array primitive). These are the machinery
the concrete glyphs in `gridded`, `primitives`, and `stats` build on, not chart
types themselves.
"""
