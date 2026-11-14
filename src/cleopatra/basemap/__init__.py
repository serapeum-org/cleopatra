"""cleopatra's networked basemap/CRS helpers (SCOPE.md's deliberate exceptions).

Deliberately re-exports nothing, matching the package root: import from the
submodule, e.g. `from cleopatra.basemap.tiles import stitch_basemap`,
`from cleopatra.basemap.reference import add_relief`. Web-tile basemaps are
added through the `add_tiles` glyph method (`GeoMixin`), not a module function.

Submodules: `geo` (`GeoMixin`), `tiles`, `ogc` (WMS/WMTS providers for the
`add_tiles` glyph method), `reference`, `projection`, `solar` (day/night terminator
and Tissot artists), and the private `_net` (shared HTTP opener, used only by
`tiles`/`reference`).
"""
