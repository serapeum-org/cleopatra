"""cleopatra's networked basemap/CRS helpers (SCOPE.md's deliberate exceptions).

Deliberately re-exports nothing, matching the package root: import from the
submodule, e.g. `from cleopatra.basemap.tiles import add_tiles`,
`from cleopatra.basemap.reference import add_relief`.

Submodules: `geo` (`GeoMixin`), `tiles`, `reference`, `projection`, and the
private `_net` (shared HTTP opener, used only by `tiles`/`reference`).
"""
