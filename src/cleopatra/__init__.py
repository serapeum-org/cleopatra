"""cleopatra - visualization package.

The package root deliberately re-exports nothing: import the public
classes/functions from their submodules, e.g.
`from cleopatra.array_glyph import ArrayGlyph`,
`from cleopatra.tiles import add_tiles`,
`from cleopatra.config import Config`.

Submodules: `array_glyph`, `glyph`, `mesh_glyph`, `statistical_glyph`,
`colors`, `styles`, `tiles`, `reference`, `projection`, `animation`,
`config`.

Importing cleopatra does not change the active matplotlib backend. If you
want the old convenience behaviour (`%matplotlib inline` in a notebook,
`Agg` in scripts) call `cleopatra.config.Config.set_matplotlib_backend()`
yourself.
"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

# Keep the package namespace minimal — nothing is re-exported here.
del PackageNotFoundError, version
