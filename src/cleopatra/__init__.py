"""cleopatra - visualization package.

Re-exports the most commonly used entry points so they can be imported
straight from the package root:

* `cleopatra.tiles.add_tiles` — overlay an XYZ web-tile basemap
  on an existing matplotlib axes.

The submodules `array_glyph`, `glyph`, `mesh_glyph`,
`statistical_glyph`, `colors`, `styles`, `tiles`, and `config`
are also publicly importable.

Importing cleopatra does not change the active matplotlib backend. If you
want the old convenience behaviour (`%matplotlib inline` in a notebook,
`Agg` in scripts) call `cleopatra.config.Config.set_matplotlib_backend()`
yourself.
"""

# Importing this binds the `config` submodule as `cleopatra.config` (used in
# `__all__`) and re-exports the `Config` class as `cleopatra.Config`.
from cleopatra.config import Config  # noqa: F401
from cleopatra.tiles import add_tiles

__all__ = [
    "add_tiles",
    "array_glyph",
    "colors",
    "config",
    "glyph",
    "mesh_glyph",
    "statistical_glyph",
    "styles",
    "tiles",
]

try:
    from importlib.metadata import PackageNotFoundError  # type: ignore
    from importlib.metadata import version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError  # type: ignore
    from importlib_metadata import version


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

