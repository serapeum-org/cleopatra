"""
cleopatra - visualization package
"""

from cleopatra.config import Config
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

config = Config()
config.set_matplotlib_backend()

