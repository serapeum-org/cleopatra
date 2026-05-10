"""Tests for the :mod:`cleopatra` package surface.

Validates the package re-exports introduced alongside the tiles module
(``add_tiles``), the ``__all__`` membership, that ``__version__`` is
defined, and that importing :mod:`cleopatra` does not switch matplotlib
to an interactive backend (the default is Agg as of 08b0728).
"""

from __future__ import annotations

import importlib

import matplotlib
import pytest


class TestPackageReExports:
    """Tests for top-level cleopatra package re-exports."""

    def test_add_tiles_is_importable(self):
        """``cleopatra.add_tiles`` is the same callable as ``cleopatra.tiles.add_tiles``."""
        import cleopatra
        from cleopatra.tiles import add_tiles as tiles_add_tiles

        assert hasattr(cleopatra, "add_tiles"), (
            "cleopatra.add_tiles re-export is missing"
        )
        assert cleopatra.add_tiles is tiles_add_tiles, (
            "cleopatra.add_tiles must be the exact callable from cleopatra.tiles"
        )

    def test_all_contains_add_tiles(self):
        """``__all__`` advertises ``add_tiles`` so ``from cleopatra import *`` works."""
        import cleopatra

        assert "add_tiles" in cleopatra.__all__, (
            f"__all__ should include 'add_tiles', got {cleopatra.__all__!r}"
        )

    def test_all_lists_known_modules(self):
        """``__all__`` lists every public submodule the package re-exports."""
        import cleopatra

        expected = {
            "add_tiles",
            "array_glyph",
            "colors",
            "config",
            "glyph",
            "mesh_glyph",
            "statistical_glyph",
            "styles",
            "tiles",
        }
        assert set(cleopatra.__all__) == expected, (
            f"__all__ mismatch: {set(cleopatra.__all__)} != {expected}"
        )

    def test_version_attribute_exists(self):
        """``cleopatra.__version__`` is defined and is a string."""
        import cleopatra

        assert hasattr(cleopatra, "__version__"), "Missing __version__"
        assert isinstance(cleopatra.__version__, str), (
            f"__version__ should be str, got {type(cleopatra.__version__)}"
        )


class TestImportSafety:
    """Tests that importing cleopatra does not break the matplotlib backend."""

    def test_import_does_not_select_tkagg(self):
        """Importing :mod:`cleopatra` never leaves matplotlib on ``TkAgg``.

        The package previously defaulted to TkAgg which opened blocking
        Tk popups on Windows. Commit ``08b0728`` switched the default to
        ``Agg``. This test guards against a regression by reloading the
        package and asserting the active backend is non-interactive.
        """
        import cleopatra

        importlib.reload(cleopatra)
        backend = matplotlib.get_backend()
        assert backend.lower() != "tkagg", (
            f"Backend leaked to TkAgg after import: {backend!r}"
        )

    def test_config_is_singleton_attribute(self):
        """``cleopatra.config`` is a :class:`Config` instance, not the module."""
        import cleopatra
        from cleopatra.config import Config

        assert isinstance(cleopatra.config, Config), (
            f"cleopatra.config should be a Config instance, "
            f"got {type(cleopatra.config)}"
        )


@pytest.mark.parametrize(
    "submodule",
    [
        "cleopatra.array_glyph",
        "cleopatra.colors",
        "cleopatra.config",
        "cleopatra.glyph",
        "cleopatra.mesh_glyph",
        "cleopatra.statistical_glyph",
        "cleopatra.styles",
        "cleopatra.tiles",
    ],
)
def test_submodule_imports_cleanly(submodule: str):
    """Each declared submodule imports without raising.

    Args:
        submodule: Dotted path to import.
    """
    mod = importlib.import_module(submodule)
    assert mod is not None, f"Failed to import {submodule}"
