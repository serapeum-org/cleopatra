"""Tests for the `cleopatra` package surface.

The package root deliberately re-exports nothing — public names live in
the submodules. These tests check that `__version__` is defined, that no
symbols leak into the top-level namespace, that importing `cleopatra` has
no side effects on the matplotlib backend, and that every submodule
imports cleanly.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

#: Submodules the package is expected to ship.
_SUBMODULES = [
    "animation",
    "array_glyph",
    "colors",
    "config",
    "flow_glyph",
    "glyph",
    "kde_glyph",
    "line_glyph",
    "mesh_glyph",
    "polygon_glyph",
    "projection",
    "scatter_glyph",
    "statistical_glyph",
    "styles",
    "tiles",
    "vector_glyph",
]


class TestPackageSurface:
    """Tests for the top-level `cleopatra` namespace."""

    def test_version_attribute_exists(self):
        """`cleopatra.__version__` is defined and is a string."""
        import cleopatra

        assert hasattr(cleopatra, "__version__"), "Missing __version__"
        assert isinstance(cleopatra.__version__, str), (
            f"__version__ should be str, got {type(cleopatra.__version__)}"
        )

    def test_no_top_level_reexports(self):
        """The package root does not re-export classes/functions or `__all__`.

        Public API lives in the submodules; importing `cleopatra` alone
        should not give you `add_tiles`, `Config`, etc.
        """
        import cleopatra

        for name in ("add_tiles", "Config", "ArrayGlyph"):
            assert not hasattr(cleopatra, name), (
                f"cleopatra should not re-export {name!r}"
            )
        assert not hasattr(cleopatra, "__all__"), (
            "cleopatra.__init__ should not define __all__"
        )
        # Only dunders and the submodule attributes Python binds on import
        # should be present — nothing else leaked from __init__.py.
        leaked = [
            n
            for n in vars(cleopatra)
            if not n.startswith("__") and n not in _SUBMODULES
        ]
        assert not leaked, f"unexpected names in cleopatra namespace: {leaked}"


class TestImportSafety:
    """Tests that importing cleopatra does not mutate the matplotlib backend."""

    def test_import_does_not_change_backend(self):
        """`import cleopatra` must not change matplotlib's active backend.

        Picking a backend is the application's job; a library must not do
        it at import time. Run in a fresh subprocess (the test session's
        `conftest.py` pins `Agg`, so an in-process check would be
        vacuous) and compare the backend before vs. after the import.
        """
        code = (
            "import matplotlib; before = matplotlib.get_backend(); "
            "import cleopatra; after = matplotlib.get_backend(); "
            "print(before); print(after)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        before, after = result.stdout.split()
        assert before == after, (
            f"`import cleopatra` changed the matplotlib backend: "
            f"{before!r} -> {after!r}"
        )

    def test_config_submodule_exposes_helpers(self):
        """`cleopatra.config` resolves to the config submodule with its API."""
        import cleopatra.config as config_mod
        from cleopatra.config import Config

        assert config_mod.Config is Config
        assert callable(config_mod.is_notebook)
        # Config carries no instance state — set_matplotlib_backend is static.
        assert callable(Config.set_matplotlib_backend)


@pytest.mark.parametrize("submodule", _SUBMODULES)
def test_submodule_imports_cleanly(submodule: str):
    """Each declared submodule imports without raising.

    Args:
        submodule: Submodule name under the `cleopatra` package.
    """
    mod = importlib.import_module(f"cleopatra.{submodule}")
    assert mod is not None, f"Failed to import cleopatra.{submodule}"
