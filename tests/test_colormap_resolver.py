"""Tests for the namespaced scientific-colormap resolver.

Covers ``cleopatra.colors.resolve_colormap`` and its ``_require_cmap`` guard
(the optional ``[science-colors]`` extra), plus the load-bearing invariant that
resolving the default colormap preserves its name so the categorical-default
fallback in ``Glyph._prepare_categorical_mapping`` keeps working.

The ``cmap`` package is an optional extra and may be absent; tests that need a
real namespaced colormap are guarded with ``pytest.importorskip("cmap")`` so the
suite stays green with and without the extra installed.
"""

from __future__ import annotations

import importlib.util

import matplotlib as mpl
import pytest
from matplotlib.colors import Colormap

from cleopatra.colors import _require_cmap, resolve_colormap
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS


@pytest.fixture(scope="function")
def without_cmap(monkeypatch):
    """Simulate the ``cmap`` package being absent.

    Patches ``importlib.util.find_spec`` so a lookup of ``"cmap"`` returns
    ``None`` (as it would when the ``[science-colors]`` extra is not installed),
    while every other spec lookup is delegated to the real implementation.

    Yields:
        None: The patch is active for the duration of the test.
    """
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "cmap":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    return None


class TestResolveColormap:
    """Tests for ``cleopatra.colors.resolve_colormap``."""

    def test_plain_name_resolves_via_matplotlib(self):
        """A plain matplotlib name resolves to the matching ``Colormap``.

        Test scenario:
            ``"viridis"`` contains no colon, so it goes straight to matplotlib
            and comes back as the built-in ``viridis`` colormap.
        """
        result = resolve_colormap("viridis")
        assert isinstance(result, Colormap), f"expected a Colormap, got {type(result)!r}"
        assert result.name == "viridis", f"expected name 'viridis', got {result.name!r}"

    def test_plain_reversed_name_resolves(self):
        """A plain ``_r`` name stays on the matplotlib path.

        Test scenario:
            ``"viridis_r"`` has no colon, so it must not touch the optional
            package and resolves to matplotlib's reversed viridis.
        """
        result = resolve_colormap("viridis_r")
        assert result.name == "viridis_r", f"expected 'viridis_r', got {result.name!r}"

    def test_plain_name_does_not_invoke_cmap_guard(self, monkeypatch):
        """A plain name never calls ``_require_cmap`` (never touches ``cmap``).

        Test scenario:
            ``_require_cmap`` is replaced with a sentinel that fails if called;
            resolving a plain name must still succeed, proving the colon dispatch
            keeps plain names entirely off the optional-package path.
        """
        def _boom(action):
            raise AssertionError(f"_require_cmap should not be called for a plain name; got {action!r}")

        monkeypatch.setattr("cleopatra.colors._require_cmap", _boom)
        result = resolve_colormap("coolwarm_r")
        assert result.name == "coolwarm_r", f"expected 'coolwarm_r', got {result.name!r}"

    def test_colormap_object_returned_unchanged(self):
        """A ``Colormap`` instance is returned unchanged (idempotent).

        Test scenario:
            Passing an already-resolved ``Colormap`` returns the very same
            object, so the resolver is safe to call on values that may already
            be colormaps (as ``DATA_STYLES`` stores).
        """
        cmap = mpl.colormaps["plasma"]
        assert resolve_colormap(cmap) is cmap, "a Colormap object must be returned unchanged"

    def test_namespaced_resolves_to_matplotlib_colormap(self):
        """A namespaced name resolves through the ``cmap`` package.

        Test scenario:
            With the ``[science-colors]`` extra present, ``"cmocean:thermal"``
            resolves to a genuine matplotlib ``Colormap``.
        """
        pytest.importorskip("cmap", reason="requires the [science-colors] extra")
        result = resolve_colormap("cmocean:thermal")
        assert isinstance(result, Colormap), f"expected a Colormap, got {type(result)!r}"

    def test_namespaced_reversed_suffix_resolves(self):
        """The ``_r`` suffix works on a namespaced name.

        Test scenario:
            ``"cmocean:thermal_r"`` resolves to a ``Colormap`` distinct from its
            forward version (reversed lookup is honoured by the ``cmap`` package).
        """
        pytest.importorskip("cmap", reason="requires the [science-colors] extra")
        forward = resolve_colormap("cmocean:thermal")
        reverse = resolve_colormap("cmocean:thermal_r")
        assert isinstance(reverse, Colormap), f"expected a Colormap, got {type(reverse)!r}"
        assert forward(0.0) != reverse(0.0), "reversed colormap should differ at the low end"

    def test_unknown_namespaced_name_raises_valueerror(self):
        """An unknown namespaced name raises ``ValueError`` naming the kwarg.

        Test scenario:
            With ``cmap`` installed, a bad collection/name pair is wrapped in a
            ``ValueError`` that names the offending keyword (default ``cmap``).
        """
        pytest.importorskip("cmap", reason="requires the [science-colors] extra")
        with pytest.raises(ValueError, match=r"cmap=") as exc_info:
            resolve_colormap("cmocean:not_a_real_colormap")
        assert "not_a_real_colormap" in str(exc_info.value), f"message should echo the bad name: {exc_info.value}"

    def test_unknown_namespaced_name_uses_param_in_message(self):
        """The ``param`` argument is woven into the error message.

        Test scenario:
            A caller-supplied ``param="face_cmap"`` appears in the ``ValueError``
            so the failure points at the right keyword.
        """
        pytest.importorskip("cmap", reason="requires the [science-colors] extra")
        with pytest.raises(ValueError, match=r"face_cmap=") as exc_info:
            resolve_colormap("cmocean:nope", param="face_cmap")
        assert "face_cmap" in str(exc_info.value), f"custom param name missing from message: {exc_info.value}"

    def test_namespaced_without_extra_raises_importerror(self, without_cmap):
        """A namespaced name without the extra raises an actionable ImportError.

        Args:
            without_cmap: Fixture making ``find_spec('cmap')`` return ``None``.

        Test scenario:
            When ``cmap`` is not installed, ``"cmocean:thermal"`` raises
            ``ImportError`` whose message names the ``[science-colors]`` extra.
        """
        with pytest.raises(ImportError, match=r"\[science-colors\]") as exc_info:
            resolve_colormap("cmocean:thermal")
        assert "cmocean:thermal" in str(exc_info.value), f"message should name the colormap: {exc_info.value}"

    def test_plain_name_works_without_extra(self, without_cmap):
        """Plain names resolve even when the extra is absent.

        Args:
            without_cmap: Fixture making ``find_spec('cmap')`` return ``None``.

        Test scenario:
            A missing ``cmap`` package must not affect built-in colormap names.
        """
        result = resolve_colormap("magma")
        assert result.name == "magma", f"expected 'magma', got {result.name!r}"


class TestRequireCmap:
    """Tests for the ``_require_cmap`` install guard."""

    def test_no_raise_when_present(self, monkeypatch):
        """The guard is a no-op when ``cmap`` is importable.

        Test scenario:
            With ``find_spec('cmap')`` returning a truthy spec, ``_require_cmap``
            returns ``None`` without raising.
        """
        monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
        assert _require_cmap("an action") is None, "_require_cmap must not raise when cmap is present"

    def test_raises_actionable_importerror_when_absent(self, without_cmap):
        """The guard raises an actionable ``ImportError`` when ``cmap`` is absent.

        Args:
            without_cmap: Fixture making ``find_spec('cmap')`` return ``None``.

        Test scenario:
            The message must echo the ``action`` context and name both the
            ``cmap`` package and the ``[science-colors]`` extra.
        """
        with pytest.raises(ImportError) as exc_info:
            _require_cmap("A namespaced colormap (cmap='cmocean:thermal')")
        message = str(exc_info.value)
        assert "cmap" in message, f"message should name the cmap package: {message}"
        assert "[science-colors]" in message, f"message should name the extra: {message}"
        assert "cmocean:thermal" in message, f"message should echo the action context: {message}"


class TestCategoricalDefaultInvariant:
    """The resolver must preserve built-in names so name comparisons hold."""

    def test_default_cmap_name_survives_resolution(self):
        """Resolving the shared default colormap preserves its name.

        Test scenario:
            ``Glyph._prepare_categorical_mapping`` decides whether to swap in the
            qualitative default by comparing the resolved colormap's ``.name`` to
            ``STYLE_DEFAULTS['cmap']``. If resolution renamed a built-in (as the
            ``cmap`` package does), that comparison would silently break. This
            asserts the default's name is unchanged by ``resolve_colormap``.
        """
        default_name = STYLE_DEFAULTS["cmap"]
        resolved = resolve_colormap(default_name)
        assert resolved.name == default_name, (
            f"default cmap name must survive resolution: {default_name!r} -> {resolved.name!r}"
        )
