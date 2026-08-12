"""Tests for grouped rendering-parameter objects in ``cleopatra.styling.params``.

Focused unit tests for the object-owned logic (currently the
``DataStyle.for_apply_style`` factory); the per-field ``to_options`` flattening
is covered by the module doctests.
"""

from __future__ import annotations

import pytest

from cleopatra.styling.params import _UNSET, DataStyle, _Unset


class TestDataStyleForApplyStyle:
    """Tests for ``DataStyle.for_apply_style``."""

    def test_hillshade_unset_omits_hillshade(self):
        """Leaving ``hillshade`` unset builds ``DataStyle(style=...)`` only.

        Test scenario:
            The default sentinel means "not passed", so ``to_options()`` emits
            just ``style`` and keeps any sticky relief shading.
        """
        ds = DataStyle.for_apply_style("dem")
        assert ds.to_options() == {"style": "dem"}, f"unexpected options: {ds.to_options()}"

    @pytest.mark.parametrize("hillshade", [True, False, {"azimuth": 315}, None])
    def test_hillshade_given_flows_through(self, hillshade):
        """An explicit ``hillshade`` (incl. ``None``/``False``) is folded in.

        Args:
            hillshade: The explicit override to forward.

        Test scenario:
            Any non-sentinel value -- a dict, ``True``/``False``, or an explicit
            ``None`` that clears sticky shading -- is emitted alongside ``style``.
        """
        ds = DataStyle.for_apply_style("dem", hillshade=hillshade)
        assert ds.to_options() == {
            "style": "dem",
            "hillshade": hillshade,
        }, f"hillshade not forwarded: {ds.to_options()}"

    def test_explicit_unset_sentinel_behaves_as_unset(self):
        """Passing ``_UNSET`` explicitly behaves like omitting ``hillshade``.

        Test scenario:
            ``for_apply_style(..., hillshade=_UNSET)`` must not emit a
            ``hillshade`` key.
        """
        ds = DataStyle.for_apply_style("dem", hillshade=_UNSET)
        assert "hillshade" not in ds.to_options(), (
            f"sentinel should not emit hillshade: {ds.to_options()}"
        )

    def test_style_none_clears_preset(self):
        """``style=None`` flows through to clear a sticky preset.

        Test scenario:
            ``for_apply_style(None)`` emits ``style=None`` (the clear signal).
        """
        ds = DataStyle.for_apply_style(None)
        assert ds.to_options() == {"style": None}, f"unexpected options: {ds.to_options()}"


class TestUnsetSentinel:
    """Tests for the `DataStyle` "not passed" sentinel `_UNSET` / `_Unset`."""

    def test_repr_is_readable(self):
        """`repr(_UNSET)` reads `<unset>` for legible `help()` / IDE tooltips.

        Test scenario:
            The sentinel exists over a plain `object()` precisely for a
            readable repr; deleting `__repr__` would regress to the default
            `<...params._Unset object at 0x...>`.
        """
        assert repr(_UNSET) == "<unset>", f"Unexpected repr: {repr(_UNSET)!r}"

    def test_is_singleton_identity(self):
        """`_UNSET` is a single shared instance, compared with `is` not `==`.

        Test scenario:
            Gates test `value is _UNSET` (or `isinstance(value, _Unset)`); a
            fresh `_Unset()` must not be the singleton nor compare equal to it.
        """
        other = _Unset()
        assert other is not _UNSET, "A fresh _Unset() must not be the _UNSET singleton"
        assert other != _UNSET, "Two distinct _Unset instances must not compare equal"
