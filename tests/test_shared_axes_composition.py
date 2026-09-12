"""Tests for composing glyphs onto one axes -- issue #346.

`VectorGlyph` documents `add_colorbar=False` as being "for shared-axes
composition", but any second glyph drawn onto an axes removed the first one's
artists, so the classic scalar-field-plus-wind-arrows figure was impossible.

Composition is opt-in via `compose=True`, because the default is what issue #210
needs: a second glyph bound to an existing axes *replaces* the first rather than
orphaning its artists. Both contracts are asserted here so neither can quietly
become the other.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cleopatra.glyphs.base.glyph import _mark_render_artists, _render_owner_token
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
from cleopatra.glyphs.gridded.vector_glyph import VectorGlyph


@pytest.fixture
def field():
    """Provide a small vector field on a 20x30 grid.

    Returns:
        tuple: `(x, y, u, v)` meshgrid coordinates and components.
    """
    x, y = np.meshgrid(np.arange(30), np.arange(20))
    rng = np.random.default_rng(0)
    return x, y, rng.random((20, 30)), rng.random((20, 30))


@pytest.fixture
def scalar():
    """Provide a 20x30 scalar field.

    Returns:
        np.ndarray: The array to render as the host layer.
    """
    return np.random.default_rng(1).random((20, 30))


class TestComposeOntoSharedAxes:
    """`compose=True` draws over an existing layer instead of replacing it."""

    def test_vector_over_scalar_keeps_the_host_image(self, field, scalar):
        """Arrows drawn over a raster leave the raster in place.

        Test scenario:
            The figure issue #346 is about. Without `compose` the scalar layer
            was removed and only the arrows survived.
        """
        x, y, u, v = field
        fig, ax = plt.subplots()
        ArrayGlyph(scalar).plot(ax=ax)
        assert len(ax.images) == 1, "precondition: the host layer is drawn"

        VectorGlyph(x, y, u, v, ax=ax, add_colorbar=False).plot(
            kind="quiver", ax=ax, compose=True
        )
        assert len(ax.images) == 1, "the host raster was cleared"
        assert len(ax.collections) == 1, "the arrows were not drawn"
        plt.close(fig)

    def test_host_colorbar_survives(self, field, scalar):
        """The host's colorbar is not removed by the overlay.

        Test scenario:
            `add_colorbar=False` exists so the host can own the single shared
            colorbar -- removing it was the opposite of that promise.
        """
        x, y, u, v = field
        fig, ax = plt.subplots()
        ArrayGlyph(scalar).plot(ax=ax)
        before = len(fig.axes)

        VectorGlyph(x, y, u, v, ax=ax, add_colorbar=False).plot(
            kind="quiver", ax=ax, compose=True
        )
        assert len(fig.axes) == before, "the host's colorbar axes was removed"
        plt.close(fig)

    def test_array_over_array_composes(self, scalar):
        """`compose` works on `ArrayGlyph` too, not only the vector overlay.

        Test scenario:
            The flag belongs to whichever glyph is drawn second, whatever it is.
        """
        fig, ax = plt.subplots()
        ArrayGlyph(scalar).plot(ax=ax)
        ArrayGlyph(scalar).plot(ax=ax, compose=True)
        assert len(ax.images) == 2, "the second raster replaced the first"
        plt.close(fig)

    def test_a_glyph_still_replaces_its_own_artists_when_composing(self, field, scalar):
        """Composing does not make a glyph orphan its *own* previous artists.

        Test scenario:
            `compose` narrows the clear to the caller's own artists -- it does
            not switch it off. Replotting the same glyph must still replace.
        """
        x, y, u, v = field
        fig, ax = plt.subplots()
        ArrayGlyph(scalar).plot(ax=ax)
        overlay = VectorGlyph(x, y, u, v, ax=ax, add_colorbar=False)
        overlay.plot(kind="quiver", ax=ax, compose=True)
        overlay.plot(kind="quiver", ax=ax, compose=True)
        overlay.plot(kind="quiver", ax=ax, compose=True)
        assert len(ax.collections) == 1, "the overlay orphaned its own arrows"
        assert len(ax.images) == 1, "the host layer was lost"
        plt.close(fig)

    def test_throwaway_host_glyph_is_not_mistaken_for_the_overlay(self, field, scalar):
        """A collected host glyph does not have its identity reused.

        Test scenario:
            `ArrayGlyph(arr).plot(ax=ax)` leaves the glyph unreferenced, so
            CPython readily hands its `id()` to the next glyph allocated. Keying
            ownership by `id()` made the overlay clear the host's artists
            anyway; the tracker uses a monotonic token instead.
        """
        x, y, u, v = field
        fig, ax = plt.subplots()
        ArrayGlyph(scalar).plot(ax=ax)  # deliberately not kept alive
        VectorGlyph(x, y, u, v, ax=ax, add_colorbar=False).plot(
            kind="quiver", ax=ax, compose=True
        )
        assert len(ax.images) == 1, "a recycled id let the overlay clear the host"
        plt.close(fig)


class TestDefaultStillReplaces:
    """Without `compose`, a second glyph replaces -- the issue #210 contract."""

    def test_vector_over_scalar_replaces_by_default(self, field, scalar):
        """The default is unchanged: the host layer is cleared.

        Test scenario:
            Issue #210's orphaned-artist defect (an animation frozen at frame 0)
            is prevented by this clearing, so it must remain the default.
        """
        x, y, u, v = field
        fig, ax = plt.subplots()
        ArrayGlyph(scalar).plot(ax=ax)
        VectorGlyph(x, y, u, v, ax=ax, add_colorbar=False).plot(kind="quiver", ax=ax)
        assert len(ax.images) == 0, "default behaviour changed; issue #210 may regress"
        plt.close(fig)


class TestRenderArtistRegistry:
    """Direct tests for the ownership bookkeeping."""

    def test_no_owner_uses_a_shared_bucket(self):
        """`None` maps to a single reserved token, not a per-call one.

        Test scenario:
            A caller with no glyph to attribute artists to still needs a stable
            bucket to mark and clear under.
        """
        assert _render_owner_token(None) == 0
        assert _render_owner_token(None) == 0, "the unowned bucket must be stable"

    def test_each_owner_gets_its_own_token(self):
        """Two glyphs never share a token.

        Test scenario:
            Sharing one would let either clear the other's artists, which is the
            defect this keying exists to prevent.
        """

        class _Owner:
            pass

        first, second = _Owner(), _Owner()
        assert _render_owner_token(first) != _render_owner_token(second)
        assert _render_owner_token(first) == _render_owner_token(first), (
            "token must be stable"
        )

    def test_stale_entries_are_pruned_on_the_next_mark(self):
        """A throwaway glyph's detached artists do not accumulate.

        Test scenario:
            `SomeGlyph(...).plot(ax=ax)` never comes back to clear its own
            entry, so without pruning the registry would grow by one per such
            call for the life of the axes.
        """

        class _Owner:
            pass

        fig, ax = plt.subplots()
        gone = ax.plot([0, 1], [0, 1])[0]
        _mark_render_artists(ax, _Owner(), gone)
        assert len(ax._cleo_render_artists) == 1, "precondition: one entry recorded"

        gone.remove()  # the artist is detached, but its entry lingers
        _mark_render_artists(ax, _Owner(), ax.plot([0, 1], [1, 0])[0])
        assert len(ax._cleo_render_artists) == 1, (
            f"stale entry not pruned: {ax._cleo_render_artists}"
        )
        plt.close(fig)


class TestArrowThinning:
    """`quiver` and `barbs` can both be thinned -- one arrow per cell is unusable."""

    @pytest.mark.parametrize("kind", ["quiver", "barbs"])
    @pytest.mark.parametrize("thin, expected", [(1, 600), (2, 150), (5, 24)])
    def test_thin_subsamples_the_grid(self, field, kind, thin, expected):
        """`thin=n` draws every nth point along each axis, for either arrow kind.

        Args:
            field: The vector-field fixture.
            kind: The per-grid-point vector kind under test.
            thin: The subsampling step.
            expected: The arrow count it should leave from a 20x30 grid.

        Test scenario:
            A 141x321 window is 45,261 arrows, so callers had to subsample the
            data and rebuild a coarser grid themselves. `barbs` draws one glyph
            per point exactly as `quiver` does and `_thinned` handles both, so
            the option cannot be quiver-only.
        """
        x, y, u, v = field
        fig, ax = plt.subplots()
        VectorGlyph(x, y, u, v, thin=thin, add_colorbar=False).plot(kind=kind, ax=ax)
        assert len(ax.collections[0].get_offsets()) == expected, (
            f"{kind} thin={thin} gave {len(ax.collections[0].get_offsets())} arrows"
        )
        plt.close(fig)

    def test_thin_on_barbs_does_not_warn(self, field):
        """`barbs` thins silently; only `streamplot` is told the option is inert.

        Args:
            field: The vector-field fixture.

        Test scenario:
            `_validate_thin` warns for `streamplot`, which seeds its own lines.
            `barbs` has a glyph per grid point to drop, so warning there would
            tell a caller their thinning did nothing when it had just worked.
        """
        x, y, u, v = field
        fig, ax = plt.subplots()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            VectorGlyph(x, y, u, v, thin=5, add_colorbar=False).plot(
                kind="barbs", ax=ax
            )
        assert len(ax.collections[0].get_offsets()) == 24, "barbs were not thinned"
        plt.close(fig)

    def test_thin_accepts_one_dimensional_coordinates(self, field):
        """1-D coordinate vectors are thinned on their only axis.

        Test scenario:
            `x`/`y` may be either 1-D vectors or a full meshgrid, and the two
            index differently.
        """
        _, _, u, v = field
        fig, ax = plt.subplots()
        VectorGlyph(
            np.arange(30), np.arange(20), u, v, thin=2, add_colorbar=False
        ).plot(kind="quiver", ax=ax)
        assert len(ax.collections) == 1, "1-D coordinates failed to thin"
        plt.close(fig)

    @pytest.mark.parametrize("kind", ["quiver", "barbs"])
    @pytest.mark.parametrize("bad", [0, -1, 2.5, "2", True])
    def test_invalid_thin_raises(self, field, kind, bad):
        """A non-positive-integer `thin` raises `ValueError` for either kind.

        Args:
            field: The vector-field fixture.
            kind: The per-grid-point vector kind under test.
            bad: The invalid step under test.

        Test scenario:
            A float or a bool would silently misindex rather than fail, and the
            check runs before any kind-specific branch, so `barbs` must be
            rejected on the same terms as `quiver`.
        """
        x, y, u, v = field
        with pytest.raises(ValueError, match="thin must be a positive integer"):
            VectorGlyph(x, y, u, v, thin=bad, add_colorbar=False).plot(kind=kind)
        plt.close("all")


class TestGroupedParameterTypeError:
    """A loose value passed to a typed style parameter says what it wanted."""

    def test_color_string_names_the_expected_type(self, field):
        """`color="black"` raises `TypeError` naming the grouped objects.

        Test scenario:
            It used to raise `'str' object has no attribute 'to_options'`, which
            named neither the parameter nor what it expected.
        """
        x, y, u, v = field
        with pytest.raises(TypeError, match="grouped parameter object"):
            VectorGlyph(x, y, u, v, add_colorbar=False).plot(
                kind="quiver", color="black"
            )
        plt.close("all")

    def test_error_mentions_a_concrete_type(self, field):
        """The message names at least one usable grouped type.

        Test scenario:
            "expected a grouped parameter object" alone would not tell a caller
            what to pass instead.
        """
        x, y, u, v = field
        with pytest.raises(TypeError, match="ColorScaling"):
            VectorGlyph(x, y, u, v, add_colorbar=False).plot(
                kind="quiver", color="black"
            )
        plt.close("all")


class TestComposeLeavesHostChromeAlone:
    """M2/M3 -- composing draws over the host without restyling its axes."""

    def test_untitled_overlay_keeps_the_host_title(self, scalar):
        """An overlay with no title of its own does not blank the host's.

        Test scenario:
            `ArrayGlyph` sets the title unconditionally, and its default is
            empty -- so composing an untitled overlay wiped the caption the host
            had put there.
        """
        fig, ax = plt.subplots()
        ArrayGlyph(scalar, title="HOST TITLE").plot(ax=ax)
        ArrayGlyph(scalar).plot(ax=ax, compose=True)
        assert ax.get_title() == "HOST TITLE", "the overlay blanked the host's title"
        plt.close(fig)

    def test_titled_overlay_still_sets_its_title(self, scalar):
        """An overlay that has a title still applies it.

        Test scenario:
            Preserving the host's caption must not stop an overlay that
            genuinely wants to retitle the axes.
        """
        fig, ax = plt.subplots()
        ArrayGlyph(scalar, title="HOST").plot(ax=ax)
        ArrayGlyph(scalar, title="OVERLAY").plot(ax=ax, compose=True)
        assert ax.get_title() == "OVERLAY"
        plt.close(fig)

    def test_replacing_render_still_retitles(self, scalar):
        """Without `compose` the title behaves exactly as before.

        Test scenario:
            A replacing render owns the axes, so its (empty) title applies.
        """
        fig, ax = plt.subplots()
        ArrayGlyph(scalar, title="HOST").plot(ax=ax)
        ArrayGlyph(scalar).plot(ax=ax)
        assert ax.get_title() == ""
        plt.close(fig)
