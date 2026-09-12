"""Regression cover for the round-2 review findings on PR #350.

Each test here corresponds to a defect the second review round found in the
round-1 fixes themselves -- the `compose=` contract leaking on `animate`, on the
extent-less path and under a styled preset, render kwargs bleeding into the set
that decides figure sizing, and the ownership registry evicting an entry whose
only *reporting* artist had gone.

It also covers the helpers the round-2 fixes introduced on their own terms:
`_artist_is_attached` asking each kind of tracked artist in its own vocabulary,
`_render_owner_token` for an owner it cannot weakly reference, `MeshGlyph`
carrying its construction-time axis options across the plot-time reset, and
`VectorGlyph` honouring the composed-overlay colorbar default.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import gc

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cleopatra.glyphs.base.glyph import (
    _apply_axis_options,
    _artist_is_attached,
    _entry_is_detached,
    _render_owner_token,
    _render_owner_tokens,
)
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
from cleopatra.glyphs.gridded.vector_glyph import VectorGlyph
from cleopatra.glyphs.stats.histogram_glyph import HistogramGlyph
from cleopatra.styling.params import DataStyle


@pytest.fixture
def arr():
    """Provide a small deterministic array to render.

    Returns:
        np.ndarray: A 20x30 array of values in [0, 1).
    """
    return np.random.default_rng(0).random((20, 30))


@pytest.fixture
def frames():
    """Provide a small deterministic frame stack to animate.

    Returns:
        np.ndarray: Three 20x30 frames of values in [0, 1).
    """
    return np.random.default_rng(1).random((3, 20, 30))


@pytest.fixture
def field():
    """Provide a small deterministic vector field on a 20x30 grid.

    Returns:
        tuple: `(x, y, u, v)` -- meshgrid coordinates and components.
    """
    grid_x, grid_y = np.meshgrid(np.arange(30), np.arange(20))
    rng = np.random.default_rng(2)
    return grid_x, grid_y, rng.random((20, 30)), rng.random((20, 30))


@pytest.fixture
def mesh():
    """Provide a two-cell quad mesh and one value per face.

    Returns:
        tuple: `(node_x, node_y, faces, data)` -- six nodes in a 3x2 lattice,
        two quad faces, and a value for each.
    """
    node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    faces = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
    return node_x, node_y, faces, np.array([1.0, 2.0])


@pytest.fixture
def samples():
    """Provide a deterministic sample for the histogram glyph.

    Returns:
        np.ndarray: 200 normal deviates.
    """
    return np.random.default_rng(3).normal(size=200)


@pytest.fixture
def host(arr):
    """Provide an axes with a titled, extent-bearing glyph already on it.

    Args:
        arr: The array fixture.

    Yields:
        matplotlib.axes.Axes: The host axes.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ArrayGlyph(arr, title="HOST", extent=[0, 0, 10, 10]).plot(ax=ax)
    yield ax
    plt.close("all")


class TestRenderKwargsDoNotResizeTheFigure:
    """`plot(figsize=...)` keeps the behaviour it had before the axis-option fix."""

    def test_plot_figsize_is_still_auto_computed(self, arr):
        """A `figsize` passed to `plot()` does not switch auto-sizing off.

        Args:
            arr: The array fixture.

        Test scenario:
            Recording render kwargs as explicit made `create_figure_axes` read
            them, which turned `plot(figsize=...)` from ignored into honoured --
            an unannounced rendering change for every existing caller. The two
            sets are now separate.
        """
        fig, _ = ArrayGlyph(arr).plot(figsize=(8, 3))
        assert fig.get_size_inches().tolist() != [8.0, 3.0], (
            "plot(figsize=) must stay auto-sized, as it was before the fix"
        )
        plt.close(fig)

    def test_constructor_figsize_is_still_honoured(self, arr):
        """A `figsize` passed to the constructor is still taken literally.

        Args:
            arr: The array fixture.

        Test scenario:
            The separation must not cost the constructor form, which is the one
            `create_figure_axes` was always meant to read.
        """
        fig, _ = ArrayGlyph(arr, figsize=(8, 3)).plot()
        assert fig.get_size_inches().tolist() == [8.0, 3.0], (
            f"constructor figsize ignored: {fig.get_size_inches()}"
        )
        plt.close(fig)

    def test_a_failed_plot_leaves_no_explicit_keys_behind(self, arr):
        """A `plot()` that raises does not change the next call's output.

        Args:
            arr: The array fixture.

        Test scenario:
            The keys were recorded before the validation loop, so
            `plot(figsize=(9, 2), nope=1)` raised and still left `figsize`
            marked explicit for every later call.
        """
        glyph = ArrayGlyph(arr)
        with pytest.raises(ValueError, match="nope"):
            glyph.plot(figsize=(9, 2), nope=1)
        assert "nope" not in getattr(glyph, "_render_explicit_options", set()), (
            "a rejected key was still recorded as explicit"
        )
        fig, _ = glyph.plot()
        assert fig.get_size_inches().tolist() != [9.0, 2.0], (
            "the failed call's figsize leaked into the next render"
        )
        plt.close(fig)


class TestDroppedOptionsStopBeingApplied:
    """The explicit set is the constructor's keys plus this call's, nothing more."""

    def test_a_later_call_does_not_re_apply_a_dropped_option(self, arr):
        """An option the caller stops passing stops being applied.

        Args:
            arr: The array fixture.

        Test scenario:
            The set only ever grew, so a second `plot(ax=ax)` that passed
            neither option still overwrote the label and grid the caller had
            since set on the axes themselves.
        """
        glyph = ArrayGlyph(arr, extent=[0, 0, 10, 10])
        fig, ax = plt.subplots()
        glyph.plot(ax=ax, xlabel="A", grid_alpha=0.3)
        ax.set_xlabel("USER SET")
        ax.grid(False)

        glyph.plot(ax=ax)
        assert ax.get_xlabel() == "USER SET", (
            f"a dropped option was re-applied: {ax.get_xlabel()!r}"
        )
        assert not any(line.get_visible() for line in ax.xaxis.get_gridlines()), (
            "a dropped grid_alpha re-drew the gridlines"
        )
        plt.close(fig)

    def test_a_constructor_option_survives_every_call(self, arr):
        """A constructor option keeps applying on calls that do not repeat it.

        Args:
            arr: The array fixture.

        Test scenario:
            Rebuilding the set per call must not lose the constructor's keys --
            those stay explicit for the life of the glyph.
        """
        glyph = ArrayGlyph(arr, xlabel="CTOR", extent=[0, 0, 10, 10])
        fig, ax = plt.subplots()
        glyph.plot(ax=ax)
        glyph.plot(ax=ax)
        assert ax.get_xlabel() == "CTOR", (
            f"constructor option lost on a later call: {ax.get_xlabel()!r}"
        )
        plt.close(fig)

    def test_a_render_kwarg_still_reaches_the_axes(self, arr):
        """The round-1 fix still holds: `plot(xlabel=...)` is not dropped.

        Args:
            arr: The array fixture.

        Test scenario:
            This is the defect the accumulation was introduced for; separating
            the two sets must not undo it.
        """
        glyph = ArrayGlyph(arr, extent=[0, 0, 10, 10])
        fig, ax = plt.subplots()
        glyph.plot(ax=ax, xlabel="PLOTX", xtick_font_size=20)
        assert ax.get_xlabel() == "PLOTX", f"render xlabel dropped: {ax.get_xlabel()!r}"
        assert ax.get_xticklabels()[0].get_fontsize() == 20.0, (
            "render xtick_font_size dropped"
        )
        plt.close(fig)


class TestComposeLeavesTheHostChromeAlone:
    """Every line that touches shared axes chrome honours `compose=`."""

    def test_animate_keeps_the_host_title(self, frames, host):
        """An untitled composed animation does not blank the host's title.

        Args:
            frames: The frame-stack fixture.
            host: The pre-rendered host axes.

        Test scenario:
            `animate` set the title unconditionally, so an overlay with no title
            of its own wrote its empty default over the host's caption.
        """
        ArrayGlyph(frames, ax=host, add_colorbar=False).animate(
            list(range(3)), compose=True
        )
        assert host.get_title() == "HOST", (
            f"composed animation blanked the host title: {host.get_title()!r}"
        )

    def test_animate_keeps_the_host_tick_labels(self, frames, host):
        """A composed animation does not strip the host's ticks.

        Args:
            frames: The frame-stack fixture.
            host: The pre-rendered host axes.

        Test scenario:
            `animate` called `set_xticklabels([])` and `set_xticks([])` on the
            host's axes regardless of `compose`.
        """
        ArrayGlyph(frames, ax=host, add_colorbar=False).animate(
            list(range(3)), compose=True
        )
        assert host.get_xticks().size, "composed animation removed the host's x ticks"
        assert host.get_yticks().size, "composed animation removed the host's y ticks"

    def test_animate_without_compose_still_replaces(self, frames, host):
        """The default `animate` still owns the axes outright.

        Args:
            frames: The frame-stack fixture.
            host: The pre-rendered host axes.

        Test scenario:
            Issue #210's replace-by-default contract must survive the guard.
        """
        ArrayGlyph(frames, ax=host, title="OVERLAY", add_colorbar=False).animate(
            list(range(3))
        )
        assert host.get_title() == "OVERLAY", (
            f"a plain animate no longer retitles: {host.get_title()!r}"
        )
        assert not host.get_xticks().size, "a plain animate no longer blanks the ticks"

    def test_an_extentless_overlay_keeps_the_host_ticks(self, arr, host):
        """An overlay built without an `extent` leaves the host's ticks alone.

        Args:
            arr: The array fixture.
            host: The pre-rendered host axes.

        Test scenario:
            The `extent is None and kind == "imshow"` branch blanked the ticks
            unconditionally, so the idiomatic bare `ArrayGlyph(arr)` overlay
            stripped the host's axes. The tick *values* legitimately move -- an
            extent-less overlay draws in pixel coordinates and widens the data
            limits -- so what is pinned here is that ticks and their labels
            still exist at all.
        """
        ArrayGlyph(arr, add_colorbar=False).plot(ax=host, compose=True)
        assert host.get_xticks().size, "extent-less overlay removed the host's x ticks"
        assert host.get_yticks().size, "extent-less overlay removed the host's y ticks"
        assert any(label.get_text() for label in host.get_xticklabels()), (
            "extent-less overlay blanked the host's tick labels"
        )

    def test_an_extentless_solo_render_still_blanks_its_ticks(self, arr):
        """Without `compose` a pixel-space render still hides its ticks.

        Args:
            arr: The array fixture.

        Test scenario:
            Row/column indices are meaningless axis labels; the guard must be
            compose-only, not a change to the solo path.
        """
        fig, ax = ArrayGlyph(arr).plot()
        assert not ax.get_xticks().size, "a solo extent-less render kept its ticks"
        plt.close(fig)

    def test_a_styled_overlay_keeps_the_host_background(self, arr, host):
        """A preset overlay does not repaint the host's axes facecolor.

        Args:
            arr: The array fixture.
            host: The pre-rendered host axes.

        Test scenario:
            `_apply_style_background` ran before the `compose` guard, so a
            dark-canvas preset blackened a light host.
        """
        before = host.get_facecolor()
        ArrayGlyph(arr, add_colorbar=False).plot(
            ax=host, compose=True, data_style=DataStyle(style="temperature_flame")
        )
        assert host.get_facecolor() == before, (
            f"composed preset repainted the host: {host.get_facecolor()}"
        )

    def test_a_solo_styled_render_still_paints_its_background(self, arr):
        """Without `compose` a dark preset still owns the canvas.

        Args:
            arr: The array fixture.

        Test scenario:
            The guard must not cost the preset its background on the path it
            does own.
        """
        fig, ax = plt.subplots()
        ArrayGlyph(arr, add_colorbar=False).plot(
            ax=ax, data_style=DataStyle(style="temperature_flame")
        )
        assert ax.get_facecolor()[:3] == (0.0, 0.0, 0.0), (
            f"the solo styled render lost its background: {ax.get_facecolor()}"
        )
        plt.close(fig)


class TestRegistryProvesDeathBeforeEvicting:
    """An entry is dropped only when every artist in it has actually gone."""

    def test_a_colorbar_outliving_its_image_keeps_the_entry(self, arr):
        """A live colorbar is not evicted because its image was removed.

        Args:
            arr: The array fixture.

        Test scenario:
            Deadness was read off the artists that expose `.axes` and the rest
            were ignored, so an entry of `[Colorbar, AxesImage]` reported
            detached the moment the image went -- dropping the still-live
            colorbar from the registry, where no later render could clear it.
        """
        fig, ax = plt.subplots()
        glyph = ArrayGlyph(arr, extent=[0, 0, 10, 10])
        glyph.plot(ax=ax)
        group = [glyph.cbar, glyph.im]
        group[1].remove()
        assert not _entry_is_detached(group), "a live colorbar was read as a dead entry"
        plt.close(fig)

    def test_an_entry_whose_artists_all_left_is_evicted(self, arr):
        """A genuinely dead entry is still pruned.

        Args:
            arr: The array fixture.

        Test scenario:
            Proving life must not become never proving death -- a throwaway
            glyph's entry has to go, or the registry only grows.
        """
        fig, ax = plt.subplots()
        glyph = ArrayGlyph(arr, extent=[0, 0, 10, 10])
        glyph.plot(ax=ax)
        group = [glyph.cbar, glyph.im]
        group[0].remove()
        group[1].remove()
        assert _entry_is_detached(group), "a dead entry was kept"
        plt.close(fig)

    def test_an_entry_nothing_can_be_asked_about_is_kept(self):
        """An entry of artists that cannot report is not assumed dead.

        Test scenario:
            An empty entry, or one holding only objects that answer neither
            `.axes` nor `.ax` nor indexing, is cheaper to keep than to guess
            about -- guessing wrong orphans live artists.
        """
        assert not _entry_is_detached([]), "an empty entry was read as dead"
        assert not _entry_is_detached([object()]), "an unaskable entry was read as dead"

    def test_a_bar_container_is_asked_through_its_children(self, samples):
        """A `BarContainer` answers through the bars it holds.

        Args:
            samples: The histogram-sample fixture.

        Test scenario:
            A `Container` exposes neither `.axes` nor `.ax`, which is what made
            "no `.axes`" read as "detached" and evicted live `HistogramGlyph`
            entries. `_artist_is_attached` indexes into it and asks a bar
            instead, so the container reports attached while its bars are on the
            axes and detached once they have all been removed.
        """
        fig, ax, _ = HistogramGlyph(samples).histogram()
        container = ax.containers[0]
        assert _artist_is_attached(container) is True, (
            "a live bar container was not recognised as attached"
        )
        for bar in list(container):
            bar.remove()
        assert _artist_is_attached(container) is False, (
            "a bar container whose bars had all gone still reported attached"
        )
        plt.close(fig)

    def test_an_artist_that_answers_nothing_is_unknown(self):
        """An object with no axes, no `.ax` and no children answers `None`.

        Test scenario:
            `None` is a third answer, distinct from `False`: it means the artist
            could not be asked, so `_entry_is_detached` discounts it rather than
            counting it as proof of death.
        """
        assert _artist_is_attached(object()) is None, (
            "an unaskable artist claimed to know whether it was attached"
        )

    def test_a_colorbar_is_asked_through_its_own_axes(self, arr):
        """A `Colorbar` answers through the `.ax` it lives on.

        Args:
            arr: The array fixture.

        Test scenario:
            A `Colorbar` has no `.axes` either; `remove()` takes its own axes
            off the figure, which is the only signal there is.
        """
        fig, ax = plt.subplots()
        glyph = ArrayGlyph(arr, extent=[0, 0, 10, 10])
        glyph.plot(ax=ax)
        assert _artist_is_attached(glyph.cbar) is True, (
            "a live colorbar was not recognised as attached"
        )
        glyph.cbar.remove()
        assert _artist_is_attached(glyph.cbar) is False, (
            "a removed colorbar still reported attached"
        )
        plt.close(fig)


class _PlainOwner:
    """A stand-in glyph with no equality of its own."""


class _SlotsOwner:
    """A stand-in glyph that cannot be weak-referenced."""

    __slots__ = ()


class _EqualOwner:
    """A stand-in glyph that compares equal to every other of its kind."""

    def __eq__(self, other):
        """Compare equal to any other `_EqualOwner`.

        Args:
            other: The object to compare against.

        Returns:
            bool: `True` for any other `_EqualOwner`.
        """
        return isinstance(other, _EqualOwner)

    def __hash__(self):
        """Hash to one bucket, so equality decides dictionary identity.

        Returns:
            int: A constant.
        """
        return 7


class TestOwnershipTokensAreKeyedByIdentity:
    """Two distinct glyphs never share a token, however they compare."""

    def test_equal_but_distinct_owners_get_distinct_tokens(self):
        """Value equality does not merge two owners.

        Test scenario:
            A `WeakKeyDictionary` keys on `__hash__`/`__eq__`, so the day a
            glyph defines value equality, two equal glyphs would have cleared
            each other's artists even under `compose=True`.
        """
        first, second = _EqualOwner(), _EqualOwner()
        assert first == second, "precondition: the two owners compare equal"
        assert _render_owner_token(first) != _render_owner_token(second), (
            "two equal-but-distinct owners shared a render token"
        )

    def test_a_repeat_lookup_does_not_assign_a_new_token(self):
        """Looking a token up neither changes it nor burns the counter.

        Test scenario:
            `setdefault(owner, next(counter) + 1)` evaluated `next()` eagerly,
            so every lookup advanced the counter -- harmless, but it
            contradicted the docstring and made the tokens unreadable.
        """
        owner = _PlainOwner()
        first = _render_owner_token(owner)
        for _ in range(5):
            assert _render_owner_token(owner) == first, "the token changed"
        assert _render_owner_token(_PlainOwner()) == first + 1, (
            "repeat lookups advanced the token counter"
        )

    def test_a_collected_owner_is_forgotten(self):
        """A glyph's entry goes when the glyph does, so its `id()` is reusable.

        Test scenario:
            Keying by `id()` is only safe while a collected owner's entry is
            dropped -- otherwise the next object handed that address inherits
            its artists.
        """
        owner = _PlainOwner()
        key = id(owner)
        _render_owner_token(owner)
        assert key in _render_owner_tokens, "precondition: the owner was tracked"
        del owner
        gc.collect()
        assert key not in _render_owner_tokens, "a collected owner's token survived it"

    def test_an_owner_that_cannot_be_weak_referenced_is_not_tracked(self):
        """A glyph with no weak-reference support falls back to bucket `0`.

        Test scenario:
            The registry is keyed by `id()` and pruned from a `weakref.finalize`
            callback, so an owner that cannot be weak-referenced has no way to
            be forgotten -- recording one would hand its artists to whatever
            object is allocated at that address next. It shares the unowned
            bucket with `owner=None` instead, and leaves no entry behind.
        """
        owner = _SlotsOwner()
        assert _render_owner_token(owner) == 0, (
            "a non-weak-referenceable owner was given a private token"
        )
        assert id(owner) not in _render_owner_tokens, (
            "a non-weak-referenceable owner was recorded in the registry"
        )
        assert _render_owner_token(None) == 0, "the unowned bucket is no longer token 0"

    def test_an_untracked_owner_does_not_burn_a_token(self, arr):
        """Falling back to bucket `0` leaves the counter where it was.

        Args:
            arr: The array fixture.

        Test scenario:
            The fallback returns before `next(_render_owner_counter)`, so a
            stream of untrackable owners cannot quietly advance the tokens
            handed to the trackable ones.
        """
        before = _render_owner_token(ArrayGlyph(arr))
        for _ in range(3):
            _render_owner_token(_SlotsOwner())
        assert _render_owner_token(ArrayGlyph(arr)) == before + 1, (
            "an untrackable owner advanced the token counter"
        )


class TestApplyAxisStyleRejectsABadGridAxis:
    """The helper validates `grid_axis` in its own vocabulary."""

    @pytest.mark.parametrize("bad", ["nope", "xy", "BOTH"])
    def test_an_unknown_grid_axis_raises(self, bad):
        """An unknown `grid_axis` names the glyph's parameter, not matplotlib's.

        Args:
            bad: The rejected value.

        Test scenario:
            The value went straight to `ax.grid(axis=...)`, so the caller got a
            matplotlib error about a parameter they had not passed.
        """
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="grid_axis must be one of"):
            _apply_axis_options(ax, {"grid_alpha": 0.5}, {"grid_alpha"}, grid_axis=bad)
        plt.close(fig)

    @pytest.mark.parametrize("good", ["both", "x", "y", None])
    def test_every_documented_grid_axis_is_accepted(self, good):
        """The four documented values still work.

        Args:
            good: The accepted value.

        Test scenario:
            The guard must not narrow the helper's own contract.
        """
        fig, ax = plt.subplots()
        _apply_axis_options(ax, {"grid_alpha": 0.5}, {"grid_alpha"}, grid_axis=good)
        plt.close(fig)


class TestAnimateHonoursTheTickOptions:
    """`animate` applies the tick sizes it accepts, as `plot` does."""

    def test_an_animation_with_an_extent_keeps_its_ticks(self, frames):
        """Real coordinates are not blanked just because this is an animation.

        Args:
            frames: The frame-stack fixture.

        Test scenario:
            `animate` cleared the tick labels and locators unconditionally,
            unlike `plot`, which only does it for a pixel-space render. So an
            animation given an `extent` lost the axis it had coordinates for.
        """
        glyph = ArrayGlyph(frames, extent=[0, 0, 10, 10], add_colorbar=False)
        glyph.animate(list(range(3)))
        assert glyph.ax.get_xticks().size, "an extent-bearing animation lost its ticks"

    def test_xtick_font_size_reaches_an_animation(self, frames):
        """`animate(xtick_font_size=...)` is no longer accepted and dropped.

        Args:
            frames: The frame-stack fixture.

        Test scenario:
            The styling ran and the labels were deleted immediately after, so
            the option had no visible effect -- the same class of defect this
            branch exists to fix.
        """
        glyph = ArrayGlyph(frames, extent=[0, 0, 10, 10], add_colorbar=False)
        glyph.animate(list(range(3)), xtick_font_size=20)
        assert glyph.ax.get_xticklabels()[0].get_fontsize() == 20.0, (
            "animate dropped xtick_font_size"
        )

    def test_a_pixel_space_animation_still_hides_its_indices(self, frames):
        """Without an `extent` the indices are still hidden.

        Args:
            frames: The frame-stack fixture.

        Test scenario:
            Row/column numbers are meaningless labels; narrowing the rule must
            not start showing them.
        """
        glyph = ArrayGlyph(frames, add_colorbar=False)
        glyph.animate(list(range(3)))
        assert not glyph.ax.get_xticks().size, (
            "a pixel-space animation showed its row/column indices"
        )


class TestAComposedOverlayDoesNotAddAColorbar:
    """A colorbar takes its space from the host axes, so composing defaults it off."""

    def test_repeated_overlays_leave_the_host_geometry_alone(self, arr, host):
        """Five overlays neither shrink the host nor add five colorbars.

        Args:
            arr: The array fixture.
            host: The pre-rendered host axes.

        Test scenario:
            `add_colorbar` defaults to `True` and `fig.colorbar()` steals space
            from the axes it is attached to, so every composed overlay
            re-laid-out the host and left another colorbar axes behind.
        """
        figure = host.get_figure()
        axes_before = len(figure.axes)
        bounds_before = host.get_position().bounds
        for _ in range(5):
            ArrayGlyph(arr, extent=[0, 0, 10, 10]).plot(ax=host, compose=True)
        assert len(figure.axes) == axes_before, (
            f"composed overlays added colorbar axes: {len(figure.axes)}"
        )
        assert host.get_position().bounds == bounds_before, (
            f"composed overlays re-laid-out the host: {host.get_position().bounds}"
        )

    @pytest.mark.parametrize("ask", ["constructor", "call"])
    def test_an_explicitly_requested_colorbar_is_still_drawn(self, arr, host, ask):
        """Composing defaults the colorbar off, it does not forbid it.

        Args:
            arr: The array fixture.
            host: The pre-rendered host axes.
            ask: Whether the colorbar is asked for at construction or on the call.

        Test scenario:
            A caller who wants a second colorbar on the host must still get one.
        """
        figure = host.get_figure()
        axes_before = len(figure.axes)
        if ask == "constructor":
            ArrayGlyph(arr, extent=[0, 0, 10, 10], add_colorbar=True).plot(
                ax=host, compose=True
            )
        else:
            ArrayGlyph(arr, extent=[0, 0, 10, 10]).plot(
                ax=host, compose=True, colorbar=True
            )
        assert len(figure.axes) == axes_before + 1, (
            f"an explicitly requested colorbar was suppressed: {len(figure.axes)}"
        )

    def test_a_composed_animation_adds_no_colorbar(self, frames, host):
        """The same default applies to `animate`.

        Args:
            frames: The frame-stack fixture.
            host: The pre-rendered host axes.

        Test scenario:
            `animate` reads the same option and draws on the same host.
        """
        figure = host.get_figure()
        axes_before = len(figure.axes)
        ArrayGlyph(frames, ax=host, extent=[0, 0, 10, 10]).animate(
            list(range(3)), compose=True
        )
        assert len(figure.axes) == axes_before, (
            f"a composed animation added a colorbar: {len(figure.axes)}"
        )

    def test_a_solo_render_still_draws_its_colorbar(self, arr):
        """Without `compose` the default is unchanged.

        Args:
            arr: The array fixture.

        Test scenario:
            The narrowing must apply only to the composing case.
        """
        fig, _ = ArrayGlyph(arr, extent=[0, 0, 10, 10]).plot()
        assert len(fig.axes) == 2, f"a solo render lost its colorbar: {len(fig.axes)}"
        plt.close(fig)

    def test_a_composed_vector_overlay_adds_no_colorbar(self, field, host):
        """Arrows drawn over a raster bring no colorbar of their own.

        Args:
            field: The vector-field fixture.
            host: The pre-rendered host axes.

        Test scenario:
            The scalar-field-plus-wind-arrows figure is the whole point of
            composing, and `VectorGlyph` defaults `add_colorbar` on -- so the
            overlay added a second colorbar and stole the host's width to make
            room for it. Counted rather than measured: an overlay legitimately
            moves the host's box by widening the data limits, so only the extra
            axes is proof of a colorbar.
        """
        x, y, u, v = field
        figure = host.get_figure()
        axes_before = len(figure.axes)
        VectorGlyph(x, y, u, v).plot(kind="quiver", ax=host, compose=True)
        assert len(figure.axes) == axes_before, (
            f"a composed vector overlay added a colorbar: {len(figure.axes)}"
        )

    @pytest.mark.parametrize("ask", ["constructor", "call", "add_colorbar"])
    def test_an_explicitly_requested_vector_colorbar_is_still_drawn(
        self, field, host, ask
    ):
        """A vector overlay that asks for a colorbar still gets one.

        Args:
            field: The vector-field fixture.
            host: The pre-rendered host axes.
            ask: Which of the three ways of asking is used.

        Test scenario:
            `VectorGlyph.plot` takes `add_colorbar=` as a parameter of its own
            as well as reading the option, so all three routes have to survive
            the composed default.
        """
        x, y, u, v = field
        figure = host.get_figure()
        axes_before = len(figure.axes)
        if ask == "constructor":
            VectorGlyph(x, y, u, v, add_colorbar=True).plot(
                kind="quiver", ax=host, compose=True
            )
        elif ask == "call":
            VectorGlyph(x, y, u, v).plot(
                kind="quiver", ax=host, compose=True, colorbar=True
            )
        else:
            VectorGlyph(x, y, u, v).plot(
                kind="quiver", ax=host, compose=True, add_colorbar=True
            )
        assert len(figure.axes) == axes_before + 1, (
            f"an explicitly requested vector colorbar was suppressed: "
            f"{len(figure.axes)}"
        )

    def test_a_solo_vector_render_still_draws_its_colorbar(self, field):
        """Without `compose` a vector render keeps its colorbar.

        Args:
            field: The vector-field fixture.

        Test scenario:
            The narrowing is for the composing case only, here as on the array
            path.
        """
        x, y, u, v = field
        fig, ax = plt.subplots()
        VectorGlyph(x, y, u, v).plot(kind="quiver", ax=ax)
        assert len(fig.axes) == 2, (
            f"a solo vector render lost its colorbar: {len(fig.axes)}"
        )
        plt.close(fig)


class TestMeshCarriesItsConstructionAxisStyle:
    """`MeshGlyph` resets `default_options` per render without losing the ctor's."""

    def test_a_call_option_overrides_the_construction_one(self, mesh):
        """The key this call passes wins over the one the constructor set.

        Args:
            mesh: The mesh fixture.

        Test scenario:
            The restore runs after the per-call merge, so it has to skip every
            key the call supplied. Writing the construction value back over it
            would make `plot(xlabel=...)` inert on exactly the glyph the restore
            was added for.
        """
        node_x, node_y, faces, data = mesh
        glyph = MeshGlyph(node_x, node_y, faces, xlabel="CTOR", ylabel="CTORY")
        _, ax = glyph.plot(data, colorbar=False, xlabel="CALL")
        assert ax.get_xlabel() == "CALL", (
            f"the construction value overwrote the call's: {ax.get_xlabel()!r}"
        )
        assert ax.get_ylabel() == "CTORY", (
            f"a construction option the call did not pass was dropped: "
            f"{ax.get_ylabel()!r}"
        )
        plt.close("all")

    def test_a_call_option_does_not_leak_into_the_next_render(self, mesh):
        """A later render without the key falls back to the construction value.

        Args:
            mesh: The mesh fixture.

        Test scenario:
            Rebuilding `default_options` per call is what stops a per-call
            option becoming sticky; restoring the construction options must not
            restore the previous call's alongside them.
        """
        node_x, node_y, faces, data = mesh
        glyph = MeshGlyph(node_x, node_y, faces, xlabel="CTOR")
        glyph.plot(data, colorbar=False, xlabel="CALL")
        _, ax = glyph.plot(data, colorbar=False)
        assert ax.get_xlabel() == "CTOR", (
            f"the previous call's xlabel became sticky: {ax.get_xlabel()!r}"
        )
        plt.close("all")

    def test_animate_keeps_the_construction_axis_style(self, mesh):
        """`animate` resets `default_options` too, and restores the same keys.

        Args:
            mesh: The mesh fixture.

        Test scenario:
            Both render entry points rebuild the options from the module
            defaults, so `MeshGlyph(xlabel=...).animate(...)` lost the label the
            same way `plot` did.
        """
        node_x, node_y, faces, _ = mesh
        glyph = MeshGlyph(node_x, node_y, faces, xlabel="CTOR", ylabel="CTORY")
        glyph.animate(np.array([[1.0, 2.0], [2.0, 3.0]]), ["t0", "t1"], colorbar=False)
        assert glyph.ax.get_xlabel() == "CTOR", (
            f"animate dropped the construction xlabel: {glyph.ax.get_xlabel()!r}"
        )
        assert glyph.ax.get_ylabel() == "CTORY", (
            f"animate dropped the construction ylabel: {glyph.ax.get_ylabel()!r}"
        )
        plt.close("all")


class TestApplyStyleRefusesToCompose:
    """`apply_style` owns the axes outright, so it cannot honour `compose`."""

    def test_compose_is_rejected_with_a_pointer_to_plot(self, arr):
        """`apply_style(..., compose=True)` raises instead of being swallowed.

        Args:
            arr: The array fixture.

        Test scenario:
            `apply_style` clears the axes before forwarding to `plot`, so the
            flag was accepted, silently defeated, and the host wiped anyway.
        """
        glyph = ArrayGlyph(arr, extent=[0, 0, 10, 10])
        with pytest.raises(ValueError, match="compose=True cannot be honoured"):
            glyph.apply_style("elevation", compose=True)
        plt.close("all")

    def test_apply_style_without_compose_is_unaffected(self, arr):
        """The guard does not disturb the ordinary call.

        Args:
            arr: The array fixture.

        Test scenario:
            Only a truthy `compose` is refused; everything else still forwards.
        """
        glyph = ArrayGlyph(arr, extent=[0, 0, 10, 10])
        _, ax = glyph.apply_style("elevation")
        assert glyph.style == "elevation", f"the style did not apply: {glyph.style}"
        assert ax.images, "apply_style rendered nothing"
        plt.close("all")
