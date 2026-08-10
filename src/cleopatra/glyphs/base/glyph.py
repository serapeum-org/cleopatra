"""Base visualization class for cleopatra glyphs.

Provides shared infrastructure for array-based and mesh-based
visualization: figure/axes lifecycle, color scale normalization,
colorbar creation, tick management, point overlays, and animation.
"""

from __future__ import annotations

import inspect
import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure, SubFigure
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle

from cleopatra.glyphs.base.animation import SUPPORTED_VIDEO_FORMAT  # noqa: F401  (re-export)
from cleopatra.glyphs.base.animation import save_animation as _save_animation
from cleopatra.styling.colors import resolve_colormap
from cleopatra.styling.scaling import MAX_DISCRETE_LEVELS  # noqa: F401  (re-export)
from cleopatra.styling.scaling import ColorScaling, levels_to_bounds
from cleopatra.styling.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS
from cleopatra.styling.styles import (
    categorize,
    classify,
    disjoint_legend,
)

#: Qualitative colormap `_prepare_categorical_mapping` falls back to when the
#: caller left `cmap` at the shared continuous/diverging default -- see the
#: fallback logic there.
CATEGORICAL_DEFAULT_CMAP = "tab10"

#: Loose-keyword option keys that moved onto grouped parameter objects.
#: Passing any of these as a flat keyword (to a constructor or to
#: `plot`/`animate`) now raises, pointing at the object to use instead.
#: The keys still live in `default_options` -- the rendering engine reads
#: them -- but they are populated only via a group object's `to_options()`.
#: Extended as each group lands (color, then contour/cells/classify/style).
_GROUPED_KWARG_HINTS: dict[str, str] = {
    "color_scale": "color=ColorScaling.<variant>(...), e.g. ColorScaling.power(gamma=0.7)",
    "gamma": "color=ColorScaling.power(gamma=...)",
    "line_threshold": "color=ColorScaling.sym_log(threshold=..., scale=...)",
    "line_scale": "color=ColorScaling.sym_log(threshold=..., scale=...)",
    "bounds": "color=ColorScaling.boundary(bounds=...)",
    "midpoint": "color=ColorScaling.midpoint(at=...)",
    "levels": "contour=Contour(levels=...)",
    "labels": "contour=Contour(labels=True, label_kw=...)",
    "label_kw": "contour=Contour(labels=True, label_kw=...)",
    "display_cell_value": "cells=CellValues(show=True, ...)",
    "num_size": "cells=CellValues(size=...)",
    "background_color_threshold": "cells=CellValues(background_threshold=...)",
    "scheme": "classify=Classify(scheme=..., k=...)",
    "k": "classify=Classify(scheme=..., k=...)",
    "category_legend_kwargs": "classify=Classify(scheme='categorical', category_legend_kwargs=...)",
    "style": "data_style=DataStyle(style=...)",
    "hillshade": "data_style=DataStyle(hillshade=...)",
    "bands": "data_style=DataStyle(bands=...)",
    "alpha_range": "data_style=DataStyle(alpha_range=...)",
    # `alpha` also moved onto `data_style=DataStyle(alpha=...)`, but only for
    # `ArrayGlyph` -- it stays a legitimate loose opacity option on other
    # glyphs (`LineGlyph`, `HistogramGlyph`), so it cannot be rejected here
    # (this hint map gates every glyph's construction). `ArrayGlyph` rejects a
    # loose `alpha=` locally instead (see its `_reject_loose_alpha`).
}


def _reject_grouped_kwargs(keys: Any) -> None:
    """Raise if any key now belongs to a grouped parameter object.

    Args:
        keys: An iterable of keyword-argument names (e.g. `kwargs`).

    Raises:
        ValueError: On the first key found in `_GROUPED_KWARG_HINTS`, with a
            message naming the grouped object to pass instead.
    """
    for key in keys:
        hint = _GROUPED_KWARG_HINTS.get(key)
        if hint is not None:
            raise ValueError(
                f"The {key!r} option moved onto a grouped parameter object; "
                f"pass {hint} instead of a loose {key}= keyword."
            )


def _get_figure_supports_root(get_figure) -> bool:
    """Return True if `get_figure` accepts a `root` keyword argument.

    The `root` parameter was added to `Axes.get_figure` in matplotlib 3.10.
    Detected by signature inspection (rather than a broad `try/except`) so an
    unrelated `TypeError` from `get_figure` itself is never swallowed.
    """
    try:
        supports = "root" in inspect.signature(get_figure).parameters
    except (TypeError, ValueError):
        supports = False
    return supports


def _root_figure(ax: Axes) -> Figure:
    """Return the top-level `Figure` that owns `ax`, across matplotlib versions.

    On matplotlib >= 3.10 this uses `Axes.get_figure(root=True)`, which returns
    the root `Figure` even when the axes lives on a `SubFigure` (and avoids the
    3.10 deprecation warning attached to the bare `get_figure()`). On older
    matplotlib (down to the project's 3.8.4 floor) the `root` keyword does not
    exist, so it climbs out of any `SubFigure` to the owning `Figure` manually.

    Args:
        ax: The axes whose top-level figure is wanted.

    Returns:
        Figure: The top-level figure for `ax`.
    """
    get_figure = ax.get_figure
    if _get_figure_supports_root(get_figure):
        root_fig = get_figure(root=True)
        # `ax` is a live, attached axes, so its root figure always resolves.
        assert root_fig is not None
        result: Figure = root_fig
    else:
        fig: Figure | SubFigure | None = get_figure()
        seen: set[int] = set()
        while isinstance(fig, SubFigure) and id(fig) not in seen:
            seen.add(id(fig))
            fig = fig.figure
        assert fig is not None
        result = cast(Figure, fig)
    return result


def _figure_is_open(fig: Figure | None) -> bool:
    """True if `fig` is a live top-level `Figure` still registered with pyplot.

    Pass a root `Figure` (resolve a `SubFigure` via `_root_figure` first); a
    figure whose window/number has been closed, or a `SubFigure` (no `.number`),
    returns `False`.
    """
    num = getattr(fig, "number", None)
    return num is not None and plt.fignum_exists(num)


def _immediate_figure(ax: Axes) -> Figure | SubFigure:
    """Return the figure `ax` is directly attached to (its immediate parent).

    Deprecation-safe counterpart to `_root_figure`: on matplotlib >= 3.10 it
    passes `root=False` explicitly; on older matplotlib it calls the bare
    `get_figure()`. For an axes on a `SubFigure` this is that `SubFigure`; for
    an ordinary axes it is the same object as `_root_figure(ax)`.

    Args:
        ax: The axes whose immediate parent figure is wanted.

    Returns:
        Figure: The figure (or sub-figure) `ax` is directly attached to.
    """
    get_figure = ax.get_figure
    if _get_figure_supports_root(get_figure):
        fig = get_figure(root=False)
    else:
        fig = get_figure()
    # `ax` is a live, attached axes, so its immediate figure always resolves.
    assert fig is not None
    return fig


def _clear_prior_render_artists(ax: Axes) -> None:
    """Remove a prior render call's tracked artists from `ax`.

    Every glyph's `plot`/`animate`-style method creates a fresh set of
    drawing artists (an image, a colorbar, a frame-label `Text`, a line
    collection, ...) on every call rather than reusing one -- matplotlib
    has no "replace the previous artist" primitive for `ax.imshow()`,
    `fig.colorbar()`, `ax.add_collection()`, etc.; each call always adds a
    new artist. Calling the method again on the same `Axes` -- from the
    same glyph instance, or a *different* one sharing it via
    `SomeGlyph(ax=..., fig=...)` (e.g. the `ax=`/`fig=` passthrough
    `pyramids`' `Dataset`/`NetCDF`/`Analysis`/`UgridDataset`.plot(ax=...,
    fig=...) expose) -- would otherwise leave the previous call's artists
    orphaned: still attached to the `Axes`/`Figure`, and driven by nothing
    once the owning glyph's own attributes move on to the new call's
    objects. Ownership is tracked on `ax` itself (not on any glyph
    instance) via a private marker, precisely so this catches both cases.
    Must be called before creating this call's own artists.

    Args:
        ax: The axes a render call is about to draw onto.
    """
    prior = getattr(ax, "_cleo_render_artists", None)
    if prior is None:
        return
    for artist in prior:
        try:
            artist.remove()
        except (KeyError, NotImplementedError, AttributeError):
            pass
    ax._cleo_render_artists = None  # type: ignore[attr-defined]


def _mark_render_artists(ax: Axes, *artists: Any) -> None:
    """Record this render call's artists on `ax` for the next call's cleanup.

    Args:
        ax: The axes this call rendered onto.
        *artists: The artists this call created, in the order they must be
            removed on the next call (e.g. a colorbar before the image it
            is attached to -- `Colorbar.remove()` reads
            `self.mappable.axes` to restore the image axes' gridspec
            position, which is only valid while the image artist is
            still attached). `None` entries (an artist this call didn't
            create, e.g. no colorbar when `add_colorbar=False`) are
            dropped.
    """
    ax._cleo_render_artists = [  # type: ignore[attr-defined]
        a for a in artists if a is not None
    ]


def _stash_projection_frame(ax: Axes, new_artists: Any) -> None:
    """Record a projection frame's artists (boundary + graticule) on `ax`.

    Tracked separately from `_cleo_render_artists` because a projection frame --
    and the frozen view / `axis("off")` state `apply_projection_frame` installs
    alongside it -- must persist across data re-renders and be undone only when a
    later render is *not* itself a projection render (see `_clear_projection_frame`
    / `_restore_flat_axes`). Stamped on the `Axes` itself so it survives across
    glyph instances that share the axes.

    Args:
        ax: The axes a projection frame was just drawn on.
        new_artists: The frame artists (boundary patch + graticule lines).
    """
    ax._cleo_projection_frame = list(new_artists)  # type: ignore[attr-defined]


def _clear_projection_frame(ax: Axes) -> bool:
    """Remove any projection frame recorded on `ax`; report whether one existed.

    Args:
        ax: The axes to strip a prior projection frame from.

    Returns:
        bool: `True` if a frame was present (so the caller knows to restore the
            flat view with `_restore_flat_axes`), else `False`.
    """
    frame = getattr(ax, "_cleo_projection_frame", None)
    ax._cleo_projection_frame = None  # type: ignore[attr-defined]
    existed = bool(frame)
    if frame:
        for artist in frame:
            try:
                artist.remove()
            except (KeyError, NotImplementedError, ValueError, AttributeError):
                pass
    return existed


def _restore_flat_axes(
    ax: Axes,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    aspect: str,
) -> None:
    """Undo the view state a projection frame installed, framing `ax` flat.

    `apply_projection_frame` freezes `xlim`/`ylim` to the orthographic radius
    (which also disables autoscaling) and calls `set_axis_off()`. A later flat
    render on the same axes would otherwise draw its (much smaller, degree-scale)
    data into that frozen view with the axis hidden -- an invisible speck. This
    re-enables the axis, restores the glyph's default `aspect`, and frames the
    axes over the data bounds (with a 5% margin) so the flat layer is visible.

    Args:
        ax: The axes to un-freeze.
        x_min: Minimum x data bound.
        x_max: Maximum x data bound.
        y_min: Minimum y data bound.
        y_max: Maximum y data bound.
        aspect: The glyph's flat-render aspect (`"equal"` for lon/lat meshes,
            `"auto"` for the array raster path).
    """
    ax.set_axis_on()
    ax.set_aspect(aspect)
    lo_x, hi_x = sorted((x_min, x_max))
    lo_y, hi_y = sorted((y_min, y_max))
    px = 0.05 * ((hi_x - lo_x) or 1.0)
    py = 0.05 * ((hi_y - lo_y) or 1.0)
    ax.set_xlim(lo_x - px, hi_x + px)
    ax.set_ylim(lo_y - py, hi_y + py)


class Glyph:
    """Base class for cleopatra visualization glyphs.

    Handles figure/axes management, default options, color scale
    normalization, colorbar creation, tick control, point overlays,
    and animation saving. Subclasses implement the actual rendering.

    The accepted option keys are exposed per subclass via the
    `DEFAULT_OPTIONS` class attribute, and can be inspected or filtered
    *before* constructing an instance with the `option_keys` and
    `filter_kwargs` classmethods (useful for safely forwarding a bag of
    user-supplied styling kwargs).

    Args:
        default_options: Default plot options dict. Subclasses provide
            their own defaults merged with `STYLE_DEFAULTS`.
        fig: Pre-existing matplotlib figure to bind. Default is None.
            An `ax` fully determines its figure, so `fig` is optional even
            when `ax` is given; when both are passed the explicit `fig`
            is kept as the figure handle. Passing a `fig` that does not own
            the given `ax` emits a `UserWarning` (the explicit `fig` is
            still honoured, but the two handles then disagree).
        ax: Pre-existing matplotlib axes to bind. Default is None. Passing
            `ax` on its own is supported — its parent figure is derived
            automatically (the axes is no longer dropped when `fig` is
            omitted).
        **kwargs: Override any key in `default_options`.

    Examples:
        - Create a Glyph and override the colormap:
            ```python
            >>> from cleopatra.glyphs.base.glyph import Glyph
            >>> from cleopatra.styling.styles import DEFAULT_OPTIONS
            >>> opts = DEFAULT_OPTIONS.copy()
            >>> opts["vmin"] = None
            >>> opts["vmax"] = None
            >>> g = Glyph(default_options=opts, cmap="plasma")
            >>> g.default_options["cmap"]
            'plasma'

            ```
        - Provide a pre-existing figure and axes:
            ```python
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.glyphs.base.glyph import Glyph
            >>> from cleopatra.styling.styles import DEFAULT_OPTIONS
            >>> opts = DEFAULT_OPTIONS.copy()
            >>> opts["vmin"] = None
            >>> opts["vmax"] = None
            >>> fig, ax = plt.subplots()
            >>> g = Glyph(default_options=opts, fig=fig, ax=ax)
            >>> g.fig is fig
            True
            >>> g.ax is ax
            True

            ```
        - Provide only an axes; the figure is derived from it:
            ```python
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.glyphs.base.glyph import Glyph
            >>> from cleopatra.styling.styles import DEFAULT_OPTIONS
            >>> opts = DEFAULT_OPTIONS.copy()
            >>> opts["vmin"] = None
            >>> opts["vmax"] = None
            >>> fig, ax = plt.subplots()
            >>> g = Glyph(default_options=opts, ax=ax)
            >>> g.ax is ax
            True
            >>> g.fig is ax.get_figure()
            True

            ```

    See Also:
        cleopatra.glyphs.gridded.array_glyph.ArrayGlyph: Glyph subclass for
            2D/3D arrays.
        cleopatra.glyphs.gridded.mesh_glyph.MeshGlyph: Glyph subclass for
            unstructured meshes.
    """

    #: The option keys this glyph accepts, as a class attribute so they can
    #: be introspected/filtered *before* an instance exists (see
    #: `option_keys`/`filter_kwargs`). Each subclass overrides this with its
    #: own option dict (built as `STYLE_DEFAULTS | <glyph-specific>`); the
    #: base value is the shared style defaults.
    DEFAULT_OPTIONS: dict = STYLE_DEFAULTS

    #: Whether this glyph's `plot()` reads back the categorical side-channel
    #: (`self._categorical`) instead of feeding raw `values` straight into
    #: the mappable. Only true for glyphs whose per-element value is a
    #: nominal class label rather than a continuous magnitude (e.g.
    #: `PolygonGlyph`, `ScatterGlyph`) — `scheme="categorical"` is rejected
    #: for any other glyph rather than silently mis-colouring it.
    _SUPPORTS_CATEGORICAL_SCHEME = False

    def __init__(
        self,
        default_options: dict,
        fig: Figure | None = None,
        ax: Axes | None = None,
        **kwargs,
    ):
        self._default_options = default_options.copy()
        self._merge_kwargs(kwargs)
        self._vmin: float | None = None
        self._vmax: float | None = None
        self.ticks_spacing: float | None = None
        #: Set by `_prepare_categorical_mapping` when `scheme="categorical"`
        #: — `{"codes", "cmap", "colors", "labels"}` — else `None`.
        self._categorical: dict | None = None
        #: Set by a subclass's `animate()`; exposed read-only via `anim`.
        self._anim: FuncAnimation | None = None
        if ax is not None:
            self.ax: Axes | None = ax
            if fig is not None:
                if fig is not _immediate_figure(ax) and fig is not _root_figure(ax):
                    warnings.warn(
                        "The given `fig` is not the figure that owns `ax`; "
                        "the axes' own figure is what will be drawn on. Pass "
                        "only `ax` (its figure is derived automatically).",
                        stacklevel=2,
                    )
                self.fig: Figure | None = fig
            else:
                self.fig = _root_figure(ax)
        elif fig is not None:
            self.fig = fig
            self.ax = None
        else:
            self.fig = None
            self.ax = None

    @property
    def vmin(self) -> float | None:
        """Minimum value for color scaling."""
        return self._vmin

    @property
    def vmax(self) -> float | None:
        """Maximum value for color scaling."""
        return self._vmax

    @property
    def default_options(self) -> dict:
        """Default plot options."""
        return self._default_options

    @classmethod
    def option_keys(cls) -> set[str]:
        """Return the keyword-argument keys this glyph accepts.

        Resolves from the class-level `DEFAULT_OPTIONS`, so the accepted
        keys can be inspected **without constructing an instance** (and
        therefore without tripping the strict unknown-kwarg check in
        `_merge_kwargs`). The keys differ per glyph subclass.

        This reports the class's *default* option set. For every concrete
        glyph subclass that equals the instance's accepted keys (each
        subclass passes the same dict to `__init__`). The base `Glyph`
        reports the shared `STYLE_DEFAULTS`; an instance built with a
        custom injected `default_options` is the one case where the two
        can differ, so base `Glyph` is not part of the introspection
        contract.

        Returns:
            set[str]: The accepted option keys for this glyph class.

        Examples:
            - Inspect the keys a glyph accepts before building one:
                ```python
                >>> from cleopatra.glyphs.primitives.scatter_glyph import ScatterGlyph
                >>> keys = ScatterGlyph.option_keys()
                >>> "cmap" in keys
                True
                >>> "totally_unknown" in keys
                False

                ```
            - Different glyphs expose different keys:
                ```python
                >>> from cleopatra.glyphs.primitives.polygon_glyph import PolygonGlyph
                >>> "edgecolor" in PolygonGlyph.option_keys()
                True

                ```

        See Also:
            filter_kwargs: Drop the keys a glyph does not accept from a dict.
        """
        return set(cls.DEFAULT_OPTIONS)

    @classmethod
    def filter_kwargs(cls, kwargs: dict) -> dict:
        """Return only the subset of `kwargs` whose keys this glyph accepts.

        A convenience for callers that forward a bag of user-supplied
        styling kwargs into a glyph: pre-filtering with this method lets
        the construction succeed instead of raising on an unknown key.
        Order and values are preserved; rejected keys are simply dropped.

        Args:
            kwargs: A mapping of candidate option keys to values.

        Returns:
            dict: The entries of `kwargs` whose keys are in `option_keys()`.

        Examples:
            - Keep only the accepted keys, then construct safely:
                ```python
                >>> from cleopatra.glyphs.primitives.polygon_glyph import PolygonGlyph
                >>> raw = {"cmap": "viridis", "edgecolor": "black", "bogus": 1}
                >>> safe = PolygonGlyph.filter_kwargs(raw)
                >>> sorted(safe)
                ['cmap', 'edgecolor']
                >>> safe["cmap"]
                'viridis'

                ```
            - An empty mapping yields an empty mapping:
                ```python
                >>> from cleopatra.glyphs.primitives.scatter_glyph import ScatterGlyph
                >>> ScatterGlyph.filter_kwargs({})
                {}

                ```

        See Also:
            option_keys: The set of keys this glyph accepts.
        """
        keys = cls.option_keys()
        return {key: val for key, val in kwargs.items() if key in keys}

    @property
    def anim(self) -> FuncAnimation:
        """Animation object created by `animate()`."""
        if self._anim is not None:
            return self._anim
        raise ValueError(
            "Please first use the animate method to create the animation object"
        )

    def _merge_kwargs(self, kwargs: dict) -> None:
        """Validate and merge keyword arguments into default_options."""
        #: Option keys the caller passed explicitly, so a subclass can tell an
        #: overridden option from one left at its default (e.g. `ArrayGlyph` only
        #: auto-sizes the figure when `figsize` was not passed).
        self._explicit_options: set[str] = set(kwargs)
        _reject_grouped_kwargs(kwargs)
        for key, val in kwargs.items():
            if key not in self._default_options:
                raise ValueError(
                    f"The given keyword argument:{key} is not correct, "
                    f"possible parameters are, {list(self._default_options.keys())}"
                )
            else:
                self._default_options[key] = val

    def _merge_group_params(self, *groups: Any) -> None:
        """Flatten grouped parameter objects into `default_options`.

        Each glyph's `plot`/`animate` accepts grouped parameter objects
        (e.g. `color=ColorScaling(...)`) in place of the loose keyword
        arguments they replaced. Every such object exposes `to_options()`,
        returning the flat `default_options` keys the rendering engine
        reads; this helper merges each non-`None` object's keys in, so the
        internal storage stays a single flat dict.

        Only keys the glyph actually supports (already present in its
        `default_options`) are applied, so a single group object can be
        passed to glyphs that support different subsets of it -- e.g. a
        `Contour` carrying `levels` + `labels` applies both on `ArrayGlyph`
        (which draws isoline labels) but only `levels` on `ScatterGlyph`
        (which has no labels). A group's `to_options()` emits only the
        fields the caller explicitly set, so unset fields never clobber a
        glyph's own defaults.

        Args:
            *groups: Grouped parameter objects (or `None` for an omitted
                group). Anything `None` is skipped; each other object must
                expose a `to_options()` returning a dict.
        """
        for group in groups:
            if group is None:
                continue
            for key, val in group.to_options().items():
                if key in self.default_options:
                    self.default_options[key] = val

    @contextmanager
    def _rollback_options_on_error(self) -> Iterator[None]:
        """Restore `default_options` if the wrapped render body raises.

        A glyph's `plot` merges grouped parameter objects (`color=`, `contour=`,
        `classify=`, ...) into the persistent `default_options` at the top of the
        call, then renders. Most glyphs keep those options across plots (they are
        sticky by design), so a render that raises *after* the merge -- an
        unsupported `scheme`, a degenerate colour scale -- would leave the
        half-applied options behind and poison later plain `plot()` calls on the
        same instance: the stale option re-triggers the same error, or silently
        renders with a colour scale that was never successfully applied.

        Wrap the render body (the merge included) in this context manager: it
        snapshots `default_options` on entry and, if the body raises, restores it
        exactly, so a failed styled render leaves the glyph's option state
        untouched. On success the merged options stay. Unlike a wrapping
        decorator, a `with` block adds no stack frame, so warnings emitted inside
        the render keep their caller-attributed `stacklevel`.

        Yields:
            None: control returns to the `with` body with the snapshot taken.
        """
        snapshot = dict(self._default_options)
        try:
            yield
        except BaseException:
            self._default_options.clear()
            self._default_options.update(snapshot)
            raise

    def create_figure_axes(self) -> tuple[Figure, Axes]:
        """Create a new figure and axes from default_options.

        Uses the `figsize` key from `default_options` to set the
        figure dimensions.

        Returns:
            tuple[Figure, Axes]: The created figure and axes.

        Examples:
            - Create a figure with custom size:
                ```python
                >>> from cleopatra.glyphs.base.glyph import Glyph
                >>> from cleopatra.styling.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None})
                >>> g = Glyph(default_options=opts, figsize=(12, 4))
                >>> fig, ax = g.create_figure_axes()
                >>> fig.get_size_inches()
                array([12.,  4.])

                ```
        """
        fig, ax = plt.subplots(figsize=self.default_options["figsize"])
        return fig, ax

    def _reset_axes_for_restyle(self) -> None:
        """Prepare `self.ax` for an in-place restyle (used by `apply_style`).

        When the glyph has a **live** axes (already plotted and its figure is
        still open), the previous render is cleared from it -- the glyph's
        colorbar, any legend / swatch inset axes, and all artists -- so the
        restyle replaces the content in place. `apply_style` therefore takes
        full ownership of this axes and must not be used on an axes shared with
        unrelated caller content. When the glyph was never plotted, its figure
        was closed, or it was built with a figure but no axes, a fresh axes is
        created instead (on the existing figure when one is still open).
        """
        ax = self.ax
        fig = self.fig
        root = _root_figure(ax) if ax is not None else fig
        ax_live = ax is not None and _figure_is_open(root)
        if ax_live:
            assert ax is not None
            for attr in ("cbar", "_cbar"):
                cbar = getattr(self, attr, None)
                if cbar is not None:
                    cbar.remove()
                    setattr(self, attr, None)
            for inset in list(ax.child_axes):
                inset.remove()
            ax.clear()
        elif fig is not None and _figure_is_open(fig):
            self.ax = fig.axes[0] if fig.axes else fig.add_subplot(111)
        else:
            self.fig, self.ax = self.create_figure_axes()

    def get_ticks(self) -> np.ndarray:
        """Compute colorbar tick locations from default_options.

        Uses `vmin`, `vmax`, and `ticks_spacing` from
        `default_options` to generate evenly-spaced tick positions.

        Returns:
            np.ndarray: Array of tick positions.

        Examples:
            - Compute ticks for a 0-10 range with spacing of 2:
                ```python
                >>> from cleopatra.glyphs.base.glyph import Glyph
                >>> from cleopatra.styling.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": 0.0, "vmax": 10.0, "ticks_spacing": 2.0})
                >>> g = Glyph(default_options=opts)
                >>> g.get_ticks()
                array([ 0.,  2.,  4.,  6.,  8., 10.])

                ```
        """
        ticks_spacing = self.default_options["ticks_spacing"]
        vmax = self.default_options["vmax"]
        vmin = self.default_options["vmin"]
        if not ticks_spacing or vmax <= vmin:
            result = np.array([vmin])
        else:
            ticks = np.arange(vmin, vmax + ticks_spacing, ticks_spacing)
            ticks = ticks[ticks <= vmax + 1e-9]
            if ticks.size == 0:
                result = np.array([vmin, vmax])
            else:
                if (vmax - ticks[-1]) > 0.04 * (vmax - vmin):
                    ticks = np.append(ticks, vmax)
                else:
                    ticks[-1] = vmax
                result = ticks
        return result

    def _create_norm_and_cbar_kw(
        self, ticks: np.ndarray
    ) -> tuple[colors.Normalize | None, dict]:
        """Create a matplotlib Normalize and colorbar kwargs.

        Honours the `color_scale` option — a `cleopatra.styling.styles.ColorScale`
        member or its string value (case-insensitive): `linear` / `power` /
        `sym-lognorm` / `boundary-norm` / `midpoint` — and the
        xarray-aligned `levels` and `extend` options when present in
        `default_options`. An unrecognised `color_scale` (including a
        non-string such as an int) raises `ValueError`.

        Behaviour for `levels`:

        * `levels` is `None` (default) — continuous norm based on
          `color_scale`.
        * `levels` is an `int` and `color_scale` is the default
          `"linear"` — switch to a `BoundaryNorm` with `levels`
          linearly-spaced edges between `vmin` and `vmax`.
        * `levels` is a sequence and `color_scale` is `"linear"` —
          use the sequence as explicit bin edges in a `BoundaryNorm`.
        * `levels` is set and `color_scale` is `"boundary-norm"`
          with no explicit `bounds` — treat `levels` as the bounds.
        * Otherwise (`color_scale` is some other enum value) — the
          user's choice wins; `levels` is left for the caller to
          forward to `contour` / `contourf`.

        Behaviour for `extend`: when present and non-None, the value
        is forwarded to the colorbar via `cbar_kw["extend"]`. The
        auto-resolution (`"both"` when `levels` is set, else
        `"neither"`) happens here only when `extend` is `None`.

        Args:
            ticks: Tick positions for the colorbar.

        Returns:
            tuple[Normalize or None, dict]: The norm (None for linear)
                and colorbar keyword arguments.

        Raises:
            ValueError: If `default_options["color_scale"]` is not a
                recognised `cleopatra.styling.styles.ColorScale` value.

        Examples:
            - Linear colour scale with no levels gives `norm=None`
                and ticks forwarded straight through:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.base.glyph import Glyph
                >>> from cleopatra.styling.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": 0.0, "vmax": 10.0})
                >>> g = Glyph(default_options=opts)
                >>> norm, cbar_kw = g._create_norm_and_cbar_kw(np.array([0.0, 5.0, 10.0]))
                >>> norm is None
                True
                >>> cbar_kw["extend"]
                'neither'
                >>> [float(t) for t in cbar_kw["ticks"]]
                [0.0, 5.0, 10.0]

                ```
            - With `levels` set and the default linear scale, a
                `BoundaryNorm` is built and `extend` defaults to
                `"both"`:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.base.glyph import Glyph
                >>> from cleopatra.styling.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": 0.0, "vmax": 10.0, "levels": 5})
                >>> g = Glyph(default_options=opts)
                >>> norm, cbar_kw = g._create_norm_and_cbar_kw(np.array([0.0, 5.0, 10.0]))
                >>> norm is None
                False
                >>> cbar_kw["extend"]
                'both'
                >>> [float(t) for t in cbar_kw["ticks"]]
                [0.0, 2.5, 5.0, 7.5, 10.0]

                ```
        """
        # The colour-scale logic lives on `ColorScaling` (see
        # `cleopatra.styling.scaling`); this method is the thin bridge from
        # the flat `default_options` storage to that object. `levels` and
        # `extend` are cross-group inputs (contour discretisation / colorbar
        # arrow extension), passed in rather than owned by the scale.
        scaling = ColorScaling.from_options(self.default_options)
        return scaling.build_norm(
            ticks,
            levels=self.default_options.get("levels"),
            extend=self.default_options.get("extend"),
        )

    @staticmethod
    def _levels_to_bounds(
        levels: int | list[float] | np.ndarray | None,
        vmin: float,
        vmax: float,
    ) -> np.ndarray | None:
        """Convert the `levels` option to an array of bin edges.

        Returns `None` when no levels are configured, signalling that
        the caller should fall back to the continuous norm path.

        Args:
            levels: Number of levels (`int`), explicit edges
                (`list` / `ndarray`), or `None` for no
                discretisation.
            vmin: Lower colour limit. Used when `levels` is an int to
                build the linspace.
            vmax: Upper colour limit. Used when `levels` is an int to
                build the linspace.

        Returns:
            np.ndarray or None: Sorted ascending array of bin edges, or
                `None` when `levels` is `None`.

        Raises:
            ValueError: If `levels` is an integer outside the range
                `[2, MAX_DISCRETE_LEVELS]` (a single edge cannot form a
                `BoundaryNorm`, and an enormous count would OOM
                `np.linspace`).

        Examples:
            - Integer `levels` becomes a `linspace` between
                `vmin` and `vmax`:
                ```python
                >>> from cleopatra.glyphs.base.glyph import Glyph
                >>> bounds = Glyph._levels_to_bounds(5, 0.0, 10.0)
                >>> [float(b) for b in bounds]
                [0.0, 2.5, 5.0, 7.5, 10.0]

                ```
            - A sequence is sorted ascending and returned as a float
                `ndarray`; `None` short-circuits to `None`:
                ```python
                >>> from cleopatra.glyphs.base.glyph import Glyph
                >>> bounds = Glyph._levels_to_bounds([10.0, 0.0, 5.0], 0.0, 10.0)
                >>> [float(b) for b in bounds]
                [0.0, 5.0, 10.0]
                >>> Glyph._levels_to_bounds(None, 0.0, 10.0) is None
                True

                ```
        """
        # Behaviour lives on `cleopatra.styling.scaling.levels_to_bounds`;
        # kept here as a thin delegator for the existing callers/doctests.
        return levels_to_bounds(levels, vmin, vmax)

    def _resolve_limits(self, values: np.ndarray) -> tuple[float, float]:
        """Resolve `(vmin, vmax)` from options, falling back to the data range.

        Reads `vmin` / `vmax` from `default_options`; whichever is `None`
        (or absent) is filled from the nan-aware min/max of `values`. This
        mirrors the simple branch of `ArrayGlyph._resolve_color_limits`
        (the `robust` / `center` / `percentile` machinery stays an
        `ArrayGlyph` concern). All-NaN input is detected and rejected here
        rather than surfacing later as an opaque failure inside
        `get_ticks()` or matplotlib.

        Args:
            values: The scalar array that will be colour-mapped. Used to
                supply data-driven limits when `vmin` / `vmax` are unset.

        Returns:
            tuple[float, float]: The resolved `(vmin, vmax)` as floats.

        Raises:
            ValueError: If a limit cannot be resolved to a finite number
                (e.g. `values` is empty or all-NaN and the corresponding
                limit was not pinned explicitly).

        Examples:
            - Auto-resolve both limits from the data:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.base.glyph import Glyph
                >>> from cleopatra.styling.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None})
                >>> g = Glyph(default_options=opts)
                >>> g._resolve_limits(np.array([1.0, 5.0, 9.0]))
                (1.0, 9.0)

                ```
            - An explicit limit is preserved; only the missing one is
                taken from the data:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.base.glyph import Glyph
                >>> from cleopatra.styling.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": 0.0, "vmax": None})
                >>> g = Glyph(default_options=opts)
                >>> g._resolve_limits(np.array([1.0, 5.0, 9.0]))
                (0.0, 9.0)

                ```
            - An all-NaN array with unpinned limits raises `ValueError`:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.base.glyph import Glyph
                >>> from cleopatra.styling.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None})
                >>> g = Glyph(default_options=opts)
                >>> g._resolve_limits(np.array([np.nan, np.nan]))
                Traceback (most recent call last):
                    ...
                ValueError: Cannot determine vmin/vmax: no finite values...

                ```
        """
        vmin = self.default_options.get("vmin")
        vmax = self.default_options.get("vmax")
        if vmin is None or vmax is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                data_min = np.nanmin(values)
                data_max = np.nanmax(values)
            vmin = data_min if vmin is None else vmin
            vmax = data_max if vmax is None else vmax
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            raise ValueError(
                "Cannot determine vmin/vmax: no finite values. Pass "
                "explicit vmin/vmax, or filter the array first."
            )
        return float(vmin), float(vmax)

    def _prepare_scalar_mapping(
        self, values: np.ndarray
    ) -> tuple[colors.Normalize | None, dict, np.ndarray]:
        """Build the `(norm, cbar_kw, ticks)` triple shared by coloured glyphs.

        This is the single home for the scalar-mapping contract that every
        colour-by-value glyph needs. It:

        1. resolves `(vmin, vmax)` from `default_options`, falling back to
           the data range via `_resolve_limits`;
        2. derives a sensible `ticks_spacing` of `(vmax - vmin) / 10` when
           the caller left it unset (`None`), guarding flat data so the
           spacing is never zero;
        3. writes `vmin`, `vmax`, and `ticks_spacing` back into
           `default_options` so the existing `get_ticks()` — which reads
           from `default_options` — can see them; and
        4. computes the ticks and forwards them to
           `_create_norm_and_cbar_kw`, honouring `levels` / `color_scale`.

        Subclasses call this instead of re-deriving the contract (which is
        easy to get subtly wrong: `get_ticks()` does not read `self._vmin`,
        and `np.arange(None, None)` raises).

        When the `scheme` option is set, the continuous steps above are
        bypassed: control is handed to `_prepare_classified_mapping`, which
        bins the data into discrete colour classes (a `BoundaryNorm`).
        `scheme="categorical"` bypasses them even earlier — before
        `_resolve_limits`, since a `vmin`/`vmax` range is meaningless for
        nominal values (and would raise for non-numeric ones) — and hands
        off to `_prepare_categorical_mapping` instead. With `scheme` unset
        (the default) the behaviour is unchanged.

        Args:
            values: The scalar array to be colour-mapped (e.g. point
                values, vector magnitudes, per-polygon values).

        Returns:
            tuple[Normalize or None, dict, np.ndarray]: the matplotlib norm
                (`None` for a plain linear scale), the colorbar keyword
                arguments from `_create_norm_and_cbar_kw`, and the computed
                tick positions.

        Raises:
            ValueError: Propagated from `_resolve_limits` when no finite
                limits can be determined.

        Examples:
            - Auto limits resolve from the data and produce a non-`None`
                `ticks_spacing` plus continuous-scale ticks:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.base.glyph import Glyph
                >>> from cleopatra.styling.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None, "ticks_spacing": None})
                >>> g = Glyph(default_options=opts)
                >>> norm, cbar_kw, ticks = g._prepare_scalar_mapping(
                ...     np.array([0.0, 5.0, 10.0])
                ... )
                >>> norm is None
                True
                >>> g.default_options["ticks_spacing"]
                1.0
                >>> [float(t) for t in ticks]
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

                ```
            - Flat data does not produce a zero spacing:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.base.glyph import Glyph
                >>> from cleopatra.styling.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None, "ticks_spacing": None})
                >>> g = Glyph(default_options=opts)
                >>> _ = g._prepare_scalar_mapping(np.array([3.0, 3.0, 3.0]))
                >>> g.default_options["ticks_spacing"]
                1.0

                ```
        """
        self._categorical = None
        if self.default_options.get("scheme") == "categorical":
            result = self._prepare_categorical_mapping(values)
        else:
            self._vmin, self._vmax = self._resolve_limits(np.asarray(values))
            if self.default_options.get("ticks_spacing") is None:
                self.ticks_spacing = (self._vmax - self._vmin) / 10 or 1.0
                self.default_options["ticks_spacing"] = self.ticks_spacing
            self.default_options["vmin"] = self._vmin
            self.default_options["vmax"] = self._vmax
            scheme = self.default_options.get("scheme")
            if scheme is not None:
                result = self._prepare_classified_mapping(values, scheme)
            else:
                ticks = self.get_ticks()
                norm, cbar_kw = self._create_norm_and_cbar_kw(ticks)
                result = (norm, cbar_kw, ticks)
        return result

    def _warn_scheme_overrides_continuous_options(self) -> None:
        """Warn when a `scheme` is set alongside continuous-only options.

        Shared by `_prepare_classified_mapping` and
        `_prepare_categorical_mapping`: either scheme owns the norm
        entirely, so a `color_scale` other than `"linear"` or an explicit
        `levels` the caller also set is silently ignored rather than
        applied -- this warns so that conflicting configuration is visible
        instead of quietly doing nothing.
        """
        if self.default_options.get("color_scale", "linear") != "linear":
            warnings.warn(
                "`scheme` is set, so `color_scale="
                f"{self.default_options['color_scale']!r}` is ignored "
                "(classification builds its own discrete norm).",
                stacklevel=5,
            )
        if self.default_options.get("levels") is not None:
            warnings.warn(
                "`scheme` is set, so `levels` is ignored (the classification "
                "scheme determines the bins).",
                stacklevel=5,
            )

    def _prepare_classified_mapping(
        self, values: np.ndarray, scheme: str | list | np.ndarray
    ) -> tuple[colors.BoundaryNorm, dict, np.ndarray]:
        """Build the `(norm, cbar_kw, ticks)` triple for classified colouring.

        The discrete sibling of the continuous branch in
        `_prepare_scalar_mapping`. When the `scheme` option is set, the
        data is binned into classes by `cleopatra.styling.styles.classify` (using
        the `k` option for the count/width schemes), and the resulting bin
        edges drive a `matplotlib.colors.BoundaryNorm` plus a colorbar
        whose ticks sit on the class boundaries — so `create_color_bar`
        renders a stepped colorbar. The `color_scale` / `levels` options
        are intentionally bypassed here; classification owns the norm.

        Args:
            values: The scalar array to classify and colour-map.
            scheme: A scheme name accepted by `classify` (e.g.
                `"quantiles"`, `"equal_interval"`) or an explicit sequence
                of bin edges.

        Returns:
            tuple[BoundaryNorm, dict, np.ndarray]: the discrete norm, the
                colorbar keyword arguments (boundary `ticks` plus
                `extend`), and the bin edges (returned in the `ticks`
                slot of the shared contract).

        Raises:
            ValueError: Propagated from `classify` (unknown scheme,
                degenerate data, or `k < 1`).

        Examples:
            - A quantile scheme yields a `BoundaryNorm` and boundary ticks:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.base.glyph import Glyph
                >>> from cleopatra.styling.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update(
                ...     {"vmin": None, "vmax": None, "scheme": "quantiles", "k": 4}
                ... )
                >>> g = Glyph(default_options=opts)
                >>> norm, cbar_kw, edges = g._prepare_classified_mapping(
                ...     np.arange(100.0), "quantiles"
                ... )
                >>> [float(b) for b in norm.boundaries]
                [0.0, 24.75, 49.5, 74.25, 99.0]
                >>> [float(t) for t in cbar_kw["ticks"]]
                [0.0, 24.75, 49.5, 74.25, 99.0]
                >>> cbar_kw["extend"]
                'neither'

                ```
        """
        self._warn_scheme_overrides_continuous_options()
        k = self.default_options.get("k", 5)
        bin_edges, norm = classify(values, scheme, k)
        extend = self.default_options.get("extend")
        cbar_kw = {
            "ticks": bin_edges,
            "extend": "neither" if extend is None else extend,
        }
        return norm, cbar_kw, bin_edges

    def _prepare_categorical_mapping(
        self, values: np.ndarray
    ) -> tuple[colors.BoundaryNorm, dict, np.ndarray]:
        """Build the `(norm, cbar_kw, edges)` triple for `scheme="categorical"`.

        The nominal sibling of `_prepare_classified_mapping`: instead of
        binning a continuous range, `cleopatra.styling.styles.categorize` assigns
        one colour per distinct value in `values` (sorted when sortable),
        and this builds a `ListedColormap` + `BoundaryNorm` over the
        resulting integer class codes — the same construction
        `colors.apply_data_style` uses for a preset's `categories`, but with
        the category table auto-derived from the data instead of
        hand-authored. The mapping (per-element codes, the `ListedColormap`,
        and the colour/label pairs) is stashed on `self._categorical` for
        the calling glyph to read back, since — unlike the continuous and
        classified paths — the array fed to the mappable is these integer
        codes, not `values` itself (which may not even be numeric).

        Only glyphs with `_SUPPORTS_CATEGORICAL_SCHEME = True` may use this
        scheme: for any other glyph, `values` are a continuous magnitude
        (e.g. vector length), where "one colour per distinct float" is
        almost never what the caller wants, and the glyph's `plot()` does
        not know to read `self._categorical` back in the first place — it
        would keep feeding the raw (mismatched) values to the mappable.

        The glyph's `cmap` option drives `categorize`'s palette, with one
        override: if `cmap` is still at the shared continuous/diverging
        default (`"coolwarm_r"`, matched by resolved name so a `Colormap`
        instance equivalent to the default is caught too, not just the bare
        string) — i.e. the caller never overrode it — it is substituted with
        `CATEGORICAL_DEFAULT_CMAP` (`"tab10"`) instead, since sampling a
        diverging gradient at N points would defeat the point of "one
        distinct colour per class". Any other `cmap`, qualitative or not,
        is always honoured as given.

        Args:
            values: The per-element nominal values to categorize.

        Returns:
            tuple[BoundaryNorm, dict, np.ndarray]: the discrete norm over
                the integer class codes, an empty colorbar-kwargs dict (a
                categorical scheme draws a `disjoint_legend`, never a
                colorbar — see `create_categorical_legend`), and the code
                boundary edges (`-0.5 .. n_categories - 0.5`).

        Raises:
            ValueError: If this glyph does not support `scheme="categorical"`,
                or (propagated from `categorize`) if `values` has no
                non-null entries.

        Examples:
            - Three distinct values map to three integer codes and colours:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.primitives.polygon_glyph import PolygonGlyph
                >>> polys = [np.zeros((3, 2))] * 3
                >>> g = PolygonGlyph(polys, values=np.array(["a", "b", "a"]))
                >>> norm, cbar_kw, edges = g._prepare_categorical_mapping(
                ...     np.array(["a", "b", "a"])
                ... )
                >>> [float(b) for b in edges]
                [-0.5, 0.5, 1.5]
                >>> [float(c) for c in g._categorical["codes"]]
                [0.0, 1.0, 0.0]

                ```
        """
        if not self._SUPPORTS_CATEGORICAL_SCHEME:
            raise ValueError(
                f"{type(self).__name__} does not support scheme='categorical' "
                "(its values are a continuous magnitude, not nominal class "
                "labels)."
            )
        self._warn_scheme_overrides_continuous_options()
        cmap = resolve_colormap(self.default_options["cmap"])
        cmap_name = cmap if isinstance(cmap, str) else getattr(cmap, "name", None)
        if cmap_name == STYLE_DEFAULTS["cmap"]:
            cmap = CATEGORICAL_DEFAULT_CMAP
        raw = np.asarray(values, dtype=object).ravel().tolist()
        categories, palette = categorize(raw, cmap=cmap)
        lookup = {category: i for i, category in enumerate(categories.tolist())}
        codes = np.array([lookup.get(v, np.nan) for v in raw], dtype=float)
        listed_cmap = colors.ListedColormap(palette)
        edges = np.arange(len(categories) + 1) - 0.5
        norm = colors.BoundaryNorm(edges, len(palette))
        self._categorical = {
            "codes": codes,
            "cmap": listed_cmap,
            "colors": palette,
            "labels": [str(c) for c in categories.tolist()],
        }
        return norm, {}, edges

    def create_categorical_legend(self, ax: Axes) -> Legend:
        """Attach the disjoint legend for a `scheme="categorical"` mapping.

        Reads the category colours/labels `_prepare_categorical_mapping`
        stashed on `self._categorical` and draws them via
        `cleopatra.styling.styles.disjoint_legend` — the discrete counterpart to
        `create_color_bar`, used instead of it whenever `scheme` is
        `"categorical"` (a colorbar would imply a false ordering over
        nominal classes). The legend's title defaults to the `cbar_label`
        option (the same label a continuous plot would put on its
        colorbar); the `category_legend_kwargs` option is merged over that
        default and forwarded to `disjoint_legend` (e.g. `loc`, `ncol`,
        `bbox_to_anchor`, or an explicit `title` override) — the categorical
        counterpart to `size_legend_kwargs`.

        Args:
            ax: The axes to attach the legend to.

        Returns:
            Legend: The created legend artist, already added to `ax`.

        Raises:
            ValueError: If `self._categorical` has not been populated yet
                (i.e. `_prepare_categorical_mapping` has not run for this
                glyph instance).

        Examples:
            - Prepare a categorical mapping, then draw and inspect the legend:
                ```python
                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.glyphs.primitives.polygon_glyph import PolygonGlyph
                >>> polys = [np.zeros((3, 2))] * 3
                >>> g = PolygonGlyph(polys, values=np.array(["a", "b", "a"]))
                >>> _ = g._prepare_categorical_mapping(np.array(["a", "b", "a"]))
                >>> fig, ax = plt.subplots()
                >>> legend = g.create_categorical_legend(ax)
                >>> [t.get_text() for t in legend.get_texts()]
                ['a', 'b']
                >>> plt.close(fig)

                ```
            - Calling it before a categorical mapping exists raises `ValueError`:
                ```python
                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.glyphs.primitives.polygon_glyph import PolygonGlyph
                >>> g = PolygonGlyph([np.zeros((3, 2))] * 2, values=np.array(["a", "b"]))
                >>> fig, ax = plt.subplots()
                >>> g.create_categorical_legend(ax)
                Traceback (most recent call last):
                    ...
                ValueError: create_categorical_legend() called before a scheme='categorical' mapping was prepared -- call _prepare_scalar_mapping (or plot()) first.
                >>> plt.close(fig)

                ```
            - `category_legend_kwargs` overrides the default title and adds
                a `loc`:
                ```python
                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.glyphs.primitives.polygon_glyph import PolygonGlyph
                >>> polys = [np.zeros((3, 2))] * 2
                >>> g = PolygonGlyph(polys, values=np.array(["a", "b"]))
                >>> g.default_options["category_legend_kwargs"] = {
                ...     "title": "Class", "loc": "upper left"
                ... }
                >>> _ = g._prepare_categorical_mapping(np.array(["a", "b"]))
                >>> fig, ax = plt.subplots()
                >>> legend = g.create_categorical_legend(ax)
                >>> legend.get_title().get_text()
                'Class'
                >>> plt.close(fig)

                ```
        """
        categorical = self._categorical
        if categorical is None:
            raise ValueError(
                "create_categorical_legend() called before a "
                "scheme='categorical' mapping was prepared -- call "
                "_prepare_scalar_mapping (or plot()) first."
            )
        legend_kwargs = {
            "title": self.default_options.get("cbar_label"),
            **(self.default_options.get("category_legend_kwargs") or {}),
        }
        return disjoint_legend(
            ax,
            categorical["colors"],
            categorical["labels"],
            **legend_kwargs,
        )

    def create_color_bar(self, ax: Axes, im: Any, cbar_kw: dict) -> Colorbar:
        """Create a colorbar with full customization from default_options.

        Reads `cbar_length`, `cbar_orientation`, `cbar_label`,
        `cbar_label_size`, and `cbar_label_location` from
        `default_options` to configure the colorbar. When the optional
        `cbar_kwargs` entry is present in `default_options` (an
        xarray-aligned dict-of-overrides), its keys are merged over the
        defaults so the user wins on any collision (e.g. `label`,
        `shrink`, `orientation`, `ticks`, `extend`).

        `cbar_kwargs` is read from `self.default_options["cbar_kwargs"]`.
        Set it via the constructor or `plot` kwargs of the calling
        glyph subclass. Keys recognised by `matplotlib.pyplot.colorbar`
        — `label`, `shrink`, `aspect`, `orientation`, `pad`,
        `ticks`, `extend` — are forwarded; `label` is special-cased
        so that label-size and label-location styling from
        `default_options` are still applied.

        Placement is controlled by `cbar_location`
        (`'left'`/`'right'`/`'top'`/`'bottom'`, which also fixes the
        orientation) and `cbar_inside`: when `True`, the colorbar is inset
        *inside* `ax` at that edge (a child of `ax`, so it tracks
        `full_bleed` instead of floating), with its tick labels facing into
        the frame and an optional `cbar_box` backing panel drawn behind it so
        the data does not show through. When `cbar_location` is `None` and
        `cbar_inside` is `False`, placement is matplotlib's default.

        Args:
            ax: Matplotlib axes.
            im: The mappable (image or contour) to attach the
                colorbar to.
            cbar_kw: Colorbar keyword arguments (ticks, format,
                extend, etc.) computed by
                `_create_norm_and_cbar_kw`.

        Returns:
            Colorbar: The created colorbar.

        Raises:
            TypeError: If `default_options["cbar_kwargs"]` is set
                but is not a `dict`.

        Examples:
            - Create a colorbar with a custom label:
                ```python
                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.glyphs.base.glyph import Glyph
                >>> from cleopatra.styling.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None})
                >>> g = Glyph(default_options=opts, cbar_label="Depth [m]")
                >>> fig, ax = plt.subplots()
                >>> im = ax.imshow(np.arange(9).reshape(3, 3))
                >>> cbar = g.create_color_bar(ax, im, {"ticks": [0, 4, 8]})
                >>> cbar.orientation
                'vertical'

                ```
            - User-supplied `cbar_kwargs` win on collision and
                `label` is applied via `set_label`:
                ```python
                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.glyphs.base.glyph import Glyph
                >>> from cleopatra.styling.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({
                ...     "vmin": None,
                ...     "vmax": None,
                ...     "cbar_kwargs": {"label": "User Label", "orientation": "horizontal"},
                ... })
                >>> g = Glyph(default_options=opts, cbar_label="Default Label")
                >>> fig, ax = plt.subplots()
                >>> im = ax.imshow(np.arange(9).reshape(3, 3))
                >>> cbar = g.create_color_bar(ax, im, {"ticks": [0, 4, 8]})
                >>> cbar.orientation
                'horizontal'
                >>> cbar.ax.get_xlabel() or cbar.ax.get_ylabel()
                'User Label'

                ```
            - Non-dict `cbar_kwargs` raises `TypeError`:
                ```python
                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.glyphs.base.glyph import Glyph
                >>> from cleopatra.styling.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None, "cbar_kwargs": "oops"})
                >>> g = Glyph(default_options=opts)
                >>> fig, ax = plt.subplots()
                >>> im = ax.imshow(np.arange(9).reshape(3, 3))
                >>> g.create_color_bar(ax, im, {"ticks": [0, 4, 8]})
                Traceback (most recent call last):
                    ...
                TypeError: cbar_kwargs must be a dict of colorbar keyword arguments, got str.

                ```
        """
        location = self.default_options.get("cbar_location")
        if location is not None and location not in ("left", "right", "top", "bottom"):
            raise ValueError(
                "cbar_location must be one of 'left', 'right', 'top', "
                f"'bottom', or None, got {location!r}."
            )
        orientation_opt = self.default_options.get("cbar_orientation")
        if orientation_opt is not None and orientation_opt not in ("vertical", "horizontal"):
            raise ValueError(
                "cbar_orientation must be 'vertical' or 'horizontal', got "
                f"{orientation_opt!r}."
            )
        inside = bool(self.default_options.get("cbar_inside", False))
        orientation = self._resolve_cbar_orientation(location)
        user_kwargs, user_label = self._cbar_user_kwargs()

        box_info = None
        if inside:
            inset_location = location or (
                "bottom" if orientation == "horizontal" else "right"
            )
            cbar, box_info = self._inside_colorbar_axes(
                ax, im, cbar_kw, inset_location, orientation, user_kwargs
            )
        else:
            cbar = self._outside_colorbar(
                ax, im, cbar_kw, location, orientation, user_kwargs
            )

        self._apply_cbar_styling(cbar, user_label)
        box = self.default_options.get("cbar_box")
        if inside and box and box_info is not None:
            self._draw_cbar_box(ax, box_info, box)
        return cbar

    def _resolve_cbar_orientation(self, location: str | None) -> str:
        """Orientation implied by `cbar_location` (else `cbar_orientation`)."""
        if location in ("left", "right"):
            orientation = "vertical"
        elif location in ("top", "bottom"):
            orientation = "horizontal"
        else:
            orientation = self.default_options["cbar_orientation"]
        return orientation

    def _cbar_user_kwargs(self) -> tuple[dict, Any]:
        """A validated copy of `cbar_kwargs` with `label` split out for set_label."""
        user_kwargs = self.default_options.get("cbar_kwargs") or {}
        if not isinstance(user_kwargs, dict):
            raise TypeError(
                "cbar_kwargs must be a dict of colorbar keyword "
                f"arguments, got {type(user_kwargs).__name__}."
            )
        user_kwargs = dict(user_kwargs)
        return user_kwargs, user_kwargs.pop("label", None)

    def _outside_colorbar(
        self,
        ax: Axes,
        im: Any,
        cbar_kw: dict,
        location: str | None,
        orientation: str,
        user_kwargs: dict,
    ) -> Colorbar:
        """Draw a normal (outside-gutter) colorbar via `fig.colorbar`."""
        fig = ax.figure
        merged_kw = {
            "shrink": self.default_options["cbar_length"],
            "pad": 0.02,
            "use_gridspec": len(fig.axes) <= 1,
        }
        if location is not None:
            # matplotlib places the bar on that side and sets orientation.
            merged_kw["location"] = location
        else:
            merged_kw["orientation"] = orientation
        merged_kw.update(cbar_kw)
        merged_kw.update(user_kwargs)
        if "location" in merged_kw:
            merged_kw.pop("orientation", None)
        return fig.colorbar(im, ax=ax, **merged_kw)

    def _apply_cbar_styling(self, cbar: Colorbar, user_label: Any) -> None:
        """Apply tick/label colours, size, location, and text to `cbar`."""
        tick_color = self.default_options.get("cbar_tick_color")
        label_color = self.default_options.get("cbar_label_color")
        cbar.ax.tick_params(
            labelsize=10, **({"colors": tick_color} if tick_color else {})
        )
        label_text = (
            user_label if user_label is not None else self.default_options["cbar_label"]
        )
        label_rotation = self.default_options.get("cbar_label_rotation")
        label_location = self.default_options["cbar_label_location"]
        if label_location is not None:
            valid = (
                ("bottom", "center", "top")
                if cbar.orientation == "vertical"
                else ("left", "center", "right")
            )
            if label_location not in valid:
                raise ValueError(
                    f"cbar_label_location={label_location!r} is not valid for a "
                    f"{cbar.orientation} colorbar; use one of {list(valid)}."
                )
        cbar.set_label(
            label_text,
            fontsize=self.default_options["cbar_label_size"],
            loc=label_location,
            **({"color": label_color} if label_color else {}),
            **({"rotation": label_rotation} if label_rotation is not None else {}),
        )

    def _inside_colorbar_axes(
        self,
        ax: Axes,
        im: Any,
        cbar_kw: dict,
        location: str,
        orientation: str,
        user_kwargs: dict,
    ) -> tuple[Colorbar, tuple]:
        """Draw the colorbar as an inset *inside* `ax` at `location`.

        The colorbar is placed in an inset axes (a child of `ax`), so it
        tracks the data axes through `full_bleed` instead of floating. Its
        tick labels are turned to face into the frame, so a backing box can
        enclose them.

        Args:
            ax: The data axes to inset the colorbar into.
            im: The mappable to attach the colorbar to.
            cbar_kw: Colorbar keyword arguments from `_create_norm_and_cbar_kw`.
            location: Edge to sit on -- `'left'`, `'right'`, `'top'`, `'bottom'`.
            orientation: `'vertical'` or `'horizontal'`.
            user_kwargs: Extra `fig.colorbar` kwargs (user `cbar_kwargs`).

        Returns:
            tuple: `(cbar, box_info)` where `box_info` is
                `(cax, inset_bounds, label_side)` for `_draw_cbar_box`.
        """
        fig = ax.figure
        default_length = STYLE_DEFAULTS["cbar_length"]
        length = self.default_options.get("cbar_length") or default_length
        long_frac = 0.72 * (length / default_length)
        long_start = 0.5 - long_frac / 2
        bounds, label_side = {
            "right": ((0.905, long_start, 0.022, long_frac), "left"),
            "left": ((0.073, long_start, 0.022, long_frac), "right"),
            "top": ((long_start, 0.905, long_frac, 0.022), "bottom"),
            "bottom": ((long_start, 0.073, long_frac, 0.022), "top"),
        }[location]
        cax = ax.inset_axes(bounds)
        cax.set_zorder(6)
        merged_kw = {"orientation": orientation, "ticklocation": label_side}
        merged_kw.update(cbar_kw)
        merged_kw.update(user_kwargs)
        cbar = fig.colorbar(im, cax=cax, **merged_kw)
        return cbar, (cax, bounds, label_side)

    def _draw_cbar_box(self, ax: Axes, box_info: tuple, box: bool | str | dict) -> None:
        """Draw a backing panel behind an inset colorbar (its bar + tick labels).

        Sized to the colorbar's tight bounding box (labels included) with an
        analytic fallback on the label side, and drawn above the data but
        below the bar so the animating field can't show through the labels.

        Args:
            ax: The data axes the colorbar is inset into.
            box_info: `(cax, inset_bounds, label_side)` from
                `_inside_colorbar_axes`.
            box: `True` for a translucent white panel, a colour string for a
                panel of that colour, or a dict of `Rectangle` kwargs.
        """
        cax, bounds, label_side = box_info
        kw: dict = {
            "facecolor": "white",
            "edgecolor": "0.6",
            "linewidth": 0.6,
        }
        if isinstance(box, str):
            kw["facecolor"] = box
        elif isinstance(box, dict):
            kw = {**kw, **box}
        fig = ax.figure
        try:
            fig.canvas.draw()
            bb = cax.get_tightbbox(fig.canvas.get_renderer())
            inv = ax.transAxes.inverted()
            x0, y0 = inv.transform((bb.x0, bb.y0))
            x1, y1 = inv.transform((bb.x1, bb.y1))
        except Exception:  # pragma: no cover - renderer unavailable on some backends
            # Fallback: the inset bounds, grown on the side the labels face.
            bx0, by0, bw, bh = bounds
            x0, y0, x1, y1 = bx0, by0, bx0 + bw, by0 + bh
            allow = 0.11 if bw < bh else 0.06
            if label_side == "left":
                x0 -= allow
            elif label_side == "right":
                x1 += allow
            elif label_side == "bottom":
                y0 -= allow
            else:
                y1 += allow
        pad = 0.014
        rect = Rectangle(
            (x0 - pad, y0 - pad),
            (x1 - x0) + 2 * pad,
            (y1 - y0) + 2 * pad,
            transform=ax.transAxes,
            zorder=5,
            clip_on=False,
            **kw,
        )
        ax.add_patch(rect)

    def adjust_ticks(
        self,
        axis: str,
        multiply_value: float | int = 1,
        add_value: float | int = 0,
        fmt: str = "{0:g}",
        visible: bool = True,
    ) -> None:
        """Adjust the axis tick labels with a linear transformation.

        Applies `tick_value * multiply_value + add_value` to each
        tick, formatted with `fmt`. Useful for converting pixel
        coordinates to real-world units.

        Args:
            axis: `"x"` or `"y"`.
            multiply_value: Multiplier for tick values. Default is 1.
            add_value: Offset added to tick values. Default is 0.
            fmt: Format string for tick labels.
                Default is `"{0:g}"`.
            visible: Whether the axis is visible. Default is True.

        Examples:
            - Scale x-axis ticks by 100 and offset by 5:
                ```python
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.glyphs.base.glyph import Glyph
                >>> from cleopatra.styling.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None})
                >>> g = Glyph(default_options=opts)
                >>> fig, ax = plt.subplots()
                >>> _ = ax.plot([0, 1, 2], [0, 1, 2])
                >>> g.fig, g.ax = fig, ax
                >>> g.adjust_ticks(axis="x", multiply_value=100, add_value=5)

                ```
        """
        assert self.ax is not None
        if axis == "x":
            ticks_fn = ticker.FuncFormatter(
                lambda x, pos: fmt.format(x * multiply_value + add_value)
            )
            self.ax.xaxis.set_major_formatter(ticks_fn)
        else:
            ticks_fn = ticker.FuncFormatter(
                lambda y, pos: fmt.format(y * multiply_value + add_value)
            )
            self.ax.yaxis.set_major_formatter(ticks_fn)

        if not visible:
            if axis == "x":
                self.ax.get_xaxis().set_visible(visible)
            else:
                self.ax.get_yaxis().set_visible(visible)

    @staticmethod
    def _plot_point_values(
        ax, point_table: np.ndarray, point_label_color, point_label_size
    ):
        """Plot point value labels on the axes."""
        write_points = lambda x: ax.text(
            x[2],
            x[1],
            x[0],
            ha="center",
            va="center",
            color=point_label_color,
            fontsize=point_label_size,
        )
        return list(map(write_points, point_table))

    def save_animation(self, path: str | os.PathLike, fps: int = 2, **kwargs) -> None:
        """Save this glyph's animation (`self.anim`) to a file.

        Thin wrapper around `cleopatra.glyphs.base.animation.save_animation`; the output
        format is determined by the file extension. GIF and WebP use an
        optimising Pillow writer; mov/avi/mp4 use FFmpeg (a system binary if
        present, otherwise the one bundled with imageio-ffmpeg).

        Args:
            path: Output file path, as a `str` or `os.PathLike` (e.g. a
                `pathlib.Path`). Extension determines format.
                Supported: gif, mov, avi, mp4, webp.
            fps: Frames per second. Default is 2.
            **kwargs: Additional keyword arguments forwarded to
                `cleopatra.glyphs.base.animation.save_animation`, e.g. `crf`, `bitrate`,
                `codec`, `preset`, `pix_fmt`, `dpi` (ffmpeg formats) or
                `optimize` and `loop` (GIF).

        Raises:
            ValueError: If `animate()` has not been called yet, if the file
                format is not supported, or if both `crf` and `bitrate`
                are given.
            FileNotFoundError: If a video format is requested but neither a
                system FFmpeg nor imageio-ffmpeg's bundled binary is found.

        Examples:
            - Check the supported video formats:
                ```python
                >>> from cleopatra.glyphs.base.glyph import SUPPORTED_VIDEO_FORMAT
                >>> sorted(SUPPORTED_VIDEO_FORMAT)
                ['avi', 'gif', 'mov', 'mp4', 'webp']

                ```
        """
        _save_animation(self.anim, path, fps=fps, **kwargs)
