"""
Module: Array.

This module provides a class, `Array`, to handle 3D arrays and perform various operations on them,
such as plotting, animating, and displaying the array.

The `Array` class has the following functionalities:
- Initialize an array object with the provided parameters.
- Plot the array with optional parameters to customize the appearance and display cell values.
- Animate the array over time with optional parameters to customize the animation speed and display points.
- Display the array with optional parameters to customize the appearance and display point IDs.

The `Array` class has the following attributes:
- `arr`: The 3D array to be handled.
- `time`: The time values for animation.
- `points`: The points to be displayed on the array.
- `default_options`: A dictionary to store default options for plotting, animating, and displaying.

The `Array` class has the following methods:
- `plot`: Plot the array with optional parameters.
- `animate`: Animate the array over time with optional parameters.
- `display`: Display the array with optional parameters.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from math import ceil
from typing import Any, Literal, TypedDict, Unpack, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from hpc.indexing import get_indices2
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from PIL import Image

from cleopatra.styling.colorbar import (
    _DEPRECATED_CBAR_KWARGS as _DEPRECATED_CBAR_KWARGS,
)
from cleopatra.styling.colorbar import (
    ColorBar,
    _resolve_colorbar,
    _swatch_text_default,
    _warn_deprecated_cbar_kwargs,
)
from cleopatra.styling.colors import (
    DATA_STYLES,
    alpha_rgba,
    apply_data_style,
    category_boundaries,
    resolve_colormap,
    resolve_single_layer_style,
    resolve_style_norm,
)
from cleopatra.basemap.geo import Basemap as Basemap
from cleopatra.basemap.geo import Feature as Feature
from cleopatra.basemap.geo import GeoMixin
from cleopatra.glyphs.base.glyph import (
    Glyph,
    _clear_prior_render_artists,
    _clear_projection_frame,
    _mark_render_artists,
    _restore_flat_axes,
    _root_figure,
    _stash_projection_frame,
)
from cleopatra.glyphs.base.hillshade import resolve_hillshade, shade_grid, shade_rgb
from cleopatra.basemap.projection import apply_projection_style, projection_draws_frame
from cleopatra.styling.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS
from cleopatra.styling.styles import (
    ColorScale,  # re-exported for convenience  # noqa: F401
    disjoint_legend,
    swatch_extend_prefixes,
    swatch_legend,
)

ARRAY_DEFAULT_OPTIONS: dict[str, Any] = {
    "vmin": None,
    "vmax": None,
    "num_size": 8,
    "display_cell_value": False,
    "background_color_threshold": None,
    "id_color": "green",
    "id_size": 20,
    "precision": 2,
    "kind": "auto",
    "levels": None,
    "robust": False,
    "center": None,
    "extend": None,
    "cbar_kwargs": None,
    "add_colorbar": True,
    "cbar_location": None,
    "cbar_inside": False,
    "cbar_box": None,
    "cbar_label_color": None,
    "cbar_tick_color": None,
    "labels": False,
    "label_kw": None,
    "hillshade": False,
    "style": None,
    "projection": None,
}
ARRAY_DEFAULT_OPTIONS = STYLE_DEFAULTS | ARRAY_DEFAULT_OPTIONS
#: Backwards-compatible alias for the array glyph's default options
#: (named like the other glyphs' `*_DEFAULT_OPTIONS` constants).
DEFAULT_OPTIONS = ARRAY_DEFAULT_OPTIONS

#: Tuple of accepted `kind=` values for `ArrayGlyph.plot`.
VALID_PLOT_KINDS = ("auto", "imshow", "pcolormesh", "contour", "contourf")
#: Tuple of accepted values for the xarray-aligned `extend` colorbar kwarg.
VALID_EXTEND_VALUES = ("neither", "both", "min", "max")
#: Default colormap auto-selected when `center` is set without an explicit `cmap`.
DIVERGING_DEFAULT_CMAP = "RdBu_r"
#: Lower percentile (2.0) used by xarray-style `robust=True` colour limits.
ROBUST_LOWER_PERCENTILE = 2.0
#: Upper percentile (98.0) used by xarray-style `robust=True` colour limits.
ROBUST_UPPER_PERCENTILE = 98.0
#: Invariant phrase in the `ValueError` raised by `ArrayGlyph._validate_coords`
#: when a coord array's shape does not match the data array. Kept stable so tests
#: can match against it without coupling to the full (shape-interpolated) message.
_COORD_SHAPE_MISMATCH = "coord array shape does not match the data array"
#: Invariant phrase in the `ValueError` raised by `ArrayGlyph._validate_coords`
#: when a coord array has a non-numeric dtype.
_COORD_DTYPE_MISMATCH = "coord arrays must be numeric (integer or float)"


class _Unset:
    """Sentinel type for "the caller did not pass this explicit parameter".

    A plain `object()` sentinel would work too, but this gives `help()` /
    IDE signature tooltips a readable `<unset>` instead of
    `<object object at 0x...>` for the parameter default that uses it
    (`ArrayGlyph.animate`'s `cell_value_text_colors`).
    """

    def __repr__(self) -> str:
        return "<unset>"


#: Sentinel default for a renamed parameter whose *real* default is
#: resolved inside `_resolve_renamed_kwarg` rather than in the method
#: signature. Distinguishes "caller didn't pass this parameter" from
#: "caller explicitly passed a value equal to its default" -- an ambiguity
#: a plain equality-with-default check cannot resolve, which previously
#: made the "both old and new given" conflict detection silently prefer
#: the deprecated value in that case (the opposite of the documented
#: "new wins" behaviour).
_UNSET = _Unset()


def _resolve_renamed_kwarg(
    kwargs: dict, old_name: str, new_name: str, new_value: Any, default: Any
) -> Any:
    """Resolve a renamed keyword argument, honouring its deprecated alias.

    Several `ArrayGlyph.animate` parameters were renamed for clarity
    (e.g. `text_colors` -> `cell_value_text_colors`). Since the old name is
    no longer an explicit parameter, a caller still using it arrives here
    through `**kwargs` instead -- this pops it out, emits a
    `DeprecationWarning`, and returns its value. Must be called *before*
    `plot`/`animate` validate `kwargs` against `self.default_options` (the
    old name is never a valid option key, so it would otherwise raise
    there instead of being resolved).

    Args:
        kwargs: The method's `**kwargs` dict; mutated in place (the old
            key, if present, is popped so it never reaches the strict
            `default_options` validation).
        old_name: The deprecated parameter name to look for in `kwargs`.
        new_name: The current parameter name, used in the warning message.
        new_value: The value the caller's `new_name` argument resolved to.
            `new_name`'s own signature default must be the `_UNSET`
            sentinel (not a concrete value) so this can tell "the caller
            didn't pass `new_name`" apart from "the caller explicitly
            passed `new_name` equal to its real default" -- the two cases
            a plain equality-with-default check cannot distinguish.
        default: `new_name`'s real default value, substituted when
            neither name was given.

    Returns:
        Any: `new_value` when `old_name` is absent from `kwargs`, or when
            both names were given (new wins, i.e. `new_value` is not
            `_UNSET`); otherwise the popped `old_name` value, or `default`
            when neither was given.

    Examples:
        - The old name is used and a `DeprecationWarning` is raised:
            ```python
            >>> import warnings
            >>> kwargs = {"text_colors": ("yellow", "purple")}
            >>> with warnings.catch_warnings(record=True) as caught:
            ...     warnings.simplefilter("always")
            ...     resolved = _resolve_renamed_kwarg(
            ...         kwargs, "text_colors", "cell_value_text_colors",
            ...         _UNSET, ("white", "black"),
            ...     )
            >>> resolved
            ('yellow', 'purple')
            >>> "text_colors" in kwargs
            False
            >>> issubclass(caught[0].category, DeprecationWarning)
            True

            ```
        - Both names given: the new one wins even when it is equal to its
            own default (the false-negative a plain equality check would
            miss):
            ```python
            >>> kwargs = {"text_colors": ("yellow", "purple")}
            >>> _resolve_renamed_kwarg(
            ...     kwargs, "text_colors", "cell_value_text_colors",
            ...     ("white", "black"), ("white", "black"),
            ... )
            ('white', 'black')

            ```
        - With no old-name alias present, the new value passes through
            untouched and `kwargs` is unaffected:
            ```python
            >>> kwargs = {"cmap": "viridis"}
            >>> _resolve_renamed_kwarg(
            ...     kwargs, "text_colors", "cell_value_text_colors",
            ...     ("yellow", "purple"), ("white", "black"),
            ... )
            ('yellow', 'purple')
            >>> kwargs
            {'cmap': 'viridis'}

            ```
    """
    if old_name not in kwargs:
        return default if new_value is _UNSET else new_value
    old_value = kwargs.pop(old_name)
    warnings.warn(
        f"`{old_name}` is deprecated; use `{new_name}` instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    if new_value is not _UNSET:
        warnings.warn(
            f"Both `{old_name}` (deprecated) and `{new_name}` were given; "
            f"`{new_name}` wins.",
            stacklevel=3,
        )
        return new_value
    return old_value


#: Static typing for the **kwargs `plot`/`animate` accept -- purely a typing
#: aid (see PEP 692 `Unpack`): with `from __future__ import annotations` the
#: `**kwargs: Unpack[...]` annotations below are never evaluated at runtime,
#: so this adds IDE autocomplete / type-checker typo-catching for `**kwargs`
#: without changing the existing `self.default_options` merge-and-validate
#: mechanism at all. Grouped to match the methods' own docstring sections;
#: each group is shared by both methods except `XarrayColourOptions` and
#: `ContourOptions`, which only `plot` documents (`animate` has no `kind`,
#: so nothing routes to `contour`/`contourf`, and does not recompute
#: `vmin`/`vmax` from `robust`/`center`).
class TitleOption(TypedDict, total=False):
    """The `title` kwarg -- `animate` only.

    `plot` also accepts a title, but as an explicit named parameter (not
    via `**kwargs`); mixing `title` into its `Unpack`'d TypedDict too would
    overlap with that parameter. `animate` has no such parameter, so its
    `**kwargs` is where `title` is actually reachable.

    Attributes:
        title: Plot title, by default `'Array Plot'`.
    """

    title: str | None


class PlotAppearanceOptions(TypedDict, total=False):
    """Colormap/colour-limit options shared by `plot` and `animate`.

    Attributes:
        title_size: Title font size, by default `15`.
        cmap: Colormap name or `Colormap` instance, by default `'coolwarm_r'`.
        vmin: Minimum value for colour scaling, by default `min(array)`.
        vmax: Maximum value for colour scaling, by default `max(array)`.
    """

    title_size: int
    cmap: str | Colormap
    vmin: float | None
    vmax: float | None


class ColorbarOptions(TypedDict, total=False):
    """Colorbar placement/label options shared by `plot` and `animate`.

    Attributes:
        add_colorbar: Whether to draw the glyph's own colorbar, by
            default `True`.
        cbar_orientation: Colorbar orientation, by default `'vertical'`.
        cbar_label_rotation: Rotation angle (degrees) of the colorbar label.
            `None` (the default) leaves matplotlib's own default orientation.
        cbar_label_location: Location of the colorbar label, by default
            `'center'`. Valid values depend on the bar orientation -- vertical:
            `'top'`/`'center'`/`'bottom'`; horizontal: `'left'`/`'center'`/`'right'`.
        cbar_length: Ratio controlling the colorbar's height/width, by
            default `0.75`.
        ticks_spacing: Spacing between colorbar ticks, by default `2`.
        cbar_label_size: Font size of the colorbar label, by default `12`.
        cbar_label: Label text for the colorbar, by default `'Value'`.

    Note:
        Colorbar *placement* (edge, inside/outside, backing box) is set
        through the `colorbar=` parameter of `plot` / `animate` with a
        `ColorBar`, not through this dict.
    """

    add_colorbar: bool
    cbar_orientation: Literal["vertical", "horizontal"]
    cbar_label_rotation: float | None
    cbar_label_location: Literal["left", "right", "top", "bottom", "center"]
    cbar_length: float
    ticks_spacing: float
    cbar_label_size: int
    cbar_label: str | None


class ColorScaleOptions(TypedDict, total=False):
    """Colour-scaling options shared by `plot` and `animate`.

    Attributes:
        color_scale: Colour scaling kind, by default `'linear'`. See
            `cleopatra.styling.styles.ColorScale`.
        gamma: Exponent for the `'power'` colour scale, by default `0.5`.
        line_threshold: Threshold for the `'sym-lognorm'` colour scale, by
            default `0.0001`.
        line_scale: Scale factor for the `'sym-lognorm'` colour scale, by
            default `0.001`.
        bounds: Boundaries for the `'boundary-norm'` colour scale, by
            default `None`.
        midpoint: Midpoint value for the `'midpoint'` colour scale, by
            default `0`.
    """

    color_scale: ColorScale | str
    gamma: float
    line_threshold: float
    line_scale: float
    bounds: list[float] | None
    midpoint: float


class XarrayColourOptions(TypedDict, total=False):
    """Xarray-aligned colour options -- `plot` only.

    Attributes:
        robust: When `True`, use the 2nd/98th percentile of the data for
            `vmin`/`vmax`, matching xarray's `robust=True`, by default
            `False`.
        center: Diverging-colormap centring value, by default `None`.
        extend: Colorbar arrow extension, by default `None` (auto-resolve).
        cbar_kwargs: Extra keyword arguments forwarded to `fig.colorbar`,
            by default `None`.
    """

    robust: bool
    center: float | None
    extend: Literal["neither", "both", "min", "max"] | None
    cbar_kwargs: dict[str, Any] | None


class ContourOptions(TypedDict, total=False):
    """Contour-line and discretisation options -- `plot` only.

    Attributes:
        levels: Discrete colour levels (xarray-aligned), by default `None`.
        labels: Draw inline numeric labels on a line `contour`'s isolines,
            by default `False`.
        label_kw: Extra keyword arguments forwarded to `ax.clabel` when
            `labels=True`, by default `None`.
    """

    levels: int | Sequence[float] | None
    labels: bool
    label_kw: dict[str, Any] | None


class CellValueOptions(TypedDict, total=False):
    """Per-cell value-text display options shared by `plot` and `animate`.

    Attributes:
        display_cell_value: Whether to display each cell's value as text,
            by default `False`.
        num_size: Font size of the cell value text, by default `8`.
        background_color_threshold: Threshold for the cell value text
            colour, by default `None` (uses `max(array) / 2`).
    """

    display_cell_value: bool
    num_size: int
    background_color_threshold: float | None


class AnimateCellValueOptions(TypedDict, total=False):
    """Per-cell value-text option specific to `animate` -- `animate` only.

    Attributes:
        precision: Decimal places each frame's cell value text is rounded
            to, by default `2`. `animate`-only: `plot`'s equivalent
            per-cell text (`ArrayGlyph._plot_text`) always rounds to 2
            decimal places internally and never reads this option.
    """

    precision: int


class DataStyleOptions(TypedDict, total=False):
    """Named data-style preset / relief-shading options shared by `plot`
    and `animate`.

    Attributes:
        style: Name of a `cleopatra.styling.colors.DATA_STYLES` preset, by default
            `None`.
        hillshade: Relief-shade a regular-grid DEM; `True` for defaults,
            or a dict tuning `vert_exag`/`azimuth`/`altitude`/
            `blend_mode`/`multidirectional`. By default `False`.
    """

    style: str | None
    hillshade: bool | dict[str, Any]


class PlotKwargs(
    PlotAppearanceOptions,
    ColorbarOptions,
    ColorScaleOptions,
    XarrayColourOptions,
    ContourOptions,
    CellValueOptions,
    DataStyleOptions,
    total=False,
):
    """The full set of `**kwargs` `ArrayGlyph.plot` accepts."""


class AnimateKwargs(
    TitleOption,
    PlotAppearanceOptions,
    ColorbarOptions,
    ColorScaleOptions,
    CellValueOptions,
    AnimateCellValueOptions,
    DataStyleOptions,
    total=False,
):
    """The full set of `**kwargs` `ArrayGlyph.animate` accepts."""


class PointOverlay:
    """A point overlay for `ArrayGlyph.plot`/`animate`: locations plus styling.

    Bundles the five point-overlay parameters (`points`, and the marker /
    value-label colour and size) that `plot`/`animate` previously accepted
    as five separate, identically-named arguments duplicated across both
    signatures. Pass an instance as `plot(points=...)` /
    `animate(points=...)` instead of the individual `point_color` /
    `point_size` / `point_label_color` / `point_label_size` keywords.

    Attributes:
        points: `(N, 3)` array: first column the value to display at each
            point, second/third columns the point's row/column index in
            the underlying array.
        color: Marker colour, by default `"red"`. Any valid matplotlib
            colour string.
        size: Marker size, by default `100`.
        label_color: Colour of the point-value text label drawn at each
            point, by default `"blue"`.
        label_size: Font size of the point-value text label, by default
            `10`.

    Examples:
        - Build an overlay and pass it to `plot`:
            ```python
            >>> import numpy as np
            >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph, PointOverlay
            >>> arr = np.arange(9, dtype=float).reshape(3, 3)
            >>> overlay = PointOverlay(np.array([[5.0, 1, 1]]), color="black")
            >>> fig, ax = ArrayGlyph(arr).plot(points=overlay)
            >>> overlay.color
            'black'
            >>> overlay.size
            100

            ```
    """

    def __init__(
        self,
        points: np.ndarray,
        *,
        color: str = "red",
        size: int | float = 100,
        label_color: str = "blue",
        label_size: int | float = 10,
    ) -> None:
        """Initialise a `PointOverlay`.

        Args:
            points: `(N, 3)` array: value, row index, column index per point.
            color: Marker colour, by default `"red"`.
            size: Marker size, by default `100`.
            label_color: Point-value label colour, by default `"blue"`.
            label_size: Point-value label font size, by default `10`.
        """
        self.points = points
        self.color = color
        self.size = size
        self.label_color = label_color
        self.label_size = label_size


class FrameLabel:
    """Styling for the per-frame time label `ArrayGlyph.animate` draws.

    Bundles the two frame-label parameters (`location`, `color`) that
    `animate` previously accepted as separate `label_location` /
    `label_color` arguments. Pass an instance as
    `animate(frame_label=...)` instead.

    Attributes:
        location: `[x, y]` position for the label, by default `None`.
            When `None`, the label is anchored just inside the top-left
            corner using axes-fraction coordinates, so it stays clear of
            the top/bottom edges regardless of the array's shape or the
            axis orientation. A very narrow axes can still overflow
            horizontally at the default font size, since no anchor choice
            can fit a long label into less horizontal space than it
            needs; pass an explicit `[x, y]` (data coordinates) in that
            case.
        color: Label text colour, by default `"black"`. Any valid
            matplotlib colour string.
        size: Label font size in points, by default `None`. When `None`,
            the label inherits the colorbar label size
            (`cbar_label_size`, `12` by default); pass a number to size
            the frame label independently of the colorbar.

    Examples:
        - Build a frame label and pass it to `animate`:
            ```python
            >>> import numpy as np
            >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph, FrameLabel
            >>> stack = np.arange(3 * 9, dtype=float).reshape(3, 3, 3)
            >>> label = FrameLabel(location=[0.1, 0.1], color="white")
            >>> glyph = ArrayGlyph(stack)
            >>> anim_obj = glyph.animate(["t0", "t1", "t2"], frame_label=label)
            >>> label.color
            'white'

            ```
    """

    def __init__(
        self,
        *,
        location: list[float] | None = None,
        color: str = "black",
        size: float | None = None,
    ) -> None:
        """Initialise a `FrameLabel`.

        Args:
            location: `[x, y]` label position, by default `None` (auto
                top-left anchor -- see the class docstring).
            color: Label text colour, by default `"black"`.
            size: Label font size in points, by default `None` (inherit
                the colorbar label size -- see the class docstring).
        """
        self.location = location
        self.color = color
        self.size = size


#: Deprecated `plot`/`animate` kwargs that `_resolve_point_overlay` folds
#: into a `PointOverlay` instead of the (now-removed) individual keywords.
#: `pid_color`/`pid_size` are the oldest generation (pre-dating even
#: `point_label_color`/`point_label_size`) and are honoured too, so the
#: deprecation chain from either generation still lands on `PointOverlay`.
_DEPRECATED_POINT_STYLE_KWARGS = (
    "point_color",
    "point_size",
    "point_label_color",
    "point_label_size",
    "pid_color",
    "pid_size",
)


def _pop_first(
    kwargs: dict, names: tuple[str, ...], default: Any
) -> tuple[Any, str | None]:
    """Pop every one of `names` present in `kwargs`; return the most-preferred value.

    All matching keys are removed (not just the winner) so none of them
    linger to fail the caller's subsequent strict `kwargs` validation --
    e.g. `plot(points=arr, point_label_color=..., pid_color=...)` must pop
    both, even though only `point_label_color`'s value is used.

    Args:
        kwargs: Dict to pop from; mutated in place.
        names: Candidate keys, most-preferred first.
        default: Value to return if none of `names` are present.

    Returns:
        tuple[Any, str | None]: The most-preferred present value (or
            `default`), and the key it came from (`None` if none were
            present).
    """
    present = [name for name in names if name in kwargs]
    values = {name: kwargs.pop(name) for name in present}
    if not present:
        return default, None
    winner = present[0]
    return values[winner], winner


def _resolve_point_overlay(
    points: np.ndarray | PointOverlay | None, kwargs: dict
) -> PointOverlay | None:
    """Normalise `plot`/`animate`'s `points` argument into a `PointOverlay`.

    Accepts the current calling convention (`points` is already a
    `PointOverlay`, or `None`) as well as two deprecated generations
    (`points` is a plain array, styled via separate `point_color` /
    `point_size` / `point_label_color` / `point_label_size` keywords, or
    the even older `pid_color` / `pid_size` for the label styling) -- any
    of these emit a `DeprecationWarning` and are folded into a
    `PointOverlay` here. The deprecated style keys are drained out of
    `kwargs` even when `points` is `None` (a no-op, matching their
    pre-`PointOverlay` behaviour as named parameters with defaults) so
    they never reach the caller's strict `kwargs` validation. Must be
    called *before* `plot`/`animate` validate `kwargs` against
    `self.default_options`, which would otherwise reject the deprecated
    keys outright.

    Args:
        points: The raw `points` argument as received by `plot`/`animate`:
            a `PointOverlay`, a plain `(N, 3)` array, or `None`.
        kwargs: The method's `**kwargs` dict; mutated in place (any
            deprecated point-style keys are popped out).

    Returns:
        PointOverlay | None: `points` unchanged if it was already a
            `PointOverlay` or `None`; otherwise a new `PointOverlay`
            wrapping the array and the (possibly deprecated) style kwargs.
    """
    if isinstance(points, PointOverlay):
        used_deprecated = [k for k in _DEPRECATED_POINT_STYLE_KWARGS if k in kwargs]
        if used_deprecated:
            warnings.warn(
                f"{used_deprecated} are ignored when `points` is a "
                "`PointOverlay` -- set them on the `PointOverlay` instance "
                "instead.",
                stacklevel=3,
            )
            for key in used_deprecated:
                kwargs.pop(key)
        return points
    if points is None:
        _, color_key = _pop_first(kwargs, ("point_color",), None)
        _, size_key = _pop_first(kwargs, ("point_size",), None)
        _, label_color_key = _pop_first(
            kwargs, ("point_label_color", "pid_color"), None
        )
        _, label_size_key = _pop_first(kwargs, ("point_label_size", "pid_size"), None)
        used = [k for k in (color_key, size_key, label_color_key, label_size_key) if k]
        if used:
            warnings.warn(
                f"{used} have no effect without `points`; pass a "
                "`cleopatra.glyphs.gridded.array_glyph.PointOverlay` as `points` instead.",
                DeprecationWarning,
                stacklevel=3,
            )
        return None
    color, color_key = _pop_first(kwargs, ("point_color",), "red")
    size, size_key = _pop_first(kwargs, ("point_size",), 100)
    label_color, label_color_key = _pop_first(
        kwargs, ("point_label_color", "pid_color"), "blue"
    )
    label_size, label_size_key = _pop_first(
        kwargs, ("point_label_size", "pid_size"), 10
    )
    used = [k for k in (color_key, size_key, label_color_key, label_size_key) if k]
    if used:
        warnings.warn(
            f"Passing `points` as a plain array together with {used} is "
            "deprecated; pass a `cleopatra.glyphs.gridded.array_glyph.PointOverlay` instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return PointOverlay(
        points, color=color, size=size, label_color=label_color, label_size=label_size
    )


#: Deprecated `animate` kwargs that `_resolve_frame_label` folds into a
#: `FrameLabel` instead of the (now-removed) individual keywords.
#: `text_loc` is the oldest generation (pre-dating `label_location`) and is
#: honoured too.
_DEPRECATED_FRAME_LABEL_KWARGS = ("label_location", "label_color", "text_loc")


def _resolve_frame_label(frame_label: Any, kwargs: dict) -> FrameLabel:
    """Normalise `animate`'s `frame_label` argument into a `FrameLabel`.

    Accepts the current calling convention (`frame_label` is already a
    `FrameLabel`, or `None` for the defaults) as well as two deprecated
    ones: separate `label_location` / `label_color` keywords (or the even
    older `text_loc` for the location), and a bare `[x, y]` value at the
    `frame_label` position itself -- the pre-`FrameLabel` calling shape,
    still reachable positionally (`animate(time, None, colors, interval,
    [x, y])`) even though `label_location`/`text_loc` were never
    positional-only. Any of these emit a `DeprecationWarning` and are
    folded into a `FrameLabel` here. Unlike `_resolve_point_overlay`, this
    never returns `None`: `animate` always draws a frame label, so
    `frame_label=None` means "use `FrameLabel`'s own defaults", not "no
    label". Must be called *before* `animate` validates `kwargs` against
    `self.default_options`, which would otherwise reject the deprecated
    keys outright.

    Args:
        frame_label: The raw `frame_label` argument as received by
            `animate`: a `FrameLabel`, `None`, or (deprecated) a plain
            `[x, y]` value passed positionally.
        kwargs: The method's `**kwargs` dict; mutated in place (any
            deprecated frame-label keys are popped out).

    Returns:
        FrameLabel: `frame_label` unchanged if it was already a
            `FrameLabel`; otherwise a new `FrameLabel` built from the
            positional value and/or the (possibly deprecated)
            location/color kwargs, or the defaults.
    """
    if isinstance(frame_label, FrameLabel):
        used_deprecated = [k for k in _DEPRECATED_FRAME_LABEL_KWARGS if k in kwargs]
        if used_deprecated:
            warnings.warn(
                f"{used_deprecated} are ignored when `frame_label` is a "
                "`FrameLabel` -- set them on the `FrameLabel` instance "
                "instead.",
                stacklevel=3,
            )
            for key in used_deprecated:
                kwargs.pop(key)
        return frame_label
    color, color_key = _pop_first(kwargs, ("label_color",), "black")
    kwarg_location, kwarg_location_key = _pop_first(
        kwargs, ("label_location", "text_loc"), None
    )
    if frame_label is not None:
        location = frame_label
        location_keys = ["frame_label (positional)"]
        if kwarg_location_key:
            location_keys.append(kwarg_location_key)
    else:
        location = kwarg_location
        location_keys = [kwarg_location_key] if kwarg_location_key else []
    used = location_keys + ([color_key] if color_key else [])
    if used:
        warnings.warn(
            f"Passing {used} directly is deprecated; pass a "
            "`cleopatra.glyphs.gridded.array_glyph.FrameLabel` as `frame_label` instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return FrameLabel(location=location, color=color)


class FacetGrid:
    """Result object for a multi-subplot facet plot.

    Mirrors xarray's `xarray.plot.facetgrid.FacetGrid` return
    shape so downstream code that already targets xarray can be reused
    without changes. Produced by `ArrayGlyph.facet`; do not
    construct directly.

    Attributes:
        fig: The shared `matplotlib.figure.Figure`.
        axes: 2-D `ndarray` of `matplotlib.axes.Axes`. Empty
            subplot slots (when `col_wrap` does not divide the stack
            evenly) are hidden via `Axes.set_visible`.
        cbar: The shared `matplotlib.colorbar.Colorbar` attached
            to the first rendered subplot. `None` when faceting an
            RGB stack (no colorbar in the RGB path).
        name_dicts: List of `{dim_name: coord_value}` dicts, one per
            rendered subplot. Mirrors
            `xarray.plot.facetgrid.FacetGrid.name_dicts` so
            callers can map subplot index to facet coordinate.

    Examples:
        - Inspect the grid shape returned by `ArrayGlyph.facet`:
            ```python
            >>> import numpy as np
            >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
            >>> stack = np.arange(4 * 5 * 5, dtype=float).reshape(4, 5, 5)
            >>> g = ArrayGlyph(stack).facet(col="t")
            >>> g.axes.shape
            (1, 4)
            >>> len(g.name_dicts)
            4

            ```
    """

    def __init__(
        self,
        fig: Figure,
        axes: np.ndarray,
        cbar: Colorbar | None,
        name_dicts: list[dict[str, Any]],
    ) -> None:
        """Initialise the `FacetGrid` result object.

        `ArrayGlyph.facet` is the only intended caller. End users
        receive an already-populated instance and should not invoke
        this constructor directly.

        Args:
            fig: The shared `matplotlib.figure.Figure` that owns
                every subplot.
            axes: 2-D `ndarray` of `matplotlib.axes.Axes` with
                shape `(nrows, ncols)`. Empty slots (when `col_wrap`
                does not divide the panel count evenly) are kept in the
                array but hidden with `Axes.set_visible(False)`.
            cbar: The shared `matplotlib.colorbar.Colorbar` for
                the grid, attached to the first rendered subplot;
                `None` for an RGB facet that has no colorbar.
            name_dicts: One `{dim_name: coord_value}` dict per
                rendered subplot, in row-major (left-to-right,
                top-to-bottom) order, mirroring
                `xarray.plot.facetgrid.FacetGrid.name_dicts`.

        Examples:
            - The result-object fields line up with the keyword args
                used to construct it:
                ```python
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.glyphs.gridded.array_glyph import FacetGrid
                >>> fig, axes = plt.subplots(1, 2, squeeze=False)
                >>> grid = FacetGrid(
                ...     fig=fig,
                ...     axes=axes,
                ...     cbar=None,
                ...     name_dicts=[{"t": 0}, {"t": 1}],
                ... )
                >>> grid.axes.shape
                (1, 2)
                >>> grid.cbar is None
                True
                >>> [d["t"] for d in grid.name_dicts]
                [0, 1]
                >>> plt.close(fig)

                ```
        """
        self.fig = fig
        self.axes = axes
        self.cbar = cbar
        self.name_dicts = name_dicts


class ArrayGlyph(GeoMixin, Glyph):
    """A class to handle arrays and perform various visualization operations on them.

    The ArrayGlyph class provides functionality for visualizing 2D and 3D arrays with
    various customization options. It supports plotting single arrays, RGB arrays,
    and creating animations from 3D arrays.

    Attributes:
        fig (matplotlib.figure.Figure): The matplotlib figure object.
        ax (matplotlib.axes.Axes): The matplotlib axes object.
        extent (List): The extent of the array [xmin, xmax, ymin, ymax].
        rgb (bool): Whether the array is an RGB array.
        num_domain_cells (int): Number of cells in the data domain — cells
            that are neither masked (via `exclude_value`) nor NaN. For a
            3-D stack this is counted on the first frame. Equals the number
            of per-cell value labels drawn when `display_cell_value=True`.
            (The legacy alias `no_elem` still works but is deprecated.)
        anim (matplotlib.animation.FuncAnimation): The animation object if created.
        im (matplotlib.cm.ScalarMappable): The colour-mapped artist produced by
            the most recent `plot`/`animate` call (e.g. the `AxesImage` for
            `imshow`, the `QuadMesh` for `pcolormesh`, the `QuadContourSet`
            for `contour`/`contourf`, or the RGB `AxesImage`). `None` before
            the first render. Lets a caller attach a colorbar/legend or query
            the colour limits without scraping `ax.images`/`ax.collections`.
        cbar (matplotlib.colorbar.Colorbar): The colorbar drawn by the glyph,
            or `None` when none was drawn (RGB, or `add_colorbar=False`).
        contour_labels (list): The inline contour-label `Text` artists from
            the most recent `plot(kind="contour", labels=True)`, or `None`
            when labelling was not requested (the default, and for every
            kind other than `"contour"`). A labelled contour with no
            isolines (e.g. a constant-value field) yields an empty list.

    Notes:
        This class provides methods for:
        - Plotting arrays with customizable color scales, color bars, and annotations
        - Creating animations from 3D arrays
        - Displaying point values on arrays
        - Customizing plot appearance

    Examples:
    - Create a simple array plot:
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
        >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> array_glyph = ArrayGlyph(arr)
        >>> fig, ax = array_glyph.plot()

        ```
    - Create an RGB plot from a 3D array:
    ```python
    >>> rgb_array = np.random.randint(0, 255, size=(3, 10, 10))
    >>> rgb_glyph = ArrayGlyph(rgb_array, rgb=[0, 1, 2])
    >>> fig, ax = rgb_glyph.plot()

    ```
    - Create an animated plot from a 3D array:
    ```python
    >>> time_series = np.random.randint(1, 10, size=(5, 10, 10))
    >>> time_labels = ["Frame 1", "Frame 2", "Frame 3", "Frame 4", "Frame 5"]
    >>> animated_glyph = ArrayGlyph(time_series)
    >>> anim = animated_glyph.animate(time_labels)

    ```
    """

    #: Option keys this glyph accepts (see `Glyph.option_keys`/`filter_kwargs`).
    DEFAULT_OPTIONS = ARRAY_DEFAULT_OPTIONS

    def __init__(
        self,
        array: np.ndarray,
        exclude_value: float | list = np.nan,
        extent: list | None = None,
        coords: tuple[np.ndarray, np.ndarray] | list[np.ndarray] | None = None,
        rgb: list[int] | None = None,
        surface_reflectance: int | None = None,
        cutoff: list | None = None,
        ax: Axes | None = None,
        fig: Figure | None = None,
        percentile: int | None = None,
        **kwargs,
    ):
        """Initialize the ArrayGlyph object with an array and optional parameters.

        Args:
            array: The array to be visualized. Can be a 2D array for single plots or a 3D array for RGB plots or animations.
            exclude_value: Value(s) used to mask cells out of the domain, by default np.nan.
                Can be a single value or a list of values to exclude.
            extent: The extent of the array in the format [xmin, ymin, xmax, ymax], by default None.
                If provided, the array will be plotted with these spatial boundaries.
                Mutually exclusive with `coords`.
            coords: Optional `(x, y)` coordinate arrays for curvilinear
                or non-uniform grids, by default None. Each element is
                either a 1-D array of cell centres (length matches the
                last/second-to-last axis of `array`) or a 2-D array
                matching the last two axes of `array`. When set,
                `kind="auto"` routes to `pcolormesh` instead of
                `imshow`. Mutually exclusive with `extent`.
            rgb: The indices of the red, green, and blue bands in the given array, by default None.
                If provided, the array will be treated as an RGB image.
                Can be a list of three values [r, g, b], or four values if alpha band is included [r, g, b, a].
            surface_reflectance: Surface reflectance value for normalizing satellite data, by default None.
                Typically 10000 for Sentinel-2 data.
            cutoff: Clip the range of pixel values for each band, by default None.
                Takes only pixel values from 0 to the value of the cutoff and scales them back to between 0 and 1.
                Should be a list with one value per band.
            ax: A pre-existing axes to plot on, by default None. Bound to
                the glyph and used by `plot`/`animate` unless `plot(ax=...)`
                overrides it. Passing `ax` alone is enough — its parent
                figure is derived automatically; if None (and no axes is
                given to `plot`), a new axes is created.
            fig: A pre-existing figure to bind, by default None. `fig` is a
                construction-time binding only (it is never a `plot`
                parameter — `plot` derives the figure from its axes). When
                `ax` is given, `fig` is optional; if both are None a new
                figure is created at render time.
            percentile: The percentile value to be used for scaling the array values, by default None.
                Used to enhance contrast by stretching the histogram.
            **kwargs: Additional keyword arguments for customizing the plot.
                Supported arguments include:
                    figsize : tuple, optional
                        Figure size, by default (8, 8).
                    vmin : float, optional
                        Minimum value for color scaling, by default min(array).
                    vmax : float, optional
                        Maximum value for color scaling, by default max(array).
                    title : str, optional
                        Title of the plot, by default 'Array Plot'.
                    title_size : int, optional
                        Title font size, by default 15.
                    cmap : str or matplotlib.colors.Colormap, optional
                        Colormap, by default 'coolwarm_r'. A plain matplotlib
                        name (e.g. 'viridis') or a `Colormap` object is used
                        as-is; a **namespaced** name such as 'cmocean:thermal'
                        or 'crameri:batlow' is resolved via the optional `cmap`
                        aggregator — install the `[science-colors]` extra
                        (`pip install cleopatra[science-colors]`). The `_r`
                        reverse suffix works on both forms.
                    kind : str, optional
                        Render kind. One of `"auto"`, `"imshow"`,
                        `"pcolormesh"`, `"contour"`, `"contourf"`.
                        Default `"auto"` (currently equivalent to
                        `"imshow"`). Stored on the instance and used
                        as the default for `plot`.
                    robust : bool, optional
                        When True, `vmin` / `vmax` are computed from
                        the 2nd and 98th percentile of the unmasked data
                        (xarray-aligned). An explicit `vmin` / `vmax`
                        wins over `robust`. Default False.
                    center : float, optional
                        Diverging-colormap centring value. When set,
                        `(vmin, vmax)` is made symmetric around
                        `center` and the cmap auto-switches to
                        `"RdBu_r"` if no explicit `cmap` was passed.
                        Default None (no centring).
                    levels : int or sequence, optional
                        Discrete colour levels (xarray-aligned). An
                        `int` selects N linearly-spaced edges between
                        `vmin` and `vmax`; a sequence is used as
                        explicit edges. Default None.
                    extend : str, optional
                        Colorbar arrow extension. One of `"neither"`,
                        `"both"`, `"min"`, `"max"`, or None to
                        auto-resolve at render time. Default None.
                    cbar_kwargs : dict, optional
                        Extra keyword arguments forwarded to
                        `fig.colorbar`; user keys win over cleopatra's
                        defaults on collision. Default None.

        Raises:
            ValueError: If an invalid keyword argument is provided.
            ValueError: If rgb is provided but the array doesn't have enough dimensions.
            ValueError: If `extend` is set to a value outside
                `{"neither", "both", "min", "max"}`.
            ValueError: If both `extent` and `coords` are supplied,
                or if a `coords` element has a shape that does not
                match `array`.
            TypeError: If `coords` is not a length-2 sequence of
                ndarrays.

        Examples:
        Basic initialization with a 2D array:
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
        >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> array_glyph = ArrayGlyph(arr)
        >>> fig, ax = array_glyph.plot()

        ```
        Initialization with custom figure size and title:
        ```python
        >>> array_glyph = ArrayGlyph(arr, figsize=(10, 8), title="Custom Array Plot")
        >>> fig, ax = array_glyph.plot()

        ```
        Initialization with RGB bands from a 3D array:
        ```python
        >>> rgb_array = np.random.randint(0, 255, size=(3, 10, 10))
        >>> rgb_glyph = ArrayGlyph(rgb_array, rgb=[0, 1, 2], surface_reflectance=255)
        >>> fig, ax = rgb_glyph.plot()

        ```
        Initialization with custom extent:
        ```python
        >>> array_glyph = ArrayGlyph(arr, extent=[0, 0, 10, 10])
        >>> fig, ax = array_glyph.plot()

        ```
        Robust colour limits (xarray-aligned `robust=True` clips the
        2nd/98th percentile so a few outliers do not dominate the
        scale):
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
        >>> data = np.arange(100, dtype=float).reshape(10, 10)
        >>> data[0, 0] = 1e6  # outlier
        >>> glyph = ArrayGlyph(data, robust=True)
        >>> round(glyph.vmin, 1), round(glyph.vmax, 1)
        (3.0, 98.0)

        ```
        Centring on a value for diverging data (auto-switches the cmap
        to `"RdBu_r"` when no `cmap` is passed):
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
        >>> anomaly = np.linspace(-3.0, 8.0, 25).reshape(5, 5)
        >>> glyph = ArrayGlyph(anomaly, center=0.0)
        >>> glyph.vmin, glyph.vmax
        (-8.0, 8.0)
        >>> glyph.default_options["cmap"]
        'RdBu_r'

        ```
        Combining `levels`, `extend` and `cbar_kwargs` (forwarded
        to `matplotlib.colorbar.Colorbar`):
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
        >>> arr = np.arange(25, dtype=float).reshape(5, 5)
        >>> glyph = ArrayGlyph(
        ...     arr,
        ...     levels=5,
        ...     extend="both",
        ...     cbar_kwargs={"shrink": 0.6},
        ... )
        >>> glyph.default_options["levels"]
        5
        >>> glyph.default_options["extend"]
        'both'
        >>> glyph.default_options["cbar_kwargs"]
        {'shrink': 0.6}

        ```
        Invalid `extend` is rejected at construction time:
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
        >>> ArrayGlyph(np.array([[0.0, 1.0]]), extend="up")
        Traceback (most recent call last):
            ...
        ValueError: Invalid extend='up'. Valid values are ('neither', 'both', 'min', 'max') or None.

        ```
        Curvilinear coords (1-D centres) auto-route to
        `pcolormesh`:
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
        >>> arr = np.arange(12, dtype=float).reshape(3, 4)
        >>> x = np.linspace(0.0, 10.0, 4)
        >>> y = np.linspace(0.0, 5.0, 3)
        >>> glyph = ArrayGlyph(arr, coords=(x, y))
        >>> glyph.coords[0].shape, glyph.coords[1].shape
        ((4,), (3,))

        ```
        `extent` and `coords` are mutually exclusive:
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
        >>> arr = np.zeros((3, 4))
        >>> x = np.linspace(0.0, 10.0, 4)
        >>> y = np.linspace(0.0, 5.0, 3)
        >>> ArrayGlyph(arr, extent=[0, 0, 1, 1], coords=(x, y))
        Traceback (most recent call last):
            ...
        ValueError: `extent` and `coords` are mutually exclusive — pass one or the other.

        ```
        """
        super().__init__(
            default_options=ARRAY_DEFAULT_OPTIONS, fig=fig, ax=ax, **kwargs
        )
        if exclude_value is not np.nan:
            values = cast(list, exclude_value)
            if len(values) > 1:
                mask = np.logical_or(
                    np.isclose(array, values[0], rtol=0.001),
                    np.isclose(array, values[1], rtol=0.001),
                )
            else:
                mask = np.isclose(array, values[0], rtol=0.0000001)
            array = ma.array(array, mask=mask, dtype=array.dtype)
        else:
            array = ma.array(array)

        # convert the extent from [xmin, ymin, xmax, ymax] to [xmin, xmax, ymin, ymax] as required by matplotlib.
        if extent is not None and coords is not None:
            raise ValueError(
                "`extent` and `coords` are mutually exclusive — pass one or the other."
            )
        if extent is not None:
            extent = [extent[0], extent[2], extent[1], extent[3]]
        self.extent = extent

        self._coords = self._validate_coords(coords, array)

        if rgb is not None:
            self.rgb = True
            if array.shape[0] < 3:
                raise ValueError(
                    f"To plot RGB plot the given array should have only 3 arrays, given array have "
                    f"{array.shape[0]}"
                )
            else:
                array = self.prepare_array(
                    array,
                    rgb=rgb,
                    surface_reflectance=surface_reflectance,
                    cutoff=cutoff,
                    percentile=percentile,
                )
        else:
            self.rgb = False

        self._exclude_value = exclude_value
        self._validate_extend(self.default_options.get("extend"))

        explicit_keys = set(kwargs.keys())
        self._style_color_overrides: dict[str, Any] = {
            key: kwargs[key]
            for key in ("vmin", "vmax", "center", "cmap")
            if key in explicit_keys and kwargs[key] is not None
        }
        #: Whether the latest plot()/animate() call explicitly requested a real
        #: colorbar (a truthy `colorbar=`), which overrides a preset's swatch.
        self._style_wants_colorbar: bool = False
        self._vmin, self._vmax = self._resolve_color_limits(
            array,
            vmin_kw=kwargs.get("vmin"),
            vmax_kw=kwargs.get("vmax"),
            robust=bool(self.default_options.get("robust", False)),
            center=self.default_options.get("center"),
            vmin_explicit="vmin" in explicit_keys,
            vmax_explicit="vmax" in explicit_keys,
        )
        if (
            self.default_options.get("center") is not None
            and "cmap" not in explicit_keys
        ):
            self.default_options["cmap"] = DIVERGING_DEFAULT_CMAP

        self._arr = array
        self.ticks_spacing = (self._vmax - self._vmin) / 10 or 1.0
        shape = array.shape
        first_frame = array[0, :, :] if len(shape) == 3 else array
        self.num_domain_cells = len(get_indices2(first_frame, [np.nan]))
        self.im: Any = None
        self.cbar: Colorbar | None = None
        self._day_text: Any = None
        self.contour_labels: list[Any] | None = None

    @property
    def arr(self):
        """The (masked) array held by the glyph.

        The array is stored as a `numpy.ma.MaskedArray`; cells matching
        `exclude_value` (or NaN) are masked so they are excluded from the
        colour range and rendered as gaps.

        Returns:
            numpy.ma.MaskedArray: The array backing this glyph.

        Examples:
            - Read the array back and inspect its shape and a value:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> glyph = ArrayGlyph(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
                >>> glyph.arr.shape
                (2, 3)
                >>> float(glyph.arr[0, 0])
                1.0

                ```
        """
        return self._arr

    @arr.setter
    def arr(self, value):
        """Set the backing array.

        Args:
            value: The new array to store (see the `arr` property).
        """
        self._arr = value

    @property
    def no_elem(self) -> int:
        """Deprecated alias for `num_domain_cells`.

        Kept for backward compatibility; emits a `DeprecationWarning`.
        Will be removed in a future release.

        Returns:
            int: Same value as `num_domain_cells`.

        Examples:
            - The deprecated alias returns the same count as
                `num_domain_cells`:
                ```python
                >>> import warnings
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> glyph = ArrayGlyph(np.array([[1.0, 2.0], [3.0, 4.0]]))
                >>> with warnings.catch_warnings():
                ...     warnings.simplefilter("ignore")
                ...     glyph.no_elem == glyph.num_domain_cells
                True

                ```
        """
        warnings.warn(
            "`ArrayGlyph.no_elem` is deprecated; use `num_domain_cells` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.num_domain_cells

    def prepare_array(
        self,
        array: np.ndarray,
        rgb: list[int] | None = None,
        surface_reflectance: int | None = None,
        cutoff: list | None = None,
        percentile: int | None = None,
    ) -> np.ndarray:
        """Prepare an array for RGB visualization.

        This method processes a multi-band array to create an RGB image suitable for visualization.
        It can normalize the data using either percentile-based scaling or surface reflectance values.

        Args:
            array: The input array containing multiple bands. For RGB visualization,
                this should be a 3D array where the first dimension represents the bands.
            rgb: The indices of the red, green, and blue bands in the given array, by default None.
                If None, assumes the order is [3, 2, 1] (common for Sentinel-2 data).
            surface_reflectance: Surface reflectance value for normalizing satellite data, by default None.
                Typically 10000 for Sentinel-2 data or 255 for 8-bit imagery.
                Used to scale values to the range [0, 1].
            cutoff: Clip the range of pixel values for each band, by default None.
                Takes only pixel values from 0 to the value of the cutoff and scales them back to between 0 and 1.
                Should be a list with one value per band.
            percentile: The percentile value to be used for scaling the array values, by default None.
                Used to enhance contrast by stretching the histogram.
                If provided, this takes precedence over surface_reflectance.

        Returns:
            np.ndarray: The prepared array with shape (height, width, 3) suitable for RGB visualization.
                Values are normalized to the range [0, 1].
                the rgb 3d array is converted into 2d array to be plotted using the plt.imshow function.
                a float32 array normalized between 0 and 1 using the `percentile` values or the `surface_reflectance`.
                if the `percentile` or `surface_reflectance` values are not given, the function just reorders the values
                to have the red-green-blue order.

        Raises:
            ValueError: If the array shape is incompatible with the provided RGB indices.

        Notes:
            - The `prepare_array` function is called in the constructor of the `ArrayGlyph` class to prepare the array,
              so you can provide the same parameters of the `prepare_array` function to the `ArrayGlyph constructor`.
            - The prepare function moves the first axes (the channel axis) to the last axes, and then scales the array
              using the percentile values. If the percentile is not given, the function scales the array using the
              surface reflectance values. If the surface reflectance is not given, the function scales the array using
              the cutoff values. If the cutoff is not given, the function scales the array using the sentinel data

        Examples:
        Prepare an array using percentile-based scaling:
            ```python
            >>> import numpy as np
            >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
            >>> # Create a 3-band array (e.g., satellite image)
            >>> bands = np.random.randint(0, 10000, size=(3, 100, 100))
            >>> glyph = ArrayGlyph(np.zeros((1, 1)))  # Dummy initialization
            >>> rgb_array = glyph.prepare_array(bands, rgb=[0, 1, 2], percentile=2)
            >>> rgb_array.shape
            (100, 100, 3)
            >>> np.all((0 <= rgb_array) & (rgb_array <= 1))
            np.True_

            ```
        Prepare an array using surface reflectance normalization:
            ```python
            >>> rgb_array = glyph.prepare_array(bands, rgb=[0, 1, 2], surface_reflectance=10000)
            >>> rgb_array.shape
            (100, 100, 3)
            >>> np.all((0 <= rgb_array) & (rgb_array <= 1))
            np.True_

            ```
        Prepare an array with cutoff values:
            ```python
            >>> rgb_array = glyph.prepare_array(
            ...     bands, rgb=[0, 1, 2], surface_reflectance=10000, cutoff=[5000, 5000, 5000]
            ... )
            >>> rgb_array.shape
            (100, 100, 3)
            >>> np.all((0 <= rgb_array) & (rgb_array <= 1))
            np.True_

            ```

        - Create an array and instantiate the `ArrayGlyph` class.
            ```python
            >>> import numpy as np
            >>> arr = np.random.randint(0, 255, size=(3, 5, 5)).astype(np.float32)
            >>> array_glyph = ArrayGlyph(arr)
            >>> print(array_glyph.arr.shape)
            (3, 5, 5)

            ```
        `rgb` channels:
            - Now let's use the `prepare_array` function with `rgb` channels as [0, 1, 2]. so the finction does not to
                reorder the chennels. but it just needs to move the first axis to the last axis.
                ```python
                >>> rgb_array = array_glyph.prepare_array(arr, rgb=[0, 1, 2])
                >>> print(rgb_array.shape)
                (5, 5, 3)

                ```
            - If we compare the values of the first channel in the original array with the first array in the rgb array it
                should be the same.
                ```python
                >>> np.testing.assert_equal(arr[0, :, :],rgb_array[:, :, 0])

                ```
        surface_reflectance:
            - if you provide the surface reflectance value, the function will scale the array using the surface reflectance
                value to a normalized rgb values.
                ```python
                >>> array_glyph = ArrayGlyph(arr)
                >>> rgb_array = array_glyph.prepare_array(arr, surface_reflectance=10000, rgb=[0, 1, 2])
                >>> print(rgb_array.shape)
                (5, 5, 3)

                ```
            - if you print the values of the first channel, you will find all the values are between 0 and 1.
                ```python
                >>> print(rgb_array[:, :, 0]) # doctest: +SKIP
                [[0.0195 0.02   0.0109 0.0211 0.0087]
                 [0.0112 0.0221 0.0035 0.0234 0.0141]
                 [0.0116 0.0188 0.0001 0.0176 0.    ]
                 [0.0014 0.0147 0.0043 0.0167 0.0117]
                 [0.0083 0.0139 0.0186 0.02   0.0058]]

                ```
            - With the `surface_reflectance` parameter, you can also use the `cutoff` parameter to affect values that
                are above it, by rescaling them.
                ```python
                >>> rgb_array = array_glyph.prepare_array(
                ...     arr, surface_reflectance=10000, rgb=[0, 1, 2], cutoff=[0.8, 0.8, 0.8]
                ... )
                >>> print(rgb_array[:, :, 0]) # doctest: +SKIP
                [[0.     0.     0.     0.     0.    ]
                 [1.     1.     1.     1.     1.    ]
                 [1.     1.     1.     1.     1.    ]
                 [0.0014 0.0147 0.0043 0.0167 0.0117]
                 [0.0083 0.0139 0.0186 0.02   0.0058]]

                ```
        """
        array = array[rgb].transpose(1, 2, 0)

        if percentile is not None:
            array = self.scale_percentile(array, percentile=percentile)
        elif surface_reflectance is not None:
            array = self._prepare_sentinel_rgb(
                array,
                rgb=rgb,
                surface_reflectance=surface_reflectance,
                cutoff=cutoff,
            )
        return array

    def _prepare_sentinel_rgb(
        self,
        array: np.ndarray,
        rgb: list[int] | None = None,
        surface_reflectance: int = 10000,
        cutoff: list | None = None,
    ) -> np.ndarray:
        """Prepare Sentinel satellite data for RGB visualization.

        This method specifically handles Sentinel satellite imagery by normalizing the data
        using the provided surface reflectance value and optional cutoff values.

        Args:
            array: The input array with shape (height, width, 3) containing RGB bands.
                This array should already be transposed from the original band-first format.
            rgb: The indices of the red, green, and blue bands in the original array, by default None.
                Used only for cutoff application.
            surface_reflectance: Surface reflectance value for normalizing satellite data, by default 10000.
                Sentinel-2 data typically uses 10000 as the maximum reflectance value.
                Used to scale values to the range [0, 1].
            cutoff: Clip the range of pixel values for each band, by default None.
                Takes only pixel values from 0 to the value of the cutoff and scales them back to between 0 and 1.
                Should be a list with one value per band.

        Returns:
            np.ndarray: The prepared array with shape (height, width, 3) suitable for RGB visualization.
                Values are normalized to the range [0, 1].

        Examples:
        Prepare Sentinel-2 data with default surface reflectance:
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
        >>> # Create a simulated Sentinel-2 RGB array
        >>> rgb_data = np.random.randint(0, 10000, size=(100, 100, 3))
        >>> glyph = ArrayGlyph(np.zeros((1, 1)))  # Dummy initialization
        >>> normalized = glyph._prepare_sentinel_rgb(rgb_data)
        >>> np.all((0 <= normalized) & (normalized <= 1))
        np.True_

        ```
        Prepare Sentinel-2 data with custom cutoff values:
        ```python
        >>> cutoffs = [8000, 7000, 9000]
        >>> normalized = glyph._prepare_sentinel_rgb(rgb_data, rgb=[0, 1, 2], cutoff=cutoffs)
        >>> np.all((0 <= normalized) & (normalized <= 1))
        np.True_

        ```
        """
        array = np.clip(array / surface_reflectance, 0, 1)
        if cutoff is not None:
            bands = cast(list, rgb)
            array[0] = np.clip(bands[0], 0, cutoff[0]) / cutoff[0]
            array[1] = np.clip(bands[1], 0, cutoff[1]) / cutoff[1]
            array[2] = np.clip(bands[2], 0, cutoff[2]) / cutoff[2]

        return array

    @staticmethod
    def scale_percentile(arr: np.ndarray, percentile: int = 1) -> np.ndarray:
        """Scale an array using percentile-based contrast stretching.

        This method enhances the contrast of an image by stretching the histogram
        based on percentile values. It calculates the lower and upper percentile values
        for each band and normalizes the data to the range [0, 1].

        Args:
            arr: The array to be scaled, with shape (height, width, bands).
                Typically an RGB image with 3 bands.
            percentile: The percentile value to be used for scaling, by default 1.
                This value determines how much of the histogram tails to exclude.
                Higher values result in more contrast stretching.
                Typical values range from 1 to 5.

        Returns:
            np.ndarray: The scaled array, normalized between 0 and 1, with the same shape as input.
                Data type is float32.

        Notes:
            The method works by:
            1. Computing the lower percentile value for each band
            2. Computing the upper percentile value (100 - percentile) for each band
            3. Normalizing each band using these percentile values
            4. Clipping values to the range [0, 1]

            This is particularly useful for visualizing satellite imagery with high dynamic range.

        Examples:
        Scale a single-band array:
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
        >>> # Create a test array with values between 0 and 10000
        >>> test_array = np.random.randint(0, 10000, size=(100, 100, 1))
        >>> scaled = ArrayGlyph.scale_percentile(test_array, percentile=2)
        >>> scaled.shape
        (100, 100, 1)
        >>> np.all((0 <= scaled) & (scaled <= 1))
        np.True_

        ```
        Scale an RGB array:
        ```python
        >>> rgb_array = np.random.randint(0, 10000, size=(100, 100, 3))
        >>> scaled = ArrayGlyph.scale_percentile(rgb_array, percentile=2)
        >>> scaled.shape
        (100, 100, 3)
        >>> np.all((0 <= scaled) & (scaled <= 1))
        np.True_

        ```
        Using different percentile values affects contrast:
        ```python
        >>> low_contrast = ArrayGlyph.scale_percentile(rgb_array, percentile=1)
        >>> high_contrast = ArrayGlyph.scale_percentile(rgb_array, percentile=5)
        >>> # Higher percentile typically results in higher contrast

        ```
        """
        rows, columns, bands = arr.shape
        arr = np.reshape(arr, [rows * columns, bands]).astype(np.float32)
        lower_percent = np.percentile(arr, percentile, axis=0)
        upper_percent = np.percentile(arr, 100 - percentile, axis=0) - lower_percent
        arr = (arr - lower_percent[None, :]) / upper_percent[None, :]
        arr = np.reshape(arr, [rows, columns, bands])
        arr = arr.clip(0, 1)

        return arr

    def __str__(self):
        """String representation of the Array object."""
        message = f"""
                    Min: {self.vmin}
                    Max: {self.vmax}
                    Exclude values: {self.exclude_value}
                    RGB: {self.rgb}
                """
        return message

    @property
    def exclude_value(self):
        """Value(s) treated as nodata and masked out of the array.

        Cells equal to `exclude_value` are masked so they are excluded
        from the colour range and rendered as gaps. Defaults to `nan`.

        Returns:
            The excluded value, or a list of excluded values.

        Examples:
            - With no explicit nodata, NaN is excluded by default:
                ```python
                >>> import math
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> glyph = ArrayGlyph(np.array([[1.0, 2.0], [3.0, 4.0]]))
                >>> math.isnan(glyph.exclude_value)
                True

                ```
            - Excluding a sentinel masks the matching cells:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> arr = np.array([[1.0, 2.0], [3.0, -9.0]])
                >>> glyph = ArrayGlyph(arr, exclude_value=[-9.0])
                >>> glyph.exclude_value
                [-9.0]
                >>> int(glyph.arr.mask.sum())
                1

                ```
        """
        return self._exclude_value

    @exclude_value.setter
    def exclude_value(self, value):
        """Set the excluded nodata value(s).

        Args:
            value: The value (or list of values) to mask out (see the
                `exclude_value` property).
        """
        self._exclude_value = value

    def _auto_figsize(self) -> tuple[float, float]:
        """A figure size whose aspect matches the data, for a filled map.

        `ArrayGlyph` draws with equal aspect (undistorted geography), so a wide
        or tall field in the default square figure collapses to a thin strip with
        an oversized-looking colorbar. When the caller did not pass an explicit
        `figsize`, this derives one from the data's own aspect ratio -- from
        `extent` (matplotlib order `[xmin, xmax, ymin, ymax]`), else the `coords`
        ranges, else the array's pixel shape -- so the map fills the figure. Any
        degenerate input falls back to the configured default `figsize`.

        Returns:
            tuple[float, float]: `(width, height)` in inches.
        """
        default = tuple(self.default_options["figsize"])
        if self.default_options.get("projection") == "globe":
            return (7.5, 6.5)  # the orthographic disc is ~square, not the lon/lat aspect
        try:
            if self.extent is not None:
                xmin, xmax, ymin, ymax = (float(v) for v in self.extent)
                width, height = abs(xmax - xmin), abs(ymax - ymin)
            elif self._coords is not None:
                xs, ys = self._coords
                width = abs(float(np.nanmax(xs)) - float(np.nanmin(xs)))
                height = abs(float(np.nanmax(ys)) - float(np.nanmin(ys)))
            else:
                arr = np.asarray(self.arr)
                if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] in (3, 4)):
                    height, width = float(arr.shape[0]), float(arr.shape[1])
                else:
                    return default
        except (TypeError, ValueError, IndexError, AttributeError):
            return default
        if not (width > 0 and height > 0):
            return default
        aspect = width / height
        plot_height = 6.0        # target plot height (inches)
        cbar_pad = 1.8           # room for the colorbar + its labels
        max_width = 14.0
        fig_w = plot_height * aspect + cbar_pad
        fig_h = plot_height
        if fig_w > max_width:    # very wide field: cap width, shrink height to keep the aspect
            fig_w = max_width
            fig_h = max(3.5, (max_width - cbar_pad) / aspect)
        fig_w = max(5.0, fig_w)
        return (round(fig_w, 1), round(fig_h, 1))

    def create_figure_axes(self) -> tuple[Figure, Axes]:
        """Create the figure/axes, sizing the figure to the data when needed.

        Overrides `Glyph.create_figure_axes` to use `_auto_figsize` whenever the
        caller left `figsize` at its default (did not pass it explicitly), so an
        equal-aspect map fills the figure instead of collapsing into a strip. An
        explicit `figsize=` is always honoured unchanged.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The new figure
            and axes.
        """
        figsize = self.default_options["figsize"]
        auto = "figsize" not in getattr(self, "_explicit_options", set())
        if auto:
            figsize = self._auto_figsize()
        fig, ax = plt.subplots(figsize=figsize)
        self._owns_figure = True
        self._auto_figure = auto
        return fig, ax

    def _tighten_figure(self, pad_inches: float = 0.02) -> None:
        """Shrink the figure to its drawn content, in place.

        `ArrayGlyph` draws with equal aspect, so the figure holding it is almost
        always larger than the map + colorbar + title, leaving a margin. Jupyter's
        inline backend hides that margin because it saves with
        `bbox_inches="tight"`, but a plain `savefig` -- or an animation writer,
        which does not crop -- keeps it, so a saved figure or GIF looks loose
        while the inline preview looked tight.

        This closes that gap at the *figure* level rather than per save call:
        measure the rendered content once, translate every axes rigidly so the
        content's lower-left sits at the origin, and resize the figure to match.
        Because the figure itself becomes tight, every export path -- a bare
        `savefig`, `embed_gif`, `to_gif`, a raw `PillowWriter` -- is tight and
        identical. The whole axes group moves together, so the relative layout
        (map, colorbar gap, title) is preserved; only the outer margin is
        removed. A small uniform `pad_inches` is kept so edge ticks/labels are
        not shaved.

        Args:
            pad_inches: Uniform margin left around the content, in inches.
        """
        fig = self.fig
        if fig is None or not fig.axes:
            return
        try:
            fig.canvas.draw()
            content = fig.get_tightbbox(fig.canvas.get_renderer())
        except Exception:  # noqa: BLE001 -- tightening is cosmetic and fully optional
            return
        if content is None:
            return
        fig_w, fig_h = (float(v) for v in fig.get_size_inches())
        new_w = (content.x1 - content.x0) + 2 * pad_inches
        new_h = (content.y1 - content.y0) + 2 * pad_inches
        if not (new_w > 0 and new_h > 0):
            return
        for axes in fig.axes:
            pos = axes.get_position()
            axes.set_position(
                [
                    (pos.x0 * fig_w - content.x0 + pad_inches) / new_w,
                    (pos.y0 * fig_h - content.y0 + pad_inches) / new_h,
                    (pos.width * fig_w) / new_w,
                    (pos.height * fig_h) / new_h,
                ]
            )
        fig.set_size_inches(new_w, new_h)

    @staticmethod
    def _validate_coords(
        coords: tuple[np.ndarray, np.ndarray] | list[np.ndarray] | None,
        array: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Validate the `coords` kwarg and return a normalised `(x, y)` tuple.

        Args:
            coords: User-provided coordinates. `None` (no curvilinear
                support), or a length-2 tuple/list of arrays. Each
                element is either 1-D (length matches the last axis of
                `array` for `x` and the second-to-last for `y`)
                or 2-D with shape `array.shape[-2:]`.
            array: The data array used to validate coordinate shapes.

        Returns:
            tuple[np.ndarray, np.ndarray] or None: The validated
                `(x, y)` pair, with each element cast to `np.ndarray`.

        Raises:
            TypeError: If `coords` is not `None` and not a length-2
                sequence.
            ValueError: If a coordinate array has a shape that does not
                match the data array, or a non-numeric dtype (bool,
                complex, object, …).

        Examples:
            - `None` short-circuits to `None`:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> ArrayGlyph._validate_coords(None, np.zeros((3, 4))) is None
                True

                ```
            - 1-D centres matching `array.shape[-1]` (x) and
                `array.shape[-2]` (y) are accepted:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> arr = np.zeros((3, 4))
                >>> x = np.array([0.0, 1.0, 2.0, 3.0])
                >>> y = np.array([0.0, 1.0, 2.0])
                >>> xs, ys = ArrayGlyph._validate_coords((x, y), arr)
                >>> xs.shape, ys.shape
                ((4,), (3,))

                ```
            - A non-tuple raises `TypeError`:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> ArrayGlyph._validate_coords("oops", np.zeros((3, 4)))
                Traceback (most recent call last):
                    ...
                TypeError: `coords` must be a length-2 sequence of arrays (x, y), got str.

                ```
        """
        if coords is None:
            result = None
        else:
            if not isinstance(coords, (tuple, list)) or len(coords) != 2:
                raise TypeError(
                    "`coords` must be a length-2 sequence of arrays "
                    f"(x, y), got {type(coords).__name__}."
                )
            x_in, y_in = coords
            x_arr = np.asarray(x_in)
            y_arr = np.asarray(y_in)
            for name, arr_ in (("x", x_arr), ("y", y_arr)):
                if not (
                    np.issubdtype(arr_.dtype, np.integer)
                    or np.issubdtype(arr_.dtype, np.floating)
                ):
                    raise ValueError(
                        f"{name}: {_COORD_DTYPE_MISMATCH}; got dtype {arr_.dtype}."
                    )
            data_shape = array.shape[-2:]
            rows, cols = data_shape
            x_ok = (x_arr.ndim == 1 and x_arr.shape[0] == cols) or (
                x_arr.ndim == 2 and x_arr.shape == data_shape
            )
            y_ok = (y_arr.ndim == 1 and y_arr.shape[0] == rows) or (
                y_arr.ndim == 2 and y_arr.shape == data_shape
            )
            if not x_ok:
                raise ValueError(
                    f"x {_COORD_SHAPE_MISMATCH}: got shape {x_arr.shape}, "
                    f"expected 1-D length {cols} or 2-D {data_shape}."
                )
            if not y_ok:
                raise ValueError(
                    f"y {_COORD_SHAPE_MISMATCH}: got shape {y_arr.shape}, "
                    f"expected 1-D length {rows} or 2-D {data_shape}."
                )
            result = (x_arr, y_arr)
        return result

    @property
    def coords(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Optional `(x, y)` coordinate arrays for curvilinear grids.

        Returns the validated coordinate pair stored at construction
        time, or `None` when the glyph was built without `coords`
        (regular pixel-grid render). When non-`None`, `plot(kind="auto")`
        routes to `pcolormesh` so the (x, y) arrays are honoured.

        Returns:
            tuple[np.ndarray, np.ndarray] or None: The `(x, y)` pair
                as stored on the instance (each cast to
                `numpy.ndarray`), or `None`.

        Examples:
            - A glyph built without `coords` reports `None`:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> glyph = ArrayGlyph(np.zeros((3, 4)))
                >>> glyph.coords is None
                True

                ```
            - A glyph built with 1-D centres exposes the validated
                arrays back through the property:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> arr = np.zeros((3, 4))
                >>> x = np.linspace(0.0, 3.0, 4)
                >>> y = np.linspace(0.0, 2.0, 3)
                >>> glyph = ArrayGlyph(arr, coords=(x, y))
                >>> xs, ys = glyph.coords
                >>> xs.shape, ys.shape
                ((4,), (3,))
                >>> float(xs[-1]), float(ys[-1])
                (3.0, 2.0)

                ```
        """
        return self._coords

    @staticmethod
    def _validate_extend(extend: str | None) -> None:
        """Validate the `extend` kwarg against the allowed values.

        Args:
            extend: User-provided value for the colorbar extension. May
                be `None` (auto-resolve at render time) or one of
                `"neither"`, `"both"`, `"min"`, `"max"`.

        Raises:
            ValueError: When `extend` is not one of the accepted
                strings (or `None`).

        Examples:
            - Accepted values return `None` silently:
                ```python
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> ArrayGlyph._validate_extend("both") is None
                True
                >>> ArrayGlyph._validate_extend(None) is None
                True

                ```
            - Unsupported values raise `ValueError`:
                ```python
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> ArrayGlyph._validate_extend("up")
                Traceback (most recent call last):
                    ...
                ValueError: Invalid extend='up'. Valid values are ('neither', 'both', 'min', 'max') or None.

                ```
        """
        if extend is None:
            return
        if extend not in VALID_EXTEND_VALUES:
            raise ValueError(
                f"Invalid extend={extend!r}. Valid values are "
                f"{VALID_EXTEND_VALUES} or None."
            )

    @staticmethod
    def _robust_limits(arr: np.ndarray) -> tuple[float, float]:
        """Compute xarray-style robust `(vmin, vmax)` from the data.

        Returns the 2nd and 98th percentile of the unmasked, finite
        values in `arr` — the same convention as xarray's
        `robust=True`. Masked entries and NaNs are excluded from the
        percentile computation.

        Args:
            arr: Input array. May be a plain ndarray or a masked array.

        Returns:
            tuple[float, float]: `(vmin_robust, vmax_robust)`.

        Raises:
            ValueError: If the array contains no finite values.

        Examples:
            - Outliers are clipped to the 2nd/98th percentile:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> arr = np.arange(100, dtype=float)
                >>> arr[0] = -1e6  # extreme low outlier
                >>> arr[-1] = 1e6  # extreme high outlier
                >>> vmin, vmax = ArrayGlyph._robust_limits(arr)
                >>> round(vmin, 1), round(vmax, 1)
                (2.0, 97.0)

                ```
            - Masked and NaN entries are excluded from the percentile
                computation:
                ```python
                >>> import numpy as np
                >>> import numpy.ma as ma
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> raw = np.array([np.nan, 0.0, 1.0, 2.0, 3.0, 4.0])
                >>> arr = ma.array(raw, mask=[True, False, False, False, False, False])
                >>> vmin, vmax = ArrayGlyph._robust_limits(arr)
                >>> round(vmin, 2), round(vmax, 2)
                (0.08, 3.92)

                ```
        """
        if isinstance(arr, ma.MaskedArray):
            values = arr.compressed()
        else:
            values = np.asarray(arr).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(
                "Cannot compute robust vmin/vmax: array has no finite values."
            )
        vmin_robust = float(np.nanpercentile(values, ROBUST_LOWER_PERCENTILE))
        vmax_robust = float(np.nanpercentile(values, ROBUST_UPPER_PERCENTILE))
        return vmin_robust, vmax_robust

    @staticmethod
    def _center_limits(vmin: float, vmax: float, center: float) -> tuple[float, float]:
        """Make `(vmin, vmax)` symmetric around `center`.

        Implements xarray's diverging-cmap centring: the larger of
        `|vmin - center|` and `|vmax - center|` becomes the half-
        range, and the result is `(center - half, center + half)`.

        Args:
            vmin: Lower colour limit before symmetrisation.
            vmax: Upper colour limit before symmetrisation.
            center: Value to centre the diverging colormap on.

        Returns:
            tuple[float, float]: Symmetric `(vmin, vmax)` around
                `center`.

        Examples:
            - Centring around zero expands the smaller side to match
                the larger one:
                ```python
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> ArrayGlyph._center_limits(-3.0, 8.0, 0.0)
                (-8.0, 8.0)

                ```
            - Centring around a non-zero value (e.g. an anomaly base
                of 5.0):
                ```python
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> low, high = ArrayGlyph._center_limits(2.0, 12.0, 5.0)
                >>> low, high
                (-2.0, 12.0)
                >>> (low + high) / 2  # centred on 5.0
                5.0

                ```
        """
        half = max(abs(vmin - center), abs(vmax - center))
        return center - half, center + half

    def _resolve_color_limits(
        self,
        arr: np.ndarray,
        vmin_kw: float | None,
        vmax_kw: float | None,
        robust: bool,
        center: float | None,
        vmin_explicit: bool,
        vmax_explicit: bool,
    ) -> tuple[float, float]:
        """Resolve final `(vmin, vmax)` for colour scaling.

        Resolution order matches xarray:

        1. Start from robust (2nd/98th percentile) limits when
           `robust=True`, else from the full data range.
        2. Override either end with an explicit `vmin` / `vmax` if
           the user provided one.
        3. If `center` is set, symmetrise around it.

        Args:
            arr: Data array (plain or masked).
            vmin_kw: Value of `vmin` from the caller's kwargs, or
                `None` if not supplied.
            vmax_kw: Value of `vmax` from the caller's kwargs, or
                `None` if not supplied.
            robust: Whether to use the 2nd/98th percentile range.
            center: Value to centre a diverging colormap on. `None`
                disables symmetrisation.
            vmin_explicit: Whether the caller explicitly passed
                `vmin` (even if its value was `None`).
            vmax_explicit: Whether the caller explicitly passed
                `vmax`.

        Returns:
            tuple[float, float]: Final `(vmin, vmax)`.

        Raises:
            ValueError: If the resolved limits are not finite — e.g. the
                array has no finite values (all NaN / fully masked) and no
                explicit `vmin` / `vmax` was supplied to fall back on.

        Examples:
            - Default path: full data range, no robust clipping, no
                centring:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> data = np.arange(25, dtype=float).reshape(5, 5)
                >>> glyph = ArrayGlyph(data)
                >>> glyph._resolve_color_limits(
                ...     data,
                ...     vmin_kw=None,
                ...     vmax_kw=None,
                ...     robust=False,
                ...     center=None,
                ...     vmin_explicit=False,
                ...     vmax_explicit=False,
                ... )
                (0.0, 24.0)

                ```
            - Explicit `vmax` overrides the data-driven upper limit
                and `center` then symmetrises around the centre:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> data = np.arange(25, dtype=float).reshape(5, 5)
                >>> glyph = ArrayGlyph(data)
                >>> glyph._resolve_color_limits(
                ...     data,
                ...     vmin_kw=None,
                ...     vmax_kw=10.0,
                ...     robust=False,
                ...     center=0.0,
                ...     vmin_explicit=False,
                ...     vmax_explicit=True,
                ... )
                (-10.0, 10.0)

                ```
        """
        if robust:
            vmin_base, vmax_base = self._robust_limits(arr)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                vmin_base = np.nanmin(arr)
                vmax_base = np.nanmax(arr)

        vmin_final = vmin_kw if vmin_explicit and vmin_kw is not None else vmin_base
        vmax_final = vmax_kw if vmax_explicit and vmax_kw is not None else vmax_base

        if not (np.isfinite(vmin_final) and np.isfinite(vmax_final)):
            raise ValueError(
                "Cannot determine vmin/vmax: the array has no finite "
                "values. Pass explicit vmin and vmax, or filter the array "
                "first."
            )

        if center is not None:
            vmin_final, vmax_final = self._center_limits(vmin_final, vmax_final, center)

        return float(vmin_final), float(vmax_final)

    def _plot_im_get_cbar_kw(
        self,
        ax: Axes,
        arr: np.ndarray,
        ticks: np.ndarray,
        kind: str = "imshow",
    ) -> tuple[Any, dict[str, str]]:
        """Render the array on `ax` and return the artist plus cbar kwargs.

        Builds the matplotlib norm from `default_options["color_scale"]`
        and dispatches to the requested `kind` of plot. All four kinds
        share the same norm/vmin/vmax resolution path so the existing
        `color_scale` enum (linear/power/sym-lognorm/boundary-norm/
        midpoint) works identically for every render kind.

        When `self._coords` is set (curvilinear / non-uniform grid),
        the `(x, y)` arrays are forwarded as the first positional
        args to `pcolormesh` / `contour` / `contourf`. `kind="imshow"`
        is incompatible with `coords` and raises `ValueError` — callers
        should use `kind="auto"` or `kind="pcolormesh"` instead.

        Args:
            ax: matplotlib figure axes.
            arr: numpy (masked) array.
            ticks: color bar ticks.
            kind: render kind. One of `"imshow"`, `"pcolormesh"`,
                `"contour"`, `"contourf"`. Default is `"imshow"`
                (preserves the historical animate/legacy call path).

        Returns:
            tuple: `(artist, cbar_kw)` where `artist` is the
                matplotlib mappable (`AxesImage` for `imshow`,
                `QuadMesh` for `pcolormesh`, `QuadContourSet` for
                contour/contourf) and `cbar_kw` is the colorbar
                keyword-argument dict.

        Raises:
            ValueError: If `kind` is `"imshow"` while `self._coords`
                is set (incompatible combination), or if `kind` is not
                one of the recognised values in `VALID_PLOT_KINDS`.
        """
        norm, cbar_kw = self._create_norm_and_cbar_kw(ticks)
        cmap = resolve_colormap(self.default_options["cmap"])
        vmin = ticks[0]
        vmax = ticks[-1]

        self.contour_labels = None

        plot_arr = arr
        if self.default_options["color_scale"].lower() == "midpoint":
            plot_arr = ma.filled(arr, np.nan)

        levels = self.default_options.get("levels")

        coords = self._coords

        im: Any
        if kind == "imshow":
            if coords is not None:
                raise ValueError("`coords` requires kind='pcolormesh' or 'auto'.")
            if norm is None:
                im = ax.matshow(
                    plot_arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=self.extent
                )
            else:
                im = ax.matshow(plot_arr, cmap=cmap, norm=norm, extent=self.extent)
        elif kind == "pcolormesh":
            pcm_args = (
                (coords[0], coords[1], plot_arr) if coords is not None else (plot_arr,)
            )
            if norm is None:
                im = ax.pcolormesh(
                    *pcm_args,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    shading="auto",
                )
            else:
                im = ax.pcolormesh(*pcm_args, cmap=cmap, norm=norm, shading="auto")
        elif kind in ("contour", "contourf"):
            if isinstance(plot_arr, ma.MaskedArray):
                plot_arr = plot_arr.filled(np.nan)
            plot_fn = ax.contour if kind == "contour" else ax.contourf
            contour_kwargs = {"cmap": cmap}
            if norm is None:
                contour_kwargs["vmin"] = vmin
                contour_kwargs["vmax"] = vmax
            else:
                contour_kwargs["norm"] = norm
            level_edges = self._levels_to_bounds(levels, vmin, vmax)
            base_args = (
                (coords[0], coords[1], plot_arr) if coords is not None else (plot_arr,)
            )
            if level_edges is not None:
                im = plot_fn(*base_args, level_edges, **contour_kwargs)
            else:
                im = plot_fn(*base_args, **contour_kwargs)
            if kind == "contour" and self.default_options.get("labels"):
                label_kw = {
                    "inline": True,
                    "fontsize": 8,
                    "fmt": "%g",
                    **(self.default_options.get("label_kw") or {}),
                }
                self.contour_labels = ax.clabel(im, **label_kw)
        else:
            raise ValueError(
                f"Invalid kind={kind!r}. Valid kinds are {VALID_PLOT_KINDS}."
            )

        hillshade = resolve_hillshade(self.default_options.get("hillshade"))
        if hillshade is not None:
            if kind == "imshow":
                hs_norm = norm if norm is not None else Normalize(vmin=vmin, vmax=vmax)
                elevation = np.asarray(
                    ma.filled(ma.asarray(plot_arr).astype(float), np.nan), dtype=float
                )
                im.set_data(shade_grid(elevation, cmap, norm=hs_norm, **hillshade))
            else:
                warnings.warn(
                    f"hillshade is only applied to kind='imshow'; ignored for "
                    f"kind={kind!r}.",
                    stacklevel=2,
                )

        return im, cbar_kw

    @property
    def style(self) -> str | None:
        """Name of the `DATA_STYLES` preset currently applied, or `None`.

        Reads back the preset set via the `style` constructor kwarg, a
        `plot(style=...)` call, or `apply_style`.
        """
        return self.default_options.get("style")

    def apply_style(self, style: str, **kwargs: Any) -> tuple[Figure, Axes]:
        """Apply a `DATA_STYLES` preset by name, re-rendering the glyph in place.

        A discoverable wrapper over `plot(style=...)` for restyling an
        already-built glyph. It redraws **in place** on the glyph's own axes
        (clearing the previous render first), so `apply_style` takes full
        ownership of that axes -- do not use it on an axes shared with unrelated
        caller content. If the glyph was never plotted (or its figure was
        closed), it renders on a fresh figure. Extra keyword arguments (e.g.
        `hillshade`, `add_colorbar`) are forwarded to `plot`. The applied style
        is **sticky** (survives a later plain `plot()`); `plot(style=None)`
        clears it.

        Args:
            style: A `cleopatra.styling.colors.DATA_STYLES` preset name (see
                `sorted(cleopatra.styling.colors.DATA_STYLES)`).
            **kwargs: Forwarded to `plot` (e.g. `hillshade`).

        Returns:
            tuple[Figure, Axes]: The figure and axes drawn on.

        Raises:
            ValueError: If `style` is unknown or names a multi-layer preset
                (raised by `plot`).

        Examples:
            - Restyle a rendered glyph by name:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> glyph = ArrayGlyph(np.arange(60.0).reshape(6, 10))
                >>> _ = glyph.plot()
                >>> _ = glyph.apply_style("topography")
                >>> glyph.style
                'topography'

                ```
        """
        resolve_single_layer_style(style)
        self._reset_axes_for_restyle()
        return self.plot(style=style, ax=self.ax, **kwargs)

    def _resolve_style_layer(self, style: str) -> str:
        """Validate a `DATA_STYLES` name and return its single layer key.

        A raster band is one field, so only single-layer presets apply. The
        style name is the layer name for every single-layer preset
        (`flow_accumulation`, `flow_direction_d8`, topography, each Magics /
        cmocean entry).

        Args:
            style: A key of `cleopatra.styling.colors.DATA_STYLES`.

        Returns:
            The preset's single layer name.

        Raises:
            ValueError: If `style` is unknown, or names a multi-layer preset
                (which cannot be applied to a single raster band).
        """
        return resolve_single_layer_style(style)[0]

    def _style_cbar_kw(self, norm: Normalize) -> dict:
        """Colorbar tick kwargs for a real colorbar drawn over a `style` preset.

        A preset's norm is often a banded `BoundaryNorm` whose many raw
        boundaries would over-crowd the axis, so derive a clean, readable set of
        ~8 ticks across `norm`'s range for `create_color_bar` instead. Falls
        back to matplotlib's auto-ticking when the norm has no finite range.

        Args:
            norm: The preset's colour norm (carries the data-range vmin/vmax).

        Returns:
            dict: `{"ticks": [...]}`, or `{}` to let matplotlib auto-tick.
        """
        lo = getattr(norm, "vmin", None)
        hi = getattr(norm, "vmax", None)
        if lo is None or hi is None or lo == hi:
            return {}
        ticks = [
            float(t) for t in MaxNLocator(nbins=8).tick_values(lo, hi) if lo <= t <= hi
        ]
        return {"ticks": ticks} if ticks else {}

    def _apply_style_background(self, cfg: dict[str, Any]) -> None:
        """Paint the preset's canvas colour on this glyph's figure + axes.

        A preset whose look depends on a tinted canvas -- e.g. the flame glow,
        which fades to transparent at the cool end and only reads on black --
        carries a `background` colour (see the preset schema). Apply it to the
        axes (behind the data) and the figure patch (the crop margin, and the
        GIF background), scoped to this glyph, so no global `rcParams` mutation
        is needed. `savefig.facecolor='auto'` means a saved still or GIF inherits
        it too.

        Args:
            cfg: The resolved layer config; its `background` key, when present,
                is the canvas colour.
        """
        background = cfg.get("background")
        if background is None:
            return
        if self.ax is not None:
            self.ax.set_facecolor(background)
        if self.fig is not None and getattr(self, "_owns_figure", False):
            self.fig.patch.set_facecolor(background)

    def _plot_with_style(self, style: str) -> tuple[Figure, Axes]:
        """Render the array with a named `DATA_STYLES` preset.

        Delegates the drawing to `cleopatra.styling.colors.apply_data_style` so the
        preset's colormap, norm (`linear`/`log`/`symlog`/diverging `center`),
        transparent nodata, optional alpha glow, and — for categorical presets
        — the discrete `disjoint_legend` are reproduced exactly. The preset's
        swatch / categorical legend stands in for the colorbar, so `self.cbar`
        is left `None`. `add_colorbar=False` suppresses that legend.

        Args:
            style: A `DATA_STYLES` name (see `_resolve_style_layer`).

        Returns:
            tuple[Figure, Axes]: The figure and axes drawn on.
        """
        layer, style_cfg = resolve_single_layer_style(style)
        _clear_prior_render_artists(self.ax)
        self._apply_style_background(style_cfg)
        self._sync_projection_frame(
            projection_draws_frame(self.default_options.get("projection"))
        )
        data = np.asarray(
            ma.filled(ma.asarray(self.arr).astype(float), np.nan), dtype=float
        )
        legend = bool(self.default_options.get("add_colorbar", True))
        override_colorbar = (
            self._style_wants_colorbar and style_cfg.get("categories") is None
        )
        draw_swatch = legend and not override_colorbar
        self.im = self._render_styled_layer(layer, data, style, draw_swatch)

        self._compose_style_hillshade(style, data)
        self.cbar = (
            self._style_override_colorbar(data, style_cfg)
            if override_colorbar
            else None
        )
        if self.extent is None and self._coords is None:
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        self.ax.set_title(
            self.default_options["title"], fontsize=self.default_options["title_size"]
        )
        _mark_render_artists(self.ax, self.cbar, self.im)
        assert self.fig is not None
        return self.fig, self.ax

    def _flat_axis_bounds(self) -> tuple[float, float, float, float]:
        """Return the `(x_min, x_max, y_min, y_max)` axis limits of the flat view.

        Used both to reframe the axes when reverting from a projection (see
        `_sync_projection_frame`, which *sorts* the values) and to seed a fresh
        axes for the pre-plot basemap builder flow (see `GeoMixin._basemap_axes`,
        which applies them verbatim). From the lon/lat coords if present, else the
        `extent`, else the pixel grid -- and the pixel branch returns the *render*
        limits of `matshow(origin="upper")` (row 0 at the top, half-pixel cell
        edges), so its y is inverted (`y_min > y_max`); a raw `set_ylim` then
        matches the plain plot instead of flipping the raster upside-down.

        Returns:
            tuple[float, float, float, float]: The flat-render axis limits.
        """
        if self._coords is not None:
            x, y = self._coords
            return float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))
        if self.extent is not None:
            x0, x1, y0, y1 = self.extent
            return float(x0), float(x1), float(y0), float(y1)
        n_rows, n_cols = np.asarray(self.arr).shape[:2]
        return -0.5, float(n_cols) - 0.5, float(n_rows) - 0.5, -0.5

    def _sync_projection_frame(self, projecting: bool) -> None:
        """Strip a prior globe frame and, when reverting to flat, restore the view.

        `ArrayGlyph` reuses its own axes across `plot()` calls, and a globe render
        freezes the view / hides the axis (`apply_projection_frame`). So before a
        non-projection render on the same axes, the stale frame must be removed and
        the flat view restored, or the flat layer is drawn into a frozen, axis-off
        view as an invisible speck. A new globe render stashes its own frame in the
        projection render path, so here we only clear the prior frame; the view is
        restored only when this render is flat.

        Args:
            projecting: Whether this render is itself a projection (globe) render.
        """
        had_frame = _clear_projection_frame(self.ax)
        if had_frame and not projecting:
            x_min, x_max, y_min, y_max = self._flat_axis_bounds()
            _restore_flat_axes(self.ax, x_min, x_max, y_min, y_max, aspect="auto")

    def _render_styled_layer(
        self, layer: str, data: np.ndarray, style: str, draw_swatch: bool
    ) -> Any:
        """Draw the styled layer via `apply_data_style`; return its image artist.

        Forwards an explicit caller vmin/vmax/center (from `_style_color_overrides`,
        which holds ONLY user-supplied limits) so it overrides the preset's own
        fixed range, and colours the swatch legend to contrast with its box.
        """
        box = self.default_options.get("cbar_box")
        swatch_kw = {
            "legend": draw_swatch,
            "swatch_text_color": self.default_options.get("cbar_label_color")
            or _swatch_text_default(box),
            "swatch_value_color": self.default_options.get("cbar_tick_color")
            or _swatch_text_default(box),
            "swatch_box": box,
        }
        override = dict(self._style_color_overrides)
        coords = self._coords
        projection = self.default_options.get("projection")
        if projection:
            if coords is None or coords[0].ndim != 1 or coords[1].ndim != 1:
                raise ValueError(
                    "projection= with a style requires 1-D lon/lat coordinate "
                    "vectors (build the glyph with coords=(lon, lat))."
                )
            before = set(map(id, self.ax.patches)) | set(map(id, self.ax.lines))
            x_edges, y_edges, masked = apply_projection_style(
                self.ax, coords[0], coords[1], data, style=projection
            )
            _stash_projection_frame(
                self.ax,
                [a for a in (*self.ax.patches, *self.ax.lines) if id(a) not in before],
            )
            images = apply_data_style(
                self.ax, {layer: masked}, style=style, x=x_edges, y=y_edges,
                shading="flat", **swatch_kw, **override,
            )
        elif coords is not None:
            images = apply_data_style(
                self.ax, {layer: data}, style=style, x=coords[0], y=coords[1],
                shading="nearest", **swatch_kw, **override,
            )
        else:
            render_kwargs: dict[str, Any] = (
                {"extent": self.extent} if self.extent is not None else {}
            )
            images = apply_data_style(
                self.ax, {layer: data}, style=style, **swatch_kw,
                **render_kwargs, **override,
            )
        return images[layer]

    def _compose_style_hillshade(self, style: str, data: np.ndarray) -> None:
        """Blend terrain hillshade into a continuous-preset image (regular grid only).

        NOT applied to a categorical preset (shading nominal class colours is
        meaningless) nor to a curvilinear `QuadMesh` (no 2D RGBA grid to light);
        both cases warn and leave the preset as drawn.
        """
        hillshade = resolve_hillshade(self.default_options.get("hillshade"))
        if hillshade is None:
            return
        categorical = (
            resolve_single_layer_style(style)[1].get("categories") is not None
        )
        if categorical or self._coords is not None:
            kind = "categorical" if categorical else "curvilinear"
            warnings.warn(
                f"hillshade is not composed with a {kind} data-style preset; "
                "the preset is applied and hillshade ignored.",
                stacklevel=2,
            )
            return
        self.im.set_data(shade_rgb(self.im.get_array(), data, **hillshade))

    def _style_override_colorbar(self, data: np.ndarray, style_cfg: dict) -> Colorbar:
        """Build a real colorbar from the preset's cmap + norm (overriding the swatch).

        The drawn image bakes RGBA, so it cannot itself drive a colorbar; hand a
        `ScalarMappable` carrying the preset's colormap + norm to
        `create_color_bar`, which honours the `ColorBar` placement.
        """
        cbar_cfg = {**style_cfg, **self._style_color_overrides}
        cbar_norm, _lo, _hi = resolve_style_norm(data, cbar_cfg)
        mappable = ScalarMappable(norm=cbar_norm, cmap=resolve_colormap(cbar_cfg["cmap"]))
        mappable.set_array([])
        return self.create_color_bar(
            self.ax, mappable, self._style_cbar_kw(cbar_norm)
        )

    def apply_colormap(self, cmap: Colormap | str) -> np.ndarray:
        """Apply a matplotlib colormap to an array.

            Create an RGB channel from the given array using the given colormap.

        Args:
            cmap: colormap.

        Returns:
            np.ndarray: 8-bit array with the colormap applied.

        Examples:
        - Create an array and instantiate the `Array` object:
        ```python
        >>> import numpy as np
        >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> array = ArrayGlyph(arr)
        >>> rgb_array = array.apply_colormap("coolwarm_r")
        >>> print(rgb_array) # doctest: +SKIP
        [[[179   3  38]
          [221  96  76]
          [244 154 123]]
         [[244 196 173]
          [220 220 221]
          [183 207 249]]
         [[139 174 253]
          [ 96 128 232]
          [ 58  76 192]]]

        >>> print(rgb_array.dtype)
        uint8

        ```
        """
        colormap = resolve_colormap(cmap)
        normed_data = (self.arr - self.arr.min()) / (self.arr.max() - self.arr.min())
        colored = colormap(normed_data)
        return np.asarray((colored[:, :, :3] * 255).astype("uint8"))

    def to_image(self, arr: np.ndarray | None = None) -> Image.Image:
        """Create an RGB image from an array.

            convert the array to an image.

        Args:
            arr: array. if None, the array in the object will be used.

        Returns:
            PIL.Image.Image: An RGB image built from the array (values
                scaled to the 0-255 `uint8` range unless already `uint8`).

        Examples:
        ```python
        >>> import numpy as np
        >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> array = ArrayGlyph(arr)
        >>> image = array.to_image()
        >>> print(image) # doctest: +SKIP
        <PIL.Image.Image image mode=RGB size=3x3 at 0x7F5E0D2F4C40>

        ```
        """
        if arr is None:
            arr = self.arr
        arr = arr if arr.dtype == "uint8" else self.scale_to_rgb()
        return Image.fromarray(arr).convert("RGB")

    def scale_to_rgb(
        self,
        arr: np.ndarray | None = None,
        per_band: bool = False,
        percentile: tuple[float, float] = (2.0, 98.0),
    ) -> np.ndarray:
        """Scale an array to the 0-255 ``uint8`` range for RGB rendering.

        Two modes are available:

        - **Global (default, `per_band=False`):** scale the whole array by a
          single maximum (`arr * 255 / arr.max()`). Suitable for a single
          band or when all bands share a range.
        - **Per-band percentile stretch (`per_band=True`):** stretch each band
          (the last axis of a ``(rows, cols, bands)`` array) independently
          between its `percentile` low/high cut, clip to that range, and map
          to 0-255. This is the contrast stretch typically wanted for true
          RGB composites where bands have different dynamic ranges. A band
          with no usable range (all-NaN, or flat where the two cuts coincide)
          has nothing to stretch and is returned as a flat zero band.

        Args:
            arr: Array to scale. If None, the glyph's own array is used.
                For `per_band=True` it must be 3-D ``(rows, cols, bands)``.
            per_band: When True, stretch each band independently using
                `percentile`. When False (default), use the legacy single
                global-max scaling. Defaults to False.
            percentile: ``(low, high)`` percentile cuts for the per-band
                stretch, by default ``(2.0, 98.0)``. Ignored when
                `per_band` is False.

        Returns:
            np.ndarray: A ``uint8`` array of the same shape as the input,
                with values in 0-255. The input array is not modified.

        Raises:
            ValueError: If `per_band=True` and `arr` is not a 3-D
                ``(rows, cols, bands)`` array.

        Examples:
            - Global scaling of a single band (default):
                ```python
                >>> import numpy as np
                >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
                >>> array = ArrayGlyph(arr)
                >>> rgb_array = array.scale_to_rgb()
                >>> print(rgb_array)
                [[28 56 85]
                 [113 141 170]
                 [198 226 255]]
                >>> print(rgb_array.dtype)
                uint8

                ```
            - Per-band percentile stretch of a 3-band composite (each band
              spans the full 0-255 range independently):
                ```python
                >>> import numpy as np
                >>> rng = np.random.default_rng(0)
                >>> stack = rng.uniform(10, 200, size=(8, 8, 3))
                >>> array = ArrayGlyph(np.zeros((4, 4)))   # any 2-D placeholder
                >>> out = array.scale_to_rgb(stack, per_band=True)
                >>> out.shape, out.dtype
                ((8, 8, 3), dtype('uint8'))
                >>> int(out[..., 0].min()), int(out[..., 0].max())
                (0, 255)

                ```
        """
        if arr is None:
            arr = self.arr

        if per_band:
            arr = np.asarray(arr, dtype="float64")
            if arr.ndim != 3:
                raise ValueError(
                    "per_band=True requires a 3-D (rows, cols, bands) array; "
                    f"got {arr.ndim}-D shape {arr.shape}."
                )
            lo_p, hi_p = percentile
            out = np.empty(arr.shape, dtype="float64")
            for band in range(arr.shape[-1]):
                values = arr[..., band]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    lo, hi = np.nanpercentile(values, [lo_p, hi_p])
                if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                    out[..., band] = 0.0
                    continue
                out[..., band] = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
            out = np.nan_to_num(out, nan=0.0)
            return (out * 255).astype("uint8")

        denominator = arr.max() or 1
        return (arr * 255 / denominator).astype("uint8")

    @staticmethod
    def _plot_text(
        ax: Axes, arr: np.ndarray, indices, default_options_dict: dict
    ) -> list:
        """plot values as a text in each cell.

        Args:
            ax: matplotlib axes.
            arr: numpy array.
            indices: array with columns, (row, col).
            default_options_dict: default options dictionary after updating the options.

        Returns:
            list: list of the text object.
        """
        add_text = lambda elem: ax.text(
            elem[1],
            elem[0],
            np.round(arr[elem[0], elem[1]], 2),
            ha="center",
            va="center",
            color="w",
            fontsize=default_options_dict["num_size"],
        )
        return list(map(add_text, indices))

    def _apply_kwargs_and_colorbar(
        self, colorbar: bool | ColorBar | None, kwargs: dict
    ) -> dict:
        """Fold loose kwargs and `colorbar=` into `default_options`; set style flags.

        Shared by `plot` and `animate`: validates and applies each loose keyword
        into `default_options`, merges the resolved `colorbar=` spec last (so it
        wins over a same-named loose key), records this call's colour-limit
        overrides for a preset render, and sets `_style_wants_colorbar` -- a
        placement-bearing `colorbar=` (`location`, `inside`, `orientation`, or
        `True`) draws a real colorbar over a preset's swatch, while a spec
        carrying only colours/box styles the swatch in place.

        Args:
            colorbar: The `colorbar=` argument (`bool`, `ColorBar`, or `None`).
            kwargs: The remaining `plot` / `animate` keyword arguments.

        Returns:
            The resolved `colorbar` option dict, so the caller can honour a
            spec-provided `ticks_spacing` before auto-computing it.
        """
        for key, val in kwargs.items():
            if key not in self.default_options.keys():
                raise ValueError(
                    f"The given keyword argument:{key} is not correct, possible parameters are,"
                    f" {DEFAULT_OPTIONS}"
                )
            else:
                self.default_options[key] = val
        resolved_colorbar = _resolve_colorbar(colorbar)
        self.default_options.update(resolved_colorbar)
        for key in ("vmin", "vmax", "center", "cmap"):
            if key in kwargs and kwargs[key] is not None:
                self._style_color_overrides[key] = kwargs[key]
        self._style_wants_colorbar = colorbar is True or (
            isinstance(colorbar, ColorBar)
            and (
                colorbar.location is not None
                or colorbar.inside
                or colorbar.orientation is not None
            )
        )
        return resolved_colorbar

    def _plot_projected(
        self, ax: Axes, arr: np.ndarray, ticks: np.ndarray
    ) -> tuple[Any, dict[str, str]]:
        """Render the array through a projection preset (`"globe"` / `"flat"`).

        Reprojects the 1-D lon/lat field with
        `cleopatra.basemap.projection.apply_projection_style` (which also draws the globe
        boundary + graticule and masks the far hemisphere), then colours the
        reprojected cells with `pcolormesh(..., shading="flat")` at the projected
        cell **edges**. The colour norm/cmap come from the same resolution path
        as the flat render, so `color_scale` / `vmin` / `vmax` / `cmap` behave
        identically. The globe path needs `pyproj` (the `[tiles]` extra).

        Args:
            ax: Axes to draw on.
            arr: The (masked) 2-D data array.
            ticks: Colorbar ticks; drive `vmin`/`vmax` when the norm is linear.

        Returns:
            tuple: `(QuadMesh, cbar_kw)` -- the mappable and its colorbar kwargs.
        """
        projection = self.default_options["projection"]
        lon, lat = self._coords
        norm, cbar_kw = self._create_norm_and_cbar_kw(ticks)
        cmap = resolve_colormap(self.default_options["cmap"])
        plot_arr = (
            ma.filled(ma.asarray(arr).astype(float), np.nan)
            if isinstance(arr, ma.MaskedArray)
            else np.asarray(arr, dtype=float)
        )
        before = set(map(id, ax.patches)) | set(map(id, ax.lines))
        x_edges, y_edges, masked = apply_projection_style(
            ax, lon, lat, plot_arr, style=projection
        )
        _stash_projection_frame(
            ax, [a for a in (*ax.patches, *ax.lines) if id(a) not in before]
        )
        if norm is None:
            im = ax.pcolormesh(
                x_edges, y_edges, masked, cmap=cmap,
                vmin=ticks[0], vmax=ticks[-1], shading="flat",
            )
        else:
            im = ax.pcolormesh(
                x_edges, y_edges, masked, cmap=cmap, norm=norm, shading="flat"
            )
        return im, cbar_kw

    def plot(
        self,
        points: np.ndarray | PointOverlay | None = None,
        kind: str = "auto",
        ax: Axes | None = None,
        title: str | None = None,
        full_bleed: bool | str = False,
        basemap: bool | dict | Basemap | Callable[[Any], None] | None = None,
        colorbar: bool | ColorBar | None = None,
        **kwargs: Unpack[PlotKwargs],
    ) -> tuple[Figure, Axes]:
        """Plot the array with customizable visualization options.

        This method creates a visualization of the array with various customization options
        including color scales, color bars, cell value display, and point annotations.
        It supports both regular arrays and RGB arrays.

        Args:
            points: Points to display on the array, by default None. A
                `PointOverlay` (locations plus marker/label styling), or a
                plain `(N, 3)` array of `[value, row, col]` per point (a
                bare array is styled with `PointOverlay`'s own defaults).
                (Styling `points` via separate `point_color` /
                `point_size` / `point_label_color` / `point_label_size`
                keywords is deprecated; pass a `PointOverlay` instead —
                the old keywords still work as `**kwargs` and emit a
                `DeprecationWarning`.)
            kind: Render kind, by default `"auto"`. One of:

                - `"auto"` — picks the best renderer for the data.
                  Routes to `"pcolormesh"` when curvilinear /
                  non-uniform `coords` were passed to the
                  constructor, otherwise falls back to `"imshow"`.
                - `"imshow"` — pixel-grid raster render via
                  `ax.imshow`/`matshow`. Honours `extent`.
                  Incompatible with `coords`.
                - `"pcolormesh"` — quadrilateral mesh render via
                  `ax.pcolormesh` with `shading="auto"`. Honours
                  `coords` (1-D centres or 2-D curvilinear).
                - `"contour"` — line contours via `ax.contour`.
                  Honours `levels` from kwargs when set.
                - `"contourf"` — filled contours via `ax.contourf`.
                  Honours `levels` from kwargs when set.

                Cell-value display and point overlays only apply to
                `"imshow"` and `"pcolormesh"`; they are silently
                skipped for `"contour"` and `"contourf"` (which have
                no per-cell grid). RGB compositing requires
                `kind="imshow"`.
            ax: Target axes to draw on, by default None. When given,
                the plot is composed into this axes (and its parent
                figure, via `ax.get_figure()`), mirroring the other
                glyphs' `plot(ax=...)`. Resolution priority is
                `plot(ax=)` > the axes bound at construction > a fresh
                figure/axes. `fig` is intentionally not a parameter
                here — it is a construction-time binding derived from
                the axes.
            title: Plot title, by default None. A convenience shortcut
                equivalent to the `title` option; when given it
                overrides the `title` set at construction.
            full_bleed: Fill the whole figure edge-to-edge with no surrounding
                margin, by default False. `True` hides ticks and spines and
                resizes the figure to the data box's aspect so the fill has no
                distortion, leaving the canvas colour untouched (masked / no-data
                cells keep the default background). Pass a colour string instead
                (e.g. `"black"`) to also paint the canvas that colour -- e.g. so
                a semi-transparent relief reads dark. Same flag as
                `animate(full_bleed=...)`. Intended for chrome-free maps -- a
                colorbar or title has no room, so pair it with
                `add_colorbar=False` and omit the title (an outside colorbar is
                otherwise left floating over the filled axes); a scale swatch
                (from `style`) still fits inside. It resizes the whole figure and
                gives its axes the entire canvas, so use a dedicated figure --
                passing `ax=` one subplot of several lets `full_bleed` take over
                the figure and hide the siblings.
            basemap: A reference backdrop drawn via the glyph's own
                `add_relief` / `add_features`, composed by `zorder` (relief
                under the data, coastline/borders over it), by default None (no
                basemap). Accepts ``True`` for a sensible default (a `"low"`
                relief plus grey `"50m"` coastline and borders), a `Basemap`
                (the typed, validated form -- `relief` / `features` /
                `resolution` / `check_alignment`, with `features` taking
                `Feature` objects), a **dict** with the same keys (see
                `GeoMixin._draw_basemap`), or a **callable** ``f(glyph)`` for
                full control. Same flag as `animate(basemap=...)`. On a
                projected axis, set `self.crs` first so the relief is warped to
                match the data. Drawing the relief needs the `[tiles]` extra
                (Pillow, and pyproj for a non-4326 `crs`).
            colorbar: Colorbar presence and placement. `None` (default) keeps
                matplotlib's placement (honouring the legacy `add_colorbar`);
                `False` draws no colorbar; `True` a default one. Pass a
                `ColorBar` for control -- an edge (`location`), an `inside`
                inset that tracks `full_bleed`, a backing `box` (defaulted on
                for an inset), and text colours (`label_color` for the title,
                `tick_color` for the tick numbers). Same flag as `animate(colorbar=)`.
                On a `style=` preset, a placement `ColorBar` (or `True`) overrides
                the swatch with a real colorbar; a colours-only `ColorBar` styles
                the swatch in place (defaults < preset < explicit).
            **kwargs: Additional keyword arguments for customizing the plot.

                Plot appearance:
                    title : str, optional
                        Title of the plot, by default 'Array Plot'.
                    title_size : int, optional
                        Title font size, by default 15.
                    cmap : str or matplotlib.colors.Colormap, optional
                        Colormap, by default 'coolwarm_r'. A plain matplotlib
                        name (e.g. 'viridis') or a `Colormap` object is used
                        as-is; a **namespaced** name such as 'cmocean:thermal'
                        or 'crameri:batlow' is resolved via the optional `cmap`
                        aggregator — install the `[science-colors]` extra
                        (`pip install cleopatra[science-colors]`). The `_r`
                        reverse suffix works on both forms.
                    vmin : float, optional
                        Minimum value for color scaling, by default min(array).
                    vmax : float, optional
                        Maximum value for color scaling, by default max(array).

                Color bar options:
                    add_colorbar : bool, optional
                        Whether to draw the glyph's own color bar, by
                        default True. Set to False for shared-axes
                        composition, where the host owns a single
                        aggregated color bar; then `self.cbar` stays
                        None and no axes space is taken by a color bar.
                        The mappable is still reachable via `self.im`.
                        Note: for a constant-value field rendered as line
                        `contour` there are no contour lines to map, so the
                        color bar is skipped (with a warning) even when
                        `add_colorbar` is True, and `self.cbar` stays None.
                    cbar_orientation : str, optional
                        Deprecated; use `colorbar=ColorBar(orientation=...)`.
                        Orientation of the color bar, by default 'vertical'.
                        Can be 'horizontal' or 'vertical'.
                    cbar_label_rotation : float, optional
                        Deprecated; use `colorbar=ColorBar(label_rotation=...)`.
                        Rotation angle (degrees) of the color bar label, by
                        default None (matplotlib's own label orientation).
                    cbar_label_location : str, optional
                        Deprecated; use `colorbar=ColorBar(label_location=...)`.
                        Location of the color bar label, by default 'center'.
                        Valid values depend on the bar orientation -- vertical:
                        'top'/'center'/'bottom'; horizontal: 'left'/'center'/'right'.
                    cbar_length : float, optional
                        Deprecated; use `colorbar=ColorBar(length=...)`. Ratio to
                        control the height/width of the color bar, by default 0.75.
                    ticks_spacing : int, optional
                        Deprecated; use `colorbar=ColorBar(ticks_spacing=...)`.
                        Spacing between ticks on the color bar, by default 5.
                    cbar_label_size : int, optional
                        Deprecated; use `colorbar=ColorBar(label_size=...)`. Font
                        size of the color bar label, by default 12.
                    cbar_label : str, optional
                        Deprecated; use `colorbar=ColorBar(label=...)`. Label text
                        for the color bar, by default None.

                Color scale options:
                    color_scale : ColorScale or str, optional
                        Type of color scaling to use, by default 'linear'.
                        Accepts a `cleopatra.styling.styles.ColorScale`
                        member or its string value (case-insensitive). An
                        unrecognised value raises `ValueError`. Options:
                        - 'linear': Linear scale
                        - 'power': Power-law normalization
                        - 'sym-lognorm': Symmetrical logarithmic scale
                        - 'boundary-norm': Discrete intervals based on boundaries
                        - 'midpoint': Scale split at a specified midpoint
                    gamma : float, optional
                        Exponent for 'power' color scale, by default 0.5.
                        Values < 1 emphasize lower values, values > 1 emphasize higher values.
                    line_threshold : float, optional
                        Threshold for 'sym-lognorm' color scale, by default 0.0001.
                    line_scale : float, optional
                        Scale factor for 'sym-lognorm' color scale, by default 0.001.
                    bounds : list, optional
                        Boundaries for 'boundary-norm' color scale, by default None.
                        Defines the discrete intervals for color mapping.
                    midpoint : float, optional
                        Midpoint value for 'midpoint' color scale, by default 0.
                    levels : int or sequence, optional
                        Discrete colour levels (xarray-aligned), by
                        default None. An `int` selects N
                        linearly-spaced edges between `vmin` and
                        `vmax`; a sequence is used as explicit edges
                        (sorted ascending). When set under the default
                        `color_scale="linear"` the norm is switched
                        to a `BoundaryNorm` so `imshow` /
                        `pcolormesh` are also discretised; under
                        `color_scale="boundary-norm"` `levels` acts
                        as the bin edges when `bounds` is unset.
                        Always forwarded as the level array to
                        `contour` / `contourf`.

                Xarray-aligned colour kwargs:
                    robust : bool, optional
                        When True, use the 2nd and 98th percentile of
                        the unmasked data for `vmin` / `vmax`,
                        matching xarray's `robust=True` default. An
                        explicit `vmin` / `vmax` always wins. By
                        default False.
                    center : float, optional
                        Diverging-colormap centring value. When set,
                        `vmin` / `vmax` are made symmetric around
                        `center` (after `robust` has been applied),
                        and the cmap auto-switches to `"RdBu_r"` if
                        the caller did not pass an explicit `cmap`.
                        By default None (no centring).
                    extend : str, optional
                        Colorbar arrow extension. One of `"neither"`,
                        `"both"`, `"min"`, `"max"`, or None to
                        auto-resolve (`"both"` when `levels` is
                        set, otherwise `"neither"`). By default
                        None.
                    cbar_kwargs : dict, optional
                        Extra keyword arguments forwarded to
                        `fig.colorbar`. Merges over the defaults
                        computed by cleopatra so user keys win on
                        collision. Common keys: `label`, `shrink`,
                        `aspect`, `orientation`, `pad`,
                        `ticks`. By default None.

                Contour options:
                    labels : bool, optional
                        Draw inline numeric labels on the isolines of a
                        line `contour` (via `ax.clabel`), by default
                        False. Ignored for `kind="contourf"` and every
                        non-contour kind (filled contours have no lines
                        to label). The label `Text` artists are stored
                        on the instance as `self.contour_labels` (an
                        empty list when the contour has no isolines).
                    label_kw : dict, optional
                        Extra keyword arguments forwarded to
                        `ax.clabel` when `labels=True`, by default None.
                        Merges over cleopatra's defaults (`inline=True`,
                        `fontsize=8`, `fmt="%g"`) so user keys win on
                        collision. Common keys: `levels` (subset of
                        levels to label), `colors`, `fmt`, `fontsize`,
                        `inline_spacing`.

                Cell value display options:
                    display_cell_value : bool, optional
                        Whether to display the values of cells as text, by default False.
                    num_size : int, optional
                        Font size of the cell value text, by default 8.
                    background_color_threshold : float, optional
                        Threshold for cell value text color, by default None.
                        If cell value > threshold, text is black; otherwise, text is white.
                        If None, uses max(array)/2 as the threshold.

                Data-style preset / relief options:
                    style : str, optional
                        Name of a `cleopatra.styling.colors.DATA_STYLES` preset (e.g.
                        `"flow_accumulation"`, `"flow_direction_d8"`,
                        `"topography"`; valid names are
                        `sorted(cleopatra.styling.colors.DATA_STYLES)`). When set, the
                        preset's colormap, norm (linear/log/symlog/diverging
                        `center`), transparent nodata, alpha glow, and — for
                        categorical presets — a discrete legend are applied via
                        `cleopatra.styling.colors.apply_data_style`. Only single-layer
                        presets apply to a single band. The preset owns the
                        colour mapping, so it takes precedence over `cmap` /
                        `color_scale` / `vmin` / `vmax` / `center`, presents its
                        scale via a legend rather than a colorbar (`self.cbar`
                        is `None`), and bypasses `points` / `display_cell_value`
                        overlays (which warn). Ignored for RGB arrays. A
                        **continuous** preset composes with `hillshade` (the
                        relief is blended into the preset colours); a
                        **categorical** preset is not shaded (shading class
                        codes is meaningless) and warns. By default None.
                    hillshade : bool or dict, optional
                        Relief-shade a regular-grid DEM so wide-range terrain
                        reads by form. `True` uses defaults; a dict tunes
                        `vert_exag`, `azimuth`, `altitude`, `blend_mode`, or
                        `multidirectional`. Applied only to `kind="imshow"`
                        (warns otherwise). By default False.
                    projection : str, optional
                        Draw the field on a projection preset: `"globe"`
                        (orthographic) or `"flat"`. Requires 1-D lon/lat
                        `coords=(lon, lat)`; the field is reprojected and, for
                        `"globe"`, the boundary + graticule are drawn. `"globe"`
                        needs `pyproj` (the `[tiles]` extra). By default None
                        (unprojected raster).

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: A tuple containing:
                - fig: The matplotlib Figure object
                - ax: The matplotlib Axes object

            The colour-mapped artist (the `ScalarMappable` — e.g. the
            `AxesImage` for `imshow`, the `QuadMesh` for
            `pcolormesh`, the `QuadContourSet` for
            `contour`/`contourf`, or the RGB `AxesImage`) is also
            stored on the instance as `self.im` after this call, so a
            caller can attach a colorbar/legend or query the colour
            limits without scraping `ax.images`/`ax.collections`.

        Raises:
            ValueError: If an invalid keyword argument is provided.

        Notes:
            This method does not call `plt.show()`; it returns the Figure and Axes so
            the caller can compose, save, or display them. In an interactive session call
            `plt.show()` yourself (or `fig.savefig(...)` to write the plot to disk)
            after `plot()` returns.

        Examples:
        - Basic array plot:

            ```python
            >>> import numpy as np
            >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
            >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Customized Plot", title_size=18)
            >>> fig, ax = array.plot()

            ```
        ![array-plot](./../images/array_glyph/array-plot.png)

        - Labelled line contours (`kind="contour"`, `labels=True`):

            - inline numeric labels are drawn on the isolines and the
                label `Text` artists are kept on `glyph.contour_labels`:
                ```python
                >>> from matplotlib.text import Text
                >>> y, x = np.mgrid[-3:3:30j, -3:3:30j]
                >>> z = np.exp(-(x**2 + y**2))
                >>> glyph = ArrayGlyph(z, figsize=(6, 6))
                >>> fig, ax = glyph.plot(kind="contour", labels=True, label_kw={"fmt": "%.2f"})
                >>> bool(glyph.contour_labels) and all(
                ...     isinstance(t, Text) for t in glyph.contour_labels
                ... )
                True

                ```
                Without `labels` (the default) no labels are drawn and
                `contour_labels` stays `None`:
                ```python
                >>> glyph = ArrayGlyph(z, figsize=(6, 6))
                >>> fig, ax = glyph.plot(kind="contour")
                >>> glyph.contour_labels is None
                True

                ```

        - Color bar customization:

            - Create an array and instantiate the `Array` object with custom options.
                ```python
                >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Customized color bar", title_size=18)
                >>> fig, ax = array.plot(
                ...     colorbar=ColorBar(
                ...         label="Discharge m3/s",
                ...         label_location="center",
                ...         length=0.7,
                ...         label_size=12,
                ...         ticks_spacing=5,
                ...         orientation="horizontal",
                ...     ),
                ...     color_scale="linear",
                ...     cmap="coolwarm_r",
                ... )

                ```
                ![color-bar-customization](./../images/array_glyph/color-bar-customization.png)

        - Display values for each cell:

            - you can display the values for each cell by using thr parameter `display_cell_value`, and customize how
                the values are displayed using the parameter `background_color_threshold` and `num_size`.

                ```python
                >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Display array values", title_size=18)
                >>> fig, ax = array.plot(
                ...     display_cell_value=True,
                ...     num_size=12
                ... )

                ```
                ![display-cell-values](./../images/array_glyph/display-cell-values.png)

        - Plot points at specific locations in the array:

            - you can display points in specific cells in the array and also display a value for each of these points.
                The point overlay's array has the first column as the values to be displayed on top of the
                points, the second and third columns are the row and column index of the point in the array.
            - A `PointOverlay`'s `color`/`size` customize the appearance of the points, while `label_color`/
                `label_size` customize the appearance of each point's value label.

                ```python
                >>> from cleopatra.glyphs.gridded.array_glyph import PointOverlay
                >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Display Points", title_size=14)
                >>> points = np.array([[1, 0, 0], [2, 1, 1], [3, 2, 2]])
                >>> overlay = PointOverlay(
                ...     points,
                ...     color="black",
                ...     size=100,
                ...     label_color="orange",
                ...     label_size=30,
                ... )
                >>> fig, ax = array.plot(points=overlay)

                ```
                ![display-points](./../images/array_glyph/display-points.png)

        - Color scale customization:

            - Power scale (with different gamma values).

                - The default power scale uses a gamma value of 0.5.

                    ```python
                    >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Power scale", title_size=18)
                    >>> fig, ax = array.plot(
                    ...     colorbar=ColorBar(label="Discharge m3/s"),
                    ...     color_scale="power",
                    ...     cmap="coolwarm_r",
                    ... )

                    ```
                    ![power-scale](./../images/array_glyph/power-scale.png)

                - change the gamma of 0.8 (emphasizes higher values less).

                    ```python
                    >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Power scale - gamma=0.8", title_size=18)
                    >>> fig, ax = array.plot(
                    ...     color_scale="power",
                    ...     gamma=0.8,
                    ...     cmap="coolwarm_r",
                    ...     colorbar=ColorBar(label="Discharge m3/s"),
                    ... )

                    ```
                    ![power-scale-gamma-0.8](./../images/array_glyph/power-scale-gamma-0.8.png)

                - change the gamma of 0.1 (emphasizes higher values more).

                    ```python
                    >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Power scale - gamma=0.1", title_size=18)
                    >>> fig, ax = array.plot(
                    ...     color_scale="power",
                    ...     gamma=0.1,
                    ...     cmap="coolwarm_r",
                    ...     colorbar=ColorBar(label="Discharge m3/s"),
                    ... )

                    ```
                    ![power-scale-gamma-0.1](./../images/array_glyph/power-scale-gamma-0.1.png)

            - Logarithmic scale.

                - the logarithmic scale uses to parameters `line_threshold` and `line_scale` with a default
                value if 0.0001, and 0.001 respectively.
                    ```python
                    >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Logarithmic scale", title_size=18)
                    >>> fig, ax = array.plot(
                    ...     colorbar=ColorBar(label="Discharge m3/s"),
                    ...     color_scale="sym-lognorm",
                    ...     cmap="coolwarm_r",
                    ... )

                    ```
                    ![log-scale](./../images/array_glyph/log-scale.png)

                - you can change the `line_threshold` and `line_scale` values.
                    ```python
                    >>> array = ArrayGlyph(
                    ...     arr, figsize=(6, 6), title="Logarithmic scale: Customized Parameter", title_size=12
                    ... )
                    >>> fig, ax = array.plot(
                    ...     colorbar=ColorBar(label="Discharge m3/s"),
                    ...     color_scale="sym-lognorm",
                    ...     cmap="coolwarm_r",
                    ...     line_threshold=0.015,
                    ...     line_scale=0.1,
                    ... )

                    ```
                    ![log-scale](./../images/array_glyph/log-scale-custom-parameters.png)

            - Defined boundary scale.
                ```python
                >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Defined boundary scale", title_size=18)
                >>> fig, ax = array.plot(
                ...     colorbar=ColorBar(label="Discharge m3/s"),
                ...     color_scale="boundary-norm",
                ...     cmap="coolwarm_r",
                ... )

                ```
                ![boundary-scale](./../images/array_glyph/boundary-scale.png)

                - You can also define the boundaries.
                    ```python
                    >>> array = ArrayGlyph(
                    ...     arr, figsize=(6, 6), title="Defined boundary scale: defined bounds", title_size=18
                    ... )
                    >>> bounds = [0, 5, 10]
                    >>> fig, ax = array.plot(
                    ...     colorbar=ColorBar(label="Discharge m3/s"),
                    ...     color_scale="boundary-norm",
                    ...     bounds=bounds,
                    ...     cmap="coolwarm_r",
                    ... )

                    ```
                    ![boundary-scale-defined-bounds](./../images/array_glyph/boundary-scale-defined-bounds.png)

            - Midpoint scale.

                in the midpoint scale you can define a value that splits the scale into half.
                ```python
                >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Midpoint scale", title_size=18)
                >>> fig, ax = array.plot(
                ...     colorbar=ColorBar(label="Discharge m3/s"),
                ...     color_scale="midpoint",
                ...     cmap="coolwarm_r",
                ...     midpoint=2,
                ... )

                ```
                ![midpoint-scale-costom-parameters](./../images/array_glyph/midpoint-scale-costom-parameters.png)

        - Render kinds (`kind=`):

            - `"pcolormesh"` for a quadrilateral mesh render. Note
                that `pcolormesh` does not honour `extent`, so the
                axes are drawn in array index space.
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> arr = np.arange(25, dtype=float).reshape(5, 5)
                >>> glyph = ArrayGlyph(arr)
                >>> fig, ax = glyph.plot(kind="pcolormesh")  # doctest: +SKIP

                ```
            - `"contourf"` for filled contours. When `levels` is set
                the level edges line up with the colorbar boundaries.
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> arr = np.arange(25, dtype=float).reshape(5, 5)
                >>> glyph = ArrayGlyph(arr, levels=5)
                >>> fig, ax = glyph.plot(kind="contourf")  # doctest: +SKIP

                ```
            - Invalid kinds are rejected with a clear error:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> arr = np.arange(9, dtype=float).reshape(3, 3)
                >>> ArrayGlyph(arr).plot(kind="heatmap")
                Traceback (most recent call last):
                    ...
                ValueError: Invalid kind='heatmap'. Valid kinds are ('auto', 'imshow', 'pcolormesh', 'contour', 'contourf').

                ```

        - xarray-aligned colour kwargs:

            - `robust=True` clips `vmin` / `vmax` to the
                2nd/98th percentile so a single outlier no longer
                dominates the colour scale:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> data = np.arange(100, dtype=float).reshape(10, 10)
                >>> data[0, 0] = 1e6  # outlier
                >>> glyph = ArrayGlyph(data, robust=True)
                >>> fig, ax = glyph.plot(robust=True)  # doctest: +SKIP
                >>> round(glyph.vmin, 1), round(glyph.vmax, 1)
                (3.0, 98.0)

                ```
            - `center=0` symmetrises the limits around zero and
                auto-switches the cmap to `"RdBu_r"` (xarray-style
                diverging default):
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> anomaly = np.linspace(-3.0, 8.0, 25).reshape(5, 5)
                >>> glyph = ArrayGlyph(anomaly, center=0.0)
                >>> fig, ax = glyph.plot(center=0.0)  # doctest: +SKIP
                >>> glyph.vmin, glyph.vmax
                (-8.0, 8.0)
                >>> glyph.default_options["cmap"]
                'RdBu_r'

                ```
            - `levels` discretises the colour scale and `extend`
                controls the colorbar arrows:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> arr = np.arange(25, dtype=float).reshape(5, 5)
                >>> glyph = ArrayGlyph(arr, levels=6, extend="both")
                >>> fig, ax = glyph.plot()  # doctest: +SKIP
                >>> glyph.default_options["levels"], glyph.default_options["extend"]
                (6, 'both')

                ```
            - `cbar_kwargs` forwards extra keyword arguments to the
                underlying `matplotlib.pyplot.colorbar` call;
                user keys win on collision:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> arr = np.arange(9, dtype=float).reshape(3, 3)
                >>> glyph = ArrayGlyph(arr, cbar_kwargs={"shrink": 0.5})
                >>> fig, ax = glyph.plot()  # doctest: +SKIP
                >>> glyph.default_options["cbar_kwargs"]
                {'shrink': 0.5}

                ```
        """
        if kind not in VALID_PLOT_KINDS:
            raise ValueError(
                f"Invalid kind={kind!r}. Valid kinds are {VALID_PLOT_KINDS}."
            )
        if self.rgb and kind not in ("imshow", "auto"):
            raise ValueError(
                f"RGB compositing requires kind='imshow'. Got kind={kind!r}."
            )

        points = _resolve_point_overlay(points, kwargs)  # type: ignore[arg-type]
        _warn_deprecated_cbar_kwargs(kwargs)

        resolved_colorbar = self._apply_kwargs_and_colorbar(colorbar, kwargs)  # type: ignore[arg-type]

        self._validate_extend(self.default_options.get("extend"))

        self.default_options["kind"] = kind
        if kind == "auto":
            effective_kind = "pcolormesh" if self._coords is not None else "imshow"
        else:
            effective_kind = kind

        if ax is not None:
            self.ax = ax
            self.fig = _root_figure(ax)
            self._auto_figure = False
            self._owns_figure = False
        elif self.fig is None:
            self.fig, self.ax = self.create_figure_axes()

        if title is not None:
            self.default_options["title"] = title

        arr = self.arr
        fig, ax = self.fig, self.ax

        style = self.default_options.get("style")
        if style is not None:
            try:
                resolve_single_layer_style(style)
            except ValueError:
                self.default_options["style"] = None
                raise
            if self.rgb:
                warnings.warn(
                    "data-style presets do not apply to RGB arrays; 'style' is "
                    "ignored and the RGB image is drawn as-is.",
                    stacklevel=2,
                )
            else:
                if points is not None or self.default_options.get("display_cell_value"):
                    warnings.warn(
                        "data-style presets bypass point and cell-value overlays; "
                        "'points' and 'display_cell_value' are ignored with 'style'.",
                        stacklevel=2,
                    )
                self._plot_with_style(style)
                if basemap is not None:
                    self._draw_basemap(basemap)
                if full_bleed:
                    self._apply_full_bleed(facecolor=full_bleed if isinstance(full_bleed, str) else None)
                elif getattr(self, "_auto_figure", False):
                    self._tighten_figure()
                return self.fig, self.ax

        if self.rgb:
            _clear_prior_render_artists(ax)
            extent = tuple(self.extent) if self.extent is not None else None
            self.im = ax.imshow(arr, extent=extent)
            self.cbar = None
        else:
            if "ticks_spacing" not in resolved_colorbar:
                if "ticks_spacing" in kwargs.keys():
                    self.default_options["ticks_spacing"] = kwargs["ticks_spacing"]
                else:
                    self.default_options["ticks_spacing"] = self.ticks_spacing

            recompute_keys = {"robust", "center", "vmin", "vmax"}
            if recompute_keys.intersection(kwargs.keys()):
                vmin_final, vmax_final = self._resolve_color_limits(
                    arr,
                    vmin_kw=kwargs.get("vmin"),
                    vmax_kw=kwargs.get("vmax"),
                    robust=bool(self.default_options.get("robust", False)),
                    center=self.default_options.get("center"),
                    vmin_explicit="vmin" in kwargs,
                    vmax_explicit="vmax" in kwargs,
                )
                self._vmin = vmin_final
                self._vmax = vmax_final
                if "ticks_spacing" not in kwargs and "ticks_spacing" not in resolved_colorbar:
                    self.ticks_spacing = (vmax_final - vmin_final) / 10 or 1.0
                    self.default_options["ticks_spacing"] = self.ticks_spacing

            if (
                "center" in kwargs
                and kwargs["center"] is not None
                and "cmap" not in kwargs
            ):
                self.default_options["cmap"] = DIVERGING_DEFAULT_CMAP

            self.default_options["vmin"] = self.vmin
            self.default_options["vmax"] = self.vmax

            ticks = self.get_ticks()
            self._create_norm_and_cbar_kw(ticks)
            projection = self.default_options.get("projection")
            if projection and (
                self._coords is None
                or self._coords[0].ndim != 1
                or self._coords[1].ndim != 1
            ):
                raise ValueError(
                    "projection= requires 1-D lon/lat coordinate vectors (build "
                    "the glyph with coords=(lon, lat)); an extent-only or "
                    "2-D-coordinate array cannot be reprojected."
                )
            _clear_prior_render_artists(ax)
            self._sync_projection_frame(projection_draws_frame(projection))
            if projection:
                if points is not None or self.default_options.get("display_cell_value"):
                    warnings.warn(
                        "'projection' draws point / cell-value overlays at raw grid "
                        "indices, not reprojected coordinates, so they are misplaced "
                        "under a projection; omit them when using 'projection'.",
                        stacklevel=2,
                    )
                if kind not in ("auto", "pcolormesh") or self.default_options.get(
                    "hillshade"
                ):
                    warnings.warn(
                        "'projection' always renders via pcolormesh and ignores "
                        "'kind' and 'hillshade'.",
                        stacklevel=2,
                    )
                im, cbar_kw = self._plot_projected(ax, arr, ticks)
            else:
                im, cbar_kw = self._plot_im_get_cbar_kw(
                    ax, arr, ticks, kind=effective_kind
                )
            self.im = im

            self.cbar = None
            degenerate_contour = (
                effective_kind == "contour" and self._vmax == self._vmin
            )
            if self.default_options["add_colorbar"]:
                if degenerate_contour:
                    warnings.warn(
                        "Constant-value field has no contour lines; skipping "
                        "the colorbar for kind='contour'.",
                        stacklevel=2,
                    )
                else:
                    self.cbar = self.create_color_bar(ax, im, cbar_kw)

        ax.set_title(
            self.default_options["title"], fontsize=self.default_options["title_size"]
        )

        if self.extent is None and effective_kind == "imshow":
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

        supports_overlay = effective_kind in ("imshow", "pcolormesh")
        optional_display: dict[str, Any] = {}
        if self.default_options["display_cell_value"] and supports_overlay:
            indices = get_indices2(arr, [np.nan])
            optional_display["cell_text_value"] = self._plot_text(
                ax, arr, indices, self.default_options
            )

        if points is not None and supports_overlay:
            row = points.points[:, 1]
            col = points.points[:, 2]
            optional_display["points_scatter"] = ax.scatter(
                col, row, color=points.color, s=points.size
            )
            optional_display["points_id"] = self._plot_point_values(
                ax, points.points, points.label_color, points.label_size
            )

        _mark_render_artists(
            ax,
            self.cbar,
            self.im,
            optional_display.get("points_scatter"),
            *(optional_display.get("points_id") or []),
            *(optional_display.get("cell_text_value") or []),
        )
        if basemap is not None:
            self._draw_basemap(basemap)
        if full_bleed:
            self._apply_full_bleed(facecolor=full_bleed if isinstance(full_bleed, str) else None)
        elif getattr(self, "_auto_figure", False):
            self._tighten_figure()
        return fig, ax

    def facet(
        self,
        *,
        col: str | None = None,
        row: str | None = None,
        col_wrap: int | None = None,
        col_coords: Sequence[Any] | None = None,
        row_coords: Sequence[Any] | None = None,
        kind: str = "auto",
        figsize: tuple[float, float] | None = None,
        extents: Sequence[Sequence[float]] | None = None,
        colorbar: bool | ColorBar | None = None,
        **kwargs,
    ) -> FacetGrid:
        """Render a grid of subplots from a 3-D or 4-D stack.

        Mirrors xarray's `xarray.plot.facetgrid.FacetGrid` API.
        `self.arr` must be 3-D `(N, H, W)` when only `col` is set,
        or 4-D `(N, M, H, W)` when both `col` and `row` are set.
        All subplots share a common colour scale (`vmin`/`vmax`
        computed over the full stack unless the user passed explicit
        limits); each panel draws its own colour bar on that shared
        scale, and `result.cbar` exposes the first panel's. Pass
        `colorbar=` to configure that bar or suppress it (see below).

        Spatial extent: every panel is a slice of the *same* array, so by
        default they all share the parent glyph's `extent` (one spatial
        domain — exactly like xarray's `FacetGrid`, which facets a single
        `DataArray` over a coordinate dimension). If your slices are
        same-shape grids covering *different* windows, pass `extents` —
        one `[xmin, ymin, xmax, ymax]` per panel. (If the slices are
        genuinely different datasets, build separate `ArrayGlyph`
        instances into your own `plt.subplots` grid instead.)

        Args:
            col: Name of the column-facet dimension (e.g. `"time"`).
                Used as a label in the per-subplot title and in
                `FacetGrid.name_dicts`. Required when `row` is
                not given.
            row: Name of the row-facet dimension (e.g. `"level"`).
                Required when faceting a 4-D stack.
            col_wrap: When only `col` is given, wrap the N subplots
                into `col_wrap` columns × `ceil(N/col_wrap)` rows.
                Ignored when `row` is set.
            col_coords: Optional sequence of coordinate labels for the
                column dimension. Length must match the column axis of
                the stack. When given, the per-subplot title contains
                the coord value instead of the integer index.
            row_coords: Optional sequence of coordinate labels for the
                row dimension. Length must match the row axis of the
                stack. Only honoured when `row` is set.
            kind: Render kind, forwarded to the per-subplot dispatch.
                One of `"auto"`, `"imshow"`, `"pcolormesh"`,
                `"contour"`, `"contourf"`. Default `"auto"`.
            figsize: Optional `(width, height)` for the shared figure.
                Defaults to `(4 * ncols, 3.5 * nrows)`.
            extents: Optional per-panel spatial extents — one
                `[xmin, ymin, xmax, ymax]` (user-facing order) for each
                rendered subplot, in row-major order (`extents[k]`
                applies to `result.axes.flat[k]`). Length must equal the
                number of panels. Mutually exclusive with the parent
                glyph's `extent` and with `coords`. `None` (default)
                reuses the parent's `extent` on every panel (or index
                space when the parent has none).
            colorbar: The shared colour bar, mirroring `plot` / `animate`.
                `None` (default) keeps each panel's default colour legend --
                a colour bar, or a preset style's swatch -- (the prior
                behaviour); `False` suppresses them (`result.cbar` is then
                `None`);
                `True` draws default ones, resetting the resettable `cbar_*`
                family to defaults so they do not inherit a prior sticky
                spec; a `ColorBar` applies its
                placement / caption / sizing to every panel (so the
                `result.cbar` returned -- the first panel's -- carries the
                spec). Prefer this typed form over the loose `cbar_*`
                kwargs, which are deprecated here as they are on
                `plot` / `animate`.

            **kwargs: Forwarded to each subplot. Recognised keys
                include the same colour / colorbar / level kwargs as
                `plot`. `vmin` / `vmax` win over the
                stack-wide auto-computed limits. Passing the loose
                `cbar_*` colorbar kwargs is deprecated -- use `colorbar`.

        Returns:
            FacetGrid: Result object exposing `fig`, `axes`,
                `cbar`, and `name_dicts`.

        Raises:
            ValueError: If neither `col` nor `row` is given, if the
                array shape does not match the requested facet
                dimensions, if `col_coords` / `row_coords` lengths
                are wrong, if `extents` is combined with the parent's
                `extent` or `coords`, or if `extents` has the wrong
                length or a non-length-4 element.

        Examples:
            - Facet a 3-D stack into a 1xN row of subplots:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> stack = np.arange(4 * 5 * 5, dtype=float).reshape(4, 5, 5)
                >>> g = ArrayGlyph(stack).facet(col="t")
                >>> g.axes.shape
                (1, 4)
                >>> g.name_dicts[0]
                {'t': 0}

                ```
            - Wrap N=6 panels into a 2x3 grid with `col_wrap=3`:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> stack = np.arange(6 * 5 * 5, dtype=float).reshape(6, 5, 5)
                >>> g = ArrayGlyph(stack).facet(col="t", col_wrap=3)
                >>> g.axes.shape
                (2, 3)

                ```
            - Per-panel extents for same-shape grids over different
                windows (one `[xmin, ymin, xmax, ymax]` per subplot):
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> stack = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
                >>> g = ArrayGlyph(stack).facet(
                ...     col="region",
                ...     extents=[[0, 0, 10, 10], [10, 0, 20, 10]],
                ... )
                >>> [tuple(int(v) for v in im.get_extent()) for im in
                ...  (ax.get_images()[0] for ax in g.axes.flat)]
                [(0, 10, 0, 10), (10, 20, 0, 10)]

                ```
            - Configure the shared colour bar with a typed `ColorBar`:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> from cleopatra.styling.colorbar import ColorBar
                >>> stack = np.arange(3 * 5 * 5, dtype=float).reshape(3, 5, 5)
                >>> g = ArrayGlyph(stack).facet(col="t", colorbar=ColorBar(label="mm"))
                >>> g.cbar.ax.get_ylabel()
                'mm'

                ```
        """
        if col is None and row is None:
            raise ValueError("at least one of `col`/`row` must be given")
        if extents is not None:
            if self.extent is not None:
                raise ValueError(
                    "`extents` (per-panel) and the glyph's `extent` "
                    "(one shared domain) are mutually exclusive."
                )
            if self._coords is not None:
                raise ValueError("`extents` and `coords` are mutually exclusive.")
            for k, e in enumerate(extents):
                if len(e) != 4:
                    raise ValueError(
                        f"`extents[{k}]` must be a length-4 sequence "
                        f"[xmin, ymin, xmax, ymax], got {e!r}."
                    )

        arr = self.arr
        if row is None:
            if arr.ndim != 3:
                raise ValueError(
                    "Faceting on `col` alone requires a 3-D array "
                    f"(N, H, W); got shape {arr.shape}."
                )
            n_col = arr.shape[0]
            if col_wrap is not None:
                if not isinstance(col_wrap, (int, np.integer)) or col_wrap < 1:
                    raise ValueError(
                        f"`col_wrap` must be a positive int, got {col_wrap!r}."
                    )
                ncols = int(col_wrap)
                nrows = int(ceil(n_col / ncols))
            else:
                ncols = n_col
                nrows = 1
            if col_coords is not None and len(col_coords) != n_col:
                raise ValueError(
                    f"`col_coords` length {len(col_coords)} does not match "
                    f"the column axis size {n_col}."
                )
            panel_indices: list[tuple[int, int | None]] = [
                (i, None) for i in range(n_col)
            ]
            n_panels = n_col
        else:
            if col is None:
                raise ValueError("Faceting on `row` requires `col` as well.")
            if arr.ndim != 4:
                raise ValueError(
                    "Faceting on `row`+`col` requires a 4-D array "
                    f"(Ncol, Nrow, H, W); got shape {arr.shape}."
                )
            n_col, n_row = arr.shape[0], arr.shape[1]
            ncols = n_col
            nrows = n_row
            if col_coords is not None and len(col_coords) != n_col:
                raise ValueError(
                    f"`col_coords` length {len(col_coords)} does not match "
                    f"the column axis size {n_col}."
                )
            if row_coords is not None and len(row_coords) != n_row:
                raise ValueError(
                    f"`row_coords` length {len(row_coords)} does not match "
                    f"the row axis size {n_row}."
                )
            panel_indices = [(i, j) for j in range(n_row) for i in range(n_col)]
            n_panels = n_col * n_row

        if extents is not None and len(extents) != n_panels:
            raise ValueError(
                f"`extents` has {len(extents)} entries but there are {n_panels} panels."
            )

        assert col is not None

        # Warn only after the structural validation above, so a malformed call
        # raises its `ValueError` without a spurious colorbar DeprecationWarning.
        _warn_deprecated_cbar_kwargs(kwargs)

        if figsize is None:
            figsize = (4.0 * ncols, 3.5 * nrows)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False
        )

        vmin_user = kwargs.get("vmin")
        vmax_user = kwargs.get("vmax")
        if vmin_user is None or vmax_user is None:
            if isinstance(arr, ma.MaskedArray):
                finite = arr.compressed()
            else:
                finite = np.asarray(arr).ravel()
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                stack_min = 0.0
                stack_max = 1.0
            else:
                stack_min = float(finite.min())
                stack_max = float(finite.max())
            shared_vmin = stack_min if vmin_user is None else float(vmin_user)
            shared_vmax = stack_max if vmax_user is None else float(vmax_user)
        else:
            shared_vmin = float(vmin_user)
            shared_vmax = float(vmax_user)

        per_subplot_kwargs = dict(kwargs)
        per_subplot_kwargs["vmin"] = shared_vmin
        per_subplot_kwargs["vmax"] = shared_vmax

        name_dicts: list[dict[str, Any]] = []
        cbar: Colorbar | None = None
        flat_axes = axes.ravel()

        for panel_idx, (col_idx, row_idx) in enumerate(panel_indices):
            ax = flat_axes[panel_idx]
            if row is None:
                panel_arr = arr[col_idx]
            else:
                panel_arr = arr[col_idx, row_idx]

            if extents is not None:
                sub_extent = list(extents[panel_idx])
            elif self.extent is None:
                sub_extent = None
            else:
                sub_extent = [
                    self.extent[0],  # xmin
                    self.extent[2],  # ymin
                    self.extent[1],  # xmax
                    self.extent[3],  # ymax
                ]
            sub = ArrayGlyph(
                panel_arr,
                coords=self._coords,
                extent=sub_extent,
                fig=fig,
                ax=ax,
                **per_subplot_kwargs,
            )
            # Route `colorbar=` through `plot` (not the constructor) so the
            # shared `_apply_kwargs_and_colorbar` logic runs per panel -- it
            # merges the resolved spec *over* any loose `cbar_*` already folded
            # into the sub-glyph's options and sets `_style_wants_colorbar`, so
            # a placement-bearing colorbar overrides a preset swatch here just
            # as it does on `plot` / `animate`.
            sub.plot(kind=kind, colorbar=colorbar)

            col_label = col_coords[col_idx] if col_coords is not None else col_idx
            name_dict: dict[str, Any] = {col: col_label}
            if row is not None:
                assert row_idx is not None
                row_label = row_coords[row_idx] if row_coords is not None else row_idx
                name_dict[row] = row_label
                title = f"{col}={col_label}, {row}={row_label}"
            else:
                title = f"{col}={col_label}"
            ax.set_title(title)
            name_dicts.append(name_dict)

            if panel_idx == 0 and getattr(sub, "cbar", None) is not None:
                cbar = sub.cbar

        for hidden_idx in range(n_panels, nrows * ncols):
            flat_axes[hidden_idx].set_visible(False)

        fig.tight_layout()
        result = FacetGrid(fig=fig, axes=axes, cbar=cbar, name_dicts=name_dicts)
        return result

    def _apply_full_bleed(self, facecolor: str | None = None) -> None:
        """Give the axes the whole figure, chrome-free (for `full_bleed=...`).

        Hides ticks and spines, resizes the figure so its aspect matches the
        georeferenced data box (from `extent`) so the map fills the frame
        without distortion, then hands the axes the entire figure area
        (`set_position([0, 0, 1, 1])`, `aspect="auto"`). Without an `extent` the
        aspect is unknown, so the axes still fills but may stretch. The caller
        skips its `tight_layout` when this runs.

        The canvas colour is left untouched unless `facecolor` is given -- so
        masked / no-data cells keep the default background, not black. Pass a
        `facecolor` (e.g. `"black"`) only when the backdrop should be painted,
        e.g. so a semi-transparent relief reads dark.

        Args:
            facecolor: Optional axes + figure background colour. `None`
                (default) leaves the canvas unchanged.
        """
        ax, fig = self.ax, self.fig
        if self.extent is not None:
            xmin, xmax, ymin, ymax = self.extent
            width, height = abs(xmax - xmin), abs(ymax - ymin)
            if width > 0 and height > 0:
                fig_width = fig.get_size_inches()[0]
                fig.set_size_inches(fig_width, fig_width * height / width, forward=True)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if facecolor is not None:
            ax.set_facecolor(facecolor)
            fig.patch.set_facecolor(facecolor)
        ax.set_aspect("auto")
        ax.set_position([0, 0, 1, 1])

    def animate(
        self,
        time: list[Any],
        points: np.ndarray | PointOverlay | None = None,
        cell_value_text_colors: tuple[str, str] | _Unset = _UNSET,
        interval: int = 200,
        frame_label: FrameLabel | None = None,
        *,
        data_getter: Callable[[int], np.ndarray] | None = None,
        full_bleed: bool | str = False,
        basemap: bool | dict | Basemap | Callable[[Any], None] | None = None,
        colorbar: bool | ColorBar | None = None,
        **kwargs: Unpack[AnimateKwargs],
    ) -> FuncAnimation:
        """Create an animation from a single-band or true-colour stack.

        This method creates an animation by iterating the first axis of the
        data, turning each slice into a frame with optional time labels, point
        annotations, and cell-value displays. Two stack layouts are accepted:

        - a 3-D `(time, rows, cols)` single-band stack, rendered as a
          colormapped field with a colorbar (the historical behaviour); and
        - a 4-D `(time, rows, cols, 3|4)` RGB / RGBA stack, where each frame is
          drawn straight through `imshow` as true colour — no norm, colormap or
          colorbar (`self.cbar` is left `None`), and `display_cell_value` is
          ignored because per-cell annotation needs a scalar field. RGB/RGBA
          frames must be display-ready (floats in `[0, 1]` or `uint8` in
          `[0, 255]`, as produced by `prepare_array`); out-of-range values are
          clipped by matplotlib.

        Every frame shares the glyph's single `extent` (one spatial domain) —
        there is no per-frame extent. For data spanning different domains, build
        one `ArrayGlyph` per domain instead of stacking them.

        Args:
            time: A list containing labels for each frame in the animation.
                These could be timestamps, frame numbers, or any other identifiers.
                The length of this list should match the first dimension of the array.
            points: Points to display on the array, by default None. A
                `PointOverlay` (locations plus marker/label styling), or a
                plain `(N, 3)` array of `[value, row, col]` per point (a
                bare array is styled with `PointOverlay`'s own defaults).
                (Styling `points` via separate `point_color` /
                `point_size` / `point_label_color` / `point_label_size`
                keywords is deprecated; pass a `PointOverlay` instead —
                the old keywords still work as `**kwargs` and emit a
                `DeprecationWarning`.)
            cell_value_text_colors: Two colors to be used for cell value
                text, by default ("white", "black"). The first color is
                used when the cell value is below the
                background_color_threshold, and the second color is used
                when the cell value is above the threshold. (Renamed from
                `text_colors`; the old name still works as a keyword and
                emits a `DeprecationWarning`.)
            interval: Delay between frames in milliseconds, by default 200.
                Controls the speed of the animation (smaller values = faster animation).
            frame_label: Styling for the per-frame time label, by default
                None (a `FrameLabel` with its own defaults: auto-anchored
                top-left, black text). See `FrameLabel` for the
                `location`/`color` fields and the top-left anchoring
                behaviour when `location` is left unset. `ArrayGlyph`-only;
                `MeshGlyph.animate()` does not yet expose this option.
                (Styling the frame label via separate `label_location` /
                `label_color` keywords — or the even older `text_loc` for
                the location — is deprecated; pass a `FrameLabel` instead —
                the old keywords still work as `**kwargs` and emit a
                `DeprecationWarning`.)
            data_getter: Optional callable `f(i) -> ndarray` that
                returns the frame for index `i`, by default None.
                When set, `self.arr` is no longer iterated; each
                frame is fetched lazily through the callback — useful
                for streaming frames from a remote / lazy source
                (e.g. a NetCDF time slab). The frame may be a 2-D
                single-band array or a `(rows, cols, 3|4)` RGB / RGBA
                array; either way its spatial dims (the first two
                axes) must match `self.arr.shape[-2:]`. When None
                (default) the existing behaviour is preserved and
                `self.arr[i]` supplies frame `i`.
            full_bleed: Fill the whole figure edge-to-edge with no chrome, by
                default False. `True` hides ticks and spines, resizes the figure
                so its aspect matches the georeferenced data box (from `extent`,
                so the fill introduces no distortion), and gives the axes the
                entire figure area (`set_position([0, 0, 1, 1])`,
                `aspect="auto"`); the internal `tight_layout` is skipped. The
                canvas colour is left untouched, so masked / no-data cells keep
                the default background rather than turning black. Pass a colour
                string instead (e.g. `"black"`) to also paint the canvas that
                colour -- e.g. so a semi-transparent relief backdrop reads dark.
                Intended for chrome-free maps -- a colorbar or title has no room,
                so pair it with `add_colorbar=False` (and no `title`). Without an
                `extent` the axes still fills the figure but may stretch.
            basemap: A reference backdrop drawn via the glyph's own
                `add_relief` / `add_features` and composed with the frames by
                `zorder` (relief under the data, coastline/borders over it), by
                default None (no basemap). Accepts ``True`` for a sensible
                default (a `"low"` relief plus grey `"50m"` coastline and
                borders), a `Basemap` (the typed, validated form -- `relief` /
                `features` / `resolution` / `check_alignment`, with `features`
                taking `Feature` objects), a **dict** with the same keys (see
                `GeoMixin._draw_basemap`), or a **callable** ``f(glyph)`` for
                full control. On a value-linked-opacity `style` (e.g.
                `temperature_flame`) the cool areas reveal the terrain while the
                data glows on top. Drawing the relief needs the `[tiles]` extra
                (Pillow).
            colorbar: Colorbar presence and placement. `None` (default) keeps
                matplotlib's placement (honouring the legacy `add_colorbar`);
                `False` draws no colorbar; `True` a default one. Pass a
                `ColorBar` for control -- an edge (`location`), an `inside`
                inset that tracks `full_bleed`, a backing `box` (defaulted on
                for an inset), and text colours (`label_color` for the title,
                `tick_color` for the tick numbers). Same flag as `plot(colorbar=)`.
                On a `style=` preset, a placement `ColorBar` (or `True`) overrides
                the swatch with a real colorbar; a colours-only `ColorBar` styles
                the swatch in place (defaults < preset < explicit).
            **kwargs: Additional keyword arguments for customizing the animation.

                Plot appearance:
                    title : str, optional
                        Title of the plot, by default 'Array Plot'.
                    title_size : int, optional
                        Title font size, by default 15.
                    cmap : str or matplotlib.colors.Colormap, optional
                        Colormap, by default 'coolwarm_r'. A plain matplotlib
                        name (e.g. 'viridis') or a `Colormap` object is used
                        as-is; a **namespaced** name such as 'cmocean:thermal'
                        or 'crameri:batlow' is resolved via the optional `cmap`
                        aggregator — install the `[science-colors]` extra
                        (`pip install cleopatra[science-colors]`). The `_r`
                        reverse suffix works on both forms.
                    vmin : float, optional
                        Minimum value for color scaling, by default min(array).
                    vmax : float, optional
                        Maximum value for color scaling, by default max(array).

                Color bar options:
                    add_colorbar : bool, optional
                        Whether to draw the glyph's own color bar, by
                        default True. Set to False for shared-axes
                        composition, where the host owns a single
                        aggregated color bar; then `self.cbar` stays
                        None and no axes space is taken by a color bar.
                        The mappable is still reachable via `self.im`.
                    cbar_orientation : str, optional
                        Deprecated; use `colorbar=ColorBar(orientation=...)`.
                        Orientation of the color bar, by default 'vertical'.
                        Can be 'horizontal' or 'vertical'.
                    cbar_label_rotation : float, optional
                        Deprecated; use `colorbar=ColorBar(label_rotation=...)`.
                        Rotation angle (degrees) of the color bar label, by
                        default None (matplotlib's own label orientation).
                    cbar_label_location : str, optional
                        Deprecated; use `colorbar=ColorBar(label_location=...)`.
                        Location of the color bar label, by default 'center'.
                        Valid values depend on the bar orientation -- vertical:
                        'top'/'center'/'bottom'; horizontal: 'left'/'center'/'right'.
                    cbar_length : float, optional
                        Deprecated; use `colorbar=ColorBar(length=...)`. Ratio to
                        control the height/width of the color bar, by default 0.75.
                    ticks_spacing : int, optional
                        Deprecated; use `colorbar=ColorBar(ticks_spacing=...)`.
                        Spacing between ticks on the color bar, by default 5.
                    cbar_label_size : int, optional
                        Deprecated; use `colorbar=ColorBar(label_size=...)`. Font
                        size of the color bar label, by default 12.
                    cbar_label : str, optional
                        Deprecated; use `colorbar=ColorBar(label=...)`. Label text
                        for the color bar, by default None.

                Color scale options:
                    color_scale : ColorScale or str, optional
                        Type of color scaling to use, by default 'linear'.
                        Accepts a `cleopatra.styling.styles.ColorScale`
                        member or its string value (case-insensitive). An
                        unrecognised value raises `ValueError`. Options:
                        - 'linear': Linear scale
                        - 'power': Power-law normalization
                        - 'sym-lognorm': Symmetrical logarithmic scale
                        - 'boundary-norm': Discrete intervals based on boundaries
                        - 'midpoint': Scale split at a specified midpoint
                    gamma : float, optional
                        Exponent for 'power' color scale, by default 0.5.
                        Values < 1 emphasize lower values, values > 1 emphasize higher values.
                    line_threshold : float, optional
                        Threshold for 'sym-lognorm' color scale, by default 0.0001.
                    line_scale : float, optional
                        Scale factor for 'sym-lognorm' color scale, by default 0.001.
                    bounds : list, optional
                        Boundaries for 'boundary-norm' color scale, by default None.
                        Defines the discrete intervals for color mapping.
                    midpoint : float, optional
                        Midpoint value for 'midpoint' color scale, by default 0.

                Cell value display options:
                    display_cell_value : bool, optional
                        Whether to display the values of cells as text, by default False.
                    num_size : int, optional
                        Font size of the cell value text, by default 8.
                    background_color_threshold : float, optional
                        Threshold for cell value text color, by default None.
                        If cell value > threshold, text is black; otherwise, text is white.
                        If None, uses max(array)/2 as the threshold.
                    precision : int, optional
                        Decimal places each frame's cell value text is
                        rounded to, by default 2. `animate`-only; `plot`'s
                        equivalent per-cell text always rounds to 2
                        decimal places internally.

                Data-style preset / relief options:
                    style : str, optional
                        Name of a `cleopatra.styling.colors.DATA_STYLES` preset applied
                        to every frame (valid names:
                        `sorted(cleopatra.styling.colors.DATA_STYLES)`). Continuous
                        presets drive the frames through the preset's cmap +
                        norm + value-linked opacity and present a swatch legend
                        (matching `plot`); categorical presets remap the class
                        codes through a discrete colormap and draw a legend (no
                        colorbar). Under a lazy `data_getter` the continuous
                        colour range is taken from frame 0. A continuous preset
                        composes with `hillshade`; a categorical preset drops it
                        (and warns). By default None.
                    hillshade : bool or dict, optional
                        Relief-shade every frame of a regular-grid DEM (same
                        options as `plot`). Composes with a continuous `style`;
                        dropped for a categorical `style`. By default False.

        Returns:
            matplotlib.animation.FuncAnimation: The animation object that can be displayed
                in a notebook or saved to a file.

            As with `plot`, the first-frame colour-mapped artist is stored on
            the instance as `self.im` (and the colorbar, when drawn, on
            `self.cbar`), so a caller can attach a host-owned
            colorbar/legend without scraping the axes.

        Raises:
            ValueError: If an invalid keyword argument is provided.
            ValueError: If the length of the time list doesn't match the first dimension of the array.
            ValueError: If `data_getter` is None and `self.arr` is
                neither a 3-D `(time, rows, cols)` nor a 4-D
                `(time, rows, cols, 3|4)` array (no time axis to
                iterate over).
            ValueError: If `data_getter` is set and a returned frame's
                spatial dims do not match `self.arr.shape[-2:]`.

        Notes:
            The animation is created by iterating through the first dimension of the array.
            For example, if the array has shape (10, 20, 30), the animation will have 10 frames,
            each showing a 20x30 slice of the array.

            This method does not call `plt.show()`; it returns the `FuncAnimation` so the
            caller controls display. In an interactive (non-notebook) session call
            `plt.show()` yourself to play it, or use `save_animation` to write it to a file.

            To display the animation in a Jupyter notebook, you may need to use:
            ```python
            from IPython.display import HTML
            HTML(anim_obj.to_jshtml())
            ```

            To save the animation to a file, use the `save_animation` method after creating
            the animation.

        Examples:
        Basic animation from a 3D array:
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
        >>> # Create a 3D array with 5 frames, each 10x10
        >>> arr = np.random.randint(1, 10, size=(5, 10, 10))
        >>> # Create labels for each frame
        >>> frame_labels = ["Frame 1", "Frame 2", "Frame 3", "Frame 4", "Frame 5"]
        >>> # Create the ArrayGlyph object
        >>> animated_array = ArrayGlyph(arr, figsize=(8, 8), title="Animated Array")
        >>> # Create the animation
        >>> anim_obj = animated_array.animate(frame_labels)

        ```
        Animation with custom interval (speed):
        ```python
        >>> animated_array = ArrayGlyph(arr, figsize=(8, 8), title="Animated Array")
        >>> # Slower animation (500ms between frames)
        >>> anim_obj = animated_array.animate(frame_labels, interval=500)
        >>> animated_array = ArrayGlyph(arr, figsize=(8, 8), title="Animated Array")
        >>> # Faster animation (100ms between frames)
        >>> anim_obj = animated_array.animate(frame_labels, interval=100)

        ```
        Animation with points:
        ```python
        >>> # Create a styled point overlay to display on the animation
        >>> from cleopatra.glyphs.gridded.array_glyph import PointOverlay
        >>> overlay = PointOverlay(
        ...     np.array([[1, 2, 3], [2, 5, 5], [3, 8, 8]]),
        ...     color="black",
        ...     size=150,
        ...     label_color="white",
        ...     label_size=12,
        ... )
        >>> animated_array = ArrayGlyph(arr, figsize=(8, 8), title="Animated Array")
        >>> anim_obj = animated_array.animate(frame_labels, points=overlay)

        ```
        Animation with cell values displayed:
        ```python
        >>> animated_array = ArrayGlyph(arr, figsize=(8, 8), title="Animated Array")
        >>> anim_obj = animated_array.animate(
        ...     frame_labels,
        ...     display_cell_value=True,
        ...     num_size=10,
        ...     cell_value_text_colors=("yellow", "blue")
        ... )

        ```
        ![animated_array](./../images/array_glyph/animated_array.gif)

        Saving the animation to a file:
        ```python
        >>> # Create the animation first
        >>> animated_array = ArrayGlyph(arr, figsize=(8, 8), title="Animated Array")
        >>> anim_obj = animated_array.animate(frame_labels)
        >>> # Then save it to a file
        >>> animated_array.save_animation("animation.gif", fps=2)

        ```
        Lazy frame streaming via `data_getter` (the callback supplies
        frame `i` on demand — useful for NetCDF time slabs or any
        source where eager loading is too expensive). The data array
        on the glyph acts as a shape template; only its last two axes
        are read.
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
        >>> template = np.arange(36, dtype=float).reshape(1, 6, 6)
        >>> glyph = ArrayGlyph(template, figsize=(4, 4), title="Lazy")
        >>> labels = ["t0", "t1", "t2"]
        >>> def get_frame(i):
        ...     return np.full((6, 6), float(i)) + np.arange(36).reshape(6, 6)
        >>> anim_obj = glyph.animate(labels, data_getter=get_frame)
        >>> anim_obj._fig is glyph.fig
        True

        ```
        True-colour animation from a 4-D `(time, rows, cols, 3)` RGB stack.
        Each frame is drawn as true colour, so no colorbar is created
        (`glyph.cbar` stays `None`):
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
        >>> rgb_stack = np.linspace(0.0, 1.0, 3 * 6 * 6 * 3).reshape(3, 6, 6, 3)
        >>> glyph = ArrayGlyph(rgb_stack, figsize=(4, 4), title="RGB")
        >>> anim_obj = glyph.animate(["t0", "t1", "t2"])
        >>> glyph.cbar is None
        True

        ```
        Full-bleed layout: no chrome (ticks/spines) and the axes taking the whole
        figure. `full_bleed=True` leaves the canvas colour alone; pass a colour
        (`full_bleed="black"`) to also paint it, so masked cells read dark:
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
        >>> stack = np.arange(3 * 6 * 8, dtype=float).reshape(3, 6, 8)
        >>> glyph = ArrayGlyph(stack, extent=[0, 0, 8, 6])
        >>> anim_obj = glyph.animate(
        ...     ["t0", "t1", "t2"], full_bleed="black", add_colorbar=False
        ... )
        >>> tuple(round(float(v), 3) for v in glyph.ax.get_position().bounds)
        (0.0, 0.0, 1.0, 1.0)
        >>> glyph.ax.get_facecolor()
        (0.0, 0.0, 0.0, 1.0)

        ```
        Compose a reference basemap under the frames (`basemap=True`) -- relief
        below, coastline and borders over. The `animate` call is `+SKIP`ped in
        doctest because it downloads the `[tiles]` assets on first use:
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
        >>> stack = np.arange(3 * 20 * 30, dtype=float).reshape(3, 20, 30)
        >>> glyph = ArrayGlyph(stack, extent=[-12, 34, 32, 64])
        >>> anim_obj = glyph.animate(  # doctest: +SKIP
        ...     ["t0", "t1", "t2"], basemap=True, full_bleed=True, add_colorbar=False
        ... )

        ```
        """
        cell_value_text_colors = cast(
            "tuple[str, str]",
            _resolve_renamed_kwarg(
                kwargs,  # type: ignore[arg-type]
                "text_colors",
                "cell_value_text_colors",
                cell_value_text_colors,
                ("white", "black"),
            ),
        )
        frame_label = _resolve_frame_label(frame_label, kwargs)  # type: ignore[arg-type]
        points = _resolve_point_overlay(points, kwargs)  # type: ignore[arg-type]
        _warn_deprecated_cbar_kwargs(kwargs)

        frame_location = frame_label.location
        label_location_is_default = frame_location is None
        if label_location_is_default:
            label_location = [0.02, 0.95]
        else:
            assert frame_location is not None
            label_location = frame_location

        resolved_colorbar = self._apply_kwargs_and_colorbar(colorbar, kwargs)  # type: ignore[arg-type]

        if "ticks_spacing" not in resolved_colorbar:
            if "ticks_spacing" in kwargs.keys():
                self.default_options["ticks_spacing"] = kwargs["ticks_spacing"]
            else:
                self.default_options["ticks_spacing"] = self.ticks_spacing

        if "vmin" in kwargs.keys():
            self.default_options["vmin"] = kwargs["vmin"]
        else:
            self.default_options["vmin"] = self.vmin

        if "vmax" in kwargs.keys():
            self.default_options["vmax"] = kwargs["vmax"]
        else:
            self.default_options["vmax"] = self.vmax

        precision = self.default_options["precision"]
        array = self.arr

        def _is_rgb_frame(frame: np.ndarray) -> bool:
            return frame.ndim == 3 and frame.shape[-1] in (3, 4)

        if data_getter is None:
            if array.ndim == 4 and array.shape[-1] in (3, 4):
                # 4-D true-colour stack: (time, rows, cols, 3|4).
                frame_0 = array[0]
                n_frames = array.shape[0]
            elif array.ndim == 3:
                # 3-D single-band stack: (time, rows, cols).
                frame_0 = array[0, :, :]
                n_frames = array.shape[0]
            else:
                raise ValueError(
                    "animate requires a 3-D (time, rows, cols) or 4-D "
                    "(time, rows, cols, 3|4) array, or a data_getter callback"
                )
        else:
            n_frames = len(time)
            frame_0 = np.asarray(data_getter(0))
            expected_hw = tuple(array.shape[-2:])
            actual_hw = frame_0.shape[:2] if _is_rgb_frame(frame_0) else frame_0.shape
            if actual_hw != expected_hw:
                raise ValueError(
                    f"`data_getter` returned shape {frame_0.shape}, whose "
                    f"spatial dims {actual_hw} do not match the data array's "
                    f"last two axes {expected_hw}."
                )

        rgb_frames = _is_rgb_frame(frame_0)
        show_cell_value = self.default_options["display_cell_value"] and not rgb_frames

        if self.fig is None:
            self.fig, self.ax = self.create_figure_axes()

        fig, ax = self.fig, self.ax

        style_render: Any = None
        style_categorical = False

        if rgb_frames:
            _clear_prior_render_artists(ax)
            im = ax.imshow(frame_0, extent=self.extent)
            self.im = im
            self.cbar = None
        else:
            ticks = self.get_ticks()
            self._create_norm_and_cbar_kw(ticks)
            _clear_prior_render_artists(ax)
            im, cbar_kw = self._plot_im_get_cbar_kw(ax, frame_0, ticks)
            self.im = im

            self.cbar = None
            if self.default_options["add_colorbar"]:
                self.cbar = self.create_color_bar(ax, im, cbar_kw)

            frame_0_scalar = np.asarray(
                ma.filled(ma.asarray(frame_0).astype(float), np.nan), dtype=float
            )
            style = self.default_options.get("style")
            if style is not None:
                if points is not None or show_cell_value:
                    warnings.warn(
                        "data-style presets bypass point and cell-value "
                        "overlays; 'points' and 'display_cell_value' are ignored "
                        "with 'style'.",
                        stacklevel=2,
                    )
                    points = None
                    show_cell_value = False
                layer = self._resolve_style_layer(style)
                cfg = {**DATA_STYLES[style][layer], **self._style_color_overrides}
                self._apply_style_background(cfg)
                hillshade_active = (
                    resolve_hillshade(self.default_options.get("hillshade")) is not None
                )
                categories = cfg.get("categories")
                if categories is not None:
                    style_categorical = True
                    if hillshade_active:
                        warnings.warn(
                            "hillshade is not composed with a categorical "
                            "data-style preset; the preset is applied and "
                            "hillshade ignored.",
                            stacklevel=2,
                        )
                    cats = sorted(categories, key=lambda c: c[0])
                    cat_values = np.array([float(c[0]) for c in cats])
                    cat_colors = [c[1] for c in cats]
                    cat_labels = [c[2] for c in cats]
                    cat_cmap = ListedColormap(cat_colors)
                    cat_norm = BoundaryNorm(
                        category_boundaries(list(cat_values)), len(cat_colors)
                    )
                    if self.cbar is not None:
                        self.cbar.remove()
                        self.cbar = None
                    im.set_data(frame_0_scalar)
                    im.set_cmap(cat_cmap)
                    im.set_norm(cat_norm)
                    if self.default_options["add_colorbar"]:
                        disjoint_legend(
                            ax,
                            cat_colors,
                            cat_labels,
                            title=cfg["label"],
                            loc="upper right",
                        )
                    style_render = ("categorical", cat_cmap, cat_norm, cat_values)
                else:
                    stack = array if data_getter is None else frame_0
                    style_norm, style_vmin, style_vmax = resolve_style_norm(
                        np.asarray(
                            ma.filled(ma.asarray(stack).astype(float), np.nan),
                            dtype=float,
                        ),
                        cfg,
                    )
                    style_cmap = resolve_colormap(cfg["cmap"])
                    im.set_data(frame_0_scalar)
                    im.set_cmap(style_cmap)
                    im.set_norm(style_norm)
                    if self.cbar is not None:
                        self.cbar.remove()
                        self.cbar = None
                    if self._style_wants_colorbar:
                        insets = list(ax.child_axes)
                        for _inset in insets:
                            _inset.remove()
                        mappable = ScalarMappable(norm=style_norm, cmap=style_cmap)
                        mappable.set_array([])
                        self.cbar = self.create_color_bar(
                            ax, mappable, self._style_cbar_kw(style_norm)
                        )
                    elif self.default_options["add_colorbar"]:
                        insets = list(ax.child_axes)
                        for _inset in insets:
                            _inset.remove()
                        vmin_prefix, vmax_prefix = swatch_extend_prefixes(style_norm)
                        swatch_legend(
                            ax,
                            style_cmap,
                            cfg["label"],
                            vmin=style_vmin,
                            vmax=style_vmax,
                            norm=style_norm,
                            vmin_prefix=vmin_prefix,
                            vmax_prefix=vmax_prefix,
                            bounds=(0.02, 0.92, 0.32, 0.06),
                            text_color=self.default_options.get("cbar_label_color")
                            or _swatch_text_default(self.default_options.get("cbar_box")),
                            value_color=self.default_options.get("cbar_tick_color")
                            or _swatch_text_default(self.default_options.get("cbar_box")),
                            box=self.default_options.get("cbar_box"),
                        )
                    alpha_vmin = cfg.get("alpha_vmin")
                    alpha_vmax = cfg.get("alpha_vmax")
                    style_alpha_norm = (
                        Normalize(vmin=alpha_vmin, vmax=alpha_vmax)
                        if alpha_vmin is not None or alpha_vmax is not None
                        else None
                    )
                    style_render = (
                        "continuous",
                        style_cmap,
                        style_norm,
                        style_alpha_norm,
                        cfg.get("alpha"),
                    )

        ax.set_title(
            self.default_options["title"], fontsize=self.default_options["title_size"]
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_xticks([])
        ax.set_yticks([])

        cell_text_value: list = []
        if show_cell_value:
            indices = get_indices2(frame_0, [np.nan])
            cell_text_value = self._plot_text(
                ax, frame_0, indices, self.default_options
            )
            indices = np.array(indices)

        points_scatter = None
        points_id: list = []
        if points is not None:
            row = points.points[:, 1]
            col = points.points[:, 2]
            points_scatter = ax.scatter(col, row, color=points.color, s=points.size)
            points_id = self._plot_point_values(
                ax, points.points, points.label_color, points.label_size
            )

        background_color_threshold = None
        if not rgb_frames:
            if self.default_options["background_color_threshold"] is not None:
                background_color_threshold = im.norm(
                    self.default_options["background_color_threshold"]
                )
            else:
                ref_for_threshold = array if data_getter is None else frame_0
                background_color_threshold = im.norm(np.nanmax(ref_for_threshold)) / 2.0

        day_text = ax.text(
            label_location[0],
            label_location[1],
            " ",
            fontsize=(
                frame_label.size
                if frame_label.size is not None
                else self.default_options["cbar_label_size"]
            ),
            color=frame_label.color,
            transform=ax.transAxes if label_location_is_default else ax.transData,
            va="top" if label_location_is_default else "baseline",
        )
        self._day_text = day_text

        def _fetch_frame(i: int) -> np.ndarray:
            """Resolve frame `i` for the animation step.

            Routes between the eager `self.arr[i]` path and the lazy
            `data_getter(i)` callback added in CLEO-7. The frame's
            spatial dims (its first two axes) must always match
            `self.arr.shape[-2:]`; the callback variant re-validates
            per call to catch upstream shape drift (e.g. a NetCDF slab
            that changed size between frames).

            Args:
                i: Zero-based frame index. Must be a valid index into
                    the time axis (`0 <= i < n_frames`).

            Returns:
                np.ndarray: The frame for index `i` — a 2-D single-band
                    array, or a `(rows, cols, 3|4)` RGB / RGBA array —
                    whose spatial dims equal `self.arr.shape[-2:]`.

            Raises:
                ValueError: If `data_getter` is set and the callback
                    returns a frame whose spatial dims do not match
                    `self.arr.shape[-2:]`.
            """
            if data_getter is None:
                frame = array[i] if rgb_frames else array[i, :, :]
            else:
                frame = np.asarray(data_getter(i))
                expected_hw = tuple(array.shape[-2:])
                actual_hw = frame.shape[:2] if _is_rgb_frame(frame) else frame.shape
                if actual_hw != expected_hw:
                    raise ValueError(
                        f"`data_getter` returned shape {frame.shape}, whose "
                        f"spatial dims {actual_hw} do not match {expected_hw}."
                    )
            return np.asarray(frame)

        hillshade_opts = resolve_hillshade(self.default_options.get("hillshade"))
        if style_categorical:
            hillshade_opts = None

        def _display_frame(frame):
            """Return the frame's image data: preset RGBA, relief-shaded, or raw."""
            if style_render is not None:
                filled = np.asarray(
                    ma.filled(ma.asarray(frame).astype(float), np.nan), dtype=float
                )
                if style_render[0] == "categorical":
                    _, cat_cmap, cat_norm, cat_values = style_render
                    masked = np.where(np.isin(filled, cat_values), filled, np.nan)
                    rgba = np.asarray(cat_cmap(cat_norm(masked)), dtype=float)
                    rgba[~np.isfinite(masked)] = 0.0
                    return rgba
                _, cmap_, norm_, alpha_norm_, const_ = style_render
                rgba = alpha_rgba(filled, cmap_, norm_, alpha_norm_, const_)
                if hillshade_opts is not None:
                    rgba = shade_rgb(rgba, filled, **hillshade_opts)
                return rgba
            if hillshade_opts is not None and not rgb_frames:
                # Cast before filling: integer masked frames reject a NaN fill.
                elevation = np.asarray(
                    ma.filled(ma.asarray(frame).astype(float), np.nan), dtype=float
                )
                return shade_grid(elevation, im.cmap, norm=im.norm, **hillshade_opts)
            return frame

        def init():
            """initialize the plot with the cached first frame"""
            im.set_data(_display_frame(frame_0))
            day_text.set_text("")
            output = [im, day_text]

            if points is not None:
                assert points_scatter is not None
                points_scatter.set_offsets(np.c_[col, row])
                output.append(points_scatter)
                update_points = lambda x: points_id[x].set_text(points.points[x, 0])
                list(map(update_points, range(len(col))))

                output += points_id

            if show_cell_value:
                vals = frame_0[indices[:, 0], indices[:, 1]]
                update_cell_value = lambda x: cell_text_value[x].set_text(vals[x])
                list(map(update_cell_value, range(len(cell_text_value))))
                output += cell_text_value

            return output

        def animate_a(i):
            """plot for each element in the iterable."""
            frame = _fetch_frame(i)
            im.set_data(_display_frame(frame))
            day_text.set_text("Date = " + str(time[i])[0:10])
            output = [im, day_text]

            if points is not None:
                assert points_scatter is not None
                points_scatter.set_offsets(np.c_[col, row])
                output.append(points_scatter)

                for x in range(len(col)):
                    points_id[x].set_text(points.points[x, 0])

                output += points_id

            if show_cell_value:
                vals = frame[indices[:, 0], indices[:, 1]]

                def update_cell_value(x):
                    """Update cell value"""
                    val = round(vals[x], precision)
                    kw = {
                        "color": cell_value_text_colors[
                            int(im.norm(vals[x]) > background_color_threshold)
                        ]
                    }
                    cell_text_value[x].update(kw)
                    cell_text_value[x].set_text(val)

                list(map(update_cell_value, range(len(cell_text_value))))

                output += cell_text_value

            return output

        if basemap is not None:
            self._draw_basemap(basemap)
        if full_bleed:
            self._apply_full_bleed(facecolor=full_bleed if isinstance(full_bleed, str) else None)
        else:
            plt.tight_layout()
            if getattr(self, "_auto_figure", False):
                self._tighten_figure()
        anim = FuncAnimation(
            fig,
            animate_a,
            init_func=init,
            frames=n_frames,
            interval=interval,
            blit=True,
        )
        self._anim = anim
        _mark_render_artists(
            ax,
            self.cbar,
            self.im,
            self._day_text,
            points_scatter,
            *points_id,
            *cell_text_value,
        )
        return anim

