"""Reusable chart / map furniture: a scale bar and a north arrow.

Two free functions that decorate an existing `matplotlib.axes.Axes` with the
two remaining pieces of standard figure furniture cleopatra did not have --
a scale bar (a segmented bar with tick numbers and a caption) and a north
arrow (a rotatable compass mark). They sit beside `stamp_mark`
(`cleopatra.styling.watermark`) and share its placement plumbing (`_CORNERS`,
`_as_margins`, `_corner_origin`) so all three anchor identically, and they
read like `cleopatra.styling.colorbar.ColorBar` (the same `box` /
`label_location` / `label_size` vocabulary).

Both are **plain matplotlib artistry** and know nothing about geography. A
scale bar is `length` axes **data units** wide with a caller-supplied `label`
string (e.g. `"100 km"`); a north arrow is rotated by a caller-supplied
`rotation` in degrees. The ellipsoidal questions -- how long is 100 km in axis
units at this latitude, what is the grid convergence here -- belong to whoever
owns the CRS (the consumer), never to this generic matplotlib layer. So a
micrograph with a micrometre bar, a floor plan, an engineering section and a
map all use the identical artist.

The bar is drawn on a frameless inset axes (the returned handle) in
axes-fraction coordinates, so it stays anchored in its corner across a dpi or
limit change rather than drifting like a data-coordinate `Rectangle`; its
tick numbers, caption and backing panel are drawn on the parent axes in
`transAxes`. Everything sits at a high zorder, above the data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Affine2D

from cleopatra.styling.watermark import _CORNERS, _as_margins, _corner_origin

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["ScaleBar", "add_scale_bar", "add_north_arrow"]

#: Furniture sits above ordinary data artists (images ~0, collections ~1-3).
_FURNITURE_ZORDER = 6.0

#: The north-arrow styles `add_north_arrow` accepts.
_NORTH_STYLES = ("arrow", "needle", "rose")

#: Tick length as an axes fraction (how far tick marks reach past the bar).
_TICK_LEN = 0.012


def _resolve_box(box: bool | str | dict | None) -> dict | None:
    """Resolve a `box` value into `Rectangle` kwargs, mirroring `ColorBar`.

    Args:
        box: `None` / `False` for no panel, `True` for a translucent white
            panel with a light-grey edge, a colour string for a panel of that
            face colour, or a dict of `matplotlib.patches.Rectangle` kwargs
            merged over the defaults.

    Returns:
        dict or None: The `Rectangle` keyword arguments for the backing panel,
        or `None` when no panel should be drawn.
    """
    if not box:
        return None
    kw: dict = {
        "facecolor": "white",
        "edgecolor": "0.6",
        "linewidth": 0.6,
        "alpha": 0.8,
    }
    if isinstance(box, str):
        kw["facecolor"] = box
    elif isinstance(box, dict):
        kw = {**kw, **box}
    return kw


def _draw_box(
    ax: Axes,
    x0: float,
    y0: float,
    width: float,
    height: float,
    box: bool | str | dict | None,
    zorder: float,
) -> None:
    """Draw a simple rectangular backing panel around a furniture rect.

    A no-op for a falsy `box`; otherwise adds one `Rectangle` (a hair larger
    than the furniture rect) on `ax` in axes-fraction coordinates, below the
    furniture. Used by `add_north_arrow`; the scale bar uses `_draw_scale_box`,
    which also leaves room for the tick numbers and caption.

    Args:
        ax: The parent axes.
        x0: The furniture rect's left edge in axes fraction.
        y0: The furniture rect's bottom edge in axes fraction.
        width: The rect width in axes fraction.
        height: The rect height in axes fraction.
        box: The `box` value (see `_resolve_box`).
        zorder: The draw order (the panel sits just below the furniture).
    """
    kw = _resolve_box(box)
    if kw is None:
        return
    pad_box = 0.01
    ax.add_patch(
        Rectangle(
            (x0 - pad_box, y0 - pad_box),
            width + 2 * pad_box,
            height + 2 * pad_box,
            transform=ax.transAxes,
            zorder=zorder,
            clip_on=False,
            **kw,
        )
    )


def _segment_fills(segments: int, color: str, edge_color: str) -> list[str]:
    """Return the alternating fill colour for each of `segments` blocks.

    Args:
        segments: The number of alternating blocks (`>= 1`).
        color: The fill for even blocks (0, 2, ...).
        edge_color: The fill for odd blocks (1, 3, ...).

    Returns:
        list[str]: One fill colour per block, alternating `color` / `edge_color`
        (a single `color` block when `segments == 1`).
    """
    return [color if i % 2 == 0 else edge_color for i in range(segments)]


@dataclass(frozen=True)
class ScaleBar:
    """Presentation options for `add_scale_bar` (everything but the axes/length).

    Grouped into one object -- mirroring `FacetLayout` / `ColorBar` -- so the
    scale-bar call stays small: `add_scale_bar(ax, length, ScaleBar(...))`.

    Attributes:
        label: The caption under (or over) the bar, e.g. `"100 km"`. Defaults
            to `f"{length:g}"` when `None`.
        location: Which corner to anchor to -- one of `"lower right"`,
            `"lower left"`, `"upper right"`, `"upper left"`.
        pad: The gap between the bar and the axes edges, as an axes fraction --
            a scalar for both axes or an `(x, y)` pair, each in `[0, 1)`.
        height: The bar thickness as an axes fraction.
        segments: The number of alternating blocks. `1` draws a plain bar.
            Must be `>= 1`.
        ticks: `True` numbers the `segments + 1` block boundaries `0 .. length`;
            a sequence places tick numbers at those data positions (each in
            `[0, length]`); `False` draws no tick numbers.
        color: The fill of the even blocks and the colour of the block outline,
            the ticks and the text.
        edge_color: The fill of the odd blocks (the alternating light blocks).
        label_location: `"top"` or `"bottom"` -- which side of the bar the tick
            numbers and caption sit on. `None` picks the side facing the axes
            interior for the chosen corner (a lower corner labels above the bar,
            an upper corner below), so the caption never spills off the edge.
        label_size: Font size (points) for the tick numbers and caption.
            `None` uses matplotlib's default.
        box: A backing panel behind the bar, using `ColorBar`'s vocabulary --
            `None` / `False` (none), `True` (translucent white), a colour
            string, or a dict of `Rectangle` kwargs.
        zorder: The draw order for the furniture. `None` uses a high default
            that sits above the data.
    """

    label: str | None = None
    location: str = "lower right"
    pad: float | tuple[float, float] = 0.025
    height: float = 0.012
    segments: int = 2
    ticks: bool | Sequence[float] = True
    color: str = "black"
    edge_color: str = "white"
    label_location: str | None = None
    label_size: float | None = None
    box: bool | str | dict | None = None
    zorder: float | None = None


def add_scale_bar(ax: Axes, length: float, spec: ScaleBar | None = None) -> Axes:
    """Draw a segmented scale bar on `ax`, sized in the axes' own data units.

    The bar is `length` **data units** wide (the caller computes that number --
    cleopatra owns no geodesy), rendered as `spec.segments` alternating blocks
    on a frameless inset axes anchored in one corner. Tick numbers at the block
    boundaries and the caption are drawn on the parent axes in axes-fraction
    coordinates, so the whole assembly stays put across a dpi or limits change
    instead of drifting like a data-coordinate `Rectangle`.

    Args:
        ax: The axes to decorate. Its current x-limits set the data-to-figure
            scale, so call this after the data is plotted and the limits are
            final.
        length: The bar length in the axes' **x data units**. Must be finite
            and `> 0`.
        spec: The presentation options as a `ScaleBar` (label, location, pad,
            segments, colours, box, ...). `None` uses all defaults.

    Returns:
        Axes: The frameless inset axes the bar was drawn on, so the caller can
        adjust it further.

    Raises:
        ValueError: If `spec.location` is not a corner, `length` is not finite
            and positive, `spec.pad` is out of range, `spec.segments < 1`,
            `spec.label_location` is not `"top"` / `"bottom"`, the axes has a
            zero-width x-range, or `pad + bar width` leaves no room on the axes.

    Examples:
        - A four-block "100 km" bar in the lower-left corner:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.styling.furniture import add_scale_bar, ScaleBar
            >>> fig, ax = plt.subplots()
            >>> ax.set_xlim(0, 500_000)
            (0.0, 500000.0)
            >>> bar = add_scale_bar(
            ...     ax, 100_000, ScaleBar(label="100 km", location="lower left", segments=4)
            ... )
            >>> len(bar.patches)   # four alternating blocks
            4
            >>> plt.close(fig)

            ```
    """
    spec = spec or ScaleBar()
    label = spec.label
    location = spec.location
    pad = spec.pad
    height = spec.height
    segments = spec.segments
    ticks = spec.ticks
    color = spec.color
    edge_color = spec.edge_color
    label_location = spec.label_location
    label_size = spec.label_size
    box = spec.box
    zorder = spec.zorder

    if location not in _CORNERS:
        raise ValueError(f"location must be one of {list(_CORNERS)}, got {location!r}.")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"length must be a finite positive number, got {length!r}.")
    if segments < 1:
        raise ValueError(f"segments must be >= 1, got {segments!r}.")
    if label_location not in ("top", "bottom", None):
        raise ValueError(
            f"label_location must be 'top', 'bottom', or None, got {label_location!r}."
        )
    if label_location is None:
        # Default: grow the tick numbers / caption toward the axes interior, so a
        # bar in a lower corner labels above the bar and one in an upper corner
        # labels below -- otherwise a default lower-corner bar spills its caption
        # off the bottom edge, over the axis tick labels.
        label_location = "top" if location.startswith("lower") else "bottom"
    pad_x, pad_y = _as_margins(pad)

    x0d, x1d = ax.get_xlim()
    data_range = abs(float(x1d) - float(x0d))
    if not data_range:
        raise ValueError("the axes has a zero-width x-range; cannot size a scale bar.")
    width = length / data_range
    if pad_x + width > 1.0:
        raise ValueError(
            f"the bar spans {width:.3g} of the axes width; with pad_x={pad_x} it does "
            "not fit. Shorten `length`, zoom out, or reduce `pad`."
        )

    z = _FURNITURE_ZORDER if zorder is None else zorder
    x0, y0 = _corner_origin(location, width, height, pad_x, pad_y)

    at_top = label_location == "top"
    # The tick marks / numbers / caption grow away from the bar on the label
    # side. `sign` points from the bar toward that side in axes fraction.
    sign = 1.0 if at_top else -1.0
    bar_edge = (y0 + height) if at_top else y0
    text_va = "bottom" if at_top else "top"

    _draw_scale_box(ax, x0, y0, width, height, sign, bar_edge, bool(ticks), box, z)
    inset = _draw_bar_segments(
        ax, x0, y0, width, height, segments, color, edge_color, z
    )
    _label_scale_bar(
        ax,
        x0,
        width,
        bar_edge,
        sign,
        text_va,
        ticks,
        length,
        segments,
        label,
        color,
        label_size,
        z,
    )
    return inset


def _draw_bar_segments(
    ax: Axes,
    x0: float,
    y0: float,
    width: float,
    height: float,
    segments: int,
    color: str,
    edge_color: str,
    zorder: float,
) -> Axes:
    """Draw the alternating scale-bar blocks on a frameless inset axes.

    Args:
        ax: The parent axes.
        x0: The bar's left edge in axes fraction.
        y0: The bar's bottom edge in axes fraction.
        width: The bar width in axes fraction.
        height: The bar thickness in axes fraction.
        segments: The number of alternating blocks.
        color: The even-block fill and the block outline.
        edge_color: The odd-block fill.
        zorder: The furniture draw order (the inset sits just above it).

    Returns:
        Axes: The frameless inset the blocks were drawn on.
    """
    inset = ax.inset_axes(
        (x0, y0, width, height), transform=ax.transAxes, zorder=zorder + 0.1
    )
    inset.set_xlim(0.0, 1.0)
    inset.set_ylim(0.0, 1.0)
    inset.set_navigate(False)
    inset.set_in_layout(False)
    inset.axis("off")
    inset.patch.set_visible(False)
    for i, fill in enumerate(_segment_fills(segments, color, edge_color)):
        inset.add_patch(
            Rectangle(
                (i / segments, 0.0),
                1.0 / segments,
                1.0,
                facecolor=fill,
                edgecolor=color,
                linewidth=0.8,
            )
        )
    return inset


def _label_scale_bar(
    ax: Axes,
    x0: float,
    width: float,
    bar_edge: float,
    sign: float,
    text_va: str,
    ticks: bool | Sequence[float],
    length: float,
    segments: int,
    label: str | None,
    color: str,
    label_size: float | None,
    zorder: float,
) -> None:
    """Draw the tick marks / numbers and the caption on the parent axes.

    Args:
        ax: The parent axes.
        x0: The bar's left edge in axes fraction.
        width: The bar width in axes fraction.
        bar_edge: The bar edge (top or bottom) the text grows from.
        sign: `+1` when the text is above the bar, `-1` when below.
        text_va: The text vertical alignment (`"top"` / `"bottom"`).
        ticks: The `ticks` option (`True` numbers the block boundaries, a
            sequence numbers those data positions, falsy draws none).
        length: The bar length in data units.
        segments: The number of blocks (sets the `True` boundary count).
        label: The caption; defaults to `f"{length:g}"` when `None`.
        color: The tick / text colour.
        label_size: Font size (points) for the tick numbers and caption.
        zorder: The furniture draw order.
    """
    if ticks is True:
        # Number the segment boundaries 0 .. length.
        tick_positions = [i / segments for i in range(segments + 1)]
        tick_values = [i * length / segments for i in range(segments + 1)]
    else:
        tick_positions, tick_values = _scale_bar_ticks(ticks, length)
    for frac, value in zip(tick_positions, tick_values):
        ax.plot(
            [x0 + frac * width, x0 + frac * width],
            [bar_edge, bar_edge + sign * _TICK_LEN],
            color=color,
            linewidth=0.8,
            transform=ax.transAxes,
            zorder=zorder + 0.1,
            clip_on=False,
        )
        ax.text(
            x0 + frac * width,
            bar_edge + sign * (_TICK_LEN + 0.006),
            f"{value:g}",
            ha="center",
            va=text_va,
            color=color,
            fontsize=label_size,
            transform=ax.transAxes,
            zorder=zorder + 0.1,
            clip_on=False,
        )

    caption = f"{length:g}" if label is None else label
    caption_offset = (_TICK_LEN + 0.05) if tick_positions else 0.008
    ax.text(
        x0 + width / 2.0,
        bar_edge + sign * caption_offset,
        caption,
        ha="center",
        va=text_va,
        color=color,
        fontsize=label_size,
        transform=ax.transAxes,
        zorder=zorder + 0.1,
        clip_on=False,
    )


def _scale_bar_ticks(
    ticks: bool | Sequence[float], length: float
) -> tuple[list[float], list[float]]:
    """Resolve the `ticks` option to `(fractions, values)` along the bar.

    Args:
        ticks: `True` for numbers at the block boundaries (handled by the
            caller, which knows `segments`), a sequence of data positions in
            `[0, length]`, or a falsy value for no ticks.
        length: The bar length in data units.

    Returns:
        tuple[list[float], list[float]]: The tick positions as fractions of the
        bar length and their data values. `True` returns empty lists here --
        boundary ticks are added by `add_scale_bar` from `segments` -- while an
        explicit sequence is validated and converted.

    Raises:
        ValueError: If an explicit position is outside `[0, length]`.
    """
    if ticks is True or not ticks:
        return [], []
    positions = [float(p) for p in ticks]
    for p in positions:
        if p < 0.0 or p > length:
            raise ValueError(
                f"tick position {p!r} is outside the bar range [0, {length:g}]."
            )
    return [p / length for p in positions], positions


def _draw_scale_box(
    ax: Axes,
    x0: float,
    y0: float,
    width: float,
    height: float,
    sign: float,
    bar_edge: float,
    has_ticks: bool,
    box: bool | str | dict | None,
    zorder: float,
) -> None:
    """Draw the backing panel behind a scale bar on the parent axes.

    Args:
        ax: The parent axes.
        x0: The bar's left edge in axes fraction.
        y0: The bar's bottom edge in axes fraction.
        width: The bar width in axes fraction.
        height: The bar height in axes fraction.
        sign: `+1` when the caption is above the bar, `-1` when below.
        bar_edge: The bar edge (top or bottom) the text grows from.
        has_ticks: Whether tick numbers are drawn (they add a row above the
            caption, which the panel must clear). The caption itself is always
            drawn, so caption room is always reserved.
        box: The `box` value (see `_resolve_box`).
        zorder: The draw order (the panel sits just below the furniture).
    """
    kw = _resolve_box(box)
    if kw is None:
        return
    hpad = 0.02
    vpad = 0.012
    # The caption is always drawn (it defaults to the length), so always reserve
    # room for it; tick numbers add a further row above it when present.
    text_room = (_TICK_LEN + 0.09) if has_ticks else 0.05
    if sign > 0:
        y_lo = y0 - vpad
        y_hi = bar_edge + text_room + vpad
    else:
        y_lo = bar_edge - text_room - vpad
        y_hi = y0 + height + vpad
    ax.add_patch(
        Rectangle(
            (x0 - hpad, y_lo),
            width + 2 * hpad,
            y_hi - y_lo,
            transform=ax.transAxes,
            zorder=zorder,
            clip_on=False,
            **kw,
        )
    )


def _north_arrow_patches(style: str, color: str, edge_color: str) -> list[Polygon]:
    """Build the north-arrow polygons in a unit `(0, 1)` box, pointing up.

    Args:
        style: One of `"arrow"` (a single filled arrow), `"needle"` (a two-tone
            compass diamond whose four spikes are each split down the middle into
            a `color` flank and an `edge_color` flank), or `"rose"` (a four-point
            compass star whose spikes likewise split into `color` / `edge_color`
            flanks).
        color: The primary fill (the whole arrow, or one flank per spike) and
            the outline colour.
        edge_color: The secondary (alternating) flank fill.

    Returns:
        list[matplotlib.patches.Polygon]: The polygons, centred on `(0.5, 0.5)`
        with the point at the top, ready to be rotated about the centre.
    """
    cx = 0.5
    if style == "arrow":
        shaft = 0.08
        return [
            Polygon(
                [
                    (cx, 0.92),
                    (cx + 0.16, 0.5),
                    (cx + shaft, 0.5),
                    (cx + shaft, 0.12),
                    (cx - shaft, 0.12),
                    (cx - shaft, 0.5),
                    (cx - 0.16, 0.5),
                ],
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
            )
        ]
    if style == "needle":
        return [
            Polygon(
                [(cx, 0.9), (cx - 0.16, 0.42), (cx, 0.5)],
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
            ),
            Polygon(
                [(cx, 0.9), (cx + 0.16, 0.42), (cx, 0.5)],
                closed=True,
                facecolor=edge_color,
                edgecolor=color,
                linewidth=0.8,
            ),
            Polygon(
                [(cx, 0.1), (cx - 0.16, 0.58), (cx, 0.5)],
                closed=True,
                facecolor=edge_color,
                edgecolor=color,
                linewidth=0.8,
            ),
            Polygon(
                [(cx, 0.1), (cx + 0.16, 0.58), (cx, 0.5)],
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
            ),
        ]
    # "rose": a four-point star, each cardinal ray split into two shaded halves.
    tips = [(cx, 0.92), (0.92, cx), (cx, 0.08), (0.08, cx)]
    patches: list[Polygon] = []
    for i, (tx, ty) in enumerate(tips):
        patches.append(
            Polygon(
                [(tx, ty), _rose_base(i, cx, "l"), (cx, cx)],
                closed=True,
                facecolor=color if i % 2 == 0 else edge_color,
                edgecolor=color,
                linewidth=0.8,
            )
        )
        patches.append(
            Polygon(
                [(tx, ty), _rose_base(i, cx, "r"), (cx, cx)],
                closed=True,
                facecolor=edge_color if i % 2 == 0 else color,
                edgecolor=color,
                linewidth=0.8,
            )
        )
    return patches


def _rose_base(i: int, cx: float, side: str) -> tuple[float, float]:
    """Return one flank vertex of the `i`-th ray of a four-point compass rose.

    Args:
        i: The ray index (0 = north, 1 = east, 2 = south, 3 = west).
        cx: The centre coordinate (`0.5`).
        side: `"l"` for the left flank vertex, `"r"` for the right.

    Returns:
        tuple[float, float]: The flank vertex in the unit box, offset from the
        centre toward the neighbouring diagonal so the two halves of each ray
        meet the centre with a visible width.
    """
    off = 0.14
    flanks = {
        0: {"l": (cx - off, cx + off), "r": (cx + off, cx + off)},  # north
        1: {"l": (cx + off, cx + off), "r": (cx + off, cx - off)},  # east
        2: {"l": (cx + off, cx - off), "r": (cx - off, cx - off)},  # south
        3: {"l": (cx - off, cx - off), "r": (cx - off, cx + off)},  # west
    }
    return flanks[i][side]


def add_north_arrow(
    ax: Axes,
    *,
    rotation: float = 0.0,
    location: str = "upper right",
    pad: float | tuple[float, float] = 0.025,
    size: float = 0.06,
    style: str = "arrow",
    label: str | None = "N",
    color: str = "black",
    edge_color: str = "white",
    label_size: float | None = None,
    box: bool | str | dict | None = None,
    zorder: float | None = None,
) -> Axes:
    """Draw a north arrow on `ax`, rotated by a caller-supplied angle.

    The arrow is drawn undistorted on a frameless inset axes anchored in one
    corner and rotated `rotation` degrees clockwise from straight up (the caller
    supplies the grid convergence -- cleopatra owns no CRS). The `"N"` label
    rotates with it.

    Args:
        ax: The axes to decorate.
        rotation: Degrees clockwise from up to rotate the arrow (e.g. the grid
            convergence at the map centre). Must be finite. Defaults to `0`.
        location: Which corner to anchor to -- one of `"upper right"` (default),
            `"upper left"`, `"lower right"`, `"lower left"`.
        pad: The gap to the axes edges, as an axes fraction -- a scalar or an
            `(x, y)` pair, each in `[0, 1)`.
        size: The arrow height as an axes fraction. Defaults to `0.06`.
        style: `"arrow"` (a single filled arrow), `"needle"` (a two-tone
            compass diamond), or `"rose"` (a four-point compass star). The
            needle and rose split each spike into a `color` flank and an
            `edge_color` flank.
        label: The label at the arrow tip. Defaults to `"N"`; `None` draws none.
        color: The primary fill and the outline / label colour.
        edge_color: The secondary (alternating) flank fill of a needle / rose.
        label_size: Font size (points) for the label. `None` uses the default.
        box: A backing panel, using `ColorBar`'s `box` vocabulary (see
            `add_scale_bar`).
        zorder: The draw order. `None` uses a high default above the data.

    Returns:
        Axes: The frameless inset axes the arrow was drawn on.

    Raises:
        ValueError: If `location` is not a corner, `style` is unknown,
            `rotation` is not finite, `pad` is out of range, or `pad + size`
            leaves no room on the axes.

    Examples:
        - A plain north arrow in the upper-right corner:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.styling.furniture import add_north_arrow
            >>> fig, ax = plt.subplots()
            >>> arrow = add_north_arrow(ax, rotation=0.0)
            >>> len(arrow.patches)   # one filled arrow
            1
            >>> plt.close(fig)

            ```
    """
    if location not in _CORNERS:
        raise ValueError(f"location must be one of {list(_CORNERS)}, got {location!r}.")
    if style not in _NORTH_STYLES:
        raise ValueError(f"style must be one of {list(_NORTH_STYLES)}, got {style!r}.")
    if not np.isfinite(rotation):
        raise ValueError(f"rotation must be a finite number, got {rotation!r}.")
    pad_x, pad_y = _as_margins(pad)

    # Keep the arrow square on screen: scale the inset's axes-fraction width by
    # the axes' inch aspect so a unit box maps to a square, and a rotation in
    # that box is a true screen rotation.
    fig_w_in, fig_h_in = ax.figure.get_size_inches()
    ax_pos = ax.get_position()
    ax_w_in = max(ax_pos.width * fig_w_in, 1e-9)
    ax_h_in = max(ax_pos.height * fig_h_in, 1e-9)
    width = size * (ax_h_in / ax_w_in)
    if pad_x + width > 1.0 or pad_y + size > 1.0:
        raise ValueError(
            f"pad + arrow size leaves no room on the axes (needs {width:.3g} x "
            f"{size:.3g} in axes fraction). Reduce `size` or `pad`."
        )

    z = _FURNITURE_ZORDER if zorder is None else zorder
    x0, y0 = _corner_origin(location, width, size, pad_x, pad_y)

    _draw_box(ax, x0, y0, width, size, box, z)

    inset = ax.inset_axes((x0, y0, width, size), transform=ax.transAxes, zorder=z + 0.1)
    inset.set_xlim(0.0, 1.0)
    inset.set_ylim(0.0, 1.0)
    inset.set_navigate(False)
    inset.set_in_layout(False)
    inset.axis("off")
    inset.patch.set_visible(False)

    rot = Affine2D().rotate_deg_around(0.5, 0.5, -rotation) + inset.transData
    for patch in _north_arrow_patches(style, color, edge_color):
        patch.set_transform(rot)
        inset.add_patch(patch)

    if label is not None:
        inset.text(
            0.5,
            0.99,
            label,
            ha="center",
            va="bottom",
            color=color,
            fontsize=label_size,
            fontweight="bold",
            rotation=-rotation,
            rotation_mode="anchor",
            transform=rot,
            clip_on=False,
        )
    return inset
