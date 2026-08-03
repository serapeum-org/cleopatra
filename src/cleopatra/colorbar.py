"""Typed colorbar configuration shared by every glyph.

`ColorBar` bundles the colorbar layout / caption / sizing choices into one value
passed as `plot(colorbar=...)` / `animate(colorbar=...)`. `_resolve_colorbar`
maps it onto the internal `cbar_*` / `ticks_spacing` options that the base
`Glyph.create_color_bar` renders, and `_warn_deprecated_cbar_kwargs` steers
callers off the loose `cbar_*` kwargs it supersedes.

Kept in its own module (rather than in any single glyph) so every glyph type can
import it without depending on `array_glyph`. `cleopatra.array_glyph` re-exports
`ColorBar` for backwards compatibility.
"""

import warnings
from typing import Any, Literal

from matplotlib.colors import to_rgb

from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS


class ColorBar:
    """Placement (and backing box) for the colorbar `plot` / `animate` draws.

    Bundles the colorbar-layout choices -- which edge it sits on, whether it
    is inset *inside* the frame, and its backing box -- into one value passed
    as `plot(colorbar=...)` / `animate(colorbar=...)`, mirroring `FrameLabel`.
    Pass `colorbar=True` / `False` / `None` for the simple cases and a
    `ColorBar` for placement control.

    Attributes:
        location: Edge the colorbar sits on -- `"left"`, `"right"`, `"top"`,
            or `"bottom"`. `None` (default) keeps matplotlib's placement
            (right of a vertical bar). Left/right force a vertical bar,
            top/bottom a horizontal one.
        orientation: Bar orientation -- `"vertical"` or `"horizontal"`. `None`
            (default) lets `location` decide, and yields a vertical bar when
            `location` is `None` too. Because a set `location` fixes the
            orientation, an `orientation` that disagrees with it is ignored
            (with a `UserWarning`) -- set only one. The resolved orientation is
            sticky on a reused glyph: a later `ColorBar()` with `orientation`
            unset does not reset a previously applied one.
        inside: When `True`, the colorbar is inset *inside* the frame at
            `location` (overlaying the data) rather than in an outside gutter,
            by default `False`. An inset is a child of the data axes, so it
            tracks the axes through `full_bleed`.
        box: Backing panel behind the scale, so the data does not show through
            its labels. `False` draws none; `True` an opaque white panel; a
            colour string a panel of that colour; a dict of
            `matplotlib.patches.Rectangle` kwargs for full control. Defaults to
            `None`, which becomes `True` when `inside` is set (an inset over
            moving data almost always wants a panel) and stays off otherwise.
            For a real colorbar the panel backs an *inside* colorbar only (it is
            ignored when `inside=False`, which sits in its own gutter); for a
            `style` preset's swatch legend it backs the swatch regardless of
            placement, and the swatch title/values then default to a colour that
            contrasts with the panel (an explicit `label_color`/`tick_color`
            still wins).
        label_color: Colour of the scale's title text -- the colorbar's axis
            label and, for a `style` preset, the swatch legend's title (the
            endpoint values take `tick_color`, not this). `None` (default) keeps
            the default: matplotlib's for a colorbar label; for the swatch, a
            colour that contrasts with `box`, else white.
        tick_color: Colour of the tick labels (the numbers) of a real colorbar
            and, for a `style` preset, the swatch legend's endpoint values.
            `None` (default) keeps matplotlib's default for a colorbar; for the
            swatch it defaults to a colour that contrasts with `box`, else white.
        label: Caption text for the scale (the colorbar's title). `None`
            (default) keeps the current default caption.
        length: Bar length as a fraction of the axis (e.g. `0.8`). `None`
            (default) keeps the default length.
        label_size: Font size of the caption. `None` (default) keeps the
            default.
        label_rotation: Rotation of the caption in degrees. `None` (default)
            leaves matplotlib's own label orientation; pass a value to rotate
            the caption (e.g. `0` for a horizontal caption).
        label_location: Where the caption sits along the bar (e.g. `"center"`).
            Distinct from `location`, which is the bar's *edge*. Valid values
            depend on orientation (vertical bar: `"top"`/`"center"`/`"bottom"`;
            horizontal bar: `"left"`/`"center"`/`"right"`). `None` (default)
            keeps the default.
        ticks_spacing: Spacing between the colorbar's ticks. `None` (default)
            keeps the default.

    Examples:
        - An inside colorbar on the right -- its box defaults on:
            ```python
            >>> from cleopatra.colorbar import ColorBar
            >>> spec = ColorBar(location="right", inside=True)
            >>> spec.inside, spec.box
            (True, True)

            ```
        - Black title + tick numbers, outside on the bottom (no box):
            ```python
            >>> from cleopatra.colorbar import ColorBar
            >>> spec = ColorBar(location="bottom", label_color="black", tick_color="black")
            >>> (spec.box, spec.label_color, spec.tick_color)
            (None, 'black', 'black')

            ```
        - A captioned bar, fully specified through the spec (no loose kwargs):
            ```python
            >>> from cleopatra.colorbar import ColorBar
            >>> spec = ColorBar(location="bottom", label="Rainfall mm/day", length=0.8)
            >>> (spec.label, spec.length)
            ('Rainfall mm/day', 0.8)

            ```
    """

    def __init__(
        self,
        *,
        location: Literal["left", "right", "top", "bottom"] | None = None,
        orientation: Literal["vertical", "horizontal"] | None = None,
        inside: bool = False,
        box: bool | str | dict | None = None,
        label_color: str | None = None,
        tick_color: str | None = None,
        label: str | None = None,
        length: float | None = None,
        label_size: float | None = None,
        label_rotation: float | None = None,
        label_location: str | None = None,
        ticks_spacing: float | None = None,
    ) -> None:
        """Initialise a `ColorBar`.

        Args:
            location: Edge to sit on (`"left"`/`"right"`/`"top"`/`"bottom"`),
                or `None` for matplotlib's default placement.
            orientation: Bar orientation (`"vertical"`/`"horizontal"`), or
                `None` to let `location` decide (a vertical bar when neither is
                set). Ignored (with a `UserWarning`) when it disagrees with the
                orientation `location` implies.
            inside: Inset the colorbar inside the frame, by default `False`.
            box: Backing panel for an inside colorbar (`True` / colour / dict),
                or `None` to default it on when `inside` is set.
            label_color: Colour of the scale title / colorbar label (and the
                swatch title for a `style` preset); `None` keeps the default.
            tick_color: Colour of the colorbar's tick numbers; `None` keeps
                matplotlib's default.
            label: Caption text (scale title); `None` keeps the default.
            length: Bar length as a fraction of the axis; `None` keeps the
                default.
            label_size: Caption font size; `None` keeps the default.
            label_rotation: Caption rotation in degrees; `None` leaves
                matplotlib's own label orientation.
            label_location: Caption placement along the bar (distinct from
                `location`, the bar's edge); valid values depend on orientation
                (vertical: top/center/bottom, horizontal: left/center/right);
                `None` keeps the default.
            ticks_spacing: Spacing between the colorbar's ticks; `None` keeps
                the default.
        """
        if orientation is not None and orientation not in ("vertical", "horizontal"):
            raise ValueError(
                "ColorBar orientation must be 'vertical' or 'horizontal', got "
                f"{orientation!r}."
            )
        self.location = location
        self.orientation = orientation
        # `location` fixes the orientation (left/right -> vertical, top/bottom
        # -> horizontal). If an explicit `orientation` disagrees it is ignored
        # downstream, so warn here rather than dropping it silently (issue #235).
        # Only a valid edge implies an orientation; an invalid `location` is left
        # for `create_color_bar` to reject with a clearer message.
        if location in ("left", "right", "top", "bottom") and orientation is not None:
            implied = "vertical" if location in ("left", "right") else "horizontal"
            if orientation != implied:
                warnings.warn(
                    f"ColorBar(orientation={orientation!r}) is ignored because "
                    f"location={location!r} already fixes the orientation to "
                    f"{implied!r}; set only one.",
                    UserWarning,
                    stacklevel=2,
                )
        # Cross-field check: valid `label_location` values depend on the bar
        # orientation, so an incompatible pair would crash deep in matplotlib.
        # Reject it up front -- but only when this spec pins the orientation
        # (via `location` or an explicit `orientation`); with neither set the
        # rendered orientation comes from the glyph's (sticky) default, which a
        # ColorBar cannot know, so we skip rather than risk a false rejection
        # (issue #241).
        effective_orientation = None
        if location in ("left", "right", "top", "bottom"):
            effective_orientation = (
                "vertical" if location in ("left", "right") else "horizontal"
            )
        elif orientation is not None:
            effective_orientation = orientation
        if effective_orientation is not None and label_location is not None:
            valid_locations = (
                ("bottom", "center", "top")
                if effective_orientation == "vertical"
                else ("left", "center", "right")
            )
            if label_location not in valid_locations:
                raise ValueError(
                    f"label_location={label_location!r} is not valid for a "
                    f"{effective_orientation} colorbar; use one of "
                    f"{list(valid_locations)}."
                )
        self.inside = inside
        # An inset over moving data almost always wants a panel: default the
        # box on when `inside` is set and the caller did not decide explicitly.
        self.box = True if (inside and box is None) else box
        self.label_color = label_color
        self.tick_color = tick_color
        self.label = label
        self.length = length
        self.label_size = label_size
        self.label_rotation = label_rotation
        self.label_location = label_location
        self.ticks_spacing = ticks_spacing


def _swatch_text_default(box: bool | str | dict | None) -> str:
    """Default swatch title/value colour that stays legible over `box`.

    With no backing panel the swatch sits directly on the map, where white
    reads on the usual dark relief/data backgrounds (the historical default).
    With a panel, choose black or white by the panel's luminance so the title
    and endpoint values never render invisibly (e.g. white-on-white when the
    default `box=True` draws a white panel).

    Args:
        box: The resolved `cbar_box` -- `None`/`False` (no panel), `True` (a
            white panel), a colour string, or a dict of `Rectangle` kwargs
            (its `facecolor` decides).

    Returns:
        str: `"white"` when there is no panel or the panel is dark, `"black"`
            when the panel is light.
    """
    if not box:
        return "white"
    if box is True:
        facecolor: Any = "white"
    elif isinstance(box, dict):
        facecolor = box.get("facecolor", "white")
    else:
        facecolor = box
    try:
        r, g, b = to_rgb(facecolor)
    except (ValueError, TypeError):
        return "white"
    # Rec. 709 relative luminance: dark text on a light panel and vice versa.
    return "black" if (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.5 else "white"


def _resolve_colorbar(colorbar: bool | ColorBar | None) -> dict:
    """Translate a `colorbar=` argument into `default_options` updates.

    Args:
        colorbar: `None` (default) leaves the colorbar options untouched --
            matplotlib's placement, honouring the legacy `add_colorbar`;
            `False` suppresses the colorbar; `True` draws a default one,
            resetting the resettable `cbar_*` family to defaults so it does not
            inherit a prior sticky spec (unlike `None`, which leaves them); a
            `ColorBar` sets its placement, caption, and sizing.

    Returns:
        dict: Updates to merge into `default_options` (empty for `None`).

    Raises:
        TypeError: If `colorbar` is not a bool, `ColorBar`, or `None`.

    Examples:
        - `False` suppresses the colorbar; a `ColorBar` maps onto the
            internal `cbar_*` keys `create_color_bar` reads:
            ```python
            >>> from cleopatra.colorbar import _resolve_colorbar, ColorBar
            >>> _resolve_colorbar(False)
            {'add_colorbar': False}
            >>> _resolve_colorbar(ColorBar(location="left", inside=True))["cbar_location"]
            'left'

            ```
        - Caption / sizing fields map onto their `cbar_*` keys only when set,
            so an unset field is omitted (leaving the existing default):
            ```python
            >>> from cleopatra.colorbar import _resolve_colorbar, ColorBar
            >>> _resolve_colorbar(ColorBar(label="Depth [m]", length=0.8))["cbar_label"]
            'Depth [m]'
            >>> "cbar_label" in _resolve_colorbar(ColorBar(location="right"))
            False

            ```
    """
    if colorbar is None:
        return {}
    if colorbar is False:
        return {"add_colorbar": False}
    if colorbar is True:
        return {
            "add_colorbar": True,
            "cbar_location": None,
            "cbar_inside": False,
            "cbar_box": None,
            "cbar_label_color": None,
            "cbar_tick_color": None,
            # A bare `True` means "default colorbar": reset the whole resettable
            # cbar_* family to its defaults so nothing sticky from a prior call
            # on a reused glyph leaks in (issue #242). `ticks_spacing` is omitted
            # on purpose -- it is auto-computed per data range when not set, and
            # seeding it here would defeat that (see plot/animate).
            "cbar_orientation": STYLE_DEFAULTS["cbar_orientation"],
            "cbar_label": STYLE_DEFAULTS["cbar_label"],
            "cbar_length": STYLE_DEFAULTS["cbar_length"],
            "cbar_label_size": STYLE_DEFAULTS["cbar_label_size"],
            "cbar_label_rotation": STYLE_DEFAULTS["cbar_label_rotation"],
            "cbar_label_location": STYLE_DEFAULTS["cbar_label_location"],
        }
    if isinstance(colorbar, ColorBar):
        updates = {
            "add_colorbar": True,
            "cbar_location": colorbar.location,
            "cbar_inside": colorbar.inside,
            "cbar_box": colorbar.box,
            "cbar_label_color": colorbar.label_color,
            "cbar_tick_color": colorbar.tick_color,
        }
        # Caption / size / spacing fields override the loose cbar_* keys only
        # when set, so an unset field leaves the existing default in place (and
        # a caller still mixing in a loose cbar_label during the transition is
        # not clobbered). Phase 2 deprecates those loose kwargs.
        optional = {
            "cbar_label": colorbar.label,
            "cbar_length": colorbar.length,
            "cbar_label_size": colorbar.label_size,
            "cbar_label_rotation": colorbar.label_rotation,
            "cbar_label_location": colorbar.label_location,
            "cbar_orientation": colorbar.orientation,
            "ticks_spacing": colorbar.ticks_spacing,
        }
        updates.update({k: v for k, v in optional.items() if v is not None})
        return updates
    raise TypeError(
        "colorbar must be a bool, ColorBar, or None, got "
        f"{type(colorbar).__name__}."
    )


#: Loose colorbar kwargs now superseded by `ColorBar` fields (issue #234).
#: Passing any of these to `plot` / `animate` is deprecated -- use the mapped
#: `ColorBar` field instead. This set includes the non-`cbar_`-prefixed
#: `ticks_spacing`, deprecated in favour of `ColorBar(ticks_spacing=...)`, not
#: only the `cbar_*` keys. They still take effect (folded into
#: `default_options`) so nothing breaks during the deprecation window.
_DEPRECATED_CBAR_KWARGS = {
    "cbar_label": "label",
    "cbar_length": "length",
    "cbar_label_size": "label_size",
    "cbar_label_rotation": "label_rotation",
    "cbar_label_location": "label_location",
    "cbar_orientation": "orientation",
    "ticks_spacing": "ticks_spacing",
}


def _warn_deprecated_cbar_kwargs(kwargs: dict) -> None:
    """Warn for any loose `cbar_*` kwarg that a `ColorBar` field now replaces.

    The kwargs still take effect (the caller folds them into `default_options`);
    this only steers callers toward the typed `colorbar=ColorBar(...)` spec.

    Args:
        kwargs: The `plot` / `animate` keyword arguments to inspect for the
            deprecated loose colorbar keys.
    """
    for key, field in _DEPRECATED_CBAR_KWARGS.items():
        if key in kwargs:
            warnings.warn(
                f"The '{key}' keyword is deprecated; pass "
                f"colorbar=ColorBar({field}=...) instead.",
                DeprecationWarning,
                stacklevel=3,
            )
