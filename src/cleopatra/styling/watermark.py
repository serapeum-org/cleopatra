"""Figure watermark / brand-mark helper.

`stamp_mark` places a logo or watermark image onto a matplotlib `Figure`,
sized as a *fraction of the figure* (so it stays proportional across the
several dpis a figure is often exported at -- an MP4 master, a smaller web
copy, a GIF) and anchored in one of the four corners, with an optional
gaussian-blurred *halo* so the mark reads on a busy or dark canvas.

The halo is centred, not offset: the mark is composited over arbitrary
imagery -- night-side ocean, sunlit cloud, a bright limb -- and a symmetric
halo reads the same whichever way the background falls, where an offset drop
shadow would imply a light direction nothing else in the frame has.

This is a presentation helper, not a glyph: it takes a finished `Figure` and
draws on top of it via a frameless inset axes in figure-fraction coordinates
(the dpi-independent counterpart of `Figure.figimage`, which is pixel-based).
Both are free functions taking a `Figure`, and both are also available as
glyph methods through `WatermarkMixin` -- `glyph.stamp_mark(logo)` rather than
importing and passing `glyph.fig` -- following the same
free-function-plus-sugar shape `styling.furniture` and `basemap.geo` already
use for the scale bar and north arrow.

`stamp_watermark` is its text counterpart: the diagonal translucent brand
text across the middle of a frame, plus an optional credit line along the
bottom. The two are designed to be used together -- a corner logo from
`stamp_mark`, brand text from `stamp_watermark` -- and share their conventions,
so a size is a fraction of the figure and a position is a margin from its edge
rather than a point size or an offset in inches.

Tiled / repeated marks and any licensing / provenance semantics remain out of
scope.

The gaussian blur for the halo uses `PIL` (Pillow), already a hard cleopatra
dependency, so no new dependency (and no SciPy) is pulled in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import patheffects
from PIL import Image, ImageFilter

from cleopatra.glyphs.base.compositing import alpha_over

if TYPE_CHECKING:
    import os

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Text

__all__ = ["WatermarkMixin", "stamp_mark", "stamp_watermark"]

#: The four corner anchors `stamp_mark` accepts.
_CORNERS = ("lower right", "lower left", "upper right", "upper left")

#: Default halo blur sigma, as a fraction of the mark's own (unpadded) width.
DEFAULT_BLUR = 0.065

#: Transparent border added around the mark before blurring, in sigmas. Three
#: sigmas hold ~99.7% of the kernel, so the halo's tail is not clipped by the
#: canvas it is drawn on -- the pad exists precisely to contain that tail.
_HALO_SIGMAS = 3.0

#: Peak halo opacity.
_HALO_ALPHA = 0.5

#: Inset zorder: the mark sits above ordinary figure content.
_MARK_ZORDER = 1_000_001

#: Watermark text sits *below* the brand mark, so a corner logo overlapping the
#: diagonal text stays legible, and both sit above ordinary figure content.
_WATERMARK_ZORDER = 1_000_000

#: The credit line sits above both -- it is the smallest element and the one
#: that must never be occluded.
_CREDIT_ZORDER = 1_000_002

#: Outline drawn behind the credit line only. The big diagonal text is left
#: plain on purpose: an outline makes it read as a solid caption rather than a
#: watermark, while the credit is small enough that it needs the stroke to stay
#: legible against arbitrary frame content.
_CREDIT_STROKE_WIDTH = 2.0
_CREDIT_STROKE_COLOR = "black"
_CREDIT_STROKE_ALPHA = 0.9

#: Fitting a font size to a figure fraction is iterative: rendered text size is
#: very nearly linear in point size, but hinting quantises glyphs to whole
#: pixels, so one correction lands within a fraction of a percent and a short
#: loop closes the rest. Both bounds are generous; the loop normally exits on
#: the tolerance after two or three passes.
_FIT_TOLERANCE = 0.002
_FIT_MAX_PASSES = 12


def stamp_mark(
    fig: Figure,
    path: str | os.PathLike | np.ndarray,
    *,
    frac: float = 0.11,
    corner: str = "lower right",
    margin: float | tuple[float, float] = 0.025,
    shadow: bool = True,
    blur: float = DEFAULT_BLUR,
) -> Axes:
    """Stamp a logo / watermark image onto a figure, sized as a fraction of it.

    Places `path` in one corner of `fig` on a frameless inset axes in
    figure-fraction coordinates, so the mark keeps the same proportion (and
    corner offset) no matter what dpi the figure is later saved at. The image
    is drawn undistorted: `frac` sets its width relative to the figure width
    and the height is derived from the image and figure aspect ratios.

    Args:
        fig: The matplotlib `Figure` to stamp. The mark is drawn on top of
            whatever the figure already contains.
        path: The mark image. A file path (any format `PIL` can open, read as
            RGBA) or an in-memory ``(H, W, 3)`` / ``(H, W, 4)`` array -- either
            ``uint8`` ``0-255`` or float ``0-1``; RGB gains an opaque alpha.
        frac: The size of the mark's *longer* on-figure side as a fraction of
            the corresponding figure side, in ``(0, 1]`` -- the width for a
            landscape mark, the height for a portrait one -- so the mark always
            fits and is never distorted. Defaults to ``0.11``.
        corner: Which corner to anchor to -- ``"lower right"`` (default),
            ``"lower left"``, ``"upper right"``, or ``"upper left"``.
        margin: The gap between the **mark** and the figure edges, as a fraction
            of the figure, each in ``[0, 1)``. Either a scalar applied to both
            axes or an ``(x, y)`` pair -- a pair is what lets a mark tuck hard
            into a corner on one axis (``margin=(0.025, 0.0)``) while keeping a
            gap on the other. Defaults to ``0.025``.
        shadow: Whether to composite a gaussian-blurred halo behind the mark so
            it separates from a busy or dark canvas. Defaults to ``True``.
        blur: Halo blur sigma, as a fraction of the mark's own **unpadded**
            width. Defaults to `DEFAULT_BLUR`. Must be non-negative (validated
            even when ``shadow=False``, where it is otherwise unused); ``0`` is
            treated as no halo.

    Returns:
        Axes: The frameless inset axes the mark was drawn on, so the caller can
        further adjust it (e.g. ``ax.set_zorder(...)``). With ``shadow=True``
        that axes holds the mark *and* its halo, so its bbox is larger than the
        mark by the halo pad -- see the sizing note below.

    Raises:
        ValueError: If `corner` is not one of the four accepted anchors, if
            `frac` is not in ``(0, 1]``, if `margin` is not a scalar or
            ``(x, y)`` pair in ``[0, 1)``, if `margin + mark size` exceeds the
            figure, if `blur` is negative, or if an image array is out of
            contract (wrong shape, a non-``uint8`` non-float dtype, or a float
            outside ``[0, 1]`` or containing NaN/inf -- see `_as_rgba`).
        FileNotFoundError: If `path` is a file path that does not exist.
        PIL.UnidentifiedImageError: If `path` is a file that is not an image
            `PIL` can decode.

    Notes:
        The mark is baked at stamp time from the figure's current size, so call
        `stamp_mark` **last** -- after any `tight_layout()` / layout
        finalization (stamping first then calling `tight_layout()` warns), and
        after the final `set_size_inches`. Placement holds across dpi but not
        across a later figure-size change. Saving with `bbox_inches="tight"`
        changes the mark's relative margin / size -- it crops surrounding
        whitespace, and a halo tucked near an edge (whose grown axes overflows
        the figure) can even *extend* the tight bbox outward; a plain ``dpi=``
        save preserves the placement.

        `frac` always sizes the **mark itself**, never the canvas it is
        composited on. The halo needs a transparent pad of
        ``_HALO_SIGMAS * blur`` on each side to hold its own tail, which makes
        that canvas ``1 + 2 * _HALO_SIGMAS * blur`` times the mark's width
        (1.39x at the defaults). The axes rect is grown by exactly that factor
        so the visible mark still measures `frac`; sizing the padded canvas to
        `frac` instead would silently render the mark at ~72% of the requested
        size, which is easy to miss because the axes bbox looks right.

        `margin` is measured to the mark, so a halo next to a small margin is
        clipped at the figure edge -- which is what you want when tucking a
        mark hard into a corner.

    Examples:
        - Stamp a logo array in the lower-right corner at 11 % of the width:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.styling.watermark import stamp_mark
            >>> fig = plt.figure(figsize=(8, 6))
            >>> logo = np.zeros((40, 80, 4), dtype=np.uint8)
            >>> logo[..., :3] = 255  # white
            >>> logo[..., 3] = 255   # opaque
            >>> ax = stamp_mark(fig, logo, frac=0.2, shadow=False)
            >>> [round(float(v), 3) for v in ax.get_position().bounds]
            [0.775, 0.025, 0.2, 0.133]
            >>> plt.close(fig)

            ```
    """
    if corner not in _CORNERS:
        raise ValueError(f"corner must be one of {list(_CORNERS)}, got {corner!r}.")
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac!r}.")
    margin_x, margin_y = _as_margins(margin)
    if blur < 0.0:
        raise ValueError(f"blur must be non-negative, got {blur!r}.")

    image = _as_rgba(path)
    img_h, img_w = image.shape[:2]
    if img_h == 0 or img_w == 0:
        raise ValueError(
            f"the mark image has a zero-size dimension {image.shape[:2]}; "
            "it must have a positive height and width."
        )
    fig_w_in, fig_h_in = fig.get_size_inches()

    width = float(frac)
    # Keep the image undistorted: its on-figure height is its width scaled by the
    # image aspect and corrected for the figure's own aspect, because a unit of
    # figure-fraction height spans fewer inches than a unit of width (or more).
    height = width * (img_h / img_w) * (fig_w_in / fig_h_in)
    # `frac` sizes the mark's *longer* on-figure side: for a landscape mark that
    # is the width (unchanged), but a portrait mark whose derived height exceeds
    # `frac` is scaled down so its height is `frac` instead -- otherwise a tall
    # logo would silently overflow the figure. Aspect is preserved either way.
    longest = max(width, height)
    if longest > frac:
        scale = frac / longest
        width *= scale
        height *= scale
    # `frac` and `margin` are each in range on their own, but their sum must
    # still leave the mark on the figure: `margin + size > 1` would place the
    # mark off the opposite edge (`x0 = 1 - margin - width < 0`).
    if margin_x + width > 1.0 or margin_y + height > 1.0:
        raise ValueError(
            f"margin + mark size exceeds the figure: margin={(margin_x, margin_y)} "
            f"leaves no room for a {width:.3g}x{height:.3g} (figure-fraction) mark. "
            "Reduce frac or margin."
        )
    x0, y0 = _corner_origin(corner, width, height, margin_x, margin_y)

    # `width`/`height` are the MARK's rect. When a halo is composited in, the
    # image handed to `imshow` is the padded canvas, so the axes rect has to grow
    # by the same ratio (about the mark's centre) or the mark would render at
    # 1/grow of the requested `frac`.
    drawn, grow_w, grow_h = (image, 1.0, 1.0)
    # `blur == 0` yields an invisible halo but still pads the canvas (min 1 px),
    # so skip the composite entirely -- it only wastes work and inflates the bbox.
    if shadow and blur > 0.0:
        drawn, grow_w, grow_h = _composite_halo(image, blur)
    rect_w = width * grow_w
    rect_h = height * grow_h
    rect_x = x0 - (rect_w - width) / 2.0
    rect_y = y0 - (rect_h - height) / 2.0

    ax = fig.add_axes(
        (rect_x, rect_y, rect_w, rect_h), frameon=False, zorder=_MARK_ZORDER
    )
    ax.imshow(drawn, aspect="auto", interpolation="antialiased")
    ax.axis("off")
    ax.set_in_layout(False)
    return ax


def _fit_text_to_frac(fig: Figure, artist: Text, frac: float) -> None:
    """Scale `artist`'s font size until its longer on-figure side is `frac`.

    Matplotlib sizes text in points, which is an *absolute* unit: a figure
    resized after the text is placed keeps the same point size and so the text
    covers a different share of the frame. Measuring the rendered extent and
    solving for the point size that hits a figure fraction is what makes the
    text behave like `stamp_mark`'s `frac` instead.

    Rendered size is very nearly linear in point size, so one correction is
    almost right; hinting quantises glyphs to whole pixels, so the loop repeats
    until the measured fraction is within `_FIT_TOLERANCE` (or the pass budget
    runs out, which leaves the closest size tried rather than failing).

    The extent is the artist's *rotated* bounding box, so a diagonal watermark
    is sized by the box it actually occupies rather than by its unrotated width.

    Args:
        fig: The figure the artist belongs to.
        artist: The text to resize, already added to `fig`.
        frac: The target fraction of the corresponding figure side.
    """
    figure_box = fig.get_window_extent()
    for _ in range(_FIT_MAX_PASSES):
        box = artist.get_window_extent()
        longest = max(box.width / figure_box.width, box.height / figure_box.height)
        if longest <= 0.0:
            # Only an empty string measures zero -- whitespace has real width
            # and height. `stamp_watermark` rejects one, so this guards the
            # helper against a direct caller rather than a reachable input.
            return
        if abs(longest - frac) <= _FIT_TOLERANCE:
            return
        artist.set_fontsize(artist.get_fontsize() * frac / longest)


def stamp_watermark(
    fig: Figure,
    text: str,
    *,
    frac: float = 0.55,
    angle: float = 30.0,
    alpha: float = 0.65,
    color: str = "white",
    credit: str | None = None,
    credit_frac: float = 0.28,
    credit_alpha: float = 1.0,
    margin: float = 0.014,
) -> tuple[Text, Text | None]:
    """Stamp diagonal brand text across a figure, sized as a fraction of it.

    The text counterpart to `stamp_mark`: translucent brand text centred on
    `fig` at an angle, with an optional credit line along the bottom edge. Like
    `stamp_mark` it sizes by a *fraction of the figure* rather than in points,
    and positions by a *margin from the edge* rather than a hardcoded offset.

    Sizing by fraction is what makes the parameter mean the same thing for any
    text. A point size scaled off the figure's width -- the obvious shortcut --
    renders a short word small and a long one straight off the canvas, because
    how much of the frame a string covers depends on how many characters it
    has. `frac` is measured on what is actually rendered, so a two-letter brand
    and a twenty-character one both land at the fraction asked for.

    The diagonal text is drawn plain and the credit line is drawn with a dark
    outline. That asymmetry is deliberate: an outline on the large text makes it
    read as a solid caption rather than a watermark, while the credit line is
    small enough that it needs the stroke to stay legible against whatever the
    frame happens to contain.

    Args:
        fig: The matplotlib `Figure` to stamp. The text is drawn on top of
            whatever the figure already contains.
        text: The brand text. Must be a non-empty string.
        frac: The size of the text's *longer* on-figure side as a fraction of
            the corresponding figure side, in ``(0, 1]`` -- measured on the
            rotated bounding box, so `angle` is accounted for. Defaults to
            ``0.55``.
        angle: Rotation in degrees, counter-clockwise. Defaults to ``30.0``.
        alpha: Opacity of the brand text, in ``[0, 1]``. Defaults to ``0.65``,
            translucent enough to read as a watermark over the frame.
        color: Any matplotlib colour for both the brand text and the credit.
            Defaults to ``"white"``, which is what reads on the dark canvases
            these watermarks are usually stamped on.
        credit: An optional credit line drawn along the bottom edge (a repo or
            attribution URL). `None` (default) draws no credit line; a blank
            string is refused rather than stamping an empty artist.
        credit_frac: The credit line's width as a fraction of the figure width,
            in ``(0, 1]``. Defaults to ``0.28``. Ignored when `credit` is
            `None`.
        credit_alpha: Opacity of the credit line, in ``[0, 1]``. Defaults to
            ``1.0`` -- the credit is information, not decoration.
        margin: The gap between the credit line and the bottom edge, as a
            fraction of the figure height, in ``[0, 1)``. Defaults to
            ``0.014``. Ignored when `credit` is `None`.

    Returns:
        tuple[Text, Text | None]: The brand-text artist and the credit-line
        artist, so either can be adjusted further. The second is `None` when no
        `credit` was given.

    Raises:
        ValueError: If `text` is not a non-empty string, if `credit` is given
            but is not a non-empty string, if `frac` or
            `credit_frac` is not in ``(0, 1]``, if `alpha` or `credit_alpha` is
            not in ``[0, 1]``, if `angle` is not finite, or if `margin` is not
            in ``[0, 1)``.

    Notes:
        Like `stamp_mark`, the size is baked at stamp time from the figure's
        current size, so call `stamp_watermark` **last** -- after any
        `tight_layout()` / layout finalization, and after the final
        `set_size_inches`. The proportion holds across dpi.

        It does **not** survive a later `set_size_inches`, and here the text
        differs from `stamp_mark`: a mark lives on an inset axes in
        figure-fraction coordinates and so keeps its share of a figure resized
        proportionally afterwards, whereas text is measured in points and keeps
        its *absolute* size, halving its share when the figure doubles. Stamp
        after the final resize and the two behave alike; stamp before one and
        only the mark follows.

        The font size is fitted by measuring the rendered text, which needs a
        renderer; on a figure that has never been drawn matplotlib creates one
        on demand, so no explicit `draw()` is required.

    Examples:
        - Stamp brand text across a figure and read back its share of it:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.styling.watermark import stamp_watermark
            >>> fig = plt.figure(figsize=(8, 4.5))
            >>> brand, credit = stamp_watermark(fig, "earthlens", frac=0.5)
            >>> credit is None
            True
            >>> box = brand.get_window_extent()
            >>> figure_box = fig.get_window_extent()
            >>> round(float(max(box.width / figure_box.width,
            ...                 box.height / figure_box.height)), 2)
            0.5
            >>> plt.close(fig)

            ```
        - Add a credit line along the bottom:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.styling.watermark import stamp_watermark
            >>> fig = plt.figure(figsize=(8, 4.5))
            >>> brand, credit = stamp_watermark(
            ...     fig, "earthlens", credit="github.com/serapeum-org/earthlens"
            ... )
            >>> credit.get_text()
            'github.com/serapeum-org/earthlens'
            >>> plt.close(fig)

            ```
        - An out-of-range opacity is refused, as on `stamp_mark`:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.styling.watermark import stamp_watermark
            >>> fig = plt.figure()
            >>> stamp_watermark(fig, "earthlens", alpha=1.5)
            Traceback (most recent call last):
                ...
            ValueError: alpha must be in [0, 1], got 1.5.

            ```

    See Also:
        stamp_mark: The image counterpart, for a corner logo.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"text must be a non-empty string, got {text!r}.")
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac!r}.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}.")
    if not np.isfinite(angle):
        raise ValueError(f"angle must be a finite number of degrees, got {angle!r}.")
    if credit is not None:
        if not isinstance(credit, str) or not credit.strip():
            raise ValueError(
                f"credit must be a non-empty string or None, got {credit!r}. A blank "
                f"credit would stamp an artist with nothing in it; pass None to draw "
                f"no credit line."
            )
        if not 0.0 < credit_frac <= 1.0:
            raise ValueError(f"credit_frac must be in (0, 1], got {credit_frac!r}.")
        if not 0.0 <= credit_alpha <= 1.0:
            raise ValueError(f"credit_alpha must be in [0, 1], got {credit_alpha!r}.")
        if not 0.0 <= margin < 1.0:
            raise ValueError(f"margin must be in [0, 1), got {margin!r}.")

    brand = fig.text(
        0.5,
        0.5,
        text,
        rotation=angle,
        ha="center",
        va="center",
        fontweight="bold",
        color=color,
        alpha=alpha,
        zorder=_WATERMARK_ZORDER,
    )
    brand.set_in_layout(False)
    _fit_text_to_frac(fig, brand, frac)

    if credit is None:
        return brand, None

    credit_artist = fig.text(
        0.5,
        margin,
        credit,
        ha="center",
        va="bottom",
        color=color,
        alpha=credit_alpha,
        zorder=_CREDIT_ZORDER,
        path_effects=[
            patheffects.withStroke(
                linewidth=_CREDIT_STROKE_WIDTH,
                foreground=_CREDIT_STROKE_COLOR,
                alpha=_CREDIT_STROKE_ALPHA,
            )
        ],
    )
    credit_artist.set_in_layout(False)
    _fit_text_to_frac(fig, credit_artist, credit_frac)
    return brand, credit_artist


def _float_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Validate a float ``0-1`` image array and scale it to ``uint8`` ``0-255``.

    A float image is the ``0-1`` contract; reject out-of-range or non-finite
    values rather than silently clipping (a ``0-255`` float array would
    otherwise flatten to white, and a ``NaN`` -- whose ``min``/``max``
    comparisons are all ``False`` -- would slip past a plain range check and
    cast to 0 with a ``RuntimeWarning``).

    Args:
        arr: A floating-dtype image array expected to hold finite values in
            ``[0, 1]``.

    Returns:
        np.ndarray: The array scaled to ``uint8`` ``0-255``.

    Raises:
        ValueError: If `arr` holds non-finite values (NaN/inf) or values
            outside ``[0, 1]``.
    """
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError(
            "a float image array must hold finite values in [0, 1]; "
            "it contains NaN or inf."
        )
    if arr.size and (arr.min() < -1e-6 or arr.max() > 1.0 + 1e-6):
        raise ValueError(
            "a float image array must hold values in [0, 1]; got range "
            f"[{float(arr.min()):.4g}, {float(arr.max()):.4g}]. "
            "For 0-255 data pass a uint8 array."
        )
    return (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)


def _as_rgba(path: str | os.PathLike | np.ndarray) -> np.ndarray:
    """Return the mark as an ``(H, W, 4)`` ``uint8`` RGBA array.

    Args:
        path: A file path `PIL` can open, or an ``(H, W, 3)`` / ``(H, W, 4)``
            array (``uint8`` ``0-255`` or float ``0-1``).

    Returns:
        np.ndarray: The image as ``uint8`` RGBA, with an opaque alpha added
        when the input is RGB.

    Raises:
        ValueError: If an array input is not ``(H, W, 3)`` / ``(H, W, 4)``, is a
            non-``uint8`` non-float array (e.g. ``uint16``, ``int32``, ``bool``),
            or is a float array with values outside ``[0, 1]`` or containing
            NaN/inf.
    """
    if isinstance(path, np.ndarray):
        arr = np.asarray(path)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(
                f"an image array must be (H, W, 3) or (H, W, 4); got shape {arr.shape}."
            )
        if np.issubdtype(arr.dtype, np.floating):
            arr = _float_to_uint8(arr)
        elif arr.dtype != np.uint8:
            # Any other non-float dtype (uint16, int32, bool, ...) would truncate
            # mod 256 under a bare uint8 cast and silently garble the mark.
            raise ValueError(
                f"a non-float image array must be uint8 (0-255); got dtype {arr.dtype}. "
                "Convert / rescale it to uint8 (or pass a float 0-1 array) first."
            )
        if arr.shape[2] == 3:
            opaque = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
            arr = np.concatenate([arr, opaque], axis=2)
        return arr
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"))


def _as_margins(margin: float | tuple[float, float]) -> tuple[float, float]:
    """Normalise `margin` to an ``(x, y)`` pair of figure fractions.

    Args:
        margin: A scalar applied to both axes, or an ``(x, y)`` pair. A pair is
            needed when a mark must tuck hard into a corner on one axis while
            keeping a gap on the other.

    Returns:
        tuple[float, float]: The ``(x, y)`` margins.

    Raises:
        ValueError: If `margin` is not a scalar or a 2-sequence, or if either
            component is outside ``[0, 1)``.
    """
    if isinstance(margin, (int, float)) and not isinstance(margin, bool):
        pair = (float(margin), float(margin))
    else:
        try:
            values = tuple(margin)  # type: ignore[arg-type]
        except TypeError:
            raise ValueError(
                f"margin must be a number or an (x, y) pair, got {margin!r}."
            ) from None
        if len(values) != 2:
            raise ValueError(
                f"margin must be a number or an (x, y) pair, got {margin!r}."
            )
        pair = (float(values[0]), float(values[1]))
    for name, value in zip(("x", "y"), pair):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"margin must be in [0, 1), got {name}={value!r}.")
    return pair


def _corner_origin(
    corner: str, width: float, height: float, margin_x: float, margin_y: float
) -> tuple[float, float]:
    """Return the ``(x0, y0)`` figure-fraction origin for a corner-anchored rect.

    Args:
        corner: One of the `_CORNERS` anchors.
        width: The mark width in figure fraction.
        height: The mark height in figure fraction.
        margin_x: The horizontal edge gap in figure fraction.
        margin_y: The vertical edge gap in figure fraction.

    Returns:
        tuple[float, float]: The bottom-left ``(x0, y0)`` of the mark's rect.
    """
    at_right = corner.endswith("right")
    at_top = corner.startswith("upper")
    x0 = (1.0 - margin_x - width) if at_right else margin_x
    y0 = (1.0 - margin_y - height) if at_top else margin_y
    return x0, y0


def _composite_halo(image: np.ndarray, blur: float) -> tuple[np.ndarray, float, float]:
    """Composite a mark over its own blurred halo, returning the growth factors.

    Pads the mark symmetrically so the blur's tail has room, blurs a black copy
    of its alpha to make the halo, and composites the mark back over it. The
    halo is *centred* on the mark rather than offset: the mark goes over
    arbitrary imagery, and a symmetric halo reads the same whichever way the
    background falls.

    Args:
        image: The RGBA ``uint8`` mark; its alpha channel drives the halo shape.
        blur: The blur sigma as a fraction of the mark's own unpadded width.

    Returns:
        tuple[np.ndarray, float, float]: The composited RGBA ``uint8`` canvas,
        and the factors by which it is wider and taller than the mark. The
        caller grows the axes rect by these so the *mark* still measures `frac`.
    """
    img_h, img_w = image.shape[:2]
    # Sigma comes off the mark's own width, so `blur` means what it says; taking
    # it off the padded width instead would inflate the effective blur and leave
    # the pad too small for the tail it was sized to contain.
    sigma = blur * img_w
    pad = max(1, int(round(_HALO_SIGMAS * sigma)))

    alpha = np.pad(image[..., 3], pad, mode="constant", constant_values=0)
    blurred = np.asarray(Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(sigma)))

    halo = np.zeros(blurred.shape + (4,), dtype=np.float64)  # black, alpha-only
    halo[..., 3] = blurred / 255.0 * _HALO_ALPHA

    mark = np.zeros_like(halo)
    mark[pad : pad + img_h, pad : pad + img_w] = image / 255.0

    out = alpha_over(mark, halo)
    pad_h, pad_w = out.shape[:2]
    return (out * 255).round().astype(np.uint8), pad_w / img_w, pad_h / img_h


class WatermarkMixin:
    """Glyph-side sugar over `stamp_mark` and `stamp_watermark`.

    Both stamps act on a whole `Figure`, so a glyph can offer them once it has
    one: `glyph.stamp_mark(logo)` instead of importing the function and passing
    `glyph.fig` by hand. The free functions remain the primitives and are
    unchanged -- this only spares the import, in the same shape
    `cleopatra.basemap.geo.GeoMixin` uses to expose `styling.furniture`'s scale
    bar and north arrow.

    A figure is what these need, and glyphs spell it differently. `Glyph` keeps
    the one it rendered on in `fig`. `HistogramGlyph` and `TexturedGlobeGlyph`
    do not inherit `Glyph` and use `_fig` for something else -- the figure bound
    at *construction*, consulted on every render to decide where to draw -- so
    they record the figure actually drawn on as `_rendered_fig`.
    `_watermark_figure` prefers that, then `fig`, then `_fig`, then the axes'
    own figure, so the mixin works on any of them without first making them
    agree on a name.

    Note that a figure may carry several glyphs. The stamp lands on the whole
    figure, not on the glyph's own axes, whichever glyph it was called through.
    """

    def _watermark_figure(self) -> Figure:
        """Resolve the figure to stamp.

        Returns:
            Figure: The glyph's figure.

        Raises:
            ValueError: If the glyph has no figure yet, which means it has not
                been rendered.
        """
        for name in ("fig", "_rendered_fig", "_fig"):
            figure = getattr(self, name, None)
            if figure is not None:
                return figure
        for name in ("ax", "_ax"):
            axes = getattr(self, name, None)
            if axes is not None:
                figure = axes.get_figure()
                if figure is not None:
                    return figure
        raise ValueError(
            f"{type(self).__name__} has no figure to stamp yet -- render it first "
            f"(e.g. plot()), or call cleopatra.styling.watermark.stamp_mark on a "
            f"figure of your own."
        )

    def stamp_mark(self, path: str | os.PathLike | np.ndarray, **kwargs: Any) -> Axes:
        """Stamp a logo image on this glyph's figure.

        Thin sugar over `cleopatra.styling.watermark.stamp_mark`; the free
        function remains available for a figure this glyph does not own.

        Args:
            path: The mark image, as `stamp_mark` accepts it.
            **kwargs: Forwarded verbatim (`frac`, `corner`, `margin`, `shadow`,
                `blur`).

        Returns:
            Axes: The frameless inset axes the mark was drawn on.

        Raises:
            ValueError: If the glyph has not been rendered yet, or as
                `stamp_mark` raises.

        See Also:
            cleopatra.styling.watermark.stamp_mark: The underlying function.
        """
        # The bare name is the module-level function, not this method: a method
        # name never enters the enclosing scope its body is resolved in.
        return stamp_mark(self._watermark_figure(), path, **kwargs)

    def stamp_watermark(self, text: str, **kwargs: Any) -> tuple[Text, Text | None]:
        """Stamp diagonal brand text on this glyph's figure.

        Thin sugar over `cleopatra.styling.watermark.stamp_watermark`; the free
        function remains available for a figure this glyph does not own.

        Args:
            text: The brand text.
            **kwargs: Forwarded verbatim (`frac`, `angle`, `alpha`, `color`,
                `credit`, `credit_frac`, `credit_alpha`, `margin`).

        Returns:
            tuple[Text, Text | None]: The brand-text artist and the credit-line
            artist, the second being `None` when no `credit` was given.

        Raises:
            ValueError: If the glyph has not been rendered yet, or as
                `stamp_watermark` raises.

        See Also:
            cleopatra.styling.watermark.stamp_watermark: The underlying
                function.

        Examples:
            - Stamp a logo and brand text without importing either:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
                >>> glyph = ArrayGlyph(np.arange(60.0).reshape(6, 10))
                >>> _ = glyph.plot()
                >>> brand, credit = glyph.stamp_watermark("cleopatra")
                >>> brand.get_text()
                'cleopatra'
                >>> plt.close("all")

                ```
        """
        return stamp_watermark(self._watermark_figure(), text, **kwargs)
