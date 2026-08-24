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
It covers only single-image, corner-anchored marks -- text watermarks,
tiled / repeated marks, and any licensing / provenance semantics are
deliberately out of scope.

The gaussian blur for the halo uses `PIL` (Pillow), already a hard cleopatra
dependency, so no new dependency (and no SciPy) is pulled in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageFilter

if TYPE_CHECKING:
    import os

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["stamp_mark"]

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


def stamp_mark(
    fig: "Figure",
    path: "str | os.PathLike | np.ndarray",
    *,
    frac: float = 0.11,
    corner: str = "lower right",
    margin: "float | tuple[float, float]" = 0.025,
    shadow: bool = True,
    blur: float = DEFAULT_BLUR,
) -> "Axes":
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
            ``(x, y)`` pair in ``[0, 1)``, if `blur` is negative, or if
            an image array is out of contract (wrong shape, a non-``uint8``
            integer dtype, or a float outside ``[0, 1]`` -- see `_as_rgba`).
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


def _as_rgba(path: "str | os.PathLike | np.ndarray") -> np.ndarray:
    """Return the mark as an ``(H, W, 4)`` ``uint8`` RGBA array.

    Args:
        path: A file path `PIL` can open, or an ``(H, W, 3)`` / ``(H, W, 4)``
            array (``uint8`` ``0-255`` or float ``0-1``).

    Returns:
        np.ndarray: The image as ``uint8`` RGBA, with an opaque alpha added
        when the input is RGB.

    Raises:
        ValueError: If an array input is not ``(H, W, 3)`` / ``(H, W, 4)``, is a
            non-``uint8`` integer array, or is a float array with values outside
            ``[0, 1]``.
    """
    if isinstance(path, np.ndarray):
        arr = np.asarray(path)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(
                f"an image array must be (H, W, 3) or (H, W, 4); got shape {arr.shape}."
            )
        if np.issubdtype(arr.dtype, np.floating):
            # A float image is the ``0-1`` contract; reject clearly out-of-range
            # values rather than silently clipping (e.g. a ``0-255`` float array
            # would otherwise flatten to all-white). Reject non-finite values
            # first: a NaN makes `min()`/`max()` NaN, whose comparisons are all
            # False, so it would otherwise slip past the range check and cast to
            # 0 with a RuntimeWarning -- the very silent garbling this guards.
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
            arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
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


def _as_margins(margin: "float | tuple[float, float]") -> tuple[float, float]:
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


def _alpha_over(foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Composite RGBA over RGBA (Porter-Duff "over"), un-premultiplied.

    The colour a pixel ends up with is the foreground's, weighted by its own
    alpha, plus the background's weighted by whatever coverage the foreground
    left over -- so the halo contributes black only where the mark does not
    already cover, including partially across a soft edge.

    Args:
        foreground: An ``(H, W, 4)`` float array in ``[0, 1]``.
        background: An ``(H, W, 4)`` float array in ``[0, 1]``, same shape.

    Returns:
        np.ndarray: The composited ``(H, W, 4)`` float array in ``[0, 1]``.

    Notes:
        This is the same operator issue #306 proposes to centralise as
        `glyphs.base.compositing.alpha_over`; swap this private helper for it
        once that lands rather than keeping two copies.
    """
    fg_a = foreground[..., 3:4]
    bg_a = background[..., 3:4]
    out_a = fg_a + bg_a * (1.0 - fg_a)
    # Guard the un-premultiply where nothing covers at all; those pixels are
    # fully transparent, so their colour is arbitrary -- keep it at zero.
    safe = np.where(out_a > 0.0, out_a, 1.0)
    rgb = (
        foreground[..., :3] * fg_a + background[..., :3] * bg_a * (1.0 - fg_a)
    ) / safe
    return np.concatenate([np.where(out_a > 0.0, rgb, 0.0), out_a], axis=2)


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

    out = _alpha_over(mark, halo)
    pad_h, pad_w = out.shape[:2]
    return (out * 255).round().astype(np.uint8), pad_w / img_w, pad_h / img_h
