"""Figure watermark / brand-mark helper.

`stamp_mark` places a logo or watermark image onto a matplotlib `Figure`,
sized as a *fraction of the figure* (so it stays proportional across the
several dpis a figure is often exported at -- an MP4 master, a smaller web
copy, a GIF) and anchored in one of the four corners, with an optional
gaussian-blurred drop shadow so the mark reads on a busy or dark canvas.

This is a presentation helper, not a glyph: it takes a finished `Figure` and
draws on top of it via a frameless inset axes in figure-fraction coordinates
(the dpi-independent counterpart of `Figure.figimage`, which is pixel-based).
It covers only single-image, corner-anchored marks -- text watermarks,
tiled / repeated marks, and any licensing / provenance semantics are
deliberately out of scope.

The gaussian blur for the shadow uses `PIL` (Pillow), already a hard cleopatra
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

#: Drop-shadow tuning, all as fractions of the mark's own size / alpha.
_SHADOW_PAD = 0.14  #: transparent border added around the mark before blurring
_SHADOW_BLUR_FRAC = 0.5  #: gaussian blur radius, as a fraction of the pad
_SHADOW_OFFSET = 0.045  #: shadow shift down-and-right, as a fraction of the mark
_SHADOW_ALPHA = 0.5  #: peak shadow opacity

#: Inset zorders: both marks sit above ordinary figure content, shadow under mark.
_SHADOW_ZORDER = 1_000_000
_MARK_ZORDER = 1_000_001


def stamp_mark(
    fig: "Figure",
    path: "str | os.PathLike | np.ndarray",
    *,
    frac: float = 0.11,
    corner: str = "lower right",
    margin: float = 0.025,
    shadow: bool = True,
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
        margin: The gap between the mark and the figure edges, as a fraction of
            the figure, in ``[0, 1)``. Defaults to ``0.025``.
        shadow: Whether to draw a gaussian-blurred drop shadow beneath the mark
            so it separates from a busy or dark canvas. Defaults to ``True``.

    Returns:
        Axes: The frameless inset axes the mark was drawn on, so the caller can
        further adjust it (e.g. ``ax.set_zorder(...)``). The optional shadow is
        drawn on its own separate axes beneath this one.

    Raises:
        ValueError: If `corner` is not one of the four accepted anchors, if
            `frac` is not in ``(0, 1]``, if `margin` is not in ``[0, 1)``, or
            if an image array does not have shape ``(H, W, 3)`` / ``(H, W, 4)``.

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
    if not 0.0 <= margin < 1.0:
        raise ValueError(f"margin must be in [0, 1), got {margin!r}.")

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
    x0, y0 = _corner_origin(corner, width, height, margin)

    if shadow:
        _stamp_shadow(fig, image, x0, y0, width, height)

    ax = fig.add_axes((x0, y0, width, height), frameon=False, zorder=_MARK_ZORDER)
    ax.imshow(image, aspect="auto", interpolation="antialiased")
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
            # would otherwise flatten to all-white).
            if arr.size and (arr.min() < -1e-6 or arr.max() > 1.0 + 1e-6):
                raise ValueError(
                    "a float image array must hold values in [0, 1]; got range "
                    f"[{float(arr.min()):.4g}, {float(arr.max()):.4g}]. "
                    "For 0-255 data pass a uint8 array."
                )
            arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
        elif arr.dtype != np.uint8:
            # Any other integer dtype (uint16, int32, bool, ...) would truncate
            # mod 256 under a bare uint8 cast and silently garble the mark.
            raise ValueError(
                f"an integer image array must be uint8 (0-255); got dtype {arr.dtype}. "
                "Convert / rescale it to uint8 (or pass a float 0-1 array) first."
            )
        if arr.shape[2] == 3:
            opaque = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
            arr = np.concatenate([arr, opaque], axis=2)
        return arr
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"))


def _corner_origin(corner: str, width: float, height: float, margin: float) -> tuple[float, float]:
    """Return the ``(x0, y0)`` figure-fraction origin for a corner-anchored rect.

    Args:
        corner: One of the `_CORNERS` anchors.
        width: The mark width in figure fraction.
        height: The mark height in figure fraction.
        margin: The edge gap in figure fraction.

    Returns:
        tuple[float, float]: The bottom-left ``(x0, y0)`` of the mark's axes.
    """
    at_right = corner.endswith("right")
    at_top = corner.startswith("upper")
    x0 = (1.0 - margin - width) if at_right else margin
    y0 = (1.0 - margin - height) if at_top else margin
    return x0, y0


def _stamp_shadow(
    fig: "Figure", image: np.ndarray, x0: float, y0: float, width: float, height: float
) -> "Axes":
    """Draw a gaussian-blurred drop shadow beneath a mark on its own axes.

    Builds the shadow from the mark's alpha channel: pad it (to give the blur
    room), gaussian-blur it, tint it black, and place it in a slightly larger,
    down-and-right-offset frameless axes under the mark so the mark separates
    from a busy / dark canvas.

    Args:
        fig: The figure to draw on.
        image: The RGBA ``uint8`` mark (its alpha drives the shadow shape).
        x0: The mark's figure-fraction left edge.
        y0: The mark's figure-fraction bottom edge.
        width: The mark's figure-fraction width.
        height: The mark's figure-fraction height.

    Returns:
        Axes: The frameless axes the shadow was drawn on.
    """
    img_h, img_w = image.shape[:2]
    pad = max(1, int(round(_SHADOW_PAD * max(img_h, img_w))))
    alpha = np.pad(image[..., 3], pad, mode="constant", constant_values=0)
    blurred = np.asarray(
        Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(pad * _SHADOW_BLUR_FRAC))
    )
    shadow = np.zeros(blurred.shape + (4,), dtype=np.uint8)
    shadow[..., 3] = (blurred.astype(np.float64) * _SHADOW_ALPHA).astype(np.uint8)

    # The mark occupies (img_w, img_h) centred within the padded (pad_w, pad_h),
    # so scale the mark's rect by the same ratio to keep the shadow aligned, then
    # nudge it down-and-right for the drop-shadow offset.
    pad_h, pad_w = shadow.shape[:2]
    shadow_w = width * pad_w / img_w
    shadow_h = height * pad_h / img_h
    shadow_x = x0 - (shadow_w - width) / 2.0 + _SHADOW_OFFSET * width
    shadow_y = y0 - (shadow_h - height) / 2.0 - _SHADOW_OFFSET * height

    sax = fig.add_axes(
        (shadow_x, shadow_y, shadow_w, shadow_h), frameon=False, zorder=_SHADOW_ZORDER
    )
    sax.imshow(shadow, aspect="auto", interpolation="antialiased")
    sax.axis("off")
    sax.set_in_layout(False)
    return sax
