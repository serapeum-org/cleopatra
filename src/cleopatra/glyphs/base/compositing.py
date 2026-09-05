"""Array-level alpha compositing.

A single, dependency-free implementation of the Porter-Duff **"over"** operator
-- stacking a semi-transparent image over another. It is the leaf primitive the
render stack reaches for whenever two image arrays are combined by alpha, so the
formula (with its two fiddly edge cases: un-premultiplying the blend, and the
divide-by-zero where the output is fully transparent) lives in exactly one place
rather than being re-derived per caller.

Arrays are **channel-last** ``(H, W, C)`` -- matplotlib's and Pillow's image
layout, and cleopatra's throughout. A caller that holds band-first ``(C, H, W)``
rasters (the GDAL/rasterio convention) transposes at its own boundary, e.g.
``np.moveaxis(arr, 0, -1)``, before calling in.

This module depends on nothing but NumPy, and must stay that way: keeping it
free of any first-party `cleopatra` import is what lets both `styling` and
`glyphs` use it without an import cycle (`styling.watermark` imports it, while
`glyphs.base` already imports `styling`).

Import from the submodule, matching the package convention (nothing is
re-exported at the package roots): `from cleopatra.glyphs.base.compositing import
alpha_over`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["alpha_over"]


def alpha_over(foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Composite an RGBA foreground over an RGB(A) background (Porter-Duff "over").

    The result is the foreground's colour weighted by its own alpha, plus the
    background's weighted by whatever coverage the foreground leaves over. The
    background's channel count decides the result:

    - **RGB background** ``(H, W, 3)`` -- an opaque canvas; the result is a
      3-band ``fg_rgb * fg_a + background * (1 - fg_a)`` with no alpha channel.
    - **RGBA background** ``(H, W, 4)`` -- the result is 4-band with
      ``out_a = fg_a + bg_a * (1 - fg_a)``; the blended colour is
      un-premultiplied by ``out_a``, guarded to stay black where nothing covers
      at all (``out_a == 0``) rather than dividing by zero.

    Args:
        foreground: An ``(H, W, 4)`` RGBA float array with values in ``[0, 1]``.
        background: An ``(H, W, 3)`` RGB or ``(H, W, 4)`` RGBA float array in
            ``[0, 1]``, sharing the foreground's ``(H, W)``.

    Returns:
        np.ndarray: The composited floating-point array -- ``(H, W, 3)`` when
        the background is RGB, ``(H, W, 4)`` when it is RGBA. Floating-point
        inputs keep their own width -- they are never upcast, so a ``float32``
        pair yields a ``float32`` result and a ``float16`` pair blends in
        ``float16`` (at ``float16`` precision, not a promise of more); integer
        and boolean inputs are promoted to ``float64``.

    Raises:
        ValueError: If the foreground is not ``(H, W, 4)``, the background is not
            ``(H, W, 3)`` or ``(H, W, 4)``, or the two differ in height or width.

    Note:
        The ``[0, 1]`` input range is a trusted precondition, not validated: the
        divide-by-zero guard is sound only for non-negative alphas, where
        ``out_a == 0`` implies both alphas are zero and the colour numerator is
        zero too. Out-of-range inputs are neither rejected nor clamped.

    Examples:
        - Half-transparent red over an opaque blue canvas blends 50/50 and drops
          the alpha channel (3-band result):
            ```python
            >>> import numpy as np
            >>> fg = np.array([[[1.0, 0.0, 0.0, 0.5]]])
            >>> bg = np.array([[[0.0, 0.0, 1.0]]])
            >>> alpha_over(fg, bg).round(2).ravel().tolist()
            [0.5, 0.0, 0.5]

            ```
        - The same mark over an RGBA canvas keeps a 4-band result and reports the
          combined coverage in the alpha channel:
            ```python
            >>> import numpy as np
            >>> fg = np.array([[[1.0, 0.0, 0.0, 0.5]]])
            >>> bg = np.array([[[0.0, 0.0, 1.0, 1.0]]])
            >>> alpha_over(fg, bg).round(2).ravel().tolist()
            [0.5, 0.0, 0.5, 1.0]

            ```
        - Fully transparent over fully transparent stays transparent instead of
          dividing by a zero output alpha:
            ```python
            >>> import numpy as np
            >>> alpha_over(np.zeros((1, 1, 4)), np.zeros((1, 1, 4))).ravel().tolist()
            [0.0, 0.0, 0.0, 0.0]

            ```
        - A background that is neither RGB nor RGBA is rejected:
            ```python
            >>> import numpy as np
            >>> alpha_over(np.zeros((1, 1, 4)), np.zeros((1, 1, 2)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: alpha_over background must be an (H, W, 3) RGB or (H, W, 4) RGBA array, got shape ...

            ```
    """
    # Coerce to floating point for the division, but keep the caller's own float
    # width -- a float32 raster stays float32 rather than doubling to float64.
    # Only non-floating inputs (int, bool) are promoted, to float64.
    foreground = np.asarray(foreground)
    background = np.asarray(background)
    if not np.issubdtype(foreground.dtype, np.floating):
        foreground = foreground.astype(float)
    if not np.issubdtype(background.dtype, np.floating):
        background = background.astype(float)

    if foreground.ndim != 3 or foreground.shape[-1] != 4:
        raise ValueError(
            "alpha_over foreground must be an (H, W, 4) RGBA array, got shape "
            f"{foreground.shape}."
        )
    if background.ndim != 3 or background.shape[-1] not in (3, 4):
        raise ValueError(
            "alpha_over background must be an (H, W, 3) RGB or (H, W, 4) RGBA "
            f"array, got shape {background.shape}."
        )
    if foreground.shape[:2] != background.shape[:2]:
        raise ValueError(
            "alpha_over foreground and background must share the same (H, W), got "
            f"{foreground.shape[:2]} and {background.shape[:2]}."
        )

    fg_rgb = foreground[..., :3]
    fg_a = foreground[..., 3:4]

    if background.shape[-1] == 3:
        # Opaque canvas: straight "over", no output alpha to carry.
        blended: np.ndarray = fg_rgb * fg_a + background * (1.0 - fg_a)
        return blended

    bg_rgb = background[..., :3]
    bg_a = background[..., 3:4]
    out_a = fg_a + bg_a * (1.0 - fg_a)
    # Un-premultiply by the output alpha, but guard the pixels nothing covers at
    # all -- their colour is arbitrary, so keep it black rather than divide by 0.
    safe = np.where(out_a > 0.0, out_a, 1.0)
    rgb = (fg_rgb * fg_a + bg_rgb * bg_a * (1.0 - fg_a)) / safe
    return np.concatenate([np.where(out_a > 0.0, rgb, 0.0), out_a], axis=-1)
