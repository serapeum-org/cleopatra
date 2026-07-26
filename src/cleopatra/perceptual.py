"""Perceptual colour-space primitives and palette generators (numpy-only).

matplotlib interpolates colormaps in RGB, which is perceptually non-uniform:
equal steps in the data map to visually uneven steps, producing banding and
dead zones. This module provides the one primitive that fixes it -- a closed-form
`sRGB <-> CIELAB` transform (pure numpy, no new dependency) -- and builds three
things on top of it:

- `interp_perceptual` / `perceptual_colormap`: interpolate colour anchors in
    CIELAB (sampled at uniform perceptual arc-length), so a hand-authored ramp
    progresses evenly to the eye. This is how cleopatra's own `HAZE`/`CAMS`/
    `FLAME` colormaps are built.
- `make_diverging`: two lightness-balanced Lab arms meeting at a neutral centre
    -- a perceptually-uniform diverging map from just two endpoint colours.
- `make_categorical`: the *glasbey* method -- greedily pick colours with the
    maximum minimum Lab distance, so N classes stay maximally distinguishable.

Nothing here imports or copies data from cmocean / cmcrameri / colorcet; the
quality comes from the *method* (designing in a perceptual space), which is pure
math. For scientific-grade sequential and cyclic maps, prefer matplotlib's own
`viridis` family and `twilight` (already optimised in CAM02-UCS); this module
earns its keep on diverging, categorical, and smoothing bespoke domain ramps.

Examples:
    >>> import numpy as np
    >>> from cleopatra.perceptual import srgb_to_lab, make_categorical
    >>> bool(np.allclose(srgb_to_lab(np.array([1.0, 1.0, 1.0])), [100, 0, 0], atol=1e-3))
    True
    >>> len(make_categorical(8))
    8
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap

__all__ = [
    "srgb_to_lab",
    "lab_to_srgb",
    "interp_perceptual",
    "perceptual_colormap",
    "make_diverging",
    "make_categorical",
    "perceptual_uniformity",
]

# sRGB (D65) <-> CIE XYZ, and the CIELAB constants. All closed-form; no deps.
_M_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
_DELTA = 6.0 / 29.0


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Undo the sRGB gamma companding (electro-optical transfer)."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Apply the sRGB gamma companding to linear light."""
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.abs(c) ** (1 / 2.4) - 0.055)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB colours to CIELAB.

    Args:
        rgb: An `(..., 3)` array (or array-like) of sRGB values in `[0, 1]`.

    Returns:
        numpy.ndarray: An `(..., 3)` array of CIELAB `(L*, a*, b*)`, where
        `L*` runs 0 (black) to 100 (white) and `a*`/`b*` are the opponent
        colour axes.

    Examples:
        ```python
        >>> import numpy as np
        >>> from cleopatra.perceptual import srgb_to_lab
        >>> bool(np.allclose(srgb_to_lab(np.array([0.0, 0.0, 0.0])), [0, 0, 0], atol=1e-6))
        True
        >>> float(round(srgb_to_lab(np.array([1.0, 1.0, 1.0]))[0], 2))
        100.0

        ```
    """
    rgb = np.asarray(rgb, dtype=float)
    xyz = _srgb_to_linear(rgb) @ _M_RGB2XYZ.T
    t = xyz / _WHITE_D65
    f = np.where(t > _DELTA**3, np.cbrt(t), t / (3 * _DELTA**2) + 4 / 29)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB colours back to sRGB, clipped to the `[0, 1]` gamut.

    Args:
        lab: An `(..., 3)` array of CIELAB `(L*, a*, b*)`.

    Returns:
        numpy.ndarray: An `(..., 3)` array of sRGB values clipped to `[0, 1]`
        (out-of-gamut Lab colours are clamped, not wrapped).

    Examples:
        ```python
        >>> import numpy as np
        >>> from cleopatra.perceptual import srgb_to_lab, lab_to_srgb
        >>> rgb = np.array([0.2, 0.6, 0.9])
        >>> bool(np.allclose(lab_to_srgb(srgb_to_lab(rgb)), rgb, atol=1e-6))
        True

        ```
    """
    lab = np.asarray(lab, dtype=float)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    t = np.where(f > _DELTA, f**3, 3 * _DELTA**2 * (f - 4 / 29))
    rgb = _linear_to_srgb((t * _WHITE_D65) @ _M_XYZ2RGB.T)
    return np.clip(rgb, 0.0, 1.0)


def _to_rgb(anchors: Sequence) -> np.ndarray:
    """Parse hex strings / names / RGB tuples into an `(m, 3)` sRGB array."""
    return np.array([mcolors.to_rgb(c) for c in anchors], dtype=float)


def interp_perceptual(anchors: Sequence, n: int = 256) -> np.ndarray:
    """Interpolate colour anchors in CIELAB at uniform perceptual arc-length.

    Unlike RGB interpolation, consecutive output colours are (near) equally
    spaced in perceived colour difference, so the ramp reads as an even
    progression. The exact endpoint anchors are preserved.

    Args:
        anchors: Two or more colours (hex strings, names, or RGB triplets) to
            interpolate between, ordered low to high.
        n: Number of output colours. Defaults to 256.

    Returns:
        numpy.ndarray: An `(n, 3)` sRGB lookup table in `[0, 1]`.

    Raises:
        ValueError: If fewer than two anchors are given, or `n < 2`.

    Examples:
        ```python
        >>> import numpy as np
        >>> from cleopatra.perceptual import interp_perceptual
        >>> lut = interp_perceptual(["#ffffff", "#ff6a00", "#2a0800"], n=16)
        >>> lut.shape
        (16, 3)
        >>> bool(np.allclose(lut[0], [1, 1, 1]))  # first anchor preserved exactly
        True

        ```
    """
    rgb = _to_rgb(anchors)
    if rgb.shape[0] < 2:
        raise ValueError("interp_perceptual needs at least two anchor colours")
    if n < 2:
        raise ValueError("n must be >= 2")
    lab = srgb_to_lab(rgb)
    seg = np.sqrt(((lab[1:] - lab[:-1]) ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] == 0:  # all anchors identical
        return np.repeat(rgb[:1], n, axis=0)
    t = cum / cum[-1]
    x = np.linspace(0.0, 1.0, n)
    out = lab_to_srgb(np.column_stack([np.interp(x, t, lab[:, k]) for k in range(3)]))
    out[0], out[-1] = rgb[0], rgb[-1]  # snap exact endpoints
    return out


def perceptual_colormap(name: str, anchors: Sequence, n: int = 256) -> LinearSegmentedColormap:
    """Build a `LinearSegmentedColormap` from anchors interpolated in CIELAB.

    A perceptually-uniform, drop-in replacement for
    `matplotlib.colors.LinearSegmentedColormap.from_list`: the same call shape and
    (continuous) return type, but the anchors are interpolated in CIELAB so the
    ramp reads as an even progression rather than banding.

    Args:
        name: Name for the resulting colormap.
        anchors: Two or more colours to interpolate between.
        n: Number of quantisation levels. Defaults to 256.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The perceptually-interpolated map.

    Examples:
        ```python
        >>> from cleopatra.perceptual import perceptual_colormap
        >>> cmap = perceptual_colormap("dust", ["#ffffff", "#ff6a00", "#2a0800"])
        >>> cmap.name
        'dust'
        >>> tuple(float(round(v, 3)) for v in cmap(0.0))  # starts at the first anchor
        (1.0, 1.0, 1.0, 1.0)

        ```
    """
    return LinearSegmentedColormap.from_list(name, interp_perceptual(anchors, n), N=n)


def make_diverging(
    low,
    high,
    n: int = 256,
    center: str = "#f4f4f4",
    balance: bool = True,
    name: str = "diverging",
) -> LinearSegmentedColormap:
    """Construct a perceptually-uniform diverging colormap from two endpoints.

    Builds two Lab-interpolated arms from a light neutral `center` out to each
    endpoint, giving the symmetric lightness profile (a peak at the centre) a
    good diverging map needs. With `balance=True` the two endpoints are forced
    to equal lightness first, so neither side visually dominates.

    Args:
        low: Colour for the low end of the scale.
        high: Colour for the high end of the scale.
        n: Total number of levels. Defaults to 256.
        center: The neutral midpoint colour. Defaults to a near-white grey.
        balance: If `True` (default), equalise the two endpoints' `L*` so the
            arms are lightness-symmetric.
        name: Name for the resulting colormap. Defaults to `"diverging"`.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The diverging colormap.

    Examples:
        ```python
        >>> from cleopatra.perceptual import make_diverging, srgb_to_lab
        >>> cmap = make_diverging("#762a83", "#1b7837")
        >>> cmap.N
        256
        >>> bool(srgb_to_lab(cmap(0.5)[:3])[0] > srgb_to_lab(cmap(0.0)[:3])[0])
        True

        ```
    """
    lo = srgb_to_lab(_to_rgb([low])[0])
    hi = srgb_to_lab(_to_rgb([high])[0])
    if balance:
        lo[0] = hi[0] = min(lo[0], hi[0])
    low_rgb, high_rgb = lab_to_srgb(lo), lab_to_srgb(hi)
    half = n // 2
    arm_lo = interp_perceptual([low_rgb, center], half)
    arm_hi = interp_perceptual([center, high_rgb], n - half)
    return LinearSegmentedColormap.from_list(name, np.vstack([arm_lo, arm_hi]), N=n)


def make_categorical(
    n: int, l_range: tuple[float, float] = (35.0, 82.0), c_min: float = 25.0
) -> list[str]:
    """Generate `n` maximally-distinguishable categorical colours (glasbey method).

    Greedily selects, from a mid-lightness / chromatic gamut, the colour whose
    *minimum* CIELAB distance to those already chosen is largest -- the same
    max-min strategy the glasbey / colorcet categorical palettes use. Fully
    deterministic and dependency-free.

    Args:
        n: Number of distinct colours to generate.
        l_range: Inclusive `(min, max)` `L*` band candidates must fall in, so
            colours are neither too dark nor too washed out. Defaults to
            `(35, 82)`.
        c_min: Minimum chroma (`sqrt(a*^2 + b*^2)`) a candidate must have, so
            colours stay vivid enough to tell apart. Defaults to `25`.

    Returns:
        list[str]: `n` hex colour strings.

    Raises:
        ValueError: If `n < 1`.

    Examples:
        ```python
        >>> from cleopatra.perceptual import make_categorical
        >>> cols = make_categorical(5)
        >>> len(cols) == len(set(cols)) == 5  # all distinct
        True
        >>> all(c.startswith("#") for c in cols)
        True

        ```
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    grid = np.linspace(0.04, 0.96, 14)
    cand = np.array(np.meshgrid(grid, grid, grid)).reshape(3, -1).T
    lab = srgb_to_lab(cand)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    keep = (lab[:, 0] >= l_range[0]) & (lab[:, 0] <= l_range[1]) & (chroma >= c_min)
    cand, lab = cand[keep], lab[keep]
    start = int(np.argmax(np.hypot(lab[:, 1], lab[:, 2])))
    chosen = [start]
    dmin = np.sqrt(((lab - lab[start]) ** 2).sum(axis=1))
    for _ in range(min(n, len(cand)) - 1):
        k = int(np.argmax(dmin))
        chosen.append(k)
        dmin = np.minimum(dmin, np.sqrt(((lab - lab[k]) ** 2).sum(axis=1)))
    return [mcolors.to_hex(cand[i]) for i in chosen]


def perceptual_uniformity(cmap: Colormap | np.ndarray, n: int = 256) -> float:
    """Score how perceptually even a colormap's steps are (0 == perfectly even).

    Returns the coefficient of variation of the per-step CIELAB distance: the
    standard deviation of `DeltaE` between consecutive samples divided by their
    mean. Lower is more uniform. Useful for comparing an RGB-interpolated ramp
    against its `interp_perceptual` counterpart.

    Args:
        cmap: A matplotlib `Colormap`, or an `(m, 3)` sRGB lookup table.
        n: Number of samples to take when `cmap` is a `Colormap`. Defaults 256.

    Returns:
        float: The coefficient of variation of the per-step `DeltaE`.

    Examples:
        ```python
        >>> from cleopatra.perceptual import perceptual_colormap, perceptual_uniformity
        >>> from matplotlib.colors import LinearSegmentedColormap
        >>> anchors = ["#ffffff", "#ff5fc9", "#200018"]
        >>> lab = perceptual_uniformity(perceptual_colormap("p", anchors))
        >>> rgb = perceptual_uniformity(LinearSegmentedColormap.from_list("r", anchors))
        >>> bool(lab < rgb)  # Lab interpolation is more even than RGB
        True

        ```
    """
    if isinstance(cmap, Colormap):
        lut = cmap(np.linspace(0, 1, n))[:, :3]
    else:
        lut = np.asarray(cmap, dtype=float)[:, :3]
    lab = srgb_to_lab(lut)
    de = np.sqrt(((lab[1:] - lab[:-1]) ** 2).sum(axis=1))
    return float(de.std() / de.mean())
