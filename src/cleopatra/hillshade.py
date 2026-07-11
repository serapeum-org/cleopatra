"""Relief shading (hillshade) for surface glyphs.

Wide-range DEMs cannot be read from colour alone: a flat coastal plain and a
high plateau land on different colours but both look featureless. The fix is to
blend terrain **illumination** into the colour-mapped elevation, so the surface
reads by *form* (slopes, ridges, valleys) independent of its elevation range.

The illumination model is shared, but the *intensity* computation is
geometry-specific, so this module exposes two primitives keyed to geometry --
any glyph reuses whichever matches its data:

* `shade_grid`  -- a regular raster grid (`ArrayGlyph`, `FacetGrid`, `KDEGlyph`).
* `shade_faces` -- a triangulated mesh (`MeshGlyph`).

Both share the same light direction and blend, provided by
`matplotlib.colors.LightSource` (`.direction`, `.blend_overlay` /
`.blend_soft_light` / `.blend_hsv`). Only the mesh face-normal illumination is
new maths. Pure numpy + matplotlib -- no new dependency.
"""
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.colors import Colormap, LightSource, Normalize

#: Default hillshade settings. Override any subset via a glyph's `hillshade`
#: option (``hillshade=True`` uses all defaults; ``hillshade={...}`` overrides).
DEFAULT_HILLSHADE: dict[str, Any] = {
    "azimuth": 315.0,       #: light compass direction in degrees (315 = NW, cartographic default)
    "altitude": 45.0,       #: light height above the horizon in degrees
    "vert_exag": 1.0,       #: vertical exaggeration -- the main relief-contrast knob
    "blend_mode": "overlay",  #: "overlay" / "soft" / "hsv" (overlay/soft suit terrain)
    "multidirectional": False,  #: False, an int N (N evenly-spaced azimuths), or a sequence of azimuths
    "fraction": 1.0,        #: increases the contrast of the illumination
    "dx": 1.0,              #: grid spacing in x (affects slope; grid path only)
    "dy": 1.0,              #: grid spacing in y (affects slope; grid path only)
}

_BLEND_MODES = ("overlay", "soft", "hsv")


def resolve_hillshade(hillshade: bool | dict | None) -> dict[str, Any] | None:
    """Normalise a glyph's `hillshade` option into a full settings dict.

    Args:
        hillshade: `False`/`None` (feature off), `True` (all defaults), or a
            dict overriding any subset of `DEFAULT_HILLSHADE`.

    Returns:
        dict | None: The merged settings, or `None` when hillshade is off.

    Raises:
        ValueError: For unknown option keys or an invalid `blend_mode`.
    """
    if not hillshade:
        return None
    opts = dict(DEFAULT_HILLSHADE)
    if isinstance(hillshade, dict):
        unknown = set(hillshade) - set(DEFAULT_HILLSHADE)
        if unknown:
            raise ValueError(
                f"unknown hillshade options {sorted(unknown)}; "
                f"allowed: {sorted(DEFAULT_HILLSHADE)}"
            )
        opts.update(hillshade)
    if opts["blend_mode"] not in _BLEND_MODES:
        raise ValueError(
            f"hillshade blend_mode must be one of {_BLEND_MODES}, got {opts['blend_mode']!r}"
        )
    return opts


def _blend_fn(ls: LightSource, blend_mode: str):
    """Return the `LightSource` blend method for `blend_mode`."""
    return {
        "overlay": ls.blend_overlay,
        "soft": ls.blend_soft_light,
        "hsv": ls.blend_hsv,
    }[blend_mode]


def _azimuths(multidirectional: Any, azimuth: float) -> list[float]:
    """Resolve the azimuth list for (optionally multidirectional) illumination."""
    if not multidirectional:
        return [azimuth]
    if isinstance(multidirectional, int) and not isinstance(multidirectional, bool):
        n = max(int(multidirectional), 1)
        return list(np.linspace(0.0, 360.0, n, endpoint=False))
    return [float(a) for a in multidirectional]


def shade_grid(
    elevation: np.ndarray,
    cmap: str | Colormap,
    *,
    norm: Normalize | None = None,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    vert_exag: float = 1.0,
    blend_mode: str = "overlay",
    multidirectional: Any = False,
    fraction: float = 1.0,
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray:
    """Colour-map `elevation` and blend hillshade relief into it (regular grid).

    Colours `elevation` with `cmap`/`norm`, computes the terrain illumination
    with `matplotlib.colors.LightSource`, and blends the two. `multidirectional`
    averages the illumination over several azimuths (an int `N` -> `N` evenly
    spaced, or a sequence of azimuths) to reduce single-direction bias. Serves
    every grid-based glyph (`ArrayGlyph`, `FacetGrid`, `KDEGlyph`).

    Args:
        elevation: 2D array of surface values (a DEM, a density field, ...).
        cmap: Colormap name or object used for the base colours.
        norm: Normalization for the colours. Defaults to an autoscaled
            `Normalize` over `elevation`'s finite range.
        azimuth: Light compass direction in degrees (315 = NW).
        altitude: Light height above the horizon in degrees.
        vert_exag: Vertical exaggeration -- raise it for more relief contrast.
        blend_mode: `"overlay"`, `"soft"`, or `"hsv"`.
        multidirectional: `False`, an int `N`, or a sequence of azimuths.
        fraction: Increases the illumination contrast.
        dx: Grid spacing in x (affects slope).
        dy: Grid spacing in y (affects slope).

    Returns:
        np.ndarray: An `(*, *, 4)` RGBA image; non-finite `elevation` cells are
        fully transparent.
    """
    cmap_obj = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    elevation = np.asarray(elevation, dtype=float)
    finite = np.isfinite(elevation)
    if norm is None:
        vals = elevation[finite]
        norm = Normalize(
            vmin=float(vals.min()) if vals.size else 0.0,
            vmax=float(vals.max()) if vals.size else 1.0,
        )

    rgb = cmap_obj(norm(elevation))[..., :3]

    intensities = [
        LightSource(azdeg=az, altdeg=altitude).hillshade(
            elevation, vert_exag=vert_exag, dx=dx, dy=dy, fraction=fraction
        )
        for az in _azimuths(multidirectional, azimuth)
    ]
    intensity = np.mean(intensities, axis=0)
    # A cell adjacent to NaN gets a NaN gradient -> neutral (mid) illumination;
    # the cell itself is made transparent below if its elevation is non-finite.
    intensity = np.where(np.isfinite(intensity), intensity, 0.5)

    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    shaded_rgb = _blend_fn(ls, blend_mode)(rgb, intensity[..., np.newaxis])
    shaded_rgb = np.clip(shaded_rgb, 0.0, 1.0)

    rgba = np.concatenate(
        [shaded_rgb, np.ones(elevation.shape + (1,))], axis=-1
    )
    rgba[~finite] = 0.0
    return rgba


def shade_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    z: np.ndarray,
    facecolors: np.ndarray,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    vert_exag: float = 1.0,
    blend_mode: str = "overlay",
    **_ignored: Any,
) -> np.ndarray:
    """Blend hillshade relief into per-face colours of a triangulated surface.

    Each triangle's 3D normal (with `vert_exag` applied to `z`) is illuminated
    by the light direction (`intensity = clip(normal . light, 0, 1)`) and
    blended into that face's colour. Serves triangulated glyphs (`MeshGlyph`).
    Extra keyword arguments (e.g. grid-only `dx`/`dy`/`multidirectional`/
    `fraction`) are accepted and ignored, so the same `hillshade` settings dict
    drives both the grid and mesh paths.

    Args:
        vertices: `(V, 2)` node x/y coordinates.
        faces: `(F, 3)` int node indices per triangle.
        z: `(V,)` surface value (elevation) per node.
        facecolors: `(F, 4)` RGBA colour per face (e.g. from `tripcolor`).
        azimuth: Light compass direction in degrees.
        altitude: Light height above the horizon in degrees.
        vert_exag: Vertical exaggeration -- raise it for more relief contrast.
        blend_mode: `"overlay"`, `"soft"`, or `"hsv"`.

    Returns:
        np.ndarray: `(F, 4)` relief-shaded RGBA per face (alpha unchanged).
    """
    verts = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    z = np.asarray(z, dtype=float)
    facecolors = np.asarray(facecolors, dtype=float)

    pts = np.column_stack([verts[:, 0], verts[:, 1], z * vert_exag])  # (V, 3)
    tri = pts[faces]  # (F, 3, 3)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # (F, 3)
    normals[normals[:, 2] < 0] *= -1.0  # orient upward
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(lengths == 0.0, 1.0, lengths)

    light = LightSource(azdeg=azimuth, altdeg=altitude).direction
    intensity = np.clip(normals @ light, 0.0, 1.0)  # (F,)

    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    rgb = facecolors[:, np.newaxis, :3]  # (F, 1, 3)
    shaded = _blend_fn(ls, blend_mode)(rgb, intensity[:, np.newaxis, np.newaxis])
    shaded = np.clip(shaded[:, 0, :], 0.0, 1.0)  # (F, 3)
    return np.column_stack([shaded, facecolors[:, 3]])
