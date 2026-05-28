"""Static projected ('globe') map frame for matplotlib axes.

Provides `apply_projection_frame` -- a single stateless helper that turns
a plain `matplotlib.axes.Axes` into a static projected map frame: it sets
the projected limits and equal aspect, draws the projection *boundary*
(the globe's circle, Robinson's rounded rectangle, ...), optionally
*clips* the existing data layers to that boundary, and draws the
*graticule* polylines.

The module is **pure matplotlib with no PROJ/CRS dependency**. It only
*receives* already-computed geometry -- boundary vertices, graticule
polylines, and projected limits -- as plain arrays. Whatever produces the
projection (reprojecting data and deriving the boundary/graticule) lives
upstream; cleopatra only renders the result. This keeps the engine split
clean: the upstream owns CRS/PROJ, cleopatra owns matplotlib.

Examples:
    Frame a plain axes as an orthographic globe and clip an image to the
    boundary circle:

    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> theta = np.linspace(0, 2 * np.pi, 200)
    >>> boundary = np.column_stack([np.cos(theta), np.sin(theta)])
    >>> fig, ax = plt.subplots()
    >>> _ = ax.imshow(np.random.rand(8, 8), extent=(-1, 1, -1, 1))
    >>> patch = apply_projection_frame(
    ...     ax, boundary_xy=boundary, xlim=(-1, 1), ylim=(-1, 1)
    ... )
    >>> ax.get_aspect()
    1.0
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path

#: Default style for the projection boundary patch. Merged with (and
#: overridden by) the caller's ``boundary_kw``.
DEFAULT_BOUNDARY_KW: dict[str, Any] = {"edgecolor": "black", "linewidth": 0.8}

#: Default style for the graticule polylines. Merged with (and overridden
#: by) the caller's ``graticule_kw``.
DEFAULT_GRATICULE_KW: dict[str, Any] = {"color": "gray", "linewidth": 0.4}


def _as_xy(array: Any, name: str) -> np.ndarray:
    """Coerce input to a float ``(N, 2)`` array of x/y vertices.

    Args:
        array: Anything array-like holding vertex coordinates.
        name: Argument name, used in error messages.

    Returns:
        numpy.ndarray: A ``(N, 2)`` float array.

    Raises:
        ValueError: If the input is not 2-D with two columns.
    """
    xy = np.asarray(array, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(
            f"{name} must be an (N, 2) array of x/y coordinates, "
            f"got shape {xy.shape}."
        )
    return xy


def apply_projection_frame(
    ax: Any,
    *,
    boundary_xy: Any,
    xlim: Sequence[float],
    ylim: Sequence[float],
    graticule_lines: Sequence[Any] | None = None,
    clip_artists: bool = True,
    boundary_kw: dict[str, Any] | None = None,
    graticule_kw: dict[str, Any] | None = None,
) -> PathPatch:
    """Turn a plain axes into a static projected ('globe') frame.

    Sets equal aspect and the projected x/y limits, draws the projection
    boundary as a `matplotlib.patches.PathPatch`, draws the graticule
    polylines, optionally clips the existing data layers to the boundary,
    and turns the axis decorations off. The boundary geometry, graticule
    polylines, and limits are supplied as plain arrays -- this function
    performs no reprojection and has no PROJ/CRS dependency.

    Args:
        ax: Matplotlib `matplotlib.axes.Axes` to frame. Any data
            layers to be clipped should already be plotted.
        boundary_xy: ``(N, 2)`` projection-boundary vertices in projected
            coordinates (e.g. the globe's circle). Array-like.
        xlim: Projected-coordinate x-limits ``(xmin, xmax)`` -- the CRS
            domain in the x direction.
        ylim: Projected-coordinate y-limits ``(ymin, ymax)``.
        graticule_lines: Optional list of ``(M, 2)`` polylines (already
            densified and projected), each drawn as a graticule line.
            `None` (default) draws no graticule.
        clip_artists: If `True` (default), clip the existing data layers
            (`ax.images`, `ax.collections`, `ax.lines`) and the drawn
            graticule to the boundary path.
        boundary_kw: Style overrides for the boundary patch, merged over
            `DEFAULT_BOUNDARY_KW`. ``facecolor`` defaults to
            ``"none"`` so the patch never hides the data.
        graticule_kw: Style overrides for the graticule lines, merged
            over `DEFAULT_GRATICULE_KW`.

    Returns:
        matplotlib.patches.PathPatch: The boundary patch added to the
        axes (also used as the clip path).

    Raises:
        TypeError: If `ax` is not a matplotlib Axes.
        ValueError: If `boundary_xy` or a graticule line is not an
            ``(N, 2)`` array, or if `xlim`/`ylim` are not 2-tuples.

    Examples:
        Frame an axes as a globe with a graticule and clip an image to
        the boundary:

        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> theta = np.linspace(0, 2 * np.pi, 200)
        >>> boundary = np.column_stack([np.cos(theta), np.sin(theta)])
        >>> meridian = np.column_stack([np.zeros(50), np.linspace(-1, 1, 50)])
        >>> fig, ax = plt.subplots()
        >>> im = ax.imshow(np.random.rand(8, 8), extent=(-1, 1, -1, 1))
        >>> patch = apply_projection_frame(
        ...     ax,
        ...     boundary_xy=boundary,
        ...     xlim=(-1, 1),
        ...     ylim=(-1, 1),
        ...     graticule_lines=[meridian],
        ... )
        >>> im.get_clip_path() is not None
        True
    """
    if not hasattr(ax, "set_xlim") or not hasattr(ax, "add_patch"):
        raise TypeError(
            "ax must be a matplotlib.axes.Axes instance, "
            f"got {type(ax).__name__}"
        )

    if len(xlim) != 2 or len(ylim) != 2:
        raise ValueError(
            f"xlim and ylim must each be a (min, max) pair, "
            f"got xlim={xlim!r}, ylim={ylim!r}."
        )

    boundary = _as_xy(boundary_xy, "boundary_xy")

    ax.set_aspect("equal")
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    patch = PathPatch(
        Path(boundary),
        facecolor="none",
        **{**DEFAULT_BOUNDARY_KW, **(boundary_kw or {})},
    )
    ax.add_patch(patch)

    graticule_style = {**DEFAULT_GRATICULE_KW, **(graticule_kw or {})}
    graticule_artists = []
    for i, line in enumerate(graticule_lines or []):
        xy = _as_xy(line, f"graticule_lines[{i}]")
        (artist,) = ax.plot(xy[:, 0], xy[:, 1], **graticule_style)
        graticule_artists.append(artist)

    if clip_artists:
        for art in (*ax.images, *ax.collections, *ax.lines):
            art.set_clip_path(patch)

    ax.set_axis_off()
    return patch
