"""One-call publication-grade map composer.

Composes cleopatra's existing pieces -- a `DATA_STYLES` style (or a colormap), an
optional `projection` preset, a title block, an optional shaded-relief basemap
(`cleopatra.reference`), and the style's units-aware legend/colorbar -- into a
single call, so a caller does not have to wire them together by hand.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from cleopatra.array_glyph import ArrayGlyph
from cleopatra.reference import add_relief

__all__ = ["publication_map"]


def publication_map(
    data: np.ndarray,
    *,
    coords: tuple[np.ndarray, np.ndarray] | None = None,
    extent: tuple[float, float, float, float] | None = None,
    style: str | None = None,
    cmap: str | Colormap | None = None,
    projection: str | None = None,
    title: str | None = None,
    relief: bool = False,
    relief_resolution: str = "low",
    figsize: tuple[float, float] | None = None,
    **plot_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Render a publication-grade map in one call.

    A thin convenience over `ArrayGlyph`: it builds the glyph with a
    `DATA_STYLES` `style` (or a `cmap`), an optional `projection` preset and a
    `title`, draws it, and -- when `relief=True` -- lays a shaded-relief basemap
    underneath (`cleopatra.reference.add_relief`, which needs the `[tiles]`
    extra). Relief only shows through a translucent overlay style (e.g. `"haze"`
    or a value-linked-opacity preset); under an opaque field it is hidden.

    Args:
        data: The 2-D field to map.
        coords: Optional `(lon, lat)` 1-D vectors (required for `projection`).
        extent: Optional `[xmin, xmax, ymin, ymax]` for an extent-based render
            (mutually exclusive with `coords`).
        style: A `cleopatra.colors.DATA_STYLES` preset name (e.g.
            `"temperature_2m"`). Takes precedence over `cmap`.
        cmap: A colormap name/object when no `style` is given (namespaced names
            like `"cmocean:thermal"` work with the `[science-colors]` extra).
        projection: `"globe"` / `"flat"` projection preset, or `None`.
        title: Title block text.
        relief: When True, draw a shaded-relief basemap under the field.
        relief_resolution: Relief resolution passed to `add_relief`.
        figsize: Figure size in inches. `None` (default) lets `ArrayGlyph` size
            the figure to the data's aspect ratio.
        **plot_kwargs: Forwarded to `ArrayGlyph.plot` (e.g. `colorbar`).

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The figure and
        axes of the rendered map.

    Examples:
        - A styled field with a title, no relief (no network needed):
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.templates import publication_map
            >>> data = np.random.default_rng(0).random((10, 12)) * 30
            >>> fig, ax = publication_map(data, style="temperature_2m", title="2 m temperature")
            >>> ax.get_title()
            '2 m temperature'
            >>> plt.close(fig)

            ```
    """
    options: dict[str, Any] = {}
    if figsize is not None:
        options["figsize"] = figsize
    if style is not None:
        options["style"] = style
    if cmap is not None:
        options["cmap"] = cmap
    if projection is not None:
        options["projection"] = projection
    if title is not None:
        options["title"] = title

    glyph = ArrayGlyph(data, coords=coords, extent=extent, **options)
    fig, ax = glyph.plot(**plot_kwargs)

    if relief:
        if projection == "globe":
            warnings.warn(
                "relief basemaps compose only with the flat / unprojected view; "
                "'relief' is skipped under projection='globe' (the axes are in "
                "projected metres, not lon/lat).",
                stacklevel=2,
            )
        else:
            # No `extent` on purpose: add_relief places the whole global image and
            # the axes limits (set by the field) crop it to the region -- passing an
            # extent for a regional lon/lat view mis-places the image.
            add_relief(ax, resolution=relief_resolution, zorder=-1)

    return fig, ax
