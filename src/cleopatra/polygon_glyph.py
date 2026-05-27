"""Filled-polygon visualization.

Provides `PolygonGlyph` for drawing a collection of polygons, optionally
coloured by a per-polygon scalar value (a choropleth). The glyph is
geometry-agnostic: it takes plain vertex arrays, never geopandas, so the
caller (e.g. Digital-Earth) is responsible for extracting polygon
vertices from whatever geometry source it has. Colour mapping reuses the
shared `Glyph._prepare_scalar_mapping` pipeline so `vmin` / `vmax`,
`ticks_spacing`, `levels`, and `color_scale` behave as for the other
glyphs.

Examples:
    - Colour two triangles by value:
        ```python
        >>> import numpy as np
        >>> from cleopatra.polygon_glyph import PolygonGlyph
        >>> polys = [
        ...     np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
        ...     np.array([[1.0, 0.0], [2.0, 0.0], [1.5, 1.0]]),
        ... ]
        >>> glyph = PolygonGlyph(polys, values=np.array([10.0, 20.0]))
        >>> fig, ax, pc = glyph.plot()

        ```
    - Draw outlines only (no fill, no colorbar):
        ```python
        >>> glyph = PolygonGlyph(polys)
        >>> fig, ax, pc = glyph.plot(outline_only=True)

        ```
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from cleopatra.glyph import Glyph
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS

#: Option keys for PolygonGlyph. `ticks_spacing` is `None` so the shared
#: `_prepare_scalar_mapping` helper auto-derives it from the data.
POLYGON_DEFAULT_OPTIONS = {
    "edgecolor": "none",
    "linewidth": 0.5,
    "vmin": None,
    "vmax": None,
    "levels": None,
    "ticks_spacing": None,
    "add_colorbar": True,
}
POLYGON_DEFAULT_OPTIONS = STYLE_DEFAULTS | POLYGON_DEFAULT_OPTIONS


class PolygonGlyph(Glyph):
    """Visualization class for collections of polygons.

    Wraps `matplotlib.collections.PolyCollection`. With a per-polygon
    `values` array, polygons are filled and colour-mapped through the
    shared scalar-mapping pipeline and a colorbar is attached. With no
    values (or `outline_only=True`), only the polygon outlines are
    drawn.

    Args:
        polygons: Sequence of polygons, each an `(n_i, 2)` array of
            `(x, y)` vertices. Polygons may have differing vertex
            counts.
        values: Optional 1D array of per-polygon scalar values for
            colour mapping. Must match the number of polygons when
            given. Default is None (outline-only).
        ax: Pre-existing axes to draw on. Default is None.
        fig: Pre-existing figure. Default is None.
        **kwargs: Override any key in `POLYGON_DEFAULT_OPTIONS`
            (e.g. `edgecolor`, `linewidth`, `cmap`, `vmin`, `vmax`,
            `levels`, `color_scale`, `ticks_spacing`, `cbar_label`,
            `figsize`, `title`). Set `add_colorbar=False` to suppress the
            per-glyph colorbar (default True) for shared-axes composition
            where the host owns a single aggregated colorbar.

    Raises:
        ValueError: If `values` is given but its length does not match
            the number of polygons.

    Examples:
        - Inspect the value array carried by the collection:
            ```python
            >>> import numpy as np
            >>> from cleopatra.polygon_glyph import PolygonGlyph
            >>> polys = [
            ...     np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
            ...     np.array([[1.0, 0.0], [2.0, 0.0], [1.5, 1.0]]),
            ... ]
            >>> glyph = PolygonGlyph(polys, values=np.array([3.0, 7.0]))
            >>> fig, ax, pc = glyph.plot()
            >>> [float(v) for v in pc.get_array()]
            [3.0, 7.0]

            ```

    See Also:
        cleopatra.glyph.Glyph._prepare_scalar_mapping: Shared
            norm/colorbar/ticks pipeline used for the filled path.
    """

    #: Option keys this glyph accepts (see `Glyph.option_keys`/`filter_kwargs`).
    DEFAULT_OPTIONS = POLYGON_DEFAULT_OPTIONS

    def __init__(
        self,
        polygons: Sequence[np.ndarray],
        values: np.ndarray | None = None,
        *,
        ax: Axes = None,
        fig: Figure = None,
        **kwargs,
    ):
        super().__init__(
            default_options=POLYGON_DEFAULT_OPTIONS, fig=fig, ax=ax, **kwargs
        )
        self.polygons = [np.asarray(p, dtype=float) for p in polygons]
        if values is not None:
            values = np.asarray(values)
            if values.shape[0] != len(self.polygons):
                raise ValueError(
                    f"values length ({values.shape[0]}) must match the number "
                    f"of polygons ({len(self.polygons)})."
                )
        self.values = values
        self.cbar = None

    def plot(
        self,
        outline_only: bool = False,
        ax: Axes = None,
        title: str | None = None,
        add_colorbar: bool | None = None,
    ) -> tuple[Figure, Axes, PolyCollection]:
        """Draw the polygons, filling by value when present.

        When `values` was supplied and `outline_only` is False, the
        polygons are filled and colour-mapped through
        `_prepare_scalar_mapping` (so `vmin` / `vmax` / `levels` /
        `color_scale` apply) with a matching colorbar. Otherwise only
        the outlines are drawn and no colorbar is added.

        Args:
            outline_only: Draw unfilled outlines even when `values` is
                present (the `shapes` use case). Default is False.
            ax: Axes to draw on. Falls back to the axes supplied at
                construction, otherwise a new figure/axes is created.
            title: Plot title. Overrides `default_options["title"]`
                when given.
            add_colorbar: Override the `add_colorbar` option for this call
                — True draws the colorbar, False suppresses it (for
                shared-axes composition). Defaults to None, which keeps the
                value set at construction.

        Returns:
            tuple[Figure, Axes, PolyCollection]: The figure, the axes,
                and the `PolyCollection` added to the axes.

        Examples:
            - Outline-only mode carries no colour array and no colorbar:
                ```python
                >>> import numpy as np
                >>> from cleopatra.polygon_glyph import PolygonGlyph
                >>> polys = [np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])]
                >>> glyph = PolygonGlyph(polys, values=np.array([5.0]))
                >>> fig, ax, pc = glyph.plot(outline_only=True)
                >>> pc.get_array() is None
                True
                >>> glyph.cbar is None
                True

                ```
        """
        if ax is not None:
            self.ax = ax
            self.fig = ax.get_figure()
        elif self.ax is None:
            self.fig, self.ax = self.create_figure_axes()
        ax = self.ax
        opts = self.default_options

        if title is not None:
            opts["title"] = title
        # Resolve the colorbar choice for this call only (a plot-time
        # override does not persist into the glyph's options).
        draw_colorbar = (
            opts["add_colorbar"] if add_colorbar is None else add_colorbar
        )

        if outline_only or self.values is None:
            pc = PolyCollection(
                self.polygons,
                facecolors="none",
                edgecolors=opts["edgecolor"],
                linewidths=opts["linewidth"],
            )
            ax.add_collection(pc)
            ax.autoscale_view()
        else:
            norm, cbar_kw, ticks = self._prepare_scalar_mapping(self.values)
            pc = PolyCollection(
                self.polygons,
                array=np.asarray(self.values),
                cmap=opts["cmap"],
                norm=norm,
                edgecolors=opts["edgecolor"],
                linewidths=opts["linewidth"],
            )
            if norm is None:
                pc.set_clim(ticks[0], ticks[-1])
            ax.add_collection(pc)
            ax.autoscale_view()
            if draw_colorbar:
                self.cbar = self.create_color_bar(ax, pc, cbar_kw)

        if opts["title"]:
            ax.set_title(opts["title"], fontsize=opts["title_size"])

        return self.fig, ax, pc
