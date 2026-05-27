"""Point-cloud visualization.

Provides `ScatterGlyph` for plotting 2D point clouds, optionally
colour-mapped by a per-point scalar value. Colour mapping reuses the
shared `Glyph._prepare_scalar_mapping` pipeline so that `vmin` / `vmax`,
`ticks_spacing`, `levels`, and `color_scale` behave exactly as they do
for `ArrayGlyph` and `MeshGlyph`.

Examples:
    - Plot an uncoloured point cloud:
        ```python
        >>> import numpy as np
        >>> from cleopatra.scatter_glyph import ScatterGlyph
        >>> x = np.array([0.0, 1.0, 2.0, 3.0])
        >>> y = np.array([0.0, 1.0, 0.0, 1.0])
        >>> glyph = ScatterGlyph(x, y)
        >>> fig, ax, paths = glyph.plot()

        ```
    - Colour points by a per-point value (adds a colorbar):
        ```python
        >>> import numpy as np
        >>> from cleopatra.scatter_glyph import ScatterGlyph
        >>> x = np.array([0.0, 1.0, 2.0, 3.0])
        >>> y = np.array([0.0, 1.0, 0.0, 1.0])
        >>> values = np.array([10.0, 20.0, 30.0, 40.0])
        >>> glyph = ScatterGlyph(x, y, values)
        >>> fig, ax, paths = glyph.plot()

        ```
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure

from cleopatra.glyph import Glyph
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS

#: Option keys for ScatterGlyph. `ticks_spacing` is `None` so the shared
#: `_prepare_scalar_mapping` helper auto-derives it from the data range.
SCATTER_DEFAULT_OPTIONS = {
    "marker": "o",
    "point_size": 20,
    "vmin": None,
    "vmax": None,
    "levels": None,
    "ticks_spacing": None,
    "add_colorbar": True,
}
SCATTER_DEFAULT_OPTIONS = STYLE_DEFAULTS | SCATTER_DEFAULT_OPTIONS


class ScatterGlyph(Glyph):
    """Visualization class for 2D point clouds.

    Wraps `matplotlib.axes.Axes.scatter`. With no `values`, points are
    drawn in a single colour. With a per-point `values` array, points
    are colour-mapped through the shared scalar-mapping pipeline and a
    matching colorbar is attached.

    Args:
        x: 1D array of point x-coordinates.
        y: 1D array of point y-coordinates. Must match the length of
            `x`.
        values: Optional 1D array of per-point scalar values used for
            colour mapping. Must match the length of `x` when given.
            Default is None (uncoloured points).
        ax: Pre-existing axes to draw on. Default is None.
        fig: Pre-existing figure. Default is None.
        **kwargs: Override any key in `SCATTER_DEFAULT_OPTIONS`
            (e.g. `marker`, `point_size`, `cmap`, `vmin`, `vmax`,
            `levels`, `color_scale`, `ticks_spacing`, `cbar_label`,
            `figsize`, `title`). Set `add_colorbar=False` to suppress the
            per-glyph colorbar (default True) for shared-axes composition
            where the host owns a single aggregated colorbar.

    Raises:
        ValueError: If `x` and `y` (or `values`, when given) have
            mismatched lengths.

    Examples:
        - Colour points by value and read back the mapped array:
            ```python
            >>> import numpy as np
            >>> from cleopatra.scatter_glyph import ScatterGlyph
            >>> glyph = ScatterGlyph(
            ...     np.array([0.0, 1.0, 2.0]),
            ...     np.array([0.0, 1.0, 0.0]),
            ...     np.array([1.0, 5.0, 9.0]),
            ... )
            >>> fig, ax, paths = glyph.plot()
            >>> [float(v) for v in paths.get_array()]
            [1.0, 5.0, 9.0]

            ```

    See Also:
        cleopatra.glyph.Glyph._prepare_scalar_mapping: Shared
            norm/colorbar/ticks pipeline used for the coloured path.
    """

    #: Option keys this glyph accepts (see `Glyph.option_keys`/`filter_kwargs`).
    DEFAULT_OPTIONS = SCATTER_DEFAULT_OPTIONS

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray | None = None,
        *,
        ax: Axes = None,
        fig: Figure = None,
        **kwargs,
    ):
        super().__init__(
            default_options=SCATTER_DEFAULT_OPTIONS, fig=fig, ax=ax, **kwargs
        )
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        if self.x.shape != self.y.shape:
            raise ValueError(
                f"x and y must have the same shape, got {self.x.shape} "
                f"and {self.y.shape}."
            )
        if values is not None:
            values = np.asarray(values)
            if values.shape != self.x.shape:
                raise ValueError(
                    f"values must match x/y shape {self.x.shape}, got "
                    f"{values.shape}."
                )
        self.values = values
        self.cbar = None

    def plot(
        self,
        ax: Axes = None,
        title: str | None = None,
    ) -> tuple[Figure, Axes, PathCollection]:
        """Draw the point cloud, colour-mapping by value when present.

        When `values` was supplied at construction, the colour scale,
        norm, ticks, and colorbar are resolved through
        `_prepare_scalar_mapping`, so `vmin` / `vmax` / `levels` /
        `color_scale` behave as for the other glyphs. With no values,
        a single-colour scatter is drawn and no colorbar is added.

        Args:
            ax: Axes to draw on. Falls back to the axes supplied at
                construction, otherwise a new figure/axes is created.
            title: Plot title. Overrides `default_options["title"]`
                when given.

        Returns:
            tuple[Figure, Axes, PathCollection]: The figure, the axes,
                and the `PathCollection` returned by `scatter` (the
                mappable for the coloured path).

        Examples:
            - Uncoloured points return a PathCollection with no array:
                ```python
                >>> import numpy as np
                >>> from cleopatra.scatter_glyph import ScatterGlyph
                >>> glyph = ScatterGlyph(
                ...     np.array([0.0, 1.0]), np.array([0.0, 1.0])
                ... )
                >>> fig, ax, paths = glyph.plot()
                >>> paths.get_array() is None
                True

                ```
            - A title passed to plot overrides the default:
                ```python
                >>> import numpy as np
                >>> from cleopatra.scatter_glyph import ScatterGlyph
                >>> glyph = ScatterGlyph(
                ...     np.array([0.0, 1.0]), np.array([0.0, 1.0])
                ... )
                >>> fig, ax, paths = glyph.plot(title="Stations")
                >>> ax.get_title()
                'Stations'

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

        if self.values is None:
            paths = ax.scatter(
                self.x,
                self.y,
                s=opts["point_size"],
                marker=opts["marker"],
            )
        else:
            norm, cbar_kw, ticks = self._prepare_scalar_mapping(self.values)
            paths = ax.scatter(
                self.x,
                self.y,
                c=np.asarray(self.values),
                s=opts["point_size"],
                marker=opts["marker"],
                cmap=opts["cmap"],
                norm=norm,
                vmin=None if norm else ticks[0],
                vmax=None if norm else ticks[-1],
            )
            if opts["add_colorbar"]:
                self.cbar = self.create_color_bar(ax, paths, cbar_kw)

        if opts["title"]:
            ax.set_title(opts["title"], fontsize=opts["title_size"])

        return self.fig, ax, paths
