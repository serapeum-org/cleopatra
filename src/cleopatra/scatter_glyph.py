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

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.legend import Legend

from cleopatra.geo import GeoMixin
from cleopatra.glyph import Glyph
from cleopatra.styles import CLASSIFY_OPTIONS
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS
from cleopatra.styles import resolve_sizes, size_legend

#: Option keys for ScatterGlyph. `ticks_spacing` is `None` so the shared
#: `_prepare_scalar_mapping` helper auto-derives it from the data range.
#: The `size_*` keys drive value→size scaling and its legend (see `plot`);
#: they are inert unless a `sizes` array is supplied at construction.
SCATTER_DEFAULT_OPTIONS = {
    "marker": "o",
    "point_size": 20,
    "vmin": None,
    "vmax": None,
    "levels": None,
    "ticks_spacing": None,
    "add_colorbar": True,
    "size_limits": (10, 200),
    "size_scale": "linear",
    "size_legend": False,
    "size_legend_values": None,
    "size_legend_kwargs": None,
}
SCATTER_DEFAULT_OPTIONS = STYLE_DEFAULTS | CLASSIFY_OPTIONS | SCATTER_DEFAULT_OPTIONS


class ScatterGlyph(GeoMixin, Glyph):
    """Visualization class for 2D point clouds.

    Wraps `matplotlib.axes.Axes.scatter`. With no `values`, points are
    drawn in a single colour. With a per-point `values` array, points
    are colour-mapped through the shared scalar-mapping pipeline and a
    matching colorbar is attached. An independent per-point `sizes` array
    scales the marker area (with an optional size legend), so colour and
    size can encode two different quantities at once.

    Args:
        x: 1D array of point x-coordinates.
        y: 1D array of point y-coordinates. Must match the length of
            `x`.
        values: Optional 1D array of per-point scalar values used for
            colour mapping. Must match the length of `x` when given.
            Default is None (uncoloured points).
        sizes: Optional 1D array of per-point magnitudes used for *size*
            mapping (kept separate from `values`, which drives colour, so a
            point can encode both). Must match the length of `x` when
            given. When None, the scalar `point_size` option is used for
            every point (the original behaviour). Default is None.
        ax: Pre-existing axes to draw on. Default is None.
        fig: Pre-existing figure. Default is None.
        **kwargs: Override any key in `SCATTER_DEFAULT_OPTIONS`
            (e.g. `marker`, `point_size`, `cmap`, `vmin`, `vmax`,
            `levels`, `color_scale`, `ticks_spacing`, `cbar_label`,
            `figsize`, `title`). Set `add_colorbar=False` to suppress the
            per-glyph colorbar (default True) for shared-axes composition
            where the host owns a single aggregated colorbar. The
            `size_limits` (min/max marker area in points², default
            `(10, 200)`), `size_scale` (`"linear"` / `"log"` / `"sqrt"`,
            default `"linear"`), `size_legend` (bool, default False),
            `size_legend_values` (explicit representative magnitudes), and
            `size_legend_kwargs` (forwarded to the legend) options control
            the `sizes` mapping and its legend.

    Raises:
        ValueError: If `x` and `y` (or `values` / `sizes`, when given)
            have mismatched lengths.

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
        - Size points by a magnitude; the areas span `size_limits`:
            ```python
            >>> import numpy as np
            >>> from cleopatra.scatter_glyph import ScatterGlyph
            >>> glyph = ScatterGlyph(
            ...     np.array([0.0, 1.0, 2.0]),
            ...     np.array([0.0, 1.0, 0.0]),
            ...     sizes=np.array([1.0, 5.0, 9.0]),
            ...     size_limits=(10, 200),
            ... )
            >>> fig, ax, paths = glyph.plot()
            >>> [float(s) for s in paths.get_sizes()]
            [10.0, 105.0, 200.0]

            ```

    See Also:
        cleopatra.glyph.Glyph._prepare_scalar_mapping: Shared
            norm/colorbar/ticks pipeline used for the coloured path.
        cleopatra.styles.resolve_sizes: The value→size helper used for the
            `sizes` mapping (reusable by other size-encoding glyphs).
        cleopatra.styles.size_legend: Builds the size legend drawn when
            `size_legend` is truthy.
    """

    #: Option keys this glyph accepts (see `Glyph.option_keys`/`filter_kwargs`).
    DEFAULT_OPTIONS = SCATTER_DEFAULT_OPTIONS
    #: Per-point `values` are a nominal class label just as often as a
    #: continuous magnitude, so `scheme="categorical"` is supported.
    _SUPPORTS_CATEGORICAL_SCHEME = True

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray | None = None,
        *,
        sizes: np.ndarray | None = None,
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
        if sizes is not None:
            sizes = np.asarray(sizes)
            if sizes.shape != self.x.shape:
                raise ValueError(
                    f"sizes must match x/y shape {self.x.shape}, got {sizes.shape}."
                )
        self.values = values
        self.sizes = sizes
        self.cbar = None
        #: The size legend created by `plot` when `size_legend` is truthy
        #: (None otherwise); built via `cleopatra.styles.size_legend`.
        self.size_legend_artist = None
        #: The disjoint legend created by `plot` when `scheme="categorical"`
        #: (`None` otherwise); built via `Glyph.create_categorical_legend`.
        self.category_legend = None

    def plot(
        self,
        ax: Axes = None,
        title: str | None = None,
        add_colorbar: bool | None = None,
    ) -> tuple[Figure, Axes, PathCollection]:
        """Draw the point cloud, colour- and/or size-mapping per point.

        When `values` was supplied at construction, the colour scale,
        norm, ticks, and colorbar are resolved through
        `_prepare_scalar_mapping`, so `vmin` / `vmax` / `levels` /
        `color_scale` behave as for the other glyphs. With no values,
        a single-colour scatter is drawn and no colorbar is added.

        The one exception is `scheme="categorical"`: `vmin` / `vmax` /
        `levels` / `color_scale` are ignored (with a warning if set), and
        instead of a colorbar a `disjoint_legend` is drawn and stored on
        `self.category_legend` (`self.cbar` stays `None`). See
        `Glyph._prepare_categorical_mapping`.

        When `sizes` was supplied, each marker's area is resolved from
        that magnitude via `cleopatra.styles.resolve_sizes` (honouring
        `size_limits` / `size_scale`); otherwise the scalar `point_size`
        option is used for every point. Colour and size are independent,
        so a point can encode two quantities at once. If `size_legend` is
        truthy, a size legend is drawn via `cleopatra.styles.size_legend`
        and stored on `self.size_legend_artist`.

        Args:
            ax: Axes to draw on. Falls back to the axes supplied at
                construction, otherwise a new figure/axes is created.
            title: Plot title. Overrides `default_options["title"]`
                when given.
            add_colorbar: Override the `add_colorbar` option for this call
                — True draws the colorbar, False suppresses it (for
                shared-axes composition). Defaults to None, which keeps the
                value set at construction.

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
            - Combine colour and size, with a size legend:
                ```python
                >>> import numpy as np
                >>> from cleopatra.scatter_glyph import ScatterGlyph
                >>> glyph = ScatterGlyph(
                ...     np.array([0.0, 1.0, 2.0]),
                ...     np.array([0.0, 1.0, 0.0]),
                ...     values=np.array([1.0, 2.0, 3.0]),
                ...     sizes=np.array([5.0, 10.0, 20.0]),
                ...     size_legend=True,
                ... )
                >>> fig, ax, paths = glyph.plot()
                >>> bool(paths.get_sizes()[0] < paths.get_sizes()[-1])
                True
                >>> glyph.size_legend_artist is not None
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
        draw_colorbar = opts["add_colorbar"] if add_colorbar is None else add_colorbar
        # Reset both artifacts unconditionally so a re-plot (e.g. switching
        # `scheme` between calls) never leaves a stale reference from the
        # previous call -- only one of the two is (re)created below.
        self.cbar = None
        self.category_legend = None

        marker_area = self._resolve_marker_area()

        if self.values is None:
            paths = ax.scatter(
                self.x,
                self.y,
                s=marker_area,
                marker=opts["marker"],
            )
        else:
            norm, cbar_kw, ticks = self._prepare_scalar_mapping(self.values)
            categorical = self._categorical
            if categorical is not None:
                color_array, cmap = categorical["codes"], categorical["cmap"]
            else:
                color_array, cmap = np.asarray(self.values), opts["cmap"]
            paths = ax.scatter(
                self.x,
                self.y,
                c=color_array,
                s=marker_area,
                marker=opts["marker"],
                cmap=cmap,
                norm=norm,
                vmin=None if norm else ticks[0],
                vmax=None if norm else ticks[-1],
            )
            if draw_colorbar:
                if categorical is not None:
                    self.category_legend = self.create_categorical_legend(ax)
                else:
                    self.cbar = self.create_color_bar(ax, paths, cbar_kw)

        if self.sizes is not None and opts["size_legend"]:
            # `Axes.legend()` is single-slot per axes: `size_legend` calls it
            # internally, which would otherwise silently evict the
            # categorical legend just drawn above (matplotlib replaces
            # `ax.legend_`, so the earlier Legend stops being part of
            # `ax.get_children()` even though `self.category_legend` still
            # references it). Re-attach it as a plain artist first, mirroring
            # the multi-legend pattern in `colors.apply_data_style`.
            if self.category_legend is not None:
                ax.add_artist(self.category_legend)
            self.size_legend_artist = self._draw_size_legend(ax, marker_area)

        if opts["title"]:
            ax.set_title(opts["title"], fontsize=opts["title_size"])

        return self.fig, ax, paths

    def _resolve_marker_area(self) -> float | np.ndarray:
        """Resolve the scatter `s` (marker area) for this glyph.

        Returns the per-point areas mapped from `sizes` when a `sizes`
        array was supplied (via `cleopatra.styles.resolve_sizes`, honouring
        the `size_limits` / `size_scale` options), or the scalar
        `point_size` option when no `sizes` were given.

        Returns:
            float or np.ndarray: A scalar area (no `sizes`) or a per-point
                area array spanning `size_limits` monotonically in `sizes`.
        """
        if self.sizes is None:
            return self.default_options["point_size"]
        size_min, size_max = self.default_options["size_limits"]
        return resolve_sizes(
            self.sizes,
            size_min,
            size_max,
            scale=self.default_options["size_scale"],
        )

    def _draw_size_legend(self, ax: Axes, marker_area: np.ndarray) -> Legend:
        """Draw a size legend for the resolved per-point areas.

        Picks representative magnitudes (`size_legend_values`, or the min /
        median / max of `sizes` when unset), maps each to its plotted
        marker area by interpolating the already-computed `(sizes ->
        marker_area)` mapping (so the swatches match the points exactly),
        and hands them to `cleopatra.styles.size_legend`.

        Args:
            ax: The axes to attach the legend to.
            marker_area: The per-point marker areas returned by
                `_resolve_marker_area`.

        Returns:
            matplotlib.legend.Legend: The size legend added to `ax`.
        """
        sizes = np.asarray(self.sizes, dtype=float)
        legend_values = self.default_options["size_legend_values"]
        if legend_values is None:
            legend_values = np.quantile(sizes, [0.0, 0.5, 1.0])
        legend_values = np.asarray(legend_values, dtype=float)
        order = np.argsort(sizes)
        legend_areas = np.interp(
            legend_values, sizes[order], np.asarray(marker_area)[order]
        )
        labels = [f"{v:g}" for v in legend_values]
        legend_kwargs = self.default_options["size_legend_kwargs"] or {}
        return size_legend(ax, legend_areas, labels, **legend_kwargs)
