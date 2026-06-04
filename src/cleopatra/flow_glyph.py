"""Flow / Sankey-style line visualization.

Provides `FlowGlyph` for drawing a collection of polylines whose **colour**
encodes a per-path magnitude and whose **width** is scaled by a (possibly
different) per-path magnitude — the rendering primitive behind a spatial
Sankey / flow map. It mirrors `VectorGlyph` (which colours an artist by a
per-item magnitude through the shared scalar-mapping pipeline) but draws a
`matplotlib.collections.LineCollection` and adds value→width scaling via
`cleopatra.styles.resolve_sizes` (shared with `ScatterGlyph`).

The glyph is geometry-agnostic: it takes plain vertex arrays, so any
great-circle interpolation or projection is the caller's job.

Examples:
    - Two flows coloured by value and scaled by width:
        ```python
        >>> import numpy as np
        >>> from cleopatra.flow_glyph import FlowGlyph
        >>> paths = [
        ...     np.array([[0.0, 0.0], [1.0, 1.0]]),
        ...     np.array([[0.0, 1.0], [1.0, 0.0]]),
        ... ]
        >>> glyph = FlowGlyph(
        ...     paths, values=np.array([1.0, 5.0]), widths=np.array([2.0, 8.0])
        ... )
        >>> fig, ax, lc = glyph.plot()

        ```
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.legend import Legend

from cleopatra.glyph import Glyph
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS
from cleopatra.styles import resolve_sizes, width_legend

#: Option keys for FlowGlyph. `ticks_spacing` is `None` so the shared
#: `_prepare_scalar_mapping` helper auto-derives it from the values. The
#: `width_*` / `size_legend*` keys drive value→width scaling and its legend;
#: they are inert unless a `widths` array is supplied at construction.
FLOW_DEFAULT_OPTIONS = {
    "width_limits": (1, 5),
    "width_scale": "linear",
    "size_legend": False,
    "size_legend_values": None,
    "size_legend_kwargs": None,
    "vmin": None,
    "vmax": None,
    "levels": None,
    "ticks_spacing": None,
    "add_colorbar": True,
}
FLOW_DEFAULT_OPTIONS = STYLE_DEFAULTS | FLOW_DEFAULT_OPTIONS


class FlowGlyph(Glyph):
    """Visualization class for magnitude-coloured, width-scaled flow paths.

    Renders a sequence of polylines as a `LineCollection`. With a per-path
    `values` array the lines are colour-mapped through the shared
    scalar-mapping pipeline and a colorbar is attached (like `VectorGlyph`);
    with a per-path `widths` array each line's width is scaled via
    `cleopatra.styles.resolve_sizes` (like `ScatterGlyph`'s `sizes`). Colour
    and width are independent, so a flow can encode two quantities at once.

    Args:
        paths: A sequence of `(n_i, 2)` arrays of `(x, y)` vertices, one
            polyline per flow. Polylines may have different vertex counts.
        values: Optional per-path magnitude (length = number of paths) used
            for colour mapping. Default is None (single-colour lines, no
            colorbar).
        widths: Optional per-path magnitude (length = number of paths) used
            for line-width scaling. When None, the scalar `line_width`
            option is used for every line. Default is None.
        ax: Pre-existing axes to draw on. Default is None.
        fig: Pre-existing figure. Default is None.
        **kwargs: Override any key in `FLOW_DEFAULT_OPTIONS`: `width_limits`
            (min/max line width in points, default `(1, 5)`), `width_scale`
            (`"linear"` / `"log"` / `"sqrt"`, default `"linear"`),
            `size_legend` (bool, default False), `size_legend_values`,
            `size_legend_kwargs`, plus the shared colour options (`cmap`,
            `vmin`, `vmax`, `levels`, `color_scale`, `ticks_spacing`,
            `cbar_label`, `figsize`, `title`). Set `add_colorbar=False` to
            suppress the per-glyph colorbar (default True).

    Raises:
        ValueError: If `values` or `widths` lengths do not match the number
            of paths.

    Examples:
        - Build flows and read back the width ordering:
            ```python
            >>> import numpy as np
            >>> from cleopatra.flow_glyph import FlowGlyph
            >>> paths = [
            ...     np.array([[0.0, 0.0], [1.0, 0.0]]),
            ...     np.array([[0.0, 1.0], [1.0, 1.0]]),
            ...     np.array([[0.0, 2.0], [1.0, 2.0]]),
            ... ]
            >>> glyph = FlowGlyph(
            ...     paths,
            ...     values=np.array([1.0, 2.0, 3.0]),
            ...     widths=np.array([10.0, 1.0, 5.0]),
            ...     width_limits=(1, 5),
            ... )
            >>> fig, ax, lc = glyph.plot()
            >>> lw = lc.get_linewidths()
            >>> bool(lw[0] == max(lw) and lw[1] == min(lw))
            True

            ```

    See Also:
        cleopatra.glyph.Glyph._prepare_scalar_mapping: Shared
            norm/colorbar/ticks pipeline used to colour by `values`.
        cleopatra.styles.resolve_sizes: The value→size helper used for line
            width (shared with `ScatterGlyph`).
        cleopatra.vector_glyph.VectorGlyph: Magnitude-coloured vector fields.
    """

    #: Option keys this glyph accepts (see `Glyph.option_keys`/`filter_kwargs`).
    DEFAULT_OPTIONS = FLOW_DEFAULT_OPTIONS

    def __init__(
        self,
        paths: Sequence[np.ndarray],
        *,
        values: np.ndarray | None = None,
        widths: np.ndarray | None = None,
        ax: Axes = None,
        fig: Figure = None,
        **kwargs,
    ):
        super().__init__(
            default_options=FLOW_DEFAULT_OPTIONS, fig=fig, ax=ax, **kwargs
        )
        self.paths = [np.asarray(p, dtype=float) for p in paths]
        n_paths = len(self.paths)
        if values is not None:
            values = np.asarray(values)
            if values.shape != (n_paths,):
                raise ValueError(
                    f"values must have one entry per path ({n_paths}), got "
                    f"shape {values.shape}."
                )
        if widths is not None:
            widths = np.asarray(widths)
            if widths.shape != (n_paths,):
                raise ValueError(
                    f"widths must have one entry per path ({n_paths}), got "
                    f"shape {widths.shape}."
                )
        self.values = values
        self.widths = widths
        self.cbar = None
        #: The width legend created by `plot` when `size_legend` is truthy
        #: (None otherwise); built via `cleopatra.styles.width_legend`.
        self.size_legend_artist = None

    def _resolve_linewidths(self) -> float | np.ndarray:
        """Resolve the per-path line widths for the collection.

        Returns the per-path widths mapped from `widths` when a `widths`
        array was supplied (via `cleopatra.styles.resolve_sizes`, honouring
        the `width_limits` / `width_scale` options), or the scalar
        `line_width` option when no `widths` were given.

        Returns:
            float or np.ndarray: A scalar width (no `widths`) or a per-path
                width array spanning `width_limits` monotonically in
                `widths`.
        """
        if self.widths is None:
            return self.default_options["line_width"]
        width_min, width_max = self.default_options["width_limits"]
        return resolve_sizes(
            self.widths,
            width_min,
            width_max,
            scale=self.default_options["width_scale"],
        )

    def _draw_width_legend(self, ax: Axes, linewidths: np.ndarray) -> Legend:
        """Draw a line-width legend for the resolved per-path widths.

        Picks representative magnitudes (`size_legend_values`, or the min /
        median / max of `widths` when unset), maps each to its plotted line
        width by interpolating the already-computed `(widths -> linewidth)`
        mapping, and hands them to `cleopatra.styles.width_legend`.

        Args:
            ax: The axes to attach the legend to.
            linewidths: The per-path widths returned by
                `_resolve_linewidths`.

        Returns:
            matplotlib.legend.Legend: The width legend added to `ax`.
        """
        widths = np.asarray(self.widths, dtype=float)
        legend_values = self.default_options["size_legend_values"]
        if legend_values is None:
            legend_values = np.quantile(widths, [0.0, 0.5, 1.0])
        legend_values = np.asarray(legend_values, dtype=float)
        order = np.argsort(widths)
        legend_widths = np.interp(
            legend_values, widths[order], np.asarray(linewidths)[order]
        )
        labels = [f"{v:g}" for v in legend_values]
        legend_kwargs = self.default_options["size_legend_kwargs"] or {}
        return width_legend(ax, legend_widths, labels, **legend_kwargs)

    def plot(
        self,
        ax: Axes = None,
        title: str | None = None,
        add_colorbar: bool | None = None,
    ) -> tuple[Figure, Axes, LineCollection]:
        """Draw the flow paths, colouring by value and scaling by width.

        Builds a `LineCollection` from `paths`. When `values` was supplied,
        the colour scale, norm, ticks, and colorbar are resolved through
        `_prepare_scalar_mapping`; otherwise a single-colour collection is
        drawn with no colorbar. Line widths come from `widths` via
        `cleopatra.styles.resolve_sizes`, falling back to the scalar
        `line_width` option. If `size_legend` is truthy and `widths` were
        given, a width legend is drawn and stored on
        `self.size_legend_artist`.

        Args:
            ax: Axes to draw on. Falls back to the axes supplied at
                construction, otherwise a new figure/axes is created.
            title: Plot title. Overrides `default_options["title"]` when
                given.
            add_colorbar: Override the `add_colorbar` option for this call
                — True draws the colorbar, False suppresses it. Defaults to
                None, which keeps the value set at construction.

        Returns:
            tuple[Figure, Axes, LineCollection]: The figure, the axes, and
                the `LineCollection` (the mappable the colorbar attaches to
                when coloured).

        Raises:
            ValueError: If the values have no finite entries (via
                `_prepare_scalar_mapping`).

        Examples:
            - Uncoloured flows draw no colorbar:
                ```python
                >>> import numpy as np
                >>> from cleopatra.flow_glyph import FlowGlyph
                >>> paths = [np.array([[0.0, 0.0], [1.0, 1.0]])]
                >>> glyph = FlowGlyph(paths)
                >>> fig, ax, lc = glyph.plot()
                >>> glyph.cbar is None
                True

                ```
            - Coloured flows expose the per-path values on the collection:
                ```python
                >>> import numpy as np
                >>> from cleopatra.flow_glyph import FlowGlyph
                >>> paths = [
                ...     np.array([[0.0, 0.0], [1.0, 1.0]]),
                ...     np.array([[0.0, 1.0], [1.0, 0.0]]),
                ... ]
                >>> glyph = FlowGlyph(paths, values=np.array([3.0, 7.0]))
                >>> fig, ax, lc = glyph.plot(add_colorbar=False)
                >>> [float(v) for v in lc.get_array()]
                [3.0, 7.0]

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
        draw_colorbar = (
            opts["add_colorbar"] if add_colorbar is None else add_colorbar
        )

        linewidths = self._resolve_linewidths()

        if self.values is None:
            lc = LineCollection(
                self.paths, colors=opts["color_1"], linewidths=linewidths
            )
            ax.add_collection(lc)
        else:
            norm, cbar_kw, ticks = self._prepare_scalar_mapping(self.values)
            lc = LineCollection(
                self.paths,
                array=np.asarray(self.values),
                cmap=opts["cmap"],
                norm=norm,
                linewidths=linewidths,
            )
            if norm is None:
                lc.set_clim(ticks[0], ticks[-1])
            ax.add_collection(lc)
            if draw_colorbar:
                self.cbar = self.create_color_bar(ax, lc, cbar_kw)

        ax.autoscale_view()

        if self.widths is not None and opts["size_legend"]:
            self.size_legend_artist = self._draw_width_legend(ax, linewidths)

        if opts["title"]:
            ax.set_title(opts["title"], fontsize=opts["title_size"])

        return self.fig, ax, lc
