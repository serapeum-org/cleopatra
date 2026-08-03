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

from collections.abc import Sequence
from typing import cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.legend import Legend

from cleopatra.colorbar import ColorBar, _resolve_colorbar, _warn_deprecated_cbar_kwargs
from cleopatra.colors import resolve_colormap, resolve_glow_options
from cleopatra.geo import GeoMixin
from cleopatra.glyph import Glyph, _root_figure
from cleopatra.styles import CLASSIFY_OPTIONS, resolve_sizes, width_legend
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS

#: Option keys for FlowGlyph. `ticks_spacing` is `None` so the shared
#: `_prepare_scalar_mapping` helper auto-derives it from the values. The
#: `width_*` / `size_legend*` keys drive value→width scaling and its legend;
#: they are inert unless a `widths` array is supplied at construction.
FLOW_DEFAULT_OPTIONS = {
    "width_limits": (1, 5),
    "width_scale": "linear",
    "draw_order": "input",
    "size_legend": False,
    "size_legend_values": None,
    "size_legend_kwargs": None,
    "vmin": None,
    "vmax": None,
    "levels": None,
    "ticks_spacing": None,
    "add_colorbar": True,
    "glow": False,
}
FLOW_DEFAULT_OPTIONS = STYLE_DEFAULTS | CLASSIFY_OPTIONS | FLOW_DEFAULT_OPTIONS


class FlowGlyph(GeoMixin, Glyph):
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
        glow: Add a soft neon halo beneath the flow polylines (tracking their
            width scaling). `True` uses the defaults; a dict overrides
            `cleopatra.colors.add_line_glow`'s `n_glow` / `alpha` /
            `linewidth_step`. Default is False.
        **kwargs: Override any key in `FLOW_DEFAULT_OPTIONS`: `width_limits`
            (min/max line width in points, default `(1, 5)`), `width_scale`
            (`"linear"` / `"log"` / `"sqrt"`, default `"linear"`),
            `draw_order` (`"input"` / `"width"`, default `"input"`; `"width"`
            paints thinnest-to-thickest so the widest paths render on top --
            e.g. high stream-order rivers over their tributaries),
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
        ax: Axes | None = None,
        fig: Figure | None = None,
        **kwargs,
    ):
        _warn_deprecated_cbar_kwargs(kwargs)
        super().__init__(default_options=FLOW_DEFAULT_OPTIONS, fig=fig, ax=ax, **kwargs)
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
        self.cbar: Colorbar | None = None
        #: The width legend created by `plot` when `size_legend` is truthy
        #: (None otherwise); built via `cleopatra.styles.width_legend`.
        self.size_legend_artist: Legend | None = None

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
            return float(self.default_options["line_width"])
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
        ax: Axes | None = None,
        title: str | None = None,
        add_colorbar: bool | None = None,
        colorbar: bool | ColorBar | None = None,
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
            colorbar: Typed `ColorBar` spec (or `True`/`False`/`None`) for the
                colorbar's placement, caption, and sizing. Resolved into the
                `cbar_*` options for this call; a `ColorBar`/`True` also enables
                the colorbar and `False` suppresses it. The spec is **sticky**,
                so a `ColorBar`/`True` persists into later plots, overriding a
                construction-time `add_colorbar=False`; an explicit
                `add_colorbar=` argument still wins the on/off decision.

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
            - `draw_order="width"` paints the widest paths last (on top), so
                high stream-order channels sit over their tributaries:
                ```python
                >>> import numpy as np
                >>> from cleopatra.flow_glyph import FlowGlyph
                >>> paths = [
                ...     np.array([[0.0, 0.0], [1.0, 0.0]]),
                ...     np.array([[0.0, 1.0], [1.0, 1.0]]),
                ... ]
                >>> glyph = FlowGlyph(
                ...     paths, values=np.array([5.0, 1.0]),
                ...     widths=np.array([5.0, 1.0]), width_limits=(1, 5),
                ...     draw_order="width",
                ... )
                >>> fig, ax, lc = glyph.plot(add_colorbar=False)
                >>> lw = lc.get_linewidths()
                >>> bool(lw[0] < lw[-1])  # thinnest drawn first, widest last
                True
                >>> [float(v) for v in lc.get_array()]  # colours follow the reorder
                [1.0, 5.0]

                ```
        """
        if ax is not None:
            self.ax = ax
            self.fig = _root_figure(ax)
        elif self.ax is None:
            self.fig, self.ax = self.create_figure_axes()
        ax = self.ax
        assert self.fig is not None
        opts = self.default_options
        # Merge a typed `colorbar=` spec (placement/caption/sizing) into the
        # options before deciding whether/how to draw the bar (issue #239).
        opts.update(_resolve_colorbar(colorbar))

        if title is not None:
            opts["title"] = title
        draw_colorbar = opts["add_colorbar"] if add_colorbar is None else add_colorbar
        if opts["draw_order"] not in ("input", "width"):
            raise ValueError(
                f"draw_order must be 'input' or 'width', got {opts['draw_order']!r}"
            )

        linewidths = self._resolve_linewidths()

        # Draw-order: with draw_order="width", paint paths from thinnest to
        # thickest so the widest (e.g. the highest stream-order main channels)
        # render on top of narrower tributaries. Reorder only the copies fed to
        # the LineCollection; self.widths/self.values and the width legend keep
        # their original order.
        draw_paths, draw_values, draw_widths = self.paths, self.values, linewidths
        if opts["draw_order"] == "width" and self.widths is not None:
            order = np.argsort(np.asarray(self.widths, dtype=float), kind="stable")
            draw_paths = [self.paths[k] for k in order]
            if isinstance(linewidths, np.ndarray):
                draw_widths = linewidths[order]
            if self.values is not None:
                draw_values = np.asarray(self.values)[order]

        if self.values is None:
            lc = LineCollection(
                draw_paths, colors=opts["color_1"], linewidths=draw_widths
            )
            ax.add_collection(lc)
        else:
            norm, cbar_kw, ticks = self._prepare_scalar_mapping(self.values)
            lc = LineCollection(
                draw_paths,
                array=np.asarray(draw_values),
                cmap=resolve_colormap(opts["cmap"]),
                norm=norm,
                linewidths=draw_widths,
            )
            if norm is None:
                lc.set_clim(ticks[0], ticks[-1])
            ax.add_collection(lc)
            if draw_colorbar:
                self.cbar = self.create_color_bar(ax, lc, cbar_kw)

        if opts["glow"]:
            glow_opts = resolve_glow_options(opts["glow"])
            n_glow = glow_opts.get("n_glow", 6)
            glow_alpha = glow_opts.get("alpha", 0.05)
            lw_step = glow_opts.get("linewidth_step", 1.0)
            under = lc.get_zorder() - 1
            for i in range(1, n_glow + 1):
                glow_widths = draw_widths + lw_step * i
                if self.values is None:
                    glow_lc = LineCollection(
                        draw_paths,
                        colors=opts["color_1"],
                        linewidths=glow_widths,
                        alpha=glow_alpha,
                        zorder=under,
                    )
                else:
                    glow_lc = LineCollection(
                        draw_paths,
                        array=np.asarray(draw_values),
                        cmap=resolve_colormap(opts["cmap"]),
                        norm=norm,
                        linewidths=glow_widths,
                        alpha=glow_alpha,
                        zorder=under,
                    )
                    if norm is None:
                        glow_lc.set_clim(ticks[0], ticks[-1])
                ax.add_collection(glow_lc)

        ax.autoscale_view()

        if self.widths is not None and opts["size_legend"]:
            # `_resolve_linewidths` returns an ndarray whenever `self.widths`
            # is not None (the branch this code is already inside).
            self.size_legend_artist = self._draw_width_legend(
                ax, cast(np.ndarray, linewidths)
            )

        if opts["title"]:
            ax.set_title(opts["title"], fontsize=opts["title_size"])

        return self.fig, ax, lc
