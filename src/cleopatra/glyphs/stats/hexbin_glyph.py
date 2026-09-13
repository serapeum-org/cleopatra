"""Hexagonal-binning visualization.

Provides `HexbinGlyph`, the discrete counterpart of `KDEGlyph`: it bins a
2-D point cloud onto a hexagonal lattice and colours each cell by a per-bin
aggregate (a count by default, or the `reduce` of a per-point `values`
array). Where `KDEGlyph` smooths the cloud into a continuous *density* with a
bandwidth, `HexbinGlyph` answers "how many observations fell here" (or "what
is their mean/sum/...") -- a value you can read straight off the colorbar --
without over-plotting the way an alpha-blended scatter does.

It wraps `matplotlib.axes.Axes.hexbin` and routes the per-bin aggregate
through the shared `Glyph._prepare_scalar_mapping` pipeline, so `vmin` /
`vmax`, `color_scale`, `levels`, `ticks_spacing`, the `ColorBar` spec and
`classify=Classify(...)` all behave exactly as for the other colour-mapped
glyphs.

The glyph is geometry- and CRS-agnostic: it takes plain `x` / `y` arrays in
whatever coordinates the axes already use. Equal-area binning on the
ellipsoid, reprojection, and the ground area of a bin are the caller's job.

Examples:
    - Per-bin counts of a point cloud (a colorbar is added):
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.stats.hexbin_glyph import HexbinGlyph
        >>> rng = np.random.default_rng(0)
        >>> x, y = rng.normal(size=500), rng.normal(size=500)
        >>> fig, ax, pc = HexbinGlyph(x, y, gridsize=12).plot()
        >>> bool(pc.get_array().max() > 1)  # some bin holds several points
        True

        ```
    - Per-bin mean of a third variable, dropping sparse bins:
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.stats.hexbin_glyph import HexbinGlyph
        >>> rng = np.random.default_rng(1)
        >>> x, y = rng.normal(size=800), rng.normal(size=800)
        >>> depth = x + y
        >>> glyph = HexbinGlyph(x, y, depth, reduce="mean", min_count=3)
        >>> fig, ax, pc = glyph.plot()

        ```
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

from cleopatra.basemap.geo import GeoMixin
from cleopatra.glyphs.base.glyph import Glyph, _root_figure
from cleopatra.styling.colorbar import ColorBar, _resolve_colorbar
from cleopatra.styling.colors import resolve_colormap
from cleopatra.styling.params import Classify, Contour
from cleopatra.styling.scaling import ColorScaling
from cleopatra.styling.styles import CLASSIFY_OPTIONS
from cleopatra.styling.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS

#: Named per-bin aggregations mapped to the reducer matplotlib's
#: `reduce_C_function` expects. `"count"` counts the observations in a bin
#: (`np.size`) even when a `values` array is supplied; a callable `reduce`
#: is passed straight through.
_REDUCE_FUNCTIONS: dict[str, Callable[[Any], Any]] = {
    "count": np.size,
    "mean": np.mean,
    "sum": np.sum,
    "min": np.min,
    "max": np.max,
    "std": np.std,
}

#: Option keys for HexbinGlyph. `ticks_spacing` is `None` so the shared
#: `_prepare_scalar_mapping` helper auto-derives it from the binned range.
HEXBIN_DEFAULT_OPTIONS = {
    "gridsize": 50,
    "reduce": "mean",
    "min_count": None,
    "extent": None,
    "edge_color": "face",
    "line_width": 0.0,
    "vmin": None,
    "vmax": None,
    "levels": None,
    "ticks_spacing": None,
    "add_colorbar": True,
}
HEXBIN_DEFAULT_OPTIONS = STYLE_DEFAULTS | CLASSIFY_OPTIONS | HEXBIN_DEFAULT_OPTIONS


class HexbinGlyph(GeoMixin, Glyph):
    """Visualization class for hexagonally-binned point density.

    Wraps `matplotlib.axes.Axes.hexbin`. With no `values`, each hexagonal
    bin is coloured by the **count** of points that fell in it; with a
    per-point `values` array, each bin is coloured by the `reduce`
    aggregate of those values (mean by default). The per-bin aggregate is
    colour-mapped through the shared scalar-mapping pipeline and a matching
    colorbar is attached, so `vmin` / `vmax` / `color_scale` / `levels` and
    `classify=Classify(...)` apply as they do for the other glyphs.

    Args:
        x: 1-D array of point x-coordinates.
        y: 1-D array of point y-coordinates. Must match the length of `x`.
        values: Optional 1-D array of per-point values aggregated per bin
            by `reduce`. Must match the length of `x` when given. Default
            is None (per-bin counts).
        ax: Pre-existing axes to draw on. Default is None.
        fig: Pre-existing figure. Default is None.
        **kwargs: Override any key in `HEXBIN_DEFAULT_OPTIONS`: `gridsize`
            (int, or an `(nx, ny)` pair, default 50), `reduce`
            (`"count"` / `"mean"` / `"sum"` / `"min"` / `"max"` / `"std"`,
            or a callable; default `"mean"`), `min_count` (drop bins with
            fewer than this many points; matplotlib's `mincnt`, default
            `None`), `extent` (`(xmin, xmax, ymin, ymax)` binning window),
            `edge_color` (default `"face"`), `line_width` (bin edge width,
            default 0.0), plus the shared appearance / colorbar / scale
            options (`cmap`, `vmin`, `vmax`, `levels`, `color_scale`,
            `ticks_spacing`, `cbar_label`, `figsize`, `title`). Set
            `add_colorbar=False` to suppress the per-glyph colorbar
            (default True).

    Note:
        Empty bins are handled differently by matplotlib's `hexbin` in the two
        modes, and with the default `min_count=None` this glyph passes that
        through: the **counts** mode (`values` is None) draws *every* lattice
        cell in the window, colouring empty ones `0` (so the colorbar starts at
        0 and the whole window is tinted), whereas the **`reduce`** mode drops
        empty cells. Pass `min_count=1` on the counts mode to blank the empty
        cells and match the `reduce` mode -- this is also required before
        `color=ColorScaling.log()`, which cannot map the `0` of an empty count
        cell.

    Raises:
        ValueError: If `x` / `y` are not 1-D or have mismatched lengths, if
            `values` (when given) does not match, if `x` is empty, or if the
            binning leaves no cells to draw (an `extent` excluding the data or
            a `min_count` above the densest cell).

    Examples:
        - Read the per-bin counts back off the drawn collection:
            ```python
            >>> import numpy as np
            >>> from cleopatra.glyphs.stats.hexbin_glyph import HexbinGlyph
            >>> x = np.array([0.0, 0.0, 0.1, 5.0])
            >>> y = np.array([0.0, 0.1, 0.0, 5.0])
            >>> fig, ax, pc = HexbinGlyph(x, y, gridsize=3).plot()
            >>> int(pc.get_array().sum())  # every point counted once
            4

            ```
        - Colour bins by the mean of a per-point value (three coincident
            points share one bin, so its value is their mean):
            ```python
            >>> import numpy as np
            >>> from cleopatra.glyphs.stats.hexbin_glyph import HexbinGlyph
            >>> x = np.array([0.0, 0.0, 0.0])
            >>> y = np.array([0.0, 0.0, 0.0])
            >>> values = np.array([2.0, 4.0, 6.0])
            >>> glyph = HexbinGlyph(x, y, values, gridsize=2, reduce="mean")
            >>> fig, ax, pc = glyph.plot()
            >>> float(pc.get_array().max())
            4.0

            ```

    See Also:
        cleopatra.glyphs.stats.kde_glyph.KDEGlyph: The continuous
            (smoothed-density) counterpart.
        cleopatra.glyphs.base.glyph.Glyph._prepare_scalar_mapping: Shared
            norm / colorbar / ticks pipeline used to colour the bins.
    """

    #: Option keys this glyph accepts (see `Glyph.option_keys`/`filter_kwargs`).
    DEFAULT_OPTIONS = HEXBIN_DEFAULT_OPTIONS
    #: A per-bin aggregate is a continuous magnitude, never a nominal class.
    _SUPPORTS_CATEGORICAL_SCHEME = False

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray | None = None,
        *,
        ax: Axes | None = None,
        fig: Figure | None = None,
        **kwargs,
    ):
        super().__init__(
            default_options=HEXBIN_DEFAULT_OPTIONS, fig=fig, ax=ax, **kwargs
        )
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError(
                f"x and y must be 1-D, got shapes {self.x.shape} and {self.y.shape}."
            )
        if self.x.shape != self.y.shape:
            raise ValueError(
                f"x and y must have the same shape, got {self.x.shape} "
                f"and {self.y.shape}."
            )
        if self.x.size == 0:
            raise ValueError("HexbinGlyph needs at least one point, got 0.")
        if values is not None:
            values = np.asarray(values, dtype=float)
            if values.shape != self.x.shape:
                raise ValueError(
                    f"values must match x/y shape {self.x.shape}, got {values.shape}."
                )
        self.values = values
        self.cbar: Colorbar | None = None
        #: The `PolyCollection` drawn by the most recent `plot` call; `None`
        #: before the first render.
        self.im: PolyCollection | None = None

    def _reduce_function(self) -> Callable[[Any], Any]:
        """Resolve the `reduce` option to the reducer `hexbin` expects.

        Returns:
            Callable: the numpy reducer for a named `reduce`, or the option
                itself when it is already a callable.

        Raises:
            ValueError: If `reduce` is neither a known name nor a callable.
        """
        reduce = self.default_options["reduce"]
        if callable(reduce):
            return reduce
        try:
            return _REDUCE_FUNCTIONS[reduce]
        except (KeyError, TypeError):
            raise ValueError(
                f"reduce must be a callable or one of "
                f"{sorted(_REDUCE_FUNCTIONS)}, got {reduce!r}."
            ) from None

    def _hexbin_kwargs(self) -> dict:
        """Build the `Axes.hexbin` keyword arguments from `default_options`.

        Returns:
            dict: `gridsize`, `reduce_C_function`, `mincnt`, `extent`,
                `edgecolors`, `linewidths`, and `C` (only when `values` was
                supplied) -- everything that defines the binning, shared by
                `plot` and `evaluate` so both bin identically.
        """
        opts = self.default_options
        kwargs: dict[str, Any] = {
            "gridsize": opts["gridsize"],
            "reduce_C_function": self._reduce_function(),
            "mincnt": opts["min_count"],
            "extent": opts["extent"],
            "edgecolors": opts["edge_color"],
            "linewidths": opts["line_width"],
        }
        if self.values is not None:
            kwargs["C"] = self.values
        return kwargs

    def evaluate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bin the points and return the result without rendering on `self.ax`.

        Draws the hexbin onto a throwaway figure (so no global state or the
        caller's axes are touched), then reads back the bin centres and the
        per-bin aggregate. Mirrors `KDEGlyph.evaluate`, and because it uses
        the same `_hexbin_kwargs` as `plot`, the returned aggregate matches
        the array of the `PolyCollection` that `plot` draws.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: the bin-centre `x`
                and `y` coordinates and the per-bin aggregate, one entry per
                drawn bin (row-aligned).

        Examples:
            - The aggregate has one value per bin centre:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.stats.hexbin_glyph import HexbinGlyph
                >>> rng = np.random.default_rng(2)
                >>> x, y = rng.normal(size=300), rng.normal(size=300)
                >>> cx, cy, agg = HexbinGlyph(x, y, gridsize=8).evaluate()
                >>> cx.shape == cy.shape == agg.shape
                True

                ```
            - The per-bin counts total the number of points:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.stats.hexbin_glyph import HexbinGlyph
                >>> x = np.array([0.0, 0.0, 1.0, 5.0])
                >>> y = np.array([0.0, 0.0, 1.0, 5.0])
                >>> _, _, agg = HexbinGlyph(x, y, gridsize=4).evaluate()
                >>> int(np.nansum(agg))
                4

                ```
        """
        fig = Figure()
        ax = fig.add_subplot(111)
        pc = ax.hexbin(self.x, self.y, **self._hexbin_kwargs())
        offsets = np.asarray(pc.get_offsets(), dtype=float)
        # `.filled(np.nan)` guards older matplotlib that masked empty /
        # `mincnt`-dropped cells; current matplotlib returns an unmasked array
        # (dropped cells are absent), so this is a no-op there.
        aggregate = np.ma.asarray(pc.get_array()).astype(float).filled(np.nan)
        return offsets[:, 0], offsets[:, 1], aggregate

    def plot(
        self,
        ax: Axes | None = None,
        title: str | None = None,
        add_colorbar: bool | None = None,
        colorbar: bool | ColorBar | None = None,
        color: ColorScaling | None = None,
        contour: Contour | None = None,
        classify: Classify | None = None,
    ) -> tuple[Figure, Axes, PolyCollection]:
        """Draw the hexagonally-binned density and colour-map the aggregate.

        The per-bin aggregate is resolved through `_prepare_scalar_mapping`,
        so `vmin` / `vmax` / `levels` / `color_scale` and the `ColorBar`
        spec behave as for the other glyphs, and `classify=Classify(...)`
        bins the aggregate into discrete colour classes. A categorical
        scheme is rejected (`_SUPPORTS_CATEGORICAL_SCHEME` is `False`): a
        per-bin aggregate is a continuous magnitude, not a nominal class.

        Args:
            ax: Axes to draw on. Falls back to the axes supplied at
                construction, otherwise a new figure/axes is created.
            title: Plot title. Overrides `default_options["title"]` when
                given.
            add_colorbar: Override the `add_colorbar` option for this call
                -- `True` draws the colorbar, `False` suppresses it (for
                shared-axes composition). `None` (default) keeps the value
                set at construction.
            colorbar: Typed `ColorBar` spec (or `True` / `False` / `None`)
                for the colorbar's placement, caption, and sizing; resolved
                into the `cbar_*` options.
            color: A `ColorScaling` selecting the colour scale
                (`vmin`/`vmax`, `color_scale`, ...) for the aggregate.
            contour: A `Contour`; only its `levels` applies here (it
                discretises the colour scale), as the hexbin draws no
                isolines.
            classify: A `Classify` binning the aggregate into discrete
                colour classes (e.g. `scheme="quantiles", k=5`).

        Returns:
            tuple[Figure, Axes, PolyCollection]: the figure, the axes, and
                the `PolyCollection` returned by `hexbin` (the mappable).

        Examples:
            - A classified hexbin steps the colorbar and still returns the
                collection:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.stats.hexbin_glyph import HexbinGlyph
                >>> from cleopatra.styling.params import Classify
                >>> rng = np.random.default_rng(3)
                >>> x, y = rng.normal(size=600), rng.normal(size=600)
                >>> glyph = HexbinGlyph(x, y, gridsize=10)
                >>> fig, ax, pc = glyph.plot(
                ...     classify=Classify(scheme="quantiles", k=4)
                ... )
                >>> glyph.cbar is not None
                True

                ```
            - Suppress the colorbar for shared-axes composition:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.stats.hexbin_glyph import HexbinGlyph
                >>> rng = np.random.default_rng(4)
                >>> x, y = rng.normal(size=200), rng.normal(size=200)
                >>> glyph = HexbinGlyph(x, y)
                >>> fig, ax, pc = glyph.plot(add_colorbar=False)
                >>> glyph.cbar is None
                True

                ```
        """
        with self._rollback_options_on_error():
            self._merge_group_params(color, contour, classify)

            if ax is not None:
                self.ax = ax
                self.fig = _root_figure(ax)
            elif self.ax is None:
                self.fig, self.ax = self.create_figure_axes()
            ax = self.ax
            assert self.fig is not None
            opts = self.default_options

            if title is not None:
                opts["title"] = title
            opts.update(_resolve_colorbar(colorbar))
            draw_colorbar = (
                opts["add_colorbar"] if add_colorbar is None else add_colorbar
            )
            self.cbar = None

            pc = ax.hexbin(
                self.x,
                self.y,
                cmap=resolve_colormap(opts["cmap"]),
                **self._hexbin_kwargs(),
            )
            # `.compressed()` guards older matplotlib, which masked empty /
            # `mincnt`-dropped cells; current matplotlib drops them as polygons
            # instead, so the two agree.
            aggregate = np.ma.asarray(pc.get_array()).compressed().astype(float)
            if aggregate.size == 0:
                raise ValueError(
                    "no hexagonal bins to draw: every cell was empty or dropped "
                    "by `min_count` / `extent`. Widen `extent` or lower "
                    "`min_count`."
                )
            norm, cbar_kw, ticks = self._prepare_scalar_mapping(aggregate)
            if norm is not None:
                pc.set_norm(norm)
            else:
                pc.set_clim(ticks[0], ticks[-1])
            self.im = pc

            if draw_colorbar:
                self.cbar = self.create_color_bar(ax, pc, cbar_kw)

            if opts["title"]:
                ax.set_title(opts["title"], fontsize=opts["title_size"])
            self._apply_axis_style(ax)

            return self.fig, ax, pc
