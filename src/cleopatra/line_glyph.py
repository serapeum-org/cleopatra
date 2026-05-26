"""Line, bar, and band visualization.

Provides `LineGlyph` for the non-colour-mapped 1D plot family: line and
marker series (`line`), bar charts (`bar`), and filled bands between two
curves (`fill_between`, used for envelopes / quantile spreads). Unlike
the colour-by-value glyphs, `LineGlyph` does not use the scalar-mapping
pipeline; it subclasses `Glyph` for the figure/axes lifecycle and the
shared style options.

Examples:
    - Plot a single line series:
        ```python
        >>> import numpy as np
        >>> from cleopatra.line_glyph import LineGlyph
        >>> x = np.array([0.0, 1.0, 2.0, 3.0])
        >>> y = np.array([0.0, 1.0, 4.0, 9.0])
        >>> glyph = LineGlyph(x, y)
        >>> fig, ax, lines = glyph.line()

        ```
    - Fill an envelope band between two curves:
        ```python
        >>> lower = np.array([-1.0, 0.0, 3.0, 8.0])
        >>> glyph = LineGlyph(x, y)
        >>> fig, ax, band = glyph.fill_between(y2=lower)

        ```
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cleopatra.glyph import Glyph
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS

#: Option keys for LineGlyph (no scalar-mapping keys; this glyph does
#: not colour by value).
LINE_DEFAULT_OPTIONS = {
    "marker": None,
    "linestyle": "-",
    "alpha": 1.0,
}
LINE_DEFAULT_OPTIONS = STYLE_DEFAULTS | LINE_DEFAULT_OPTIONS


class LineGlyph(Glyph):
    """Visualization class for line, bar, and band plots.

    Wraps `Axes.plot`, `Axes.bar`, and `Axes.fill_between`. `y` may be
    1D (a single series) or 2D (`line`/`bar` draw one series per
    column). Styling comes from the shared options (`color_1`,
    `line_width`, `marker`, `linestyle`, `alpha`).

    Args:
        x: 1D array of x-coordinates.
        y: y-values. 1D for a single series, or 2D `(n_points,
            n_series)` for multiple series.
        ax: Pre-existing axes to draw on. Default is None.
        fig: Pre-existing figure. Default is None.
        **kwargs: Override any key in `LINE_DEFAULT_OPTIONS`
            (e.g. `marker`, `linestyle`, `alpha`, `color_1`,
            `line_width`, `figsize`, `title`).

    Raises:
        ValueError: If the length of `x` does not match the number of
            rows in `y`.

    Examples:
        - Read back the y-data of the drawn line:
            ```python
            >>> import numpy as np
            >>> from cleopatra.line_glyph import LineGlyph
            >>> glyph = LineGlyph(np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 2.0]))
            >>> fig, ax, lines = glyph.line()
            >>> [float(v) for v in lines[0].get_ydata()]
            [1.0, 3.0, 2.0]

            ```
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        ax: Axes = None,
        fig: Figure = None,
        **kwargs,
    ):
        super().__init__(
            default_options=LINE_DEFAULT_OPTIONS, fig=fig, ax=ax, **kwargs
        )
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        n_rows = self.y.shape[0]
        if self.x.shape[0] != n_rows:
            raise ValueError(
                f"x length ({self.x.shape[0]}) must match the number of rows "
                f"in y ({n_rows})."
            )

    def _series(self) -> list[np.ndarray]:
        """Split `y` into one 1D array per series (column for 2D)."""
        if self.y.ndim == 1:
            return [self.y]
        return [self.y[:, i] for i in range(self.y.shape[1])]

    def _resolve_ax(self, ax: Axes | None) -> Axes:
        """Set up and return the axes to draw on, creating one if needed."""
        if ax is not None:
            self.ax = ax
            self.fig = ax.get_figure()
        elif self.ax is None:
            self.fig, self.ax = self.create_figure_axes()
        return self.ax

    def _apply_title(self, ax: Axes, title: str | None) -> None:
        """Apply a title override (or the option default) to `ax`."""
        if title is not None:
            self.default_options["title"] = title
        if self.default_options["title"]:
            ax.set_title(
                self.default_options["title"],
                fontsize=self.default_options["title_size"],
            )

    def line(
        self,
        ax: Axes = None,
        title: str | None = None,
        label: str | list[str] | None = None,
        color=None,
        **kwargs,
    ):
        """Draw line/marker series with `Axes.plot`.

        Args:
            ax: Axes to draw on. Falls back to the construction axes or
                a new figure/axes.
            title: Plot title override.
            label: Legend label(s); a single string for 1D `y` or one
                per series for 2D `y`.
            color: Line colour. Defaults to the `color_1` option.
            **kwargs: Forwarded to `Axes.plot`.

        Returns:
            tuple[Figure, Axes, list[Line2D]]: The figure, the axes, and
                the list of drawn lines (one per series).

        Examples:
            - Two series produce two lines:
                ```python
                >>> import numpy as np
                >>> from cleopatra.line_glyph import LineGlyph
                >>> x = np.array([0.0, 1.0, 2.0])
                >>> y = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
                >>> glyph = LineGlyph(x, y)
                >>> fig, ax, lines = glyph.line()
                >>> len(lines)
                2

                ```
        """
        ax = self._resolve_ax(ax)
        opts = self.default_options
        color = color if color is not None else opts["color_1"]
        series = self._series()
        labels = (
            label if isinstance(label, (list, tuple))
            else [label] * len(series)
        )
        lines = []
        for col, lab in zip(series, labels):
            drawn = ax.plot(
                self.x, col,
                linestyle=opts["linestyle"],
                marker=opts["marker"],
                linewidth=opts["line_width"],
                alpha=opts["alpha"],
                label=lab,
                **({"color": color} if len(series) == 1 else {}),
                **kwargs,
            )
            lines.extend(drawn)
        self._apply_title(ax, title)
        return self.fig, ax, lines

    def bar(
        self,
        ax: Axes = None,
        title: str | None = None,
        color=None,
        **kwargs,
    ):
        """Draw a bar chart of a single series with `Axes.bar`.

        Args:
            ax: Axes to draw on. Falls back to the construction axes or
                a new figure/axes.
            title: Plot title override.
            color: Bar colour. Defaults to the `color_1` option.
            **kwargs: Forwarded to `Axes.bar`.

        Returns:
            tuple[Figure, Axes, BarContainer]: The figure, the axes, and
                the bar container.

        Raises:
            ValueError: If `y` is not 1D (bar charts take a single
                series).

        Examples:
            - One bar per x value:
                ```python
                >>> import numpy as np
                >>> from cleopatra.line_glyph import LineGlyph
                >>> glyph = LineGlyph(np.array([0.0, 1.0, 2.0]), np.array([3.0, 1.0, 2.0]))
                >>> fig, ax, bars = glyph.bar()
                >>> len(bars)
                3

                ```
        """
        if self.y.ndim != 1:
            raise ValueError(
                f"bar requires 1D y (a single series); got {self.y.ndim}D."
            )
        ax = self._resolve_ax(ax)
        opts = self.default_options
        color = color if color is not None else opts["color_1"]
        bars = ax.bar(
            self.x, self.y, color=color, alpha=opts["alpha"], **kwargs
        )
        self._apply_title(ax, title)
        return self.fig, ax, bars

    def fill_between(
        self,
        y2: float | np.ndarray = 0.0,
        ax: Axes = None,
        title: str | None = None,
        color=None,
        alpha: float | None = None,
        **kwargs,
    ):
        """Fill the band between `y` and `y2` with `Axes.fill_between`.

        Useful for envelopes and quantile spreads: `y` is the upper
        curve and `y2` the lower (a scalar baseline or a matching
        array). Requires 1D `y`.

        Args:
            y2: Lower curve, scalar or array matching `x`. Default is
                0.0.
            ax: Axes to draw on. Falls back to the construction axes or
                a new figure/axes.
            title: Plot title override.
            color: Fill colour. Defaults to the `color_1` option.
            alpha: Fill transparency. Defaults to `0.3` for a band look
                (overrides the glyph's `alpha` option for the fill).
            **kwargs: Forwarded to `Axes.fill_between`.

        Returns:
            tuple[Figure, Axes, PolyCollection]: The figure, the axes,
                and the filled band collection.

        Raises:
            ValueError: If `y` is not 1D.

        Examples:
            - Band between an upper curve and a scalar baseline:
                ```python
                >>> import numpy as np
                >>> from cleopatra.line_glyph import LineGlyph
                >>> glyph = LineGlyph(np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 2.0]))
                >>> fig, ax, band = glyph.fill_between(y2=0.0)
                >>> band.get_paths() is not None
                True

                ```
        """
        if self.y.ndim != 1:
            raise ValueError(
                f"fill_between requires 1D y; got {self.y.ndim}D."
            )
        ax = self._resolve_ax(ax)
        opts = self.default_options
        color = color if color is not None else opts["color_1"]
        alpha = alpha if alpha is not None else 0.3
        band = ax.fill_between(
            self.x, self.y, y2, color=color, alpha=alpha, **kwargs
        )
        self._apply_title(ax, title)
        return self.fig, ax, band
