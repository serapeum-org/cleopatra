"""Vector-field visualization.

Provides `VectorGlyph` for plotting 2D vector fields as arrows
(`quiver`), wind barbs (`barbs`), or streamlines (`streamplot`), with
arrows/lines coloured by vector magnitude. Magnitude colour mapping
reuses the shared `Glyph._prepare_scalar_mapping` pipeline so `vmin` /
`vmax`, `ticks_spacing`, `levels`, and `color_scale` behave exactly as
for the other glyphs.

Examples:
    - Quiver plot coloured by magnitude:
        ```python
        >>> import numpy as np
        >>> from cleopatra.vector_glyph import VectorGlyph
        >>> x, y = np.meshgrid(np.arange(3), np.arange(3))
        >>> u = np.ones_like(x, dtype=float)
        >>> v = np.zeros_like(y, dtype=float)
        >>> glyph = VectorGlyph(x, y, u, v)
        >>> fig, ax, im = glyph.plot(kind="quiver")

        ```
    - Add a reference-arrow key:
        ```python
        >>> key = glyph.add_key(im, value=1.0, label="1 m/s")

        ```
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.quiver import QuiverKey

from cleopatra.glyph import Glyph
from cleopatra.styles import CLASSIFY_OPTIONS
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS

#: Vector kinds dispatched by `VectorGlyph.plot`.
VECTOR_KINDS = ("quiver", "barbs", "streamplot")

#: Option keys for VectorGlyph. `ticks_spacing` is `None` so the shared
#: `_prepare_scalar_mapping` helper auto-derives it from the magnitude.
VECTOR_DEFAULT_OPTIONS = {
    "density": 1.0,
    "scale": None,
    "vmin": None,
    "vmax": None,
    "levels": None,
    "ticks_spacing": None,
    "add_colorbar": True,
}
VECTOR_DEFAULT_OPTIONS = STYLE_DEFAULTS | CLASSIFY_OPTIONS | VECTOR_DEFAULT_OPTIONS


class VectorGlyph(Glyph):
    """Visualization class for 2D vector fields.

    Renders a `(u, v)` vector field over `(x, y)` positions as arrows,
    wind barbs, or streamlines, with the artist coloured by the vector
    magnitude `hypot(u, v)` through the shared scalar-mapping pipeline.

    Args:
        x: x-coordinates of the vector positions.
        y: y-coordinates of the vector positions.
        u: x-components of the vectors. Must broadcast against `x`/`y`.
        v: y-components of the vectors. Must broadcast against `x`/`y`.
        ax: Pre-existing axes to draw on. Default is None.
        fig: Pre-existing figure. Default is None.
        **kwargs: Override any key in `VECTOR_DEFAULT_OPTIONS`
            (e.g. `density`, `scale`, `cmap`, `vmin`, `vmax`, `levels`,
            `color_scale`, `ticks_spacing`, `cbar_label`, `figsize`,
            `title`). Set `add_colorbar=False` to suppress the per-glyph
            colorbar (default True) for shared-axes composition where the
            host owns a single aggregated colorbar.

    Examples:
        - Build a field and inspect the stored magnitude:
            ```python
            >>> import numpy as np
            >>> from cleopatra.vector_glyph import VectorGlyph
            >>> x, y = np.meshgrid(np.arange(2), np.arange(2))
            >>> u = np.array([[3.0, 0.0], [0.0, 3.0]])
            >>> v = np.array([[4.0, 0.0], [0.0, 4.0]])
            >>> glyph = VectorGlyph(x, y, u, v)
            >>> float(glyph.magnitude.max())
            5.0

            ```

    See Also:
        cleopatra.glyph.Glyph._prepare_scalar_mapping: Shared
            norm/colorbar/ticks pipeline used to colour by magnitude.
    """

    #: Option keys this glyph accepts (see `Glyph.option_keys`/`filter_kwargs`).
    DEFAULT_OPTIONS = VECTOR_DEFAULT_OPTIONS

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        *,
        ax: Axes = None,
        fig: Figure = None,
        **kwargs,
    ):
        super().__init__(
            default_options=VECTOR_DEFAULT_OPTIONS, fig=fig, ax=ax, **kwargs
        )
        self.x, self.y, self.u, self.v = (np.asarray(a) for a in (x, y, u, v))
        if self.u.shape != self.v.shape:
            raise ValueError(
                f"u and v must have the same shape, got {self.u.shape} "
                f"and {self.v.shape}."
            )
        self.cbar = None

    @property
    def magnitude(self) -> np.ndarray:
        """Per-vector magnitude `hypot(u, v)` used for colour mapping."""
        return np.hypot(self.u, self.v)

    def plot(
        self,
        kind: str = "quiver",
        ax: Axes = None,
        title: str | None = None,
        add_colorbar: bool | None = None,
    ):
        """Render the vector field, coloured by magnitude.

        Dispatches to `Axes.quiver`, `Axes.barbs`, or `Axes.streamplot`
        based on `kind`. The colour scale, norm, ticks, and colorbar are
        resolved from the magnitude via `_prepare_scalar_mapping`.

        Args:
            kind: One of `"quiver"`, `"barbs"`, or `"streamplot"`.
                Default is `"quiver"`.
            ax: Axes to draw on. Falls back to the axes supplied at
                construction, otherwise a new figure/axes is created.
            title: Plot title. Overrides `default_options["title"]`
                when given.
            add_colorbar: Override the `add_colorbar` option for this call
                — True draws the colorbar, False suppresses it (for
                shared-axes composition). Defaults to None, which keeps the
                value set at construction.

        Returns:
            tuple[Figure, Axes, Any]: The figure, the axes, and the
                mappable artist (the `Quiver`, `Barbs`, or the
                streamplot's `LineCollection`) that the colorbar is
                attached to.

        Raises:
            ValueError: If `kind` is not a recognised vector kind, or if
                the magnitude has no finite values (via
                `_prepare_scalar_mapping`).

        Examples:
            - A barbs plot returns the Barbs mappable with the magnitude
                array:
                ```python
                >>> import numpy as np
                >>> from cleopatra.vector_glyph import VectorGlyph
                >>> x, y = np.meshgrid(np.arange(3), np.arange(3))
                >>> u = np.full_like(x, 2.0, dtype=float)
                >>> v = np.zeros_like(y, dtype=float)
                >>> glyph = VectorGlyph(x, y, u, v)
                >>> fig, ax, im = glyph.plot(kind="barbs")
                >>> float(im.get_array().max())
                2.0

                ```
            - An unknown kind raises ValueError:
                ```python
                >>> import numpy as np
                >>> from cleopatra.vector_glyph import VectorGlyph
                >>> x, y = np.meshgrid(np.arange(2), np.arange(2))
                >>> glyph = VectorGlyph(x, y, np.ones_like(x), np.ones_like(y))
                >>> glyph.plot(kind="spirograph")
                Traceback (most recent call last):
                    ...
                ValueError: unknown vector kind 'spirograph'; expected one of ...

                ```
        """
        if kind not in VECTOR_KINDS:
            raise ValueError(
                f"unknown vector kind {kind!r}; expected one of "
                f"{', '.join(VECTOR_KINDS)}."
            )

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

        mag = self.magnitude
        norm, cbar_kw, ticks = self._prepare_scalar_mapping(mag)
        cmap = opts["cmap"]
        clim = {} if norm else {"clim": (ticks[0], ticks[-1])}

        if kind == "quiver":
            im = ax.quiver(
                self.x, self.y, self.u, self.v, mag,
                cmap=cmap, norm=norm, scale=opts["scale"], **clim,
            )
        elif kind == "barbs":
            im = ax.barbs(
                self.x, self.y, self.u, self.v, mag,
                cmap=cmap, norm=norm, **clim,
            )
        else:  # streamplot
            stream = ax.streamplot(
                self.x, self.y, self.u, self.v,
                color=mag, cmap=cmap, norm=norm, density=opts["density"],
            )
            im = stream.lines
            if im.get_array() is None:
                im.set_array(np.asarray(mag).ravel())
            # streamplot has no clim kwarg; pin its LineCollection to the
            # same tick range quiver/barbs use so colours and colorbar
            # agree on the linear (norm is None) path.
            if norm is None:
                im.set_clim(ticks[0], ticks[-1])

        if draw_colorbar:
            self.cbar = self.create_color_bar(ax, im, cbar_kw)

        if opts["title"]:
            ax.set_title(opts["title"], fontsize=opts["title_size"])

        return self.fig, ax, im

    def add_key(
        self,
        im,
        x: float = 0.9,
        y: float = 1.02,
        value: float = 10.0,
        label: str | None = None,
        labelpos: str = "E",
        **kwargs,
    ) -> QuiverKey:
        """Add a reference-arrow key to a quiver plot.

        Wraps `Axes.quiverkey` to draw a sample arrow of known length
        with a text label, the standard legend for a quiver field.

        Args:
            im: The `Quiver` artist returned by `plot(kind="quiver")`.
            x: Key x-position in axes fraction coordinates.
                Default is 0.9.
            y: Key y-position in axes fraction coordinates.
                Default is 1.02.
            value: The reference vector length the key represents.
                Default is 10.0.
            label: Text drawn beside the key. Default is `None`, which
                renders the numeric `value` as the label.
            labelpos: Side of the arrow for the label (`"N"`, `"S"`,
                `"E"`, `"W"`). Default is `"E"`.
            **kwargs: Forwarded to `Axes.quiverkey`.

        Returns:
            QuiverKey: The created key artist.

        Examples:
            - Add a 5 m/s reference key to a quiver:
                ```python
                >>> import numpy as np
                >>> from cleopatra.vector_glyph import VectorGlyph
                >>> x, y = np.meshgrid(np.arange(3), np.arange(3))
                >>> u = np.ones_like(x, dtype=float)
                >>> v = np.ones_like(y, dtype=float)
                >>> glyph = VectorGlyph(x, y, u, v)
                >>> fig, ax, im = glyph.plot(kind="quiver")
                >>> key = glyph.add_key(im, value=5.0, label="5 m/s")
                >>> key.text.get_text()
                '5 m/s'

                ```
        """
        text = label if label is not None else f"{value:g}"
        return self.ax.quiverkey(im, x, y, value, text, labelpos=labelpos, **kwargs)
