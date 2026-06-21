"""Geographic basemap convenience methods for glyphs.

`GeoMixin` adds three convenience methods -- `add_tiles`, `add_features`,
and `add_relief` -- to the glyph classes that plot geographic data, so a
basemap can be dropped under a plot without importing the standalone
helpers and without repeating the axes:

    >>> glyph.plot()                 # doctest: +SKIP
    >>> glyph.add_relief("low")      # doctest: +SKIP
    >>> glyph.add_features("coastline", "50m")  # doctest: +SKIP

Each method is a thin wrapper that draws on the glyph's own axes
(`self.ax`) and delegates to the single implementation in
`cleopatra.tiles` / `cleopatra.reference`. The standalone functions remain
the source of truth; this mixin only removes the import + explicit-axes
boilerplate for the geographic glyphs (`ArrayGlyph`, `MeshGlyph`,
`VectorGlyph`, `FlowGlyph`, `PolygonGlyph`, `ScatterGlyph`). Non-geographic
glyphs (line/bar charts, statistical plots) deliberately do not inherit
it.

The basemap helpers are imported lazily inside each method so importing a
glyph never pulls in the optional `cleopatra[tiles]` extra unless a basemap
is actually requested.
"""

from __future__ import annotations

from typing import Any


class GeoMixin:
    """Mixin giving geographic glyphs `add_tiles` / `add_features` / `add_relief`.

    The host class is expected to expose the plotted axes as `self.ax`
    (every `cleopatra.glyph.Glyph` subclass does). Call these after
    plotting, or pass `ax=` explicitly.
    """

    #: Set by `cleopatra.glyph.Glyph`; the axes the basemap is drawn on.
    ax: Any

    def _basemap_axes(self, ax: Any = None) -> Any:
        """Return the axes to draw a basemap on (`ax` or `self.ax`).

        Args:
            ax: An explicit axes to use instead of the glyph's own.

        Returns:
            The resolved matplotlib axes.

        Raises:
            RuntimeError: If neither `ax` nor `self.ax` is available -- the
                glyph has not been plotted yet.
        """
        target = ax if ax is not None else getattr(self, "ax", None)
        if target is None:
            raise RuntimeError(
                "No axes to draw on. Plot the glyph first (or pass ax=) "
                "before adding a basemap layer."
            )
        return target

    def add_tiles(self, *args: Any, ax: Any = None, **kwargs: Any) -> Any:
        """Overlay a web-tile basemap on the glyph's axes.

        Thin wrapper over `cleopatra.tiles.add_tiles`; positional and
        keyword arguments are forwarded unchanged (e.g. `source`, `crs`,
        `zoom`, `alpha`). Requires the `cleopatra[tiles]` extra.

        Args:
            *args: Positional arguments for `cleopatra.tiles.add_tiles`
                (after the axes).
            ax: Axes to draw on. Defaults to the glyph's `self.ax`.
            **kwargs: Keyword arguments for `cleopatra.tiles.add_tiles`.

        Returns:
            matplotlib.axes.Axes: The axes, for chaining.

        Raises:
            RuntimeError: If the glyph has no axes yet and `ax` is not given.

        See Also:
            cleopatra.tiles.add_tiles: The underlying implementation and its
                full parameter list.
        """
        from cleopatra.tiles import add_tiles

        return add_tiles(self._basemap_axes(ax), *args, **kwargs)

    def add_features(self, *args: Any, ax: Any = None, **kwargs: Any) -> Any:
        """Draw a Natural Earth reference layer on the glyph's axes.

        Thin wrapper over `cleopatra.reference.add_features`; arguments are
        forwarded unchanged (e.g. `layer`, `resolution`, `crs`, and style
        keywords).

        Args:
            *args: Positional arguments for
                `cleopatra.reference.add_features` (after the axes), such as
                `layer` and `resolution`.
            ax: Axes to draw on. Defaults to the glyph's `self.ax`.
            **kwargs: Keyword arguments for
                `cleopatra.reference.add_features`.

        Returns:
            matplotlib.axes.Axes: The axes, for chaining.

        Raises:
            RuntimeError: If the glyph has no axes yet and `ax` is not given.

        See Also:
            cleopatra.reference.add_features: The underlying implementation
                and its full parameter list.
        """
        from cleopatra.reference import add_features

        return add_features(self._basemap_axes(ax), *args, **kwargs)

    def add_relief(self, *args: Any, ax: Any = None, **kwargs: Any) -> Any:
        """Draw a hypsometric relief backdrop under the glyph's data.

        Thin wrapper over `cleopatra.reference.add_relief`; arguments are
        forwarded unchanged (e.g. `resolution`, `extent`, `alpha`).
        Requires the `cleopatra[tiles]` extra (Pillow).

        Args:
            *args: Positional arguments for
                `cleopatra.reference.add_relief` (after the axes), such as
                `resolution`.
            ax: Axes to draw on. Defaults to the glyph's `self.ax`.
            **kwargs: Keyword arguments for
                `cleopatra.reference.add_relief`.

        Returns:
            matplotlib.axes.Axes: The axes, for chaining.

        Raises:
            RuntimeError: If the glyph has no axes yet and `ax` is not given.

        See Also:
            cleopatra.reference.add_relief: The underlying implementation and
                its full parameter list.
        """
        from cleopatra.reference import add_relief

        return add_relief(self._basemap_axes(ax), *args, **kwargs)
