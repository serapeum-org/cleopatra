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

Importing this module (and the `cleopatra.tiles` / `cleopatra.reference`
modules it calls) does not require the optional `cleopatra[tiles]` extra:
those modules gate their `[tiles]` dependencies (`mercantile`, `pyproj`,
`Pillow`, ...) behind their own internal lazy imports, so the extra is
only needed when a basemap is actually drawn.
"""

from __future__ import annotations

from typing import Any

from cleopatra import reference, tiles


class GeoMixin:
    """Mixin giving geographic glyphs `add_tiles` / `add_features` / `add_relief`.

    The host class is expected to expose the plotted axes as `self.ax`
    (every `cleopatra.glyph.Glyph` subclass does). Call these after
    plotting, or pass `ax=` explicitly.

    Set `self.crs` to the CRS of the data plotted on the axes (an EPSG code
    or CRS string) and `add_features` / `add_tiles` default their `crs=`
    argument to it, so the reference layer is placed in matching
    coordinates without restating it on every call. An explicit `crs=`
    still wins; leaving `self.crs` as `None` preserves each helper's own
    default. `add_relief` ignores `crs` -- relief is a fixed EPSG:4326
    raster placed by `extent`.
    """

    #: Set by `cleopatra.glyph.Glyph`; the axes the basemap is drawn on.
    ax: Any

    #: CRS of the data plotted on `self.ax` (EPSG code or CRS string). When
    #: set, `add_features` / `add_tiles` default `crs=` to it. `None` keeps
    #: each helper's own default.
    crs: int | str | None = None

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

    def _basemap_kwargs(self, kwargs: dict) -> dict:
        """Default `crs` to `self.crs` when the caller did not set it.

        Only injects when `self.crs` is set and `crs` is absent (or `None`)
        in `kwargs`, so the default `self.crs is None` is a pure pass-through
        and an explicit `crs=` always wins. Pass `crs` as a keyword (not
        positionally) for this defaulting to apply.

        Args:
            kwargs: The keyword arguments destined for the basemap helper.

        Returns:
            dict: `kwargs`, with `crs` filled in from `self.crs` when needed.
        """
        if self.crs is not None and kwargs.get("crs") is None:
            return {**kwargs, "crs": self.crs}
        return kwargs

    def add_tiles(self, *args: Any, ax: Any = None, **kwargs: Any) -> Any:
        """Overlay a web-tile basemap on the glyph's axes.

        Thin wrapper over `cleopatra.tiles.add_tiles`; positional and
        keyword arguments are forwarded unchanged (e.g. `source`, `crs`,
        `zoom`, `alpha`). When `crs` is omitted it defaults to `self.crs`.
        Requires the `cleopatra[tiles]` extra.

        Args:
            *args: Positional arguments for `cleopatra.tiles.add_tiles`
                (after the axes).
            ax: Axes to draw on. Defaults to the glyph's `self.ax`.
            **kwargs: Keyword arguments for `cleopatra.tiles.add_tiles`. A
                `crs` keyword is defaulted to `self.crs` when omitted; an
                explicit `crs=` overrides it.

        Returns:
            matplotlib.axes.Axes: The axes, for chaining.

        Raises:
            RuntimeError: If the glyph has no axes yet and `ax` is not given.

        See Also:
            cleopatra.tiles.add_tiles: The underlying implementation and its
                full parameter list.
        """
        return tiles.add_tiles(
            self._basemap_axes(ax), *args, **self._basemap_kwargs(kwargs)
        )

    def add_features(self, *args: Any, ax: Any = None, **kwargs: Any) -> Any:
        """Draw a Natural Earth reference layer on the glyph's axes.

        Thin wrapper over `cleopatra.reference.add_features`; arguments are
        forwarded unchanged (e.g. `layer`, `resolution`, `crs`, and style
        keywords). When `crs` is omitted it defaults to `self.crs`.

        Args:
            *args: Positional arguments for
                `cleopatra.reference.add_features` (after the axes), such as
                `layer` and `resolution`.
            ax: Axes to draw on. Defaults to the glyph's `self.ax`.
            **kwargs: Keyword arguments for
                `cleopatra.reference.add_features`. A `crs` keyword is
                defaulted to `self.crs` when omitted; an explicit `crs=`
                overrides it.

        Returns:
            matplotlib.axes.Axes: The axes, for chaining.

        Raises:
            RuntimeError: If the glyph has no axes yet and `ax` is not given.

        See Also:
            cleopatra.reference.add_features: The underlying implementation
                and its full parameter list.
        """
        return reference.add_features(
            self._basemap_axes(ax), *args, **self._basemap_kwargs(kwargs)
        )

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
        return reference.add_relief(self._basemap_axes(ax), *args, **kwargs)
