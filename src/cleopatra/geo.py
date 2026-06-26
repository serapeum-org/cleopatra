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

import importlib.util
from typing import Any

from cleopatra import reference, tiles


def _validate_crs(crs: int | str | None) -> int | str | None:
    """Validate a value assigned to `GeoMixin.crs`, returning it unchanged.

    Cheap type/shape checks always run (and need no third-party package);
    full CRS-resolvability is additionally checked with `pyproj` **only when
    it is installed**, so setting `crs` never requires the optional
    `cleopatra[tiles]` extra. `None` is always accepted. When `pyproj` is
    absent, an unresolvable-but-well-typed CRS is still caught later, at
    draw time, by `add_features` / `add_tiles`.

    Args:
        crs: An int EPSG code, a CRS string, or `None`.

    Returns:
        The validated `crs`, unchanged.

    Raises:
        TypeError: If `crs` is not an int, str, or `None` (`bool` is
            rejected).
        ValueError: If `crs` is a non-positive EPSG code, an empty string,
            or (when `pyproj` is installed) an unresolvable CRS.

    Examples:
        - `None` and well-formed values pass through unchanged:
            ```python
            >>> from cleopatra.geo import _validate_crs
            >>> _validate_crs(None) is None
            True
            >>> _validate_crs(4326)
            4326

            ```
        - Wrong types are rejected immediately:
            ```python
            >>> from cleopatra.geo import _validate_crs
            >>> _validate_crs([4326])
            Traceback (most recent call last):
                ...
            TypeError: crs must be an int EPSG code, a CRS string, or None, got list

            ```
    """
    if crs is None:
        return None
    if isinstance(crs, bool) or not isinstance(crs, (int, str)):
        raise TypeError(
            "crs must be an int EPSG code, a CRS string, or None, got "
            f"{type(crs).__name__}"
        )
    if isinstance(crs, int) and crs <= 0:
        raise ValueError(f"crs EPSG code must be a positive int, got {crs}")
    if isinstance(crs, str) and not crs.strip():
        raise ValueError("crs string must be a non-empty CRS identifier")
    if importlib.util.find_spec("pyproj") is not None:
        from pyproj import CRS
        from pyproj.exceptions import CRSError

        try:
            CRS.from_user_input(crs)
        except CRSError as e:
            raise ValueError(f"Invalid CRS {crs!r}: {e}") from e
    return crs


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

    #: Backing store for the validated `crs` property; `None` means unset.
    _crs: int | str | None = None

    @property
    def crs(self) -> int | str | None:
        """CRS of the data plotted on `self.ax` (EPSG code or CRS string).

        When set, `add_features` / `add_tiles` default `crs=` to it; `None`
        keeps each helper's own default. The value is **validated on
        assignment** (see `cleopatra.geo._validate_crs`) so mistakes surface
        at `glyph.crs = ...` rather than later, when a basemap is drawn.

        Raises:
            TypeError: If assigned something other than an int, str, or
                `None`.
            ValueError: If assigned a non-positive EPSG code, an empty
                string, or (when `pyproj` is installed) an unresolvable CRS.
        """
        return self._crs

    @crs.setter
    def crs(self, value: int | str | None) -> None:
        self._crs = _validate_crs(value)

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
        and an explicit `crs=` always wins. `crs` is keyword-only in both
        `add_features` and `add_tiles`, so it always arrives via `kwargs`.

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
