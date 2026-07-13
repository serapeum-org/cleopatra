"""Base visualization class for cleopatra glyphs.

Provides shared infrastructure for array-based and mesh-based
visualization: figure/axes lifecycle, color scale normalization,
colorbar creation, tick management, point overlays, and animation.
"""

from __future__ import annotations

import inspect
import math
import os
import warnings
from typing import Any

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure, SubFigure
from matplotlib.legend import Legend
from matplotlib.ticker import LogFormatter

# `SUPPORTED_VIDEO_FORMAT` is re-imported (not redefined) so the constant has
# a single source of truth in `cleopatra.animation`, while the historical
# `from cleopatra.glyph import SUPPORTED_VIDEO_FORMAT` path keeps working.
from cleopatra.animation import SUPPORTED_VIDEO_FORMAT  # noqa: F401  (re-export)
from cleopatra.animation import save_animation as _save_animation
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS
from cleopatra.styles import (
    ColorScale,
    MidpointNormalize,
    categorize,
    classify,
    disjoint_legend,
)

#: Upper bound for an integer `levels` value (number of discrete colour
#: levels / contour lines). A larger request is almost certainly a mistake
#: and `np.linspace` with a huge count would exhaust memory.
MAX_DISCRETE_LEVELS = 1000

#: Qualitative colormap `_prepare_categorical_mapping` falls back to when the
#: caller left `cmap` at the shared continuous/diverging default -- see the
#: fallback logic there.
CATEGORICAL_DEFAULT_CMAP = "tab10"


def _get_figure_supports_root(get_figure) -> bool:
    """Return True if `get_figure` accepts a `root` keyword argument.

    The `root` parameter was added to `Axes.get_figure` in matplotlib 3.10.
    Detected by signature inspection (rather than a broad `try/except`) so an
    unrelated `TypeError` from `get_figure` itself is never swallowed.
    """
    try:
        return "root" in inspect.signature(get_figure).parameters
    except (TypeError, ValueError):
        return False


def _root_figure(ax: Axes) -> Figure:
    """Return the top-level `Figure` that owns `ax`, across matplotlib versions.

    On matplotlib >= 3.10 this uses `Axes.get_figure(root=True)`, which returns
    the root `Figure` even when the axes lives on a `SubFigure` (and avoids the
    3.10 deprecation warning attached to the bare `get_figure()`). On older
    matplotlib (down to the project's 3.8.4 floor) the `root` keyword does not
    exist, so it climbs out of any `SubFigure` to the owning `Figure` manually.

    Args:
        ax: The axes whose top-level figure is wanted.

    Returns:
        Figure: The top-level figure for `ax`.
    """
    get_figure = ax.get_figure
    if _get_figure_supports_root(get_figure):
        return get_figure(root=True)
    fig = get_figure()
    seen: set[int] = set()
    while isinstance(fig, SubFigure) and id(fig) not in seen:
        seen.add(id(fig))
        fig = fig.figure
    return fig


def _figure_is_open(fig: Figure | None) -> bool:
    """True if `fig` is a live top-level `Figure` still registered with pyplot.

    Pass a root `Figure` (resolve a `SubFigure` via `_root_figure` first); a
    figure whose window/number has been closed, or a `SubFigure` (no `.number`),
    returns `False`.
    """
    num = getattr(fig, "number", None)
    return num is not None and plt.fignum_exists(num)


def _immediate_figure(ax: Axes) -> Figure:
    """Return the figure `ax` is directly attached to (its immediate parent).

    Deprecation-safe counterpart to `_root_figure`: on matplotlib >= 3.10 it
    passes `root=False` explicitly; on older matplotlib it calls the bare
    `get_figure()`. For an axes on a `SubFigure` this is that `SubFigure`; for
    an ordinary axes it is the same object as `_root_figure(ax)`.

    Args:
        ax: The axes whose immediate parent figure is wanted.

    Returns:
        Figure: The figure (or sub-figure) `ax` is directly attached to.
    """
    get_figure = ax.get_figure
    if _get_figure_supports_root(get_figure):
        return get_figure(root=False)
    return get_figure()


class Glyph:
    """Base class for cleopatra visualization glyphs.

    Handles figure/axes management, default options, color scale
    normalization, colorbar creation, tick control, point overlays,
    and animation saving. Subclasses implement the actual rendering.

    The accepted option keys are exposed per subclass via the
    `DEFAULT_OPTIONS` class attribute, and can be inspected or filtered
    *before* constructing an instance with the `option_keys` and
    `filter_kwargs` classmethods (useful for safely forwarding a bag of
    user-supplied styling kwargs).

    Args:
        default_options: Default plot options dict. Subclasses provide
            their own defaults merged with `STYLE_DEFAULTS`.
        fig: Pre-existing matplotlib figure to bind. Default is None.
            An `ax` fully determines its figure, so `fig` is optional even
            when `ax` is given; when both are passed the explicit `fig`
            is kept as the figure handle. Passing a `fig` that does not own
            the given `ax` emits a `UserWarning` (the explicit `fig` is
            still honoured, but the two handles then disagree).
        ax: Pre-existing matplotlib axes to bind. Default is None. Passing
            `ax` on its own is supported — its parent figure is derived
            automatically (the axes is no longer dropped when `fig` is
            omitted).
        **kwargs: Override any key in `default_options`.

    Examples:
        - Create a Glyph and override the colormap:
            ```python
            >>> from cleopatra.glyph import Glyph
            >>> from cleopatra.styles import DEFAULT_OPTIONS
            >>> opts = DEFAULT_OPTIONS.copy()
            >>> opts["vmin"] = None
            >>> opts["vmax"] = None
            >>> g = Glyph(default_options=opts, cmap="plasma")
            >>> g.default_options["cmap"]
            'plasma'

            ```
        - Provide a pre-existing figure and axes:
            ```python
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.glyph import Glyph
            >>> from cleopatra.styles import DEFAULT_OPTIONS
            >>> opts = DEFAULT_OPTIONS.copy()
            >>> opts["vmin"] = None
            >>> opts["vmax"] = None
            >>> fig, ax = plt.subplots()
            >>> g = Glyph(default_options=opts, fig=fig, ax=ax)
            >>> g.fig is fig
            True
            >>> g.ax is ax
            True

            ```
        - Provide only an axes; the figure is derived from it:
            ```python
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.glyph import Glyph
            >>> from cleopatra.styles import DEFAULT_OPTIONS
            >>> opts = DEFAULT_OPTIONS.copy()
            >>> opts["vmin"] = None
            >>> opts["vmax"] = None
            >>> fig, ax = plt.subplots()
            >>> g = Glyph(default_options=opts, ax=ax)
            >>> g.ax is ax
            True
            >>> g.fig is ax.get_figure()
            True

            ```

    See Also:
        cleopatra.array_glyph.ArrayGlyph: Glyph subclass for
            2D/3D arrays.
        cleopatra.mesh_glyph.MeshGlyph: Glyph subclass for
            unstructured meshes.
    """

    #: The option keys this glyph accepts, as a class attribute so they can
    #: be introspected/filtered *before* an instance exists (see
    #: `option_keys`/`filter_kwargs`). Each subclass overrides this with its
    #: own option dict (built as `STYLE_DEFAULTS | <glyph-specific>`); the
    #: base value is the shared style defaults.
    DEFAULT_OPTIONS: dict = STYLE_DEFAULTS

    #: Whether this glyph's `plot()` reads back the categorical side-channel
    #: (`self._categorical`) instead of feeding raw `values` straight into
    #: the mappable. Only true for glyphs whose per-element value is a
    #: nominal class label rather than a continuous magnitude (e.g.
    #: `PolygonGlyph`, `ScatterGlyph`) — `scheme="categorical"` is rejected
    #: for any other glyph rather than silently mis-colouring it.
    _SUPPORTS_CATEGORICAL_SCHEME = False

    def __init__(
        self,
        default_options: dict,
        fig: Figure = None,
        ax: Axes = None,
        **kwargs,
    ):
        self._default_options = default_options.copy()
        self._merge_kwargs(kwargs)
        self._vmin: float | None = None
        self._vmax: float | None = None
        self.ticks_spacing: float | None = None
        #: Set by `_prepare_categorical_mapping` when `scheme="categorical"`
        #: — `{"codes", "cmap", "colors", "labels"}` — else `None`.
        self._categorical: dict | None = None
        # Resolve the (fig, ax) binding. An `ax` fully determines its
        # figure, so accept `ax` on its own and derive the figure from it
        # rather than dropping the axes when `fig` is omitted. An explicit
        # `fig` is honoured (and wins for the figure handle when both are
        # given); passing neither leaves both unset until render time.
        if ax is not None:
            self.ax = ax
            if fig is not None:
                # A mismatched (fig, ax) pair leaves self.fig and
                # self.ax.figure disagreeing — almost always a caller mistake.
                # `fig` is fine if it is either the axes' immediate parent
                # (e.g. a SubFigure) or its top-level root figure.
                if fig is not _immediate_figure(ax) and fig is not _root_figure(ax):
                    warnings.warn(
                        "The given `fig` is not the figure that owns `ax`; "
                        "the axes' own figure is what will be drawn on. Pass "
                        "only `ax` (its figure is derived automatically).",
                        stacklevel=2,
                    )
                self.fig = fig
            else:
                self.fig = _root_figure(ax)
        elif fig is not None:
            self.fig = fig
            self.ax = None
        else:
            self.fig = None
            self.ax = None

    @property
    def vmin(self) -> float | None:
        """Minimum value for color scaling."""
        return self._vmin

    @property
    def vmax(self) -> float | None:
        """Maximum value for color scaling."""
        return self._vmax

    @property
    def default_options(self) -> dict:
        """Default plot options."""
        return self._default_options

    @classmethod
    def option_keys(cls) -> set[str]:
        """Return the keyword-argument keys this glyph accepts.

        Resolves from the class-level `DEFAULT_OPTIONS`, so the accepted
        keys can be inspected **without constructing an instance** (and
        therefore without tripping the strict unknown-kwarg check in
        `_merge_kwargs`). The keys differ per glyph subclass.

        This reports the class's *default* option set. For every concrete
        glyph subclass that equals the instance's accepted keys (each
        subclass passes the same dict to `__init__`). The base `Glyph`
        reports the shared `STYLE_DEFAULTS`; an instance built with a
        custom injected `default_options` is the one case where the two
        can differ, so base `Glyph` is not part of the introspection
        contract.

        Returns:
            set[str]: The accepted option keys for this glyph class.

        Examples:
            - Inspect the keys a glyph accepts before building one:
                ```python
                >>> from cleopatra.scatter_glyph import ScatterGlyph
                >>> keys = ScatterGlyph.option_keys()
                >>> "cmap" in keys
                True
                >>> "totally_unknown" in keys
                False

                ```
            - Different glyphs expose different keys:
                ```python
                >>> from cleopatra.polygon_glyph import PolygonGlyph
                >>> "edgecolor" in PolygonGlyph.option_keys()
                True

                ```

        See Also:
            filter_kwargs: Drop the keys a glyph does not accept from a dict.
        """
        return set(cls.DEFAULT_OPTIONS)

    @classmethod
    def filter_kwargs(cls, kwargs: dict) -> dict:
        """Return only the subset of `kwargs` whose keys this glyph accepts.

        A convenience for callers that forward a bag of user-supplied
        styling kwargs into a glyph: pre-filtering with this method lets
        the construction succeed instead of raising on an unknown key.
        Order and values are preserved; rejected keys are simply dropped.

        Args:
            kwargs: A mapping of candidate option keys to values.

        Returns:
            dict: The entries of `kwargs` whose keys are in `option_keys()`.

        Examples:
            - Keep only the accepted keys, then construct safely:
                ```python
                >>> from cleopatra.polygon_glyph import PolygonGlyph
                >>> raw = {"cmap": "viridis", "edgecolor": "black", "bogus": 1}
                >>> safe = PolygonGlyph.filter_kwargs(raw)
                >>> sorted(safe)
                ['cmap', 'edgecolor']
                >>> safe["cmap"]
                'viridis'

                ```
            - An empty mapping yields an empty mapping:
                ```python
                >>> from cleopatra.scatter_glyph import ScatterGlyph
                >>> ScatterGlyph.filter_kwargs({})
                {}

                ```

        See Also:
            option_keys: The set of keys this glyph accepts.
        """
        keys = cls.option_keys()
        return {key: val for key, val in kwargs.items() if key in keys}

    @property
    def anim(self) -> FuncAnimation:
        """Animation object created by `animate()`."""
        if hasattr(self, "_anim") and self._anim is not None:
            return self._anim
        raise ValueError(
            "Please first use the animate method to create the animation object"
        )

    def _merge_kwargs(self, kwargs: dict) -> None:
        """Validate and merge keyword arguments into default_options."""
        for key, val in kwargs.items():
            if key not in self._default_options:
                raise ValueError(
                    f"The given keyword argument:{key} is not correct, "
                    f"possible parameters are, {list(self._default_options.keys())}"
                )
            else:
                self._default_options[key] = val

    def create_figure_axes(self) -> tuple[Figure, Axes]:
        """Create a new figure and axes from default_options.

        Uses the `figsize` key from `default_options` to set the
        figure dimensions.

        Returns:
            tuple[Figure, Axes]: The created figure and axes.

        Examples:
            - Create a figure with custom size:
                ```python
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None})
                >>> g = Glyph(default_options=opts, figsize=(12, 4))
                >>> fig, ax = g.create_figure_axes()
                >>> fig.get_size_inches()
                array([12.,  4.])

                ```
        """
        fig, ax = plt.subplots(figsize=self.default_options["figsize"])
        return fig, ax

    def _reset_axes_for_restyle(self) -> None:
        """Prepare `self.ax` for an in-place restyle (used by `apply_style`).

        When the glyph has a **live** axes (already plotted and its figure is
        still open), the previous render is cleared from it -- the glyph's
        colorbar, any legend / swatch inset axes, and all artists -- so the
        restyle replaces the content in place. `apply_style` therefore takes
        full ownership of this axes and must not be used on an axes shared with
        unrelated caller content. When the glyph was never plotted, its figure
        was closed, or it was built with a figure but no axes, a fresh axes is
        created instead (on the existing figure when one is still open).
        """
        ax = getattr(self, "ax", None)
        fig = getattr(self, "fig", None)
        # Decide liveness by the ROOT figure's number: a SubFigure has no number
        # of its own, so resolving the root detects a closed parent Figure too.
        root = _root_figure(ax) if ax is not None else fig
        ax_live = ax is not None and _figure_is_open(root)
        if ax_live:
            for attr in ("cbar", "_cbar"):
                cbar = getattr(self, attr, None)
                if cbar is not None:
                    cbar.remove()
                    setattr(self, attr, None)
            for inset in list(self.ax.child_axes):
                inset.remove()
            self.ax.clear()
        elif _figure_is_open(fig):
            # A live figure with no (live) axes -- e.g. a `fig`-only construction:
            # reuse an existing axes on it if present, else add one, rather than
            # crashing when `plot` dereferences `self.ax` (or overlapping a
            # caller's own axes with a fresh `111` subplot).
            self.ax = fig.axes[0] if fig.axes else fig.add_subplot(111)
        else:
            self.fig, self.ax = self.create_figure_axes()

    def get_ticks(self) -> np.ndarray:
        """Compute colorbar tick locations from default_options.

        Uses `vmin`, `vmax`, and `ticks_spacing` from
        `default_options` to generate evenly-spaced tick positions.

        Returns:
            np.ndarray: Array of tick positions.

        Examples:
            - Compute ticks for a 0-10 range with spacing of 2:
                ```python
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": 0.0, "vmax": 10.0, "ticks_spacing": 2.0})
                >>> g = Glyph(default_options=opts)
                >>> g.get_ticks()
                array([ 0.,  2.,  4.,  6.,  8., 10.])

                ```
        """
        ticks_spacing = self.default_options["ticks_spacing"]
        vmax = self.default_options["vmax"]
        vmin = self.default_options["vmin"]
        # A degenerate colour range (e.g. a constant-value array where
        # vmax == vmin, so ticks_spacing is 0) has no meaningful tick
        # spacing; return a single tick at the value rather than dividing
        # by zero in `np.arange` / `math.remainder` below.
        if not ticks_spacing or vmax <= vmin:
            return np.array([vmin])
        ticks = np.arange(vmin, vmax + ticks_spacing, ticks_spacing)
        # If vmax is not evenly divisible by spacing, append one more tick.
        remainder = np.round(math.remainder(vmax, ticks_spacing), 3)
        if remainder != 0:
            ticks = np.append(
                ticks,
                [int(vmax / ticks_spacing) * ticks_spacing + ticks_spacing],
            )
        return ticks

    def _create_norm_and_cbar_kw(
        self, ticks: np.ndarray
    ) -> tuple[colors.Normalize | None, dict]:
        """Create a matplotlib Normalize and colorbar kwargs.

        Honours the `color_scale` option — a `cleopatra.styles.ColorScale`
        member or its string value (case-insensitive): `linear` / `power` /
        `sym-lognorm` / `boundary-norm` / `midpoint` — and the
        xarray-aligned `levels` and `extend` options when present in
        `default_options`. An unrecognised `color_scale` (including a
        non-string such as an int) raises `ValueError`.

        Behaviour for `levels`:

        * `levels` is `None` (default) — continuous norm based on
          `color_scale`.
        * `levels` is an `int` and `color_scale` is the default
          `"linear"` — switch to a `BoundaryNorm` with `levels`
          linearly-spaced edges between `vmin` and `vmax`.
        * `levels` is a sequence and `color_scale` is `"linear"` —
          use the sequence as explicit bin edges in a `BoundaryNorm`.
        * `levels` is set and `color_scale` is `"boundary-norm"`
          with no explicit `bounds` — treat `levels` as the bounds.
        * Otherwise (`color_scale` is some other enum value) — the
          user's choice wins; `levels` is left for the caller to
          forward to `contour` / `contourf`.

        Behaviour for `extend`: when present and non-None, the value
        is forwarded to the colorbar via `cbar_kw["extend"]`. The
        auto-resolution (`"both"` when `levels` is set, else
        `"neither"`) happens here only when `extend` is `None`.

        Args:
            ticks: Tick positions for the colorbar.

        Returns:
            tuple[Normalize or None, dict]: The norm (None for linear)
                and colorbar keyword arguments.

        Raises:
            ValueError: If `default_options["color_scale"]` is not a
                recognised `cleopatra.styles.ColorScale` value.

        Examples:
            - Linear colour scale with no levels gives `norm=None`
                and ticks forwarded straight through:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": 0.0, "vmax": 10.0})
                >>> g = Glyph(default_options=opts)
                >>> norm, cbar_kw = g._create_norm_and_cbar_kw(np.array([0.0, 5.0, 10.0]))
                >>> norm is None
                True
                >>> cbar_kw["extend"]
                'neither'
                >>> [float(t) for t in cbar_kw["ticks"]]
                [0.0, 5.0, 10.0]

                ```
            - With `levels` set and the default linear scale, a
                `BoundaryNorm` is built and `extend` defaults to
                `"both"`:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": 0.0, "vmax": 10.0, "levels": 5})
                >>> g = Glyph(default_options=opts)
                >>> norm, cbar_kw = g._create_norm_and_cbar_kw(np.array([0.0, 5.0, 10.0]))
                >>> norm is None
                False
                >>> cbar_kw["extend"]
                'both'
                >>> [float(t) for t in cbar_kw["ticks"]]
                [0.0, 2.5, 5.0, 7.5, 10.0]

                ```
        """
        raw_scale = self.default_options["color_scale"]
        try:
            color_scale = ColorScale(raw_scale)
        except ValueError as e:
            valid = ", ".join(repr(m.value) for m in ColorScale)
            raise ValueError(
                f"Invalid color_scale {raw_scale!r}. Expected one of "
                f"{valid} (or a cleopatra.styles.ColorScale member)."
            ) from e
        vmin = ticks[0]
        vmax = ticks[-1]
        levels = self.default_options.get("levels")
        bounds_from_levels = self._levels_to_bounds(levels, vmin, vmax)

        if color_scale == ColorScale.LINEAR:
            if bounds_from_levels is not None:
                norm = colors.BoundaryNorm(boundaries=bounds_from_levels, ncolors=256)
                cbar_kw = {"ticks": bounds_from_levels}
            else:
                norm = None
                cbar_kw = {"ticks": ticks}
        elif color_scale == ColorScale.POWER:
            norm = colors.PowerNorm(
                gamma=self.default_options["gamma"], vmin=vmin, vmax=vmax
            )
            cbar_kw = {"ticks": ticks}
        elif color_scale == ColorScale.SYM_LOGNORM:
            norm = colors.SymLogNorm(
                linthresh=self.default_options["line_threshold"],
                linscale=self.default_options["line_scale"],
                base=np.e,
                vmin=vmin,
                vmax=vmax,
            )
            formatter = LogFormatter(10, labelOnlyBase=False)
            cbar_kw = {"ticks": ticks, "format": formatter}
        elif color_scale == ColorScale.BOUNDARY_NORM:
            explicit_bounds = self.default_options["bounds"]
            if explicit_bounds:
                bounds = explicit_bounds
                cbar_kw = {"ticks": explicit_bounds}
            elif bounds_from_levels is not None:
                bounds = bounds_from_levels
                cbar_kw = {"ticks": bounds_from_levels}
            else:
                bounds = ticks
                cbar_kw = {"ticks": ticks}
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        elif color_scale == ColorScale.MIDPOINT:
            norm = MidpointNormalize(
                midpoint=self.default_options["midpoint"],
                vmin=vmin,
                vmax=vmax,
            )
            cbar_kw = {"ticks": ticks}
        else:  # pragma: no cover - a ColorScale member without a branch
            raise ValueError(
                f"No norm branch implemented for color_scale={color_scale!r}."
            )

        extend = self.default_options.get("extend")
        if extend is None:
            extend_effective = "both" if levels is not None else "neither"
        else:
            extend_effective = extend
        cbar_kw["extend"] = extend_effective

        return norm, cbar_kw

    @staticmethod
    def _levels_to_bounds(
        levels: int | list[float] | np.ndarray | None,
        vmin: float,
        vmax: float,
    ) -> np.ndarray | None:
        """Convert the `levels` option to an array of bin edges.

        Returns `None` when no levels are configured, signalling that
        the caller should fall back to the continuous norm path.

        Args:
            levels: Number of levels (`int`), explicit edges
                (`list` / `ndarray`), or `None` for no
                discretisation.
            vmin: Lower colour limit. Used when `levels` is an int to
                build the linspace.
            vmax: Upper colour limit. Used when `levels` is an int to
                build the linspace.

        Returns:
            np.ndarray or None: Sorted ascending array of bin edges, or
                `None` when `levels` is `None`.

        Raises:
            ValueError: If `levels` is an integer outside the range
                `[2, MAX_DISCRETE_LEVELS]` (a single edge cannot form a
                `BoundaryNorm`, and an enormous count would OOM
                `np.linspace`).

        Examples:
            - Integer `levels` becomes a `linspace` between
                `vmin` and `vmax`:
                ```python
                >>> from cleopatra.glyph import Glyph
                >>> bounds = Glyph._levels_to_bounds(5, 0.0, 10.0)
                >>> [float(b) for b in bounds]
                [0.0, 2.5, 5.0, 7.5, 10.0]

                ```
            - A sequence is sorted ascending and returned as a float
                `ndarray`; `None` short-circuits to `None`:
                ```python
                >>> from cleopatra.glyph import Glyph
                >>> bounds = Glyph._levels_to_bounds([10.0, 0.0, 5.0], 0.0, 10.0)
                >>> [float(b) for b in bounds]
                [0.0, 5.0, 10.0]
                >>> Glyph._levels_to_bounds(None, 0.0, 10.0) is None
                True

                ```
        """
        bounds: np.ndarray | None
        if levels is None:
            bounds = None
        elif isinstance(levels, (int, np.integer)) and not isinstance(levels, bool):
            n = int(levels)
            if not 2 <= n <= MAX_DISCRETE_LEVELS:
                raise ValueError(
                    f"`levels` as an integer must be between 2 and "
                    f"{MAX_DISCRETE_LEVELS}, got {n}."
                )
            bounds = np.linspace(float(vmin), float(vmax), n)
        else:
            bounds = np.sort(np.asarray(levels, dtype=float))
        return bounds

    def _resolve_limits(self, values: np.ndarray) -> tuple[float, float]:
        """Resolve `(vmin, vmax)` from options, falling back to the data range.

        Reads `vmin` / `vmax` from `default_options`; whichever is `None`
        (or absent) is filled from the nan-aware min/max of `values`. This
        mirrors the simple branch of `ArrayGlyph._resolve_color_limits`
        (the `robust` / `center` / `percentile` machinery stays an
        `ArrayGlyph` concern). All-NaN input is detected and rejected here
        rather than surfacing later as an opaque failure inside
        `get_ticks()` or matplotlib.

        Args:
            values: The scalar array that will be colour-mapped. Used to
                supply data-driven limits when `vmin` / `vmax` are unset.

        Returns:
            tuple[float, float]: The resolved `(vmin, vmax)` as floats.

        Raises:
            ValueError: If a limit cannot be resolved to a finite number
                (e.g. `values` is empty or all-NaN and the corresponding
                limit was not pinned explicitly).

        Examples:
            - Auto-resolve both limits from the data:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None})
                >>> g = Glyph(default_options=opts)
                >>> g._resolve_limits(np.array([1.0, 5.0, 9.0]))
                (1.0, 9.0)

                ```
            - An explicit limit is preserved; only the missing one is
                taken from the data:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": 0.0, "vmax": None})
                >>> g = Glyph(default_options=opts)
                >>> g._resolve_limits(np.array([1.0, 5.0, 9.0]))
                (0.0, 9.0)

                ```
            - An all-NaN array with unpinned limits raises `ValueError`:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None})
                >>> g = Glyph(default_options=opts)
                >>> g._resolve_limits(np.array([np.nan, np.nan]))
                Traceback (most recent call last):
                    ...
                ValueError: Cannot determine vmin/vmax: no finite values...

                ```
        """
        vmin = self.default_options.get("vmin")
        vmax = self.default_options.get("vmax")
        if vmin is None or vmax is None:
            # nanmin/nanmax on an all-NaN array return NaN with a
            # RuntimeWarning; compute quietly and validate below so the
            # failure is a clear ValueError rather than a downstream crash.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                data_min = np.nanmin(values)
                data_max = np.nanmax(values)
            vmin = data_min if vmin is None else vmin
            vmax = data_max if vmax is None else vmax
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            raise ValueError(
                "Cannot determine vmin/vmax: no finite values. Pass "
                "explicit vmin/vmax, or filter the array first."
            )
        return float(vmin), float(vmax)

    def _prepare_scalar_mapping(
        self, values: np.ndarray
    ) -> tuple[colors.Normalize | None, dict, np.ndarray]:
        """Build the `(norm, cbar_kw, ticks)` triple shared by coloured glyphs.

        This is the single home for the scalar-mapping contract that every
        colour-by-value glyph needs. It:

        1. resolves `(vmin, vmax)` from `default_options`, falling back to
           the data range via `_resolve_limits`;
        2. derives a sensible `ticks_spacing` of `(vmax - vmin) / 10` when
           the caller left it unset (`None`), guarding flat data so the
           spacing is never zero;
        3. writes `vmin`, `vmax`, and `ticks_spacing` back into
           `default_options` so the existing `get_ticks()` — which reads
           from `default_options` — can see them; and
        4. computes the ticks and forwards them to
           `_create_norm_and_cbar_kw`, honouring `levels` / `color_scale`.

        Subclasses call this instead of re-deriving the contract (which is
        easy to get subtly wrong: `get_ticks()` does not read `self._vmin`,
        and `np.arange(None, None)` raises).

        When the `scheme` option is set, the continuous steps above are
        bypassed: control is handed to `_prepare_classified_mapping`, which
        bins the data into discrete colour classes (a `BoundaryNorm`).
        `scheme="categorical"` bypasses them even earlier — before
        `_resolve_limits`, since a `vmin`/`vmax` range is meaningless for
        nominal values (and would raise for non-numeric ones) — and hands
        off to `_prepare_categorical_mapping` instead. With `scheme` unset
        (the default) the behaviour is unchanged.

        Args:
            values: The scalar array to be colour-mapped (e.g. point
                values, vector magnitudes, per-polygon values).

        Returns:
            tuple[Normalize or None, dict, np.ndarray]: the matplotlib norm
                (`None` for a plain linear scale), the colorbar keyword
                arguments from `_create_norm_and_cbar_kw`, and the computed
                tick positions.

        Raises:
            ValueError: Propagated from `_resolve_limits` when no finite
                limits can be determined.

        Examples:
            - Auto limits resolve from the data and produce a non-`None`
                `ticks_spacing` plus continuous-scale ticks:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None, "ticks_spacing": None})
                >>> g = Glyph(default_options=opts)
                >>> norm, cbar_kw, ticks = g._prepare_scalar_mapping(
                ...     np.array([0.0, 5.0, 10.0])
                ... )
                >>> norm is None
                True
                >>> g.default_options["ticks_spacing"]
                1.0
                >>> [float(t) for t in ticks]
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

                ```
            - Flat data does not produce a zero spacing:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None, "ticks_spacing": None})
                >>> g = Glyph(default_options=opts)
                >>> _ = g._prepare_scalar_mapping(np.array([3.0, 3.0, 3.0]))
                >>> g.default_options["ticks_spacing"]
                1.0

                ```
        """
        self._categorical = None
        if self.default_options.get("scheme") == "categorical":
            return self._prepare_categorical_mapping(values)
        self._vmin, self._vmax = self._resolve_limits(np.asarray(values))
        if self.default_options.get("ticks_spacing") is None:
            # `or 1.0` guards flat data (vmax == vmin -> spacing 0), which
            # would make get_ticks()'s np.arange produce an empty array.
            self.ticks_spacing = (self._vmax - self._vmin) / 10 or 1.0
            self.default_options["ticks_spacing"] = self.ticks_spacing
        self.default_options["vmin"] = self._vmin
        self.default_options["vmax"] = self._vmax
        scheme = self.default_options.get("scheme")
        if scheme is not None:
            # Categorical (classified) colouring short-circuits the
            # continuous `color_scale` / `levels` machinery: the bin edges
            # fully determine the discrete `BoundaryNorm` and its ticks.
            return self._prepare_classified_mapping(values, scheme)
        ticks = self.get_ticks()
        norm, cbar_kw = self._create_norm_and_cbar_kw(ticks)
        return norm, cbar_kw, ticks

    def _warn_scheme_overrides_continuous_options(self) -> None:
        """Warn when a `scheme` is set alongside continuous-only options.

        Shared by `_prepare_classified_mapping` and
        `_prepare_categorical_mapping`: either scheme owns the norm
        entirely, so a `color_scale` other than `"linear"` or an explicit
        `levels` the caller also set is silently ignored rather than
        applied -- this warns so that conflicting configuration is visible
        instead of quietly doing nothing.
        """
        if self.default_options.get("color_scale", "linear") != "linear":
            warnings.warn(
                "`scheme` is set, so `color_scale="
                f"{self.default_options['color_scale']!r}` is ignored "
                "(classification builds its own discrete norm).",
                stacklevel=5,
            )
        if self.default_options.get("levels") is not None:
            warnings.warn(
                "`scheme` is set, so `levels` is ignored (the classification "
                "scheme determines the bins).",
                stacklevel=5,
            )

    def _prepare_classified_mapping(
        self, values: np.ndarray, scheme: str | list | np.ndarray
    ) -> tuple[colors.BoundaryNorm, dict, np.ndarray]:
        """Build the `(norm, cbar_kw, ticks)` triple for classified colouring.

        The discrete sibling of the continuous branch in
        `_prepare_scalar_mapping`. When the `scheme` option is set, the
        data is binned into classes by `cleopatra.styles.classify` (using
        the `k` option for the count/width schemes), and the resulting bin
        edges drive a `matplotlib.colors.BoundaryNorm` plus a colorbar
        whose ticks sit on the class boundaries — so `create_color_bar`
        renders a stepped colorbar. The `color_scale` / `levels` options
        are intentionally bypassed here; classification owns the norm.

        Args:
            values: The scalar array to classify and colour-map.
            scheme: A scheme name accepted by `classify` (e.g.
                `"quantiles"`, `"equal_interval"`) or an explicit sequence
                of bin edges.

        Returns:
            tuple[BoundaryNorm, dict, np.ndarray]: the discrete norm, the
                colorbar keyword arguments (boundary `ticks` plus
                `extend`), and the bin edges (returned in the `ticks`
                slot of the shared contract).

        Raises:
            ValueError: Propagated from `classify` (unknown scheme,
                degenerate data, or `k < 1`).

        Examples:
            - A quantile scheme yields a `BoundaryNorm` and boundary ticks:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update(
                ...     {"vmin": None, "vmax": None, "scheme": "quantiles", "k": 4}
                ... )
                >>> g = Glyph(default_options=opts)
                >>> norm, cbar_kw, edges = g._prepare_classified_mapping(
                ...     np.arange(100.0), "quantiles"
                ... )
                >>> [float(b) for b in norm.boundaries]
                [0.0, 24.75, 49.5, 74.25, 99.0]
                >>> [float(t) for t in cbar_kw["ticks"]]
                [0.0, 24.75, 49.5, 74.25, 99.0]
                >>> cbar_kw["extend"]
                'neither'

                ```
        """
        self._warn_scheme_overrides_continuous_options()
        k = self.default_options.get("k", 5)
        bin_edges, norm = classify(values, scheme, k)
        extend = self.default_options.get("extend")
        cbar_kw = {
            "ticks": bin_edges,
            "extend": "neither" if extend is None else extend,
        }
        return norm, cbar_kw, bin_edges

    def _prepare_categorical_mapping(
        self, values: np.ndarray
    ) -> tuple[colors.BoundaryNorm, dict, np.ndarray]:
        """Build the `(norm, cbar_kw, edges)` triple for `scheme="categorical"`.

        The nominal sibling of `_prepare_classified_mapping`: instead of
        binning a continuous range, `cleopatra.styles.categorize` assigns
        one colour per distinct value in `values` (sorted when sortable),
        and this builds a `ListedColormap` + `BoundaryNorm` over the
        resulting integer class codes — the same construction
        `colors.apply_data_style` uses for a preset's `categories`, but with
        the category table auto-derived from the data instead of
        hand-authored. The mapping (per-element codes, the `ListedColormap`,
        and the colour/label pairs) is stashed on `self._categorical` for
        the calling glyph to read back, since — unlike the continuous and
        classified paths — the array fed to the mappable is these integer
        codes, not `values` itself (which may not even be numeric).

        Only glyphs with `_SUPPORTS_CATEGORICAL_SCHEME = True` may use this
        scheme: for any other glyph, `values` are a continuous magnitude
        (e.g. vector length), where "one colour per distinct float" is
        almost never what the caller wants, and the glyph's `plot()` does
        not know to read `self._categorical` back in the first place — it
        would keep feeding the raw (mismatched) values to the mappable.

        The glyph's `cmap` option drives `categorize`'s palette, with one
        override: if `cmap` is still at the shared continuous/diverging
        default (`"coolwarm_r"`, matched by resolved name so a `Colormap`
        instance equivalent to the default is caught too, not just the bare
        string) — i.e. the caller never overrode it — it is substituted with
        `CATEGORICAL_DEFAULT_CMAP` (`"tab10"`) instead, since sampling a
        diverging gradient at N points would defeat the point of "one
        distinct colour per class". Any other `cmap`, qualitative or not,
        is always honoured as given.

        Args:
            values: The per-element nominal values to categorize.

        Returns:
            tuple[BoundaryNorm, dict, np.ndarray]: the discrete norm over
                the integer class codes, an empty colorbar-kwargs dict (a
                categorical scheme draws a `disjoint_legend`, never a
                colorbar — see `create_categorical_legend`), and the code
                boundary edges (`-0.5 .. n_categories - 0.5`).

        Raises:
            ValueError: If this glyph does not support `scheme="categorical"`,
                or (propagated from `categorize`) if `values` has no
                non-null entries.

        Examples:
            - Three distinct values map to three integer codes and colours:
                ```python
                >>> import numpy as np
                >>> from cleopatra.polygon_glyph import PolygonGlyph
                >>> polys = [np.zeros((3, 2))] * 3
                >>> g = PolygonGlyph(polys, values=np.array(["a", "b", "a"]))
                >>> norm, cbar_kw, edges = g._prepare_categorical_mapping(
                ...     np.array(["a", "b", "a"])
                ... )
                >>> [float(b) for b in edges]
                [-0.5, 0.5, 1.5]
                >>> [float(c) for c in g._categorical["codes"]]
                [0.0, 1.0, 0.0]

                ```
        """
        if not self._SUPPORTS_CATEGORICAL_SCHEME:
            raise ValueError(
                f"{type(self).__name__} does not support scheme='categorical' "
                "(its values are a continuous magnitude, not nominal class "
                "labels)."
            )
        self._warn_scheme_overrides_continuous_options()
        cmap = self.default_options["cmap"]
        # Compare by resolved name, not raw `==`: `Colormap` does not
        # implement equality, so a `Colormap` *instance* equivalent to the
        # default (e.g. `mpl.colormaps["coolwarm_r"]`, a legitimate way to
        # pass a colormap) would otherwise never match the string default
        # and silently bypass the fallback below.
        cmap_name = cmap if isinstance(cmap, str) else getattr(cmap, "name", None)
        if cmap_name == STYLE_DEFAULTS["cmap"]:
            # `cmap` is still at the shared continuous/diverging default
            # ("coolwarm_r") -- nobody chose it *for* a categorical mapping,
            # they just never overrode it. Sampling a diverging gradient at
            # N points defeats the point of "one distinct colour per class",
            # so fall back to `categorize`'s own qualitative default instead.
            # An explicit non-default `cmap` (qualitative or not) is always
            # honoured as given.
            cmap = CATEGORICAL_DEFAULT_CMAP
        # Flatten `values` once and reuse it for both `categorize` (which
        # would otherwise redo the identical `np.asarray(...).ravel()` on
        # the original, possibly nested/2-D `values`) and the codes lookup
        # below -- passing an already-flat list back through `categorize`
        # is a cheap no-op re-wrap, not a second real flattening pass.
        raw = np.asarray(values, dtype=object).ravel().tolist()
        categories, palette = categorize(raw, cmap=cmap)
        lookup = {category: i for i, category in enumerate(categories.tolist())}
        codes = np.array([lookup.get(v, np.nan) for v in raw], dtype=float)
        listed_cmap = colors.ListedColormap(palette)
        edges = np.arange(len(categories) + 1) - 0.5
        norm = colors.BoundaryNorm(edges, len(palette))
        self._categorical = {
            "codes": codes,
            "cmap": listed_cmap,
            "colors": palette,
            "labels": [str(c) for c in categories.tolist()],
        }
        return norm, {}, edges

    def create_categorical_legend(self, ax: Axes) -> Legend:
        """Attach the disjoint legend for a `scheme="categorical"` mapping.

        Reads the category colours/labels `_prepare_categorical_mapping`
        stashed on `self._categorical` and draws them via
        `cleopatra.styles.disjoint_legend` — the discrete counterpart to
        `create_color_bar`, used instead of it whenever `scheme` is
        `"categorical"` (a colorbar would imply a false ordering over
        nominal classes). The legend's title defaults to the `cbar_label`
        option (the same label a continuous plot would put on its
        colorbar); the `category_legend_kwargs` option is merged over that
        default and forwarded to `disjoint_legend` (e.g. `loc`, `ncol`,
        `bbox_to_anchor`, or an explicit `title` override) — the categorical
        counterpart to `size_legend_kwargs`.

        Args:
            ax: The axes to attach the legend to.

        Returns:
            Legend: The created legend artist, already added to `ax`.

        Raises:
            ValueError: If `self._categorical` has not been populated yet
                (i.e. `_prepare_categorical_mapping` has not run for this
                glyph instance).

        Examples:
            - Prepare a categorical mapping, then draw and inspect the legend:
                ```python
                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.polygon_glyph import PolygonGlyph
                >>> polys = [np.zeros((3, 2))] * 3
                >>> g = PolygonGlyph(polys, values=np.array(["a", "b", "a"]))
                >>> _ = g._prepare_categorical_mapping(np.array(["a", "b", "a"]))
                >>> fig, ax = plt.subplots()
                >>> legend = g.create_categorical_legend(ax)
                >>> [t.get_text() for t in legend.get_texts()]
                ['a', 'b']
                >>> plt.close(fig)

                ```
            - Calling it before a categorical mapping exists raises `ValueError`:
                ```python
                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.polygon_glyph import PolygonGlyph
                >>> g = PolygonGlyph([np.zeros((3, 2))] * 2, values=np.array(["a", "b"]))
                >>> fig, ax = plt.subplots()
                >>> g.create_categorical_legend(ax)
                Traceback (most recent call last):
                    ...
                ValueError: create_categorical_legend() called before a scheme='categorical' mapping was prepared -- call _prepare_scalar_mapping (or plot()) first.
                >>> plt.close(fig)

                ```
            - `category_legend_kwargs` overrides the default title and adds
                a `loc`:
                ```python
                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.polygon_glyph import PolygonGlyph
                >>> polys = [np.zeros((3, 2))] * 2
                >>> g = PolygonGlyph(
                ...     polys, values=np.array(["a", "b"]),
                ...     category_legend_kwargs={"title": "Class", "loc": "upper left"},
                ... )
                >>> _ = g._prepare_categorical_mapping(np.array(["a", "b"]))
                >>> fig, ax = plt.subplots()
                >>> legend = g.create_categorical_legend(ax)
                >>> legend.get_title().get_text()
                'Class'
                >>> plt.close(fig)

                ```
        """
        categorical = self._categorical
        if categorical is None:
            raise ValueError(
                "create_categorical_legend() called before a "
                "scheme='categorical' mapping was prepared -- call "
                "_prepare_scalar_mapping (or plot()) first."
            )
        legend_kwargs = {
            "title": self.default_options.get("cbar_label"),
            **(self.default_options.get("category_legend_kwargs") or {}),
        }
        return disjoint_legend(
            ax,
            categorical["colors"],
            categorical["labels"],
            **legend_kwargs,
        )

    def create_color_bar(self, ax: Axes, im: Any, cbar_kw: dict) -> Colorbar:
        """Create a colorbar with full customization from default_options.

        Reads `cbar_length`, `cbar_orientation`, `cbar_label`,
        `cbar_label_size`, and `cbar_label_location` from
        `default_options` to configure the colorbar. When the optional
        `cbar_kwargs` entry is present in `default_options` (an
        xarray-aligned dict-of-overrides), its keys are merged over the
        defaults so the user wins on any collision (e.g. `label`,
        `shrink`, `orientation`, `ticks`, `extend`).

        `cbar_kwargs` is read from `self.default_options["cbar_kwargs"]`.
        Set it via the constructor or `plot` kwargs of the calling
        glyph subclass. Keys recognised by `matplotlib.pyplot.colorbar`
        — `label`, `shrink`, `aspect`, `orientation`, `pad`,
        `ticks`, `extend` — are forwarded; `label` is special-cased
        so that label-size and label-location styling from
        `default_options` are still applied.

        Args:
            ax: Matplotlib axes.
            im: The mappable (image or contour) to attach the
                colorbar to.
            cbar_kw: Colorbar keyword arguments (ticks, format,
                extend, etc.) computed by
                `_create_norm_and_cbar_kw`.

        Returns:
            Colorbar: The created colorbar.

        Raises:
            TypeError: If `default_options["cbar_kwargs"]` is set
                but is not a `dict`.

        Examples:
            - Create a colorbar with a custom label:
                ```python
                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None})
                >>> g = Glyph(default_options=opts, cbar_label="Depth [m]")
                >>> fig, ax = plt.subplots()
                >>> im = ax.imshow(np.arange(9).reshape(3, 3))
                >>> cbar = g.create_color_bar(ax, im, {"ticks": [0, 4, 8]})
                >>> cbar.orientation
                'vertical'

                ```
            - User-supplied `cbar_kwargs` win on collision and
                `label` is applied via `set_label`:
                ```python
                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({
                ...     "vmin": None,
                ...     "vmax": None,
                ...     "cbar_kwargs": {"label": "User Label", "orientation": "horizontal"},
                ... })
                >>> g = Glyph(default_options=opts, cbar_label="Default Label")
                >>> fig, ax = plt.subplots()
                >>> im = ax.imshow(np.arange(9).reshape(3, 3))
                >>> cbar = g.create_color_bar(ax, im, {"ticks": [0, 4, 8]})
                >>> cbar.orientation
                'horizontal'
                >>> cbar.ax.get_xlabel() or cbar.ax.get_ylabel()
                'User Label'

                ```
            - Non-dict `cbar_kwargs` raises `TypeError`:
                ```python
                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None, "cbar_kwargs": "oops"})
                >>> g = Glyph(default_options=opts)
                >>> fig, ax = plt.subplots()
                >>> im = ax.imshow(np.arange(9).reshape(3, 3))
                >>> g.create_color_bar(ax, im, {"ticks": [0, 4, 8]})
                Traceback (most recent call last):
                    ...
                TypeError: cbar_kwargs must be a dict of colorbar keyword arguments, got str.

                ```
        """
        fig = ax.figure
        is_subplot = len(fig.axes) > 1
        merged_kw = {
            "shrink": self.default_options["cbar_length"],
            "orientation": self.default_options["cbar_orientation"],
            "use_gridspec": not is_subplot,
        }
        merged_kw.update(cbar_kw)
        # Pull the user-supplied `label` (if any) out of cbar_kwargs
        # before forwarding to `fig.colorbar` so we can apply it via
        # `cbar.set_label` and preserve label-size/location styling.
        user_kwargs = self.default_options.get("cbar_kwargs") or {}
        if not isinstance(user_kwargs, dict):
            raise TypeError(
                "cbar_kwargs must be a dict of colorbar keyword "
                f"arguments, got {type(user_kwargs).__name__}."
            )
        user_kwargs = dict(user_kwargs)
        user_label = user_kwargs.pop("label", None)
        merged_kw.update(user_kwargs)
        cbar = fig.colorbar(im, ax=ax, **merged_kw)
        cbar.ax.tick_params(labelsize=10)
        label_text = (
            user_label if user_label is not None else self.default_options["cbar_label"]
        )
        cbar.set_label(
            label_text,
            fontsize=self.default_options["cbar_label_size"],
            loc=self.default_options["cbar_label_location"],
        )
        return cbar

    def adjust_ticks(
        self,
        axis: str,
        multiply_value: float | int = 1,
        add_value: float | int = 0,
        fmt: str = "{0:g}",
        visible: bool = True,
    ) -> None:
        """Adjust the axis tick labels with a linear transformation.

        Applies `tick_value * multiply_value + add_value` to each
        tick, formatted with `fmt`. Useful for converting pixel
        coordinates to real-world units.

        Args:
            axis: `"x"` or `"y"`.
            multiply_value: Multiplier for tick values. Default is 1.
            add_value: Offset added to tick values. Default is 0.
            fmt: Format string for tick labels.
                Default is `"{0:g}"`.
            visible: Whether the axis is visible. Default is True.

        Examples:
            - Scale x-axis ticks by 100 and offset by 5:
                ```python
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None})
                >>> g = Glyph(default_options=opts)
                >>> fig, ax = plt.subplots()
                >>> _ = ax.plot([0, 1, 2], [0, 1, 2])
                >>> g.fig, g.ax = fig, ax
                >>> g.adjust_ticks(axis="x", multiply_value=100, add_value=5)

                ```
        """
        if axis == "x":
            ticks_fn = ticker.FuncFormatter(
                lambda x, pos: fmt.format(x * multiply_value + add_value)
            )
            self.ax.xaxis.set_major_formatter(ticks_fn)
        else:
            ticks_fn = ticker.FuncFormatter(
                lambda y, pos: fmt.format(y * multiply_value + add_value)
            )
            self.ax.yaxis.set_major_formatter(ticks_fn)

        if not visible:
            if axis == "x":
                self.ax.get_xaxis().set_visible(visible)
            else:
                self.ax.get_yaxis().set_visible(visible)

    @staticmethod
    def _plot_point_values(ax, point_table: np.ndarray, pid_color, pid_size):
        """Plot point value labels on the axes."""
        write_points = lambda x: ax.text(
            x[2],
            x[1],
            x[0],
            ha="center",
            va="center",
            color=pid_color,
            fontsize=pid_size,
        )
        return list(map(write_points, point_table))

    def save_animation(self, path: str | os.PathLike, fps: int = 2, **kwargs) -> None:
        """Save this glyph's animation (`self.anim`) to a file.

        Thin wrapper around `cleopatra.animation.save_animation`; the output
        format is determined by the file extension. GIF and WebP use an
        optimising Pillow writer; mov/avi/mp4 use FFmpeg (a system binary if
        present, otherwise the one bundled with imageio-ffmpeg).

        Args:
            path: Output file path, as a `str` or `os.PathLike` (e.g. a
                `pathlib.Path`). Extension determines format.
                Supported: gif, mov, avi, mp4, webp.
            fps: Frames per second. Default is 2.
            **kwargs: Additional keyword arguments forwarded to
                `cleopatra.animation.save_animation`, e.g. ``crf``, ``bitrate``,
                ``codec``, ``preset``, ``pix_fmt``, ``dpi`` (ffmpeg formats) or
                ``optimize`` and ``loop`` (GIF).

        Raises:
            ValueError: If `animate()` has not been called yet, if the file
                format is not supported, or if both ``crf`` and ``bitrate``
                are given.
            FileNotFoundError: If a video format is requested but neither a
                system FFmpeg nor imageio-ffmpeg's bundled binary is found.

        Examples:
            - Check the supported video formats:
                ```python
                >>> from cleopatra.glyph import SUPPORTED_VIDEO_FORMAT
                >>> sorted(SUPPORTED_VIDEO_FORMAT)
                ['avi', 'gif', 'mov', 'mp4', 'webp']

                ```
        """
        _save_animation(self.anim, path, fps=fps, **kwargs)
