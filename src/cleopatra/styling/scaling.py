"""Colour-scaling parameter object.

`ColorScaling` groups the six loose colour-scale options every colormap
glyph used to accept as flat keyword arguments (`color_scale`, `gamma`,
`line_threshold`, `line_scale`, `bounds`, `midpoint`) into a single,
discoverable object, and owns the logic that turns them into a matplotlib
norm plus colorbar keyword arguments.

The flat options are mutually exclusive by scale kind -- `gamma` only
applies to `power`, `line_threshold`/`line_scale` only to `sym-lognorm`,
`bounds` only to `boundary-norm`, `midpoint` only to `midpoint`. The
variant constructors (`ColorScaling.power`, `.sym_log`, `.log`,
`.boundary`, `.midpoint`, `.linear`) expose only the knobs each scale
actually uses,
so an invalid combination cannot be expressed.

Examples:
    - A power scale exposes only `gamma`:
        ```python
        >>> from cleopatra.styling.scaling import ColorScaling
        >>> scale = ColorScaling.power(gamma=0.7)
        >>> scale.kind.value
        'power'
        >>> scale.gamma
        0.7

        ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.colors as colors
import numpy as np
from matplotlib.ticker import LogFormatter

from cleopatra.styling.colors import build_log_norm
from cleopatra.styling.styles import ColorScale, MidpointNormalize

#: Upper bound on an integer `levels` count. A single edge cannot form a
#: `BoundaryNorm`, and an enormous count would OOM `np.linspace`.
MAX_DISCRETE_LEVELS = 1000

#: Defaults for the colour-scale options, matching
#: `cleopatra.styling.styles.DEFAULT_OPTIONS`. Kept here so
#: `ColorScaling.from_options` can fill a missing key rather than raising.
_SCALE_DEFAULTS: dict[str, Any] = {
    "color_scale": "linear",
    "gamma": 0.5,
    "line_threshold": 0.0001,
    "line_scale": 0.001,
    "bounds": None,
    "midpoint": 0,
}


def levels_to_bounds(
    levels: int | list[float] | np.ndarray | None,
    vmin: float,
    vmax: float,
) -> np.ndarray | None:
    """Convert a `levels` option to an array of bin edges.

    Returns `None` when no levels are configured, signalling that the
    caller should fall back to the continuous norm path.

    Args:
        levels: Number of levels (`int`), explicit edges (`list` /
            `ndarray`), or `None` for no discretisation.
        vmin: Lower colour limit. Used when `levels` is an int to build
            the linspace.
        vmax: Upper colour limit. Used when `levels` is an int to build
            the linspace.

    Returns:
        np.ndarray or None: Sorted ascending array of bin edges, or `None`
            when `levels` is `None`.

    Raises:
        ValueError: If `levels` is an integer outside the range
            `[2, MAX_DISCRETE_LEVELS]`.

    Examples:
        - Integer `levels` becomes a `linspace` between `vmin` and `vmax`:
            ```python
            >>> from cleopatra.styling.scaling import levels_to_bounds
            >>> [float(b) for b in levels_to_bounds(5, 0.0, 10.0)]
            [0.0, 2.5, 5.0, 7.5, 10.0]

            ```
        - A sequence is sorted ascending; `None` short-circuits to `None`:
            ```python
            >>> from cleopatra.styling.scaling import levels_to_bounds
            >>> [float(b) for b in levels_to_bounds([10.0, 0.0, 5.0], 0.0, 10.0)]
            [0.0, 5.0, 10.0]
            >>> levels_to_bounds(None, 0.0, 10.0) is None
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


@dataclass(frozen=True)
class ColorScaling:
    """The colour-scale group: a scale kind plus its scale-specific knobs.

    Prefer the variant constructors (`linear`, `power`, `sym_log`, `log`,
    `boundary`, `midpoint`) over the raw dataclass -- each exposes only
    the fields its scale uses, so nonsensical combinations (e.g. a
    `midpoint` on a `linear` scale) cannot be built.

    Attributes:
        kind: The scale kind (`cleopatra.styling.styles.ColorScale`).
        gamma: Exponent for the `power` scale. Ignored by other kinds.
        line_threshold: Linear-region threshold for `sym-lognorm`.
        line_scale: Linear-region scale factor for `sym-lognorm`.
        bounds: Explicit bin edges for `boundary-norm`.
        center: Centre value for the `midpoint` scale (the value pinned to
            the colormap centre). Named `center` rather than `midpoint` so
            the field does not shadow the `midpoint()` variant constructor.
    """

    kind: ColorScale = ColorScale.LINEAR
    gamma: float = 0.5
    line_threshold: float = 0.0001
    line_scale: float = 0.001
    bounds: list[float] | None = None
    center: float = 0

    @classmethod
    def linear(cls) -> ColorScaling:
        """A plain linear colour scale (matplotlib's default norm).

        Examples:
            - The linear scale carries no extra knobs:
                ```python
                >>> from cleopatra.styling.scaling import ColorScaling
                >>> ColorScaling.linear().kind.value
                'linear'

                ```
        """
        return cls(kind=ColorScale.LINEAR)

    @classmethod
    def power(cls, gamma: float = 0.5) -> ColorScaling:
        """A power-law (`PowerNorm`) colour scale.

        Args:
            gamma: The power exponent. Defaults to `0.5`.

        Examples:
            - Only `gamma` is exposed:
                ```python
                >>> from cleopatra.styling.scaling import ColorScaling
                >>> ColorScaling.power(gamma=2.0).gamma
                2.0

                ```
        """
        return cls(kind=ColorScale.POWER, gamma=gamma)

    @classmethod
    def sym_log(cls, threshold: float = 0.0001, scale: float = 0.001) -> ColorScaling:
        """A symmetric-log (`SymLogNorm`) colour scale.

        Args:
            threshold: The linear-region half-width (`linthresh`).
                Defaults to `0.0001`.
            scale: The linear-region scale factor (`linscale`). Defaults
                to `0.001`.

        Examples:
            - Exposes the two `sym-lognorm` knobs:
                ```python
                >>> from cleopatra.styling.scaling import ColorScaling
                >>> s = ColorScaling.sym_log(threshold=0.01, scale=0.1)
                >>> (s.line_threshold, s.line_scale)
                (0.01, 0.1)

                ```
        """
        return cls(kind=ColorScale.SYM_LOGNORM, line_threshold=threshold, line_scale=scale)

    @classmethod
    def log(cls) -> ColorScaling:
        """A logarithmic (`LogNorm`) colour scale for strictly-positive data.

        The plain-log counterpart of `sym_log`: `LogNorm` needs a positive
        value range, so for data that spans zero or negative values use
        `sym_log` (a symmetric-log scale) instead. Like `linear`, it carries
        no extra knobs -- `vmin`/`vmax` come from the tick range at render
        time.

        Examples:
            - The log scale exposes no extra knobs:
                ```python
                >>> from cleopatra.styling.scaling import ColorScaling
                >>> ColorScaling.log().kind.value
                'lognorm'

                ```
        """
        return cls(kind=ColorScale.LOGNORM)

    @classmethod
    def boundary(cls, bounds: list[float] | None = None) -> ColorScaling:
        """A discrete (`BoundaryNorm`) colour scale.

        Args:
            bounds: Explicit bin edges. When `None`, the edges are derived
                from `levels` (if set) or the tick positions at render
                time.

        Examples:
            - Explicit edges are carried through:
                ```python
                >>> from cleopatra.styling.scaling import ColorScaling
                >>> ColorScaling.boundary([0, 1, 5, 10]).bounds
                [0, 1, 5, 10]

                ```
        """
        return cls(kind=ColorScale.BOUNDARY_NORM, bounds=bounds)

    @classmethod
    def midpoint(cls, at: float = 0) -> ColorScaling:
        """A midpoint-anchored diverging colour scale.

        Args:
            at: The value pinned to the colormap centre. Defaults to `0`.

        Examples:
            - Anchor the colormap centre at a chosen value:
                ```python
                >>> from cleopatra.styling.scaling import ColorScaling
                >>> ColorScaling.midpoint(at=100).center
                100

                ```
        """
        return cls(kind=ColorScale.MIDPOINT, center=at)

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> ColorScaling:
        """Build a `ColorScaling` from a flat `default_options` dict.

        The bridge between the legacy flat-key storage every glyph still
        uses internally and this object's behaviour. Reads the six
        colour-scale keys, validating `color_scale` with the same
        actionable error the flat path raised.

        Args:
            options: A glyph's `default_options` (or any mapping carrying
                the colour-scale keys).

        Returns:
            ColorScaling: The reconstructed scale object.

        Raises:
            ValueError: If `options["color_scale"]` is not a recognised
                `cleopatra.styling.styles.ColorScale` value.

        Examples:
            - Round-trips the flat keys back into an object:
                ```python
                >>> from cleopatra.styling.scaling import ColorScaling
                >>> ColorScaling.from_options({"color_scale": "power", "gamma": 0.7}).gamma
                0.7

                ```
        """
        raw_scale = options.get("color_scale", _SCALE_DEFAULTS["color_scale"])
        try:
            kind = ColorScale(raw_scale)
        except ValueError as e:
            valid = ", ".join(repr(m.value) for m in ColorScale)
            raise ValueError(
                f"Invalid color_scale {raw_scale!r}. Expected one of "
                f"{valid} (or a cleopatra.styling.styles.ColorScale member)."
            ) from e
        return cls(
            kind=kind,
            gamma=options.get("gamma", _SCALE_DEFAULTS["gamma"]),
            line_threshold=options.get("line_threshold", _SCALE_DEFAULTS["line_threshold"]),
            line_scale=options.get("line_scale", _SCALE_DEFAULTS["line_scale"]),
            bounds=options.get("bounds", _SCALE_DEFAULTS["bounds"]),
            center=options.get("midpoint", _SCALE_DEFAULTS["midpoint"]),
        )

    def to_options(self) -> dict[str, Any]:
        """Flatten back to the `default_options` keys the engine reads.

        Returns:
            dict: The six colour-scale keys, with `color_scale` as the
                plain string value.

        Examples:
            - Emits the flat keys a glyph merges into `default_options`:
                ```python
                >>> from cleopatra.styling.scaling import ColorScaling
                >>> ColorScaling.power(gamma=0.7).to_options()["color_scale"]
                'power'

                ```
        """
        return {
            "color_scale": self.kind.value,
            "gamma": self.gamma,
            "line_threshold": self.line_threshold,
            "line_scale": self.line_scale,
            "bounds": self.bounds,
            "midpoint": self.center,
        }

    def build_norm(
        self,
        ticks: np.ndarray,
        levels: int | list[float] | np.ndarray | None = None,
        extend: str | None = None,
    ) -> tuple[colors.Normalize | None, dict[str, Any]]:
        """Build the matplotlib norm and colorbar keyword arguments.

        The colour-scale logic that used to live in
        `Glyph._create_norm_and_cbar_kw`. `vmin`/`vmax` are read from the
        first and last tick; `levels` and `extend` are cross-group inputs
        (contour discretisation and colorbar arrow extension) passed in by
        the caller.

        Args:
            ticks: Tick positions for the colorbar; `ticks[0]`/`ticks[-1]`
                supply `vmin`/`vmax`.
            levels: Optional discretisation for the `linear`/`boundary`
                kinds (int count or explicit edges).
            extend: Colorbar arrow extension. When `None`, auto-resolves to
                `"both"` if `levels` is set, else `"neither"`.

        Returns:
            tuple[Normalize or None, dict]: The norm (`None` for a plain
                linear scale) and the colorbar keyword arguments.

        Examples:
            - A linear scale with no levels yields no norm and passes the
                ticks straight through:
                ```python
                >>> import numpy as np
                >>> from cleopatra.styling.scaling import ColorScaling
                >>> norm, cbar_kw = ColorScaling.linear().build_norm(
                ...     np.array([0.0, 5.0, 10.0])
                ... )
                >>> norm is None
                True
                >>> cbar_kw["extend"]
                'neither'

                ```
            - `levels` on the linear scale builds a `BoundaryNorm` and
                defaults `extend` to `"both"`:
                ```python
                >>> import numpy as np
                >>> from cleopatra.styling.scaling import ColorScaling
                >>> norm, cbar_kw = ColorScaling.linear().build_norm(
                ...     np.array([0.0, 5.0, 10.0]), levels=5
                ... )
                >>> norm is None
                False
                >>> cbar_kw["extend"]
                'both'

                ```
        """
        vmin = ticks[0]
        vmax = ticks[-1]
        bounds_from_levels = levels_to_bounds(levels, vmin, vmax)

        norm: colors.Normalize | None
        cbar_kw: dict[str, Any]
        if self.kind == ColorScale.LINEAR:
            norm, cbar_kw = self._linear_norm(ticks, bounds_from_levels)
        elif self.kind == ColorScale.POWER:
            norm = colors.PowerNorm(gamma=self.gamma, vmin=vmin, vmax=vmax)
            cbar_kw = {"ticks": ticks}
        elif self.kind == ColorScale.SYM_LOGNORM:
            norm = colors.SymLogNorm(
                linthresh=self.line_threshold,
                linscale=self.line_scale,
                base=np.e,
                vmin=vmin,
                vmax=vmax,
            )
            cbar_kw = {"ticks": ticks, "format": LogFormatter(10, labelOnlyBase=False)}
        elif self.kind == ColorScale.LOGNORM:
            lo, hi = float(vmin), float(vmax)
            # A constant *positive* field yields a single tick (vmin == vmax); a
            # log scale cannot span a zero-width range, so widen it -- matching
            # the data-style norm='log' path, which bumps vmax = vmin + 1.0. Only
            # widen a positive constant: a non-positive one must raise, and its
            # error should report the real bound, not a widened one.
            if hi == lo and lo > 0.0:
                hi = lo + 1.0
            norm = build_log_norm(
                lo, hi, context="ColorScaling.log()", remedy="use ColorScaling.sym_log()"
            )
            cbar_kw = {"ticks": ticks, "format": LogFormatter(10, labelOnlyBase=False)}
        elif self.kind == ColorScale.BOUNDARY_NORM:
            norm, cbar_kw = self._boundary_norm(ticks, bounds_from_levels)
        elif self.kind == ColorScale.MIDPOINT:
            norm = MidpointNormalize(midpoint=self.center, vmin=vmin, vmax=vmax)
            cbar_kw = {"ticks": ticks}
        else:  # pragma: no cover - a ColorScale member without a branch
            raise ValueError(
                f"No norm branch implemented for color_scale={self.kind!r}."
            )

        if extend is None:
            extend = "both" if levels is not None else "neither"
        cbar_kw["extend"] = extend
        return norm, cbar_kw

    def _linear_norm(
        self, ticks: np.ndarray, bounds_from_levels: np.ndarray | None
    ) -> tuple[colors.Normalize | None, dict[str, Any]]:
        """Linear-scale norm: a `BoundaryNorm` when `levels` are given, else no norm."""
        if bounds_from_levels is not None:
            norm = colors.BoundaryNorm(boundaries=bounds_from_levels, ncolors=256)
            return norm, {"ticks": bounds_from_levels}
        return None, {"ticks": ticks}

    def _boundary_norm(
        self, ticks: np.ndarray, bounds_from_levels: np.ndarray | None
    ) -> tuple[colors.Normalize, dict[str, Any]]:
        """Explicit-bounds norm: own `bounds` win, then `levels`, then the ticks."""
        if self.bounds:
            bounds = self.bounds
        elif bounds_from_levels is not None:
            bounds = bounds_from_levels
        else:
            bounds = ticks
        return colors.BoundaryNorm(boundaries=bounds, ncolors=256), {"ticks": bounds}
