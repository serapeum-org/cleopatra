"""Colour-scaling parameter object.

`ColorScaling` groups the six loose colour-scale options every colormap
glyph used to accept as flat keyword arguments (`color_scale`, `gamma`,
`line_threshold`, `line_scale`, `bounds`, `midpoint`) into a single,
discoverable object, and owns the logic that turns them into a matplotlib
norm plus colorbar keyword arguments.

The flat options are mutually exclusive by scale kind -- `gamma` only
applies to `power`, `line_threshold`/`line_scale` only to `sym-lognorm`,
`bounds` only to `boundary-norm`, `midpoint` only to `midpoint`, `samples`
only to `equalize` (the continuous rank-equalising scale). The variant
constructors (`ColorScaling.power`, `.sym_log`, `.log`, `.boundary`,
`.midpoint`, `.equalize`, `.linear`) expose only the knobs each scale
actually uses, so an invalid combination cannot be expressed.

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
from matplotlib.ticker import (
    FuncFormatter,
    Locator,
    LogLocator,
    SymmetricalLogLocator,
)

from cleopatra.styling.colors import build_log_norm
from cleopatra.styling.styles import ColorScale, MidpointNormalize

#: Upper bound on an integer `levels` count. A single edge cannot form a
#: `BoundaryNorm`, and an enormous count would OOM `np.linspace`.
MAX_DISCRETE_LEVELS = 1000


def _format_tick_value(value: float, _pos: int | None = None) -> str:
    """Label a colorbar tick with its plain numeric value, sign-correct.

    `format(v, "g")` keeps whole numbers compact (`100.0` -> `"100"`) and negatives
    signed. `value + 0.0` normalises a signed zero so `-0.0` renders as `"0"`, not
    `"-0"`. Note `"g"` switches to scientific notation and ~6 significant figures
    for very large or very small magnitudes (`1e6` -> `"1e+06"`); the default log
    ticks are clean powers of ten, so this only shows for a caller's own large
    `set_ticks` value.

    Args:
        value: The tick value to label.
        _pos: The tick index matplotlib passes; unused.

    Returns:
        str: The formatted label.
    """
    return f"{value + 0.0:g}"


def _plain_tick_formatter() -> FuncFormatter:
    """A colorbar formatter that labels every tick with its plain numeric value.

    Used for the non-linear scales (`sym_log`, `log`) so their bars are readable.
    Unlike matplotlib's `LogFormatter` -- which blanks any position that is not
    decade-aligned and drops the sign of negative symlog values -- this labels
    whatever positions it is given, signed. That keeps the default bar readable
    and also means a later `cbar.set_ticks([...])` is labelled as asked, without
    needing a paired `set_ticklabels`.

    Returns:
        matplotlib.ticker.FuncFormatter: Wraps the module-level `_format_tick_value`.
    """
    return FuncFormatter(_format_tick_value)


def _decades_in_range(
    locator: Locator, vmin: float, vmax: float, fallback: np.ndarray
) -> np.ndarray:
    """Run a matplotlib locator over `[vmin, vmax]`, clipped, or fall back.

    Shared by the sym_log and log tick helpers: take the locator's tick values,
    keep only those inside the colour range, and fall back to the caller's linear
    ladder when fewer than two land in range (e.g. a range spanning less than one
    decade).

    Args:
        locator: A matplotlib tick locator (`SymmetricalLogLocator` / `LogLocator`).
        vmin: Lower bound of the colour range.
        vmax: Upper bound of the colour range.
        fallback: Tick positions to use when the locator is too sparse.

    Returns:
        numpy.ndarray: The in-range locator positions, or `fallback`.
    """
    positions = np.asarray(locator.tick_values(vmin, vmax), dtype=float)
    positions = positions[(positions >= vmin) & (positions <= vmax)]
    return positions if positions.size >= 2 else np.asarray(fallback, dtype=float)


def _symlog_tick_positions(
    vmin: float, vmax: float, linthresh: float, fallback: np.ndarray
) -> np.ndarray:
    """Scale-aware symlog tick positions within `[vmin, vmax]`.

    Linear positions are meaningless on a symlog axis, so place the bar's ticks
    with matplotlib's `SymmetricalLogLocator` (the natural choice for a
    `SymLogNorm`) at the base-10 decades. Falls back to the caller's linear ladder
    if the locator yields fewer than two in-range positions (e.g. a range entirely
    inside `linthresh`), so the return is decade-aligned only in the common case.

    Args:
        vmin: Lower bound of the colour range.
        vmax: Upper bound of the colour range.
        linthresh: The symlog linear threshold (the norm's `linthresh`).
        fallback: Tick positions to use when the locator is too sparse.

    Returns:
        numpy.ndarray: The symlog-appropriate tick positions.
    """
    # base=10 ticks are intentional even though the SymLogNorm uses base=e: the
    # labels people read are base-10 decades, and consecutive base-10 decades stay
    # evenly spaced on a base-e symlog transform (ln(10x) - ln(x) = ln(10)).
    locator = SymmetricalLogLocator(base=10.0, linthresh=linthresh)
    return _decades_in_range(locator, vmin, vmax, fallback)


def _log_tick_positions(vmin: float, vmax: float, fallback: np.ndarray) -> np.ndarray:
    """Scale-aware log tick positions within `[vmin, vmax]`.

    Places the bar's ticks with matplotlib's `LogLocator` (the natural choice for
    a `LogNorm`) at the base-10 decades, falling back to the caller's linear ladder
    if fewer than two decades land in range.

    Args:
        vmin: Lower (strictly positive) bound of the colour range.
        vmax: Upper bound of the colour range.
        fallback: Tick positions to use when the locator is too sparse.

    Returns:
        numpy.ndarray: The log-appropriate tick positions.
    """
    return _decades_in_range(LogLocator(base=10.0), vmin, vmax, fallback)


#: Fraction of the data's peak magnitude used to size an auto-derived symlog
#: `linthresh` when the caller passes no explicit `threshold`. `linthresh` is the
#: half-width of the zero-centred linear band, so at 1% the band spans +/-1% of
#: the data's peak magnitude -- tying the log decades to the data's magnitude
#: instead of the fixed `0.0001` that ran arbitrarily far below it (issue #337).
#: This bounds how far the decades reach below the data, not the exact smallest
#: decade -- an O(1) range straddling zero can still show a sub-unit decade or two
#: below its peak.
_AUTO_LINTHRESH_FRACTION = 0.01


def _auto_linthresh(vmin: float, vmax: float) -> float:
    """Derive a symlog `linthresh` matched to the data's magnitude.

    A fixed `linthresh` far below the data (the old `0.0001` default) pushes a
    wide-ranging field almost entirely into the log region, so the colour bar
    fills with near-zero sub-scale decades. Sizing it as a small fraction of the
    peak magnitude ties the linear band to the data's magnitude, so the log
    decades stay near the data's scale rather than running arbitrarily far below
    it. This bounds how far the decades reach, not the exact smallest one: a
    large-magnitude field shows only decades near its scale, while an O(1) range
    straddling zero can still show a sub-unit decade or two below its peak.

    Args:
        vmin: Lower bound of the colour range.
        vmax: Upper bound of the colour range.

    Returns:
        float: A strictly-positive `linthresh`. Falls back to `1.0` for an
            all-zero range (`vmin == vmax == 0`), where the value is immaterial
            because the tick helper then uses the linear fallback anyway.
    """
    peak = max(abs(float(vmin)), abs(float(vmax)))
    linthresh = peak * _AUTO_LINTHRESH_FRACTION
    return linthresh if linthresh > 0.0 else 1.0


#: `linscale` used when the caller passes no explicit `scale`, paired with an
#: auto-derived `linthresh`. matplotlib's own default, `1.0`, gives the linear
#: band one decade of colour-bar width -- comparable to each log decade -- so the
#: now data-sized band stays legible and its in-band ticks don't overprint. The
#: old vendored `0.001` only suited the tiny fixed `0.0001` threshold; against a
#: data-sized band it crushed the near-zero region to a sliver (issue #337).
_AUTO_LINSCALE = 1.0


#: Defaults for the colour-scale options, matching
#: `cleopatra.styling.styles.DEFAULT_OPTIONS`. Kept here so
#: `ColorScaling.from_options` can fill a missing key rather than raising.
_SCALE_DEFAULTS: dict[str, Any] = {
    "color_scale": "linear",
    "gamma": 0.5,
    "line_threshold": None,
    "line_scale": None,
    "bounds": None,
    "midpoint": 0,
    "samples": 512,
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
    `boundary`, `midpoint`, `equalize`) over the raw dataclass -- each exposes
    only the fields its scale uses, so nonsensical combinations (e.g. a
    `midpoint` on a `linear` scale) cannot be built.

    Attributes:
        kind: The scale kind (`cleopatra.styling.styles.ColorScale`).
        gamma: Exponent for the `power` scale. Ignored by other kinds.
        line_threshold: Linear-region threshold (`linthresh`) for
            `sym-lognorm`. `None` (the default) auto-derives it from the data
            range at render time; an explicit value is used as given.
        line_scale: Linear-region scale factor (`linscale`) for `sym-lognorm`.
            `None` (the default) pairs a sensible width (matplotlib's `1.0`)
            with an auto-derived `line_threshold`; an explicit value is used as
            given.
        bounds: Explicit bin edges for `boundary-norm`.
        center: Centre value for the `midpoint` scale (the value pinned to
            the colormap centre). Named `center` rather than `midpoint` so
            the field does not shadow the `midpoint()` variant constructor.
        samples: Number of quantile samples for the `equalize` scale -- the
            resolution of the empirical-CDF table. Ignored by other kinds.
    """

    kind: ColorScale = ColorScale.LINEAR
    gamma: float = 0.5
    line_threshold: float | None = None
    line_scale: float | None = None
    bounds: list[float] | None = None
    center: float = 0
    samples: int = 512

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
    def sym_log(
        cls, threshold: float | None = None, scale: float | None = None
    ) -> ColorScaling:
        """A symmetric-log (`SymLogNorm`) colour scale.

        Args:
            threshold: The linear-region half-width (`linthresh`) -- the
                boundary between the linear band around zero and the log tail.
                Defaults to `None`, which auto-derives it from the data range at
                render time (a small fraction of the data's peak magnitude), so
                the log decades track the data's own scale instead of running
                far below it. Pass an explicit value to pin the band near a
                scale you care about; an explicit `threshold` always wins over
                the auto-derivation.
            scale: The linear-region scale factor (`linscale`) -- how much
                colour-bar width the linear band around zero occupies. Defaults
                to `None`, which pairs a sensible width (matplotlib's `1.0`)
                with the auto-derived `threshold` so the widened linear band
                stays legible. Pass an explicit value to override it; an
                explicit `scale` always wins.

        Examples:
            - Exposes the two `sym-lognorm` knobs:
                ```python
                >>> from cleopatra.styling.scaling import ColorScaling
                >>> s = ColorScaling.sym_log(threshold=0.01, scale=0.1)
                >>> (s.line_threshold, s.line_scale)
                (0.01, 0.1)

                ```
            - The default defers the threshold to the data range:
                ```python
                >>> ColorScaling.sym_log().line_threshold is None
                True

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

        On `ArrayGlyph`, an un-pinned `vmin` is floored at the smallest positive
        value that is not an extreme low outlier (`ArrayGlyph._log_safe_vmin`),
        so a lone near-zero pixel does not drag the bar's decades below the
        data's bulk (issue #339); pass an explicit `vmin` to keep the raw
        minimum.

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
    def equalize(cls, samples: int = 512) -> ColorScaling:
        """A continuous rank-equalising colour scale (histogram equalisation).

        Spreads the colour ramp by rank rather than by value, so every quantile
        of the data receives an equal share of the ramp. On a skewed field
        (bathymetry, population, discharge) this reveals the bulk that a linear
        norm flattens into one tone -- and, unlike `boundary`, it stays
        continuous, so it does not posterise a shaded-relief surface. It is
        backed by a `matplotlib.colors.FuncNorm` built from the data's own
        empirical CDF at render time.

        It ranks within the resolved display window, so `vmin`/`vmax` and
        `robust=True` clip the field before ranking (handy for taming outliers
        on a skewed surface); with no limits it ranks the whole field.

        The scale is data-driven, so it is wired for `ArrayGlyph` (which can
        supply its cell values); using it where the values are unavailable
        raises a clear error rather than guessing.

        Args:
            samples: Number of quantile samples in the empirical-CDF table --
                its resolution. Must be at least 2. Defaults to `512`.

        Raises:
            ValueError: If `samples` is less than 2.

        Examples:
            - The equalize scale carries its sample count:
                ```python
                >>> from cleopatra.styling.scaling import ColorScaling
                >>> ColorScaling.equalize().kind.value
                'equalize'
                >>> ColorScaling.equalize(samples=256).samples
                256

                ```
        """
        if samples < 2:
            raise ValueError(f"equalize needs samples >= 2, got {samples}.")
        return cls(kind=ColorScale.EQUALIZE, samples=samples)

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
            samples=options.get("samples", _SCALE_DEFAULTS["samples"]),
        )

    def to_options(self) -> dict[str, Any]:
        """Flatten back to the `default_options` keys the engine reads.

        Returns:
            dict: The colour-scale keys, with `color_scale` as the plain
                string value and `norm` reset to `None` (a scale clears any
                raw-norm escape hatch).

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
            "samples": self.samples,
            # A scale is a full reset: choosing one clears any raw-norm escape
            # hatch (`plot(norm=...)`) so a later `color=ColorScaling.*` is not
            # silently shadowed by a sticky caller norm.
            "norm": None,
        }

    def build_norm(
        self,
        ticks: np.ndarray,
        levels: int | list[float] | np.ndarray | None = None,
        extend: str | None = None,
        values: np.ndarray | None = None,
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
            values: The data's own values, used only by the `equalize` scale
                to build its empirical-CDF table. `None` (the default) is fine
                for every other kind; `equalize` raises when it is `None`.

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
            norm, cbar_kw = self._sym_log_norm(ticks, vmin, vmax)
        elif self.kind == ColorScale.LOGNORM:
            norm, cbar_kw = self._log_norm(ticks, vmin, vmax)
        elif self.kind == ColorScale.BOUNDARY_NORM:
            norm, cbar_kw = self._boundary_norm(ticks, bounds_from_levels)
        elif self.kind == ColorScale.MIDPOINT:
            norm = MidpointNormalize(midpoint=self.center, vmin=vmin, vmax=vmax)
            cbar_kw = {"ticks": ticks}
        elif self.kind == ColorScale.EQUALIZE:
            norm, cbar_kw = self._equalize_norm(ticks, values)
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

    def _sym_log_norm(
        self, ticks: np.ndarray, vmin: Any, vmax: Any
    ) -> tuple[colors.Normalize, dict[str, Any]]:
        """Symmetric-log norm, deriving the linear band from the data when unset.

        A `None` threshold means "match the data": derive `linthresh` from the
        range so the log decades stay near the data's scale instead of running
        arbitrarily far below it (issue #337). The same value drives the norm
        (the rendered image) and the bar ticks, so they stay consistent. A `None`
        scale likewise pairs matplotlib's `1.0` `linscale` with that wider band,
        so the near-zero region keeps a legible share of the bar and its in-band
        ticks don't overprint. An explicit `threshold`/`scale` is used as given.
        """
        linthresh = (
            _auto_linthresh(vmin, vmax)
            if self.line_threshold is None
            else self.line_threshold
        )
        linscale = _AUTO_LINSCALE if self.line_scale is None else self.line_scale
        norm = colors.SymLogNorm(
            linthresh=linthresh, linscale=linscale, base=np.e, vmin=vmin, vmax=vmax
        )
        cbar_kw = {
            "ticks": _symlog_tick_positions(vmin, vmax, linthresh, ticks),
            "format": _plain_tick_formatter(),
        }
        return norm, cbar_kw

    def _log_norm(
        self, ticks: np.ndarray, vmin: Any, vmax: Any
    ) -> tuple[colors.Normalize, dict[str, Any]]:
        """Plain-log norm over a strictly-positive range, widening a constant field.

        A constant *positive* field yields a single tick (`vmin == vmax`); a log
        scale cannot span a zero-width range, so widen it -- matching the
        data-style `norm='log'` path, which bumps `vmax = vmin + 1.0`. Only widen
        a positive constant: a non-positive one must raise, and its error should
        report the real bound, not a widened one.
        """
        lo, hi = float(vmin), float(vmax)
        if hi == lo and lo > 0.0:
            hi = lo + 1.0
        norm = build_log_norm(
            lo, hi, context="ColorScaling.log()", remedy="use ColorScaling.sym_log()"
        )
        cbar_kw = {
            "ticks": _log_tick_positions(lo, hi, ticks),
            "format": _plain_tick_formatter(),
        }
        return norm, cbar_kw

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

    def _equalize_norm(
        self, ticks: np.ndarray, values: np.ndarray | None
    ) -> tuple[colors.Normalize, dict[str, Any]]:
        """Rank-equalising norm: a `FuncNorm` over the data's own empirical CDF.

        Maps each value to its quantile rank in `[0, 1]`, so every quantile of
        the data gets an equal share of the ramp. Needs the data itself (not
        just the tick range), so `values` is required; the colour bar's ticks
        are placed at the data's quantiles rather than linearly, so they sit
        evenly on the equalised axis instead of implying a linear one.

        Ranks within the resolved display window `[ticks[0], ticks[-1]]`, so an
        explicit `vmin`/`vmax` or `robust=True` clips the field before ranking
        (out-of-window outliers then take the end colours rather than flattening
        the in-window distribution). The default window is the data range, so it
        keeps every cell.
        """
        if values is None:
            raise ValueError(
                "ColorScaling.equalize() needs the data values to build its "
                "quantile table. It is wired for ArrayGlyph (which supplies its "
                "cell values); pass values= to build_norm() to use it directly."
            )
        data = np.asarray(values, dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            raise ValueError("ColorScaling.equalize() got no finite values to rank.")
        if ticks is not None and len(ticks) >= 2:
            lo_lim, hi_lim = float(ticks[0]), float(ticks[-1])
            if hi_lim > lo_lim:
                in_window = data[(data >= lo_lim) & (data <= hi_lim)]
                if in_window.size:
                    data = in_window
        q = np.linspace(0.0, 1.0, self.samples)
        qv = np.quantile(data, q)
        # A flat plateau repeats a data value across several quantiles, giving
        # np.interp a zero-width interval; keep a strictly increasing support by
        # dropping the repeats (np.unique returns sorted-unique + first index).
        qv_unique, first = np.unique(qv, return_index=True)
        q_unique = q[first]
        if qv_unique.size < 2:
            # A constant / fully-tied field has no rank spread to apply: fall
            # back to a degenerate linear norm rather than dividing by zero.
            lo = float(qv_unique[0])
            return colors.Normalize(vmin=lo, vmax=lo), {"ticks": np.array([lo])}
        lo, hi = float(qv_unique[0]), float(qv_unique[-1])
        norm = colors.FuncNorm(
            (
                lambda x, xp=qv_unique, fp=q_unique: np.interp(x, xp, fp),
                lambda y, xp=q_unique, fp=qv_unique: np.interp(y, xp, fp),
            ),
            vmin=lo,
            vmax=hi,
        )
        n_ticks = len(ticks) if ticks is not None and len(ticks) >= 2 else 8
        # Reuse the CDF table (qv) rather than a second np.quantile sort of the
        # full field; interpolating it at the tick quantiles gives the same
        # quantile-spaced positions.
        tick_vals = np.unique(np.interp(np.linspace(0.0, 1.0, n_ticks), q, qv))
        return norm, {"ticks": tick_vals, "format": _plain_tick_formatter()}
