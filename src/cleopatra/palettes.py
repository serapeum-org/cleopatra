"""One unified palette record and registry.

Every colour ramp cleopatra knows about -- its own `HAZE`/`CAMS`/`FLAME`
families, and anything you generate with `cleopatra.perceptual` -- is described
by a single `Palette` record: a name, a `kind` (what it is *for*), the colours,
and a `source` (provenance only). One schema, one registry, one lookup API, so
adding a family is data, not a new code path.

`kind` (a `PaletteKind`) is the single knob that decides how a palette should be
turned into a colormap and, downstream, its natural norm/legend:

- `sequential` / `diverging` / `cyclic`: continuous -- the colours are anchors
    interpolated perceptually (in CIELAB, via `cleopatra.perceptual`).
- `qualitative`: discrete -- the colours are the exact class swatches, kept as a
    `ListedColormap` with no interpolation.

Example:
    >>> from cleopatra.palettes import Palette, PaletteKind, register, get_palette
    >>> p = register(Palette("demo", PaletteKind.SEQUENTIAL, ("#ffffff", "#004cff")))
    >>> get_palette("demo").kind
    <PaletteKind.SEQUENTIAL: 'sequential'>
    >>> get_palette("demo").to_colormap().name
    'demo'
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (
    BoundaryNorm,
    CenteredNorm,
    Colormap,
    ListedColormap,
    Normalize,
)
from matplotlib.figure import Figure

from cleopatra.perceptual import make_diverging, perceptual_colormap

__all__ = [
    "PaletteKind",
    "Palette",
    "register",
    "get_palette",
    "available_palettes",
    "PALETTES",
    "preview_palettes",
    "HAZE_COLORMAPS",
    "CAMS_AOD_COLORMAPS",
    "FLAME_COLORMAPS",
]


class PaletteKind(StrEnum):
    """What a palette is for -- drives colormap construction and default norm.

    Members are plain strings (`StrEnum`), so `PaletteKind.SEQUENTIAL ==
    "sequential"` and construction is case-insensitive
    (`PaletteKind("Diverging") is PaletteKind.DIVERGING`).

    Examples:
        ```python
        >>> from cleopatra.palettes import PaletteKind
        >>> PaletteKind.DIVERGING == "diverging"
        True
        >>> PaletteKind("Qualitative") is PaletteKind.QUALITATIVE
        True

        ```
    """

    SEQUENTIAL = "sequential"
    DIVERGING = "diverging"
    CYCLIC = "cyclic"
    QUALITATIVE = "qualitative"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            return cls.__members__.get(value.upper().replace("-", "_"))
        return None


def _auto_bounds(
    data: np.ndarray | None, vmin: float | None, vmax: float | None
) -> tuple[float | None, float | None]:
    """Fill missing `vmin`/`vmax` from the finite range of `data`.

    A bound already given is kept; a bound left `None` with no usable data stays
    `None`, so the norm autoscales at draw time.
    """
    if data is None or (vmin is not None and vmax is not None):
        return vmin, vmax
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return vmin, vmax
    if vmin is None:
        vmin = float(finite.min())
    if vmax is None:
        vmax = float(finite.max())
    return vmin, vmax


def _centered_norm(
    vmin: float | None, vmax: float | None, center: float | None
) -> CenteredNorm:
    """A `CenteredNorm` symmetric about `center` (default 0).

    The halfrange is derived from `vmin`/`vmax` when both are given (so the map
    spans the data symmetrically); otherwise it is left to autoscale.
    """
    vcenter = 0.0 if center is None else float(center)
    halfrange = None
    if vmin is not None and vmax is not None:
        halfrange = max(abs(vmin - vcenter), abs(vmax - vcenter)) or None
    return CenteredNorm(vcenter=vcenter, halfrange=halfrange)


@dataclass(frozen=True)
class Palette:
    """One colour palette: name, kind, colours, and provenance.

    Args:
        name: Unique registry key / colormap name.
        kind: A `PaletteKind` (or its string value). Coerced to the enum.
        colors: The palette colours (hex strings or names) -- interpolation
            anchors for continuous kinds, exact class swatches for
            `qualitative`.
        source: Free-text provenance (e.g. `"cleopatra"`, `"ecmwf"`); metadata
            only. Defaults to `"cleopatra"`.

    Examples:
        ```python
        >>> from cleopatra.palettes import Palette, PaletteKind
        >>> Palette("d", "diverging", ("#762a83", "#f4f4f4", "#1b7837")).kind
        <PaletteKind.DIVERGING: 'diverging'>

        ```
    """

    name: str
    kind: PaletteKind
    colors: tuple[str, ...]
    source: str = "cleopatra"

    def __post_init__(self):
        # Coerce a string kind / list of colours to the canonical types so the
        # record is uniform however it was constructed.
        object.__setattr__(self, "kind", PaletteKind(self.kind))
        object.__setattr__(self, "colors", tuple(self.colors))

    def to_colormap(self, n: int = 256) -> Colormap:
        """Build a matplotlib `Colormap` from this palette.

        The colormap is constructed according to `kind`:

        - `qualitative`: the exact swatches as a `ListedColormap` (no interpolation).
        - `diverging`: `make_diverging` from the first and last colours, so the
            neutral centre lands exactly on the midpoint and the arms are
            lightness-balanced. A three-colour diverging palette uses its middle
            colour as the neutral centre; otherwise a near-white default is used.
            A palette with more than three colours has its interior colours
            ignored (only the first/last shape the ramp) and a warning is
            emitted.
        - `sequential` / `cyclic`: the colours interpolated perceptually (CIELAB).

        Args:
            n: Levels for continuous kinds. Defaults to 256. Ignored for
                `qualitative`.

        Returns:
            matplotlib.colors.Colormap: The colormap for this palette.
        """
        if self.kind is PaletteKind.QUALITATIVE:
            return ListedColormap(list(self.colors), name=self.name)
        if self.kind is PaletteKind.DIVERGING:
            if len(self.colors) > 3:
                warnings.warn(
                    f"diverging palette {self.name!r} has {len(self.colors)} "
                    f"colours; to_colormap builds the ramp from only the first "
                    f"and last (a 3-colour palette also uses its middle as the "
                    f"centre), so the interior colours are ignored.",
                    stacklevel=2,
                )
            center = self.colors[1] if len(self.colors) == 3 else "#f4f4f4"
            return make_diverging(
                self.colors[0], self.colors[-1], n, center=center, name=self.name
            )
        return perceptual_colormap(self.name, list(self.colors), n)

    def default_norm(
        self,
        data: np.ndarray | None = None,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        center: float | None = None,
    ) -> Normalize:
        """Return the matplotlib norm that suits this palette's `kind`.

        The companion to `to_colormap`: pairing a palette's colormap with the norm
        its kind implies gives a sensible default rendering without hand-picking a
        norm every time.

        - `sequential` / `cyclic`: a linear `Normalize` over `[vmin, vmax]`.
        - `diverging`: a `CenteredNorm` symmetric about `center` (default `0.0`),
            so the colormap's neutral midpoint lands on the centre and both ends are
            equidistant.
        - `qualitative`: a `BoundaryNorm` over the `N` discrete class indices, so an
            integer class `k` maps to swatch `k`.

        Concrete bounds are taken from `vmin`/`vmax` when given, else from `data`'s
        finite range; a missing bound left as `None` autoscales at draw time. `data`
        and the bounds are ignored for `qualitative`.

        Args:
            data: Optional array to auto-range from when `vmin`/`vmax` are omitted.
            vmin: Lower bound (continuous kinds).
            vmax: Upper bound (continuous kinds).
            center: Centre for a `diverging` norm. Defaults to `0.0`.

        Returns:
            matplotlib.colors.Normalize: The norm for this palette's kind.

        Examples:
            ```python
            >>> from cleopatra.palettes import Palette
            >>> from matplotlib.colors import BoundaryNorm, CenteredNorm, Normalize
            >>> seq = Palette("s", "sequential", ("#ffffff", "#000000"))
            >>> type(seq.default_norm(vmin=0, vmax=10)) is Normalize
            True
            >>> div = Palette("d", "diverging", ("#0000ff", "#ffffff", "#ff0000"))
            >>> isinstance(div.default_norm(vmin=-5, vmax=8), CenteredNorm)
            True
            >>> Palette("q", "qualitative", ("#f00", "#0f0", "#00f")).default_norm().Ncmap
            3

            ```
        """
        if self.kind is PaletteKind.QUALITATIVE:
            n = len(self.colors)
            return BoundaryNorm(np.arange(n + 1) - 0.5, n)
        vmin, vmax = _auto_bounds(data, vmin, vmax)
        if self.kind is PaletteKind.DIVERGING:
            return _centered_norm(vmin, vmax, center)
        return Normalize(vmin=vmin, vmax=vmax)


#: The global palette registry, keyed by name. Populate it with `register`.
PALETTES: dict[str, Palette] = {}


def register(palette: Palette) -> Palette:
    """Add (or replace) a palette in the registry and return it.

    Args:
        palette: The `Palette` to register under its `name`.

    Returns:
        Palette: The same palette, for convenient chaining.
    """
    PALETTES[palette.name] = palette
    return palette


def get_palette(name: str) -> Palette:
    """Look up a registered palette by name.

    Args:
        name: The palette's registry key.

    Returns:
        Palette: The registered palette.

    Raises:
        KeyError: If no palette is registered under `name`.
    """
    try:
        return PALETTES[name]
    except KeyError:
        raise KeyError(
            f"unknown palette {name!r}; registered: {available_palettes()}"
        ) from None


def available_palettes(kind: PaletteKind | str | None = None) -> list[str]:
    """List registered palette names, optionally filtered by kind.

    Args:
        kind: If given, return only palettes of this `PaletteKind` (or its
            string value). Defaults to `None` (all palettes).

    Returns:
        list[str]: Sorted palette names.

    Examples:
        ```python
        >>> from cleopatra.palettes import available_palettes
        >>> isinstance(available_palettes("sequential"), list)
        True

        ```
    """
    if kind is None:
        return sorted(PALETTES)
    kind = PaletteKind(kind)
    return sorted(n for n, p in PALETTES.items() if p.kind is kind)


#: Order kinds are grouped in for display (`preview_palettes`).
_KIND_ORDER = (
    PaletteKind.SEQUENTIAL,
    PaletteKind.DIVERGING,
    PaletteKind.CYCLIC,
    PaletteKind.QUALITATIVE,
)


def preview_palettes(
    kind: PaletteKind | str | None = None,
    *,
    names: Sequence[str] | None = None,
    n: int = 256,
) -> Figure:
    """Render registered palettes as a grouped swatch grid.

    Each palette is drawn as a horizontal strip of its colormap -- continuous
    kinds show a smooth ramp, `qualitative` shows its discrete class swatches --
    labelled with its name and `source`, under a bold heading per `PaletteKind`.
    A quick way to browse the registry (including anything you have `register`ed).

    Args:
        kind: Show only this kind (a `PaletteKind` or its string value). Ignored
            when `names` is given. Defaults to `None` (every registered palette).
        names: An explicit list of palette names to show instead of filtering by
            `kind`; still grouped by kind in the grid.
        n: Levels used to build each continuous colormap. Defaults to 256.

    Returns:
        matplotlib.figure.Figure: The swatch-grid figure (save or show it
        yourself; cleopatra never changes the active backend).

    Raises:
        KeyError: If a name in `names` is not registered.
        ValueError: If no palettes match the selection.

    Examples:
        ```python
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import matplotlib.pyplot as plt
        >>> from cleopatra.palettes import preview_palettes
        >>> fig = preview_palettes("diverging")
        >>> len(fig.axes) > 0
        True
        >>> plt.close(fig)

        ```
    """
    if names is not None:
        selected = [get_palette(name) for name in names]
    elif kind is not None:
        selected = [PALETTES[name] for name in available_palettes(kind)]
    else:
        selected = [PALETTES[name] for name in available_palettes()]
    if not selected:
        raise ValueError("no palettes to preview for the given selection")

    # A header row per non-empty kind group, then that group's palette rows.
    rows: list[tuple[str, object]] = []
    for group in _KIND_ORDER:
        members = [p for p in selected if p.kind is group]
        if not members:
            continue
        rows.append(("header", f"{group.value.title()}  ({len(members)})"))
        rows.extend(("palette", p) for p in members)

    grad = np.linspace(0, 1, n).reshape(1, -1)
    fig, axes = plt.subplots(len(rows), 1, figsize=(8.5, 0.42 * len(rows)))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.28, right=0.985, top=0.99, bottom=0.02, hspace=0.55)
    for ax, (row_type, payload) in zip(axes, rows):
        ax.set_xticks([])
        ax.set_yticks([])
        if row_type == "header":
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.text(0.0, 0.5, payload, fontsize=12, fontweight="bold",
                    va="center", ha="left", transform=ax.transAxes)
            continue
        ax.imshow(grad, aspect="auto", cmap=payload.to_colormap(n), extent=(0, 1, 0, 1))
        for spine in ax.spines.values():
            spine.set_edgecolor("0.6")
            spine.set_linewidth(0.5)
        ax.text(-0.02, 0.5, payload.name, fontsize=9, family="monospace",
                va="center", ha="right", transform=ax.transAxes)
        ax.text(1.008, 0.5, payload.source, fontsize=6.5, color="0.45",
                va="center", ha="left", transform=ax.transAxes)
    return fig


# --------------------------------------------------------------------------
# Built-in colour families -- registered at import, so the registry is
# populated whether you import `cleopatra.palettes` or `cleopatra.colors`.
# --------------------------------------------------------------------------

#: Sequential "haze" ramps (white at 0.0, saturating toward the named hue) -- the
#: value-modulated-alpha glow of ECMWF/CAMS aerosol animations.
_HAZE_ANCHORS: dict[str, list[str]] = {
    "organic_matter": ["#ffffff", "#ffd9f2", "#ff5fc9", "#c400a0", "#5c0050", "#200018"],
    "dust": ["#ffffff", "#fff2b3", "#ffcc33", "#ff6a00", "#7a1500", "#2a0800"],
}

#: The official ECMWF/CAMS aerosol-optical-depth (AOD at 550 nm) scales, colour
#: stops transcribed from the open-source Magics engine (`ecmwf/magics`,
#: Apache-2.0); only colour data, no code. The Magics style name is noted per
#: entry. Opacity ramps in the Magics originals are handled by cleopatra's
#: separate opacity axis, not baked into the colour.
_CAMS_AOD_ANCHORS: dict[str, list[str]] = {
    # Magics `sh_BuYlRd_aod` -- the canonical CAMS AOD scale.
    "blue_yellow_red": [
        "#d3d7eb", "#a8afd7", "#8892bf", "#a3a891", "#bebd65", "#d8d239",
        "#f3e70b", "#f4c60a", "#f6a508", "#f88406", "#f96205", "#fb4103",
        "#fd2001", "#ff0000",
    ],
    # Magics `sh_BuYlRdBr_aod` -- like blue_yellow_red but fading to dark maroon.
    "blue_yellow_red_brown": [
        "#d2d2ff", "#a1a1ff", "#7070ff", "#8787c7", "#b8b876", "#e9e926",
        "#ffda00", "#ff8a00", "#ff3900", "#f40000", "#c40000", "#930000",
        "#640000",
    ],
    # Magics `sh_all_aod` / `sh_all_aod550` -- blue->cyan->green->yellow->red.
    "blue_red": [
        "#0000f1", "#004cff", "#00b1ff", "#29ffce", "#7dff7a", "#ceff29",
        "#ffc400", "#ff6800", "#f10800", "#800000",
    ],
    # Magics `sh_Oranges_aod` -- white->dark-orange.
    "oranges": [
        "#ffefe0", "#fee9d4", "#fee2c6", "#fdd9b4", "#fdd0a2", "#fdc38d",
        "#fdb576", "#fda762", "#fd9a4e", "#fd8c3b", "#f87f2c", "#f3701b",
        "#ec620f", "#e25508", "#d84801", "#c54102", "#b03903", "#9e3303",
        "#8e2d04", "#7f2704",
    ],
}

#: Flame/heat ramps for rendering a scalar field (typically temperature) as a
#: glowing plume -- the aerosol technique recoloured for heat.
_FLAME_ANCHORS: dict[str, list[str]] = {
    # A stand-alone copy of matplotlib's `afmhot` (black->red->yellow->white).
    "white_hot": ["#000000", "#4d0000", "#990000", "#e02a00", "#ff7a00", "#ffbf1a", "#fff29a", "#ffffff"],
    "amber": ["#240000", "#7a0000", "#c81800", "#ff5a00", "#ff9a00", "#ffd21e", "#fff2a8"],
}

for _prefix, _anchor_map, _src in [
    ("haze", _HAZE_ANCHORS, "cleopatra"),
    ("cams_aod", _CAMS_AOD_ANCHORS, "ecmwf-magics"),
    ("flame", _FLAME_ANCHORS, "cleopatra"),
]:
    for _key, _anchors in _anchor_map.items():
        register(Palette(f"{_prefix}_{_key}", PaletteKind.SEQUENTIAL, _anchors, source=_src))
del _prefix, _anchor_map, _src, _key, _anchors

#: Backward-compatible `name -> Colormap` dicts, perceptually interpolated. New
#: code should prefer the registry (`get_palette`); these remain for existing
#: `from cleopatra.colors import HAZE_COLORMAPS` imports and `DATA_STYLES` wiring.
HAZE_COLORMAPS: dict[str, Colormap] = {
    k: get_palette(f"haze_{k}").to_colormap() for k in _HAZE_ANCHORS
}
CAMS_AOD_COLORMAPS: dict[str, Colormap] = {
    k: get_palette(f"cams_aod_{k}").to_colormap() for k in _CAMS_AOD_ANCHORS
}
FLAME_COLORMAPS: dict[str, Colormap] = {
    k: get_palette(f"flame_{k}").to_colormap() for k in _FLAME_ANCHORS
}


# --------------------------------------------------------------------------
# Curated palettes -- net-new and generated with this module's own tools, not
# vendored from any package. Diverging maps are built on demand from their two
# endpoints by `to_colormap` (via `make_diverging`, which balances the arms and
# lands the neutral centre on the midpoint). The categorical swatches were
# generated once with `make_categorical` (greedy max-min in CIELAB) and frozen
# here so each named palette has a stable identity.
# --------------------------------------------------------------------------

for _name, _low, _high in [
    ("diverging_blue_red", "#2166ac", "#b2182b"),
    ("diverging_purple_green", "#762a83", "#1b7837"),
    ("diverging_brown_teal", "#8c510a", "#01665e"),
]:
    register(Palette(_name, PaletteKind.DIVERGING, (_low, _high)))
del _name, _low, _high

register(
    Palette(
        "category12",
        PaletteKind.QUALITATIVE,
        (
            "#520af5", "#0ae30a", "#f50a0a", "#2e9bbf", "#f564bf", "#89760a",
            "#0ae3ad", "#5276f5", "#894040", "#0a6440", "#f50af5", "#e3bff5",
        ),
    )
)
register(
    Palette(
        "category20",
        PaletteKind.QUALITATIVE,
        (
            "#520af5", "#0ae30a", "#f50a0a", "#2e9bbf", "#f564bf", "#89760a",
            "#0ae3ad", "#5276f5", "#894040", "#0a6440", "#f50af5", "#e3bff5",
            "#f5890a", "#f5bf9b", "#add11c", "#645289", "#0a890a", "#f54064",
            "#891cad", "#add189",
        ),
    )
)
