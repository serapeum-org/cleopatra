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

from dataclasses import dataclass
from enum import StrEnum

from matplotlib.colors import Colormap, ListedColormap

from cleopatra.perceptual import perceptual_colormap

__all__ = [
    "PaletteKind",
    "Palette",
    "register",
    "get_palette",
    "available_palettes",
    "PALETTES",
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

        Continuous kinds interpolate the colours perceptually (CIELAB); a
        `qualitative` palette keeps its exact swatches as a `ListedColormap`.

        Args:
            n: Levels for continuous kinds. Defaults to 256. Ignored for
                `qualitative`.

        Returns:
            matplotlib.colors.Colormap: The colormap for this palette.
        """
        if self.kind is PaletteKind.QUALITATIVE:
            return ListedColormap(list(self.colors), name=self.name)
        return perceptual_colormap(self.name, list(self.colors), n)


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
