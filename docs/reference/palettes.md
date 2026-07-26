# Palettes (registry)

The `cleopatra.palettes` module is the single home for every colour ramp cleopatra
knows about. One `Palette` record — a name, a `kind`, the colours, and a `source`
(provenance) — describes each palette; one registry looks them up. Adding a colour
family is therefore *data*, not a new code path.

A palette's [`PaletteKind`][cleopatra.palettes.PaletteKind] decides how it becomes a
colormap (and, downstream, its natural norm and legend): continuous kinds
(`sequential`/`diverging`/`cyclic`) interpolate their colours perceptually in CIELAB
(via [`cleopatra.perceptual`](perceptual.md)); a `qualitative` palette keeps its
exact class swatches as a `ListedColormap`.

The built-in **haze / CAMS-AOD / flame** families live here and register at import,
so the registry is populated whether you import `cleopatra.palettes` or
`cleopatra.colors`. Their `name → Colormap` dicts (`HAZE_COLORMAPS`,
`CAMS_AOD_COLORMAPS`, `FLAME_COLORMAPS`) are still importable from `cleopatra.colors`
for backward compatibility.

## PaletteKind

::: cleopatra.palettes.PaletteKind
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Palette

::: cleopatra.palettes.Palette
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Registry

Register a palette, look one up, or list what's available (optionally filtered by
kind). `PALETTES` is the underlying `name → Palette` mapping.

::: cleopatra.palettes.register
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: cleopatra.palettes.get_palette
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: cleopatra.palettes.available_palettes
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Examples

### Register and use a palette

```python
from cleopatra.palettes import Palette, PaletteKind, register, get_palette, available_palettes

# register a diverging palette (interpolated perceptually when built)
register(Palette("temp_anomaly", PaletteKind.DIVERGING, ("#762a83", "#f4f4f4", "#1b7837")))

cmap = get_palette("temp_anomaly").to_colormap()   # a LinearSegmentedColormap
print(available_palettes("diverging"))             # ['temp_anomaly', ...]
```

### Discover the built-in families

```python
from cleopatra.palettes import available_palettes, get_palette

print(available_palettes("sequential"))            # includes 'haze_dust', 'cams_aod_blue_red', ...
print(get_palette("cams_aod_blue_red").source)     # 'ecmwf-magics'
```
