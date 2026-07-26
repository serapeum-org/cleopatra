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

The kind is also what drives `Palette.default_norm` — pair it with `to_colormap` to
get both the colours and the matching matplotlib norm in one step: a symmetric
`CenteredNorm` for `diverging`, a `BoundaryNorm` over the class indices for
`qualitative`, and a linear `Normalize` otherwise.

The built-in **haze / CAMS-AOD / flame** families live here and register at import,
so the registry is populated whether you import `cleopatra.palettes` or
`cleopatra.colors`. Their `name → Colormap` dicts (`HAZE_COLORMAPS`,
`CAMS_AOD_COLORMAPS`, `FLAME_COLORMAPS`) are still importable from `cleopatra.colors`
for backward compatibility.

## Curated palettes

A small set of net-new palettes ships pre-registered — **generated** with this
package's own tools (`make_diverging` / `make_categorical`), not vendored or copied
from any other library:

| Name | Kind | Notes |
|------|------|-------|
| `diverging_blue_red` | diverging | blue ↔ red, lightness-balanced, neutral centre |
| `diverging_purple_green` | diverging | purple ↔ green |
| `diverging_brown_teal` | diverging | brown ↔ teal (moisture/precip anomalies) |
| `category12` | qualitative | 12 maximally-distinguishable class colours |
| `category20` | qualitative | 20 class colours (the first 12 match `category12`) |

The diverging maps are built on demand from their two endpoints (so the centre lands
exactly on the midpoint); the categorical swatches were generated once with
`make_categorical` (greedy max-min in CIELAB) and frozen for a stable identity. Fetch
any of them like the built-ins:

```python
from cleopatra.palettes import get_palette, available_palettes

available_palettes("diverging")     # ['diverging_blue_red', 'diverging_brown_teal', ...]
cmap = get_palette("category12").to_colormap()   # a 12-colour ListedColormap
```

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

p = get_palette("temp_anomaly")
cmap = p.to_colormap()                # a LinearSegmentedColormap
norm = p.default_norm(vmin=-4, vmax=6)  # a CenteredNorm symmetric about 0
print(available_palettes("diverging"))  # ['temp_anomaly', ...]
# ... then: ax.imshow(data, cmap=cmap, norm=norm)
```

### Discover the built-in families

```python
from cleopatra.palettes import available_palettes, get_palette

print(available_palettes("sequential"))            # includes 'haze_dust', 'cams_aod_blue_red', ...
print(get_palette("cams_aod_blue_red").source)     # 'ecmwf-magics'
```
