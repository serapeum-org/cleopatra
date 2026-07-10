# Colors Module

The `cleopatra.colors` module provides the `Colors` class for working with colors
— converting between formats (hex, RGB), validating color values, and getting the
type of color — plus the composable **"haze" data-style** helpers: ready-made
colormaps, value-tied alpha rendering, and a one-call multi-layer preset.

## Colors Class

The `Colors` class converts between different color formats (hex, RGB), validates
color values, extracts colour ramps from images, and builds matplotlib colormaps.

::: cleopatra.colors.Colors
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Composable data styles ("haze")

These module-level helpers render one or more value layers with per-pixel opacity
tied to value — the ECMWF/CAMS aerosol look — without constructing a glyph. They
pair with [`projection.apply_projection_style`](projection.md) to compose the same
style on a flat map or an orthographic globe.

- `HAZE_COLORMAPS` — ready `LinearSegmentedColormap` objects (`"organic_matter"`,
  `"dust"`); white → saturated hue, not registered in matplotlib's global registry.
- `DATA_STYLES` — named data-style presets; `"haze"` maps each layer to a colormap,
  label, and value/alpha limits.
- `alpha_scaled_image` — draw a 2D array with `imshow` where low values fade to
  transparent (NaN is always fully transparent).
- `alpha_scaled_mesh` — the `pcolormesh` / curvilinear-grid counterpart, for
  `orthographic_grid` output.
- `apply_data_style` — draw one or more named layers with a `DATA_STYLES` preset
  (colour + swatch legend) in one call; returns the artists keyed by layer name.

```python
import matplotlib
matplotlib.use("Agg")  # any backend
import matplotlib.pyplot as plt
import numpy as np

from cleopatra.colors import apply_data_style

# two synthetic aerosol layers on a lon/lat grid
yy, xx = np.mgrid[0:90, 0:180]
organic_matter = np.exp(-(((xx - 60) ** 2) / 400.0 + ((yy - 55) ** 2) / 200.0))
dust = np.exp(-(((xx - 80) ** 2) / 300.0 + ((yy - 35) ** 2) / 150.0))

fig, ax = plt.subplots(figsize=(6, 4))
artists = apply_data_style(
    ax,
    {"organic_matter": organic_matter, "dust": dust},
    style="haze",
    extent=[-180, 180, -90, 90],
    origin="lower",
)
```

::: cleopatra.colors.apply_data_style
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: cleopatra.colors.alpha_scaled_image
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: cleopatra.colors.alpha_scaled_mesh
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Examples

### Creating Color Objects

```python
from cleopatra.colors import Colors

# Create a Colors object with a hex color
hex_color = Colors("#FF5733")

# Create a Colors object with an RGB color (normalized)
rgb_color = Colors((1.0, 0.34, 0.2))

# Create a Colors object with an RGB color (0-255)
rgb_255_color = Colors((255, 87, 51))

# Create a Colors object with a named color
named_color = Colors("red")

# Create a Colors object with a list of colors
color_list = Colors(["red", "green", "blue"])
```

### Converting Between Color Formats

```python
# Convert to hex
hex_value = rgb_color.to_hex()
print(hex_value)  # "#FF5733"

# Convert to RGB (normalized)
rgb_value = hex_color.to_rgb(normalized=True)
print(rgb_value)  # (1.0, 0.34, 0.2)

# Convert to RGB (0-255)
rgb_255_value = hex_color.to_rgb(normalized=False)
print(rgb_255_value)  # (255, 87, 51)
```

### Validating Color Values

```python
# Check if a hex color is valid
is_valid = hex_color.is_valid_hex()
print(is_valid)  # True

# Check if an RGB color is valid
is_valid = rgb_color.is_valid_rgb()
print(is_valid)  # True
```
