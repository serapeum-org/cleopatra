# Colors Class

The `Colors` class provides functionality for working with colors, including converting between different color formats (hex, RGB), validating color values, and getting the type of color.

## Class Documentation

::: cleopatra.colors.Colors
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
