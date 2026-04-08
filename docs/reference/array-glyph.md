# Array Class

The `ArrayGlyph` class provides functionality for visualizing and manipulating arrays, including plotting, animating, and saving animations.

## Class Documentation

::: cleopatra.array_glyph.ArrayGlyph
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Examples

### Basic Array Plot

```python
import numpy as np
from cleopatra.array_glyph import ArrayGlyph

# Create a sample array
array = np.random.rand(10, 10)

# Create an ArrayGlyph object
array_glyph = ArrayGlyph(array)

# Plot the array
fig, ax, im, cbar = array_glyph.plot()
```

![Array Plot Example](../_images/array-plot.png)

### Display Cell Values

```python
# Plot the array with cell values displayed
fig, ax, im, cbar = array_glyph.plot(display_cell_values=True)
```

![Display Cell Values Example](../_images/display-cell-values.png)

### Display Points

```python
# Create some points to display on the array
points = np.array([[2, 3, 1], [5, 7, 2], [8, 1, 3]])

# Plot the array with points
fig, ax, im, cbar = array_glyph.plot(points=points)
```

![Display Points Example](../_images/display-points.png)

### Animation

```python
import numpy as np
from cleopatra.array_glyph import ArrayGlyph

# Create a time series of arrays
time_series = [np.random.rand(10, 10) for _ in range(5)]
time_labels = ["t1", "t2", "t3", "t4", "t5"]

# Create an ArrayGlyph object with the first array
array_glyph = ArrayGlyph(time_series[0])

# Animate the array over time
anim = array_glyph.animate(time=time_labels, points=points)

# Save the animation
array_glyph.save_animation("animation.gif", fps=2)
```

![Animation Example](../_images/animated_array.gif)
