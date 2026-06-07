# StatisticalGlyph Class

The `statistical_glyph` module provides the `StatisticalGlyph` class for creating
statistical plots — **histograms**, **boxplots**, **multi-boxplots**, and
**warming-stripe** bands. It handles both 1D (single-dimensional) and 2D
(multi-dimensional, one series per column) data.

## Class Documentation

::: cleopatra.statistical_glyph.StatisticalGlyph
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Examples

### 1D Data Example

```python
import numpy as np
import matplotlib.pyplot as plt
from cleopatra.statistical_glyph import StatisticalGlyph

# Create some random 1D data
np.random.seed(1)
data_1d = 4 + np.random.normal(0, 1.5, 200)

# Create a Statistic object with the 1D data
stat_plot_1d = StatisticalGlyph(data_1d)

# Generate a histogram plot for the 1D data
fig_1d, ax_1d, hist_1d = stat_plot_1d.histogram()
```

![One Histogram Example](../images/statistical_glyph/one-histogram.png)

### 2D Data Example

```python
# Create some random 2D data
data_2d = 4 + np.random.normal(0, 1.5, (200, 3))

# Create a Statistic object with the 2D data
stat_plot_2d = StatisticalGlyph(data_2d, color=["red", "green", "blue"], alpha=0.4, rwidth=0.8)

# Generate a histogram plot for the 2D data
fig_2d, ax_2d, hist_2d = stat_plot_2d.histogram()
```

![Three Histogram Example](../images/statistical_glyph/three-histogram.png)

### Boxplot

```python
import numpy as np
from cleopatra.statistical_glyph import StatisticalGlyph

# one box per column for 2D data
data = np.random.default_rng(0).normal(0, 1, (200, 3))
fig, ax, artists = StatisticalGlyph(data).boxplot(labels=["a", "b", "c"], notch=True)
```

### Grouped boxes at explicit positions (multiboxplot)

```python
# place boxes at caller-controlled x positions (e.g. lead times, months)
data = np.random.default_rng(0).normal(0, 1, (200, 4))
fig, ax, artists = StatisticalGlyph(data).multiboxplot(
    positions=[1, 3, 6, 12], labels=["1h", "3h", "6h", "12h"], widths=0.4
)
```

### Warming stripes

```python
# one full-height coloured stripe per value (1D), the Ed-Hawkins idiom
yearly = np.random.default_rng(0).normal(0, 1, 50).cumsum()
fig, ax, bars = StatisticalGlyph(yearly).stripes(cmap="RdBu_r")
```
