import matplotlib

matplotlib.use("TkAgg")
import numpy as np

from cleopatra.statistical_glyph import StatisticalGlyph

# %%
np.random.seed(1)
x = 4 + np.random.normal(0, 1.5, 200)
stat_plot = StatisticalGlyph(x)
fig, ax, hist = stat_plot.histogram()
# %%
np.random.seed(1)
x = 4 + np.random.normal(0, 1.5, (200, 3))
# x = 4 + np.random.normal(0, 1.5, 200)
colors = ["red", "green", "blue"]
stat_plot = StatisticalGlyph(x, color=colors, alpha=0.4, rwidth=0.8)
res = stat_plot.histogram()
