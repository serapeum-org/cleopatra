import matplotlib
import numpy as np

matplotlib.use("agg")
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cleopatra.statistical_glyph import StatisticalGlyph


def test_histogram_one_sample():
    # make data
    np.random.seed(1)
    x = 4 + np.random.normal(0, 1.5, 200)
    stat_plot = StatisticalGlyph(x)
    fig, ax, hist = stat_plot.histogram()
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(hist, dict)
    assert ["n", "bins", "patches"] == list(hist.keys())


def test_histogram_multiple_sample():
    # make data
    np.random.seed(1)
    x = 4 + np.random.normal(0, 1.5, (200, 3))
    colors = ["red", "green", "blue"]
    stat_plot = StatisticalGlyph(x, color=colors)
    fig, ax, hist = stat_plot.histogram()
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(hist, dict)
    assert ["n", "bins", "patches"] == list(hist.keys())
