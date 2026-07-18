"""
The `statistical_glyph` module provides a class for creating statistical plots, specifically histograms. The class,
`StatisticalGlyph`, is designed to handle both 1D (single-dimensional) and 2D (multi-dimensional) data.

The class has the following key features:

1. Initialization: The class is initialized with a set of values (1D or 2D array) and optional keyword arguments.
    The keyword arguments can be used to customize the appearance of the histogram.

2. Properties:
    The class includes properties for accessing the values and default options.

3. Histogram method:
    The class has a method named `histogram` that generates a histogram plot based on the provided values and options.
    The method handles both 1D and 2D data, and it allows customization of various aspects of the plot, such as the
    number of bins, color, transparency, and axis labels.

4. Error handling:
    The class includes error handling mechanisms to ensure that the number of colors provided matches the number of
    samples in the data. It also checks for invalid keyword arguments.

5. Examples:
    The class includes examples demonstrating how to use the histogram method with 1D and 2D data. The examples also
    include doctests to verify the correctness of the code.

Here's an example of how to use the `StatisticalGlyph` class:

```python
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from cleopatra.statistical_glyph import StatisticalGlyph

# Create some random 1D data
np.random.seed(1)
data_1d = 4 + np.random.normal(0, 1.5, 200)

# Create a StatisticalGlyph object with the 1D data
stat_plot_1d = StatisticalGlyph(data_1d)

# Generate a histogram plot for the 1D data
fig_1d, ax_1d, hist_1d = stat_plot_1d.histogram()

# Create some random 2D data
data_2d = 4 + np.random.normal(0, 1.5, (200, 3))

# Create a StatisticalGlyph object with the 2D data
stat_plot_2d = StatisticalGlyph(data_2d, color=["red", "green", "blue"], alpha=0.4, rwidth=0.8)

# Generate a histogram plot for the 2D data
fig_2d, ax_2d, hist_2d = stat_plot_2d.histogram()

# histogram() no longer calls plt.show() for you; display the figures yourself.
plt.show()
```
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

from cleopatra.glyph import _clear_prior_render_artists, _mark_render_artists
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS

STATISTICAL_DEFAULT_OPTIONS = {
    "figsize": (5, 5),
    "bins": 15,
    "color": ["#0504aa"],
    "alpha": 0.7,
    "rwidth": 0.85,
}
STATISTICAL_DEFAULT_OPTIONS = STYLE_DEFAULTS | STATISTICAL_DEFAULT_OPTIONS
#: Backwards-compatible alias for the statistical glyph's default options
#: (named like the other glyphs' `*_DEFAULT_OPTIONS` constants).
DEFAULT_OPTIONS = STATISTICAL_DEFAULT_OPTIONS


class StatisticalGlyph:
    """A class for creating statistical plots, specifically histograms.

    This class provides methods for initializing the class with numerical values and optional keyword arguments,
    and for creating histograms from the given values.

    The accepted option keys are exposed via the `DEFAULT_OPTIONS` class
    attribute and can be inspected or filtered before constructing an
    instance with the `option_keys` and `filter_kwargs` classmethods
    (mirroring `cleopatra.glyph.Glyph`, though this is a standalone class).

    Attributes:
        values: The numerical values to be plotted as histograms.
        default_options: The default options for creating histograms, including:
            - bins: Number of histogram bins
            - color: Colors for the histogram bars
            - alpha: Transparency of the bars
            - rwidth: Width of the bars
            - grid_alpha: Transparency of the grid
            - xlabel, ylabel: Axis labels
            - xlabel_font_size, ylabel_font_size: Font sizes for axis labels
            - xtick_font_size, ytick_font_size: Font sizes for axis ticks

    Methods:
        histogram(**kwargs): Creates a histogram from the given values with customizable options.

    Notes:
        The class can handle both 1D data (single histogram) and 2D data (multiple histograms
        overlaid on the same plot). For 2D data, the number of colors provided should match
        the number of data series (columns in the array).

    Examples:
        Create a histogram from 1D data:
        ```python
        >>> import numpy as np
        >>> from cleopatra.statistical_glyph import StatisticalGlyph
        >>> np.random.seed(1)
        >>> x = 4 + np.random.normal(0, 1.5, 200)
        >>> stat_plot = StatisticalGlyph(x)
        >>> fig, ax, hist = stat_plot.histogram()

        ```
        Create a histogram from 2D data with custom colors:
        ```python
        >>> np.random.seed(1)
        >>> x = 4 + np.random.normal(0, 1.5, (200, 3))
        >>> stat_plot = StatisticalGlyph(x, color=["red", "green", "blue"], alpha=0.4, rwidth=0.8)
        >>> fig, ax, hist = stat_plot.histogram()

        ```

        Example usage:
        ```python
        >>> np.random.seed(1)
        >>> x = 4 + np.random.normal(0, 1.5, 200)
        >>> stat_plot = StatisticalGlyph(x)
        >>> fig, ax, hist = stat_plot.histogram()
        >>> print(hist) # doctest: +SKIP
        {'n': [array([ 2.,  4.,  3., 10., 11., 20., 30., 27., 31., 25., 17.,  8.,  5.,
                            6.,  1.])], 'bins': [array([0.34774335, 0.8440597 , 1.34037605, 1.8366924 , 2.33300874,
                           2.82932509, 3.32564144, 3.82195778, 4.31827413, 4.81459048,
                           5.31090682, 5.80722317, 6.30353952, 6.79985587, 7.29617221,
                           7.79248856])], 'patches': [<BarContainer object of 15 artists>]}

        ```
        ![one-histogram](./../images/statistical_glyph/one-histogram.png)
    """

    #: Option keys this glyph accepts, exposed as a class attribute so they
    #: can be introspected/filtered before an instance exists (see
    #: `option_keys`/`filter_kwargs`).
    DEFAULT_OPTIONS = STATISTICAL_DEFAULT_OPTIONS

    def __init__(
        self,
        values: Union[List, np.ndarray],
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ):
        """Initialize the StatisticalGlyph object with values and optional customization parameters.

        Args:
            values: The numerical values to be plotted as histograms. Can be:
                - 1D array/list for a single histogram
                - 2D array/list for multiple histograms (one per column)
            fig: Pre-existing matplotlib Figure to draw on. Honoured in two ways
                by ``histogram()``: if ``ax`` is also given, ``fig`` is returned
                as-is alongside it; if ``ax`` is None, a new axes is created on
                this figure (the figure is not replaced). In the fig-only case
                the figure must be empty — if it already contains axes,
                ``histogram()`` raises ``ValueError`` and asks you to pass the
                target ``ax`` explicitly. If both ``fig`` and ``ax`` are None, a
                brand-new figure is created.
            ax: Pre-existing matplotlib Axes to draw on. If None, new axes
                are created when ``histogram()`` is called. When supplied,
                the histogram is composed into the given axes and its parent
                figure is inferred (unless ``fig`` is also passed explicitly).
                The box/stripe renderers (``boxplot``, ``multiboxplot``,
                ``stripes``) also fall back to this axes when their own
                ``ax`` argument is omitted. Note these renderers take ``ax``
                per call but never ``fig`` — the figure is bound here, at
                construction.
            **kwargs: Additional keyword arguments to customize the histogram appearance.
                Supported arguments include:
                - figsize: Figure size as (width, height) in inches, by default (5, 5).
                - bins: Number of histogram bins, by default 15.
                - color: Colors for the histogram bars, by default ["#0504aa"].
                    For 2D data, the number of colors should match the number of columns.
                - alpha: Transparency of the histogram bars, by default 0.7.
                    Values range from 0 (transparent) to 1 (opaque).
                - rwidth: Relative width of the bars, by default 0.85.
                    Values range from 0 to 1.
                - grid_alpha: Transparency of the grid lines, by default 0.75.
                - xlabel, ylabel: Labels for the x and y axes.
                - xlabel_font_size, ylabel_font_size: Font sizes for the axis labels.
                - xtick_font_size, ytick_font_size: Font sizes for the axis tick labels.

        Examples:
            Initialize with default options:
            ```python
            >>> import numpy as np
            >>> from cleopatra.statistical_glyph import StatisticalGlyph
            >>> np.random.seed(1)
            >>> x = np.random.normal(0, 1, 100)
            >>> stat = StatisticalGlyph(x)

            ```
            Initialize with custom options:
            ```python
            >>> stat_custom = StatisticalGlyph(
            ...     x,
            ...     figsize=(8, 6),
            ...     bins=20,
            ...     color=["#FF5733"],
            ...     alpha=0.5,
            ...     rwidth=0.9,
            ...     xlabel="Values",
            ...     ylabel="Frequency",
            ...     xlabel_font_size=14,
            ...     ylabel_font_size=14
            ... )

            ```
            Initialize with 2D data:
            ```python
            >>> data_2d = np.random.normal(0, 1, (100, 3))
            >>> stat_2d = StatisticalGlyph(
            ...     data_2d,
            ...     color=["red", "green", "blue"],
            ...     alpha=0.4
            ... )

            ```
            Compose into a pre-existing axes:
            ```python
            >>> import matplotlib.pyplot as plt
            >>> fig, ax = plt.subplots()
            >>> stat = StatisticalGlyph(x, fig=fig, ax=ax)
            >>> fig2, ax2, hist = stat.histogram()
            >>> ax2 is ax
            True

            ```
        """
        self._values = values
        self._fig = fig
        self._ax = ax
        options_dict = STATISTICAL_DEFAULT_OPTIONS.copy()
        options_dict.update(kwargs)
        self._default_options = options_dict

    @property
    def values(self):
        """Get the numerical values to be plotted.

        Returns:
            numpy.ndarray or list: The numerical values stored in the object, which can be:
                - 1D array/list for a single histogram
                - 2D array/list for multiple histograms (one per column)

        Examples:
            >>> import numpy as np
            >>> from cleopatra.statistical_glyph import StatisticalGlyph
            >>> np.random.seed(1)
            >>> x = np.random.normal(0, 1, 100)
            >>> stat = StatisticalGlyph(x)
            >>> values = stat.values
            >>> values.shape
            (100,)
        """
        return self._values

    @values.setter
    def values(self, values):
        """Set the numerical values to be plotted.

        Args:
            values: The new numerical values to be plotted as histograms. Can be:
                - 1D array/list for a single histogram
                - 2D array/list for multiple histograms (one per column)

        Examples:
            ```python
            >>> import numpy as np
            >>> from cleopatra.statistical_glyph import StatisticalGlyph
            >>> np.random.seed(1)
            >>> x1 = np.random.normal(0, 1, 100)
            >>> stat = StatisticalGlyph(x1)
            >>> # Update with new values
            >>> x2 = np.random.normal(5, 2, 100)
            >>> stat.values = x2

            ```
        """
        self._values = values

    @property
    def default_options(self) -> Dict:
        """Get the default options for histogram plotting.

        This property returns the dictionary of default options used for creating
        histogram plots. These options can be modified by passing keyword arguments
        to the class constructor or to the histogram method.

        Returns:
            Dict: A dictionary containing the default options for histogram plotting, including:
                - figsize: Figure size as (width, height) in inches.
                - bins: Number of histogram bins.
                - color: Colors for the histogram bars.
                - alpha: Transparency of the histogram bars.
                - rwidth: Relative width of the bars.
                - grid_alpha: Transparency of the grid lines.
                - xlabel, ylabel: Labels for the x and y axes.
                - xlabel_font_size, ylabel_font_size: Font sizes for the axis labels.
                - xtick_font_size, ytick_font_size: Font sizes for the axis tick labels.

        Examples:
            ```python
            >>> import numpy as np
            >>> from cleopatra.statistical_glyph import StatisticalGlyph
            >>> np.random.seed(1)
            >>> x = np.random.normal(0, 1, 100)
            >>> stat = StatisticalGlyph(x)
            >>> options = stat.default_options
            >>> print(options['bins'])
            15
            >>> print(options['alpha'])
            0.7

            ```
        """
        return self._default_options

    @classmethod
    def option_keys(cls) -> set[str]:
        """Return the keyword-argument keys this glyph accepts.

        Resolves from the class-level ``DEFAULT_OPTIONS`` so the accepted
        keys can be inspected without constructing an instance. Mirrors
        ``cleopatra.glyph.Glyph.option_keys`` (``StatisticalGlyph`` is a
        standalone class, not a ``Glyph`` subclass).

        Returns:
            set: The accepted option keys for this glyph class.

        Examples:
            - Inspect the accepted keys before building one:
                ```python
                >>> from cleopatra.statistical_glyph import StatisticalGlyph
                >>> keys = StatisticalGlyph.option_keys()
                >>> "bins" in keys
                True
                >>> "totally_unknown" in keys
                False

                ```

        See Also:
            filter_kwargs: Drop the keys this glyph does not accept.
        """
        return set(cls.DEFAULT_OPTIONS)

    @classmethod
    def filter_kwargs(cls, kwargs: dict) -> dict:
        """Return only the subset of ``kwargs`` whose keys this glyph accepts.

        Args:
            kwargs: A mapping of candidate option keys to values.

        Returns:
            Dict: The entries of ``kwargs`` whose keys are in ``option_keys()``.

        Examples:
            - Keep only the accepted keys:
                ```python
                >>> from cleopatra.statistical_glyph import StatisticalGlyph
                >>> raw = {"bins": 20, "alpha": 0.5, "bogus": 1}
                >>> safe = StatisticalGlyph.filter_kwargs(raw)
                >>> sorted(safe)
                ['alpha', 'bins']
                >>> safe["bins"]
                20

                ```

        See Also:
            option_keys: The set of keys this glyph accepts.
        """
        keys = cls.option_keys()
        return {key: val for key, val in kwargs.items() if key in keys}

    def histogram(self, **kwargs) -> Tuple[Figure, Axes, Dict]:
        """Create a histogram from the stored numerical values.

        This method generates a histogram visualization of the numerical values stored
        in the object. It can handle both 1D data (single histogram) and 2D data
        (multiple histograms overlaid on the same plot).

        Args:
            **kwargs: Additional keyword arguments to customize the histogram appearance.
                These will override any options set during initialization.
                Supported arguments include:
                - figsize: Figure size as (width, height) in inches, by default (5, 5).
                - bins: Number of histogram bins, by default 15.
                - color: Colors for the histogram bars, by default ["#0504aa"].
                    For 2D data, the number of colors should match the number of columns.
                - alpha: Transparency of the histogram bars, by default 0.7.
                    Values range from 0 (transparent) to 1 (opaque).
                - rwidth: Relative width of the bars, by default 0.85.
                    Values range from 0 to 1.
                - grid_alpha: Transparency of the grid lines, by default 0.75.
                - xlabel, ylabel: Labels for the x and y axes.
                - xlabel_font_size, ylabel_font_size: Font sizes for the axis labels.
                - xtick_font_size, ytick_font_size: Font sizes for the axis tick labels.

        Returns:
            Figure: The matplotlib Figure object containing the histogram.
            Axes: The matplotlib Axes object on which the histogram is drawn.
            Dict: A dictionary containing the histogram data with keys:
                - 'n': List of arrays containing the histogram bin counts
                - 'bins': List of arrays containing the bin edges
                - 'patches': List of BarContainer objects representing the histogram bars

        Raises:
            ValueError: If an invalid keyword argument is provided.
            ValueError: If the number of colors provided doesn't match the number of data series
                (columns) in 2D data.
            ValueError: If a ``fig`` was supplied without an ``ax`` and that figure already
                contains axes (pass the target ``ax`` explicitly in that case).

        Notes:
            For 2D data, multiple histograms will be overlaid on the same plot with
            different colors. The transparency (alpha) can be adjusted to make overlapping
            regions visible.

            The figure and axes used depend on what was passed to ``__init__``:

            - ``ax`` given: the histogram is drawn into that axes; the returned
              figure is the one explicitly passed as ``fig`` if any, otherwise
              the axes' own parent figure.
            - ``fig`` given without ``ax``: a new axes is added to that figure
              and used for drawing (the figure is reused, not replaced). The
              figure must be empty; if it already contains axes a ``ValueError``
              is raised so the caller passes the target ``ax`` explicitly.
            - neither given: a new figure and axes are created with ``figsize``.

        Examples:
            - 1D data.

                - Create a histogram from 1D data:

                    ```python
                    >>> import numpy as np
                    >>> from cleopatra.statistical_glyph import StatisticalGlyph
                    >>> np.random.seed(1)
                    >>> x = 4 + np.random.normal(0, 1.5, 200)
                    >>> stat_plot = StatisticalGlyph(x)
                    >>> fig, ax, hist = stat_plot.histogram()
                    >>> print(hist) # doctest: +SKIP
                    {'n': [array([ 2.,  4.,  3., 10., 11., 20., 30., 27., 31., 25., 17.,  8.,  5.,
                            6.,  1.])], 'bins': [array([0.34774335, 0.8440597 , 1.34037605, 1.8366924 , 2.33300874,
                           2.82932509, 3.32564144, 3.82195778, 4.31827413, 4.81459048,
                           5.31090682, 5.80722317, 6.30353952, 6.79985587, 7.29617221,
                           7.79248856])], 'patches': [<BarContainer object of 15 artists>]}
                    ```
                    ![one-histogram](./../images/statistical_glyph/one-histogram.png)

                - Create a histogram with custom bin count and labels:

                    ```python
                    >>> fig, ax, hist = stat_plot.histogram(
                    ...     bins=20,
                    ...     xlabel="Values",
                    ...     ylabel="Frequency",
                    ...     xlabel_font_size=14,
                    ...     ylabel_font_size=14
                    ... )

                    ```

            - 2D data.

                - Create a histogram with custom bin count and labels:
                    ```python
                    >>> np.random.seed(1)
                    >>> x = 4 + np.random.normal(0, 1.5, (200, 3))
                    >>> stat_plot = StatisticalGlyph(x, color=["red", "green", "blue"], alpha=0.4, rwidth=0.8)
                    >>> fig, ax, hist = stat_plot.histogram()
                    >>> print(hist) # doctest: +SKIP
                    {'n': [array([ 1.,  2.,  4., 10., 13., 19., 20., 32., 27., 23., 24., 11.,  5.,
                            5.,  4.]), array([ 3.,  4.,  9., 12., 20., 41., 29., 32., 25., 14.,  9.,  1.,  0.,
                            0.,  1.]), array([ 3.,  4.,  6.,  7., 25., 26., 31., 24., 30., 19., 11.,  9.,  4.,
                            0.,  1.])], 'bins': [array([-0.1896275 ,  0.33461786,  0.85886323,  1.38310859,  1.90735396,
                            2.43159932,  2.95584469,  3.48009005,  4.00433542,  4.52858078,
                            5.05282615,  5.57707151,  6.10131688,  6.62556224,  7.14980761,
                            7.67405297]), array([-0.1738017 ,  0.50031202,  1.17442573,  1.84853945,  2.52265317,
                            3.19676688,  3.8708806 ,  4.54499432,  5.21910804,  5.89322175,
                            6.56733547,  7.24144919,  7.9155629 ,  8.58967662,  9.26379034,
                            9.93790406]), array([0.24033902, 0.7940688 , 1.34779857, 1.90152835, 2.45525813,
                           3.0089879 , 3.56271768, 4.11644746, 4.67017723, 5.22390701,
                           5.77763679, 6.33136656, 6.88509634, 7.43882612, 7.99255589,
                           8.54628567])], 'patches': [<BarContainer object of 15 artists>,
                           <BarContainer object of 15 artists>, <BarContainer object of 15 artists>]}
                    ```
                    ![three-histogram](./../images/statistical_glyph/three-histogram.png)

                Access the histogram data:

                    ```python
                    >>> # Get the bin counts for the first data series
                    >>> bin_counts = hist['n'][0]
                    >>> # Get the bin edges for the first data series
                    >>> bin_edges = hist['bins'][0]

                    ```
        """
        for key, val in kwargs.items():
            if key not in self.default_options.keys():
                raise ValueError(
                    f"The given keyword argument:{key} is not correct, possible parameters are,"
                    f" {self.default_options}"
                )
            else:
                self.default_options[key] = val

        if self._ax is not None:
            ax = self._ax
            fig = self._fig if self._fig is not None else ax.get_figure()
        elif self._fig is not None:
            fig = self._fig
            if fig.axes:
                raise ValueError(
                    "The supplied `fig` already contains axes; pass the target axes via "
                    "`ax=` so the histogram is drawn on a specific axes instead of being "
                    "overlaid on the existing ones."
                )
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots(figsize=self.default_options["figsize"])

        # See `_clear_prior_render_artists`: a prior `histogram`/`boxplot`/
        # `multiboxplot`/`stripes` call on this Axes (this glyph's own, or
        # a different glyph sharing it via `StatisticalGlyph(ax=..., ...)`)
        # leaves its bars/boxes orphaned unless removed first.
        _clear_prior_render_artists(ax)

        n = []
        bins = []
        patches = []
        bins_val = self.default_options["bins"]
        color = self.default_options["color"]
        alpha = self.default_options["alpha"]
        rwidth = self.default_options["rwidth"]
        if self.values.ndim == 2:
            num_samples = self.values.shape[1]
            if len(color) != num_samples:
                raise ValueError(
                    f"The number of colors:{len(color)} should be equal to the number of samples:{num_samples}"
                )
        else:
            num_samples = 1

        for i in range(num_samples):
            if self.values.ndim == 1:
                vals = self.values
            else:
                vals = self.values[:, i]

            n_i, bins_i, patches_i = ax.hist(
                x=vals,
                bins=bins_val,
                color=color[i],
                alpha=alpha,
                rwidth=rwidth,
            )
            n.append(n_i)
            bins.append(bins_i)
            patches.append(patches_i)

        ax.grid(axis="y", alpha=self.default_options["grid_alpha"])
        ax.set_xlabel(
            self.default_options["xlabel"],
            fontsize=self.default_options["xlabel_font_size"],
        )
        ax.set_ylabel(
            self.default_options["ylabel"],
            fontsize=self.default_options["ylabel_font_size"],
        )
        ax.tick_params(axis="x", labelsize=self.default_options["xtick_font_size"])
        ax.tick_params(axis="y", labelsize=self.default_options["ytick_font_size"])
        hist = {"n": n, "bins": bins, "patches": patches}
        _mark_render_artists(ax, *patches)
        return fig, ax, hist

    @staticmethod
    def _reject_fig_kwarg(kwargs: dict) -> None:
        """Reject a per-call `fig=` with a clear migration message.

        `fig` is a construction-time binding, not a renderer parameter.
        Because the renderers forward `**kwargs` to matplotlib, an absorbed
        `fig=` would otherwise surface as a confusing downstream error, so
        catch it explicitly here.

        Args:
            kwargs: The renderer's forwarded keyword arguments.

        Raises:
            TypeError: If `fig` is present in `kwargs`.
        """
        if "fig" in kwargs:
            raise TypeError(
                "`fig` is not a parameter of boxplot/multiboxplot/stripes; "
                "bind the figure at construction instead: "
                "StatisticalGlyph(values, fig=...)."
            )

    def _resolve_fig_ax(self, ax: Axes | None) -> Tuple[Figure, Axes]:
        """Return a `(fig, ax)` pair, creating one when none is given.

        These renderers compose into a caller-supplied axes when provided
        (and never call `plt.show()`), so they can be laid out into a
        larger figure. The figure is always derived from the axes —
        `fig` is a construction-time binding (`StatisticalGlyph(..., fig=)`),
        not a per-call parameter.

        Resolution priority: the method's `ax` argument, then the axes
        bound at construction (`self._ax`), then a new axes on the
        figure bound at construction (`self._fig`), otherwise a brand-new
        figure/axes.

        Args:
            ax: An existing axes to draw on, or None.

        Returns:
            Tuple[Figure, Axes]: The figure/axes to render into.
        """
        if ax is None:
            ax = self._ax
        if ax is not None:
            return ax.figure, ax
        if self._fig is not None:
            return self._fig, self._fig.add_subplot(111)
        return plt.subplots(figsize=self.default_options["figsize"])

    def _columns(self) -> List[np.ndarray]:
        """Split the stored values into one 1D array per series.

        Returns:
            List[np.ndarray]: A single-element list for 1D values, or
                one array per column for 2D values.
        """
        values = np.asarray(self.values)
        if values.ndim == 1:
            return [values]
        return [values[:, i] for i in range(values.shape[1])]

    def _apply_axis_labels(self, ax: Axes) -> None:
        """Apply the styled x/y axis labels from default_options to `ax`."""
        opts = self.default_options
        ax.set_xlabel(opts["xlabel"], fontsize=opts["xlabel_font_size"])
        ax.set_ylabel(opts["ylabel"], fontsize=opts["ylabel_font_size"])
        ax.tick_params(axis="x", labelsize=opts["xtick_font_size"])
        ax.tick_params(axis="y", labelsize=opts["ytick_font_size"])

    def boxplot(
        self,
        ax: Axes = None,
        labels: Sequence[str] | None = None,
        notch: bool = False,
        showfliers: bool = True,
        **kwargs,
    ) -> Tuple[Figure, Axes, Dict]:
        """Draw a box-and-whisker plot of the stored values.

        One box is drawn for 1D values; for 2D values one box is drawn
        per column. Boxes are filled with the `color` option (cycled if
        there are more series than colours). Composes into a supplied
        `ax`/`fig` and does not call `plt.show()`.

        Args:
            ax: Axes to draw on. Falls back to the axes/figure bound at
                construction (`StatisticalGlyph(..., ax=/fig=)`), and a
                brand-new figure/axes is created when none is available.
                `fig` is a construction-time binding, not a parameter
                here.
            labels: Tick labels, one per box. Defaults to 1-based
                series indices.
            notch: Draw notched boxes (a rough CI around the median).
                Default is False.
            showfliers: Draw outlier points beyond the whiskers.
                Default is True.
            **kwargs: Forwarded to `Axes.boxplot`.

        Returns:
            Tuple[Figure, Axes, Dict]: The figure, the axes, and the
                dict returned by `Axes.boxplot` (keys: `boxes`,
                `medians`, `whiskers`, ...).

        Examples:
            - One box per column for 2D data:
                ```python
                >>> import numpy as np
                >>> from cleopatra.statistical_glyph import StatisticalGlyph
                >>> np.random.seed(1)
                >>> data = np.random.normal(0, 1, (50, 3))
                >>> stat = StatisticalGlyph(data, color=["r", "g", "b"])
                >>> fig, ax, bp = stat.boxplot()
                >>> len(bp["boxes"])
                3

                ```
        """
        self._reject_fig_kwarg(kwargs)
        fig, ax = self._resolve_fig_ax(ax)
        # See `_clear_prior_render_artists`: a prior `histogram`/`boxplot`/
        # `multiboxplot`/`stripes` call on this Axes (this glyph's own, or
        # a different glyph sharing it) leaves its bars/boxes orphaned
        # unless removed first.
        _clear_prior_render_artists(ax)
        columns = self._columns()
        tick_labels = (
            list(labels)
            if labels is not None
            else [str(i + 1) for i in range(len(columns))]
        )
        # Honour a caller-supplied `positions` (forwarded via **kwargs)
        # so the tick labels below land under the boxes; default is the
        # 1..n that matplotlib's boxplot uses.
        positions = list(kwargs.get("positions", range(1, len(columns) + 1)))
        bp = ax.boxplot(
            columns,
            notch=notch,
            showfliers=showfliers,
            patch_artist=True,
            **kwargs,
        )
        # Set tick labels after the fact rather than via boxplot's label
        # kwarg, whose name differs across matplotlib versions (`labels`
        # before 3.9, `tick_labels` from 3.9). This keeps the floor at
        # matplotlib 3.8.4 (see pyproject) working.
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels)
        palette = self.default_options["color"]
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(palette[i % len(palette)])
            box.set_alpha(self.default_options["alpha"])
        ax.grid(axis="y", alpha=self.default_options["grid_alpha"])
        self._apply_axis_labels(ax)
        _mark_render_artists(ax, *(a for artists in bp.values() for a in artists))
        return fig, ax, bp

    def multiboxplot(
        self,
        positions: Sequence[float] | None = None,
        labels: Sequence[str] | None = None,
        ax: Axes = None,
        widths: float = 0.5,
        **kwargs,
    ) -> Tuple[Figure, Axes, Dict]:
        """Draw grouped boxes at explicit x positions.

        Like `boxplot`, but the boxes are placed at caller-controlled
        `positions` along the x axis (e.g. lead times, months) — the
        usual layout for comparing ensembles side by side.
        Requires 2D values (one column per box).

        Args:
            positions: x positions for the boxes, one per column.
                Defaults to `1..n`.
            labels: Tick labels, one per box. Defaults to the string of
                each position.
            ax: Axes to draw on. Falls back to the axes/figure bound at
                construction, and a brand-new figure/axes is created
                when none is available. `fig` is a construction-time
                binding, not a parameter here.
            widths: Box width in data units. Default is 0.5.
            **kwargs: Forwarded to `Axes.boxplot`.

        Returns:
            Tuple[Figure, Axes, Dict]: The figure, the axes, and the
                `Axes.boxplot` dict.

        Raises:
            ValueError: If the values are not 2D, or if `positions` /
                `labels` length does not match the number of columns.

        Examples:
            - Place three boxes at custom positions:
                ```python
                >>> import numpy as np
                >>> from cleopatra.statistical_glyph import StatisticalGlyph
                >>> np.random.seed(1)
                >>> data = np.random.normal(0, 1, (40, 3))
                >>> stat = StatisticalGlyph(data, color=["r", "g", "b"])
                >>> fig, ax, bp = stat.multiboxplot(positions=[1, 2, 4])
                >>> [int(line.get_xdata().mean()) for line in bp["medians"]]
                [1, 2, 4]

                ```
        """
        self._reject_fig_kwarg(kwargs)
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise ValueError(
                "multiboxplot requires 2D values (one column per box); got "
                f"{values.ndim}D."
            )
        columns = self._columns()
        n = len(columns)
        if positions is None:
            positions = list(range(1, n + 1))
        if len(positions) != n:
            raise ValueError(
                f"positions length ({len(positions)}) must match the number "
                f"of columns ({n})."
            )
        if labels is not None and len(labels) != n:
            raise ValueError(
                f"labels length ({len(labels)}) must match the number of "
                f"columns ({n})."
            )

        fig, ax = self._resolve_fig_ax(ax)
        # See `_clear_prior_render_artists`: a prior `histogram`/`boxplot`/
        # `multiboxplot`/`stripes` call on this Axes (this glyph's own, or
        # a different glyph sharing it) leaves its bars/boxes orphaned
        # unless removed first.
        _clear_prior_render_artists(ax)
        bp = ax.boxplot(
            columns,
            positions=list(positions),
            widths=widths,
            patch_artist=True,
            **kwargs,
        )
        palette = self.default_options["color"]
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(palette[i % len(palette)])
            box.set_alpha(self.default_options["alpha"])
        ax.set_xticks(list(positions))
        ax.set_xticklabels(
            [str(p) for p in positions] if labels is None else list(labels)
        )
        ax.grid(axis="y", alpha=self.default_options["grid_alpha"])
        self._apply_axis_labels(ax)
        _mark_render_artists(ax, *(a for artists in bp.values() for a in artists))
        return fig, ax, bp

    def stripes(
        self,
        ax: Axes = None,
        cmap=None,
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs,
    ) -> Tuple[Figure, Axes, BarContainer]:
        """Draw a warming-stripes band: one colour bar per value.

        Each stored value becomes a full-height vertical stripe coloured
        by `cmap` / the resolved `(vmin, vmax)` normalization — the
        Ed-Hawkins "warming stripes" idiom. Requires 1D values. Composes
        into a supplied `ax`/`fig` and does not call `plt.show()`.

        Args:
            ax: Axes to draw on. Falls back to the axes/figure bound at
                construction, and a brand-new figure/axes is created
                when none is available. `fig` is a construction-time
                binding, not a parameter here.
            cmap: Colormap name or object. Defaults to the `cmap`
                option.
            vmin: Lower colour limit. Defaults to the data minimum.
            vmax: Upper colour limit. Defaults to the data maximum.
            **kwargs: Forwarded to `Axes.bar`.

        Returns:
            Tuple[Figure, Axes, BarContainer]: The figure, the axes, and
                the bar container (one bar per value).

        Raises:
            ValueError: If the values are not 1D.

        Examples:
            - One stripe per yearly value:
                ```python
                >>> import numpy as np
                >>> from cleopatra.statistical_glyph import StatisticalGlyph
                >>> series = np.array([0.1, 0.3, 0.2, 0.6, 0.9, 0.7])
                >>> stat = StatisticalGlyph(series)
                >>> fig, ax, bars = stat.stripes(cmap="coolwarm")
                >>> len(bars)
                6

                ```
        """
        self._reject_fig_kwarg(kwargs)
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1:
            raise ValueError(f"stripes requires 1D values; got {values.ndim}D.")
        fig, ax = self._resolve_fig_ax(ax)
        # See `_clear_prior_render_artists`: a prior `histogram`/`boxplot`/
        # `multiboxplot`/`stripes` call on this Axes (this glyph's own, or
        # a different glyph sharing it) leaves its bars/boxes orphaned
        # unless removed first.
        _clear_prior_render_artists(ax)
        cmap = cmap if cmap is not None else self.default_options["cmap"]
        cmap_obj = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        lo = float(np.nanmin(values)) if vmin is None else vmin
        hi = float(np.nanmax(values)) if vmax is None else vmax
        norm = Normalize(vmin=lo, vmax=hi)
        bar_colors = cmap_obj(norm(values))
        bars = ax.bar(
            np.arange(values.size),
            np.ones(values.size),
            width=1.0,
            color=bar_colors,
            **kwargs,
        )
        ax.set_yticks([])
        ax.set_xlim(-0.5, values.size - 0.5)
        self._apply_axis_labels(ax)
        _mark_render_artists(ax, bars)
        return fig, ax, bars
