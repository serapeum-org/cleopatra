"""
Module: Array.

This module provides a class, `Array`, to handle 3D arrays and perform various operations on them,
such as plotting, animating, and displaying the array.

The `Array` class has the following functionalities:
- Initialize an array object with the provided parameters.
- Plot the array with optional parameters to customize the appearance and display cell values.
- Animate the array over time with optional parameters to customize the animation speed and display points.
- Display the array with optional parameters to customize the appearance and display point IDs.

The `Array` class has the following attributes:
- `arr`: The 3D array to be handled.
- `time`: The time values for animation.
- `points`: The points to be displayed on the array.
- `default_options`: A dictionary to store default options for plotting, animating, and displaying.

The `Array` class has the following methods:
- `plot`: Plot the array with optional parameters.
- `animate`: Animate the array over time with optional parameters.
- `display`: Display the array with optional parameters.
"""

from __future__ import annotations

import warnings
from math import ceil
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from hpc.indexing import get_indices2
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from PIL import Image

from cleopatra.glyph import Glyph
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS
from cleopatra.styles import ColorScale  # re-exported for convenience  # noqa: F401

DEFAULT_OPTIONS = {
    "vmin": None,
    "vmax": None,
    "num_size": 8,
    "display_cell_value": False,
    "background_color_threshold": None,
    "id_color": "green",
    "id_size": 20,
    "precision": 2,
    "kind": "auto",
    "levels": None,
    "robust": False,
    "center": None,
    "extend": None,
    "cbar_kwargs": None,
}
DEFAULT_OPTIONS = STYLE_DEFAULTS | DEFAULT_OPTIONS

#: Tuple of accepted ``kind=`` values for :meth:`ArrayGlyph.plot`.
VALID_PLOT_KINDS = ("auto", "imshow", "pcolormesh", "contour", "contourf")
#: Tuple of accepted values for the xarray-aligned ``extend`` colorbar kwarg.
VALID_EXTEND_VALUES = ("neither", "both", "min", "max")
#: Default colormap auto-selected when ``center`` is set without an explicit ``cmap``.
DIVERGING_DEFAULT_CMAP = "RdBu_r"
#: Lower percentile (2.0) used by xarray-style ``robust=True`` colour limits.
ROBUST_LOWER_PERCENTILE = 2.0
#: Upper percentile (98.0) used by xarray-style ``robust=True`` colour limits.
ROBUST_UPPER_PERCENTILE = 98.0
#: Invariant phrase in the ``ValueError`` raised by :meth:`ArrayGlyph._validate_coords`
#: when a coord array's shape does not match the data array. Kept stable so tests
#: can match against it without coupling to the full (shape-interpolated) message.
_COORD_SHAPE_MISMATCH = "coord array shape does not match the data array"
#: Invariant phrase in the ``ValueError`` raised by :meth:`ArrayGlyph._validate_coords`
#: when a coord array has a non-numeric dtype.
_COORD_DTYPE_MISMATCH = "coord arrays must be numeric (integer or float)"


class FacetGrid:
    """Result object for a multi-subplot facet plot.

    Mirrors xarray's :class:`xarray.plot.facetgrid.FacetGrid` return
    shape so downstream code that already targets xarray can be reused
    without changes. Produced by :meth:`ArrayGlyph.facet`; do not
    construct directly.

    Attributes:
        fig: The shared :class:`matplotlib.figure.Figure`.
        axes: 2-D ``ndarray`` of :class:`matplotlib.axes.Axes`. Empty
            subplot slots (when ``col_wrap`` does not divide the stack
            evenly) are hidden via :meth:`Axes.set_visible`.
        cbar: The shared :class:`matplotlib.colorbar.Colorbar` attached
            to the first rendered subplot. ``None`` when faceting an
            RGB stack (no colorbar in the RGB path).
        name_dicts: List of ``{dim_name: coord_value}`` dicts, one per
            rendered subplot. Mirrors
            :attr:`xarray.plot.facetgrid.FacetGrid.name_dicts` so
            callers can map subplot index to facet coordinate.

    Examples:
        - Inspect the grid shape returned by :meth:`ArrayGlyph.facet`:
            ```python
            >>> import numpy as np
            >>> from cleopatra.array_glyph import ArrayGlyph
            >>> stack = np.arange(4 * 5 * 5, dtype=float).reshape(4, 5, 5)
            >>> g = ArrayGlyph(stack).facet(col="t")
            >>> g.axes.shape
            (1, 4)
            >>> len(g.name_dicts)
            4

            ```
    """

    def __init__(
        self,
        fig: Figure,
        axes: np.ndarray,
        cbar: Colorbar | None,
        name_dicts: list[dict[str, Any]],
    ) -> None:
        """Initialise the :class:`FacetGrid` result object.

        :meth:`ArrayGlyph.facet` is the only intended caller. End users
        receive an already-populated instance and should not invoke
        this constructor directly.

        Args:
            fig: The shared :class:`matplotlib.figure.Figure` that owns
                every subplot.
            axes: 2-D ``ndarray`` of :class:`matplotlib.axes.Axes` with
                shape ``(nrows, ncols)``. Empty slots (when ``col_wrap``
                does not divide the panel count evenly) are kept in the
                array but hidden with :meth:`Axes.set_visible(False)`.
            cbar: The shared :class:`matplotlib.colorbar.Colorbar` for
                the grid, attached to the first rendered subplot;
                ``None`` for an RGB facet that has no colorbar.
            name_dicts: One ``{dim_name: coord_value}`` dict per
                rendered subplot, in row-major (left-to-right,
                top-to-bottom) order, mirroring
                :attr:`xarray.plot.facetgrid.FacetGrid.name_dicts`.

        Examples:
            - The result-object fields line up with the keyword args
                used to construct it:
                ```python
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.array_glyph import FacetGrid
                >>> fig, axes = plt.subplots(1, 2, squeeze=False)
                >>> grid = FacetGrid(
                ...     fig=fig,
                ...     axes=axes,
                ...     cbar=None,
                ...     name_dicts=[{"t": 0}, {"t": 1}],
                ... )
                >>> grid.axes.shape
                (1, 2)
                >>> grid.cbar is None
                True
                >>> [d["t"] for d in grid.name_dicts]
                [0, 1]
                >>> plt.close(fig)

                ```
        """
        self.fig = fig
        self.axes = axes
        self.cbar = cbar
        self.name_dicts = name_dicts


class ArrayGlyph(Glyph):
    """A class to handle arrays and perform various visualization operations on them.

    The ArrayGlyph class provides functionality for visualizing 2D and 3D arrays with
    various customization options. It supports plotting single arrays, RGB arrays,
    and creating animations from 3D arrays.

    Attributes:
        fig (matplotlib.figure.Figure): The matplotlib figure object.
        ax (matplotlib.axes.Axes): The matplotlib axes object.
        extent (List): The extent of the array [xmin, xmax, ymin, ymax].
        rgb (bool): Whether the array is an RGB array.
        num_domain_cells (int): Number of cells in the data domain — cells
            that are neither masked (via ``exclude_value``) nor NaN. For a
            3-D stack this is counted on the first frame. Equals the number
            of per-cell value labels drawn when ``display_cell_value=True``.
            (The legacy alias ``no_elem`` still works but is deprecated.)
        anim (matplotlib.animation.FuncAnimation): The animation object if created.

    Notes:
        This class provides methods for:
        - Plotting arrays with customizable color scales, color bars, and annotations
        - Creating animations from 3D arrays
        - Displaying point values on arrays
        - Customizing plot appearance

    Examples:
    - Create a simple array plot:
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> array_glyph = ArrayGlyph(arr)
        >>> fig, ax = array_glyph.plot()

        ```
    - Create an RGB plot from a 3D array:
    ```python
    >>> rgb_array = np.random.randint(0, 255, size=(3, 10, 10))
    >>> rgb_glyph = ArrayGlyph(rgb_array, rgb=[0, 1, 2])
    >>> fig, ax = rgb_glyph.plot()

    ```
    - Create an animated plot from a 3D array:
    ```python
    >>> time_series = np.random.randint(1, 10, size=(5, 10, 10))
    >>> time_labels = ["Frame 1", "Frame 2", "Frame 3", "Frame 4", "Frame 5"]
    >>> animated_glyph = ArrayGlyph(time_series)
    >>> anim = animated_glyph.animate(time_labels)

    ```
    """

    def __init__(
        self,
        array: np.ndarray,
        exclude_value: list = np.nan,
        extent: list = None,
        coords: tuple[np.ndarray, np.ndarray] | list[np.ndarray] | None = None,
        rgb: list[int] = None,
        surface_reflectance: int = None,
        cutoff: list = None,
        ax: Axes = None,
        fig: Figure = None,
        percentile: int = None,
        **kwargs,
    ):
        """Initialize the ArrayGlyph object with an array and optional parameters.

        Args:
            array: The array to be visualized. Can be a 2D array for single plots or a 3D array for RGB plots or animations.
            exclude_value: Value(s) used to mask cells out of the domain, by default np.nan.
                Can be a single value or a list of values to exclude.
            extent: The extent of the array in the format [xmin, ymin, xmax, ymax], by default None.
                If provided, the array will be plotted with these spatial boundaries.
                Mutually exclusive with ``coords``.
            coords: Optional ``(x, y)`` coordinate arrays for curvilinear
                or non-uniform grids, by default None. Each element is
                either a 1-D array of cell centres (length matches the
                last/second-to-last axis of ``array``) or a 2-D array
                matching the last two axes of ``array``. When set,
                ``kind="auto"`` routes to ``pcolormesh`` instead of
                ``imshow``. Mutually exclusive with ``extent``.
            rgb: The indices of the red, green, and blue bands in the given array, by default None.
                If provided, the array will be treated as an RGB image.
                Can be a list of three values [r, g, b], or four values if alpha band is included [r, g, b, a].
            surface_reflectance: Surface reflectance value for normalizing satellite data, by default None.
                Typically 10000 for Sentinel-2 data.
            cutoff: Clip the range of pixel values for each band, by default None.
                Takes only pixel values from 0 to the value of the cutoff and scales them back to between 0 and 1.
                Should be a list with one value per band.
            ax: A pre-existing axes to plot on, by default None.
                If None, a new axes will be created.
            fig: A pre-existing figure to plot on, by default None.
                If None, a new figure will be created.
            percentile: The percentile value to be used for scaling the array values, by default None.
                Used to enhance contrast by stretching the histogram.
            **kwargs: Additional keyword arguments for customizing the plot.
                Supported arguments include:
                    figsize : tuple, optional
                        Figure size, by default (8, 8).
                    vmin : float, optional
                        Minimum value for color scaling, by default min(array).
                    vmax : float, optional
                        Maximum value for color scaling, by default max(array).
                    title : str, optional
                        Title of the plot, by default 'Array Plot'.
                    title_size : int, optional
                        Title font size, by default 15.
                    cmap : str, optional
                        Colormap name, by default 'coolwarm_r'.
                    kind : str, optional
                        Render kind. One of ``"auto"``, ``"imshow"``,
                        ``"pcolormesh"``, ``"contour"``, ``"contourf"``.
                        Default ``"auto"`` (currently equivalent to
                        ``"imshow"``). Stored on the instance and used
                        as the default for :meth:`plot`.
                    robust : bool, optional
                        When True, ``vmin`` / ``vmax`` are computed from
                        the 2nd and 98th percentile of the unmasked data
                        (xarray-aligned). An explicit ``vmin`` / ``vmax``
                        wins over ``robust``. Default False.
                    center : float, optional
                        Diverging-colormap centring value. When set,
                        ``(vmin, vmax)`` is made symmetric around
                        ``center`` and the cmap auto-switches to
                        ``"RdBu_r"`` if no explicit ``cmap`` was passed.
                        Default None (no centring).
                    levels : int or sequence, optional
                        Discrete colour levels (xarray-aligned). An
                        ``int`` selects N linearly-spaced edges between
                        ``vmin`` and ``vmax``; a sequence is used as
                        explicit edges. Default None.
                    extend : str, optional
                        Colorbar arrow extension. One of ``"neither"``,
                        ``"both"``, ``"min"``, ``"max"``, or None to
                        auto-resolve at render time. Default None.
                    cbar_kwargs : dict, optional
                        Extra keyword arguments forwarded to
                        ``fig.colorbar``; user keys win over cleopatra's
                        defaults on collision. Default None.

        Raises:
            ValueError: If an invalid keyword argument is provided.
            ValueError: If rgb is provided but the array doesn't have enough dimensions.
            ValueError: If ``extend`` is set to a value outside
                ``{"neither", "both", "min", "max"}``.
            ValueError: If both ``extent`` and ``coords`` are supplied,
                or if a ``coords`` element has a shape that does not
                match ``array``.
            TypeError: If ``coords`` is not a length-2 sequence of
                ndarrays.

        Examples:
        Basic initialization with a 2D array:
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> array_glyph = ArrayGlyph(arr)
        >>> fig, ax = array_glyph.plot()

        ```
        Initialization with custom figure size and title:
        ```python
        >>> array_glyph = ArrayGlyph(arr, figsize=(10, 8), title="Custom Array Plot")
        >>> fig, ax = array_glyph.plot()

        ```
        Initialization with RGB bands from a 3D array:
        ```python
        >>> rgb_array = np.random.randint(0, 255, size=(3, 10, 10))
        >>> rgb_glyph = ArrayGlyph(rgb_array, rgb=[0, 1, 2], surface_reflectance=255)
        >>> fig, ax = rgb_glyph.plot()

        ```
        Initialization with custom extent:
        ```python
        >>> array_glyph = ArrayGlyph(arr, extent=[0, 0, 10, 10])
        >>> fig, ax = array_glyph.plot()

        ```
        Robust colour limits (xarray-aligned ``robust=True`` clips the
        2nd/98th percentile so a few outliers do not dominate the
        scale):
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> data = np.arange(100, dtype=float).reshape(10, 10)
        >>> data[0, 0] = 1e6  # outlier
        >>> glyph = ArrayGlyph(data, robust=True)
        >>> round(glyph.vmin, 1), round(glyph.vmax, 1)
        (3.0, 98.0)

        ```
        Centring on a value for diverging data (auto-switches the cmap
        to ``"RdBu_r"`` when no ``cmap`` is passed):
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> anomaly = np.linspace(-3.0, 8.0, 25).reshape(5, 5)
        >>> glyph = ArrayGlyph(anomaly, center=0.0)
        >>> glyph.vmin, glyph.vmax
        (-8.0, 8.0)
        >>> glyph.default_options["cmap"]
        'RdBu_r'

        ```
        Combining ``levels``, ``extend`` and ``cbar_kwargs`` (forwarded
        to :class:`matplotlib.colorbar.Colorbar`):
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> arr = np.arange(25, dtype=float).reshape(5, 5)
        >>> glyph = ArrayGlyph(
        ...     arr,
        ...     levels=5,
        ...     extend="both",
        ...     cbar_kwargs={"shrink": 0.6},
        ... )
        >>> glyph.default_options["levels"]
        5
        >>> glyph.default_options["extend"]
        'both'
        >>> glyph.default_options["cbar_kwargs"]
        {'shrink': 0.6}

        ```
        Invalid ``extend`` is rejected at construction time:
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> ArrayGlyph(np.array([[0.0, 1.0]]), extend="up")
        Traceback (most recent call last):
            ...
        ValueError: Invalid extend='up'. Valid values are ('neither', 'both', 'min', 'max') or None.

        ```
        Curvilinear coords (1-D centres) auto-route to
        ``pcolormesh``:
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> arr = np.arange(12, dtype=float).reshape(3, 4)
        >>> x = np.linspace(0.0, 10.0, 4)
        >>> y = np.linspace(0.0, 5.0, 3)
        >>> glyph = ArrayGlyph(arr, coords=(x, y))
        >>> glyph.coords[0].shape, glyph.coords[1].shape
        ((4,), (3,))

        ```
        ``extent`` and ``coords`` are mutually exclusive:
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> arr = np.zeros((3, 4))
        >>> x = np.linspace(0.0, 10.0, 4)
        >>> y = np.linspace(0.0, 5.0, 3)
        >>> ArrayGlyph(arr, extent=[0, 0, 1, 1], coords=(x, y))
        Traceback (most recent call last):
            ...
        ValueError: `extent` and `coords` are mutually exclusive — pass one or the other.

        ```
        """
        super().__init__(default_options=DEFAULT_OPTIONS, fig=fig, ax=ax, **kwargs)
        # first replace the no_data_value by nan
        # convert the array to float32 to be able to replace the no data value with nan
        if exclude_value is not np.nan:
            if len(exclude_value) > 1:
                mask = np.logical_or(
                    np.isclose(array, exclude_value[0], rtol=0.001),
                    np.isclose(array, exclude_value[1], rtol=0.001),
                )
            else:
                mask = np.isclose(array, exclude_value[0], rtol=0.0000001)
            array = ma.array(array, mask=mask, dtype=array.dtype)
        else:
            array = ma.array(array)

        # convert the extent from [xmin, ymin, xmax, ymax] to [xmin, xmax, ymin, ymax] as required by matplotlib.
        if extent is not None and coords is not None:
            raise ValueError(
                "`extent` and `coords` are mutually exclusive — pass one or the other."
            )
        if extent is not None:
            extent = [extent[0], extent[2], extent[1], extent[3]]
        self.extent = extent

        # Validate and normalise ``coords`` (curvilinear / non-uniform support).
        # Stored as ``self._coords = (x, y)`` or ``None``.
        self._coords = self._validate_coords(coords, array)

        if rgb is not None:
            self.rgb = True
            # prepare to plot rgb plot only if there are three arrays
            if array.shape[0] < 3:
                raise ValueError(
                    f"To plot RGB plot the given array should have only 3 arrays, given array have "
                    f"{array.shape[0]}"
                )
            else:
                array = self.prepare_array(
                    array,
                    rgb=rgb,
                    surface_reflectance=surface_reflectance,
                    cutoff=cutoff,
                    percentile=percentile,
                )
        else:
            self.rgb = False

        self._exclude_value = exclude_value
        # Validate the extend kwarg once at construction time so users get
        # a clear error before the first render call.
        self._validate_extend(self.default_options.get("extend"))

        explicit_keys = set(kwargs.keys())
        self._vmin, self._vmax = self._resolve_color_limits(
            array,
            vmin_kw=kwargs.get("vmin"),
            vmax_kw=kwargs.get("vmax"),
            robust=bool(self.default_options.get("robust", False)),
            center=self.default_options.get("center"),
            vmin_explicit="vmin" in explicit_keys,
            vmax_explicit="vmax" in explicit_keys,
        )
        # Auto-switch the colormap to a diverging default when ``center`` is
        # set and the user did not pick a cmap explicitly. ``cmap`` always
        # exists in ``default_options`` (it has a non-None default of
        # ``"coolwarm_r"``), so the explicit-vs-default check must rely on
        # whether the user actually passed it.
        if (
            self.default_options.get("center") is not None
            and "cmap" not in explicit_keys
        ):
            self.default_options["cmap"] = DIVERGING_DEFAULT_CMAP

        self._arr = array
        # get the tick spacing that has 10 ticks only
        self.ticks_spacing = (self._vmax - self._vmin) / 10
        shape = array.shape
        # Cells in the data domain (not masked, not NaN). Use the same
        # predicate plot/animate use to place per-cell labels so the counts
        # match; ``MaskedArray.count()`` would miss NaN cells.
        first_frame = array[0, :, :] if len(shape) == 3 else array
        self.num_domain_cells = len(get_indices2(first_frame, [np.nan]))

    @property
    def arr(self):
        """array"""
        return self._arr

    @arr.setter
    def arr(self, value):
        self._arr = value

    @property
    def no_elem(self) -> int:
        """Deprecated alias for :attr:`num_domain_cells`.

        Kept for backward compatibility; emits a :class:`DeprecationWarning`.
        Will be removed in a future release.

        Returns:
            int: Same value as :attr:`num_domain_cells`.
        """
        warnings.warn(
            "`ArrayGlyph.no_elem` is deprecated; use `num_domain_cells` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.num_domain_cells

    def prepare_array(
        self,
        array: np.ndarray,
        rgb: list[int] = None,
        surface_reflectance: int = None,
        cutoff: list = None,
        percentile: int = None,
    ) -> np.ndarray:
        """Prepare an array for RGB visualization.

        This method processes a multi-band array to create an RGB image suitable for visualization.
        It can normalize the data using either percentile-based scaling or surface reflectance values.

        Args:
            array: The input array containing multiple bands. For RGB visualization,
                this should be a 3D array where the first dimension represents the bands.
            rgb: The indices of the red, green, and blue bands in the given array, by default None.
                If None, assumes the order is [3, 2, 1] (common for Sentinel-2 data).
            surface_reflectance: Surface reflectance value for normalizing satellite data, by default None.
                Typically 10000 for Sentinel-2 data or 255 for 8-bit imagery.
                Used to scale values to the range [0, 1].
            cutoff: Clip the range of pixel values for each band, by default None.
                Takes only pixel values from 0 to the value of the cutoff and scales them back to between 0 and 1.
                Should be a list with one value per band.
            percentile: The percentile value to be used for scaling the array values, by default None.
                Used to enhance contrast by stretching the histogram.
                If provided, this takes precedence over surface_reflectance.

        Returns:
            np.ndarray: The prepared array with shape (height, width, 3) suitable for RGB visualization.
                Values are normalized to the range [0, 1].
                the rgb 3d array is converted into 2d array to be plotted using the plt.imshow function.
                a float32 array normalized between 0 and 1 using the `percentile` values or the `surface_reflectance`.
                if the `percentile` or `surface_reflectance` values are not given, the function just reorders the values
                to have the red-green-blue order.

        Raises:
            ValueError: If the array shape is incompatible with the provided RGB indices.

        Notes:
            - The `prepare_array` function is called in the constructor of the `ArrayGlyph` class to prepare the array,
              so you can provide the same parameters of the `prepare_array` function to the `ArrayGlyph constructor`.
            - The prepare function moves the first axes (the channel axis) to the last axes, and then scales the array
              using the percentile values. If the percentile is not given, the function scales the array using the
              surface reflectance values. If the surface reflectance is not given, the function scales the array using
              the cutoff values. If the cutoff is not given, the function scales the array using the sentinel data

        Examples:
        Prepare an array using percentile-based scaling:
            ```python
            >>> import numpy as np
            >>> from cleopatra.array_glyph import ArrayGlyph
            >>> # Create a 3-band array (e.g., satellite image)
            >>> bands = np.random.randint(0, 10000, size=(3, 100, 100))
            >>> glyph = ArrayGlyph(np.zeros((1, 1)))  # Dummy initialization
            >>> rgb_array = glyph.prepare_array(bands, rgb=[0, 1, 2], percentile=2)
            >>> rgb_array.shape
            (100, 100, 3)
            >>> np.all((0 <= rgb_array) & (rgb_array <= 1))
            np.True_

            ```
        Prepare an array using surface reflectance normalization:
            ```python
            >>> rgb_array = glyph.prepare_array(bands, rgb=[0, 1, 2], surface_reflectance=10000)
            >>> rgb_array.shape
            (100, 100, 3)
            >>> np.all((0 <= rgb_array) & (rgb_array <= 1))
            np.True_

            ```
        Prepare an array with cutoff values:
            ```python
            >>> rgb_array = glyph.prepare_array(
            ...     bands, rgb=[0, 1, 2], surface_reflectance=10000, cutoff=[5000, 5000, 5000]
            ... )
            >>> rgb_array.shape
            (100, 100, 3)
            >>> np.all((0 <= rgb_array) & (rgb_array <= 1))
            np.True_

            ```

        - Create an array and instantiate the `ArrayGlyph` class.
            ```python
            >>> import numpy as np
            >>> arr = np.random.randint(0, 255, size=(3, 5, 5)).astype(np.float32)
            >>> array_glyph = ArrayGlyph(arr)
            >>> print(array_glyph.arr.shape)
            (3, 5, 5)

            ```
        `rgb` channels:
            - Now let's use the `prepare_array` function with `rgb` channels as [0, 1, 2]. so the finction does not to
                reorder the chennels. but it just needs to move the first axis to the last axis.
                ```python
                >>> rgb_array = array_glyph.prepare_array(arr, rgb=[0, 1, 2])
                >>> print(rgb_array.shape)
                (5, 5, 3)

                ```
            - If we compare the values of the first channel in the original array with the first array in the rgb array it
                should be the same.
                ```python
                >>> np.testing.assert_equal(arr[0, :, :],rgb_array[:, :, 0])

                ```
        surface_reflectance:
            - if you provide the surface reflectance value, the function will scale the array using the surface reflectance
                value to a normalized rgb values.
                ```python
                >>> array_glyph = ArrayGlyph(arr)
                >>> rgb_array = array_glyph.prepare_array(arr, surface_reflectance=10000, rgb=[0, 1, 2])
                >>> print(rgb_array.shape)
                (5, 5, 3)

                ```
            - if you print the values of the first channel, you will find all the values are between 0 and 1.
                ```python
                >>> print(rgb_array[:, :, 0]) # doctest: +SKIP
                [[0.0195 0.02   0.0109 0.0211 0.0087]
                 [0.0112 0.0221 0.0035 0.0234 0.0141]
                 [0.0116 0.0188 0.0001 0.0176 0.    ]
                 [0.0014 0.0147 0.0043 0.0167 0.0117]
                 [0.0083 0.0139 0.0186 0.02   0.0058]]

                ```
            - With the `surface_reflectance` parameter, you can also use the `cutoff` parameter to affect values that
                are above it, by rescaling them.
                ```python
                >>> rgb_array = array_glyph.prepare_array(
                ...     arr, surface_reflectance=10000, rgb=[0, 1, 2], cutoff=[0.8, 0.8, 0.8]
                ... )
                >>> print(rgb_array[:, :, 0]) # doctest: +SKIP
                [[0.     0.     0.     0.     0.    ]
                 [1.     1.     1.     1.     1.    ]
                 [1.     1.     1.     1.     1.    ]
                 [0.0014 0.0147 0.0043 0.0167 0.0117]
                 [0.0083 0.0139 0.0186 0.02   0.0058]]

                ```
        """
        # take the rgb arrays and reorder them to have the red-green-blue, if the order is not given, assume the
        # order as sentinel data. [3, 2, 1]
        array = array[rgb].transpose(1, 2, 0)

        if percentile is not None:
            array = self.scale_percentile(array, percentile=percentile)
        elif surface_reflectance is not None:
            array = self._prepare_sentinel_rgb(
                array,
                rgb=rgb,
                surface_reflectance=surface_reflectance,
                cutoff=cutoff,
            )
        return array

    def _prepare_sentinel_rgb(
        self,
        array: np.ndarray,
        rgb: list[int] = None,
        surface_reflectance: int = 10000,
        cutoff: list = None,
    ) -> np.ndarray:
        """Prepare Sentinel satellite data for RGB visualization.

        This method specifically handles Sentinel satellite imagery by normalizing the data
        using the provided surface reflectance value and optional cutoff values.

        Args:
            array: The input array with shape (height, width, 3) containing RGB bands.
                This array should already be transposed from the original band-first format.
            rgb: The indices of the red, green, and blue bands in the original array, by default None.
                Used only for cutoff application.
            surface_reflectance: Surface reflectance value for normalizing satellite data, by default 10000.
                Sentinel-2 data typically uses 10000 as the maximum reflectance value.
                Used to scale values to the range [0, 1].
            cutoff: Clip the range of pixel values for each band, by default None.
                Takes only pixel values from 0 to the value of the cutoff and scales them back to between 0 and 1.
                Should be a list with one value per band.

        Returns:
            np.ndarray: The prepared array with shape (height, width, 3) suitable for RGB visualization.
                Values are normalized to the range [0, 1].

        Examples:
        Prepare Sentinel-2 data with default surface reflectance:
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> # Create a simulated Sentinel-2 RGB array
        >>> rgb_data = np.random.randint(0, 10000, size=(100, 100, 3))
        >>> glyph = ArrayGlyph(np.zeros((1, 1)))  # Dummy initialization
        >>> normalized = glyph._prepare_sentinel_rgb(rgb_data)
        >>> np.all((0 <= normalized) & (normalized <= 1))
        np.True_

        ```
        Prepare Sentinel-2 data with custom cutoff values:
        ```python
        >>> cutoffs = [8000, 7000, 9000]
        >>> normalized = glyph._prepare_sentinel_rgb(rgb_data, rgb=[0, 1, 2], cutoff=cutoffs)
        >>> np.all((0 <= normalized) & (normalized <= 1))
        np.True_

        ```
        """
        array = np.clip(array / surface_reflectance, 0, 1)
        if cutoff is not None:
            array[0] = np.clip(rgb[0], 0, cutoff[0]) / cutoff[0]
            array[1] = np.clip(rgb[1], 0, cutoff[1]) / cutoff[1]
            array[2] = np.clip(rgb[2], 0, cutoff[2]) / cutoff[2]

        return array

    @staticmethod
    def scale_percentile(arr: np.ndarray, percentile: int = 1) -> np.ndarray:
        """Scale an array using percentile-based contrast stretching.

        This method enhances the contrast of an image by stretching the histogram
        based on percentile values. It calculates the lower and upper percentile values
        for each band and normalizes the data to the range [0, 1].

        Args:
            arr: The array to be scaled, with shape (height, width, bands).
                Typically an RGB image with 3 bands.
            percentile: The percentile value to be used for scaling, by default 1.
                This value determines how much of the histogram tails to exclude.
                Higher values result in more contrast stretching.
                Typical values range from 1 to 5.

        Returns:
            np.ndarray: The scaled array, normalized between 0 and 1, with the same shape as input.
                Data type is float32.

        Notes:
            The method works by:
            1. Computing the lower percentile value for each band
            2. Computing the upper percentile value (100 - percentile) for each band
            3. Normalizing each band using these percentile values
            4. Clipping values to the range [0, 1]

            This is particularly useful for visualizing satellite imagery with high dynamic range.

        Examples:
        Scale a single-band array:
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> # Create a test array with values between 0 and 10000
        >>> test_array = np.random.randint(0, 10000, size=(100, 100, 1))
        >>> scaled = ArrayGlyph.scale_percentile(test_array, percentile=2)
        >>> scaled.shape
        (100, 100, 1)
        >>> np.all((0 <= scaled) & (scaled <= 1))
        np.True_

        ```
        Scale an RGB array:
        ```python
        >>> rgb_array = np.random.randint(0, 10000, size=(100, 100, 3))
        >>> scaled = ArrayGlyph.scale_percentile(rgb_array, percentile=2)
        >>> scaled.shape
        (100, 100, 3)
        >>> np.all((0 <= scaled) & (scaled <= 1))
        np.True_

        ```
        Using different percentile values affects contrast:
        ```python
        >>> low_contrast = ArrayGlyph.scale_percentile(rgb_array, percentile=1)
        >>> high_contrast = ArrayGlyph.scale_percentile(rgb_array, percentile=5)
        >>> # Higher percentile typically results in higher contrast

        ```
        """
        rows, columns, bands = arr.shape
        # flatten image.
        arr = np.reshape(arr, [rows * columns, bands]).astype(np.float32)
        # lower percentile values (one value for each band).
        lower_percent = np.percentile(arr, percentile, axis=0)
        # 98 percentile values.
        upper_percent = np.percentile(arr, 100 - percentile, axis=0) - lower_percent
        # normalize the 3 bands using the percentile values for each band.
        arr = (arr - lower_percent[None, :]) / upper_percent[None, :]
        arr = np.reshape(arr, [rows, columns, bands])
        # discard outliers.
        arr = arr.clip(0, 1)

        return arr

    def __str__(self):
        """String representation of the Array object."""
        message = f"""
                    Min: {self.vmin}
                    Max: {self.vmax}
                    Exclude values: {self.exclude_value}
                    RGB: {self.rgb}
                """
        return message

    @property
    def exclude_value(self):
        """exclude_value"""
        return self._exclude_value

    @exclude_value.setter
    def exclude_value(self, value):
        self._exclude_value = value

    @staticmethod
    def _validate_coords(
        coords: tuple[np.ndarray, np.ndarray] | list[np.ndarray] | None,
        array: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Validate the ``coords`` kwarg and return a normalised ``(x, y)`` tuple.

        Args:
            coords: User-provided coordinates. ``None`` (no curvilinear
                support), or a length-2 tuple/list of arrays. Each
                element is either 1-D (length matches the last axis of
                ``array`` for ``x`` and the second-to-last for ``y``)
                or 2-D with shape ``array.shape[-2:]``.
            array: The data array used to validate coordinate shapes.

        Returns:
            tuple[np.ndarray, np.ndarray] or None: The validated
                ``(x, y)`` pair, with each element cast to ``np.ndarray``.

        Raises:
            TypeError: If ``coords`` is not ``None`` and not a length-2
                sequence.
            ValueError: If a coordinate array has a shape that does not
                match the data array, or a non-numeric dtype (bool,
                complex, object, …).

        Examples:
            - ``None`` short-circuits to ``None``:
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> ArrayGlyph._validate_coords(None, np.zeros((3, 4))) is None
                True

                ```
            - 1-D centres matching ``array.shape[-1]`` (x) and
                ``array.shape[-2]`` (y) are accepted:
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> arr = np.zeros((3, 4))
                >>> x = np.array([0.0, 1.0, 2.0, 3.0])
                >>> y = np.array([0.0, 1.0, 2.0])
                >>> xs, ys = ArrayGlyph._validate_coords((x, y), arr)
                >>> xs.shape, ys.shape
                ((4,), (3,))

                ```
            - A non-tuple raises :class:`TypeError`:
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> ArrayGlyph._validate_coords("oops", np.zeros((3, 4)))
                Traceback (most recent call last):
                    ...
                TypeError: `coords` must be a length-2 sequence of arrays (x, y), got str.

                ```
        """
        if coords is None:
            result = None
        else:
            if not isinstance(coords, (tuple, list)) or len(coords) != 2:
                raise TypeError(
                    "`coords` must be a length-2 sequence of arrays "
                    f"(x, y), got {type(coords).__name__}."
                )
            x_in, y_in = coords
            x_arr = np.asarray(x_in)
            y_arr = np.asarray(y_in)
            for name, arr_ in (("x", x_arr), ("y", y_arr)):
                if not (
                    np.issubdtype(arr_.dtype, np.integer)
                    or np.issubdtype(arr_.dtype, np.floating)
                ):
                    raise ValueError(
                        f"{name}: {_COORD_DTYPE_MISMATCH}; got dtype "
                        f"{arr_.dtype}."
                    )
            data_shape = array.shape[-2:]
            rows, cols = data_shape
            x_ok = (x_arr.ndim == 1 and x_arr.shape[0] == cols) or (
                x_arr.ndim == 2 and x_arr.shape == data_shape
            )
            y_ok = (y_arr.ndim == 1 and y_arr.shape[0] == rows) or (
                y_arr.ndim == 2 and y_arr.shape == data_shape
            )
            if not x_ok:
                raise ValueError(
                    f"x {_COORD_SHAPE_MISMATCH}: got shape {x_arr.shape}, "
                    f"expected 1-D length {cols} or 2-D {data_shape}."
                )
            if not y_ok:
                raise ValueError(
                    f"y {_COORD_SHAPE_MISMATCH}: got shape {y_arr.shape}, "
                    f"expected 1-D length {rows} or 2-D {data_shape}."
                )
            result = (x_arr, y_arr)
        return result

    @property
    def coords(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Optional ``(x, y)`` coordinate arrays for curvilinear grids.

        Returns the validated coordinate pair stored at construction
        time, or ``None`` when the glyph was built without ``coords``
        (regular pixel-grid render). When non-``None``, ``plot(kind=
        "auto")`` routes to ``pcolormesh`` so the (x, y) arrays are
        honoured.

        Returns:
            tuple[np.ndarray, np.ndarray] or None: The ``(x, y)`` pair
                as stored on the instance (each cast to
                :class:`numpy.ndarray`), or ``None``.

        Examples:
            - A glyph built without ``coords`` reports ``None``:
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> glyph = ArrayGlyph(np.zeros((3, 4)))
                >>> glyph.coords is None
                True

                ```
            - A glyph built with 1-D centres exposes the validated
                arrays back through the property:
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> arr = np.zeros((3, 4))
                >>> x = np.linspace(0.0, 3.0, 4)
                >>> y = np.linspace(0.0, 2.0, 3)
                >>> glyph = ArrayGlyph(arr, coords=(x, y))
                >>> xs, ys = glyph.coords
                >>> xs.shape, ys.shape
                ((4,), (3,))
                >>> float(xs[-1]), float(ys[-1])
                (3.0, 2.0)

                ```
        """
        return self._coords

    @staticmethod
    def _validate_extend(extend: str | None) -> None:
        """Validate the ``extend`` kwarg against the allowed values.

        Args:
            extend: User-provided value for the colorbar extension. May
                be ``None`` (auto-resolve at render time) or one of
                ``"neither"``, ``"both"``, ``"min"``, ``"max"``.

        Raises:
            ValueError: When ``extend`` is not one of the accepted
                strings (or ``None``).

        Examples:
            - Accepted values return ``None`` silently:
                ```python
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> ArrayGlyph._validate_extend("both") is None
                True
                >>> ArrayGlyph._validate_extend(None) is None
                True

                ```
            - Unsupported values raise :class:`ValueError`:
                ```python
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> ArrayGlyph._validate_extend("up")
                Traceback (most recent call last):
                    ...
                ValueError: Invalid extend='up'. Valid values are ('neither', 'both', 'min', 'max') or None.

                ```
        """
        if extend is None:
            return
        if extend not in VALID_EXTEND_VALUES:
            raise ValueError(
                f"Invalid extend={extend!r}. Valid values are "
                f"{VALID_EXTEND_VALUES} or None."
            )

    @staticmethod
    def _robust_limits(arr: np.ndarray) -> tuple[float, float]:
        """Compute xarray-style robust ``(vmin, vmax)`` from the data.

        Returns the 2nd and 98th percentile of the unmasked, finite
        values in ``arr`` — the same convention as xarray's
        ``robust=True``. Masked entries and NaNs are excluded from the
        percentile computation.

        Args:
            arr: Input array. May be a plain ndarray or a masked array.

        Returns:
            tuple[float, float]: ``(vmin_robust, vmax_robust)``.

        Raises:
            ValueError: If the array contains no finite values.

        Examples:
            - Outliers are clipped to the 2nd/98th percentile:
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> arr = np.arange(100, dtype=float)
                >>> arr[0] = -1e6  # extreme low outlier
                >>> arr[-1] = 1e6  # extreme high outlier
                >>> vmin, vmax = ArrayGlyph._robust_limits(arr)
                >>> round(vmin, 1), round(vmax, 1)
                (2.0, 97.0)

                ```
            - Masked and NaN entries are excluded from the percentile
                computation:
                ```python
                >>> import numpy as np
                >>> import numpy.ma as ma
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> raw = np.array([np.nan, 0.0, 1.0, 2.0, 3.0, 4.0])
                >>> arr = ma.array(raw, mask=[True, False, False, False, False, False])
                >>> vmin, vmax = ArrayGlyph._robust_limits(arr)
                >>> round(vmin, 2), round(vmax, 2)
                (0.08, 3.92)

                ```
        """
        if isinstance(arr, ma.MaskedArray):
            values = arr.compressed()
        else:
            values = np.asarray(arr).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(
                "Cannot compute robust vmin/vmax: array has no finite "
                "values."
            )
        vmin_robust = float(
            np.nanpercentile(values, ROBUST_LOWER_PERCENTILE)
        )
        vmax_robust = float(
            np.nanpercentile(values, ROBUST_UPPER_PERCENTILE)
        )
        return vmin_robust, vmax_robust

    @staticmethod
    def _center_limits(
        vmin: float, vmax: float, center: float
    ) -> tuple[float, float]:
        """Make ``(vmin, vmax)`` symmetric around ``center``.

        Implements xarray's diverging-cmap centring: the larger of
        ``|vmin - center|`` and ``|vmax - center|`` becomes the half-
        range, and the result is ``(center - half, center + half)``.

        Args:
            vmin: Lower colour limit before symmetrisation.
            vmax: Upper colour limit before symmetrisation.
            center: Value to centre the diverging colormap on.

        Returns:
            tuple[float, float]: Symmetric ``(vmin, vmax)`` around
                ``center``.

        Examples:
            - Centring around zero expands the smaller side to match
                the larger one:
                ```python
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> ArrayGlyph._center_limits(-3.0, 8.0, 0.0)
                (-8.0, 8.0)

                ```
            - Centring around a non-zero value (e.g. an anomaly base
                of 5.0):
                ```python
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> low, high = ArrayGlyph._center_limits(2.0, 12.0, 5.0)
                >>> low, high
                (-2.0, 12.0)
                >>> (low + high) / 2  # centred on 5.0
                5.0

                ```
        """
        half = max(abs(vmin - center), abs(vmax - center))
        return center - half, center + half

    def _resolve_color_limits(
        self,
        arr: np.ndarray,
        vmin_kw: float | None,
        vmax_kw: float | None,
        robust: bool,
        center: float | None,
        vmin_explicit: bool,
        vmax_explicit: bool,
    ) -> tuple[float, float]:
        """Resolve final ``(vmin, vmax)`` for colour scaling.

        Resolution order matches xarray:

        1. Start from robust (2nd/98th percentile) limits when
           ``robust=True``, else from the full data range.
        2. Override either end with an explicit ``vmin`` / ``vmax`` if
           the user provided one.
        3. If ``center`` is set, symmetrise around it.

        Args:
            arr: Data array (plain or masked).
            vmin_kw: Value of ``vmin`` from the caller's kwargs, or
                ``None`` if not supplied.
            vmax_kw: Value of ``vmax`` from the caller's kwargs, or
                ``None`` if not supplied.
            robust: Whether to use the 2nd/98th percentile range.
            center: Value to centre a diverging colormap on. ``None``
                disables symmetrisation.
            vmin_explicit: Whether the caller explicitly passed
                ``vmin`` (even if its value was ``None``).
            vmax_explicit: Whether the caller explicitly passed
                ``vmax``.

        Returns:
            tuple[float, float]: Final ``(vmin, vmax)``.

        Raises:
            ValueError: If the resolved limits are not finite — e.g. the
                array has no finite values (all NaN / fully masked) and no
                explicit ``vmin`` / ``vmax`` was supplied to fall back on.

        Examples:
            - Default path: full data range, no robust clipping, no
                centring:
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> data = np.arange(25, dtype=float).reshape(5, 5)
                >>> glyph = ArrayGlyph(data)
                >>> glyph._resolve_color_limits(
                ...     data,
                ...     vmin_kw=None,
                ...     vmax_kw=None,
                ...     robust=False,
                ...     center=None,
                ...     vmin_explicit=False,
                ...     vmax_explicit=False,
                ... )
                (0.0, 24.0)

                ```
            - Explicit ``vmax`` overrides the data-driven upper limit
                and ``center`` then symmetrises around the centre:
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> data = np.arange(25, dtype=float).reshape(5, 5)
                >>> glyph = ArrayGlyph(data)
                >>> glyph._resolve_color_limits(
                ...     data,
                ...     vmin_kw=None,
                ...     vmax_kw=10.0,
                ...     robust=False,
                ...     center=0.0,
                ...     vmin_explicit=False,
                ...     vmax_explicit=True,
                ... )
                (-10.0, 10.0)

                ```
        """
        if robust:
            vmin_base, vmax_base = self._robust_limits(arr)
        else:
            # nanmin/nanmax on an all-NaN / fully-masked array return NaN
            # (with a RuntimeWarning). Compute quietly and validate the
            # resolved limits below — an unusable colour range should fail
            # loudly here, not later inside ``get_ticks()``/matplotlib.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                vmin_base = np.nanmin(arr)
                vmax_base = np.nanmax(arr)

        vmin_final = (
            vmin_kw if vmin_explicit and vmin_kw is not None else vmin_base
        )
        vmax_final = (
            vmax_kw if vmax_explicit and vmax_kw is not None else vmax_base
        )

        if not (np.isfinite(vmin_final) and np.isfinite(vmax_final)):
            raise ValueError(
                "Cannot determine vmin/vmax: the array has no finite "
                "values. Pass explicit vmin and vmax, or filter the array "
                "first."
            )

        if center is not None:
            vmin_final, vmax_final = self._center_limits(
                vmin_final, vmax_final, center
            )

        return float(vmin_final), float(vmax_final)

    def _plot_im_get_cbar_kw(
        self,
        ax: Axes,
        arr: np.ndarray,
        ticks: np.ndarray,
        kind: str = "imshow",
    ) -> tuple[Any, dict[str, str]]:
        """Render the array on ``ax`` and return the artist plus cbar kwargs.

        Builds the matplotlib norm from ``default_options["color_scale"]``
        and dispatches to the requested ``kind`` of plot. All four kinds
        share the same norm/vmin/vmax resolution path so the existing
        ``color_scale`` enum (linear/power/sym-lognorm/boundary-norm/
        midpoint) works identically for every render kind.

        When ``self._coords`` is set (curvilinear / non-uniform grid),
        the ``(x, y)`` arrays are forwarded as the first positional
        args to ``pcolormesh`` / ``contour`` / ``contourf``. ``kind=
        "imshow"`` is incompatible with ``coords`` and raises
        :class:`ValueError` — callers should use ``kind="auto"`` or
        ``kind="pcolormesh"`` instead.

        Args:
            ax: matplotlib figure axes.
            arr: numpy (masked) array.
            ticks: color bar ticks.
            kind: render kind. One of ``"imshow"``, ``"pcolormesh"``,
                ``"contour"``, ``"contourf"``. Default is ``"imshow"``
                (preserves the historical animate/legacy call path).

        Returns:
            tuple: ``(artist, cbar_kw)`` where ``artist`` is the
                matplotlib mappable (``AxesImage`` for ``imshow``,
                ``QuadMesh`` for ``pcolormesh``, ``QuadContourSet`` for
                contour/contourf) and ``cbar_kw`` is the colorbar
                keyword-argument dict.

        Raises:
            ValueError: If ``kind`` is ``"imshow"`` while ``self._coords``
                is set (incompatible combination), or if ``kind`` is not
                one of the recognised values in :data:`VALID_PLOT_KINDS`.
        """
        norm, cbar_kw = self._create_norm_and_cbar_kw(ticks)
        cmap = self.default_options["cmap"]
        vmin = ticks[0]
        vmax = ticks[-1]

        # midpoint normalization needs unmasked NaN-filled data; the
        # other kinds can handle a masked array directly but contour /
        # contourf misbehave on masked arrays as well, so fill there too.
        plot_arr = arr
        if self.default_options["color_scale"].lower() == "midpoint":
            plot_arr = arr.filled(np.nan)

        levels = self.default_options.get("levels")

        coords = self._coords

        if kind == "imshow":
            if coords is not None:
                raise ValueError(
                    "`coords` requires kind='pcolormesh' or 'auto'."
                )
            if norm is None:
                im = ax.matshow(
                    plot_arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=self.extent
                )
            else:
                im = ax.matshow(plot_arr, cmap=cmap, norm=norm, extent=self.extent)
        elif kind == "pcolormesh":
            pcm_args = (coords[0], coords[1], plot_arr) if coords is not None else (plot_arr,)
            if norm is None:
                im = ax.pcolormesh(
                    *pcm_args,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    shading="auto",
                )
            else:
                im = ax.pcolormesh(
                    *pcm_args, cmap=cmap, norm=norm, shading="auto"
                )
        elif kind in ("contour", "contourf"):
            # contour/contourf cannot consume masked arrays cleanly;
            # convert to a NaN-filled view if we're holding a mask.
            if isinstance(plot_arr, ma.MaskedArray):
                plot_arr = plot_arr.filled(np.nan)
            plot_fn = ax.contour if kind == "contour" else ax.contourf
            contour_kwargs = {"cmap": cmap}
            if norm is None:
                contour_kwargs["vmin"] = vmin
                contour_kwargs["vmax"] = vmax
            else:
                contour_kwargs["norm"] = norm
            # Pass the resolved level edges when ``levels`` is set so the
            # contour rings line up with the colorbar boundaries computed
            # in ``_create_norm_and_cbar_kw``.
            level_edges = self._levels_to_bounds(levels, vmin, vmax)
            # When curvilinear coords are present forward them as the
            # first two positional args (matplotlib contour signature is
            # ``contour([X, Y,] Z, [levels], **kwargs)``).
            if coords is not None:
                base_args = (coords[0], coords[1], plot_arr)
            else:
                base_args = (plot_arr,)
            if level_edges is not None:
                im = plot_fn(*base_args, level_edges, **contour_kwargs)
            else:
                im = plot_fn(*base_args, **contour_kwargs)
        else:
            raise ValueError(
                f"Invalid kind={kind!r}. Valid kinds are {VALID_PLOT_KINDS}."
            )

        return im, cbar_kw

    def apply_colormap(self, cmap: Colormap | str) -> np.ndarray:
        """Apply a matplotlib colormap to an array.

            Create an RGB channel from the given array using the given colormap.

        Args:
            cmap: colormap.

        Returns:
            np.ndarray: 8-bit array with the colormap applied.

        Examples:
        - Create an array and instantiate the `Array` object:
        ```python
        >>> import numpy as np
        >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> array = ArrayGlyph(arr)
        >>> rgb_array = array.apply_colormap("coolwarm_r")
        >>> print(rgb_array) # doctest: +SKIP
        [[[179   3  38]
          [221  96  76]
          [244 154 123]]
         [[244 196 173]
          [220 220 221]
          [183 207 249]]
         [[139 174 253]
          [ 96 128 232]
          [ 58  76 192]]]

        >>> print(rgb_array.dtype)
        uint8

        ```
        """
        colormap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        normed_data = (self.arr - self.arr.min()) / (self.arr.max() - self.arr.min())
        colored = colormap(normed_data)
        return (colored[:, :, :3] * 255).astype("uint8")

    def to_image(self, arr: np.ndarray = None) -> Image.Image:
        """Create an RGB image from an array.

            convert the array to an image.

        Args:
            arr: array. if None, the array in the object will be used.

        Examples:
        ```python
        >>> import numpy as np
        >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> array = ArrayGlyph(arr)
        >>> image = array.to_image()
        >>> print(image) # doctest: +SKIP
        <PIL.Image.Image image mode=RGB size=3x3 at 0x7F5E0D2F4C40>

        ```
        """
        if arr is None:
            arr = self.arr
        # This is done to scale the values between 0 and 255
        arr = arr if arr.dtype == "uint8" else self.scale_to_rgb()
        return Image.fromarray(arr).convert("RGB")

    def scale_to_rgb(self, arr: np.ndarray = None) -> np.ndarray:
        """Create an RGB image.

        Args:
            arr: array. if None, the array in the object will be used.

        Examples:
        ```python
        >>> import numpy as np
        >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> array = ArrayGlyph(arr)
        >>> rgb_array = array.scale_to_rgb()
        >>> print(rgb_array)
        [[28 56 85]
         [113 141 170]
         [198 226 255]]
        >>> print(rgb_array.dtype)
        uint8

        ```
        """
        if arr is None:
            arr = self.arr
        # This is done to scale the values between 0 and 255
        return (arr * 255 / arr.max()).astype("uint8")

    @staticmethod
    def _plot_text(
        ax: Axes, arr: np.ndarray, indices, default_options_dict: dict
    ) -> list:
        """plot values as a text in each cell.

        Args:
            ax: matplotlib axes.
            arr: numpy array.
            indices: array with columns, (row, col).
            default_options_dict: default options dictionary after updating the options.

        Returns:
            list: list of the text object.
        """
        # https://github.com/serapeum-org/cleopatra/issues/75
        # add text for the cell values
        add_text = lambda elem: ax.text(
            elem[1],
            elem[0],
            np.round(arr[elem[0], elem[1]], 2),
            ha="center",
            va="center",
            color="w",
            fontsize=default_options_dict["num_size"],
        )
        return list(map(add_text, indices))

    def plot(
        self,
        points: np.ndarray = None,
        point_color: str = "red",
        point_size: int | float = 100,
        pid_color: str = "blue",
        pid_size: int | float = 10,
        kind: str = "auto",
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """Plot the array with customizable visualization options.

        This method creates a visualization of the array with various customization options
        including color scales, color bars, cell value display, and point annotations.
        It supports both regular arrays and RGB arrays.

        Args:
            points: Points to display on the array, by default None.
                Should be a 3-column array where:
                - First column: values to display for each point
                - Second column: row indices of the points in the array
                - Third column: column indices of the points in the array
            point_color: Color of the points, by default "red".
                Any valid matplotlib color string.
            point_size: Size of the points, by default 100.
                Controls the marker size.
            pid_color: Color of the point value annotations, by default "blue".
                Any valid matplotlib color string.
            pid_size: Size of the point value annotations, by default 10.
                Controls the font size of the annotations.
            kind: Render kind, by default ``"auto"``. One of:

                - ``"auto"`` — picks the best renderer for the data.
                  Routes to ``"pcolormesh"`` when curvilinear /
                  non-uniform ``coords`` were passed to the
                  constructor, otherwise falls back to ``"imshow"``.
                - ``"imshow"`` — pixel-grid raster render via
                  ``ax.imshow``/``matshow``. Honours ``extent``.
                  Incompatible with ``coords``.
                - ``"pcolormesh"`` — quadrilateral mesh render via
                  ``ax.pcolormesh`` with ``shading="auto"``. Honours
                  ``coords`` (1-D centres or 2-D curvilinear).
                - ``"contour"`` — line contours via ``ax.contour``.
                  Honours ``levels`` from kwargs when set.
                - ``"contourf"`` — filled contours via ``ax.contourf``.
                  Honours ``levels`` from kwargs when set.

                Cell-value display and point overlays only apply to
                ``"imshow"`` and ``"pcolormesh"``; they are silently
                skipped for ``"contour"`` and ``"contourf"`` (which have
                no per-cell grid). RGB compositing requires
                ``kind="imshow"``.
            **kwargs: Additional keyword arguments for customizing the plot.

                Plot appearance:
                    title : str, optional
                        Title of the plot, by default 'Array Plot'.
                    title_size : int, optional
                        Title font size, by default 15.
                    cmap : str, optional
                        Colormap name, by default 'coolwarm_r'.
                    vmin : float, optional
                        Minimum value for color scaling, by default min(array).
                    vmax : float, optional
                        Maximum value for color scaling, by default max(array).

                Color bar options:
                    cbar_orientation : str, optional
                        Orientation of the color bar, by default 'vertical'.
                        Can be 'horizontal' or 'vertical'.
                    cbar_label_rotation : float, optional
                        Rotation angle of the color bar label, by default -90.
                    cbar_label_location : str, optional
                        Location of the color bar label, by default 'bottom'.
                        Options: 'top', 'bottom', 'center', 'baseline', 'center_baseline'.
                    cbar_length : float, optional
                        Ratio to control the height/width of the color bar, by default 0.75.
                    ticks_spacing : int, optional
                        Spacing between ticks on the color bar, by default 2.
                    cbar_label_size : int, optional
                        Font size of the color bar label, by default 12.
                    cbar_label : str, optional
                        Label text for the color bar, by default 'Value'.

                Color scale options:
                    color_scale : ColorScale or str, optional
                        Type of color scaling to use, by default 'linear'.
                        Accepts a :class:`cleopatra.styles.ColorScale`
                        member or its string value (case-insensitive). An
                        unrecognised value raises ``ValueError``. Options:
                        - 'linear': Linear scale
                        - 'power': Power-law normalization
                        - 'sym-lognorm': Symmetrical logarithmic scale
                        - 'boundary-norm': Discrete intervals based on boundaries
                        - 'midpoint': Scale split at a specified midpoint
                    gamma : float, optional
                        Exponent for 'power' color scale, by default 0.5.
                        Values < 1 emphasize lower values, values > 1 emphasize higher values.
                    line_threshold : float, optional
                        Threshold for 'sym-lognorm' color scale, by default 0.0001.
                    line_scale : float, optional
                        Scale factor for 'sym-lognorm' color scale, by default 0.001.
                    bounds : list, optional
                        Boundaries for 'boundary-norm' color scale, by default None.
                        Defines the discrete intervals for color mapping.
                    midpoint : float, optional
                        Midpoint value for 'midpoint' color scale, by default 0.
                    levels : int or sequence, optional
                        Discrete colour levels (xarray-aligned), by
                        default None. An ``int`` selects N
                        linearly-spaced edges between ``vmin`` and
                        ``vmax``; a sequence is used as explicit edges
                        (sorted ascending). When set under the default
                        ``color_scale="linear"`` the norm is switched
                        to a ``BoundaryNorm`` so ``imshow`` /
                        ``pcolormesh`` are also discretised; under
                        ``color_scale="boundary-norm"`` ``levels`` acts
                        as the bin edges when ``bounds`` is unset.
                        Always forwarded as the level array to
                        ``contour`` / ``contourf``.

                Xarray-aligned colour kwargs:
                    robust : bool, optional
                        When True, use the 2nd and 98th percentile of
                        the unmasked data for ``vmin`` / ``vmax``,
                        matching xarray's ``robust=True`` default. An
                        explicit ``vmin`` / ``vmax`` always wins. By
                        default False.
                    center : float, optional
                        Diverging-colormap centring value. When set,
                        ``vmin`` / ``vmax`` are made symmetric around
                        ``center`` (after ``robust`` has been applied),
                        and the cmap auto-switches to ``"RdBu_r"`` if
                        the caller did not pass an explicit ``cmap``.
                        By default None (no centring).
                    extend : str, optional
                        Colorbar arrow extension. One of ``"neither"``,
                        ``"both"``, ``"min"``, ``"max"``, or None to
                        auto-resolve (``"both"`` when ``levels`` is
                        set, otherwise ``"neither"``). By default
                        None.
                    cbar_kwargs : dict, optional
                        Extra keyword arguments forwarded to
                        ``fig.colorbar``. Merges over the defaults
                        computed by cleopatra so user keys win on
                        collision. Common keys: ``label``, ``shrink``,
                        ``aspect``, ``orientation``, ``pad``,
                        ``ticks``. By default None.

                Cell value display options:
                    display_cell_value : bool, optional
                        Whether to display the values of cells as text, by default False.
                    num_size : int, optional
                        Font size of the cell value text, by default 8.
                    background_color_threshold : float, optional
                        Threshold for cell value text color, by default None.
                        If cell value > threshold, text is black; otherwise, text is white.
                        If None, uses max(array)/2 as the threshold.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: A tuple containing:
                - fig: The matplotlib Figure object
                - ax: The matplotlib Axes object

        Raises:
            ValueError: If an invalid keyword argument is provided.

        Examples:
        - Basic array plot:

            ```python
            >>> import numpy as np
            >>> from cleopatra.array_glyph import ArrayGlyph
            >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Customized Plot", title_size=18)
            >>> fig, ax = array.plot()

            ```
        ![array-plot](./../images/array_glyph/array-plot.png)

        - Color bar customization:

            - Create an array and instantiate the `Array` object with custom options.
                ```python
                >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Customized color bar", title_size=18)
                >>> fig, ax = array.plot(
                ...     cbar_orientation="horizontal",
                ...     cbar_label_rotation=-90,
                ...     cbar_label_location="center",
                ...     cbar_length=0.7,
                ...     cbar_label_size=12,
                ...     cbar_label="Discharge m3/s",
                ...     ticks_spacing=5,
                ...     color_scale="linear",
                ...     cmap="coolwarm_r",
                ... )

                ```
                ![color-bar-customization](./../images/array_glyph/color-bar-customization.png)

        - Display values for each cell:

            - you can display the values for each cell by using thr parameter `display_cell_value`, and customize how
                the values are displayed using the parameter `background_color_threshold` and `num_size`.

                ```python
                >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Display array values", title_size=18)
                >>> fig, ax = array.plot(
                ...     display_cell_value=True,
                ...     num_size=12
                ... )

                ```
                ![display-cell-values](./../images/array_glyph/display-cell-values.png)

        - Plot points at specific locations in the array:

            - you can display points in specific cells in the array and also display a value for each of these points.
                The point parameter takes an array with the first column as the values to be displayed on top of the
                points, the second and third columns are the row and column index of the point in the array.
            - The `point_color` and `point_size` parameters are used to customize the appearance of the points,
                while the `pid_color` and `pid_size` parameters are used to customize the appearance of the point
                IDs/text.

                ```python
                >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Display Points", title_size=14)
                >>> points = np.array([[1, 0, 0], [2, 1, 1], [3, 2, 2]])
                >>> fig, ax = array.plot(
                ...     points=points,
                ...     point_color="black",
                ...     point_size=100,
                ...     pid_color="orange",
                ...     pid_size=30,
                ... )

                ```
                ![display-points](./../images/array_glyph/display-points.png)

        - Color scale customization:

            - Power scale (with different gamma values).

                - The default power scale uses a gamma value of 0.5.

                    ```python
                    >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Power scale", title_size=18)
                    >>> fig, ax = array.plot(
                    ...     cbar_label="Discharge m3/s",
                    ...     color_scale="power",
                    ...     cmap="coolwarm_r",
                    ...     cbar_label_rotation=-90,
                    ... )

                    ```
                    ![power-scale](./../images/array_glyph/power-scale.png)

                - change the gamma of 0.8 (emphasizes higher values less).

                    ```python
                    >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Power scale - gamma=0.8", title_size=18)
                    >>> fig, ax = array.plot(
                    ...     color_scale="power",
                    ...     gamma=0.8,
                    ...     cmap="coolwarm_r",
                    ...     cbar_label_rotation=-90,
                    ...     cbar_label="Discharge m3/s",
                    ... )

                    ```
                    ![power-scale-gamma-0.8](./../images/array_glyph/power-scale-gamma-0.8.png)

                - change the gamma of 0.1 (emphasizes higher values more).

                    ```python
                    >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Power scale - gamma=0.1", title_size=18)
                    >>> fig, ax = array.plot(
                    ...     color_scale="power",
                    ...     gamma=0.1,
                    ...     cmap="coolwarm_r",
                    ...     cbar_label_rotation=-90,
                    ...     cbar_label="Discharge m3/s",
                    ... )

                    ```
                    ![power-scale-gamma-0.1](./../images/array_glyph/power-scale-gamma-0.1.png)

            - Logarithmic scale.

                - the logarithmic scale uses to parameters `line_threshold` and `line_scale` with a default
                value if 0.0001, and 0.001 respectively.
                    ```python
                    >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Logarithmic scale", title_size=18)
                    >>> fig, ax = array.plot(
                    ...     cbar_label="Discharge m3/s",
                    ...     color_scale="sym-lognorm",
                    ...     cmap="coolwarm_r",
                    ...     cbar_label_rotation=-90,
                    ... )

                    ```
                    ![log-scale](./../images/array_glyph/log-scale.png)

                - you can change the `line_threshold` and `line_scale` values.
                    ```python
                    >>> array = ArrayGlyph(
                    ...     arr, figsize=(6, 6), title="Logarithmic scale: Customized Parameter", title_size=12
                    ... )
                    >>> fig, ax = array.plot(
                    ...     cbar_label_rotation=-90,
                    ...     cbar_label="Discharge m3/s",
                    ...     color_scale="sym-lognorm",
                    ...     cmap="coolwarm_r",
                    ...     line_threshold=0.015,
                    ...     line_scale=0.1,
                    ... )

                    ```
                    ![log-scale](./../images/array_glyph/log-scale-custom-parameters.png)

            - Defined boundary scale.
                ```python
                >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Defined boundary scale", title_size=18)
                >>> fig, ax = array.plot(
                ...     cbar_label_rotation=-90,
                ...     cbar_label="Discharge m3/s",
                ...     color_scale="boundary-norm",
                ...     cmap="coolwarm_r",
                ... )

                ```
                ![boundary-scale](./../images/array_glyph/boundary-scale.png)

                - You can also define the boundaries.
                    ```python
                    >>> array = ArrayGlyph(
                    ...     arr, figsize=(6, 6), title="Defined boundary scale: defined bounds", title_size=18
                    ... )
                    >>> bounds = [0, 5, 10]
                    >>> fig, ax = array.plot(
                    ...     cbar_label_rotation=-90,
                    ...     cbar_label="Discharge m3/s",
                    ...     color_scale="boundary-norm",
                    ...     bounds=bounds,
                    ...     cmap="coolwarm_r",
                    ... )

                    ```
                    ![boundary-scale-defined-bounds](./../images/array_glyph/boundary-scale-defined-bounds.png)

            - Midpoint scale.

                in the midpoint scale you can define a value that splits the scale into half.
                ```python
                >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Midpoint scale", title_size=18)
                >>> fig, ax = array.plot(
                ...     cbar_label_rotation=-90,
                ...     cbar_label="Discharge m3/s",
                ...     color_scale="midpoint",
                ...     cmap="coolwarm_r",
                ...     midpoint=2,
                ... )

                ```
                ![midpoint-scale-costom-parameters](./../images/array_glyph/midpoint-scale-costom-parameters.png)

        - Render kinds (``kind=``):

            - ``"pcolormesh"`` for a quadrilateral mesh render. Note
                that ``pcolormesh`` does not honour ``extent``, so the
                axes are drawn in array index space.
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> arr = np.arange(25, dtype=float).reshape(5, 5)
                >>> glyph = ArrayGlyph(arr)
                >>> fig, ax = glyph.plot(kind="pcolormesh")  # doctest: +SKIP

                ```
            - ``"contourf"`` for filled contours. When ``levels`` is set
                the level edges line up with the colorbar boundaries.
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> arr = np.arange(25, dtype=float).reshape(5, 5)
                >>> glyph = ArrayGlyph(arr, levels=5)
                >>> fig, ax = glyph.plot(kind="contourf")  # doctest: +SKIP

                ```
            - Invalid kinds are rejected with a clear error:
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> arr = np.arange(9, dtype=float).reshape(3, 3)
                >>> ArrayGlyph(arr).plot(kind="heatmap")
                Traceback (most recent call last):
                    ...
                ValueError: Invalid kind='heatmap'. Valid kinds are ('auto', 'imshow', 'pcolormesh', 'contour', 'contourf').

                ```

        - xarray-aligned colour kwargs:

            - ``robust=True`` clips ``vmin`` / ``vmax`` to the
                2nd/98th percentile so a single outlier no longer
                dominates the colour scale:
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> data = np.arange(100, dtype=float).reshape(10, 10)
                >>> data[0, 0] = 1e6  # outlier
                >>> glyph = ArrayGlyph(data, robust=True)
                >>> fig, ax = glyph.plot(robust=True)  # doctest: +SKIP
                >>> round(glyph.vmin, 1), round(glyph.vmax, 1)
                (3.0, 98.0)

                ```
            - ``center=0`` symmetrises the limits around zero and
                auto-switches the cmap to ``"RdBu_r"`` (xarray-style
                diverging default):
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> anomaly = np.linspace(-3.0, 8.0, 25).reshape(5, 5)
                >>> glyph = ArrayGlyph(anomaly, center=0.0)
                >>> fig, ax = glyph.plot(center=0.0)  # doctest: +SKIP
                >>> glyph.vmin, glyph.vmax
                (-8.0, 8.0)
                >>> glyph.default_options["cmap"]
                'RdBu_r'

                ```
            - ``levels`` discretises the colour scale and ``extend``
                controls the colorbar arrows:
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> arr = np.arange(25, dtype=float).reshape(5, 5)
                >>> glyph = ArrayGlyph(arr, levels=6, extend="both")
                >>> fig, ax = glyph.plot()  # doctest: +SKIP
                >>> glyph.default_options["levels"], glyph.default_options["extend"]
                (6, 'both')

                ```
            - ``cbar_kwargs`` forwards extra keyword arguments to the
                underlying :func:`matplotlib.pyplot.colorbar` call;
                user keys win on collision:
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> arr = np.arange(9, dtype=float).reshape(3, 3)
                >>> glyph = ArrayGlyph(arr, cbar_kwargs={"shrink": 0.5})
                >>> fig, ax = glyph.plot()  # doctest: +SKIP
                >>> glyph.default_options["cbar_kwargs"]
                {'shrink': 0.5}

                ```
        """
        if kind not in VALID_PLOT_KINDS:
            raise ValueError(
                f"Invalid kind={kind!r}. Valid kinds are {VALID_PLOT_KINDS}."
            )
        if self.rgb and kind not in ("imshow", "auto"):
            raise ValueError(
                "RGB compositing requires kind='imshow'. "
                f"Got kind={kind!r}."
            )

        for key, val in kwargs.items():
            if key not in self.default_options.keys():
                raise ValueError(
                    f"The given keyword argument:{key} is not correct, possible parameters are,"
                    f" {DEFAULT_OPTIONS}"
                )
            else:
                self.default_options[key] = val

        self._validate_extend(self.default_options.get("extend"))

        self.default_options["kind"] = kind
        # Resolve "auto": when curvilinear / non-uniform coords are set
        # we route to ``pcolormesh`` (it honours the (x, y) arrays);
        # otherwise we fall back to the original ``imshow`` path.
        if kind == "auto":
            effective_kind = "pcolormesh" if self._coords is not None else "imshow"
        else:
            effective_kind = kind

        if self.fig is None:
            self.fig, self.ax = self.create_figure_axes()

        arr = self.arr
        fig, ax = self.fig, self.ax

        if self.rgb:
            ax.imshow(arr, extent=self.extent)
        else:
            # if user did not input ticks spacing use the calculated one.
            if "ticks_spacing" in kwargs.keys():
                self.default_options["ticks_spacing"] = kwargs["ticks_spacing"]
            else:
                self.default_options["ticks_spacing"] = self.ticks_spacing

            # Recompute (vmin, vmax) for this plot call if any of the
            # xarray-aligned colour kwargs (``robust``, ``center``) or the
            # explicit ``vmin``/``vmax`` were passed here. We honour the
            # values stashed on ``self`` from the constructor when none of
            # these are present in *this* call.
            recompute_keys = {"robust", "center", "vmin", "vmax"}
            if recompute_keys.intersection(kwargs.keys()):
                vmin_final, vmax_final = self._resolve_color_limits(
                    arr,
                    vmin_kw=kwargs.get("vmin"),
                    vmax_kw=kwargs.get("vmax"),
                    robust=bool(self.default_options.get("robust", False)),
                    center=self.default_options.get("center"),
                    vmin_explicit="vmin" in kwargs,
                    vmax_explicit="vmax" in kwargs,
                )
                self._vmin = vmin_final
                self._vmax = vmax_final
                # Keep ticks_spacing in sync with the new colour range
                # unless the caller pinned it explicitly.
                if "ticks_spacing" not in kwargs:
                    self.ticks_spacing = (vmax_final - vmin_final) / 10
                    self.default_options["ticks_spacing"] = self.ticks_spacing

            # Auto-switch the colormap to a diverging default when the
            # caller passes ``center`` here without an explicit cmap.
            if (
                "center" in kwargs
                and kwargs["center"] is not None
                and "cmap" not in kwargs
            ):
                self.default_options["cmap"] = DIVERGING_DEFAULT_CMAP

            self.default_options["vmin"] = self.vmin
            self.default_options["vmax"] = self.vmax

            # creating the ticks/bounds
            ticks = self.get_ticks()
            im, cbar_kw = self._plot_im_get_cbar_kw(
                ax, arr, ticks, kind=effective_kind
            )

            # Create colorbar
            self.cbar = self.create_color_bar(ax, im, cbar_kw)

        ax.set_title(
            self.default_options["title"], fontsize=self.default_options["title_size"]
        )

        if self.extent is None and effective_kind == "imshow":
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

        # Cell-value annotations and point overlays are only meaningful
        # for raster-style renderings (imshow / pcolormesh). For contour
        # / contourf they are skipped silently — there is no per-cell
        # grid to annotate.
        supports_overlay = effective_kind in ("imshow", "pcolormesh")
        optional_display = {}
        if self.default_options["display_cell_value"] and supports_overlay:
            indices = get_indices2(arr, [np.nan])
            optional_display["cell_text_value"] = self._plot_text(
                ax, arr, indices, self.default_options
            )

        if points is not None and supports_overlay:
            row = points[:, 1]
            col = points[:, 2]
            optional_display["points_scatter"] = ax.scatter(
                col, row, color=point_color, s=point_size
            )
            optional_display["points_id"] = self._plot_point_values(
                ax, points, pid_color, pid_size
            )

        # # Normalize the threshold to the image color range.
        # if self.default_options["background_color_threshold"] is not None:
        #     im.norm(self.default_options["background_color_threshold"])
        # else:
        #     im.norm(self.vmax) / 2.0
        plt.show()
        return fig, ax

    def facet(
        self,
        *,
        col: str | None = None,
        row: str | None = None,
        col_wrap: int | None = None,
        col_coords: Sequence[Any] | None = None,
        row_coords: Sequence[Any] | None = None,
        kind: str = "auto",
        figsize: tuple[float, float] | None = None,
        extents: Sequence[Sequence[float]] | None = None,
        **kwargs,
    ) -> FacetGrid:
        """Render a grid of subplots from a 3-D or 4-D stack.

        Mirrors xarray's :class:`xarray.plot.facetgrid.FacetGrid` API.
        ``self.arr`` must be 3-D ``(N, H, W)`` when only ``col`` is set,
        or 4-D ``(N, M, H, W)`` when both ``col`` and ``row`` are set.
        All subplots share a common colour scale (``vmin``/``vmax``
        computed over the full stack unless the user passed explicit
        limits) and a single shared colorbar attached to the first
        rendered subplot.

        Spatial extent: every panel is a slice of the *same* array, so by
        default they all share the parent glyph's ``extent`` (one spatial
        domain — exactly like xarray's ``FacetGrid``, which facets a single
        ``DataArray`` over a coordinate dimension). If your slices are
        same-shape grids covering *different* windows, pass ``extents`` —
        one ``[xmin, ymin, xmax, ymax]`` per panel. (If the slices are
        genuinely different datasets, build separate :class:`ArrayGlyph`
        instances into your own ``plt.subplots`` grid instead.)

        Args:
            col: Name of the column-facet dimension (e.g. ``"time"``).
                Used as a label in the per-subplot title and in
                :attr:`FacetGrid.name_dicts`. Required when ``row`` is
                not given.
            row: Name of the row-facet dimension (e.g. ``"level"``).
                Required when faceting a 4-D stack.
            col_wrap: When only ``col`` is given, wrap the N subplots
                into ``col_wrap`` columns × ``ceil(N/col_wrap)`` rows.
                Ignored when ``row`` is set.
            col_coords: Optional sequence of coordinate labels for the
                column dimension. Length must match the column axis of
                the stack. When given, the per-subplot title contains
                the coord value instead of the integer index.
            row_coords: Optional sequence of coordinate labels for the
                row dimension. Length must match the row axis of the
                stack. Only honoured when ``row`` is set.
            kind: Render kind, forwarded to the per-subplot dispatch.
                One of ``"auto"``, ``"imshow"``, ``"pcolormesh"``,
                ``"contour"``, ``"contourf"``. Default ``"auto"``.
            figsize: Optional ``(width, height)`` for the shared figure.
                Defaults to ``(4 * ncols, 3.5 * nrows)``.
            extents: Optional per-panel spatial extents — one
                ``[xmin, ymin, xmax, ymax]`` (user-facing order) for each
                rendered subplot, in row-major order (``extents[k]``
                applies to ``result.axes.flat[k]``). Length must equal the
                number of panels. Mutually exclusive with the parent
                glyph's ``extent`` and with ``coords``. ``None`` (default)
                reuses the parent's ``extent`` on every panel (or index
                space when the parent has none).

            **kwargs: Forwarded to each subplot. Recognised keys
                include the same colour / colorbar / level kwargs as
                :meth:`plot`. ``vmin`` / ``vmax`` win over the
                stack-wide auto-computed limits.

        Returns:
            FacetGrid: Result object exposing ``fig``, ``axes``,
                ``cbar``, and ``name_dicts``.

        Raises:
            ValueError: If neither ``col`` nor ``row`` is given, if the
                array shape does not match the requested facet
                dimensions, if ``col_coords`` / ``row_coords`` lengths
                are wrong, if ``extents`` is combined with the parent's
                ``extent`` or ``coords``, or if ``extents`` has the wrong
                length or a non-length-4 element.

        Examples:
            - Facet a 3-D stack into a 1xN row of subplots:
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> stack = np.arange(4 * 5 * 5, dtype=float).reshape(4, 5, 5)
                >>> g = ArrayGlyph(stack).facet(col="t")
                >>> g.axes.shape
                (1, 4)
                >>> g.name_dicts[0]
                {'t': 0}

                ```
            - Wrap N=6 panels into a 2x3 grid with ``col_wrap=3``:
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> stack = np.arange(6 * 5 * 5, dtype=float).reshape(6, 5, 5)
                >>> g = ArrayGlyph(stack).facet(col="t", col_wrap=3)
                >>> g.axes.shape
                (2, 3)

                ```
            - Per-panel extents for same-shape grids over different
                windows (one ``[xmin, ymin, xmax, ymax]`` per subplot):
                ```python
                >>> import numpy as np
                >>> from cleopatra.array_glyph import ArrayGlyph
                >>> stack = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
                >>> g = ArrayGlyph(stack).facet(
                ...     col="region",
                ...     extents=[[0, 0, 10, 10], [10, 0, 20, 10]],
                ... )
                >>> [tuple(int(v) for v in im.get_extent()) for im in
                ...  (ax.get_images()[0] for ax in g.axes.flat)]
                [(0, 10, 0, 10), (10, 20, 0, 10)]

                ```
        """
        if col is None and row is None:
            raise ValueError(
                "at least one of `col`/`row` must be given"
            )
        if extents is not None:
            if self.extent is not None:
                raise ValueError(
                    "`extents` (per-panel) and the glyph's `extent` "
                    "(one shared domain) are mutually exclusive."
                )
            if self._coords is not None:
                raise ValueError(
                    "`extents` and `coords` are mutually exclusive."
                )
            for k, e in enumerate(extents):
                if len(e) != 4:
                    raise ValueError(
                        f"`extents[{k}]` must be a length-4 sequence "
                        f"[xmin, ymin, xmax, ymax], got {e!r}."
                    )

        arr = self.arr
        if row is None:
            if arr.ndim != 3:
                raise ValueError(
                    "Faceting on `col` alone requires a 3-D array "
                    f"(N, H, W); got shape {arr.shape}."
                )
            n_col = arr.shape[0]
            if col_wrap is not None:
                if not isinstance(col_wrap, (int, np.integer)) or col_wrap < 1:
                    raise ValueError(
                        f"`col_wrap` must be a positive int, got {col_wrap!r}."
                    )
                ncols = int(col_wrap)
                nrows = int(ceil(n_col / ncols))
            else:
                ncols = n_col
                nrows = 1
            if col_coords is not None and len(col_coords) != n_col:
                raise ValueError(
                    f"`col_coords` length {len(col_coords)} does not match "
                    f"the column axis size {n_col}."
                )
            panel_indices = [(i, None) for i in range(n_col)]
            n_panels = n_col
        else:
            if col is None:
                raise ValueError(
                    "Faceting on `row` requires `col` as well."
                )
            if arr.ndim != 4:
                raise ValueError(
                    "Faceting on `row`+`col` requires a 4-D array "
                    f"(Ncol, Nrow, H, W); got shape {arr.shape}."
                )
            n_col, n_row = arr.shape[0], arr.shape[1]
            ncols = n_col
            nrows = n_row
            if col_coords is not None and len(col_coords) != n_col:
                raise ValueError(
                    f"`col_coords` length {len(col_coords)} does not match "
                    f"the column axis size {n_col}."
                )
            if row_coords is not None and len(row_coords) != n_row:
                raise ValueError(
                    f"`row_coords` length {len(row_coords)} does not match "
                    f"the row axis size {n_row}."
                )
            panel_indices = [
                (i, j) for j in range(n_row) for i in range(n_col)
            ]
            n_panels = n_col * n_row

        if extents is not None and len(extents) != n_panels:
            raise ValueError(
                f"`extents` has {len(extents)} entries but there are "
                f"{n_panels} panels."
            )

        if figsize is None:
            figsize = (4.0 * ncols, 3.5 * nrows)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False
        )

        vmin_user = kwargs.get("vmin")
        vmax_user = kwargs.get("vmax")
        if vmin_user is None or vmax_user is None:
            if isinstance(arr, ma.MaskedArray):
                finite = arr.compressed()
            else:
                finite = np.asarray(arr).ravel()
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                stack_min = 0.0
                stack_max = 1.0
            else:
                stack_min = float(finite.min())
                stack_max = float(finite.max())
            shared_vmin = stack_min if vmin_user is None else float(vmin_user)
            shared_vmax = stack_max if vmax_user is None else float(vmax_user)
        else:
            shared_vmin = float(vmin_user)
            shared_vmax = float(vmax_user)

        per_subplot_kwargs = dict(kwargs)
        per_subplot_kwargs["vmin"] = shared_vmin
        per_subplot_kwargs["vmax"] = shared_vmax

        name_dicts: list[dict[str, Any]] = []
        cbar: Colorbar | None = None
        flat_axes = axes.ravel()

        for panel_idx, (col_idx, row_idx) in enumerate(panel_indices):
            ax = flat_axes[panel_idx]
            # Use plain slicing so ``numpy.ma.MaskedArray`` inputs keep
            # their mask on each per-panel sub-array. ``np.asarray``
            # would drop the mask and render masked cells as data.
            if row is None:
                panel_arr = arr[col_idx]
            else:
                panel_arr = arr[col_idx, row_idx]

            # Per-panel ``extents`` win if given (already in the
            # user-facing ``[xmin, ymin, xmax, ymax]`` order the
            # constructor wants). Otherwise reuse the parent's ``extent``,
            # which is stored in matplotlib's ``[xmin, xmax, ymin, ymax]``
            # order — convert it back before forwarding.
            if extents is not None:
                sub_extent = list(extents[panel_idx])
            elif self.extent is None:
                sub_extent = None
            else:
                sub_extent = [
                    self.extent[0],  # xmin
                    self.extent[2],  # ymin
                    self.extent[1],  # xmax
                    self.extent[3],  # ymax
                ]
            sub = ArrayGlyph(
                panel_arr,
                coords=self._coords,
                extent=sub_extent,
                fig=fig,
                ax=ax,
                **per_subplot_kwargs,
            )
            sub.plot(kind=kind)

            col_label = (
                col_coords[col_idx] if col_coords is not None else col_idx
            )
            name_dict: dict[str, Any] = {col: col_label}
            if row is not None:
                row_label = (
                    row_coords[row_idx] if row_coords is not None else row_idx
                )
                name_dict[row] = row_label
                title = f"{col}={col_label}, {row}={row_label}"
            else:
                title = f"{col}={col_label}"
            ax.set_title(title)
            name_dicts.append(name_dict)

            if panel_idx == 0 and getattr(sub, "cbar", None) is not None:
                cbar = sub.cbar

        for hidden_idx in range(n_panels, nrows * ncols):
            flat_axes[hidden_idx].set_visible(False)

        fig.tight_layout()
        result = FacetGrid(fig=fig, axes=axes, cbar=cbar, name_dicts=name_dicts)
        return result

    def animate(
        self,
        time: list[Any],
        points: np.ndarray = None,
        text_colors: tuple[str, str] = ("white", "black"),
        interval: int = 200,
        text_loc: list[Any, Any] = None,
        point_color: str = "red",
        point_size: int = 100,
        pid_color: str = "blue",
        pid_size: int = 10,
        *,
        data_getter: Callable[[int], np.ndarray] | None = None,
        **kwargs,
    ) -> FuncAnimation:
        """Create an animation from a 3D array.

        This method creates an animation by iterating through the first dimension of a 3D array.
        Each slice of the array becomes a frame in the animation, with optional time labels,
        point annotations, and cell value displays. Every frame is a slice of the *same* array,
        so they all share the glyph's single ``extent`` (one spatial domain) — there is no
        per-frame extent. For data spanning different domains, build one :class:`ArrayGlyph`
        per domain instead of stacking them into a single 3-D array.

        Args:
            time: A list containing labels for each frame in the animation.
                These could be timestamps, frame numbers, or any other identifiers.
                The length of this list should match the first dimension of the array.
            points: Points to display on the array, by default None.
                Should be a 3-column array where:
                - First column: values to display for each point
                - Second column: row indices of the points in the array
                - Third column: column indices of the points in the array
            text_colors: Two colors to be used for cell value text, by default ("white", "black").
                The first color is used when the cell value is below the background_color_threshold,
                and the second color is used when the cell value is above the threshold.
            interval: Delay between frames in milliseconds, by default 200.
                Controls the speed of the animation (smaller values = faster animation).
            text_loc: Location of the time label text as [x, y] coordinates, by default None.
                If None, defaults to [0.1, 0.2].
            point_color: Color of the points, by default "red".
                Any valid matplotlib color string.
            point_size: Size of the points, by default 100.
                Controls the marker size.
            pid_color: Color of the point value annotations, by default "blue".
                Any valid matplotlib color string.
            pid_size: Size of the point value annotations, by default 10.
                Controls the font size of the annotations.
            data_getter: Optional callable ``f(i) -> ndarray`` that
                returns the 2-D frame for index ``i``, by default
                None. When set, ``self.arr`` is no longer iterated;
                each frame is fetched lazily through the callback —
                useful for streaming frames from a remote / lazy
                source (e.g. a NetCDF time slab). The returned array
                must have shape ``self.arr.shape[-2:]``. When None
                (default) the existing behaviour is preserved and
                ``self.arr[i]`` supplies frame ``i``.
            **kwargs: Additional keyword arguments for customizing the animation.

                Plot appearance:
                    title : str, optional
                        Title of the plot, by default 'Array Plot'.
                    title_size : int, optional
                        Title font size, by default 15.
                    cmap : str, optional
                        Colormap name, by default 'coolwarm_r'.
                    vmin : float, optional
                        Minimum value for color scaling, by default min(array).
                    vmax : float, optional
                        Maximum value for color scaling, by default max(array).

                Color bar options:
                    cbar_orientation : str, optional
                        Orientation of the color bar, by default 'vertical'.
                        Can be 'horizontal' or 'vertical'.
                    cbar_label_rotation : float, optional
                        Rotation angle of the color bar label, by default -90.
                    cbar_label_location : str, optional
                        Location of the color bar label, by default 'bottom'.
                        Options: 'top', 'bottom', 'center', 'baseline', 'center_baseline'.
                    cbar_length : float, optional
                        Ratio to control the height/width of the color bar, by default 0.75.
                    ticks_spacing : int, optional
                        Spacing between ticks on the color bar, by default 2.
                    cbar_label_size : int, optional
                        Font size of the color bar label, by default 12.
                    cbar_label : str, optional
                        Label text for the color bar, by default 'Value'.

                Color scale options:
                    color_scale : ColorScale or str, optional
                        Type of color scaling to use, by default 'linear'.
                        Accepts a :class:`cleopatra.styles.ColorScale`
                        member or its string value (case-insensitive). An
                        unrecognised value raises ``ValueError``. Options:
                        - 'linear': Linear scale
                        - 'power': Power-law normalization
                        - 'sym-lognorm': Symmetrical logarithmic scale
                        - 'boundary-norm': Discrete intervals based on boundaries
                        - 'midpoint': Scale split at a specified midpoint
                    gamma : float, optional
                        Exponent for 'power' color scale, by default 0.5.
                        Values < 1 emphasize lower values, values > 1 emphasize higher values.
                    line_threshold : float, optional
                        Threshold for 'sym-lognorm' color scale, by default 0.0001.
                    line_scale : float, optional
                        Scale factor for 'sym-lognorm' color scale, by default 0.001.
                    bounds : list, optional
                        Boundaries for 'boundary-norm' color scale, by default None.
                        Defines the discrete intervals for color mapping.
                    midpoint : float, optional
                        Midpoint value for 'midpoint' color scale, by default 0.

                Cell value display options:
                    display_cell_value : bool, optional
                        Whether to display the values of cells as text, by default False.
                    num_size : int, optional
                        Font size of the cell value text, by default 8.
                    background_color_threshold : float, optional
                        Threshold for cell value text color, by default None.
                        If cell value > threshold, text is black; otherwise, text is white.
                        If None, uses max(array)/2 as the threshold.

        Returns:
            matplotlib.animation.FuncAnimation: The animation object that can be displayed
                in a notebook or saved to a file.

        Raises:
            ValueError: If an invalid keyword argument is provided.
            ValueError: If the length of the time list doesn't match the first dimension of the array.
            ValueError: If ``data_getter`` is None and ``self.arr``
                is 2-D (no time axis to iterate over).

        Notes:
            The animation is created by iterating through the first dimension of the array.
            For example, if the array has shape (10, 20, 30), the animation will have 10 frames,
            each showing a 20x30 slice of the array.

            To display the animation in a Jupyter notebook, you may need to use:
            ```python
            from IPython.display import HTML
            HTML(anim_obj.to_jshtml())
            ```

            To save the animation to a file, use the `save_animation` method after creating
            the animation.

        Examples:
        Basic animation from a 3D array:
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> # Create a 3D array with 5 frames, each 10x10
        >>> arr = np.random.randint(1, 10, size=(5, 10, 10))
        >>> # Create labels for each frame
        >>> frame_labels = ["Frame 1", "Frame 2", "Frame 3", "Frame 4", "Frame 5"]
        >>> # Create the ArrayGlyph object
        >>> animated_array = ArrayGlyph(arr, figsize=(8, 8), title="Animated Array")
        >>> # Create the animation
        >>> anim_obj = animated_array.animate(frame_labels)

        ```
        Animation with custom interval (speed):
        ```python
        >>> animated_array = ArrayGlyph(arr, figsize=(8, 8), title="Animated Array")
        >>> # Slower animation (500ms between frames)
        >>> anim_obj = animated_array.animate(frame_labels, interval=500)
        >>> animated_array = ArrayGlyph(arr, figsize=(8, 8), title="Animated Array")
        >>> # Faster animation (100ms between frames)
        >>> anim_obj = animated_array.animate(frame_labels, interval=100)

        ```
        Animation with points:
        ```python
        >>> # Create points to display on the animation
        >>> points = np.array([[1, 2, 3], [2, 5, 5], [3, 8, 8]])
        >>> animated_array = ArrayGlyph(arr, figsize=(8, 8), title="Animated Array")
        >>> anim_obj = animated_array.animate(
        ...     frame_labels,
        ...     points=points,
        ...     point_color="black",
        ...     point_size=150,
        ...     pid_color="white",
        ...     pid_size=12
        ... )

        ```
        Animation with cell values displayed:
        ```python
        >>> animated_array = ArrayGlyph(arr, figsize=(8, 8), title="Animated Array")
        >>> anim_obj = animated_array.animate(
        ...     frame_labels,
        ...     display_cell_value=True,
        ...     num_size=10,
        ...     text_colors=("yellow", "blue")
        ... )

        ```
        ![animated_array](./../images/array_glyph/animated_array.gif)

        Saving the animation to a file:
        ```python
        >>> # Create the animation first
        >>> animated_array = ArrayGlyph(arr, figsize=(8, 8), title="Animated Array")
        >>> anim_obj = animated_array.animate(frame_labels)
        >>> # Then save it to a file
        >>> animated_array.save_animation("animation.gif", fps=2)

        ```
        Lazy frame streaming via ``data_getter`` (the callback supplies
        frame ``i`` on demand — useful for NetCDF time slabs or any
        source where eager loading is too expensive). The data array
        on the glyph acts as a shape template; only its last two axes
        are read.
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> template = np.arange(36, dtype=float).reshape(1, 6, 6)
        >>> glyph = ArrayGlyph(template, figsize=(4, 4), title="Lazy")
        >>> labels = ["t0", "t1", "t2"]
        >>> def get_frame(i):
        ...     return np.full((6, 6), float(i)) + np.arange(36).reshape(6, 6)
        >>> anim_obj = glyph.animate(labels, data_getter=get_frame)
        >>> anim_obj._fig is glyph.fig
        True

        ```
        """
        if text_loc is None:
            text_loc = [0.1, 0.2]

        for key, val in kwargs.items():
            if key not in self.default_options.keys():
                raise ValueError(
                    f"The given keyword argument:{key} is not correct, possible parameters are,"
                    f" {DEFAULT_OPTIONS}"
                )
            else:
                self.default_options[key] = val

        # if user did not input ticks spacing use the calculated one.
        if "ticks_spacing" in kwargs.keys():
            self.default_options["ticks_spacing"] = kwargs["ticks_spacing"]
        else:
            self.default_options["ticks_spacing"] = self.ticks_spacing

        if "vmin" in kwargs.keys():
            self.default_options["vmin"] = kwargs["vmin"]
        else:
            self.default_options["vmin"] = self.vmin

        if "vmax" in kwargs.keys():
            self.default_options["vmax"] = kwargs["vmax"]
        else:
            self.default_options["vmax"] = self.vmax

        # if optional_display
        precision = self.default_options["precision"]
        array = self.arr

        # ``data_getter`` is the lazy-frame escape hatch added in
        # CLEO-7. When ``None`` (default) we fall back to the eager
        # ``self.arr[i]`` path. We require a 3-D arr in the eager
        # path; with a callback, ``self.arr`` is used only for the
        # frame shape so a 2-D template is fine.
        if data_getter is None:
            if array.ndim != 3:
                raise ValueError(
                    "animate requires either a 3-D arr or a "
                    "data_getter callback"
                )
            frame_0 = array[0, :, :]
            n_frames = np.shape(array)[0]
        else:
            n_frames = len(time)
            frame_0 = np.asarray(data_getter(0))
            expected_shape = tuple(array.shape[-2:])
            if frame_0.shape != expected_shape:
                raise ValueError(
                    f"`data_getter` returned shape {frame_0.shape}, "
                    f"expected {expected_shape} (matching the data "
                    "array's last two axes)."
                )

        if self.fig is None:
            self.fig, self.ax = self.create_figure_axes()

        fig, ax = self.fig, self.ax

        ticks = self.get_ticks()
        im, cbar_kw = self._plot_im_get_cbar_kw(ax, frame_0, ticks)

        # Create colorbar (stored on the instance, mirroring ``plot``).
        self.cbar = self.create_color_bar(ax, im, cbar_kw)

        ax.set_title(
            self.default_options["title"], fontsize=self.default_options["title_size"]
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_xticks([])
        ax.set_yticks([])

        if self.default_options["display_cell_value"]:
            indices = get_indices2(frame_0, [np.nan])
            cell_text_value = self._plot_text(
                ax, frame_0, indices, self.default_options
            )
            indices = np.array(indices)

        if points is not None:
            row = points[:, 1]
            col = points[:, 2]
            points_scatter = ax.scatter(col, row, color=point_color, s=point_size)
            points_id = self._plot_point_values(ax, points, pid_color, pid_size)

        # Normalize the threshold to the image color range. With a
        # lazy ``data_getter`` we only have ``frame_0`` available
        # cheaply, so fall back to its max — callers who care can
        # set ``background_color_threshold`` explicitly.
        if self.default_options["background_color_threshold"] is not None:
            background_color_threshold = im.norm(
                self.default_options["background_color_threshold"]
            )
        else:
            ref_for_threshold = array if data_getter is None else frame_0
            background_color_threshold = im.norm(np.nanmax(ref_for_threshold)) / 2.0

        day_text = ax.text(
            text_loc[0],
            text_loc[1],
            " ",
            fontsize=self.default_options["cbar_label_size"],
        )

        def _fetch_frame(i: int) -> np.ndarray:
            """Resolve frame ``i`` for the animation step.

            Routes between the eager ``self.arr[i]`` path and the lazy
            ``data_getter(i)`` callback added in CLEO-7. The returned
            frame must always match ``self.arr.shape[-2:]``; the
            callback variant re-validates per call to catch upstream
            shape drift (e.g. a NetCDF slab that changed size between
            frames).

            Args:
                i: Zero-based frame index. Must be a valid index into
                    the time axis (``0 <= i < n_frames``).

            Returns:
                np.ndarray: The 2-D frame for index ``i``, with shape
                    equal to ``self.arr.shape[-2:]``.

            Raises:
                ValueError: If ``data_getter`` is set and the callback
                    returns an array whose shape does not match the
                    expected ``(H, W)``.
            """
            if data_getter is None:
                frame = array[i, :, :]
            else:
                frame = np.asarray(data_getter(i))
                if frame.shape != tuple(array.shape[-2:]):
                    raise ValueError(
                        f"`data_getter` returned shape {frame.shape}, "
                        f"expected {tuple(array.shape[-2:])}."
                    )
            return frame

        def init():
            """initialize the plot with the cached first frame"""
            im.set_data(frame_0)
            day_text.set_text("")
            output = [im, day_text]

            if points is not None:
                points_scatter.set_offsets(np.c_[col, row])
                output.append(points_scatter)
                update_points = lambda x: points_id[x].set_text(points[x, 0])
                list(map(update_points, range(len(col))))

                output += points_id

            if self.default_options["display_cell_value"]:
                vals = frame_0[indices[:, 0], indices[:, 1]]
                update_cell_value = lambda x: cell_text_value[x].set_text(vals[x])
                # Iterate over the actual artist list, not
                # ``self.num_domain_cells`` — keep the loop bound tied to the
                # thing being indexed so it can never raise ``IndexError``.
                list(map(update_cell_value, range(len(cell_text_value))))
                output += cell_text_value

            return output

        def animate_a(i):
            """plot for each element in the iterable."""
            frame = _fetch_frame(i)
            im.set_data(frame)
            day_text.set_text("Date = " + str(time[i])[0:10])
            output = [im, day_text]

            if points is not None:
                points_scatter.set_offsets(np.c_[col, row])
                output.append(points_scatter)

                for x in range(len(col)):
                    points_id[x].set_text(points[x, 0])

                output += points_id

            if self.default_options["display_cell_value"]:
                vals = frame[indices[:, 0], indices[:, 1]]

                def update_cell_value(x):
                    """Update cell value"""
                    val = round(vals[x], precision)
                    kw = {
                        "color": text_colors[
                            int(im.norm(vals[x]) > background_color_threshold)
                        ]
                    }
                    cell_text_value[x].update(kw)
                    cell_text_value[x].set_text(val)

                # See ``init`` above: iterate the artist list, not
                # ``self.num_domain_cells``, so the bound matches the index.
                list(map(update_cell_value, range(len(cell_text_value))))

                output += cell_text_value

            return output

        plt.tight_layout()
        anim = FuncAnimation(
            fig,
            animate_a,
            init_func=init,
            frames=n_frames,
            interval=interval,
            blit=True,
        )
        self._anim = anim
        plt.show()
        return anim

    # @staticmethod
    # def plot_type_1(
    #     Y1,
    #     Y2,
    #     Points,
    #     PointsY,
    #     PointMaxSize=200,
    #     PointMinSize=1,
    #     X_axis_label="X Axis",
    #     LegendNum=5,
    #     LegendLoc=(1.3, 1),
    #     PointLegendTitle="Output 2",
    #     Ylim=[0, 180],
    #     Y2lim=[-2, 14],
    #     color1="#27408B",
    #     color2="#DC143C",
    #     color3="grey",
    #     linewidth=4,
    #     **kwargs,
    # ):
    #     """Plot_Type1.
    #
    #     !TODO Needs docs
    #
    #     Parameters
    #     ----------
    #     Y1 : TYPE
    #         DESCRIPTION.
    #     Y2 : TYPE
    #         DESCRIPTION.
    #     Points : TYPE
    #         DESCRIPTION.
    #     PointsY : TYPE
    #         DESCRIPTION.
    #     PointMaxSize : TYPE, optional
    #         DESCRIPTION. The default is 200.
    #     PointMinSize : TYPE, optional
    #         DESCRIPTION. The default is 1.
    #     X_axis_label : TYPE, optional
    #         DESCRIPTION. The default is 'X Axis'.
    #     LegendNum : TYPE, optional
    #         DESCRIPTION. The default is 5.
    #     LegendLoc : TYPE, optional
    #         DESCRIPTION. The default is (1.3, 1).
    #     PointLegendTitle : TYPE, optional
    #         DESCRIPTION. The default is "Output 2".
    #     Ylim : TYPE, optional
    #         DESCRIPTION. The default is [0,180].
    #     Y2lim : TYPE, optional
    #         DESCRIPTION. The default is [-2,14].
    #     color1 : TYPE, optional
    #         DESCRIPTION. The default is '#27408B'.
    #     color2 : TYPE, optional
    #         DESCRIPTION. The default is '#DC143C'.
    #     color3 : TYPE, optional
    #         DESCRIPTION. The default is "grey".
    #     linewidth : TYPE, optional
    #         DESCRIPTION. The default is 4.
    #     **kwargs : TYPE
    #         DESCRIPTION.
    #
    #     Returns
    #     -------
    #     ax1 : TYPE
    #         DESCRIPTION.
    #     TYPE
    #         DESCRIPTION.
    #     fig : TYPE
    #         DESCRIPTION.
    #     """
    #     fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    #
    #     ax2 = ax1.twinx()
    #
    #     ax1.plot(
    #         Y1[:, 0],
    #         Y1[:, 1],
    #         zorder=1,
    #         color=color1,
    #         linestyle=Styles.get_line_style(0),
    #         linewidth=linewidth,
    #         label="Model 1 Output1",
    #     )
    #
    #     if "Y1_2" in kwargs.keys():
    #         Y1_2 = kwargs["Y1_2"]
    #
    #         rows_axis1, cols_axis1 = np.shape(Y1_2)
    #
    #         if "Y1_2_label" in kwargs.keys():
    #             label = kwargs["Y2_2_label"]
    #         else:
    #             label = ["label"] * (cols_axis1 - 1)
    #         # first column is the x axis
    #         for i in range(1, cols_axis1):
    #             ax1.plot(
    #                 Y1_2[:, 0],
    #                 Y1_2[:, i],
    #                 zorder=1,
    #                 color=color2,
    #                 linestyle=Styles.get_line_style(i),
    #                 linewidth=linewidth,
    #                 label=label[i - 1],
    #             )
    #
    #     ax2.plot(
    #         Y2[:, 0],
    #         Y2[:, 1],
    #         zorder=1,
    #         color=color3,
    #         linestyle=Styles.get_line_style(6),
    #         linewidth=2,
    #         label="Output1-Diff",
    #     )
    #
    #     if "Y2_2" in kwargs.keys():
    #         Y2_2 = kwargs["Y2_2"]
    #         rows_axis2, cols_axis2 = np.shape(Y2_2)
    #
    #         if "Y2_2_label" in kwargs.keys():
    #             label = kwargs["Y2_2_label"]
    #         else:
    #             label = ["label"] * (cols_axis2 - 1)
    #
    #         for i in range(1, cols_axis2):
    #             ax1.plot(
    #                 Y2_2[:, 0],
    #                 Y2_2[:, i],
    #                 zorder=1,
    #                 color=color2,
    #                 linestyle=Styles.get_line_style(i),
    #                 linewidth=linewidth,
    #                 label=label[i - 1],
    #             )
    #
    #     if "Points1" in kwargs.keys():
    #         # first axis in the x axis
    #         Points1 = kwargs["Points1"]
    #
    #         vmax = np.max(Points1[:, 1:])
    #         vmin = np.min(Points1[:, 1:])
    #
    #         vmax = max(Points[:, 1].max(), vmax)
    #         vmin = min(Points[:, 1].min(), vmin)
    #
    #     else:
    #         vmax = max(Points)
    #         vmin = min(Points)
    #
    #     vmaxnew = PointMaxSize
    #     vminnew = PointMinSize
    #
    #     Points_scaled = [
    #         Scale.rescale(x, vmin, vmax, vminnew, vmaxnew) for x in Points[:, 1]
    #     ]
    #     f1 = np.ones(shape=(len(Points))) * PointsY
    #     scatter = ax2.scatter(
    #         Points[:, 0],
    #         f1,
    #         zorder=1,
    #         c=color1,
    #         s=Points_scaled,
    #         label="Model 1 Output 2",
    #     )
    #
    #     if "Points1" in kwargs.keys():
    #         row_points, col_points = np.shape(Points1)
    #         PointsY1 = kwargs["PointsY1"]
    #         f2 = np.ones_like(Points1[:, 1:])
    #
    #         for i in range(col_points - 1):
    #             Points1_scaled = [
    #                 Scale.rescale(x, vmin, vmax, vminnew, vmaxnew)
    #                 for x in Points1[:, i]
    #             ]
    #             f2[:, i] = PointsY1[i]
    #
    #             ax2.scatter(
    #                 Points1[:, 0],
    #                 f2[:, i],
    #                 zorder=1,
    #                 c=color2,
    #                 s=Points1_scaled,
    #                 label="Model 2 Output 2",
    #             )
    #
    #     # produce a legend with the unique colors from the scatter
    #     legend1 = ax2.legend(
    #         *scatter.legend_elements(), bbox_to_anchor=(1.1, 0.2)
    #     )  # loc="lower right", title="RIM"
    #
    #     ax2.add_artist(legend1)
    #
    #     # produce a legend with a cross section of sizes from the scatter
    #     handles, labels = scatter.legend_elements(
    #         prop="sizes", alpha=0.6, num=LegendNum
    #     )
    #     # L = [vminnew] + [float(i[14:-2]) for i in labels] + [vmaxnew]
    #     L = [float(i[14:-2]) for i in labels]
    #     labels1 = [
    #         round(Scale.rescale(x, vminnew, vmaxnew, vmin, vmax) / 1000) for x in L
    #     ]
    #
    #     legend2 = ax2.legend(
    #         handles, labels1, bbox_to_anchor=LegendLoc, title=PointLegendTitle
    #     )
    #     ax2.add_artist(legend2)
    #
    #     ax1.set_ylim(Ylim)
    #     ax2.set_ylim(Y2lim)
    #     #
    #     ax1.set_ylabel("Output 1 (m)", fontsize=12)
    #     ax2.set_ylabel("Output 1 - Diff (m)", fontsize=12)
    #     ax1.set_xlabel(X_axis_label, fontsize=12)
    #     ax1.xaxis.set_minor_locator(plt.MaxNLocator(10))
    #     ax1.tick_params(which="minor", length=5)
    #     fig.legend(
    #         loc="lower center",
    #         bbox_to_anchor=(1.3, 0.3),
    #         bbox_transform=ax1.transAxes,
    #         fontsize=10,
    #     )
    #     plt.rcParams.update({"ytick.major.size": 3.5})
    #     plt.rcParams.update({"font.size": 12})
    #     plt.title("Model Output Comparison", fontsize=15)
    #
    #     plt.subplots_adjust(right=0.7)
    #     # plt.tight_layout()
    #
    #     return (ax1, ax2), fig
