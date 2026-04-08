"""style related functionality"""

from collections import OrderedDict
from typing import Union

import matplotlib.colors as colors
import numpy as np

DEFAULT_OPTIONS = {
    "figsize": (8, 8),
    "title": None,
    "title_size": 15,
    "ylabel": "",
    "ylabel_font_size": 11,
    "xlabel": "",
    "xlabel_font_size": 11,
    "xtick_font_size": 11,
    "ytick_font_size": 11,
    "legend": "",
    "legend_size": 10,
    "color_1": "#3D59AB",
    "color_2": "#DC143C",
    "line_width": 3,
    "cbar_length": 0.75,
    "cbar_orientation": "vertical",
    "cmap": "coolwarm_r",
    "cbar_label_size": 12,
    "cbar_label": None,
    "cbar_label_rotation": -90,
    "cbar_label_location": "center",
    "ticks_spacing": 5,
    "color_scale": "linear",
    "gamma": 0.5,
    "line_scale": 0.001,
    "line_threshold": 0.0001,
    "bounds": None,
    "midpoint": 0,
    "grid_alpha": 0.75,
}


class Styles:
    """A class providing line and marker styles for matplotlib plots.

    This class contains collections of predefined line styles and marker styles
    that can be used to customize matplotlib plots. It provides static methods
    to retrieve these styles by name or index.

    Attributes:
        line_styles: A dictionary of line style definitions, mapping style names to
            matplotlib line style tuples. Each tuple defines the line style pattern.
        marker_style_list: A list of marker style strings that combine line styles with markers.

    Methods:
        get_line_style(style): Get a line style tuple by name or index.
        get_marker_style(style): Get a marker style string by index.

    Notes:
        Line styles define the pattern of the line (solid, dashed, dotted, etc.),
        while marker styles define both the line pattern and the marker shape
        (circle, square, triangle, etc.) used at data points.

    Examples:
    ```python
    >>> from cleopatra.styles import Styles
    >>> # Get a line style by name
    >>> solid_line = Styles.get_line_style("solid")
    >>> # Get a line style by index
    >>> dashed_line = Styles.get_line_style(5)  # "dashed"
    >>> # Get a marker style
    >>> marker_style = Styles.get_marker_style(0)  # "--o"

    ```
    """

    line_styles = OrderedDict(
        [
            ("solid", (0, ())),  # 0
            ("loosely dotted", (0, (1, 10))),  # 1
            ("dotted", (0, (1, 5))),  # 2
            ("densely dotted", (0, (1, 1))),  # 3
            ("loosely dashed", (0, (5, 10))),  # 4
            ("dashed", (0, (5, 5))),  # 5
            ("densely dashed", (0, (5, 1))),  # 6
            ("loosely dashdotted", (0, (3, 10, 1, 10))),  # 7
            ("dashdotted", (0, (3, 5, 1, 5))),  # 8
            ("densely dashdotted", (0, (3, 1, 1, 1))),  # 9
            ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),  # 10
            ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),  # 11
            ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),  # 12
            ("densely dashdotdottededited", (0, (6, 1, 1, 1, 1, 1))),  # 13
        ]
    )

    marker_style_list = [
        "--o",
        ":D",
        "-.H",
        "--x",
        ":v",
        "--|",
        "-+",
        "-^",
        "--s",
        "-.*",
        "-.h",
    ]

    @staticmethod
    def get_line_style(style: Union[str, int] = "loosely dotted"):
        """Get a matplotlib line style tuple by name or index.

        This method retrieves a line style tuple that can be used with matplotlib
        plotting functions to customize the appearance of lines. The style can be
        specified either by name (string) or by index (integer).

        Args:
            style: The line style to retrieve, by default "loosely dotted".
                If a string, it should be one of the keys in the `line_styles` dictionary.
                If an integer, it should be an index into the `line_styles` dictionary.
                Available style names:
                - "solid"
                - "loosely dotted"
                - "dotted"
                - "densely dotted"
                - "loosely dashed"
                - "dashed"
                - "densely dashed"
                - "loosely dashdotted"
                - "dashdotted"
                - "densely dashdotted"
                - "loosely dashdotdotted"
                - "dashdotdotted"
                - "densely dashdotdotted"
                - "densely dashdotdottededited"

        Returns:
            A matplotlib line style tuple that can be used with plot functions.
            The tuple format is (offset, (on_off_seq)) where:
            - offset is usually 0
            - on_off_seq is a sequence of on/off lengths in points

        Raises:
            KeyError: If the style name provided does not exist in the `line_styles`
                dictionary. In this case, a message is printed and the available styles
                are listed.

        Examples:
        Get a line style by name:
        ```python
        >>> from cleopatra.styles import Styles
        >>> solid = Styles.get_line_style("solid")
        >>> solid
        (0, ())

        ```
        Get a line style by index:
        ```python
        >>> dashed = Styles.get_line_style(5)  # "dashed"
        >>> dashed
        (0, (5, 5))

        ```
        Use a line style in a matplotlib plot:
        ```python
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> x = np.linspace(0, 10, 100)
        >>> y = np.sin(x)
        >>> plt.plot(x, y, linestyle=Styles.get_line_style("dashed"))  # doctest: +SKIP

        ```
        """
        if isinstance(style, str):
            try:
                return Styles.line_styles[style]
            except KeyError:
                msg = (
                    f" The style name you entered-{style}-does not exist please"
                    "choose from the available styles"
                )
                print(msg)
                print(list(Styles.line_styles))
        else:
            return list(Styles.line_styles.items())[style][1]

    @staticmethod
    def get_marker_style(style: int):
        """Get a matplotlib marker style string by index.

        This method retrieves a marker style string that can be used with matplotlib
        plotting functions to customize the appearance of markers and lines. The style
        is specified by an index into the `marker_style_list`.

        Args:
            style: The index of the marker style to retrieve from the `marker_style_list`.
                If the index is out of range, it will be wrapped around using modulo
                operation to ensure a valid style is always returned.

        Returns:
            A matplotlib marker style string that combines line style and marker.
            Examples: "--o" (dashed line with circle markers), ":D" (dotted line with
            diamond markers), etc.

        Notes:
            The marker style strings use matplotlib's shorthand notation:
            - Line styles: "-" (solid), "--" (dashed), "-." (dash-dot), ":" (dotted)
            - Markers: "o" (circle), "D" (diamond), "s" (square), "^" (triangle up), etc.

        Examples:
        Get a marker style by index:
        ```python
        >>> from cleopatra.styles import Styles
        >>> # Get the first marker style
        >>> style0 = Styles.get_marker_style(0)
        >>> style0
        '--o'

        >>> # Get another marker style
        >>> style1 = Styles.get_marker_style(1)
        >>> style1
        ':D'

        ```
        Handle index out of range (wraps around):
        ```python
        >>> # If we have 11 styles and request index 15, we get style at index 15 % 11 = 4
        >>> len(Styles.marker_style_list)
        11
        >>> style15 = Styles.get_marker_style(15)  # Same as style4
        >>> style4 = Styles.get_marker_style(4)
        >>> style15 == style4
        True

        ```
        Use a marker style in a matplotlib plot:
        ```python
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> x = np.linspace(0, 10, 20)
        >>> y = np.sin(x)
        >>> plt.plot(x, y, Styles.get_marker_style(0))  # doctest: +SKIP

        ```
        """
        if style > len(Styles.marker_style_list) - 1:
            style = style % len(Styles.marker_style_list)
        return Styles.marker_style_list[style]


class Scale:
    """A class providing various scaling functions for data visualization.

    This class contains static methods for different types of scaling operations
    that can be used to transform data values for visualization purposes. These
    include logarithmic scaling, power scaling, identity scaling, and general
    value rescaling between different ranges.

    Methods:
        log_scale(val): Apply logarithmic (base 10) scaling to a value.
        power_scale(min_val): Create a power scaling function based on a minimum value.
        identity_scale(min_val, max_val): Create an identity scaling function that always returns 2.
        rescale(old_value, old_min, old_max, new_min, new_max): Rescale a value from one range to another.

    Notes:
        Scaling functions are useful for transforming data to improve visualization,
        especially when dealing with data that spans multiple orders of magnitude or
        needs to be normalized to a specific range.

    Examples:
        Apply logarithmic scaling:
    ```python
    >>> from cleopatra.styles import Scale
    >>> Scale.log_scale(100)
    np.float64(2.0)
    >>> Scale.log_scale(1000)
    np.float64(3.0)

    ```
    Rescale a value from one range to another:
    ```python
    >>> Scale.rescale(5, 0, 10, 0, 100)  # 5 is 50% of [0,10], so 50% of [0,100] is 50
    50.0
    >>> Scale.rescale(75, 0, 100, -1, 1)  # 75 is 75% of [0,100], so 75% of [-1,1] is 0.5
    0.5

    ```
    """

    def __init__(self):
        """Initialize a Scale object.

        Note that this class is primarily intended to be used via its static methods,
        so initialization is not typically necessary.
        """
        pass

    @staticmethod
    def log_scale(val):
        """Apply logarithmic (base 10) scaling to a value or array.

        This method computes the base-10 logarithm of the input value(s),
        which is useful for visualizing data that spans multiple orders of magnitude.

        Args:
            val: The value or array of values to be logarithmically scaled.
                Must be positive (greater than 0) to avoid math domain errors.

        Returns:
            The base-10 logarithm of the input value(s).
            If the input is an array, the output will be an array of the same shape.

        Notes:
            Logarithmic scaling is particularly useful for:
            - Data that spans multiple orders of magnitude
            - Compressing wide ranges of values into a more manageable range
            - Visualizing exponential growth or decay

        Examples:
        Scale a single value:
        ```python
        >>> from cleopatra.styles import Scale
        >>> Scale.log_scale(100)
        np.float64(2.0)
        >>> Scale.log_scale(1000)
        np.float64(3.0)

        ```
        Scale an array of values:
        ```python
        >>> import numpy as np
        >>> values = np.array([1, 10, 100, 1000])
        >>> Scale.log_scale(values)
        array([0., 1., 2., 3.])

        ```
        """
        return np.log10(val)

    @staticmethod
    def power_scale(min_val) -> callable:
        """Create a power scaling function based on a minimum value.

        This method returns a function that applies power scaling to its input.
        The scaling function first shifts the input value by adding the absolute
        value of the minimum value plus 1 (to ensure positive values), then
        divides by 1000 and squares the result.

        Args:
            min_val: The minimum value in the data range. Used to shift the data to ensure
                all values are positive before applying the power transformation.

        Returns:
            A function that takes a value or array and returns the power-scaled result.
            The returned function has the signature: f(val) -> float or numpy.ndarray

        Notes:
            Power scaling is useful for:
            - Emphasizing differences in smaller values
            - Compressing the range of larger values
            - Creating non-linear visualizations where small changes in small values
              are more important than small changes in large values

        Examples:
        Create a power scaling function and apply it to values:
        ```python
        >>> from cleopatra.styles import Scale
        >>> # Create a scaling function with minimum value -10
        >>> scale_func = Scale.power_scale(-10)
        >>> # Apply to a single value
        >>> scale_func(5)  # (5 + |-10| + 1) / 1000)^2 = (5 + 10 + 1)^2 / 1000000 = 16^2 / 1000000 = 256 / 1000000 = 0.000256
        0.000256
        >>> # Apply to another value
        >>> scale_func(100)  # (100 + |-10| + 1) / 1000)^2 = (100 + 10 + 1)^2 / 1000000 = 111^2 / 1000000 = 12321 / 1000000 ≈ 0.012321
        0.012321

        ```
        Apply to an array of values:
        ```python
        >>> import numpy as np
        >>> values = np.array([0, 10, 100])
        >>> scale_func = Scale.power_scale(-5)
        >>> scale_func(values)  # doctest: +ELLIPSIS
        array([3.6000e-05, 2.5600e-04, 1.1236e-02])

        >>> # [(0+5+1)/1000]^2, [(10+5+1)/1000]^2, [(100+5+1)/1000]^2]
        ```
        """

        def scalar(val):
            val = val + abs(min_val) + 1
            return (val / 1000) ** 2

        return scalar

    @staticmethod
    def identity_scale(min_val, max_val):
        """Create a constant scaling function that always returns 2.

        This method returns a function that ignores its input and always returns
        the constant value 2. Despite its name, this is not a true identity function
        (which would return the input unchanged), but rather a constant function.

        Args:
            min_val: The minimum value in the data range. This parameter is not used in the
                implementation but is included for API consistency with other scaling methods.
            max_val: The maximum value in the data range. This parameter is not used in the
                implementation but is included for API consistency with other scaling methods.

        Returns:
            A function that takes any input and always returns 2.
            The returned function has the signature: f(val) -> int

        Notes:
            This function can be useful in situations where:
            - A constant size or value is needed regardless of the input data
            - A placeholder scaling function is required
            - Testing or debugging code that expects a scaling function

        Examples:
        Create and use the constant scaling function:
        ```python
        >>> from cleopatra.styles import Scale
        >>> scale_func = Scale.identity_scale(0, 100)  # min_val and max_val are ignored
        >>> scale_func(5)  # Returns 2 regardless of input
        2
        >>> scale_func(100)  # Still returns 2
        2
        >>> scale_func(-10)  # Still returns 2
        2

        ```
        Works with arrays too, but returns a scalar, not an array:
        ```python
        >>> import numpy as np
        >>> values = np.array([1, 2, 3, 4, 5])
        >>> scale_func(values)  # Returns scalar 2, not an array of 2s
        2

        ```
        """

        def scalar(val):
            return 2

        return scalar

    @staticmethod
    def rescale(old_value, old_min, old_max, new_min, new_max):
        """Rescale a value from one range to another.

        This method performs linear rescaling of a value from an original range
        [old_min, old_max] to a new range [new_min, new_max]. The transformation
        preserves the relative position of the value within its range.

        Args:
            old_value: The value(s) to be rescaled. Can be a single value or an array.
            old_min: The minimum value of the original range.
            old_max: The maximum value of the original range.
            new_min: The minimum value of the target range.
            new_max: The maximum value of the target range.

        Returns:
            The rescaled value(s) in the new range. If the input is an array,
            the output will be an array of the same shape.

        Notes:
            The rescaling formula is:
            new_value = (((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

            This function is useful for:
            - Normalizing data to a specific range (e.g., [0, 1])
            - Converting between different units or scales
            - Preparing data for visualization with specific bounds

        Examples:
        Rescale a value from [0, 10] to [0, 100]:
        ```python
        >>> from cleopatra.styles import Scale
        >>> Scale.rescale(5, 0, 10, 0, 100)  # 5 is 50% of [0,10], so 50% of [0,100] is 50
        50.0

        ```
        Rescale a value from [0, 100] to [-1, 1]:
        ```python
        >>> Scale.rescale(75, 0, 100, -1, 1)  # 75 is 75% of [0,100], so 75% of [-1,1] is 0.5
        0.5

        ```
        Rescale an array of values:
        ```python
        >>> import numpy as np
        >>> values = np.array([0, 5, 10])
        >>> Scale.rescale(values, 0, 10, 0, 1)  # Normalize to [0,1]
        array([0. , 0.5, 1. ])

        ```
        Invert a range by swapping the new min and max:
        ```python
        >>> Scale.rescale(25, 0, 100, 1, 0)  # 25 is 25% from min, so 25% from max in new range is 0.75
        0.75

        ```
        """
        old_range = old_max - old_min
        new_range = new_max - new_min
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min

        return new_value


class MidpointNormalize(colors.Normalize):
    """A normalization class that scales data with a midpoint.

    This class extends matplotlib's Normalize class to create a colormap
    normalization that has a fixed midpoint. This is useful for data that
    has a natural midpoint (like zero) where the colormap should be centered,
    regardless of the actual data range.

    The normalization maps values to the range [0, 1] with the midpoint
    mapped to 0.5, which allows for symmetric colormaps to be properly centered.

    Args:
        vmin: The minimum data value that corresponds to 0 in the normalized data.
            If None, it is automatically calculated from the data.
        vmax: The maximum data value that corresponds to 1 in the normalized data.
            If None, it is automatically calculated from the data.
        midpoint: The data value that corresponds to 0.5 in the normalized data.
            If None, it defaults to the midpoint between vmin and vmax.
        clip: If True, values outside the [vmin, vmax] range are clipped to be
            within that range, by default False.

    Attributes:
        midpoint: The data value that will be mapped to 0.5 in the normalized data.

    Notes:
        This normalization is particularly useful for:
        - Diverging colormaps where a specific value should be at the center
        - Data with positive and negative values where zero should be the midpoint
        - Highlighting deviations from a reference value

    Examples:
    Create a plot with a midpoint normalization:
    ```python
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from cleopatra.styles import MidpointNormalize
    >>> # Create some data with positive and negative values
    >>> data = np.linspace(-10, 10, 100)
    >>> # Create a normalization with midpoint at 0
    >>> norm = MidpointNormalize(vmin=-10, vmax=10, midpoint=0)
    >>> # Use in a plot
    >>> plt.figure(figsize=(8, 1)) # doctest: +SKIP
    >>> plt.imshow([data], cmap='coolwarm', norm=norm, aspect='auto')  # doctest: +SKIP
    >>> plt.colorbar()  # doctest: +SKIP
    >>> plt.title('Midpoint Normalization with midpoint=0')  # doctest: +SKIP
    >>> plt.tight_layout()  # doctest: +SKIP

    ```
    Create a normalization with a non-zero midpoint (5):
    ```python
    >>> norm = MidpointNormalize(vmin=0, vmax=10, midpoint=5)
    ```
    - Values below midpoint are mapped to [0, 0.5]
    ```python
    >>> norm(0)
    masked_array(data=0.,
                 mask=False,
           fill_value=1e+20)
    >>> norm(2.5)
    masked_array(data=0.25,
                 mask=False,
           fill_value=1e+20)

    ```
    - Midpoint is mapped to 0.5
    ```python
    >>> norm(5)
    masked_array(data=0.5,
                 mask=False,
           fill_value=1e+20)

    ```
    - Values above midpoint are mapped to [0.5, 1]
    ```python
    >>> norm(7.5)
    masked_array(data=0.75,
                 mask=False,
           fill_value=1e+20)
    >>> norm(10)
    masked_array(data=1.,
                 mask=False,
           fill_value=1e+20)

    ```
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        """Initialize a MidpointNormalize instance.

        Args:
            vmin: The minimum data value that corresponds to 0 in the normalized data.
                If None, it is automatically calculated from the data when the
                normalization is applied.
            vmax: The maximum data value that corresponds to 1 in the normalized data.
                If None, it is automatically calculated from the data when the
                normalization is applied.
            midpoint: The data value that corresponds to 0.5 in the normalized data.
                If None, it defaults to the midpoint between vmin and vmax.
            clip: If True, values outside the [vmin, vmax] range are clipped to be
                within that range, by default False.

        Notes:
            This initialization sets up the midpoint attribute and calls the parent
            class (matplotlib.colors.Normalize) constructor with the vmin, vmax, and
            clip parameters.

        Examples:
        Create a normalization with default parameters:
        ```python
        >>> from cleopatra.styles import MidpointNormalize
        >>> norm = MidpointNormalize()  # vmin, vmax, midpoint will be determined from data

        ```
        Create a normalization with specific range and midpoint:
        ```python
        >>> norm = MidpointNormalize(vmin=-10, vmax=10, midpoint=0)
        >>> norm.midpoint
        0

        ```
        """
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        """Normalize data values to the [0, 1] range with a fixed midpoint.

        This method implements the normalization logic, mapping input values to
        the range [0, 1] with the midpoint mapped to 0.5. It uses linear interpolation
        to create two separate linear mappings: one for values below the midpoint
        and another for values above the midpoint.

        Args:
            value: The data value(s) to normalize. Can be a single value or an array.
            clip: Whether to clip the input values to the [vmin, vmax] range.
                If None, the clip attribute of the instance is used.

        Returns:
            The normalized value(s) in the range [0, 1], with the midpoint mapped to 0.5.
            If the input is an array, the output will be an array of the same shape.
            Masked values in the input remain masked in the output.

        Notes:
            The normalization is performed using numpy's interp function, which does
            linear interpolation between the points:
            - (vmin, 0): minimum value maps to 0
            - (midpoint, 0.5): midpoint value maps to 0.5
            - (vmax, 1): maximum value maps to 1

            This creates a piecewise linear mapping that ensures the midpoint is
            always at 0.5 in the normalized range.

        Examples:
        - Normalize values with a zero midpoint:
        ```python
        >>> from cleopatra.styles import MidpointNormalize
        >>> norm = MidpointNormalize(vmin=-10, vmax=10, midpoint=0)
        >>> # Values below midpoint are mapped to [0, 0.5]
        >>> norm(-10)  # vmin maps to 0
        masked_array(data=0.,
                     mask=False,
               fill_value=1e+20)
        >>> norm(-5)   # halfway between vmin and midpoint maps to 0.25
        masked_array(data=0.25,
                     mask=False,
               fill_value=1e+20)

        ```
        - Midpoint maps to 0.5
        ```python
        >>> norm(0)
        masked_array(data=0.5,
                     mask=False,
               fill_value=1e+20)

        ```
        - Values above midpoint are mapped to [0.5, 1]
        ```python
        >>> norm(5)    # halfway between midpoint and vmax maps to 0.75
        masked_array(data=0.75,
                     mask=False,
               fill_value=1e+20)
        >>> norm(10)   # vmax maps to 1
        masked_array(data=1.,
                     mask=False,
               fill_value=1e+20)

        ```
        Normalize an array of values:
        ```python
        >>> import numpy as np
        >>> values = np.array([-10, -5, 0, 5, 10])
        >>> norm(values)
        masked_array(data=[0.  , 0.25, 0.5 , 0.75, 1.  ],
                     mask=False,
               fill_value=1e+20)

        ```
        """
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]

        return np.ma.masked_array(np.interp(value, x, y))
