"""style related functionality"""

from __future__ import annotations

import copy
from collections import OrderedDict
from enum import StrEnum
from typing import Callable, Sequence

import matplotlib as mpl
import matplotlib.colors as colors
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.container import BarContainer
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


class ColorScale(StrEnum):
    """Accepted values for the `color_scale` option of cleopatra glyphs.

    Members are plain strings (`StrEnum`), so `ColorScale.LINEAR == "linear"`
    holds and any code that treats the value as a string keeps working
    whether the caller passes the enum member or the bare string. Lookup is
    case-insensitive: `ColorScale("Linear") is ColorScale.LINEAR`.

    Examples:
        - The members behave like their string values:
            ```python
            >>> from cleopatra.styles import ColorScale
            >>> ColorScale.LINEAR == "linear"
            True
            >>> str(ColorScale.POWER)
            'power'

            ```
        - Construction is case-insensitive; bad values raise `ValueError`:
            ```python
            >>> from cleopatra.styles import ColorScale
            >>> ColorScale("Boundary-Norm") is ColorScale.BOUNDARY_NORM
            True
            >>> ColorScale("nope")
            Traceback (most recent call last):
                ...
            ValueError: 'nope' is not a valid ColorScale

            ```
    """

    LINEAR = "linear"
    POWER = "power"
    SYM_LOGNORM = "sym-lognorm"
    BOUNDARY_NORM = "boundary-norm"
    MIDPOINT = "midpoint"

    @classmethod
    def _missing_(cls, value: object) -> "ColorScale | None":
        """Resolve a case-insensitive string to a member, else `None`.

        Called by `enum.Enum` when a direct value lookup fails. Only
        strings are coerced (lower-cased and re-matched); anything else
        (an int, `None`, …) returns `None` so `ColorScale(value)`
        raises the usual `ValueError`.

        Args:
            value: The value passed to `ColorScale(value)` that did not
                match a member directly.

        Returns:
            ColorScale or None: The matching member, or `None` to let
                `Enum` raise `ValueError`.
        """
        if isinstance(value, str):
            lowered = value.lower()
            for member in cls:
                if member.value == lowered:
                    return member
        return None


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
    "scheme": None,
    "k": 5,
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
    def get_line_style(style: str | int = "loosely dotted") -> tuple[int, tuple[int, ...]] | None:
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
    def get_marker_style(style: int) -> str:
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
    def log_scale(val: float | np.ndarray) -> np.floating | np.ndarray:
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
    def power_scale(min_val: float) -> Callable:
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
    def identity_scale(min_val: float, max_val: float) -> Callable:
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
    def rescale(old_value: float | np.ndarray, old_min: float, old_max: float, new_min: float, new_max: float) -> float | np.ndarray:
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


#: Accepted values for the `size_scale` of `resolve_sizes` — the transform
#: applied to the magnitudes before they are linearly rescaled into the output
#: range. `"sqrt"` matches perceived marker *area*; `"log"` suits magnitudes
#: that span orders of magnitude.
SIZE_SCALES = ("linear", "log", "sqrt")


def resolve_sizes(
    values: np.ndarray | Sequence[float],
    out_min: float,
    out_max: float,
    scale: str = "linear",
) -> np.ndarray:
    """Map per-item magnitudes to a visual size range.

    The reusable value→size primitive shared by size-encoding glyphs: it
    turns a per-item magnitude array into an array of visual sizes spanning
    `[out_min, out_max]`, optionally pre-transforming the magnitudes
    (`"log"` / `"sqrt"`) before the linear rescale. `ScatterGlyph` uses it
    for marker area (`s`); a future `FlowGlyph` can reuse it for line width.
    The linear rescale itself is delegated to `Scale.rescale`, so this never
    re-implements the range mapping.

    The mapping is monotonic in the input, so larger magnitudes always map
    to larger sizes. When every (finite) magnitude is equal, there is no
    spread to encode and the midpoint of the output range is returned for
    each item.

    Args:
        values: The per-item magnitudes to map. Non-finite entries are kept
            in place in the output (mapped from a domain that ignores them)
            — callers that pre-filter their data will not hit this.
        out_min: The smallest output size (maps to the minimum magnitude).
        out_max: The largest output size (maps to the maximum magnitude).
        scale: The pre-transform: `"linear"` (identity), `"log"`
            (`log10`, requires strictly positive magnitudes), or `"sqrt"`
            (requires non-negative magnitudes). Case-insensitive. Default
            is `"linear"`.

    Returns:
        np.ndarray: The mapped sizes, the same shape as `values`, spanning
            `[out_min, out_max]` for non-degenerate input.

    Raises:
        ValueError: If `values` has no finite entries, if `scale` is not one
            of `SIZE_SCALES`, if `scale="log"` and any magnitude is
            non-positive, or if `scale="sqrt"` and any magnitude is negative.

    Examples:
        - Linear mapping spans the output range, smallest→`out_min`:
            ```python
            >>> import numpy as np
            >>> from cleopatra.styles import resolve_sizes
            >>> sizes = resolve_sizes(np.array([0.0, 5.0, 10.0]), 10.0, 200.0)
            >>> [float(s) for s in sizes]
            [10.0, 105.0, 200.0]

            ```
        - The mapping is monotonic, so ranking is preserved:
            ```python
            >>> import numpy as np
            >>> from cleopatra.styles import resolve_sizes
            >>> sizes = resolve_sizes(np.array([3.0, 1.0, 2.0]), 0.0, 1.0)
            >>> bool(sizes[1] < sizes[2] < sizes[0])
            True

            ```
        - All-equal magnitudes map to the output midpoint:
            ```python
            >>> import numpy as np
            >>> from cleopatra.styles import resolve_sizes
            >>> [float(s) for s in resolve_sizes(np.full(3, 4.0), 10.0, 50.0)]
            [30.0, 30.0, 30.0]

            ```
    """
    values = np.asarray(values, dtype=float)
    scale = str(scale).lower()
    if scale not in SIZE_SCALES:
        valid = ", ".join(repr(s) for s in SIZE_SCALES)
        raise ValueError(
            f"Invalid size_scale {scale!r}. Expected one of {valid}."
        )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(
            "Cannot resolve sizes: `values` has no finite entries."
        )
    if scale == "log":
        if np.any(finite <= 0):
            raise ValueError(
                "size_scale='log' requires strictly positive magnitudes."
            )
        transformed = Scale.log_scale(values)
        domain = Scale.log_scale(finite)
    elif scale == "sqrt":
        if np.any(finite < 0):
            raise ValueError(
                "size_scale='sqrt' requires non-negative magnitudes."
            )
        transformed = np.sqrt(values)
        domain = np.sqrt(finite)
    else:  # linear
        transformed = values
        domain = finite

    lo, hi = float(domain.min()), float(domain.max())
    if hi == lo:
        midpoint = (out_min + out_max) / 2.0
        return np.full(values.shape, midpoint, dtype=float)
    return Scale.rescale(transformed, lo, hi, out_min, out_max)


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

    def __init__(self, vmin: float | None = None, vmax: float | None = None, midpoint: float | None = None, clip: bool = False) -> None:
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

    def __call__(self, value: float | np.ndarray, clip: bool | None = None) -> np.ma.MaskedArray:
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


def disjoint_legend(
    ax: Axes,
    colors: Sequence,
    labels: Sequence[str],
    *,
    edgecolor: str = "none",
    **kwargs,
) -> Legend:
    """Attach a categorical (disjoint) swatch legend to an axes.

    Builds one filled rectangle (`matplotlib.patches.Patch`) per
    category and registers them as a legend on `ax`. This is the
    discrete counterpart to a colorbar: use it when categories are
    nominal/disjoint (land-cover classes, region names, ...) rather
    than samples of a continuous scale, where a colorbar would imply a
    false ordering.

    Args:
        ax: The axes the legend is attached to.
        colors: One color per category, in any matplotlib color form
            (name, hex, or RGB(A) tuple). Must be the same length as
            `labels`.
        labels: The category label drawn next to each swatch. Must be
            the same length as `colors`.
        edgecolor: Outline color for every swatch. Defaults to
            `"none"` (no border), matching cleopatra's flat look.
        **kwargs: Forwarded verbatim to `Axes.legend` (e.g. `title`,
            `loc`, `ncol`, `bbox_to_anchor`, `fontsize`).

    Returns:
        Legend: The created legend artist, already added to `ax`.

    Raises:
        ValueError: If `colors` and `labels` have different lengths.

    Examples:
        - Build a three-class legend and read back its labels:
            ```python
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.styles import disjoint_legend
            >>> fig, ax = plt.subplots()
            >>> legend = disjoint_legend(
            ...     ax,
            ...     ["#1b9e77", "#d95f02", "#7570b3"],
            ...     ["water", "urban", "forest"],
            ... )
            >>> [t.get_text() for t in legend.get_texts()]
            ['water', 'urban', 'forest']

            ```
        - Forward `Axes.legend` kwargs such as a title and column count:
            ```python
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.styles import disjoint_legend
            >>> fig, ax = plt.subplots()
            >>> legend = disjoint_legend(
            ...     ax, ["red", "blue"], ["hot", "cold"], title="Class", ncol=2
            ... )
            >>> legend.get_title().get_text()
            'Class'

            ```
        - Mismatched lengths raise `ValueError`:
            ```python
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.styles import disjoint_legend
            >>> fig, ax = plt.subplots()
            >>> disjoint_legend(ax, ["red", "blue"], ["only-one"])
            Traceback (most recent call last):
                ...
            ValueError: colors and labels must have the same length, got 2 and 1.

            ```
    """
    colors = list(colors)
    labels = list(labels)
    if len(colors) != len(labels):
        raise ValueError(
            "colors and labels must have the same length, got "
            f"{len(colors)} and {len(labels)}."
        )
    handles = [
        Patch(facecolor=color, edgecolor=edgecolor, label=label)
        for color, label in zip(colors, labels)
    ]
    return ax.legend(handles=handles, **kwargs)


def size_legend(
    ax: Axes,
    marker_sizes: Sequence[float],
    labels: Sequence[str],
    *,
    color: str = "0.4",
    marker: str = "o",
    **kwargs,
) -> Legend:
    """Attach a legend whose marker *sizes* encode magnitude.

    The size counterpart to `disjoint_legend`: where that varies the swatch
    *colour*, this varies the marker *size*, so it is the right legend for a
    bubble / size-scaled scatter (e.g. `ScatterGlyph(..., sizes=...)`). One
    proxy marker is drawn per entry, sized to match the points it
    represents.

    `marker_sizes` are scatter-style **areas** (points², the same unit as a
    glyph's resolved `s`); each is converted to the matplotlib `Line2D`
    `markersize` (a diameter in points) via `sqrt`, so the swatches match
    the plotted points visually.

    Args:
        ax: The axes the legend is attached to.
        marker_sizes: The representative marker areas (points²), one per
            legend entry. Must be the same length as `labels`.
        labels: The text drawn next to each marker. Must be the same length
            as `marker_sizes`.
        color: Fill colour for every proxy marker. Defaults to a neutral
            grey (`"0.4"`) because the legend encodes size, not colour.
        marker: The marker style for the proxies. Defaults to `"o"`.
        **kwargs: Forwarded verbatim to `Axes.legend` (e.g. `title`, `loc`,
            `labelspacing`, `bbox_to_anchor`).

    Returns:
        Legend: The created legend artist, already added to `ax`.

    Raises:
        ValueError: If `marker_sizes` and `labels` have different lengths.

    Examples:
        - Build a three-entry size legend and read back its labels:
            ```python
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.styles import size_legend
            >>> fig, ax = plt.subplots()
            >>> legend = size_legend(ax, [20.0, 100.0, 200.0], ["low", "mid", "high"])
            >>> [t.get_text() for t in legend.get_texts()]
            ['low', 'mid', 'high']

            ```
        - Larger areas produce larger proxy markers (diameters in points):
            ```python
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.styles import size_legend
            >>> fig, ax = plt.subplots()
            >>> legend = size_legend(ax, [16.0, 64.0], ["small", "big"])
            >>> handles = legend.legend_handles
            >>> round(handles[0].get_markersize(), 1), round(handles[1].get_markersize(), 1)
            (4.0, 8.0)

            ```
    """
    marker_sizes = list(marker_sizes)
    labels = list(labels)
    if len(marker_sizes) != len(labels):
        raise ValueError(
            "marker_sizes and labels must have the same length, got "
            f"{len(marker_sizes)} and {len(labels)}."
        )
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=np.sqrt(max(size, 0.0)),
            label=label,
        )
        for size, label in zip(marker_sizes, labels)
    ]
    return ax.legend(handles=handles, **kwargs)


def colorbar_legend(mappable: ScalarMappable, ax: Axes = None, **kwargs) -> Colorbar:
    """Attach a continuous colorbar legend for a mappable.

    A thin, glyph-agnostic wrapper over `Figure.colorbar` for callers
    that already hold a mappable (the artist returned by
    `scatter` / `imshow` / `quiver` / a glyph's `plot`) and just want a
    matching colorbar. For full cleopatra colorbar styling (label size,
    location, shrink) use `Glyph.create_color_bar` instead; this helper
    is the minimal counterpart that sits alongside `disjoint_legend`
    and `histogram_legend`.

    Args:
        mappable: A `matplotlib.cm.ScalarMappable` (e.g. the result of
            `ax.scatter(..., c=values)`), carrying the cmap/norm to map.
        ax: Axes to steal space from for the colorbar. Defaults to the
            mappable's own axes. The parent figure is inferred from
            whichever axes is used.
        **kwargs: Forwarded to `Figure.colorbar` (e.g. `label`,
            `orientation`, `shrink`, `ticks`, `extend`).

    Returns:
        Colorbar: The created colorbar.

    Raises:
        ValueError: If no axes can be determined (the mappable is not
            attached to an axes and `ax` is None).

    Examples:
        - Build a colorbar for a coloured scatter and read its label:
            ```python
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.styles import colorbar_legend
            >>> fig, ax = plt.subplots()
            >>> sc = ax.scatter([0, 1, 2], [0, 1, 0], c=[10, 20, 30])
            >>> cbar = colorbar_legend(sc, ax, label="depth")
            >>> cbar.ax.get_ylabel()
            'depth'

            ```
    """
    parent_ax = ax if ax is not None else getattr(mappable, "axes", None)
    if parent_ax is None:
        raise ValueError(
            "Cannot determine an axes for the colorbar: pass `ax` or use a "
            "mappable already attached to an axes."
        )
    fig = parent_ax.figure
    return fig.colorbar(mappable, ax=parent_ax, **kwargs)


def histogram_legend(
    ax: Axes,
    values: np.ndarray | None = None,
    *,
    mappable: ScalarMappable | None = None,
    cmap=None,
    norm: colors.Normalize | None = None,
    bins: int = 20,
    orientation: str = "vertical",
    **bar_kwargs,
) -> BarContainer:
    """Draw a colour-mapped histogram as a distribution legend.

    Renders a histogram of `values` whose bars are coloured by the same
    colormap/norm used for the data, so the legend doubles as a
    distribution plot — the third legend style alongside the continuous
    colorbar and the categorical `disjoint_legend`. The colour mapping
    can be taken straight from a `mappable` (so it matches a glyph's
    plot exactly) or supplied explicitly via `cmap` / `norm`.

    Args:
        ax: Axes to draw the histogram on (typically a small companion
            axes beside the main plot).
        values: 1D data to histogram. Non-finite entries are dropped.
            Defaults to the mappable's array when `values` is None.
        mappable: Optional `ScalarMappable` to inherit `cmap`, `norm`,
            and (when `values` is None) the data array from.
        cmap: Colormap name or object. Falls back to the mappable's
            cmap, then to matplotlib's default. Ignored when a
            `mappable` provides one and `cmap` is None.
        norm: Normalization for mapping bin centres to colours. Falls
            back to the mappable's norm, then to a linear norm spanning
            the data.
        bins: Number of histogram bins. Default is 20.
        orientation: `"vertical"` (bars rise with count) or
            `"horizontal"` (bars extend rightwards). Default is
            `"vertical"`.
        **bar_kwargs: Forwarded to `Axes.bar` / `Axes.barh`
            (e.g. `edgecolor`, `alpha`).

    Returns:
        BarContainer: The bars drawn, one per bin.

    Raises:
        ValueError: If neither `values` nor a `mappable` with an array
            is provided, if there are no finite values, or if
            `orientation` is not `"vertical"` / `"horizontal"`.

    Examples:
        - Histogram legend from explicit values and a colormap:
            ```python
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.styles import histogram_legend
            >>> fig, ax = plt.subplots()
            >>> bars = histogram_legend(
            ...     ax, [0.0, 1.0, 1.0, 2.0, 2.0, 2.0], cmap="viridis", bins=3
            ... )
            >>> len(bars)
            3

            ```
        - Inherit cmap/norm/data straight from a mappable:
            ```python
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.styles import histogram_legend
            >>> fig, (ax, legend_ax) = plt.subplots(1, 2)
            >>> sc = ax.scatter([0, 1, 2, 3], [0, 1, 0, 1], c=[1, 2, 3, 4], cmap="plasma")
            >>> bars = histogram_legend(legend_ax, mappable=sc, bins=4)
            >>> len(bars)
            4

            ```
    """
    if orientation not in ("vertical", "horizontal"):
        raise ValueError(
            f"orientation must be 'vertical' or 'horizontal', got "
            f"{orientation!r}."
        )

    if values is None:
        if mappable is None or mappable.get_array() is None:
            raise ValueError(
                "Provide `values` or a `mappable` carrying a data array."
            )
        values = np.asarray(mappable.get_array()).ravel()
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite values to histogram.")

    if cmap is None and mappable is not None:
        cmap = mappable.cmap
    cmap_obj = mpl.colormaps[cmap] if isinstance(cmap, str) else (
        cmap if cmap is not None else mpl.colormaps[mpl.rcParams["image.cmap"]]
    )
    if norm is None and mappable is not None:
        # Copy so that mapping bin centres below cannot mutate the
        # caller's norm (an unscaled norm would otherwise be
        # autoscaled in place by the norm(centers) call). copy.copy
        # preserves the norm subtype (e.g. BoundaryNorm).
        norm = copy.copy(mappable.norm)

    counts, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    if norm is None:
        norm = colors.Normalize(vmin=float(edges[0]), vmax=float(edges[-1]))
    bar_colors = cmap_obj(norm(centers))

    if orientation == "vertical":
        return ax.bar(centers, counts, width=widths, color=bar_colors, **bar_kwargs)
    return ax.barh(centers, counts, height=widths, color=bar_colors, **bar_kwargs)


#: Classification schemes that need nothing beyond numpy. Anything outside
#: this set (the Jenks-family `"fisher_jenks"` / `"natural_breaks"`) is routed
#: to the optional, lazily-imported `cleopatra[classify]` extra (mapclassify)
#: so the default install stays numpy + matplotlib only.
NUMPY_SCHEMES = ("quantiles", "equal_interval", "percentiles", "std_mean")

#: Jenks-family schemes that require the optional `cleopatra[classify]` extra.
JENKS_SCHEMES = ("fisher_jenks", "natural_breaks")

#: Standard-deviation multiples used by the `"std_mean"` scheme (mean ± nσ).
#: Matches mapclassify's `StdMean` default; `k` is ignored for this scheme.
_STD_MEAN_MULTIPLES = (-2.0, -1.0, 0.0, 1.0, 2.0)


def classify(
    values: np.ndarray | Sequence[float],
    scheme: str | Sequence[float],
    k: int = 5,
) -> tuple[np.ndarray, colors.BoundaryNorm]:
    """Bin a continuous array into discrete colour classes.

    The shared building block behind categorical (classified) colouring:
    it turns a continuous data column into an array of bin edges plus a
    matching `matplotlib.colors.BoundaryNorm`, so any colour-by-value glyph
    can render a stepped colorbar / class legend instead of a continuous
    ramp. It is the classification counterpart to `Scale` and
    `MidpointNormalize`.

    The numpy-only schemes (no dependency beyond numpy) are:

    * `"quantiles"` — `k` equal-count classes via
      `np.quantile(values, np.linspace(0, 1, k + 1))`.
    * `"equal_interval"` — `k` equal-width classes spanning the data range.
    * `"percentiles"` — `k` equal-count classes via `np.percentile` on the
      same evenly-spaced probabilities; numerically equivalent to
      `"quantiles"` (it differs only in the `[0, 100]` vs `[0, 1]`
      convention) and is kept as a familiar alias.
    * `"std_mean"` — fixed breaks at `mean + nσ` for `n` in
      `(-2, -1, 0, 1, 2)`, clipped to the data range. `k` is **ignored**
      for this scheme (the number of classes follows from the multiples).

    The Jenks-family schemes `"fisher_jenks"` and `"natural_breaks"` are
    routed to the optional `cleopatra[classify]` extra (`mapclassify`),
    imported lazily so the default install never needs it.

    A non-string `scheme` is treated as an explicit, already-chosen
    sequence of bin edges (sorted ascending); `k` is ignored.

    Args:
        values: The data to classify. Non-finite entries (`NaN` / `inf`)
            are ignored when computing the edges. Can be any array-like.
        scheme: A scheme name (see above, case-insensitive) **or** an
            explicit sequence of bin edges to use verbatim.
        k: The number of classes for the count/width schemes. Must be
            `>= 1`. Default is 5. Ignored for `"std_mean"` and for an
            explicit edge sequence.

    Returns:
        tuple[np.ndarray, matplotlib.colors.BoundaryNorm]: The sorted,
            de-duplicated bin edges (length = classes + 1) and a
            `BoundaryNorm` built from them (with `ncolors=256`, matching
            the package's other boundary norms).

    Raises:
        ValueError: If `values` has no finite entries, if `k < 1`, if the
            (finite) data has no spread so fewer than two distinct edges
            result, or if `scheme` is an unrecognised name.
        ModuleNotFoundError: If a Jenks-family scheme is requested but the
            optional `cleopatra[classify]` extra (mapclassify) is not
            installed.

    Examples:
        - Equal-interval edges on a 0–10 ramp:
            ```python
            >>> import numpy as np
            >>> from cleopatra.styles import classify
            >>> edges, norm = classify(np.arange(11.0), "equal_interval", k=5)
            >>> [float(e) for e in edges]
            [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
            >>> [float(b) for b in norm.boundaries]
            [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

            ```
        - Quantile edges put equal counts in each class:
            ```python
            >>> import numpy as np
            >>> from cleopatra.styles import classify
            >>> edges, _ = classify(np.arange(100.0), "quantiles", k=4)
            >>> [float(e) for e in edges]
            [0.0, 24.75, 49.5, 74.25, 99.0]

            ```
        - An explicit edge sequence is used verbatim (sorted):
            ```python
            >>> import numpy as np
            >>> from cleopatra.styles import classify
            >>> edges, _ = classify(np.arange(11.0), [10.0, 0.0, 5.0])
            >>> [float(e) for e in edges]
            [0.0, 5.0, 10.0]

            ```
        - An unknown scheme name is rejected:
            ```python
            >>> from cleopatra.styles import classify
            >>> classify([1.0, 2.0, 3.0], "rainbow")
            Traceback (most recent call last):
                ...
            ValueError: Unknown classification scheme 'rainbow'. ...

            ```
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError(
            "Cannot classify: `values` has no finite entries to bin."
        )

    if isinstance(scheme, str):
        edges = _scheme_edges(finite, scheme, k)
    else:
        # An explicit sequence of bin edges supplied by the caller.
        edges = np.sort(np.asarray(scheme, dtype=float))

    # BoundaryNorm needs strictly increasing edges; quantiles on skewed or
    # discrete data can repeat, so collapse duplicates and require spread.
    edges = np.unique(edges)
    if edges.size < 2:
        raise ValueError(
            "Cannot classify: the data has no spread (fewer than two "
            "distinct bin edges). Pin explicit edges or widen the data."
        )
    norm = colors.BoundaryNorm(boundaries=edges, ncolors=256)
    return edges, norm


def _scheme_edges(finite: np.ndarray, scheme: str, k: int) -> np.ndarray:
    """Compute bin edges for a named classification `scheme`.

    Helper for `classify`: `finite` is already the finite-only data and
    `scheme` is a (case-insensitive) scheme name. Returns the raw edges
    before de-duplication, which `classify` performs.

    Args:
        finite: The finite-only data to derive edges from.
        scheme: The scheme name (see `classify`).
        k: The number of classes for the count/width schemes.

    Returns:
        np.ndarray: The (possibly non-unique) bin edges.

    Raises:
        ValueError: If `k < 1` (for the `k`-driven schemes) or `scheme`
            is not a recognised name.
        ModuleNotFoundError: If a Jenks-family scheme is requested without
            the optional `cleopatra[classify]` extra.
    """
    name = scheme.lower()
    if name in NUMPY_SCHEMES:
        if name == "std_mean":
            mean, std = float(finite.mean()), float(finite.std())
            breaks = [mean + mult * std for mult in _STD_MEAN_MULTIPLES]
            lo, hi = float(finite.min()), float(finite.max())
            inner = [b for b in breaks if lo < b < hi]
            return np.array([lo, *inner, hi], dtype=float)
        if k < 1:
            raise ValueError(f"`k` must be >= 1, got {k}.")
        if name == "quantiles":
            return np.quantile(finite, np.linspace(0.0, 1.0, k + 1))
        if name == "percentiles":
            return np.percentile(finite, np.linspace(0.0, 100.0, k + 1))
        # name == "equal_interval"
        return np.linspace(float(finite.min()), float(finite.max()), k + 1)

    if name in JENKS_SCHEMES:
        return _jenks_edges(finite, name, k)

    valid = ", ".join(repr(s) for s in (*NUMPY_SCHEMES, *JENKS_SCHEMES))
    raise ValueError(
        f"Unknown classification scheme {scheme!r}. Expected one of "
        f"{valid}, or an explicit sequence of bin edges."
    )


def _jenks_edges(finite: np.ndarray, name: str, k: int) -> np.ndarray:
    """Compute Jenks-family bin edges via the optional mapclassify extra.

    `mapclassify` is a soft dependency (the `cleopatra[classify]` extra),
    not a hard requirement, so it is imported lazily here and re-raised
    with an actionable install hint when missing — the default install
    stays numpy + matplotlib only.

    Args:
        finite: The finite-only data to classify.
        name: Either `"fisher_jenks"` or `"natural_breaks"`.
        k: The number of classes.

    Returns:
        np.ndarray: Bin edges `[min, *upper_bounds]` from the classifier.

    Raises:
        ValueError: If `k < 1`.
        ModuleNotFoundError: If `mapclassify` is not installed.
    """
    if k < 1:
        raise ValueError(f"`k` must be >= 1, got {k}.")
    try:
        import mapclassify
    except ModuleNotFoundError as exc:  # pragma: no cover - extra not installed
        raise ModuleNotFoundError(
            f"The {name!r} classification scheme requires the optional "
            "'classify' extra. Install it with: "
            "pip install 'cleopatra[classify]'."
        ) from exc
    classifier_name = "FisherJenks" if name == "fisher_jenks" else "NaturalBreaks"
    classifier = getattr(mapclassify, classifier_name)(finite, k=k)
    return np.concatenate([[float(finite.min())], np.asarray(classifier.bins)])
