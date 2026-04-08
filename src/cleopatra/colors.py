from pathlib import Path
from typing import Any, List, Tuple, Union

from matplotlib import colors as mcolors
from matplotlib.colors import Colormap, LinearSegmentedColormap
from PIL import Image, UnidentifiedImageError


class Colors:
    """A class for handling and converting between different color formats.

    The Colors class provides functionality for working with different color formats
    including hexadecimal colors, RGB colors (normalized between 0 and 1), and
    RGB colors (with values between 0 and 255). It supports validation, conversion,
    and manipulation of colors.

    Attributes:
        color_value: The color values stored in the class, can be hex strings or RGB tuples.

    Methods:
        get_type(): Determine the type of each color (hex, rgb, rgb-normalized).
        to_hex(): Convert all colors to hexadecimal format.
        to_rgb(normalized=True): Convert all colors to RGB format.
        is_valid_hex(): Check if each color is a valid hex color.
        is_valid_rgb(): Check if each color is a valid RGB color.

    Examples:
    Create a Colors object with a hex color:
    ```python
    >>> from cleopatra.colors import Colors
    >>> hex_color = Colors("#ff0000")
    >>> hex_color.color_value
    ['#ff0000']
    >>> hex_color.get_type()
    ['hex']

    ```
    Create a Colors object with an RGB color (values between 0 and 1):
    ```python
    >>> rgb_norm = Colors((0.5, 0.2, 0.8))
    >>> rgb_norm.color_value
    [(0.5, 0.2, 0.8)]
    >>> rgb_norm.get_type()
    ['rgb-normalized']

    ```

    Create a Colors object with an RGB color (values between 0 and 255):
    ```python
    >>> rgb_255 = Colors((128, 51, 204))
    >>> rgb_255.color_value
    [(128, 51, 204)]
    >>> rgb_255.get_type()
    ['rgb']

    ```
    Convert between color formats:
    ```python
    >>> hex_color.to_rgb()  # Convert hex to RGB (normalized)
    [(1.0, 0.0, 0.0)]
    >>> rgb_norm.to_hex()  # Convert RGB to hex
    ['#8033cc']

    ```
    """

    def __init__(
        self,
        color_value: Union[
            List[str], str, Tuple[float, float, float], List[Tuple[float, float, float]]
        ],
    ):
        """Initialize a Colors object with the given color value(s).

        Args:
            color_value: The color value(s) to initialize the object with. Can be:
                - A single hex color string (e.g., "#ff0000" or "ff0000")
                - A single RGB tuple with values between 0-1 (e.g., (1.0, 0.0, 0.0))
                - A single RGB tuple with values between 0-255 (e.g., (255, 0, 0))
                - A list of hex color strings
                - A list of RGB tuples

        Raises:
            ValueError: If the color_value is not a string, tuple, or list of strings/tuples.

        Notes:
        - Hex colors can be provided with or without the leading "#"
        - RGB tuples with float values between 0-1 are treated as normalized RGB
        - RGB tuples with integer values between 0-255 are treated as standard RGB
        - The class automatically detects the type of color format provided

        Examples:
        - Initialize with a hex color:

            ```python
            >>> from cleopatra.colors import Colors
            >>> # With hash symbol
            >>> color1 = Colors("#ff0000")
            >>> color1.color_value
            ['#ff0000']
            >>> # Without hash symbol
            >>> color2 = Colors("ff0000")
            >>> color2.color_value
            ['ff0000']

            ```

        - Initialize with an RGB color (normalized, values between 0 and 1):

            ```python
            >>> rgb_norm = Colors((1.0, 0.0, 0.0))
            >>> rgb_norm.color_value
            [(1.0, 0.0, 0.0)]
            >>> rgb_norm.get_type()
            ['rgb-normalized']

            ```

        - Initialize with an RGB color (values between 0 and 255):

            ```python
            >>> rgb_255 = Colors((255, 0, 0))
            >>> rgb_255.color_value
            [(255, 0, 0)]
            >>> rgb_255.get_type()
            ['rgb']

            ```

        - Initialize with a list of colors:

            ```python
            >>> mixed_colors = Colors(["#ff0000", (0, 255, 0), (0.0, 0.0, 1.0)])
            >>> mixed_colors.color_value
            ['#ff0000', (0, 255, 0), (0.0, 0.0, 1.0)]
            >>> mixed_colors.get_type()
            ['hex', 'rgb', 'rgb-normalized']

            ```
        """
        # convert the hex color to a list if it is a string
        if isinstance(color_value, str) or isinstance(color_value, tuple):
            color_value = [color_value]
        elif not isinstance(color_value, list):
            raise ValueError(
                "The color_value must be a list of hex colors, list of tuples (RGB color), a single hex "
                "or single RGB tuple color."
            )

        self._color_value = color_value

    @classmethod
    def create_from_image(cls, path: str) -> "Colors":
        """Create a color object from an image.

        if you have an image of a color ramp, and you want to extract the colors from it, you can use this method.

        ![color-ramp](./../images/colors/color-ramp.png)

        Args:
            path: The path to the image file.

        Returns:
            Colors: A color object.

        Raises:
            FileNotFoundError: If the file does not exist.

        Examples:
        ```python
        >>> path = "examples/data/colors/color-ramp.png"
        >>> colors = Colors.create_from_image(path)
        >>> print(colors.color_value) # doctest: +SKIP
        [(9, 63, 8), (8, 68, 9), (5, 78, 7), (1, 82, 3), (0, 84, 0), (0, 85, 0), (1, 83, 0), (1, 81, 0), (1, 80, 1)

        ```
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"The file {path} does not exist.")
        try:
            image = Image.open(path).convert("RGB")
        except UnidentifiedImageError:
            raise ValueError(f"The file {path} is not a valid image.")
        width, height = image.size
        color_values = [image.getpixel((x, int(height / 2))) for x in range(width)]

        return cls(color_values)

    def get_type(self) -> List[str]:
        """Determine the type of each color value.

        This method analyzes each color value stored in the object and determines
        its type: hex, rgb (values 0-255), or rgb-normalized (values 0-1).

        Returns:
            List[str]: A list of strings indicating the type of each color value.
                Possible values are:
                - 'hex': Hexadecimal color string
                - 'rgb': RGB tuple with values between 0-255
                - 'rgb-normalized': RGB tuple with values between 0-1

        Notes:
            The method uses the following criteria to determine color types:
            - If the value is a string and is a valid hex color, it's classified as 'hex'
            - If the value is a tuple of 3 floats between 0-1, it's classified as 'rgb-normalized'
            - If the value is a tuple of 3 integers between 0-255, it's classified as 'rgb'

        Examples:
        - Determine the type of a hex color:

            ```python
            >>> from cleopatra.colors import Colors
            >>> hex_color = Colors("#23a9dd")
            >>> hex_color.get_type()
            ['hex']

            ```

        - Determine the type of an RGB color with normalized values (0-1):

            ```python
            >>> rgb_norm = Colors((0.5, 0.2, 0.8))
            >>> rgb_norm.get_type()
            ['rgb-normalized']

            ```

        - Determine the type of an RGB color with values between 0-255:

            ```python
            >>> rgb_255 = Colors((128, 51, 204))
            >>> rgb_255.get_type()
            ['rgb']

            ```

        - Determine types of mixed color formats:

            ```python
            >>> mixed = Colors(["#ff0000", (0, 255, 0), (0.0, 0.0, 1.0)])
            >>> mixed.get_type()
            ['hex', 'rgb', 'rgb-normalized']

            ```
        """
        color_type = []
        for color_i in self.color_value:
            if self._is_valid_rgb_norm(color_i):
                color_type.append("rgb-normalized")
            elif self._is_valid_rgb_255(color_i):
                color_type.append("rgb")
            elif self._is_valid_hex_i(color_i):
                color_type.append("hex")

        return color_type

    @property
    def color_value(self) -> Union[List[str], List[Tuple[float, float, float]]]:
        """Get the color values stored in the object.

        This property returns the color values that were provided when initializing
        the Colors object or set afterwards. The values can be hex color strings,
        RGB tuples with values between 0-255, or normalized RGB tuples with values
        between 0-1.

        Returns:
            Union[List[str], List[Tuple[float, float, float]]]: A list containing the color values. Each element can be:
                - A hex color string (e.g., "#ff0000" or "ff0000")
                - An RGB tuple with values between 0-255 (e.g., (255, 0, 0))
                - A normalized RGB tuple with values between 0-1 (e.g., (1.0, 0.0, 0.0))

        Examples:
        Get color values from a Colors object with hex colors:
        ```python
        >>> from cleopatra.colors import Colors
        >>> hex_colors = Colors(["#ff0000", "#00ff00", "#0000ff"])
        >>> hex_colors.color_value
        ['#ff0000', '#00ff00', '#0000ff']

        ```

        Get color values from a Colors object with RGB colors:
        ```python
        >>> rgb_colors = Colors([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
        >>> rgb_colors.color_value
        [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        ```
        Get color values from a Colors object with mixed color formats:
        ```python
        >>> mixed = Colors(["#ff0000", (0, 255, 0), (0.0, 0.0, 1.0)])
        >>> mixed.color_value
        ['#ff0000', (0, 255, 0), (0.0, 0.0, 1.0)]

        ```
        """
        return self._color_value

    def to_hex(self) -> List[str]:
        """Convert all color values to hexadecimal format.

        This method converts all color values stored in the object to hexadecimal format.
        RGB tuples (both normalized and 0-255 range) are converted to their hex equivalents.
        Hex colors remain unchanged.

        Returns:
            List[str]: A list of hexadecimal color strings. Each string is in the format '#RRGGBB'.

        Notes:
            - RGB tuples with values between 0-255 are first normalized to 0-1 range before conversion
            - RGB tuples with values already between 0-1 are directly converted
            - Existing hex colors are returned as-is
            - All returned hex colors include the leading '#' character

        Examples:
        Convert RGB colors to hex:
        ```python
        >>> from cleopatra.colors import Colors
        >>> # RGB colors (0-255 range)
        >>> rgb_255 = Colors([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
        >>> rgb_255.to_hex()
        ['#ff0000', '#00ff00', '#0000ff']

        ```
        >>> # RGB colors (normalized 0-1 range)
        >>> rgb_norm = Colors([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        >>> rgb_norm.to_hex()
        ['#ff0000', '#00ff00', '#0000ff']

        ```
        Convert a mix of color formats to hex:
        ```python
        >>> mixed = Colors([(128, 51, 204), "#23a9dd", (0.5, 0.2, 0.8)])
        >>> mixed.to_hex()
        ['#8033cc', '#23a9dd', '#8033cc']

        ```
        Hex colors are returned as-is:
        ```python
        >>> hex_colors = Colors(["#ff0000", "#00ff00", "#0000ff"])
        >>> hex_colors.to_hex()
        ['#ff0000', '#00ff00', '#0000ff']

        ```
        """
        converted_color = []
        color_type = self.get_type()
        for ind, color_i in enumerate(self.color_value):
            if color_type[ind] == "hex":
                converted_color.append(color_i)
            elif color_type[ind] == "rgb":
                # Normalize the RGB values to be between 0 and 1
                rgb_color_normalized = tuple(value / 255 for value in color_i)
                converted_color.append(mcolors.to_hex(rgb_color_normalized))
            else:
                converted_color.append(mcolors.to_hex(color_i))
        return converted_color

    def is_valid_hex(self) -> List[bool]:
        """Check if each color value is a valid hexadecimal color.

        This method checks each color value stored in the object to determine
        if it is a valid hexadecimal color string.

        Returns:
            List[bool]: A list of boolean values, one for each color value in the object.
                True indicates the color is a valid hex color, False otherwise.

        Notes:
            - The method uses matplotlib's is_color_like function to validate hex colors
            - Both formats with and without the leading '#' are supported
            - RGB tuples will return False as they are not hex colors

        Examples:
        Check if hex colors are valid:
        ```python
        >>> from cleopatra.colors import Colors
        >>> hex_colors = Colors(["#ff0000", "#00ff00", "#0000ff"])
        >>> hex_colors.is_valid_hex()
        [True, True, True]

        ```
        Check if RGB colors are valid hex colors (they're not):
        ```python
        >>> rgb_colors = Colors([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
        >>> rgb_colors.is_valid_hex()
        [False, False, False]

        ```
        Check a mix of color formats:
        ```python
        >>> mixed = Colors(["#ff0000", (0, 255, 0), "not-a-color"])
        >>> mixed.is_valid_hex()
        [True, False, False]

        ```
        """
        return [self._is_valid_hex_i(col) for col in self.color_value]

    @staticmethod
    def _is_valid_hex_i(hex_color: str) -> bool:
        """Check if a single color value is a valid hexadecimal color.

        This static method checks if the provided color value is a valid
        hexadecimal color string.

        Args:
            hex_color: A color string to validate as a hexadecimal color.
                Can be in the format "#RRGGBB" or "RRGGBB".

        Returns:
            bool: True if the color is a valid hexadecimal color, False otherwise.

        Notes:
            - The method uses matplotlib's is_color_like function to validate hex colors
            - Both formats with and without the leading '#' are supported
            - Non-string values will return False

        Examples:
        Check valid hex colors:
        ```python
        >>> from cleopatra.colors import Colors
        >>> Colors._is_valid_hex_i("#ff0000")
        True
        >>> Colors._is_valid_hex_i("00ff00")
        False
        >>> Colors._is_valid_hex_i("#0000FF")
        True

        ```

        Check invalid hex colors:
        ```python
        >>> Colors._is_valid_hex_i("not-a-color")
        False
        >>> Colors._is_valid_hex_i("#12345")  # Too short
        False
        >>> Colors._is_valid_hex_i((255, 0, 0))  # doctest: +ELLIPSIS
        False

        ```
        """
        if not isinstance(hex_color, str):
            return False
        else:
            return True if mcolors.is_color_like(hex_color) else False

    def is_valid_rgb(self) -> List[bool]:
        """Check if each color value is a valid RGB color.

        This method checks each color value stored in the object to determine
        if it is a valid RGB color tuple (either with values between 0-255 or
        normalized values between 0-1).

        Returns:
            List[bool]: A list of boolean values, one for each color value in the object.
                True indicates the color is a valid RGB tuple, False otherwise.

        Notes:
            - The method checks for both RGB formats: values between 0-255 and normalized values between 0-1
            - A valid RGB tuple must have exactly 3 values (R, G, B)
            - Hex color strings will return False as they are not RGB tuples

        Examples:
        Check if RGB colors are valid:
        ```python
        >>> from cleopatra.colors import Colors
        >>> # RGB colors (0-255 range)
        >>> rgb_255 = Colors([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
        >>> rgb_255.is_valid_rgb()
        [True, True, True]

        >>> # RGB colors (normalized 0-1 range)
        >>> rgb_norm = Colors([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        >>> rgb_norm.is_valid_rgb()
        [True, True, True]

        ```
        Check if hex colors are valid RGB colors (they're not):
        ```python
        >>> hex_colors = Colors(["#ff0000", "#00ff00", "#0000ff"])
        >>> hex_colors.is_valid_rgb()
        [False, False, False]

        ```
        Check a mix of color formats:
        ```python
        >>> mixed = Colors([(255, 0, 0), "#00ff00", (0.0, 0.0, 1.0)])
        >>> mixed.is_valid_rgb()
        [True, False, True]

        ```
        """
        return [
            self._is_valid_rgb_norm(col) or self._is_valid_rgb_255(col)
            for col in self.color_value
        ]

    @staticmethod
    def _is_valid_rgb_255(rgb_tuple: Any) -> bool:
        """Check if a single color value is a valid RGB tuple with values between 0-255.

        This static method checks if the provided value is a valid RGB tuple with
        integer values between 0 and 255.

        Args:
            rgb_tuple: The value to check. Should be a tuple of 3 integers between 0 and 255
                to be considered valid.

        Returns:
            bool: True if the value is a valid RGB tuple with values between 0-255,
                False otherwise.

        Examples:
        Check valid RGB tuples (0-255 range):
        ```python
        >>> from cleopatra.colors import Colors
        >>> Colors._is_valid_rgb_255((255, 0, 0))
        True
        >>> Colors._is_valid_rgb_255((128, 64, 32))
        True
        >>> Colors._is_valid_rgb_255((0, 0, 0))
        True

        ```
        Check invalid RGB tuples:
        ```python
        >>> Colors._is_valid_rgb_255((1.0, 0.0, 0.0))  # Floats, not integers
        False
        >>> Colors._is_valid_rgb_255((256, 0, 0))  # Value > 255
        False
        >>> Colors._is_valid_rgb_255((0, 0))  # Not 3 values
        False
        >>> Colors._is_valid_rgb_255("#ff0000")  # Not a tuple
        False

        ```
        """
        if isinstance(rgb_tuple, tuple) and len(rgb_tuple) == 3:
            if all(isinstance(value, int) for value in rgb_tuple):
                return all(0 <= value <= 255 for value in rgb_tuple)
        return False

    @staticmethod
    def _is_valid_rgb_norm(rgb_tuple: Any) -> bool:
        """Check if a single color value is a valid normalized RGB tuple with values between 0-1.

        This static method checks if the provided value is a valid RGB tuple with
        float values between 0.0 and 1.0.

        Args:
            rgb_tuple: The value to check. Should be a tuple of 3 floats between 0.0 and 1.0
                to be considered valid.

        Returns:
            bool: True if the value is a valid normalized RGB tuple with values between 0.0-1.0,
                False otherwise.

        Examples:
        Check valid normalized RGB tuples:
        ```python
        >>> from cleopatra.colors import Colors
        >>> Colors._is_valid_rgb_norm((1.0, 0.0, 0.0))
        True
        >>> Colors._is_valid_rgb_norm((0.5, 0.5, 0.5))
        True
        >>> Colors._is_valid_rgb_norm((0.0, 0.0, 0.0))
        True

        ```
        Check invalid normalized RGB tuples:
        ```python
        >>> Colors._is_valid_rgb_norm((255, 0, 0))  # Integers, not floats
        False
        >>> Colors._is_valid_rgb_norm((1.2, 0.0, 0.0))  # Value > 1.0
        False
        >>> Colors._is_valid_rgb_norm((0.5, 0.5))  # Not 3 values
        False
        >>> Colors._is_valid_rgb_norm("#ff0000")  # Not a tuple
        False

        ```
        """
        if isinstance(rgb_tuple, tuple) and len(rgb_tuple) == 3:
            if all(isinstance(value, float) for value in rgb_tuple):
                return all(0.0 <= value <= 1.0 for value in rgb_tuple)
        return False

    def to_rgb(
        self, normalized: bool = True
    ) -> List[Tuple[Union[int, float], Union[int, float], Union[int, float]]]:
        """Convert all color values to RGB format.

        This method converts all color values stored in the object to RGB format.
        Hex colors are converted to their RGB equivalents. RGB colors remain unchanged
        but may be normalized or denormalized based on the 'normalized' parameter.

        Args:
            normalized: Whether to return normalized RGB values (between 0 and 1) or standard RGB values
                (between 0 and 255). Defaults to True.
                - If True, returns RGB values scaled between 0 and 1
                - If False, returns RGB values scaled between 0 and 255

        Returns:
            List[Tuple[Union[int, float], Union[int, float], Union[int, float]]]: A list of RGB tuples.
                Each tuple contains three values (R, G, B).
                - If normalized=True, values are floats between 0.0 and 1.0
                - If normalized=False, values are integers between 0 and 255

        Examples:
        - Convert hex colors to normalized RGB (0-1 range):
            ```python
            >>> from cleopatra.colors import Colors
            >>> hex_colors = Colors(["#ff0000", "#00ff00", "#0000ff"])
            >>> hex_colors.to_rgb(normalized=True)
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

            ```

        - Convert hex colors to standard RGB (0-255 range):
            ```python
            >>> hex_colors.to_rgb(normalized=False)
            [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

            ```
        - Convert RGB colors and maintain their format:
            There are two types of RGB coor values (0-255), and (0-1), you can get the RGB values in any format, the
            default is the normalized format (0-1):

            ```python
            >>> rgb_255 = Colors([(255, 0, 0), (0, 255, 0)])
            >>> rgb_255.to_rgb(normalized=False)  # Keep as 0-255 range
            [(255, 0, 0), (0, 255, 0)]
            >>> rgb_255.to_rgb(normalized=True)  # Convert to 0-1 range
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

            >>> rgb_norm = Colors([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
            >>> rgb_norm.to_rgb(normalized=True)  # Keep as 0-1 range
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
            >>> rgb_norm.to_rgb(normalized=False)  # Convert to 0-255 range
            [(255, 0, 0), (0, 255, 0)]

            ```

        Convert mixed color formats:
        ```python
        >>> mixed = Colors(["#ff0000", (0, 255, 0), (0.0, 0.0, 1.0)])
        >>> mixed.to_rgb(normalized=True)
        [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

        ```
        """
        color_type = self.get_type()
        rgb = []
        if normalized:
            for ind, color_i in enumerate(self.color_value):
                # if the color is in RGB format (0-255), normalize the values to be between 0 and 1
                if color_type[ind] == "rgb":
                    rgb_color_normalized = tuple(value / 255 for value in color_i)
                    rgb.append(rgb_color_normalized)
                else:
                    # any other format, just convert it to RGB
                    rgb.append(mcolors.to_rgb(color_i))
        else:
            for ind, color_i in enumerate(self.color_value):
                # if the color is in RGB format (0-255), normalize the values to be between 0 and 1
                if color_type[ind] == "rgb":
                    rgb.append(color_i)
                else:
                    # any other format, just convert it to RGB
                    rgb.append(tuple([int(c * 255) for c in mcolors.to_rgb(color_i)]))

        return rgb

    def get_color_map(self, name: str = None) -> Colormap:
        """Get color ramp from a color values in stored in the object.

        Args:
            name: The name of the color ramp. Defaults to None.

        Returns:
            Colormap: A color map.

        Examples:
        - Create a color object from an image and get the color ramp:
            ```python
            >>> path = "examples/data/colors/color-ramp.png"
            >>> colors = Colors.create_from_image(path)
            >>> color_ramp = colors.get_color_map()
            >>> print(color_ramp) # doctest: +SKIP
            <matplotlib.colors.LinearSegmentedColormap object at 0x7f8a2e1b5e50>

            ```
        """
        vals = self.to_rgb(normalized=True)
        name = "custom_color_map" if name is None else name
        return LinearSegmentedColormap.from_list(name, vals)
