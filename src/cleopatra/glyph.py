"""Base visualization class for cleopatra glyphs.

Provides shared infrastructure for array-based and mesh-based
visualization: figure/axes lifecycle, color scale normalization,
colorbar creation, tick management, point overlays, and animation.
"""

from __future__ import annotations

from typing import Any

import math

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.ticker import LogFormatter

from cleopatra.styles import MidpointNormalize

SUPPORTED_VIDEO_FORMAT = ["gif", "mov", "avi", "mp4"]


class Glyph:
    """Base class for cleopatra visualization glyphs.

    Handles figure/axes management, default options, color scale
    normalization, colorbar creation, tick control, point overlays,
    and animation saving. Subclasses implement the actual rendering.

    Args:
        default_options: Default plot options dict. Subclasses provide
            their own defaults merged with ``STYLE_DEFAULTS``.
        fig: Pre-existing matplotlib figure. Default is None.
        ax: Pre-existing matplotlib axes. Default is None.
        **kwargs: Override any key in ``default_options``.

    Examples:
        - Create a Glyph and override the colormap:
            ```python
            >>> from cleopatra.glyph import Glyph
            >>> from cleopatra.styles import DEFAULT_OPTIONS
            >>> opts = DEFAULT_OPTIONS.copy()
            >>> opts["vmin"] = None
            >>> opts["vmax"] = None
            >>> g = Glyph(default_options=opts, cmap="plasma")
            >>> g.default_options["cmap"]
            'plasma'

            ```
        - Provide a pre-existing figure and axes:
            ```python
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.glyph import Glyph
            >>> from cleopatra.styles import DEFAULT_OPTIONS
            >>> opts = DEFAULT_OPTIONS.copy()
            >>> opts["vmin"] = None
            >>> opts["vmax"] = None
            >>> fig, ax = plt.subplots()
            >>> g = Glyph(default_options=opts, fig=fig, ax=ax)
            >>> g.fig is fig
            True
            >>> g.ax is ax
            True

            ```

    See Also:
        cleopatra.array_glyph.ArrayGlyph: Glyph subclass for
            2D/3D arrays.
        cleopatra.mesh_glyph.MeshGlyph: Glyph subclass for
            unstructured meshes.
    """

    def __init__(
        self,
        default_options: dict,
        fig: Figure = None,
        ax: Axes = None,
        **kwargs,
    ):
        self._default_options = default_options.copy()
        self._merge_kwargs(kwargs)
        self._vmin: float | None = None
        self._vmax: float | None = None
        self.ticks_spacing: float | None = None
        if fig is not None:
            self.fig = fig
            self.ax = ax
        else:
            self.fig = None
            self.ax = None

    @property
    def vmin(self) -> float | None:
        """Minimum value for color scaling."""
        return self._vmin

    @property
    def vmax(self) -> float | None:
        """Maximum value for color scaling."""
        return self._vmax

    @property
    def default_options(self) -> dict:
        """Default plot options."""
        return self._default_options

    @property
    def anim(self) -> FuncAnimation:
        """Animation object created by ``animate()``."""
        if hasattr(self, "_anim") and self._anim is not None:
            return self._anim
        raise ValueError(
            "Please first use the animate method to create the animation object"
        )

    def _merge_kwargs(self, kwargs: dict) -> None:
        """Validate and merge keyword arguments into default_options."""
        for key, val in kwargs.items():
            if key not in self._default_options:
                raise ValueError(
                    f"The given keyword argument:{key} is not correct, "
                    f"possible parameters are, {list(self._default_options.keys())}"
                )
            else:
                self._default_options[key] = val

    def create_figure_axes(self) -> tuple[Figure, Axes]:
        """Create a new figure and axes from default_options.

        Uses the ``figsize`` key from ``default_options`` to set the
        figure dimensions.

        Returns:
            tuple[Figure, Axes]: The created figure and axes.

        Examples:
            - Create a figure with custom size:
                ```python
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None})
                >>> g = Glyph(default_options=opts, figsize=(12, 4))
                >>> fig, ax = g.create_figure_axes()
                >>> fig.get_size_inches()
                array([12.,  4.])

                ```
        """
        fig, ax = plt.subplots(figsize=self.default_options["figsize"])
        return fig, ax

    def get_ticks(self) -> np.ndarray:
        """Compute colorbar tick locations from default_options.

        Uses ``vmin``, ``vmax``, and ``ticks_spacing`` from
        ``default_options`` to generate evenly-spaced tick positions.

        Returns:
            np.ndarray: Array of tick positions.

        Examples:
            - Compute ticks for a 0-10 range with spacing of 2:
                ```python
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": 0.0, "vmax": 10.0, "ticks_spacing": 2.0})
                >>> g = Glyph(default_options=opts)
                >>> g.get_ticks()
                array([ 0.,  2.,  4.,  6.,  8., 10.])

                ```
        """
        ticks_spacing = self.default_options["ticks_spacing"]
        vmax = self.default_options["vmax"]
        vmin = self.default_options["vmin"]
        ticks = np.arange(vmin, vmax + ticks_spacing, ticks_spacing)
        # If vmax is not evenly divisible by spacing, append one more tick.
        remainder = np.round(math.remainder(vmax, ticks_spacing), 3)
        if remainder != 0:
            ticks = np.append(
                ticks,
                [int(vmax / ticks_spacing) * ticks_spacing + ticks_spacing],
            )
        return ticks

    def _create_norm_and_cbar_kw(
        self, ticks: np.ndarray
    ) -> tuple[colors.Normalize | None, dict]:
        """Create a matplotlib Normalize and colorbar kwargs.

        Args:
            ticks: Tick positions for the colorbar.

        Returns:
            tuple[Normalize or None, dict]: The norm (None for linear)
                and colorbar keyword arguments.
        """
        color_scale = self.default_options["color_scale"]
        vmin = ticks[0]
        vmax = ticks[-1]

        if color_scale.lower() == "linear":
            norm = None
            cbar_kw = {"ticks": ticks}
        elif color_scale.lower() == "power":
            norm = colors.PowerNorm(
                gamma=self.default_options["gamma"], vmin=vmin, vmax=vmax
            )
            cbar_kw = {"ticks": ticks}
        elif color_scale.lower() == "sym-lognorm":
            norm = colors.SymLogNorm(
                linthresh=self.default_options["line_threshold"],
                linscale=self.default_options["line_scale"],
                base=np.e,
                vmin=vmin,
                vmax=vmax,
            )
            formatter = LogFormatter(10, labelOnlyBase=False)
            cbar_kw = {"ticks": ticks, "format": formatter}
        elif color_scale.lower() == "boundary-norm":
            if not self.default_options["bounds"]:
                bounds = ticks
                cbar_kw = {"ticks": ticks}
            else:
                bounds = self.default_options["bounds"]
                cbar_kw = {"ticks": self.default_options["bounds"]}
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        elif color_scale.lower() == "midpoint":
            norm = MidpointNormalize(
                midpoint=self.default_options["midpoint"],
                vmin=vmin,
                vmax=vmax,
            )
            cbar_kw = {"ticks": ticks}
        else:
            raise ValueError(
                f"Invalid color scale option: {color_scale}. Use 'linear', "
                "'power', 'sym-lognorm', 'boundary-norm', 'midpoint'"
            )
        return norm, cbar_kw

    def create_color_bar(self, ax: Axes, im: Any, cbar_kw: dict) -> Colorbar:
        """Create a colorbar with full customization from default_options.

        Reads ``cbar_length``, ``cbar_orientation``, ``cbar_label``,
        ``cbar_label_size``, and ``cbar_label_location`` from
        ``default_options`` to configure the colorbar.

        Args:
            ax: Matplotlib axes.
            im: The mappable (image or contour) to attach the
                colorbar to.
            cbar_kw: Colorbar keyword arguments (ticks, format, etc.).

        Returns:
            Colorbar: The created colorbar.

        Examples:
            - Create a colorbar with a custom label:
                ```python
                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None})
                >>> g = Glyph(default_options=opts, cbar_label="Depth [m]")
                >>> fig, ax = plt.subplots()
                >>> im = ax.imshow(np.arange(9).reshape(3, 3))
                >>> cbar = g.create_color_bar(ax, im, {"ticks": [0, 4, 8]})
                >>> cbar.orientation
                'vertical'

                ```
        """
        fig = ax.figure
        is_subplot = len(fig.axes) > 1
        cbar = fig.colorbar(
            im,
            ax=ax,
            shrink=self.default_options["cbar_length"],
            orientation=self.default_options["cbar_orientation"],
            use_gridspec=not is_subplot,
            **cbar_kw,
        )
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(
            self.default_options["cbar_label"],
            fontsize=self.default_options["cbar_label_size"],
            loc=self.default_options["cbar_label_location"],
        )
        return cbar

    def adjust_ticks(
        self,
        axis: str,
        multiply_value: float | int = 1,
        add_value: float | int = 0,
        fmt: str = "{0:g}",
        visible: bool = True,
    ) -> None:
        """Adjust the axis tick labels with a linear transformation.

        Applies ``tick_value * multiply_value + add_value`` to each
        tick, formatted with ``fmt``. Useful for converting pixel
        coordinates to real-world units.

        Args:
            axis: ``"x"`` or ``"y"``.
            multiply_value: Multiplier for tick values. Default is 1.
            add_value: Offset added to tick values. Default is 0.
            fmt: Format string for tick labels.
                Default is ``"{0:g}"``.
            visible: Whether the axis is visible. Default is True.

        Examples:
            - Scale x-axis ticks by 100 and offset by 5:
                ```python
                >>> import matplotlib.pyplot as plt
                >>> from cleopatra.glyph import Glyph
                >>> from cleopatra.styles import DEFAULT_OPTIONS
                >>> opts = DEFAULT_OPTIONS.copy()
                >>> opts.update({"vmin": None, "vmax": None})
                >>> g = Glyph(default_options=opts)
                >>> fig, ax = plt.subplots()
                >>> _ = ax.plot([0, 1, 2], [0, 1, 2])
                >>> g.fig, g.ax = fig, ax
                >>> g.adjust_ticks(axis="x", multiply_value=100, add_value=5)

                ```
        """
        if axis == "x":
            ticks_fn = ticker.FuncFormatter(
                lambda x, pos: fmt.format(x * multiply_value + add_value)
            )
            self.ax.xaxis.set_major_formatter(ticks_fn)
        else:
            ticks_fn = ticker.FuncFormatter(
                lambda y, pos: fmt.format(y * multiply_value + add_value)
            )
            self.ax.yaxis.set_major_formatter(ticks_fn)

        if not visible:
            if axis == "x":
                self.ax.get_xaxis().set_visible(visible)
            else:
                self.ax.get_yaxis().set_visible(visible)

    @staticmethod
    def _plot_point_values(ax, point_table: np.ndarray, pid_color, pid_size):
        """Plot point value labels on the axes."""
        write_points = lambda x: ax.text(
            x[2],
            x[1],
            x[0],
            ha="center",
            va="center",
            color=pid_color,
            fontsize=pid_size,
        )
        return list(map(write_points, point_table))

    def save_animation(self, path: str, fps: int = 2) -> None:
        """Save the animation to a file.

        The output format is determined by the file extension. GIF uses
        ``PillowWriter``; mov/avi/mp4 require FFmpeg to be installed.

        Args:
            path: Output file path. Extension determines format.
                Supported: gif, mov, avi, mp4.
            fps: Frames per second. Default is 2.

        Raises:
            ValueError: If the file format is not supported.

        Examples:
            - Check the supported video formats:
                ```python
                >>> from cleopatra.glyph import SUPPORTED_VIDEO_FORMAT
                >>> sorted(SUPPORTED_VIDEO_FORMAT)
                ['avi', 'gif', 'mov', 'mp4']

                ```
        """
        video_format = path.split(".")[-1]
        if video_format not in SUPPORTED_VIDEO_FORMAT:
            raise ValueError(
                f"The given extension {video_format} implies a format that is "
                f"not supported, only {SUPPORTED_VIDEO_FORMAT} are supported"
            )

        if video_format == "gif":
            writer_gif = animation.PillowWriter(fps=fps)
            self.anim.save(path, writer=writer_gif)
        else:
            try:
                writer_video = animation.FFMpegWriter(fps=fps, bitrate=1800)
                self.anim.save(path, writer=writer_video)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    "FFmpeg not found. Please visit https://ffmpeg.org/ "
                    "and download a version compatible with your OS."
                ) from e
