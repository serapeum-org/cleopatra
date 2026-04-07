"""Base visualization class for cleopatra glyphs.

Provides shared infrastructure for array-based and mesh-based
visualization: figure/axes lifecycle, color scale normalization,
colorbar creation, tick management, point overlays, and animation.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

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

    Parameters
    ----------
    default_options : dict
        Default plot options dict. Subclasses provide their own
        defaults merged with ``STYLE_DEFAULTS``.
    fig : Figure or None, optional
        Pre-existing matplotlib figure. Default is None.
    ax : Axes or None, optional
        Pre-existing matplotlib axes. Default is None.
    **kwargs
        Override any key in ``default_options``.
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
        if hasattr(self, "_anim"):
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

    def create_figure_axes(self) -> Tuple[Figure, Axes]:
        """Create a new figure and axes from default_options.

        Returns
        -------
        tuple[Figure, Axes]
            The created figure and axes.
        """
        fig, ax = plt.subplots(figsize=self.default_options["figsize"])
        return fig, ax

    def get_ticks(self) -> np.ndarray:
        """Compute colorbar tick locations from default_options.

        Returns
        -------
        np.ndarray
            Array of tick positions.
        """
        ticks_spacing = self.default_options["ticks_spacing"]
        vmax = self.default_options["vmax"]
        vmin = self.default_options["vmin"]
        remainder = np.round(math.remainder(vmax, ticks_spacing), 3)
        if remainder == 0:
            ticks = np.arange(vmin, vmax + ticks_spacing, ticks_spacing)
        else:
            try:
                ticks = np.arange(vmin, vmax + ticks_spacing, ticks_spacing)
            except ValueError:
                raise ValueError(
                    "The number of ticks exceeded the max allowed size, possible errors"
                    " is the value of the NodataValue you entered"
                )
            ticks = np.append(
                ticks,
                [int(vmax / ticks_spacing) * ticks_spacing + ticks_spacing],
            )
        return ticks

    def _create_norm_and_cbar_kw(
        self, ticks: np.ndarray
    ) -> Tuple[colors.Normalize | None, dict]:
        """Create a matplotlib Normalize and colorbar kwargs.

        Parameters
        ----------
        ticks : np.ndarray
            Tick positions for the colorbar.

        Returns
        -------
        tuple[Normalize or None, dict]
            The norm (None for linear) and colorbar keyword arguments.
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

    def create_color_bar(self, ax: Axes, im, cbar_kw: dict) -> Colorbar:
        """Create a colorbar with full customization from default_options.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes.
        im : AxesImage
            The mappable (image or contour) to attach the colorbar to.
        cbar_kw : dict
            Colorbar keyword arguments (ticks, format, etc.).

        Returns
        -------
        Colorbar
            The created colorbar.
        """
        cbar = ax.figure.colorbar(
            im,
            ax=ax,
            shrink=self.default_options["cbar_length"],
            orientation=self.default_options["cbar_orientation"],
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
        multiply_value: Union[float, int] = 1,
        add_value: Union[float, int] = 0,
        fmt: str = "{0:g}",
        visible: bool = True,
    ) -> None:
        """Adjust the axis tick labels.

        Parameters
        ----------
        axis : str
            ``"x"`` or ``"y"``.
        multiply_value : float or int, optional
            Multiplier for tick values. Default is 1.
        add_value : float or int, optional
            Offset added to tick values. Default is 0.
        fmt : str, optional
            Format string for tick labels. Default is ``"{0:g}"``.
        visible : bool, optional
            Whether the axis is visible. Default is True.
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

        plt.show()

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

        Parameters
        ----------
        path : str
            Output file path. Extension determines format.
            Supported: gif, mov, avi, mp4.
        fps : int, optional
            Frames per second. Default is 2.

        Raises
        ------
        ValueError
            If the file format is not supported.
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
            except FileNotFoundError:
                print(
                    "Please visit https://ffmpeg.org/ and download a version "
                    "of ffmpeg compatible with your operating system"
                )
