"""
The `textured_globe_glyph` module wraps an equirectangular (lon/lat) RGB texture onto a 3-D sphere drawn on a
matplotlib `Axes3D`, with an optional axial tilt and a `spin` angle so the globe can be rotated frame-by-frame for
animation.

It is cleopatra's one deliberate 3-D glyph. Every other glyph targets a 2-D axes; this one is a single, self-contained
exception. It adds no dependency -- `mpl_toolkits.mplot3d` ships with matplotlib -- and keeps the
"NumPy in -> matplotlib artist out" contract: you bring an `(H, W, 3)` or `(H, W, 4)` equirectangular array (for
example `cleopatra.basemap.reference.relief()`), and you get back a matplotlib `Figure`/`Axes3D`.

Like `HistogramGlyph`, `TexturedGlobeGlyph` is a standalone class rather than a `Glyph` subclass, because the `Glyph`
base class owns the 2-D figure/axes/colorbar lifecycle that does not apply to a textured sphere.

Design
------

- **Sample once, rotate per frame.** The texture is resampled to the mesh resolution and turned into a `facecolors`
  array exactly once (cached on the instance). Spinning the globe only rotates the pre-computed vertex coordinates by a
  small 3x3 matrix multiply and re-draws -- the texture is never re-sampled. This is what makes `animate()` cheap.
- **Spin the globe, not the camera.** `spin` rotates the sphere about its own (tilted) polar axis, so the axial tilt
  stays fixed in space while the surface turns underneath it -- physically what a spinning planet does. The camera
  (`elev`/`azim`) is held still.
- **Resolution is the cost driver.** matplotlib's 3-D surface is drawn on the CPU as one polygon per mesh face, so the
  render time grows with `n_lon * n_lat`. Measured first-draw times (matplotlib 3.10, Agg):

  | `n_lon` x `n_lat` | faces   | first draw |
  | ----------------- | ------- | ---------- |
  | 180 x 90 (default)|  ~15900 | ~1.5 s     |
  | 360 x 180         |  ~64000 | ~7.7 s     |
  | 720 x 360         | ~258000 | ~27 s      |

  The default (180 x 90) renders a recognisable globe quickly; raise `n_lon`/`n_lat` for a sharper still, lower them
  for a smooth animation.

Example
-------

```python
import matplotlib.pyplot as plt
import numpy as np
from cleopatra.glyphs.globe.textured_globe_glyph import TexturedGlobeGlyph

# A cheap synthetic texture (or use cleopatra.basemap.reference.relief()).
texture = np.zeros((180, 360, 3), dtype=np.uint8)
texture[:90] = (40, 90, 180)     # northern hemisphere blue
texture[90:] = (180, 120, 40)    # southern hemisphere ochre

globe = TexturedGlobeGlyph(texture, tilt_deg=23.44)
fig, ax = globe.draw(spin=60.0)
plt.show()
```
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from cleopatra.glyphs.base.glyph import (
    _clear_prior_render_artists,
    _mark_render_artists,
    _root_figure,
)

#: Earth's axial tilt (obliquity of the ecliptic), in degrees -- the default lean of the polar axis.
EARTH_TILT_DEG = 23.44

#: Default render options for :class:`TexturedGlobeGlyph` (the module constant; the class re-exposes it as
#: ``TexturedGlobeGlyph.DEFAULT_OPTIONS`` for introspection).
GLOBE_DEFAULT_OPTIONS = {
    "figsize": (6, 6),
    "elev": 15.0,
    "azim": 0.0,
    "background": None,
}


class TexturedGlobeGlyph:
    """Wrap an equirectangular texture onto a tilted, spinnable 3-D sphere.

    The glyph takes an equirectangular (plate-carree) `(H, W, 3)` or `(H, W, 4)` array -- rows running north (row 0,
    +90 deg) to south (last row, -90 deg), columns running west (col 0, -180 deg) to east (last col, +180 deg), exactly
    the layout of `cleopatra.basemap.reference.relief()` -- and paints it onto a unit sphere on a matplotlib `Axes3D`.

    The polar axis is tilted `tilt_deg` from vertical, and `draw(spin=...)` rotates the sphere about that axis so the
    same instance can render a whole rotation without re-sampling the texture (see the module docstring).

    Attributes:
        texture: The normalised RGBA texture, float in `[0, 1]`, shape `(H, W, 4)`.
        tilt_deg: Axial tilt of the polar axis from vertical, in degrees.
        n_lon: Number of longitude samples in the sphere mesh.
        n_lat: Number of latitude samples in the sphere mesh.
        brightness: Multiplier applied to the RGB channels (clipped to `[0, 1]`).
        default_options: The resolved render options (`figsize`, `elev`, `azim`, `background`).

    Methods:
        draw(ax=None, *, spin=0.0, **kwargs): Render the globe onto a 3-D axes at a given spin angle.
        animate(ax=None, n_frames=60, revolutions=1.0, ...): Return a `FuncAnimation` spinning the globe.

    Notes:
        `TexturedGlobeGlyph` is a standalone class, not a `Glyph` subclass (like `HistogramGlyph`). The accepted option
        keys are exposed via the `DEFAULT_OPTIONS` class attribute and can be inspected/filtered with the `option_keys`
        and `filter_kwargs` classmethods.

    Examples:
        Build a globe from a small synthetic texture and render it:
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.globe.textured_globe_glyph import TexturedGlobeGlyph
        >>> texture = np.zeros((16, 32, 3), dtype=np.uint8)
        >>> texture[:8] = (40, 90, 180)
        >>> globe = TexturedGlobeGlyph(texture, n_lon=48, n_lat=24)
        >>> fig, ax = globe.draw(spin=45.0)
        >>> ax.name
        '3d'
        >>> globe.surface is not None
        True

        ```
    """

    #: Option keys this glyph accepts, exposed as a class attribute so they can be introspected/filtered before an
    #: instance exists (see `option_keys`/`filter_kwargs`).
    DEFAULT_OPTIONS = GLOBE_DEFAULT_OPTIONS

    def __init__(
        self,
        texture: np.ndarray,
        *,
        tilt_deg: float = EARTH_TILT_DEG,
        n_lon: int = 180,
        n_lat: int = 90,
        brightness: float = 1.0,
        fig: Figure | None = None,
        ax: Axes3D | None = None,
        **kwargs,
    ):
        """Initialize the globe from an equirectangular texture.

        Args:
            texture: An equirectangular RGB(A) array of shape `(H, W, 3)` or `(H, W, 4)`, `H >= 2` and `W >= 2`.
                Integer arrays are read as 8-bit (0-255); float arrays are assumed to be in `[0, 1]` and, only if a
                channel exceeds 1, are normalised by their own peak. A float `(H, W, 4)` array's alpha must already
                be in `[0, 1]`. NaN cells render black. Row 0 is the northern edge (+90 deg), the last row the
                southern edge (-90 deg); column 0 is -180 deg, the last column +180 deg.
            tilt_deg: Axial tilt of the polar axis from vertical, in degrees. Defaults to Earth's 23.44 deg.
            n_lon: Longitude samples in the sphere mesh (>= 2). Higher is sharper but quadratically slower to draw.
            n_lat: Latitude samples in the sphere mesh (>= 2). Higher is sharper but quadratically slower to draw.
            brightness: Multiplier applied to the RGB channels before clipping to `[0, 1]`. `< 1` darkens, `> 1`
                brightens. Must be `>= 0`.
            fig: Pre-existing matplotlib `Figure` to draw on when `draw`/`animate` are called without their own `ax`.
            ax: Pre-existing 3-D matplotlib `Axes` (`Axes3D`) to draw on. If given it must be a 3-D axes.
            **kwargs: Render options overriding `DEFAULT_OPTIONS` (`figsize`, `elev`, `azim`, `background`). An
                unrecognised key raises `ValueError`.

        Raises:
            ValueError: If `texture` is not an `(H, W, 3)`/`(H, W, 4)` array with `H, W >= 2`, if `n_lon`/`n_lat` are
                `< 2`, if `brightness` is negative, if `ax` is given but is not a 3-D axes, or if an unknown render
                option is passed.

        Examples:
            ```python
            >>> import numpy as np
            >>> from cleopatra.glyphs.globe.textured_globe_glyph import TexturedGlobeGlyph
            >>> globe = TexturedGlobeGlyph(np.zeros((8, 16, 3), dtype=np.uint8), n_lon=24, n_lat=12)
            >>> globe.n_lon, globe.n_lat
            (24, 12)

            ```
        """
        if int(n_lon) < 2 or int(n_lat) < 2:
            raise ValueError(
                f"n_lon and n_lat must each be >= 2; got n_lon={n_lon}, n_lat={n_lat}."
            )
        if brightness < 0:
            raise ValueError(f"brightness must be >= 0; got {brightness}.")
        if ax is not None and not isinstance(ax, Axes3D):
            raise ValueError(
                "TexturedGlobeGlyph needs a 3-D axes; create one with fig.add_subplot(projection='3d')."
            )

        self._brightness = float(brightness)
        self._texture = self._normalize_texture(texture, self._brightness)
        self._tilt_deg = float(tilt_deg)
        self._n_lon = int(n_lon)
        self._n_lat = int(n_lat)
        self._fig = fig
        self._ax = ax

        self._reject_unknown_options(kwargs)
        options_dict = GLOBE_DEFAULT_OPTIONS.copy()
        options_dict.update(kwargs)
        self._default_options = options_dict

        # Filled lazily and cached by `_prepare` (sample-once contract).
        self._base_xyz: np.ndarray | None = None
        self._facecolors: np.ndarray | None = None
        self._tilt_matrix: np.ndarray | None = None
        self._surface = None

    # ------------------------------------------------------------------ #
    # Construction helpers                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_texture(texture: np.ndarray, brightness: float) -> np.ndarray:
        """Return `texture` as a float `(H, W, 4)` RGBA array in `[0, 1]`, brightness-scaled.

        Args:
            texture: An `(H, W, 3)` or `(H, W, 4)` RGB(A) array. Integer arrays are read as 8-bit
                (0-255) and divided by 255; float arrays are assumed to be in `[0, 1]` and, only if a
                channel exceeds 1, normalised by their own peak (NaN ignored). NaN cells render black.
            brightness: Multiplier applied to the RGB channels before clipping.

        Returns:
            numpy.ndarray: A contiguous float `(H, W, 4)` array in `[0, 1]`.

        Raises:
            ValueError: If `texture` is not an `(H, W, 3)`/`(H, W, 4)` array with `H, W >= 2`.
        """
        arr = np.asarray(texture)
        if (
            arr.ndim != 3
            or arr.shape[-1] not in (3, 4)
            or arr.shape[0] < 2
            or arr.shape[1] < 2
        ):
            raise ValueError(
                "texture must be an (H, W, 3) or (H, W, 4) array with H, W >= 2; "
                f"got shape {arr.shape}."
            )
        rgba = arr.astype(float)
        if np.issubdtype(arr.dtype, np.integer):
            # Integer textures are 8-bit 0-255 by convention (alpha included).
            rgba = rgba / 255.0
        else:
            # Float textures are assumed to be in [0, 1]; if any channel exceeds 1 the
            # texture is normalised by its own peak so a stray highlight cannot black out
            # the globe. The peak is taken over the finite RGB values only, so a NaN cell
            # neither disables normalisation nor triggers an all-NaN warning.
            rgb = rgba[..., :3]
            finite_rgb = rgb[np.isfinite(rgb)]
            peak = finite_rgb.max() if finite_rgb.size else 0.0
            if peak > 1.0:
                # Scale only the RGB channels; alpha is already assumed to be in [0, 1].
                rgba[..., :3] = rgb / peak
        rgb = np.nan_to_num(np.clip(rgba[..., :3] * brightness, 0.0, 1.0), nan=0.0)
        if arr.shape[-1] == 4:
            alpha = np.nan_to_num(np.clip(rgba[..., 3:4], 0.0, 1.0), nan=1.0)
        else:
            alpha = np.ones((*rgb.shape[:2], 1))
        return np.ascontiguousarray(np.concatenate([rgb, alpha], axis=-1))

    def _prepare(self) -> None:
        """Sample the texture and build the base sphere mesh once, caching the results on the instance.

        Computes and caches the un-spun vertex coordinates `(3, n_lat * n_lon)`, the per-face `facecolors`
        `(n_lat - 1, n_lon - 1, 4)` sampled at face centres, and the fixed axial-tilt rotation matrix. Idempotent:
        repeated calls (e.g. one per animation frame) return immediately.
        """
        if self._base_xyz is not None:
            return

        lat_edges = np.linspace(90.0, -90.0, self._n_lat)
        lon_edges = np.linspace(-180.0, 180.0, self._n_lon)
        lon_grid, lat_grid = np.meshgrid(np.deg2rad(lon_edges), np.deg2rad(lat_edges))
        x = np.cos(lat_grid) * np.cos(lon_grid)
        y = np.cos(lat_grid) * np.sin(lon_grid)
        z = np.sin(lat_grid)
        self._base_xyz = np.stack([x.ravel(), y.ravel(), z.ravel()])

        # Sample the texture at each face centre -> (n_lat - 1, n_lon - 1, 4).
        lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        height, width = self._texture.shape[:2]
        rows = np.clip(
            np.round((90.0 - lat_centers) / 180.0 * (height - 1)).astype(int),
            0,
            height - 1,
        )
        cols = np.clip(
            np.round((lon_centers + 180.0) / 360.0 * (width - 1)).astype(int),
            0,
            width - 1,
        )
        row_idx, col_idx = np.meshgrid(rows, cols, indexing="ij")
        self._facecolors = self._texture[row_idx, col_idx]

        self._tilt_matrix = self._rotation_x(self._tilt_deg)

    @staticmethod
    def _rotation_x(deg: float) -> np.ndarray:
        """Return the 3x3 matrix rotating a point cloud by `deg` degrees about the x-axis."""
        rad = np.deg2rad(deg)
        cos, sin = np.cos(rad), np.sin(rad)
        return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])

    @staticmethod
    def _rotation_z(deg: float) -> np.ndarray:
        """Return the 3x3 matrix rotating a point cloud by `deg` degrees about the z-axis (the polar axis)."""
        rad = np.deg2rad(deg)
        cos, sin = np.cos(rad), np.sin(rad)
        return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])

    def _spun_mesh(self, spin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the `(x, y, z)` sphere mesh spun `spin` degrees about its tilted polar axis.

        The globe is rotated about its own polar axis (`z` in the body frame) and then the fixed axial tilt is applied,
        so the tilt stays put in space while the surface turns under it. Only the pre-computed base vertices are
        rotated -- the `facecolors` are untouched.

        Args:
            spin: Rotation about the polar axis, in degrees.

        Returns:
            tuple: Three `(n_lat, n_lon)` arrays `(x, y, z)` for `Axes3D.plot_surface`.
        """
        coords = self._tilt_matrix @ (self._rotation_z(spin) @ self._base_xyz)
        return tuple(coords.reshape(3, self._n_lat, self._n_lon))

    def _resolve_axes(self, ax: Axes3D | None, options: dict) -> tuple[Figure, Axes3D]:
        """Resolve the 3-D `(fig, ax)` to draw on, creating a 3-D axes if none was supplied.

        Args:
            ax: An explicit 3-D axes for this call, or `None` to fall back to the instance's `ax`/`fig` or a new one.
            options: The resolved render options (uses `figsize` when a new figure is made).

        Returns:
            tuple: `(fig, ax)` where `ax` is an `Axes3D`.

        Raises:
            ValueError: If a supplied axes is not a 3-D axes.
        """
        target = ax if ax is not None else self._ax
        if target is None:
            fig = self._fig or plt.figure(figsize=options["figsize"])
            target = fig.add_subplot(projection="3d")
        else:
            if not isinstance(target, Axes3D):
                raise ValueError(
                    "TexturedGlobeGlyph needs a 3-D axes; create one with fig.add_subplot(projection='3d')."
                )
            fig = _root_figure(target)
        return fig, target

    # ------------------------------------------------------------------ #
    # Introspection (mirrors Glyph / HistogramGlyph)                      #
    # ------------------------------------------------------------------ #
    @property
    def texture(self) -> np.ndarray:
        """The normalised RGBA texture (float `(H, W, 4)` in `[0, 1]`)."""
        return self._texture

    @property
    def tilt_deg(self) -> float:
        """Axial tilt of the polar axis from vertical, in degrees."""
        return self._tilt_deg

    @property
    def n_lon(self) -> int:
        """Number of longitude samples in the sphere mesh."""
        return self._n_lon

    @property
    def n_lat(self) -> int:
        """Number of latitude samples in the sphere mesh."""
        return self._n_lat

    @property
    def brightness(self) -> float:
        """The brightness multiplier applied to the RGB channels."""
        return self._brightness

    @property
    def default_options(self) -> dict:
        """The resolved render options (`figsize`, `elev`, `azim`, `background`)."""
        return self._default_options

    @classmethod
    def option_keys(cls) -> set[str]:
        """Return the keyword-argument keys this glyph accepts.

        Resolves from the class-level `DEFAULT_OPTIONS` so the accepted keys can be inspected without constructing an
        instance. Mirrors `cleopatra.glyphs.base.glyph.Glyph.option_keys` (`TexturedGlobeGlyph` is a standalone class,
        not a `Glyph` subclass).

        Returns:
            set: The accepted option keys for this glyph class.

        Examples:
            ```python
            >>> from cleopatra.glyphs.globe.textured_globe_glyph import TexturedGlobeGlyph
            >>> "elev" in TexturedGlobeGlyph.option_keys()
            True

            ```
        """
        return set(cls.DEFAULT_OPTIONS)

    @classmethod
    def filter_kwargs(cls, kwargs: dict) -> dict:
        """Return only the subset of `kwargs` whose keys this glyph accepts.

        Args:
            kwargs: A mapping of candidate option keys to values.

        Returns:
            dict: The entries of `kwargs` whose keys are in `option_keys()`.

        Examples:
            ```python
            >>> from cleopatra.glyphs.globe.textured_globe_glyph import TexturedGlobeGlyph
            >>> sorted(TexturedGlobeGlyph.filter_kwargs({"elev": 30, "bogus": 1}))
            ['elev']

            ```
        """
        keys = cls.option_keys()
        return {key: val for key, val in kwargs.items() if key in keys}

    @classmethod
    def _reject_unknown_options(cls, kwargs: dict) -> None:
        """Raise `ValueError` if `kwargs` holds keys this glyph does not accept.

        Mirrors the rest of the package (`Glyph._merge_kwargs`,
        `HistogramGlyph._apply_options`), which reject unknown options rather than
        silently ignoring them, so a typo like `elevv=` surfaces immediately.

        Args:
            kwargs: The render options passed to `__init__`/`draw`/`animate`.

        Raises:
            ValueError: If any key is not in `option_keys()`.
        """
        unknown = set(kwargs) - cls.option_keys()
        if unknown:
            raise ValueError(
                f"Unknown option(s) {sorted(unknown)}; accepted keys are "
                f"{sorted(cls.option_keys())}."
            )

    # ------------------------------------------------------------------ #
    # Rendering                                                            #
    # ------------------------------------------------------------------ #
    def draw(
        self, ax: Axes3D | None = None, *, spin: float = 0.0, **kwargs
    ) -> tuple[Figure, Axes3D]:
        """Render the textured globe onto a 3-D axes at a given spin angle.

        Args:
            ax: A 3-D matplotlib axes (`Axes3D`) to draw on. If `None`, the instance's `ax` is used, or a new 3-D axes
                is created on the instance's `fig` (or a new figure sized by the `figsize` option).
            spin: Rotation of the globe about its tilted polar axis, in degrees. The camera stays put.
            **kwargs: Render options overriding the instance defaults for this call (`figsize`, `elev`, `azim`,
                `background`).

        Returns:
            tuple: `(fig, ax)` -- the figure and the 3-D axes the globe was drawn on. The surface artist is also
                available as the `surface` attribute.

        Raises:
            ValueError: If `ax` is supplied but is not a 3-D axes, or if an unknown render option is passed.

        Examples:
            ```python
            >>> import numpy as np
            >>> from cleopatra.glyphs.globe.textured_globe_glyph import TexturedGlobeGlyph
            >>> texture = np.zeros((16, 32, 3), dtype=np.uint8)
            >>> texture[:, :16] = (200, 40, 40)
            >>> fig, ax = TexturedGlobeGlyph(texture, n_lon=36, n_lat=18).draw(spin=90.0)
            >>> ax.name
            '3d'

            ```
        """
        self._reject_unknown_options(kwargs)
        options = self._default_options.copy()
        options.update(kwargs)
        fig, target = self._resolve_axes(ax, options)
        self._prepare()
        _clear_prior_render_artists(target)

        if options["background"] is not None:
            fig.set_facecolor(options["background"])
            target.set_facecolor(options["background"])

        x, y, z = self._spun_mesh(spin)
        surface = target.plot_surface(
            x,
            y,
            z,
            facecolors=self._facecolors,
            rstride=1,
            cstride=1,
            shade=False,
            antialiased=False,
            linewidth=0,
        )
        target.set_box_aspect((1, 1, 1))
        target.set_xlim(-1, 1)
        target.set_ylim(-1, 1)
        target.set_zlim(-1, 1)
        target.set_axis_off()
        target.view_init(elev=options["elev"], azim=options["azim"])

        _mark_render_artists(target, surface)
        self._surface = surface
        return fig, target

    def animate(
        self,
        ax: Axes3D | None = None,
        *,
        n_frames: int = 60,
        revolutions: float = 1.0,
        start_spin: float = 0.0,
        interval: int = 50,
        **kwargs,
    ) -> FuncAnimation:
        """Return a `FuncAnimation` that spins the globe about its polar axis.

        The texture is sampled once (via `draw`'s cached `_prepare`); each frame only rotates the pre-computed vertices
        and re-draws, so the per-frame cost is dominated by matplotlib's surface draw at the chosen mesh resolution.
        Save it with `cleopatra.glyphs.base.animation` (or matplotlib's own writers).

        Args:
            ax: A 3-D axes to animate on, or `None` to create one (see `draw`).
            n_frames: Number of frames in the animation.
            revolutions: How many full turns the globe makes over `n_frames` (`1.0` = one 360 deg rotation).
            start_spin: Spin angle of the first frame, in degrees.
            interval: Delay between frames in milliseconds (matplotlib playback hint).
            **kwargs: Render options forwarded to `draw` on every frame (`figsize`, `elev`, `azim`, `background`).

        Returns:
            matplotlib.animation.FuncAnimation: The animation, ready to save or embed.

        Raises:
            ValueError: If `ax` is supplied but is not a 3-D axes, or if an unknown render option is passed.

        Examples:
            ```python
            >>> import numpy as np
            >>> from cleopatra.glyphs.globe.textured_globe_glyph import TexturedGlobeGlyph
            >>> globe = TexturedGlobeGlyph(np.zeros((8, 16, 3), dtype=np.uint8), n_lon=24, n_lat=12)
            >>> anim = globe.animate(n_frames=4)
            >>> list(anim.new_frame_seq())    # one entry per rendered frame
            [0, 1, 2, 3]

            ```
        """
        self._reject_unknown_options(kwargs)
        options = self._default_options.copy()
        options.update(kwargs)
        fig, target = self._resolve_axes(ax, options)
        self._prepare()
        angles = start_spin + np.linspace(
            0.0, 360.0 * revolutions, n_frames, endpoint=False
        )

        def _update(frame_index: int):
            self.draw(target, spin=float(angles[frame_index]), **kwargs)
            return (self._surface,)

        return FuncAnimation(
            fig, _update, frames=n_frames, interval=interval, blit=False
        )

    @property
    def surface(self):
        """The `Poly3DCollection` from the most recent `draw`, or `None` before the first draw."""
        return self._surface
