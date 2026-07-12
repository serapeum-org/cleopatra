"""2-D kernel-density (isochrone) visualization.

Provides `KDEGlyph` for drawing a 2-D Gaussian kernel-density estimate of a
point cloud as filled (`contourf`) or line (`contour`) density bands. The
density is evaluated on a regular grid with a few lines of numpy — **no
scipy** — and coloured through the shared `Glyph._prepare_scalar_mapping`
pipeline, so `vmin` / `vmax`, `levels`, `ticks_spacing`, and `color_scale`
behave exactly as they do for the other glyphs. The glyph is geometry- and
CRS-agnostic: it takes plain `x` / `y` arrays plus an optional matplotlib
clip path.

The estimator uses an isotropic Gaussian kernel with Scott's-rule bandwidth
(scaled by an optional `bw_method` multiplier). It is intended for typical
scientific point clouds; it is **not** a drop-in for
`scipy.stats.gaussian_kde` (no anisotropic/diagonal bandwidth, no weights).

Examples:
    - Filled density of a small cluster:
        ```python
        >>> import numpy as np
        >>> from cleopatra.kde_glyph import KDEGlyph
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(size=200)
        >>> y = rng.normal(size=200)
        >>> glyph = KDEGlyph(x, y, gridsize=40)
        >>> fig, ax, cs = glyph.plot()

        ```
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath

from cleopatra.colors import resolve_single_layer_style, resolve_style_norm
from cleopatra.glyph import Glyph
from cleopatra.hillshade import resolve_hillshade, shade_grid
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS

#: Upper bound on the number of (grid-cell × data-point) products evaluated in
#: a single numpy block. The density sum is chunked over the data points so a
#: large `gridsize` / point count never materialises one giant array; this caps
#: the temporary at ~`MAX_KDE_BLOCK` floats (a few tens of MB).
MAX_KDE_BLOCK = 4_000_000

#: Option keys for KDEGlyph. `ticks_spacing` is `None` so the shared
#: `_prepare_scalar_mapping` helper auto-derives it from the density range.
KDE_DEFAULT_OPTIONS = {
    "levels": 10,
    "shade": True,
    "bw_method": None,
    "gridsize": 100,
    "vmin": None,
    "vmax": None,
    "ticks_spacing": None,
    "add_colorbar": True,
    "hillshade": False,
    "style": None,
}
KDE_DEFAULT_OPTIONS = STYLE_DEFAULTS | KDE_DEFAULT_OPTIONS


class KDEGlyph(Glyph):
    """Visualization class for 2-D kernel-density estimates.

    Evaluates an isotropic Gaussian KDE of a `(x, y)` point cloud on a
    regular grid (numpy only, no scipy) and draws it as filled or line
    density contours, coloured through the shared scalar-mapping pipeline.

    Args:
        x: 1D array of point x-coordinates.
        y: 1D array of point y-coordinates. Must match the length of `x`.
        clip_path: Optional matplotlib `Path` or `Patch` that clips the
            drawn contours (e.g. a country/basin outline supplied by the
            caller). A `Patch` is used directly; a `Path` is interpreted in
            data coordinates. Default is None (no clipping).
        ax: Pre-existing axes to draw on. Default is None.
        fig: Pre-existing figure. Default is None.
        **kwargs: Override any key in `KDE_DEFAULT_OPTIONS`: `levels` (int
            count or explicit sequence of density levels, default 10),
            `shade` (filled `contourf` vs line `contour`, default True),
            `bw_method` (None for Scott's rule, or a positive float
            bandwidth multiplier), `gridsize` (density grid resolution,
            default 100), plus the shared colour options (`cmap`, `vmin`,
            `vmax`, `color_scale`, `ticks_spacing`, `cbar_label`,
            `figsize`, `title`). Set `add_colorbar=False` to suppress the
            per-glyph colorbar (default True).

    Raises:
        ValueError: If `x` and `y` have mismatched shapes, if fewer than
            two points are given, if `bw_method` is non-positive, or if a
            coordinate has zero spread (a degenerate kernel).

    Examples:
        - Evaluate the density grid directly (no rendering):
            ```python
            >>> import numpy as np
            >>> from cleopatra.kde_glyph import KDEGlyph
            >>> rng = np.random.default_rng(1)
            >>> x = rng.normal(size=500)
            >>> y = rng.normal(size=500)
            >>> glyph = KDEGlyph(x, y, gridsize=50)
            >>> gx, gy, density = glyph.evaluate()
            >>> density.shape
            (50, 50)
            >>> bool(density.sum() > 0)
            True

            ```

    See Also:
        cleopatra.glyph.Glyph._prepare_scalar_mapping: Shared
            norm/colorbar/ticks pipeline used to colour the density.
        cleopatra.mesh_glyph.MeshGlyph: Contour rendering for unstructured
            meshes.
    """

    #: Option keys this glyph accepts (see `Glyph.option_keys`/`filter_kwargs`).
    DEFAULT_OPTIONS = KDE_DEFAULT_OPTIONS

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        clip_path: MplPath | Patch | None = None,
        ax: Axes = None,
        fig: Figure = None,
        **kwargs,
    ):
        super().__init__(default_options=KDE_DEFAULT_OPTIONS, fig=fig, ax=ax, **kwargs)
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        if self.x.shape != self.y.shape:
            raise ValueError(
                f"x and y must have the same shape, got {self.x.shape} "
                f"and {self.y.shape}."
            )
        if self.x.size < 2:
            raise ValueError(f"KDE needs at least 2 points, got {self.x.size}.")
        bw_method = self.default_options["bw_method"]
        if bw_method is not None and bw_method <= 0:
            raise ValueError(
                f"bw_method must be a positive float or None, got {bw_method}."
            )
        self.clip_path = clip_path
        self.cbar = None

    def _bandwidth(self) -> float:
        """Return Scott's-rule bandwidth, scaled by the `bw_method` option.

        Scott's rule in `d` dimensions is `n ** (-1 / (d + 4))`; for the 2-D
        estimator here that is `n ** (-1 / 6)`. The optional `bw_method`
        multiplier (default 1.0) widens (`> 1`) or narrows (`< 1`) the kernel.

        Returns:
            float: The bandwidth factor applied to each coordinate's
                standard deviation.
        """
        n = self.x.size
        multiplier = self.default_options["bw_method"] or 1.0
        return multiplier * n ** (-1.0 / 6.0)

    def evaluate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the KDE on a regular grid spanning the point bounds.

        Builds a `gridsize × gridsize` grid over the `[x.min, x.max] ×
        [y.min, y.max]` bounding box and sums an isotropic Gaussian kernel
        (Scott's-rule bandwidth) over the points. The sum is chunked over
        the data points so memory stays bounded (see `MAX_KDE_BLOCK`) even
        for large `gridsize` or point counts.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The grid `gx`, `gy`
                (each `gridsize × gridsize`) and the density evaluated on
                that grid (same shape), normalised to integrate to ~1.

        Raises:
            ValueError: If either coordinate has zero spread (its standard
                deviation is 0), which would give a degenerate kernel.

        Examples:
            - The density peaks near a tight synthetic cluster:
                ```python
                >>> import numpy as np
                >>> from cleopatra.kde_glyph import KDEGlyph
                >>> rng = np.random.default_rng(2)
                >>> pts = rng.normal(scale=0.1, size=300)
                >>> x = np.concatenate([pts, pts + 5.0])
                >>> y = np.concatenate([pts, pts + 5.0])
                >>> gx, gy, d = KDEGlyph(x, y, gridsize=60).evaluate()
                >>> peak = np.unravel_index(int(np.argmax(d)), d.shape)
                >>> bool(min(abs(gx[peak] - 0.0), abs(gx[peak] - 5.0)) < 1.0)
                True

                ```
        """
        x, y = self.x, self.y
        n = x.size
        bw = self._bandwidth()
        sx, sy = x.std() * bw, y.std() * bw
        if sx == 0 or sy == 0:
            raise ValueError(
                "Cannot build a KDE: a coordinate has zero spread "
                "(degenerate kernel). Provide points that vary in x and y."
            )

        gridsize = int(self.default_options["gridsize"])
        gx, gy = np.meshgrid(
            np.linspace(x.min(), x.max(), gridsize),
            np.linspace(y.min(), y.max(), gridsize),
        )
        gx_flat = gx.ravel()[:, None]
        gy_flat = gy.ravel()[:, None]

        # Sum the kernel over the points in blocks so the temporary
        # (grid-cells × block) array never exceeds ~MAX_KDE_BLOCK floats.
        block = max(1, MAX_KDE_BLOCK // gx_flat.shape[0])
        density_flat = np.zeros(gx_flat.shape[0], dtype=float)
        for start in range(0, n, block):
            xs = x[start : start + block]
            ys = y[start : start + block]
            dx = (gx_flat - xs) / sx
            dy = (gy_flat - ys) / sy
            density_flat += np.exp(-0.5 * (dx**2 + dy**2)).sum(axis=1)

        density = density_flat.reshape(gx.shape) / (2.0 * np.pi * sx * sy * n)
        return gx, gy, density

    def _resolve_levels(self, density: np.ndarray) -> np.ndarray:
        """Resolve the `levels` option to explicit, increasing density edges.

        An integer becomes that many evenly-spaced edges across the density
        range; an explicit sequence is sorted and used verbatim. Returning
        explicit edges (rather than an int) keeps `contourf`/`contour` in
        step with the `BoundaryNorm` the shared pipeline builds from the
        same `levels` option.

        Args:
            density: The evaluated density grid (for its value range).

        Returns:
            np.ndarray: The sorted, increasing contour level edges.
        """
        levels = self.default_options["levels"]
        if isinstance(levels, (int, np.integer)) and not isinstance(levels, bool):
            return np.linspace(float(density.min()), float(density.max()), int(levels))
        return np.sort(np.asarray(levels, dtype=float))

    def _apply_clip(self, contour_set: Any) -> None:
        """Clip the drawn contour set to `self.clip_path`, if any.

        A `Patch` clips in data coordinates; a `Path` is clipped in data
        coordinates (`ax.transData`). No-op when no clip path was supplied.

        A `Patch` clips through its own transform. A patch the caller just
        constructed (and has not added to an axes) carries an identity
        transform, which would clip in display space rather than data space.
        Rather than mutate the caller's patch, an unattached patch is clipped
        against its geometry directly — its `Path` under
        `patch_transform + ax.transData` — which is what `Axes.add_patch`
        would resolve to. A patch already added to an axes is used as-is
        (its own transform is honoured).

        Args:
            contour_set: The `QuadContourSet` returned by
                `contourf`/`contour`.

        Raises:
            TypeError: If `clip_path` is neither a matplotlib `Path` nor a
                `Patch`.
        """
        clip = self.clip_path
        if clip is None:
            return
        if isinstance(clip, Patch):
            if clip.axes is None:
                # Clip in data coordinates without mutating the caller's patch.
                transform = clip.get_patch_transform() + self.ax.transData
                contour_set.set_clip_path(clip.get_path(), transform)
            else:
                contour_set.set_clip_path(clip)
        elif isinstance(clip, MplPath):
            contour_set.set_clip_path(clip, transform=self.ax.transData)
        else:
            raise TypeError(
                "clip_path must be a matplotlib Path or Patch, got "
                f"{type(clip).__name__}."
            )

    @property
    def style(self) -> str | None:
        """Name of the `DATA_STYLES` preset currently applied, or `None`.

        Reads back the preset set via the `style` constructor kwarg, a
        `plot(style=...)` call, or `apply_style`.
        """
        return self.default_options.get("style")

    def apply_style(
        self,
        style: str,
        *,
        hillshade: bool | dict | None = None,
        add_colorbar: bool | None = None,
        title: str | None = None,
    ):
        """Apply a continuous `DATA_STYLES` preset by name, re-rendering in place.

        A discoverable wrapper over `plot(style=...)` for restyling an
        already-built glyph: it redraws on the glyph's existing axes (clearing
        the previous render first) or, if the glyph has not been plotted yet,
        on a new figure.

        Args:
            style: A continuous `cleopatra.colors.DATA_STYLES` preset name.
            hillshade: Optional relief shading, forwarded to `plot`.
            add_colorbar: Optional colorbar toggle, forwarded to `plot`.
            title: Optional title, forwarded to `plot`.

        Returns:
            tuple[Figure, Axes, QuadContourSet]: The `plot` result.

        Raises:
            ValueError: If `style` is unknown, or is categorical (a density is
                continuous), raised by `plot`.
        """
        if getattr(self, "ax", None) is not None:
            if self.cbar is not None:
                self.cbar.remove()
                self.cbar = None
            for inset in list(self.ax.child_axes):
                inset.remove()
            self.ax.clear()
            return self.plot(
                ax=self.ax, title=title, add_colorbar=add_colorbar,
                hillshade=hillshade, style=style,
            )
        return self.plot(
            title=title, add_colorbar=add_colorbar, hillshade=hillshade, style=style
        )

    def plot(
        self,
        ax: Axes = None,
        title: str | None = None,
        add_colorbar: bool | None = None,
        hillshade: bool | dict | None = None,
        style: str | None = None,
    ):
        """Render the 2-D density as filled or line contours.

        Evaluates the KDE via `evaluate`, colours it through
        `_prepare_scalar_mapping`, and draws `contourf` (when `shade`) or
        `contour` (otherwise). An optional `clip_path` restricts the drawn
        contours.

        Args:
            ax: Axes to draw on. Falls back to the axes supplied at
                construction, otherwise a new figure/axes is created.
            title: Plot title. Overrides `default_options["title"]` when
                given.
            add_colorbar: Override the `add_colorbar` option for this call
                — True draws the colorbar, False suppresses it. Defaults to
                None, which keeps the value set at construction.
            hillshade: Relief-shade the density surface for this call (`True`
                or an options dict; see `cleopatra.hillshade`). Defaults to
                None, which keeps the value set at construction. Accepting it
                here mirrors `ArrayGlyph.plot`/`MeshGlyph.plot`, so `hillshade`
                works the same way across all three glyphs.
            style: Name of a continuous `cleopatra.colors.DATA_STYLES` preset
                to colour the density with (its cmap + norm; composes with
                `hillshade`). Defaults to None, keeping the construction value.
                A categorical preset has no meaning for a continuous density
                and raises `ValueError`. Valid names:
                `sorted(cleopatra.colors.DATA_STYLES)`.

        Returns:
            tuple[Figure, Axes, QuadContourSet]: The figure, the axes, and
                the contour set (the mappable the colorbar attaches to).

        Raises:
            ValueError: If a coordinate has zero spread (via `evaluate`).
            TypeError: If `clip_path` is an unsupported type (via the clip
                step).

        Examples:
            - Filled contours add a colorbar by default:
                ```python
                >>> import numpy as np
                >>> from cleopatra.kde_glyph import KDEGlyph
                >>> rng = np.random.default_rng(3)
                >>> x, y = rng.normal(size=300), rng.normal(size=300)
                >>> glyph = KDEGlyph(x, y, gridsize=40)
                >>> fig, ax, cs = glyph.plot()
                >>> glyph.cbar is not None
                True

                ```
            - Line contours (`shade=False`) and no colorbar:
                ```python
                >>> import numpy as np
                >>> from cleopatra.kde_glyph import KDEGlyph
                >>> rng = np.random.default_rng(4)
                >>> x, y = rng.normal(size=300), rng.normal(size=300)
                >>> glyph = KDEGlyph(x, y, gridsize=40, shade=False)
                >>> fig, ax, cs = glyph.plot(add_colorbar=False)
                >>> glyph.cbar is None
                True

                ```
        """
        if ax is not None:
            self.ax = ax
            self.fig = ax.get_figure()
        elif self.ax is None:
            self.fig, self.ax = self.create_figure_axes()
        ax = self.ax
        opts = self.default_options

        if title is not None:
            opts["title"] = title
        draw_colorbar = opts["add_colorbar"] if add_colorbar is None else add_colorbar

        gx, gy, density = self.evaluate()
        level_edges = self._resolve_levels(density)
        norm, cbar_kw, _ = self._prepare_scalar_mapping(density)
        cmap = opts["cmap"]

        # Named data-style preset: a continuous preset overrides the density's
        # cmap + norm (and composes with hillshade below). A categorical preset
        # has no meaning for a continuous density surface, so reject it. The
        # cmap is resolved into a LOCAL (not `opts`), so a per-call `style` does
        # not leak its colormap into a later `plot()` on the same instance.
        # Persist a plot-time `style` name into the options so `self.style`
        # reads it back (matching ArrayGlyph, whose **kwargs persist it). Only
        # the name is stored -- the resolved cmap stays a local (no leak).
        if style is not None:
            opts["style"] = style
        style = opts.get("style")
        if style is not None:
            _, cfg = resolve_single_layer_style(style)
            if cfg.get("categories") is not None:
                raise ValueError(
                    f"data style {style!r} is categorical; KDEGlyph colours a "
                    "continuous density, so only continuous presets apply"
                )
            cmap = cfg["cmap"]
            norm, _, _ = resolve_style_norm(np.asarray(density, dtype=float), cfg)
            # Drop the linear ticks so the colorbar matches the preset norm.
            cbar_kw.pop("ticks", None)

        hillshade = resolve_hillshade(
            hillshade if hillshade is not None else opts.get("hillshade")
        )
        if hillshade is not None:
            # Treat the density as a surface and relief-shade it, so the
            # "density terrain" (peaks and ridges) reads by form. The shaded
            # RGBA image carries no scalar array, so the colorbar attaches to a
            # ScalarMappable proxy carrying the same cmap/norm.
            hs_norm = norm if norm is not None else Normalize(
                vmin=float(density.min()), vmax=float(density.max())
            )
            rgba = shade_grid(density, cmap, norm=hs_norm, **hillshade)
            extent = [float(gx.min()), float(gx.max()), float(gy.min()), float(gy.max())]
            mappable = ax.imshow(rgba, extent=extent, origin="lower", aspect="auto")
            self._apply_clip(mappable)
            self.im = mappable
            if draw_colorbar:
                proxy = ScalarMappable(norm=hs_norm, cmap=cmap)
                proxy.set_array(density)
                self.cbar = self.create_color_bar(ax, proxy, cbar_kw)
            if opts["title"]:
                ax.set_title(opts["title"], fontsize=opts["title_size"])
            return self.fig, ax, mappable

        render = ax.contourf if opts["shade"] else ax.contour
        contour_set = render(
            gx, gy, density, levels=level_edges, cmap=cmap, norm=norm
        )
        self._apply_clip(contour_set)
        self.im = contour_set

        if draw_colorbar:
            self.cbar = self.create_color_bar(ax, contour_set, cbar_kw)

        if opts["title"]:
            ax.set_title(opts["title"], fontsize=opts["title_size"])

        return self.fig, ax, contour_set
