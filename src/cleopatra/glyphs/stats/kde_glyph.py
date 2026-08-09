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
        >>> from cleopatra.glyphs.stats.kde_glyph import KDEGlyph
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
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath

from cleopatra.styling.colorbar import ColorBar, _resolve_colorbar, _warn_deprecated_cbar_kwargs
from cleopatra.styling.params import Contour, DataStyle
from cleopatra.styling.scaling import ColorScaling
from cleopatra.styling.colors import (
    resolve_colormap,
    resolve_single_layer_style,
    resolve_style_norm,
)
from cleopatra.glyphs.base.glyph import Glyph, _root_figure
from cleopatra.glyphs.base.hillshade import resolve_hillshade, shade_grid
from cleopatra.styling.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS

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
        **kwargs: Construction-time overrides for the non-grouped
            `KDE_DEFAULT_OPTIONS`: `shade` (filled `contourf` vs line
            `contour`, default True), `bw_method` (None for Scott's rule,
            or a positive float bandwidth multiplier), `gridsize` (density
            grid resolution, default 100), plus the shared appearance /
            colorbar options (`cmap`, `vmin`, `vmax`, `ticks_spacing`,
            `cbar_label`, `figsize`, `title`). Set `add_colorbar=False` to
            suppress the per-glyph colorbar (default True). The colour
            scale, density `levels`, and preset / relief shading are no
            longer construction kwargs -- pass them to `plot()` via
            `color=ColorScaling(...)`, `contour=Contour(levels=...)`, and
            `data_style=DataStyle(...)` respectively (a loose `color_scale`
            / `levels` / `style` keyword now raises).

    Raises:
        ValueError: If `x` and `y` have mismatched shapes, if fewer than
            two points are given, if `bw_method` is non-positive, or if a
            coordinate has zero spread (a degenerate kernel).

    Examples:
        - Evaluate the density grid directly (no rendering):
            ```python
            >>> import numpy as np
            >>> from cleopatra.glyphs.stats.kde_glyph import KDEGlyph
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
        cleopatra.glyphs.base.glyph.Glyph._prepare_scalar_mapping: Shared
            norm/colorbar/ticks pipeline used to colour the density.
        cleopatra.glyphs.gridded.mesh_glyph.MeshGlyph: Contour rendering for unstructured
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
        ax: Axes | None = None,
        fig: Figure | None = None,
        **kwargs,
    ):
        _warn_deprecated_cbar_kwargs(kwargs)
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
        self.cbar: Colorbar | None = None
        #: The `AxesImage` (hillshaded) or `QuadContourSet` mappable from
        #: the most recent `plot` call; `None` before first render.
        self.im: Any = None

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
        return float(multiplier * n ** (-1.0 / 6.0))

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
                >>> from cleopatra.glyphs.stats.kde_glyph import KDEGlyph
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
        # `_apply_clip` is only called from `plot()` after `self.ax` is resolved.
        assert self.ax is not None
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
        `plot(data_style=DataStyle(style=...))` call, or `apply_style`.
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

        A discoverable wrapper over `plot(data_style=DataStyle(style=...))` for restyling an
        already-built glyph. It redraws **in place** on the glyph's own axes
        (taking full ownership -- do not use on a shared axes), or on a fresh
        figure if the glyph was never plotted or its figure was closed. The
        applied style is **sticky** (survives a later plain `plot()`);
        `plot(data_style=DataStyle(style=None))` clears it.

        Args:
            style: A continuous `cleopatra.styling.colors.DATA_STYLES` preset name.
            hillshade: Optional relief shading, forwarded to `plot`.
            add_colorbar: Optional colorbar toggle, forwarded to `plot`.
            title: Optional title, forwarded to `plot`.

        Returns:
            tuple[Figure, Axes, QuadContourSet]: The `plot` result.

        Raises:
            ValueError: If `style` is unknown, or is categorical (a density is
                continuous).
        """
        _, cfg = resolve_single_layer_style(style)
        if cfg.get("categories") is not None:
            raise ValueError(
                f"data style {style!r} is categorical; KDEGlyph colours a "
                "continuous density, so only continuous presets apply"
            )
        self._reset_axes_for_restyle()
        # Only override hillshade when the caller passed one; leaving it
        # unset keeps any sticky relief shading (a plain None would clear it).
        data_style = (
            DataStyle(style=style)
            if hillshade is None
            else DataStyle(style=style, hillshade=hillshade)
        )
        return self.plot(
            ax=self.ax,
            title=title,
            add_colorbar=add_colorbar,
            data_style=data_style,
        )

    def plot(
        self,
        ax: Axes | None = None,
        title: str | None = None,
        add_colorbar: bool | None = None,
        colorbar: bool | ColorBar | None = None,
        color: ColorScaling | None = None,
        contour: Contour | None = None,
        data_style: DataStyle | None = None,
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
            colorbar: Typed `ColorBar` spec (or `True`/`False`/`None`) for the
                colorbar's placement, caption, and sizing; resolved into the
                `cbar_*` options. A `ColorBar`/`True` also enables the bar and is
                **sticky** -- it persists into later plots, overriding a
                construction-time `add_colorbar=False`; an explicit
                `add_colorbar=` argument still wins the on/off decision.
            hillshade: Relief-shade the density surface for this call (`True`
                or an options dict; see `cleopatra.glyphs.base.hillshade`). Defaults to
                None, which keeps the value set at construction. Accepting it
                here mirrors `ArrayGlyph.plot`/`MeshGlyph.plot`, so `hillshade`
                works the same way across all three glyphs.
            style: Name of a continuous `cleopatra.styling.colors.DATA_STYLES` preset
                to colour the density with (its cmap + norm; composes with
                `hillshade`). The preset name is **sticky** -- once set it
                persists into `default_options` and survives later plain
                `plot()` calls (like `ArrayGlyph`), and `self.style` reads it
                back; the resolved cmap is not persisted, so it never leaks.
                Not passing `style` keeps the current preset; passing
                `style=None` clears it back to the plain density colouring
                (unlike `hillshade`, which reverts to its construction value).
                A categorical preset has no meaning for a continuous density
                and raises `ValueError`. Valid names:
                `sorted(cleopatra.styling.colors.DATA_STYLES)`.

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
                >>> from cleopatra.glyphs.stats.kde_glyph import KDEGlyph
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
                >>> from cleopatra.glyphs.stats.kde_glyph import KDEGlyph
                >>> rng = np.random.default_rng(4)
                >>> x, y = rng.normal(size=300), rng.normal(size=300)
                >>> glyph = KDEGlyph(x, y, gridsize=40, shade=False)
                >>> fig, ax, cs = glyph.plot(add_colorbar=False)
                >>> glyph.cbar is None
                True

                ```
        """
        # KDE keeps its persistent default_options across plots, so snapshot
        # every option key these group objects are about to touch (colour
        # scale, contour levels, style/hillshade) before merging. If the new
        # preset turns out invalid below, ALL of them are rolled back -- not
        # just style -- so a co-passed color=/contour= cannot leak into a
        # later plain plot.
        prev_group_opts = {}
        for grp in (color, contour, data_style):
            if grp is not None:
                for key in grp.to_options():
                    if key in self.default_options:
                        prev_group_opts[key] = self.default_options[key]
        self._merge_group_params(color, contour, data_style)

        if ax is not None:
            self.ax = ax
            self.fig = _root_figure(ax)
        elif self.ax is None:
            self.fig, self.ax = self.create_figure_axes()
        ax = self.ax
        opts = self.default_options

        if title is not None:
            opts["title"] = title
        opts.update(_resolve_colorbar(colorbar))
        draw_colorbar = opts["add_colorbar"] if add_colorbar is None else add_colorbar

        gx, gy, density = self.evaluate()
        level_edges = self._resolve_levels(density)
        norm, cbar_kw, _ = self._prepare_scalar_mapping(density)
        cmap = resolve_colormap(opts["cmap"])

        style = opts.get("style")
        if style is not None:
            try:
                _, cfg = resolve_single_layer_style(style)
                if cfg.get("categories") is not None:
                    raise ValueError(
                        f"data style {style!r} is categorical; KDEGlyph colours "
                        "a continuous density, so only continuous presets apply"
                    )
            except ValueError:
                for key, value in prev_group_opts.items():
                    opts[key] = value
                raise
            cfg = {
                **cfg,
                **{k: opts[k] for k in ("vmin", "vmax") if opts.get(k) is not None},
            }
            cmap = resolve_colormap(cfg["cmap"])
            norm, _, _ = resolve_style_norm(np.asarray(density, dtype=float), cfg)
            # Drop the linear ticks so the colorbar matches the preset norm.
            cbar_kw.pop("ticks", None)

        hillshade = resolve_hillshade(opts.get("hillshade"))
        if hillshade is not None:
            hs_norm = (
                norm
                if norm is not None
                else Normalize(vmin=float(density.min()), vmax=float(density.max()))
            )
            rgba = shade_grid(density, cmap, norm=hs_norm, **hillshade)
            extent = (
                float(gx.min()),
                float(gx.max()),
                float(gy.min()),
                float(gy.max()),
            )
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
        contour_set = render(gx, gy, density, levels=level_edges, cmap=cmap, norm=norm)
        self._apply_clip(contour_set)
        self.im = contour_set

        if draw_colorbar:
            self.cbar = self.create_color_bar(ax, contour_set, cbar_kw)

        if opts["title"]:
            ax.set_title(opts["title"], fontsize=opts["title_size"])

        return self.fig, ax, contour_set
