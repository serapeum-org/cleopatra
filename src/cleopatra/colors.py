import importlib.resources
import json
import os
from pathlib import Path
from typing import Any, List, Tuple, Union

import matplotlib as mpl
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.image import AxesImage
from PIL import Image, UnidentifiedImageError

from cleopatra.styles import disjoint_legend, swatch_legend

#: Sequential colormaps for the "haze" data style (white at 0.0, saturating
#: toward the named hue at 1.0) -- the value-modulated-alpha, glowing-rim look
#: used by ECMWF/CAMS aerosol-optical-depth animations. Each value is a ready
#: `matplotlib.colors.Colormap` -- pass it directly as a `cmap` argument
#: anywhere cleopatra or matplotlib accepts one (e.g. `alpha_scaled_image`,
#: `swatch_legend`, `plt.imshow(..., cmap=HAZE_COLORMAPS["dust"])`). Plain
#: constants, not registered into matplotlib's global colormap namespace, so
#: importing this module has no global side effect. See `alpha_scaled_image`
#: and `swatch_legend` for runnable examples using these colormaps.
HAZE_COLORMAPS: dict[str, Colormap] = {
    "organic_matter": LinearSegmentedColormap.from_list(
        "haze_organic_matter",
        ["#ffffff", "#ffd9f2", "#ff5fc9", "#c400a0", "#5c0050", "#200018"],
    ),
    "dust": LinearSegmentedColormap.from_list(
        "haze_dust",
        ["#ffffff", "#fff2b3", "#ffcc33", "#ff6a00", "#7a1500", "#2a0800"],
    ),
}

#: The official ECMWF/CAMS aerosol-optical-depth (AOD at 550 nm) colour scales,
#: as opposed to the stylised `HAZE_COLORMAPS` inspired by them. Colour stops are
#: transcribed verbatim from the open-source Magics rendering engine
#: (`ecmwf/magics`, Apache-2.0 -- `share/magics/styles/default/palettes.json`,
#: the palettes tagged `cams` / "Aerosol optical depth at 550 nm"); only the
#: engine's colour data is vendored, none of its code. Each value is a ready
#: `matplotlib.colors.Colormap`, usable anywhere cleopatra or matplotlib accepts
#: a `cmap` argument (e.g. `plt.imshow(..., cmap=CAMS_AOD_COLORMAPS["blue_yellow_red"])`).
#: Keys describe the ramp; the originating Magics style name is given per entry.
#:
#: These are pure-colour maps (fully opaque), consistent with `HAZE_COLORMAPS`.
#: Magics' `sh_Oranges_aod` additionally ramps *opacity* linearly with value
#: (alpha 0.05 -> 1.0); that alpha is intentionally not baked in here -- reproduce
#: it with cleopatra's separate opacity axis via `alpha_scaled_image(...,
#: alpha_norm=Normalize(vmin, vmax))` or a `DATA_STYLES` entry carrying
#: `alpha_vmin`/`alpha_vmax`, keeping colour and opacity independent.
CAMS_AOD_COLORMAPS: dict[str, Colormap] = {
    # Magics `sh_BuYlRd_aod` -- the canonical CAMS AOD scale.
    "blue_yellow_red": LinearSegmentedColormap.from_list(
        "cams_aod_blue_yellow_red",
        ["#d3d7eb", "#a8afd7", "#8892bf", "#a3a891", "#bebd65", "#d8d239",
         "#f3e70b", "#f4c60a", "#f6a508", "#f88406", "#f96205", "#fb4103",
         "#fd2001", "#ff0000"],
    ),
    # Magics `sh_BuYlRdBr_aod` -- like blue_yellow_red but fading to dark maroon.
    "blue_yellow_red_brown": LinearSegmentedColormap.from_list(
        "cams_aod_blue_yellow_red_brown",
        ["#d2d2ff", "#a1a1ff", "#7070ff", "#8787c7", "#b8b876", "#e9e926",
         "#ffda00", "#ff8a00", "#ff3900", "#f40000", "#c40000", "#930000",
         "#640000"],
    ),
    # Magics `sh_all_aod` / `sh_all_aod550` -- the blue->cyan->green->yellow->red scale.
    "blue_red": LinearSegmentedColormap.from_list(
        "cams_aod_blue_red",
        ["#0000f1", "#004cff", "#00b1ff", "#29ffce", "#7dff7a", "#ceff29",
         "#ffc400", "#ff6800", "#f10800", "#800000"],
    ),
    # Magics `sh_Oranges_aod` -- white->dark-orange (natively alpha-ramped; see above).
    "oranges": LinearSegmentedColormap.from_list(
        "cams_aod_oranges",
        ["#ffefe0", "#fee9d4", "#fee2c6", "#fdd9b4", "#fdd0a2", "#fdc38d",
         "#fdb576", "#fda762", "#fd9a4e", "#fd8c3b", "#f87f2c", "#f3701b",
         "#ec620f", "#e25508", "#d84801", "#c54102", "#b03903", "#9e3303",
         "#8e2d04", "#7f2704"],
    ),
}


def alpha_scaled_image(
    ax: Axes,
    data: np.ndarray,
    cmap: str | Colormap,
    *,
    norm: mcolors.Normalize | None = None,
    alpha_norm: mcolors.Normalize | None = None,
    constant_alpha: float | None = None,
    **imshow_kwargs: Any,
) -> AxesImage:
    """Draw `data` on `ax` with per-pixel opacity tied to its value.

    Builds an RGBA image from `cmap(norm(data))` and overwrites the alpha
    channel with `alpha_norm(data)`, so low values fade toward fully
    transparent instead of being drawn at full opacity in a pale colour.
    This is the "smoke fading into haze" look used by ECMWF/CAMS aerosol
    animations: whatever is plotted underneath (a basemap, another layer)
    shows through wherever the value is near zero. Any non-finite entry in
    `data` (NaN) is drawn fully transparent regardless of `alpha_norm`.

    This is a generic rendering primitive -- it takes any 2D array and any
    colormap, so it composes with any other cleopatra or matplotlib styling
    (a different basemap, a different colormap, a flat or projected axes).

    Args:
        ax: Axes to draw on.
        data: 2D array of values to map.
        cmap: Colormap name or object, e.g. `HAZE_COLORMAPS["dust"]`.
        norm: Normalization mapping `data` to colour. Defaults to
            `Normalize(vmin, vmax)` over the finite range of `data`.
        alpha_norm: Normalization mapping `data` to opacity. Defaults to
            `norm`, so colour and opacity are driven by the same scale; pass
            a separate instance to decouple them (e.g. a steeper alpha ramp
            so faint values vanish sooner than their colour would suggest).
        constant_alpha: If given, draw every finite cell at this fixed opacity
            (clipped to `[0, 1]`) and ignore `alpha_norm` -- e.g. `1.0` for a
            plain opaque field. Non-finite (NaN) cells stay transparent.
        **imshow_kwargs: Forwarded to `ax.imshow` (e.g. `extent`, `origin`,
            `zorder`, `interpolation`).

    Returns:
        AxesImage: The image artist added to `ax`.

    Raises:
        ValueError: If `data` is not 2-dimensional.

    Examples:
        - Low values fade to transparent, high values are opaque:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.colors import alpha_scaled_image, HAZE_COLORMAPS
            >>> fig, ax = plt.subplots()
            >>> data = np.array([[0.0, 1.0], [0.5, 1.0]])
            >>> img = alpha_scaled_image(ax, data, HAZE_COLORMAPS["dust"])
            >>> rgba = img.get_array()
            >>> rgba[0, 0, 3]  # value 0.0 -> fully transparent
            np.float64(0.0)
            >>> rgba[0, 1, 3]  # value 1.0 -> fully opaque
            np.float64(1.0)
            >>> plt.close(fig)

            ```
        - NaN pixels are always transparent, independent of `alpha_norm`:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.colors import alpha_scaled_image
            >>> fig, ax = plt.subplots()
            >>> data = np.array([[np.nan, 1.0]])
            >>> img = alpha_scaled_image(ax, data, "viridis")
            >>> img.get_array()[0, 0, 3]
            np.float64(0.0)
            >>> plt.close(fig)

            ```

    See Also:
        HAZE_COLORMAPS: Ready-made colormaps designed for this function.
        swatch_legend: A matching two-stop legend for the same data.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-dimensional, got shape {data.shape}")

    rgba = _alpha_rgba(data, cmap, norm, alpha_norm, constant_alpha)
    return ax.imshow(rgba, **imshow_kwargs)


def _alpha_rgba(
    data: np.ndarray,
    cmap: str | Colormap,
    norm: mcolors.Normalize | None,
    alpha_norm: mcolors.Normalize | None,
    constant_alpha: float | None = None,
) -> np.ndarray:
    """Shared colour+alpha computation behind `alpha_scaled_image`/`_mesh`.

    Args:
        data: 2D array of values, already validated by the caller.
        cmap: Colormap name or object.
        norm: Normalization for colour, or `None` to default to the finite
            range of `data`.
        alpha_norm: Normalization for opacity, or `None` to reuse `norm`.
        constant_alpha: If given, every finite cell is drawn at this fixed
            opacity (clipped to `[0, 1]`) and `alpha_norm` is ignored -- for a
            plain opaque field (`1.0`) or a uniform semi-transparent overlay.
            Non-finite cells stay fully transparent either way.

    Returns:
        np.ndarray: An `(*, *, 4)` RGBA array; non-finite `data` cells are
        fully transparent.
    """
    cmap_obj = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    if norm is None:
        finite = data[np.isfinite(data)]
        vmin = float(finite.min()) if finite.size else 0.0
        vmax = float(finite.max()) if finite.size else 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    alpha_norm = norm if alpha_norm is None else alpha_norm

    rgba = cmap_obj(norm(data))
    if constant_alpha is not None:
        alpha = np.full(data.shape, float(np.clip(constant_alpha, 0.0, 1.0)))
    else:
        alpha = np.clip(np.asarray(alpha_norm(data), dtype=float), 0.0, 1.0)
    finite_mask = np.isfinite(data)
    rgba[..., 3] = np.where(finite_mask, alpha, 0.0)
    return rgba


def alpha_scaled_mesh(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    cmap: str | Colormap,
    *,
    norm: mcolors.Normalize | None = None,
    alpha_norm: mcolors.Normalize | None = None,
    constant_alpha: float | None = None,
    **pcolormesh_kwargs: Any,
) -> Any:
    """Draw `data` on a curvilinear `(x, y)` mesh with per-cell opacity.

    The `pcolormesh` counterpart to `alpha_scaled_image`. Use this instead of
    `alpha_scaled_image` whenever the grid is not a plain rectangle in axes
    coordinates -- e.g. data reprojected onto an orthographic globe by
    `cleopatra.projection.orthographic_grid`, or any other curvilinear
    `(x, y)` grid. Builds the same value-modulated-alpha RGBA colouring as
    `alpha_scaled_image`, then paints it onto the mesh via `set_facecolor`:
    `pcolormesh`'s own `cmap`/`norm`/`alpha` machinery is bypassed because its
    `alpha` argument is a single scalar and cannot vary per cell.

    Args:
        ax: Axes to draw on.
        x: 2D array of cell x-coordinates, in `Axes.pcolormesh`'s `(X, Y, C)`
            convention (either one larger than `data` per axis for exact
            cell edges, or the same shape with `shading="auto"`/`"nearest"`).
        y: 2D array of cell y-coordinates, same convention as `x`.
        data: 2D array of values, one per mesh cell.
        cmap: Colormap name or object, e.g. `HAZE_COLORMAPS["dust"]`.
        norm: Normalization mapping `data` to colour. Defaults to
            `Normalize(vmin, vmax)` over the finite range of `data`.
        alpha_norm: Normalization mapping `data` to opacity. Defaults to
            `norm`.
        constant_alpha: If given, paint every finite cell at this fixed opacity
            (clipped to `[0, 1]`) and ignore `alpha_norm` -- e.g. `1.0` for a
            plain opaque field. Non-finite (NaN) cells stay transparent.
        **pcolormesh_kwargs: Forwarded to `ax.pcolormesh`. `shading` defaults
            to `"auto"` if not given.

    Returns:
        QuadMesh: The mesh artist added to `ax`.

    Raises:
        ValueError: If `data` is not 2-dimensional.

    Examples:
        - A 2x2 curvilinear mesh with opacity fading toward zero:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.colors import alpha_scaled_mesh
            >>> fig, ax = plt.subplots()
            >>> x, y = np.meshgrid(np.arange(3), np.arange(3))
            >>> data = np.array([[0.0, 1.0], [0.5, 1.0]])
            >>> mesh = alpha_scaled_mesh(ax, x, y, data, "viridis", shading="flat")
            >>> alpha = mesh.get_facecolor()[:, 3]
            >>> alpha[0]  # first cell, value 0.0 -> transparent
            np.float64(0.0)
            >>> alpha[1]  # second cell, value 1.0 -> opaque
            np.float64(1.0)
            >>> plt.close(fig)

            ```

    See Also:
        alpha_scaled_image: The regular-grid counterpart (uses `imshow`).
        cleopatra.projection.orthographic_grid: Produces the `(x, y, data)`
            triple this function is designed to render.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-dimensional, got shape {data.shape}")

    pcolormesh_kwargs.setdefault("shading", "auto")
    rgba = _alpha_rgba(data, cmap, norm, alpha_norm, constant_alpha)
    mesh = ax.pcolormesh(x, y, data, **pcolormesh_kwargs)
    mesh.set_array(None)
    mesh.set_facecolor(rgba.reshape(-1, 4))
    return mesh


#: Named "data style" presets for `apply_data_style` -- each entry maps a
#: layer name to the config used to draw it with `alpha_scaled_image` /
#: `alpha_scaled_mesh` and label it with `swatch_legend`. Per-layer keys:
#:
#: - ``cmap`` (required): a `Colormap` object or a matplotlib colormap name.
#: - ``label`` (required): the legend caption.
#: - ``vmin`` / ``vmax`` (optional): colour range; **omit to auto-range** from
#:   each field's finite values -- the right default for real GIS/climate data
#:   whose absolute range varies (temperature in K vs C, elevation in m, ...).
#: - ``center`` (optional): render as a **diverging** map symmetric around this
#:   value (colormap midpoint lands on it) -- for anomaly fields, usually ``0``.
#: - ``norm`` (optional): ``"linear"`` (default), ``"log"`` (`LogNorm`), or
#:   ``"symlog"`` (`SymLogNorm`, linear within ``+/-linthresh`` so ``0`` maps
#:   cleanly -- the robust choice for heavily-skewed, zero-containing fields
#:   like flow accumulation). ``linthresh`` (optional) sets the symlog threshold.
#: - Opacity policy (choose at most one): omit all alpha keys for the default
#:   value-linked opacity (transparent where the value is low -- the overlay
#:   look); set ``alpha`` to a constant (e.g. ``1.0``) for a plain opaque
#:   field; or set ``alpha_vmin``/``alpha_vmax`` to decouple opacity from
#:   colour (the "haze" glowing rim).
#: - ``categories`` (optional): a **categorical** preset instead of a colormap.
#:   A list of ``(class_value, colour, label)`` triples for discrete integer
#:   class codes (e.g. flood status); the layer is drawn opaque with a
#:   `ListedColormap`/`BoundaryNorm` and gets a discrete (disjoint) legend
#:   rather than a gradient swatch. ``cmap``/``vmin``/``vmax``/``center`` are
#:   ignored when ``categories`` is set; only ``label`` (the legend title) is used.
#:
#: This is the colour/legend half of the ECMWF/CAMS look; pair it with a
#: `cleopatra.projection` projection-style preset (globe or flat) -- the two
#: are independent and neither requires the other.
#:
#: `"haze"`'s layers also set `alpha_vmin`/`alpha_vmax`, decoupling opacity
#: from colour: opacity saturates over a much narrower band (0.1-0.5) than
#: colour (0.0-1.0), so the vivid mid-colormap tones are fully opaque well
#: before the data reaches its maximum. This reproduces the bright, glowing
#: "flame" rim ECMWF/CAMS aerosol maps show at a plume's edge -- with a
#: single shared curve, that rim's colour is barely visible because it sits
#: at low, nearly-transparent opacity.
#:
#: `"cams_aod"` is the plainer, official counterpart: a single `"aod"` layer
#: drawn with the canonical `CAMS_AOD_COLORMAPS["blue_yellow_red"]` scale. It
#: sets no `alpha_vmin`/`alpha_vmax`, so opacity tracks the colour norm
#: linearly -- transparent where AOD is ~0, opaque red where it is high --
#: the natural behaviour for overlaying a single aerosol-optical-depth field
#: on a basemap (the common `pyramids` raster/NetCDF case).
DATA_STYLES: dict[str, dict[str, dict[str, Any]]] = {
    "haze": {
        "organic_matter": {
            "cmap": HAZE_COLORMAPS["organic_matter"],
            "label": "Organic Matter",
            "vmin": 0.0,
            "vmax": 1.0,
            "alpha_vmin": 0.1,
            "alpha_vmax": 0.5,
        },
        "dust": {
            "cmap": HAZE_COLORMAPS["dust"],
            "label": "Dust",
            "vmin": 0.0,
            "vmax": 1.0,
            "alpha_vmin": 0.1,
            "alpha_vmax": 0.5,
        },
    },
    "cams_aod": {
        "aod": {
            "cmap": CAMS_AOD_COLORMAPS["blue_yellow_red"],
            "label": "Aerosol Optical Depth",
            "vmin": 0.0,
            "vmax": 1.0,
        },
    },
    # --- Ready-to-use presets for common pyramids GIS/NetCDF-climate fields. ---
    # Opaque full fields (auto-ranged from the data): the whole field is drawn.
    "temperature": {
        "temperature": {
            "cmap": "RdYlBu_r",  # blue (cold) -> red (hot)
            "label": "Temperature",
            "alpha": 1.0,
        },
    },
    "elevation": {
        "elevation": {
            "cmap": "terrain",
            "label": "Elevation",
            "alpha": 1.0,
        },
    },
    "vegetation": {
        "vegetation": {
            "cmap": "YlGn",  # sparse (pale) -> dense (green)
            "label": "Vegetation (NDVI)",
            "alpha": 1.0,
        },
    },
    "wind_speed": {
        "wind_speed": {
            "cmap": "viridis",  # perceptually uniform
            "label": "Wind speed",
            "alpha": 1.0,
        },
    },
    # Diverging anomaly field, symmetric around zero (0 -> white).
    "anomaly": {
        "anomaly": {
            "cmap": "RdBu_r",  # negative (blue) -> 0 (white) -> positive (red)
            "label": "Anomaly",
            "center": 0.0,
            "alpha": 1.0,
        },
    },
    # Overlay field: value-linked opacity, so it is transparent where dry and
    # opaque where it rains -- ideal for compositing over a basemap.
    "precipitation": {
        "precipitation": {
            "cmap": "YlGnBu",  # light (light rain) -> dark blue (heavy)
            "label": "Precipitation",
        },
    },
    # Categorical preset: discrete integer class codes -> fixed colours + a
    # discrete (disjoint) legend, instead of a continuous colormap. The
    # near-universal NWS/USGS 5-class river-flood status scale (0..4).
    "flood_status": {
        "flood_status": {
            "categories": [
                (0, "#2c7fb8", "Normal"),
                (1, "#31a354", "Action"),
                (2, "#ffeb3b", "Minor"),
                (3, "#ff7f00", "Moderate"),
                (4, "#e31a1c", "Major"),
            ],
            "label": "Flood status",
        },
    },
    # Hydrology rasters derived from a DEM.
    # D8 flow direction: the 8 ESRI direction codes (powers of two) are discrete
    # classes, coloured cyclically (twilight) so adjacent compass directions are
    # similar and opposites distinct. (D-infinity flow *angle* is continuous and
    # cyclic -- use the "phase" preset for that.)
    "flow_direction_d8": {
        "flow_direction_d8": {
            "categories": [
                (1, "#e2d9e2", "E"),
                (2, "#95b5c7", "SE"),
                (4, "#6276ba", "S"),
                (8, "#592a8f", "SW"),
                (16, "#2f1436", "W"),
                (32, "#741e4f", "NW"),
                (64, "#b25652", "N"),
                (128, "#cca389", "NE"),
            ],
            "label": "Flow direction (D8)",
        },
    },
    # Flow accumulation is extremely skewed (most cells ~0, channels huge), so a
    # symmetric-log norm is used; opacity tracks it, so low cells fade and the
    # channel network stands out. Composes over a hillshaded DEM.
    "flow_accumulation": {
        "flow_accumulation": {
            "cmap": "Blues",
            "label": "Flow accumulation",
            "norm": "symlog",
        },
    },
}


def _load_preset_asset(
    resource: str, cmap_prefix: str
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build `DATA_STYLES` entries from a vendored preset asset under `cleopatra.data`.

    Shared by the ECMWF/Magics and cmocean preset libraries. Each asset maps a
    preset key to a `palette` (hex control points), a `label`, an `opacity`
    policy (`"opaque"` -> a plain field via constant alpha; otherwise a
    value-linked overlay), and an optional diverging `center`. Every preset is a
    single layer keyed by its own name and carries no `vmin`/`vmax`, so it
    auto-ranges.

    Args:
        resource: The asset filename inside the `cleopatra.data` package.
        cmap_prefix: A prefix for the generated colormap names (e.g. `"magics"`).

    Returns:
        dict: `DATA_STYLES`-shaped presets, or an empty mapping if the asset is
        unavailable. Never raises, so a partial install degrades to the
        hand-authored presets rather than breaking `import cleopatra`.
    """
    # Outer guard: a missing, unreadable, or malformed-JSON asset (or a
    # non-mapping structure) degrades to the hand-authored presets rather than
    # breaking `import cleopatra`.
    try:
        source = (
            importlib.resources.files("cleopatra.data")
            .joinpath(resource)
            .read_text(encoding="utf-8")
        )
        records = json.loads(source).get("presets", {}).items()
    except (
        FileNotFoundError, ModuleNotFoundError, OSError,
        json.JSONDecodeError, AttributeError,
    ):
        return {}

    # Inner guard: a single structurally-broken record (missing palette/label,
    # an unparseable colour, a <2-colour palette) is skipped, keeping every
    # other well-formed preset in the asset.
    presets: dict[str, dict[str, dict[str, Any]]] = {}
    for key, rec in records:
        try:
            layer: dict[str, Any] = {
                "cmap": LinearSegmentedColormap.from_list(
                    f"{cmap_prefix}_{key}", rec["palette"]
                ),
                "label": rec["label"],
            }
            if rec.get("opacity") == "opaque":
                layer["alpha"] = 1.0  # value-linked opacity (overlay) is the default otherwise
            if rec.get("center") is not None:
                layer["center"] = rec["center"]
            presets[key] = {key: layer}
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return presets


def _load_magics_presets() -> dict[str, dict[str, dict[str, Any]]]:
    """Load the ECMWF/Magics parameter-preset library (Apache-2.0).

    Colour ramps and parameter labels derived from ecmwf/magics (see
    `MAGICS_NOTICE.txt`), keyed by GRIB shortName. Thin wrapper over
    `_load_preset_asset`.
    """
    return _load_preset_asset("magics_presets.json", "magics")


#: Register the vendored preset libraries into `DATA_STYLES` at import, alongside
#: the hand-authored presets above: the full ECMWF/Magics parameter set (keyed
#: by GRIB shortName, e.g. `"2t"`, `"tp"`, `"aod550"`) and the cmocean
#: ocean/hydrology/DEM set (keyed by variable, e.g. `"salinity"`,
#: `"bathymetry"`). List them all with `sorted(DATA_STYLES)`.
DATA_STYLES.update(_load_magics_presets())
DATA_STYLES.update(_load_preset_asset("cmocean_presets.json", "cmocean"))


def _category_boundaries(values: list[float]) -> list[float]:
    """Bin edges for a `BoundaryNorm` over discrete category values.

    Interior edges are the midpoints between consecutive (sorted) class
    values; the two outer edges extend by the same half-gap, so each value
    lands in the middle of its own bin (for integer class codes this is the
    usual ``+/-0.5``).

    Args:
        values: The category class values (need not be pre-sorted).

    Returns:
        list[float]: ``len(values) + 1`` ascending bin edges.
    """
    vals = sorted(values)
    if len(vals) == 1:
        return [vals[0] - 0.5, vals[0] + 0.5]
    mids = [(vals[i] + vals[i + 1]) / 2.0 for i in range(len(vals) - 1)]
    lower = vals[0] - (mids[0] - vals[0])
    upper = vals[-1] + (vals[-1] - mids[-1])
    return [lower] + mids + [upper]


def _resolve_style_norm(
    data: np.ndarray, cfg: dict[str, Any]
) -> tuple[mcolors.Normalize, float, float]:
    """Resolve the colour `Normalize` (and its concrete bounds) for one layer.

    Honours a `DATA_STYLES` layer's optional `vmin`/`vmax` (auto-ranged from
    the data's finite values when omitted -- essential for real GIS/climate
    fields whose absolute range varies) and an optional diverging `center`.
    When `center` is set and a bound is missing, the range is made symmetric
    around it (`center +/- max|data - center|`), so the colormap's midpoint
    lands exactly on `center` -- the anomaly-map convention.

    Args:
        data: The layer's 2D data array (finite values drive auto-ranging).
        cfg: The layer's `DATA_STYLES` config dict.

    Returns:
        tuple: `(norm, vmin, vmax)` -- the colour normalization and the
        concrete bounds it resolved to (reused for the layer's legend).
    """
    vmin = cfg.get("vmin")
    vmax = cfg.get("vmax")
    center = cfg.get("center")
    finite = data[np.isfinite(data)]
    if center is not None and (vmin is None or vmax is None):
        if finite.size:
            radius = max(
                abs(float(finite.min()) - center),
                abs(float(finite.max()) - center),
            )
        else:
            radius = 1.0
        radius = radius or 1.0
        vmin = center - radius if vmin is None else vmin
        vmax = center + radius if vmax is None else vmax
    else:
        if vmin is None:
            vmin = float(finite.min()) if finite.size else 0.0
        if vmax is None:
            vmax = float(finite.max()) if finite.size else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0

    norm_kind = cfg.get("norm")
    if norm_kind in (None, "linear") and center is not None:
        # Diverging: put `center` on the colormap midpoint regardless of how
        # the bounds were resolved (auto-symmetric or explicit vmin/vmax).
        if not (vmin < center < vmax):
            raise ValueError(
                f"diverging 'center' ({center}) must lie strictly between "
                f"vmin ({vmin}) and vmax ({vmax})"
            )
        norm: mcolors.Normalize = mcolors.TwoSlopeNorm(
            vcenter=center, vmin=vmin, vmax=vmax
        )
    elif norm_kind in (None, "linear"):
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    elif norm_kind == "log":
        # LogNorm needs a strictly positive range. Derive the lower bound from
        # an explicit positive vmin, else the smallest positive finite value;
        # the upper bound must be positive too. Data with no positive value (or
        # an inverted explicit range) has no valid log window -- fail clearly
        # here instead of letting matplotlib raise an opaque "vmin must be less
        # or equal to vmax" deep inside the draw.
        positive = finite[finite > 0] if finite.size else finite
        lo = vmin if (vmin is not None and vmin > 0) else (
            float(positive.min()) if positive.size else None
        )
        # `lo >= vmax` (not just `>`) so a single-positive-value range like
        # data [0, 5] -- where the only positive value is both the lower and
        # upper bound -- fails clearly rather than building a degenerate
        # LogNorm(vmin==vmax) that renders the whole layer flat at one colour.
        if lo is None or vmax is None or vmax <= 0 or lo >= vmax:
            raise ValueError(
                "data style norm='log' needs positive data with a positive "
                f"value range (resolved vmin={lo!r}, vmax={vmax!r}); use "
                "norm='symlog' for data that spans zero or negative values."
            )
        # Report the clamped positive lower bound so the legend matches the
        # colours the LogNorm actually starts from (not a 0/negative vmin).
        vmin = lo
        norm = mcolors.LogNorm(vmin=lo, vmax=vmax)
    elif norm_kind == "symlog":
        # Symmetric log: linear within +/- linthresh (so 0 maps cleanly) and
        # logarithmic beyond -- the robust choice for skewed, zero-containing
        # fields such as flow accumulation.
        norm = mcolors.SymLogNorm(
            linthresh=float(cfg.get("linthresh", 1.0)), vmin=vmin, vmax=vmax
        )
    else:
        raise ValueError(
            f"data style 'norm' must be 'linear', 'log', or 'symlog', got {norm_kind!r}"
        )
    return norm, vmin, vmax


def apply_data_style(
    ax: Axes,
    layers: dict[str, np.ndarray],
    style: str = "haze",
    *,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    legend: bool = True,
    legend_bounds: list[tuple[float, float, float, float]] | None = None,
    **render_kwargs: Any,
) -> dict[str, Any]:
    """Draw one or more named data layers with a registered `DATA_STYLES` preset.

    Applies `alpha_scaled_image` (and, if `legend`, a stacked `swatch_legend`
    per layer) to each array in `layers`, using the colormap/label/range that
    `style` defines for that layer name in `DATA_STYLES`. Calling this with
    `layers={"organic_matter": ..., "dust": ...}` reproduces the ECMWF/CAMS
    aerosol look in one call -- but it is only a thin orchestration over
    `alpha_scaled_image` + `swatch_legend`, so nothing about it requires the
    orthographic globe: it works on a plain flat axes, an existing
    `"ecmwf"`/`"ecmwf-dark"` reference map (`cleopatra.geo`), or any other
    projection just as well. Pass `x`/`y` (e.g. from
    `cleopatra.projection.orthographic_grid`) to render on a curvilinear grid
    via `alpha_scaled_mesh` instead of the default `imshow`-based
    `alpha_scaled_image`.

    Args:
        ax: Axes to draw on.
        layers: Mapping of layer name to its 2D data array. Every key must be
            a layer defined by `style` (e.g. `"organic_matter"`/`"dust"` for
            `"haze"`); pass a subset to draw only some of a style's layers.
            For a **categorical** preset (one that defines `categories`, e.g.
            `"flow_direction_d8"`), the array is matched to the declared class
            codes by exact float equality, so it must be integer-coded (D8
            powers of two, flood classes 0..4 — all exactly representable in
            float). Any cell that is not bit-exactly a declared code (nodata,
            sinks, or a value perturbed by a lossy float transform) is treated
            as out-of-range and rendered transparent.
        style: A name from `DATA_STYLES`. Defaults to `"haze"`.
        x: Optional 2D curvilinear x-coordinates (see `alpha_scaled_mesh`).
            When given (together with `y`), every layer is drawn with
            `alpha_scaled_mesh` instead of `alpha_scaled_image`.
        y: Optional 2D curvilinear y-coordinates, paired with `x`.
        legend: If `True` (default), attach one `swatch_legend` per layer,
            stacked top-to-bottom in the top-left.
        legend_bounds: Explicit `(x0, y0, width, height)` per layer legend,
            in the same order as `layers`, overriding the auto-stacked
            default.
        **render_kwargs: Forwarded to every `alpha_scaled_image` (or
            `alpha_scaled_mesh`, when `x`/`y` are given) call.

    Returns:
        dict[str, Any]: The image (or mesh) artist for each layer, keyed by
        name, in the same order as `layers`.

    Raises:
        KeyError: If `style` is not registered, or `layers` names a layer the
            style does not define.
        ValueError: If exactly one of `x`/`y` is given (they must be given
            together, or both omitted).

    Examples:
        - Draw both haze layers and read back the images and their labels:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.colors import apply_data_style
            >>> fig, ax = plt.subplots()
            >>> layers = {
            ...     "dust": np.array([[0.0, 1.0]]),
            ...     "organic_matter": np.array([[0.2, 0.8]]),
            ... }
            >>> images = apply_data_style(ax, layers)
            >>> sorted(images)
            ['dust', 'organic_matter']
            >>> [t.get_text() for c in ax.child_axes for t in c.texts][:2]
            ['Dust', '0']
            >>> plt.close(fig)

            ```
        - Passing `x`/`y` renders on a curvilinear mesh instead of `imshow`:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.collections import QuadMesh
            >>> from cleopatra.colors import apply_data_style
            >>> fig, ax = plt.subplots()
            >>> x, y = np.meshgrid(np.arange(3), np.arange(3))
            >>> images = apply_data_style(
            ...     ax, {"dust": np.array([[0.0, 1.0], [0.5, 1.0]])},
            ...     x=x, y=y, shading="flat",
            ... )
            >>> isinstance(images["dust"], QuadMesh)
            True
            >>> plt.close(fig)

            ```
        - An unknown layer name raises `KeyError` before drawing anything:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.colors import apply_data_style
            >>> fig, ax = plt.subplots()
            >>> apply_data_style(ax, {"smoke": np.array([[0.0, 1.0]])})
            Traceback (most recent call last):
                ...
            KeyError: "['smoke'] not defined for data style 'haze'; available layers: ['dust', 'organic_matter']"
            >>> plt.close(fig)

            ```

    See Also:
        alpha_scaled_image: The regular-grid rendering primitive this composes.
        alpha_scaled_mesh: The curvilinear-grid rendering primitive this
            composes when `x`/`y` are given.
        swatch_legend: The per-layer legend primitive this composes.
        cleopatra.projection.apply_projection_style: The companion
            projection-style axis (globe vs flat).
    """
    if style not in DATA_STYLES:
        raise KeyError(
            f"Unknown data style {style!r}; available: {sorted(DATA_STYLES)}"
        )
    preset = DATA_STYLES[style]
    unknown = sorted(set(layers) - set(preset))
    if unknown:
        raise KeyError(
            f"{unknown} not defined for data style {style!r}; "
            f"available layers: {sorted(preset)}"
        )
    if (x is None) != (y is None):
        raise ValueError(
            "x and y must be given together (or both omitted); got "
            f"x={'given' if x is not None else None}, "
            f"y={'given' if y is not None else None}"
        )

    curvilinear = x is not None and y is not None
    if curvilinear:
        # cleopatra.projection.apply_projection_style always returns cell
        # EDGE coordinates (one larger per axis than data): matplotlib's
        # automatic centre-to-edge inference ("auto"/"nearest") is unreliable
        # for a globe's extreme local distortion, so shading="flat" (which
        # trusts the given edges exactly) is the correct default here.
        render_kwargs.setdefault("shading", "flat")
    images: dict[str, Any] = {}
    for i, (name, data) in enumerate(layers.items()):
        cfg = preset[name]
        data = np.asarray(data, dtype=float)

        categories = cfg.get("categories")
        if categories is not None:
            cats = sorted(categories, key=lambda c: c[0])
            cat_values = [float(c[0]) for c in cats]
            cat_colors = [c[1] for c in cats]
            cat_labels = [c[2] for c in cats]
            cat_cmap = mcolors.ListedColormap(cat_colors)
            cat_norm = mcolors.BoundaryNorm(
                _category_boundaries(cat_values), len(cat_colors)
            )
            # Only cells whose value is one of the declared class codes are
            # drawn; anything else (nodata sentinels, D8 sinks, out-of-range
            # codes) is masked to NaN so it renders transparent instead of
            # being clamped to an end category at full opacity. Matching is by
            # exact float equality, so categorical presets expect integer-coded
            # input (D8 powers of two, flood classes 0..4 -- all exactly
            # representable); a value that is not bit-exactly a declared code
            # (e.g. one perturbed by a lossy float transform) is treated as
            # out-of-range and silently rendered transparent.
            cat_data = np.where(np.isin(data, cat_values), data, np.nan)
            if curvilinear:
                images[name] = alpha_scaled_mesh(
                    ax, x, y, cat_data, cat_cmap, norm=cat_norm, constant_alpha=1.0,
                    **render_kwargs,
                )
            else:
                images[name] = alpha_scaled_image(
                    ax, cat_data, cat_cmap, norm=cat_norm, constant_alpha=1.0,
                    **render_kwargs,
                )
            if legend:
                # Honour legend_bounds' (x0, y0) as an anchor when given;
                # otherwise default to the top-right. Re-add any earlier
                # categorical legend so a second categorical layer stacks
                # instead of replacing it (matplotlib keeps one legend/axes).
                prior_legend = ax.get_legend()
                if legend_bounds is not None:
                    x0, y0 = legend_bounds[i][0], legend_bounds[i][1]
                    leg = disjoint_legend(
                        ax, cat_colors, cat_labels, title=cfg["label"],
                        loc="upper left", bbox_to_anchor=(x0, y0),
                    )
                else:
                    leg = disjoint_legend(
                        ax, cat_colors, cat_labels, title=cfg["label"], loc="upper right"
                    )
                if prior_legend is not None and prior_legend is not leg:
                    ax.add_artist(prior_legend)
            continue

        norm, resolved_vmin, resolved_vmax = _resolve_style_norm(data, cfg)

        alpha_const = cfg.get("alpha")
        alpha_vmin = cfg.get("alpha_vmin")
        alpha_vmax = cfg.get("alpha_vmax")
        if alpha_const is not None and (alpha_vmin is not None or alpha_vmax is not None):
            raise ValueError(
                f"data style layer {name!r} sets both a constant 'alpha' and "
                "'alpha_vmin'/'alpha_vmax'; those are mutually exclusive"
            )
        alpha_norm = (
            mcolors.Normalize(vmin=alpha_vmin, vmax=alpha_vmax)
            if alpha_vmin is not None or alpha_vmax is not None
            else None
        )
        if curvilinear:
            images[name] = alpha_scaled_mesh(
                ax, x, y, data, cfg["cmap"], norm=norm, alpha_norm=alpha_norm,
                constant_alpha=alpha_const, **render_kwargs,
            )
        else:
            images[name] = alpha_scaled_image(
                ax, data, cfg["cmap"], norm=norm, alpha_norm=alpha_norm,
                constant_alpha=alpha_const, **render_kwargs,
            )
        if legend:
            bounds = (
                legend_bounds[i]
                if legend_bounds is not None
                else (0.02, 0.92 - 0.12 * i, 0.32, 0.06)
            )
            swatch_legend(
                ax,
                cfg["cmap"],
                cfg["label"],
                vmin=resolved_vmin,
                vmax=resolved_vmax,
                bounds=bounds,
                norm=norm,
            )
    return images


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
    def create_from_image(cls, path: Union[str, os.PathLike]) -> "Colors":
        """Create a color object from an image.

        if you have an image of a color ramp, and you want to extract the colors from it, you can use this method.

        ![color-ramp](./../images/colors/color-ramp.png)

        Args:
            path: The path to the image file, as a `str` or `os.PathLike`
                (e.g. a `pathlib.Path`).

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
        path = os.fspath(path)
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
