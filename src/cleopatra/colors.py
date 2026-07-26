import importlib.resources
import json
import os
import warnings
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.image import AxesImage
from PIL import Image, UnidentifiedImageError

from cleopatra.palettes import CAMS_AOD_COLORMAPS, FLAME_COLORMAPS, HAZE_COLORMAPS
from cleopatra.styles import disjoint_legend, swatch_extend_prefixes, swatch_legend

#: The haze / CAMS-AOD / flame colour families now live in `cleopatra.palettes`
#: (built there via perceptual CIELAB interpolation and registered in the unified
#: palette registry). They are re-exported here (and used by `DATA_STYLES` below)
#: so existing `from cleopatra.colors import HAZE_COLORMAPS` imports keep working.


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

    rgba = alpha_rgba(data, cmap, norm, alpha_norm, constant_alpha)
    return ax.imshow(rgba, **imshow_kwargs)


def alpha_rgba(
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
    return np.asarray(rgba)


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
    rgba = alpha_rgba(data, cmap, norm, alpha_norm, constant_alpha)
    mesh = ax.pcolormesh(x, y, data, **pcolormesh_kwargs)
    mesh.set_array(None)
    # matplotlib's Collection.set_facecolor accepts an (N, 4) RGBA array at
    # runtime; its stub only spells out scalar/sequence-of-scalar color forms.
    mesh.set_facecolor(rgba.reshape(-1, 4))  # type: ignore[arg-type]
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
    # A general-purpose temperature ramp: the muted `Spectral_r` colours (blue
    # cold -> red hot), auto-ranged from the data so it fits ANY continuous field
    # (a Celsius raster, a KDE density, a normalized index). Pass `vmin`/`vmax`
    # to pin the scale to a chosen range (e.g. `vmin=-40, vmax=40` for the ECMWF
    # window); for the fixed, discretely-banded ECMWF 2 m-temperature look in one
    # word, use the `"temperature_2m"` preset instead (fixed -40..40 degC,
    # `extend="both"`).
    "temperature": {
        "temperature": {
            "cmap": "Spectral_r",  # muted spectral, blue (cold) -> red (hot)
            "label": "Temperature",
            "alpha": 1.0,  # opaque full field (like elevation/wind_speed), not a glow
        },
    },
    # Temperature (or any heat field) rendered as a glowing flame/plume: the CAMS
    # aerosol technique (value-linked opacity -- cool fades to transparent so the
    # terrain shows, hot glows opaque) recoloured for heat. Compose over a dark
    # hillshaded backdrop (`apply_blank_canvas` + a `cleopatra.reference` relief),
    # the way the "haze" style is composed. Colour spans 0..40, opacity ramps in
    # over 6..32 -- sensible for surface air temperature in degC; override
    # `vmin`/`vmax` for other ranges. Two flavours: `white_hot` (blows out to
    # yellow-white) and `amber` (warmer gold/orange, less blown-out).
    "temperature_flame": {
        "temperature_flame": {
            "cmap": FLAME_COLORMAPS["white_hot"],
            "label": "Temperature",
            "vmin": 0.0,
            "vmax": 40.0,
            "alpha_vmin": 6.0,
            "alpha_vmax": 32.0,
        },
    },
    "temperature_flame_amber": {
        "temperature_flame_amber": {
            "cmap": FLAME_COLORMAPS["amber"],
            "label": "Temperature",
            "vmin": 0.0,
            "vmax": 40.0,
            "alpha_vmin": 6.0,
            "alpha_vmax": 32.0,
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
    """Build `DATA_STYLES` entries from a vendored continuous-colormap preset asset.

    Used for the cmocean ocean/hydrology/DEM preset library. Each asset maps a
    preset key to a `palette` (hex control points sampled from a continuous
    colormap), a `label`, an `opacity` policy (`"opaque"` -> a plain field via
    constant alpha; otherwise a value-linked overlay), and an optional diverging
    `center`. Every preset is a single layer keyed by its own name and carries no
    `vmin`/`vmax`, so it auto-ranges.

    Args:
        resource: The asset filename inside the `cleopatra.data` package.
        cmap_prefix: A prefix for the generated colormap names (e.g. `"cmocean"`).

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
        records = json.loads(source).items()
    except (
        ModuleNotFoundError,
        OSError,
        json.JSONDecodeError,
        AttributeError,
    ):
        return {}

    # Inner guard: a single structurally-broken record (missing palette/label,
    # an unparseable colour, a <2-colour palette) is skipped, keeping every
    # other well-formed preset in the asset.
    presets: dict[str, dict[str, dict[str, Any]]] = {}
    for key, rec in records:
        try:
            palette = rec["palette"]
            cmap: Colormap = LinearSegmentedColormap.from_list(
                f"{cmap_prefix}_{key}", palette
            )
            layer: dict[str, Any] = {"cmap": cmap, "label": rec["label"]}
            if rec.get("opacity") == "opaque":
                layer["alpha"] = (
                    1.0  # value-linked opacity (overlay) is the default otherwise
                )
            if rec.get("center") is not None:
                layer["center"] = rec["center"]
            presets[key] = {key: layer}
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return presets


def _load_weather_presets() -> dict[str, dict[str, dict[str, Any]]]:
    """Load the merged ECMWF weather preset library (Apache-2.0), keyed by a descriptive name.

    Merged from two sources at build time (see `tools/build_weather_presets.py`)
    into one record per GRIB shortName, then renamed to its descriptive key --
    each record is one of three shapes:

    - **Equal-width banded** (vendored from Magics): a discrete `colors` list
      plus a `bands` count (`len(colors)`), rendered as a `ListedColormap` with
      `bands` equal-width intervals. A `vmin`/`vmax` is also present when the
      parameter's original Magics style name encoded a fixed range; otherwise
      the bands auto-range to the data.
    - **Explicit contour levels** (vendored from earthkit-plots' curated ECMWF
      defaults, which supersede the Magics record for the same shortName): a
      `colors` list or matplotlib colormap name, plus explicit `levels` and an
      `extend` cap, rendered as a `BoundaryNorm` at those exact boundaries.
    - **Continuous** (colour list with neither `bands` nor `levels`, e.g.
      `total_precipitation`'s rain gradient): a genuine `LinearSegmentedColormap`.

    Never raises: a missing/malformed asset degrades to `{}`.
    """
    try:
        raw = (
            importlib.resources.files("cleopatra.data")
            .joinpath("weather_presets.json")
            .read_text(encoding="utf-8")
        )
    except (ModuleNotFoundError, OSError):
        return {}
    try:
        records = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(records, dict):
        return {}

    presets: dict[str, dict[str, dict[str, Any]]] = {}
    for key, rec in records.items():
        try:
            colors = rec["colors"]
            levels = rec.get("levels")
            bands = rec.get("bands")
            if levels:
                # A colour LIST with explicit levels is a discrete band palette:
                # keep the exact ECMWF colours via a ListedColormap. A colormap
                # NAME (str) is resolved at draw time instead.
                cmap: Any = (
                    mcolors.ListedColormap(colors, name=f"weather_{key}")
                    if isinstance(colors, list)
                    else colors
                )
            elif bands:
                # A Magics preset renders as flat colour bands, not a smooth
                # ramp -- a continuous interpolation of these saturated colours
                # reads as a glossy, over-exposed sheen.
                cmap = mcolors.ListedColormap(colors, name=f"weather_{key}")
            elif isinstance(colors, str):
                cmap = colors
            else:
                # A colour list with neither levels nor bands (e.g.
                # `total_precipitation`'s white->blue gradient) is a genuine
                # continuous ramp.
                cmap = LinearSegmentedColormap.from_list(f"weather_{key}", colors)
            layer: dict[str, Any] = {"cmap": cmap, "label": rec["label"]}
            if rec.get("opacity") == "opaque":
                layer["alpha"] = 1.0
            if levels:
                layer["levels"] = levels
                layer["extend"] = rec.get("extend", "neither")
            elif bands:
                layer["bands"] = bands
                if "vmin" in rec:
                    layer["vmin"], layer["vmax"] = rec["vmin"], rec["vmax"]
            presets[key] = {key: layer}
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return presets


#: Register the vendored preset libraries into `DATA_STYLES` at import, alongside
#: the hand-authored presets above: the merged ECMWF weather parameter set
#: (keyed by a descriptive parameter name, e.g. `"temperature_2m"`,
#: `"total_precipitation"`, `"aerosol_optical_depth_550nm"`) and the cmocean
#: ocean/hydrology/DEM set (keyed by variable, e.g. `"salinity"`,
#: `"bathymetry"`). List them all with `sorted(DATA_STYLES)`.
DATA_STYLES.update(_load_preset_asset("ocean_presets.json", "ocean"))
DATA_STYLES.update(_load_weather_presets())


def category_boundaries(values: list[float]) -> list[float]:
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


def _warn_if_outside_fixed_range(data: np.ndarray, lo: float, hi: float) -> None:
    """Warn when finite `data` lies entirely outside a preset's fixed scale `[lo, hi]`.

    A preset that fixes its colour range (contour `levels` or a decoded Magics
    `vmin`/`vmax`) renders the whole field as one edge colour if the data is in
    the wrong units -- the classic footgun being 2 m temperature in Kelvin
    (~250-320) hitting a Celsius scale. Surface it instead of failing silently.
    """
    finite = data[np.isfinite(data)]
    if finite.size and (float(finite.min()) > hi or float(finite.max()) < lo):
        warnings.warn(
            f"data range [{float(finite.min()):g}, {float(finite.max()):g}] lies entirely "
            f"outside the style's fixed scale [{lo:g}, {hi:g}]; the whole field will render "
            "as one edge colour. Is the data in the expected units (e.g. degC, not K)?",
            stacklevel=3,
        )


def resolve_style_norm(
    data: np.ndarray, cfg: dict[str, Any]
) -> tuple[mcolors.Normalize, float, float]:
    """Resolve the colour `Normalize` (and its concrete bounds) for one layer.

    Resolution order:

    - **Explicit `levels`** (the ECMWF / earthkit contour model) resolve to a
      discrete `BoundaryNorm` over those boundaries, with `extend` capping the
      out-of-range ends -- unless the caller supplied `vmin`/`vmax`/`center` or a
      non-linear `norm` kind (`"log"`/`"symlog"`) merged into `cfg`, which take
      precedence and fall through to the continuous path below so the override
      actually rescales the map.
    - Otherwise the bounds come from the layer's `vmin`/`vmax` (auto-ranged from
      the data's finite values when omitted -- essential for real GIS/climate
      fields whose absolute range varies) and an optional diverging `center`
      (a missing bound is made symmetric around it, `center +/- max|data -
      center|`, so the colormap midpoint lands on `center`).
    - A layer with `bands` (a Magics discrete shade) becomes a `BoundaryNorm`
      partitioning `[vmin, vmax]` into `bands` equal intervals; a `norm` kind of
      `"log"`/`"symlog"` selects the matching non-linear norm.

    When a fixed scale (`levels`, or an explicit `vmin`/`vmax`) does not overlap
    the data at all, a `UserWarning` is emitted (the units-mismatch footgun,
    e.g. Kelvin data on a Celsius scale).

    Args:
        data: The layer's 2D data array (finite values drive auto-ranging).
        cfg: The layer's `DATA_STYLES` config dict.

    Returns:
        tuple: `(norm, vmin, vmax)` -- the colour normalization and the
        concrete bounds it resolved to (reused for the layer's legend).
    """
    levels = cfg.get("levels")
    norm_kind = cfg.get("norm")
    # A caller-supplied vmin/vmax/center -- or a non-linear norm kind
    # ("log"/"symlog") -- merged into cfg by apply_data_style's style override
    # takes precedence over the preset's own fixed levels: fall through to the
    # continuous vmin/vmax path below so the override actually rescales the map
    # rather than being silently ignored. Honouring a string norm kind here is
    # what keeps it consistent with a Normalize *instance* override (which
    # apply_data_style applies after this call); otherwise "log"/"symlog" would
    # be dropped on a levels preset while an instance of the same norm is kept.
    caller_override = any(
        cfg.get(key) is not None for key in ("vmin", "vmax", "center")
    ) or (isinstance(norm_kind, str) and norm_kind in ("log", "symlog"))
    if levels is not None and not caller_override:
        # Explicit contour LEVELS (the ECMWF / earthkit-plots model): discrete
        # bands at fixed boundaries with `extend` capping the out-of-range ends
        # -- the look of a professional weather-service map.
        edges = [float(v) for v in levels]
        _warn_if_outside_fixed_range(data, edges[0], edges[-1])
        cmap_obj = cfg["cmap"]
        if not isinstance(cmap_obj, Colormap):
            cmap_obj = mpl.colormaps[cmap_obj]
        # Honour `extend` unless the colormap lacks a spare colour for each
        # reserved under/over slot. A continuous colormap (256 entries) always
        # has room; a one-colour-per-band ListedColormap (aod550=9, 10si=6) does
        # not, so `extend` is dropped there and out-of-range values clamp to the
        # end bands -- but a colour-rich list (cape=255 over 16 bands) keeps it.
        extend = cfg.get("extend", "neither")
        reserved = {"neither": 0, "min": 1, "max": 1, "both": 2}.get(extend, 0)
        if cmap_obj.N < (len(edges) - 1) + reserved:
            extend = "neither"
        norm: mcolors.Normalize = mcolors.BoundaryNorm(
            edges, ncolors=cmap_obj.N, extend=extend
        )
        return norm, edges[0], edges[-1]

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

    bands = cfg.get("bands")
    if bands and norm_kind in (None, "linear") and center is None:
        # Discrete contour bands (Magics-style shade), each mapped to one entry
        # of the paired ListedColormap: the flat, banded ECMWF look, not a smooth
        # (over-exposed) interpolation of the same colours. Partition [vmin, vmax]
        # into `bands` equal intervals so the edges stay within the declared range
        # -- every palette colour is reachable and the legend agrees (a
        # step-aligned partition could overshoot vmax and strand the top colours).
        if cfg.get("vmin") is not None or cfg.get("vmax") is not None:
            _warn_if_outside_fixed_range(data, vmin, vmax)
        boundaries = np.linspace(vmin, vmax, bands + 1)
        return mcolors.BoundaryNorm(boundaries, bands), vmin, vmax
    if norm_kind in (None, "linear") and center is not None:
        # Diverging: put `center` on the colormap midpoint regardless of how
        # the bounds were resolved (auto-symmetric or explicit vmin/vmax).
        if not (vmin < center < vmax):
            raise ValueError(
                f"diverging 'center' ({center}) must lie strictly between "
                f"vmin ({vmin}) and vmax ({vmax})"
            )
        norm = mcolors.TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
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
        lo = (
            vmin
            if (vmin is not None and vmin > 0)
            else (float(positive.min()) if positive.size else None)
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


#: Backward-compatible private aliases for symbols that were renamed public.
_resolve_style_norm = resolve_style_norm
_alpha_rgba = alpha_rgba
_category_boundaries = category_boundaries


def resolve_single_layer_style(style: str) -> tuple[str, dict[str, Any]]:
    """Resolve a single-layer `DATA_STYLES` preset to its `(layer, config)`.

    A single glyph field (a raster band, a mesh's node/face values, a density)
    maps to exactly one preset layer, so multi-layer presets are rejected. Used
    by the glyph `style=` options to look up the preset's cmap/norm/categories.

    Args:
        style: A key of `DATA_STYLES`.

    Returns:
        tuple: `(layer_name, layer_config)` for the preset's single layer.

    Raises:
        ValueError: If `style` is unknown, or names a multi-layer preset.
    """
    if style not in DATA_STYLES:
        raise ValueError(
            f"unknown data style {style!r}; valid styles are {sorted(DATA_STYLES)}"
        )
    layers = DATA_STYLES[style]
    if len(layers) != 1:
        raise ValueError(
            f"data style {style!r} defines multiple layers {sorted(layers)}; a "
            "single glyph field maps to one layer. Use apply_data_style directly "
            "for multi-layer styles."
        )
    name = next(iter(layers))
    return name, layers[name]


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
            `alpha_scaled_mesh`, when `x`/`y` are given) call. A `vmin`/`vmax`/
            `center` here overrides the preset's own colour scale (e.g. a fixed
            Magics range or contour `levels`); a string `norm` (`"log"`/
            `"symlog"`) overrides the preset's norm kind and a `Normalize`
            instance is used directly as the colour norm.

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
    # A caller-supplied `vmin`/`vmax`/`center` overrides the preset's own colour
    # scale (e.g. a Magics preset's decoded fixed range). These are colour-scale
    # keys, not `imshow` kwargs, so pull them out of `render_kwargs` and merge
    # them over each layer's config below.
    style_override: dict[str, Any] = {}
    for key in ("vmin", "vmax", "center"):
        if key in render_kwargs:
            value = render_kwargs.pop(key)
            # Pop it (so it never reaches imshow), but only override the preset when
            # it is actually set -- an explicit None must not wipe a fixed range.
            if value is not None:
                style_override[key] = value
    # A caller `norm=`: a string kind ("linear"/"log"/"symlog") overrides the
    # preset's norm kind via cfg; a Normalize *instance* is used directly as the
    # colour norm. Pop it either way so it never collides with the norm
    # alpha_scaled_image is already given, nor is mis-read as a kind string.
    norm_override = render_kwargs.pop("norm", None)
    if isinstance(norm_override, str):
        style_override["norm"] = norm_override
        norm_override = None
    images: dict[str, Any] = {}
    for i, (name, data) in enumerate(layers.items()):
        cfg = {**preset[name], **style_override}
        data = np.asarray(data, dtype=float)

        categories = cfg.get("categories")
        if categories is not None:
            cats = sorted(categories, key=lambda c: c[0])
            cat_values = [float(c[0]) for c in cats]
            cat_colors = [c[1] for c in cats]
            cat_labels = [c[2] for c in cats]
            cat_cmap = mcolors.ListedColormap(cat_colors)
            cat_norm = mcolors.BoundaryNorm(
                category_boundaries(cat_values), len(cat_colors)
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
                assert x is not None and y is not None
                images[name] = alpha_scaled_mesh(
                    ax,
                    x,
                    y,
                    cat_data,
                    cat_cmap,
                    norm=cat_norm,
                    constant_alpha=1.0,
                    **render_kwargs,
                )
            else:
                images[name] = alpha_scaled_image(
                    ax,
                    cat_data,
                    cat_cmap,
                    norm=cat_norm,
                    constant_alpha=1.0,
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
                        ax,
                        cat_colors,
                        cat_labels,
                        title=cfg["label"],
                        loc="upper left",
                        bbox_to_anchor=(x0, y0),
                    )
                else:
                    leg = disjoint_legend(
                        ax,
                        cat_colors,
                        cat_labels,
                        title=cfg["label"],
                        loc="upper right",
                    )
                if prior_legend is not None and prior_legend is not leg:
                    ax.add_artist(prior_legend)
            continue

        norm, resolved_vmin, resolved_vmax = resolve_style_norm(data, cfg)
        if norm_override is not None:
            # A caller-supplied Normalize instance is used directly as the norm.
            # Label the legend with the INSTANCE's own range, not the preset's
            # resolved bounds: otherwise a levels preset would label the swatch
            # with its fixed level endpoints (e.g. -40..40) while the map uses the
            # instance's scale, and the swatch gradient -- sampled through the
            # instance across [vmin, vmax] -- would feed a LogNorm the preset's
            # negative levels. Fall back to the data's own range for any bound the
            # instance leaves unset (matplotlib autoscales the map the same way).
            norm = norm_override
            finite = data[np.isfinite(data)]
            data_lo = float(finite.min()) if finite.size else resolved_vmin
            data_hi = float(finite.max()) if finite.size else resolved_vmax
            resolved_vmin = norm.vmin if norm.vmin is not None else data_lo
            resolved_vmax = norm.vmax if norm.vmax is not None else data_hi

        alpha_const = cfg.get("alpha")
        alpha_vmin = cfg.get("alpha_vmin")
        alpha_vmax = cfg.get("alpha_vmax")
        if alpha_const is not None and (
            alpha_vmin is not None or alpha_vmax is not None
        ):
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
            assert x is not None and y is not None
            images[name] = alpha_scaled_mesh(
                ax,
                x,
                y,
                data,
                cfg["cmap"],
                norm=norm,
                alpha_norm=alpha_norm,
                constant_alpha=alpha_const,
                **render_kwargs,
            )
        else:
            images[name] = alpha_scaled_image(
                ax,
                data,
                cfg["cmap"],
                norm=norm,
                alpha_norm=alpha_norm,
                constant_alpha=alpha_const,
                **render_kwargs,
            )
        if legend:
            bounds = (
                legend_bounds[i]
                if legend_bounds is not None
                else (0.02, 0.92 - 0.12 * i, 0.32, 0.06)
            )
            # Mark each endpoint as capped ("≤"/"≥") only where the norm reserves
            # an out-of-range slot, so the legend states the capping the map
            # applies -- a two-sided BoundaryNorm caps both ends, a `neither`/
            # downgraded one caps neither, and a continuous norm keeps the
            # open-ended "≥". Derived in one helper shared with animate().
            vmin_prefix, vmax_prefix = swatch_extend_prefixes(norm)
            swatch_legend(
                ax,
                cfg["cmap"],
                cfg["label"],
                vmin=resolved_vmin,
                vmax=resolved_vmax,
                vmin_prefix=vmin_prefix,
                vmax_prefix=vmax_prefix,
                bounds=bounds,
                norm=norm,
            )
    return images


#: A single color as the `Colors` class accepts it: a hex string (with or
#: without the leading "#") or an RGB tuple (0-1 normalized or 0-255 range).
_ColorEntry = str | tuple[float, float, float]
#: A `Colors.__init__`/`color_value` value: one color, or a list of colors
#: (hex strings and RGB tuples may be freely mixed within the list).
ColorValue = _ColorEntry | list[_ColorEntry]


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
        color_value: ColorValue,
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
        color_list: list[_ColorEntry]
        if isinstance(color_value, str) or isinstance(color_value, tuple):
            color_list = [color_value]
        elif isinstance(color_value, list):
            color_list = color_value
        else:
            raise ValueError(
                "The color_value must be a list of hex colors, list of tuples (RGB color), a single hex "
                "or single RGB tuple color."
            )

        self._color_value: list[_ColorEntry] = color_list

    @classmethod
    def create_from_image(cls, path: str | os.PathLike) -> "Colors":
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
        # `.convert("RGB")` above guarantees a 3-int-tuple pixel at every
        # coordinate; `Image.getpixel`'s general stub is looser (it also
        # covers single-band/palette modes), so tell mypy what mode this is.
        color_values = cast(
            "list[_ColorEntry]",
            [image.getpixel((x, int(height / 2))) for x in range(width)],
        )

        return cls(color_values)

    def get_type(self) -> list[str]:
        """Determine the type of each color value.

        This method analyzes each color value stored in the object and determines
        its type: hex, rgb (values 0-255), or rgb-normalized (values 0-1).

        Returns:
            list[str]: A list of strings indicating the type of each color value.
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
    def color_value(self) -> list[_ColorEntry]:
        """Get the color values stored in the object.

        This property returns the color values that were provided when initializing
        the Colors object or set afterwards. The values can be hex color strings,
        RGB tuples with values between 0-255, or normalized RGB tuples with values
        between 0-1.

        Returns:
            list[_ColorEntry]: A list containing the color values. Each element can be:
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

    def to_hex(self) -> list[str]:
        """Convert all color values to hexadecimal format.

        This method converts all color values stored in the object to hexadecimal format.
        RGB tuples (both normalized and 0-255 range) are converted to their hex equivalents.
        Hex colors remain unchanged.

        Returns:
            list[str]: A list of hexadecimal color strings. Each string is in the format '#RRGGBB'.

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
        converted_color: list[str] = []
        color_type = self.get_type()
        for ind, color_i in enumerate(self.color_value):
            if color_type[ind] == "hex":
                # get_type() tagged this entry "hex", so it is a str.
                converted_color.append(cast(str, color_i))
            elif color_type[ind] == "rgb":
                # get_type() tagged this entry "rgb", so it is a 3-tuple.
                r, g, b = cast("tuple[float, float, float]", color_i)
                rgb_color_normalized = (r / 255, g / 255, b / 255)
                converted_color.append(mcolors.to_hex(rgb_color_normalized))
            else:
                converted_color.append(mcolors.to_hex(color_i))
        return converted_color

    def is_valid_hex(self) -> list[bool]:
        """Check if each color value is a valid hexadecimal color.

        This method checks each color value stored in the object to determine
        if it is a valid hexadecimal color string.

        Returns:
            list[bool]: A list of boolean values, one for each color value in the object.
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
    def _is_valid_hex_i(hex_color: _ColorEntry) -> bool:
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

    def is_valid_rgb(self) -> list[bool]:
        """Check if each color value is a valid RGB color.

        This method checks each color value stored in the object to determine
        if it is a valid RGB color tuple (either with values between 0-255 or
        normalized values between 0-1).

        Returns:
            list[bool]: A list of boolean values, one for each color value in the object.
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
    ) -> list[tuple[int | float, int | float, int | float]]:
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
            list[tuple[int | float, int | float, int | float]]: A list of RGB tuples.
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
        rgb: list[tuple[int | float, int | float, int | float]] = []
        if normalized:
            for ind, color_i in enumerate(self.color_value):
                # if the color is in RGB format (0-255), normalize the values to be between 0 and 1
                if color_type[ind] == "rgb":
                    r, g, b = cast("tuple[float, float, float]", color_i)
                    rgb.append((r / 255, g / 255, b / 255))
                else:
                    # any other format, just convert it to RGB
                    rgb.append(mcolors.to_rgb(color_i))
        else:
            for ind, color_i in enumerate(self.color_value):
                # if the color is in RGB format (0-255), normalize the values to be between 0 and 1
                if color_type[ind] == "rgb":
                    rgb.append(cast("tuple[int, int, int]", color_i))
                else:
                    # any other format, just convert it to RGB
                    r, g, b = mcolors.to_rgb(color_i)
                    rgb.append((int(r * 255), int(g * 255), int(b * 255)))

        return rgb

    def get_color_map(self, name: str | None = None) -> Colormap:
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
