#!/usr/bin/env python3
"""Build cleopatra's merged weather preset asset (maintainer-side, offline).

This script is **not** part of the shipped package and is **not** run at import
or install time. It is the one-off, maintainer-machine tool that derives
``src/cleopatra/styling/data/weather_presets.json`` from two open sources, keyed
by GRIB shortName:

- **magics** (Apache-2.0): the full operational parameter library, each
  style's discrete colour bands resolved from ``share/magics/styles/default/
  palettes.json`` + ``contours.json`` + the named-colour table in
  ``src/common/Colour.cc``. Every Magics preset renders as equal-width bands
  (``bands`` = its colour count) over a fixed ``vmin``/``vmax`` decoded from its
  style name's ``f<from>t<to>[i<interval>]`` grammar, when the name carries one.
- **reference styles** (Apache-2.0): a curated subset of parameters using
  the ``optimal`` style variant -- the actual professional look
  (explicit, non-uniform contour ``levels`` + an ``extend`` cap, or a genuine
  continuous ramp when a parameter has no discrete levels).

Where both sources cover the same shortName, the reference record is kept
and the Magics one dropped (see ``REFERENCE_OVERRIDE_KEYS``) -- the reference set is the
professional weather-service look; only one record per shortName ships.

Only each source's colour *data* and parameter/label associations are vendored,
never any upstream code.

Maintainer dependencies: ``matplotlib`` (already a cleopatra dependency, used to
resolve Magics' named colours) and ``PyYAML`` (``import yaml``, used to parse
the reference style files) -- neither is a cleopatra runtime dependency; install
PyYAML in the maintainer environment before re-running.

Re-run (from the repo root)::

    python tools/build_weather_presets.py src/cleopatra/styling/data/weather_presets.json [<magics_ref>] [<reference_ref>]

``<magics_ref>`` defaults to ``develop``, ``<reference_ref>`` defaults to ``main``.
"""

import colorsys
import json
import re
import sys
import urllib.request
from pathlib import Path

import yaml
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgba

# Opener restricted to http(s): only the HTTP(S) handlers are registered (plus
# the supporting redirect/error-processing handlers `urlopen` needs), so a
# file:///ftp:///data: URL structurally cannot be opened through it -- no
# handler claims it.
_HTTP_ONLY_OPENER = urllib.request.OpenerDirector()
for _handler in (
    urllib.request.HTTPHandler(),
    urllib.request.HTTPSHandler(),
    urllib.request.HTTPErrorProcessor(),
    urllib.request.HTTPRedirectHandler(),
    urllib.request.HTTPDefaultErrorHandler(),
    urllib.request.UnknownHandler(),
):
    _HTTP_ONLY_OPENER.add_handler(_handler)

#: GRIB shortNames reference-source curated defaults supersede -- dropped from
#: the Magics side so the merged file carries exactly one record per shortName.
REFERENCE_OVERRIDE_KEYS = {
    "2d", "2t", "aod550", "duaod550", "tp",
    # CAMS composition shortNames whose reference `optimal` look supersedes the
    # Magics palette for the same field (reference-only shortNames -- the CH4/CO
    # level slices, uvi, uvics -- need no entry; they have no Magics record).
    "suaod550", "tcco", "tcco2", "tcch4", "gtco3", "tcso2",
    "no2", "go3", "frpfire", "pm10", "pm2p5",
    # Operational long-tail fields the reference curated look supersedes for a
    # shortName Magics also carries (waves, cloud cover, humidity, the 2 m
    # extremes, and the instability indices). Reference-only long-tail shortNames
    # -- the pressure-level slices, hydrology, and single-field additions below --
    # need no entry; Magics carries no record for them.
    "swh", "mwp", "mpts", "mpww", "shts", "shww", "sh10",
    "hcc", "mcc", "lcc", "q", "mn2t", "mx2t", "kx", "totalx", "cin",
}

# --- Magics -----------------------------------------------------------------

MAGICS_BASE_TEMPLATE = (
    "https://raw.githubusercontent.com/ecmwf/magics/{ref}/share/magics/styles"
)
MAGICS_REPO_TEMPLATE = "https://raw.githubusercontent.com/ecmwf/magics/{ref}"

#: Matches a Magics named-colour definition in ``src/common/Colour.cc``:
#: ``colours_["greenish_blue"] = Rgb(0.0000, 0.5000, 1.0000);`` (float 0-1
#: components; the ``undefined`` sentinel is ``Rgb(-1., -1., -1.)``).
_RGB_DEF = re.compile(
    r'colours_\["([^"]+)"\]\s*=\s*Rgb\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)'
)

#: A Magics style name encodes its contour range in an
#: ``f<from>t<to>[i<interval>]`` grammar (``M`` = minus). The interval is
#: optional and, once decoded, unused (see `transform_magics`) -- only the
#: range matters for the merged asset.
_MAGICS_RANGE = re.compile(r"f(M?\d+(?:_\d+)?)t(M?\d+(?:_\d+)?)(?:i(\d+(?:_\d+)?))?")


def _magics_num(token):
    """Decode one numeric token of a Magics range name.

    The grammar: ``M`` prefixes a negative; ``_`` is the decimal point
    (``1_5`` -> 1.5); a leading ``0`` on a multi-digit token marks a decimal
    (``05`` -> 0.5, ``01`` -> 0.1); ``0`` alone is zero; any other token is an
    integer (``10`` -> 10, not 1.0).
    """
    negative = token.startswith("M")
    if negative:
        token = token[1:]
    if "_" in token:
        value = float(token.replace("_", "."))
    elif len(token) > 1 and token[0] == "0":
        value = float("0." + token[1:])
    else:
        value = float(token)
    return -value if negative else value


def decode_magics_range(magics_style):
    """A Magics style name -> ``(vmin, vmax)``, or ``None`` if it carries no usable range."""
    match = _MAGICS_RANGE.search(magics_style or "")
    if match is None:
        return None
    vmin = _magics_num(match.group(1))
    vmax = _magics_num(match.group(2))
    if vmin >= vmax:
        return None
    return vmin, vmax


def fetch_magics(base, path):
    with _HTTP_ONLY_OPENER.open(f"{base}/{path}") as r:
        return json.load(r)


def fetch_magics_colours(magics_ref):
    """Magics' named-colour table (name -> hex), parsed from ``src/common/Colour.cc``."""
    url = MAGICS_REPO_TEMPLATE.format(ref=magics_ref) + "/src/common/Colour.cc"
    with _HTTP_ONLY_OPENER.open(url) as r:
        src = r.read().decode("utf-8", "replace")
    table = {}
    for name, red, green, blue in _RGB_DEF.findall(src):
        rgb = (float(red), float(green), float(blue))
        if min(rgb) < 0:  # the 'undefined' sentinel
            continue
        table[name] = to_hex(rgb)
    return table


def _parse_rgb(c):
    """A Magics ``rgb()``/``rgba()`` string -> (hex, alpha|None), or None."""
    is_float = "." in c
    vals = [float(x) for x in re.findall(r"[\d.]+", c.replace(" ", ""))]
    if len(vals) < 3:
        return None
    rgb, alpha = vals[:3], (vals[3] if len(vals) > 3 else None)
    out = [
        min(255, max(0, round(v * 255) if (is_float and v <= 1.0) else round(v)))
        for v in rgb
    ]
    return "#{:02x}{:02x}{:02x}".format(*out), (
        round(alpha, 4) if alpha is not None else None
    )


def parse_color(c, named_colours):
    """A Magics colour string -> (hex, alpha|None), or None if unresolvable."""
    c = c.strip()
    if c.lower().startswith(("rgb(", "rgba(")):
        return _parse_rgb(c)
    if c.lower().startswith("hsl("):
        nums = [float(x) for x in re.findall(r"[-\d.]+", c)]
        if len(nums) < 3:
            return None
        red, green, blue = colorsys.hls_to_rgb(nums[0] / 360.0, nums[2], nums[1])
        return to_hex((red, green, blue)), None
    if c in named_colours:
        return named_colours[c], None
    try:
        r, g, b, a = to_rgba(c)
        return to_hex((r, g, b)), (round(float(a), 4) if a < 1.0 else None)
    except (ValueError, TypeError):
        return None


def _magics_style_to_palette(palettes):
    """Map each Magics style name (as tagged in `palettes.json`) to its palette record."""
    style_to_palette = {}
    for pval in palettes.values():
        for tag in pval.get("tags", []):
            style_to_palette.setdefault(str(tag), pval)
    return style_to_palette


def _resolve_palette_colours(pal, named_colours):
    """Resolve one Magics palette's colour strings -> (hexes, alphas, unresolved).

    A colour we still cannot resolve would silently truncate the palette
    (mis-weighting the ramp), so it is returned for the caller's report
    rather than dropped unseen.
    """
    hexes, alphas, unresolved = [], [], set()
    for c in pal.get("values", []):
        parsed = parse_color(c, named_colours)
        if parsed is None:
            unresolved.add(c.strip())
            continue
        hexes.append(parsed[0])
        alphas.append(parsed[1])
    return hexes, alphas, unresolved


def build_magics(magics_ref):
    """Fetch + resolve every Magics parameter style -> (presets, skipped, unresolved)."""
    base = MAGICS_BASE_TEMPLATE.format(ref=magics_ref)
    palettes = fetch_magics(base, "default/palettes.json")
    contours = fetch_magics(base, "default/contours.json")
    named_colours = fetch_magics_colours(magics_ref)
    style_to_palette = _magics_style_to_palette(palettes)

    presets, skipped, unresolved = {}, [], set()
    for entry in contours:
        crit = entry.get("criteria", {})
        short = crit.get("shortName")
        style = entry.get("style")
        if not short or not style:
            continue
        pal = style_to_palette.get(style)
        if pal is None:
            skipped.append((short, style))
            continue
        hexes, alphas, pal_unresolved = _resolve_palette_colours(pal, named_colours)
        unresolved |= pal_unresolved
        if len(hexes) < 2:
            skipped.append((short, style))
            continue
        has_alpha = any(a is not None for a in alphas)
        presets[short] = {
            "label": crit.get("long_name") or short,
            "palette": hexes,
            "opacity": "overlay" if has_alpha else "opaque",
            "magics_style": style,
        }
    return presets, skipped, sorted(unresolved)


# --- reference styles -----------------------------------------------------------

REFERENCE_RAW = (
    "https://raw.githubusercontent.com/ecmwf/earthkit-plots/{ref}"
    "/src/earthkit/plots/data/styles/auto-styles/{stem}.yml"
)

#: reference auto-style file stem -> (GRIB shortName cleopatra keys it by, label).
#: The `optimal` variant of each is vendored as the parameter's default style.
REFERENCE_PARAMS = {
    "2t": ("2t", "2 m temperature"),
    "2t_dewpoint": ("2d", "2 m dewpoint temperature"),
    "composition_aod550": ("aod550", "Total aerosol optical depth at 550 nm"),
    "composition_duaod550": ("duaod550", "Dust aerosol optical depth at 550 nm"),
    "10u": ("10u", "10 m U wind component"),
    "10v": ("10v", "10 m V wind component"),
    "wind-speed-at-10m": ("10si", "10 m wind speed"),
    "total-precipitation": ("tp", "Total precipitation"),
    "cape": ("cape", "Convective available potential energy"),
    # CAMS atmospheric composition. aod550/duaod550 are covered above; the
    # rest of reference's `composition_*` auto-styles are added here. The
    # pressure-level CH4/CO slices have no distinct GRIB shortName, so they key
    # by a synthetic code (`ch4_850`, `co_700`, ...) mapped to a descriptive
    # name in SHORTNAME_TO_NAME -- style-library entries, not shortName-matched.
    "composition_suaod550": ("suaod550", "Sulphate aerosol optical depth at 550 nm"),
    "composition_o3_surface": ("go3", "Ozone"),
    "composition_o3_totalcolumn": ("gtco3", "Total column ozone"),
    "composition_no2_surface": ("no2", "Nitrogen dioxide"),
    "composition_so2_totalcolumn": ("tcso2", "Total column sulphur dioxide"),
    "composition_co_totalcolumn": ("tcco", "Total column carbon monoxide"),
    "composition_co700": ("co_700", "Carbon monoxide at 700 hPa"),
    "composition_co_500hpa": ("co_500", "Carbon monoxide at 500 hPa"),
    "composition_co2_totalcolumn": ("tcco2", "CO2 column-mean molar fraction"),
    "composition_ch4_totalcolumn": ("tcch4", "CH4 column-mean molar fraction"),
    "composition_ch4_surface": ("ch4sfc", "Methane at the surface"),
    "composition_ch4_850": ("ch4_850", "Methane at 850 hPa"),
    "composition_ch4_500": ("ch4_500", "Methane at 500 hPa"),
    "composition_ch4_300": ("ch4_300", "Methane at 300 hPa"),
    "composition_ch4_50": ("ch4_50", "Methane at 50 hPa"),
    "composition_pm10": ("pm10", "Particulate matter d < 10 um"),
    "composition_pm2p5": ("pm2p5", "Particulate matter d < 2.5 um"),
    "composition_uvindex": ("uvi", "UV index"),
    "composition_uvindex_clearsky": ("uvics", "UV index (clear sky)"),
    "composition_fire": ("frpfire", "Wildfire radiative power"),
    # Operational long tail: the deterministic curated auto-styles beyond the
    # CAMS composition set. Each vendors the `optimal` variant of a shadable
    # Style. Fields that already have a GRIB shortName in SHORTNAME_TO_NAME reuse
    # it (and appear in REFERENCE_OVERRIDE_KEYS above); the pressure-level slices,
    # hydrology fields, and single additions with no distinct GRIB code key by a
    # synthetic code mapped to a descriptive name in SHORTNAME_TO_NAME.
    # Ocean waves.
    "wave_swh": ("swh", "Significant wave height"),
    "wave_mwp": ("mwp", "Mean wave period"),
    "wave_mpts": ("mpts", "Mean period of total swell"),
    "wave_mpww": ("mpww", "Mean period of wind waves"),
    "wave_shts": ("shts", "Significant height of total swell"),
    "wave_shww": ("shww", "Significant height of wind waves"),
    "wave_sh10": ("sh10", "Significant wave height of waves over 10 s"),
    # Sea ice.
    "sea_ice_cover": ("ci", "Sea ice cover"),
    # Cloud cover and cloud geometry.
    "hcc": ("hcc", "High cloud cover"),
    "mcc": ("mcc", "Medium cloud cover"),
    "lcc": ("lcc", "Low cloud cover"),
    "cbh": ("cbh", "Cloud base height"),
    "ceiling": ("ceil", "Cloud ceiling"),
    # Humidity.
    "rh1000": ("rh1000", "Relative humidity at 1000 hPa"),
    "q1000": ("q", "Specific humidity"),
    # Dynamics: vorticity, potential vorticity, divergence, vertical velocity.
    "700vorticity": ("vo700", "Relative vorticity at 700 hPa"),
    "850vorticity": ("vo850", "Relative vorticity at 850 hPa"),
    "315Kpotvort": ("pv315k", "Potential vorticity on the 315 K surface"),
    "1000divergence": ("d1000", "Divergence at 1000 hPa"),
    "700w": ("w700", "Vertical velocity at 700 hPa"),
    # Precipitation, snow, and hydrology.
    "tp_rate": ("tprate", "Total precipitation rate"),
    "snow-water-equivalent": ("swe", "Snow water equivalent"),
    "total-runoff-water-equivalent": ("trwe", "Total runoff water equivalent"),
    "river-discharge": ("dis", "River discharge"),
    "soil-wetness-index": ("swi", "Soil wetness index"),
    # 2 m temperature extremes and pressure-level temperatures.
    "mn2t": ("mn2t", "Minimum 2 m temperature"),
    "mx2t": ("mx2t", "Maximum 2 m temperature"),
    "t3": ("t3", "Temperature at 3 hPa"),
    "t250": ("t250", "Temperature at 250 hPa"),
    "t500": ("t500", "Temperature at 500 hPa"),
    "t925": ("t925", "Temperature at 925 hPa"),
    # Wind speed on pressure levels.
    "200_windspeed_field": ("ws200", "Wind speed at 200 hPa"),
    "250ws": ("ws250", "Wind speed at 250 hPa"),
    "300_windspeed": ("ws300", "Wind speed at 300 hPa"),
    "500ws": ("ws500", "Wind speed at 500 hPa"),
    "800ws": ("ws800", "Wind speed at 800 hPa"),
    "925ws": ("ws925", "Wind speed at 925 hPa"),
    # Other single fields.
    "visibility": ("vis", "Visibility"),
    "fzra": ("fzra", "Freezing rain"),
    "ssr": ("ssr", "Surface net solar radiation"),
    "kindex": ("kx", "K index"),
    "totalx": ("totalx", "Total totals index"),
    "cin": ("cin", "Convective inhibition"),
    "wbpt850": ("wbpt850", "Wet-bulb potential temperature at 850 hPa"),
}

_REFERENCE_RANGE = re.compile(r"range\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def parse_levels(levels):
    """reference `levels` (a ``range(a, b, c)`` string or an explicit list) -> list | None."""
    if levels is None:
        return None
    if isinstance(levels, str):
        match = _REFERENCE_RANGE.search(levels)
        if match is None:
            return None
        start, stop, step = (int(x) for x in match.groups())
        return list(range(start, stop, step))
    return [float(v) for v in levels]


def clean_colors(colors):
    """Keep a matplotlib cmap NAME as-is; normalise a colour LIST to ``#rrggbb``.

    A listed palette entry may arrive as a ``#rrggbbaa`` hex (the trailing alpha
    is dropped) or as an ``[r, g, b]`` / ``[r, g, b, a]`` float triple/quad in the
    0-1 range -- both are folded to a plain ``#rrggbb`` string so the vendored
    ``colors`` is always a list of hex strings the on-disk schema accepts.
    """
    if not isinstance(colors, list):
        return colors
    out = []
    for c in colors:
        if isinstance(c, str) and c.startswith("#") and len(c) == 9:
            out.append(c[:7])
        elif isinstance(c, (list, tuple)):
            out.append(to_hex(tuple(c[:3])))
        else:
            out.append(c)
    return out


def fit_colors_to_levels(colors, levels):
    """Interpolate a short listed palette up to its `levels`-defined bin count.

    A `BoundaryNorm` over N `levels` has N-1 bins, and a `ListedColormap` must
    supply at least that many colours (the `extend` cap reuses the end colours
    via set_under/over, so it needs no extra listed colour). A few reference
    ``optimal`` palettes ship one colour short of their bins -- the source resamples
    them internally; cleopatra's strict `BoundaryNorm` would raise. Interpolate
    the palette up to N-1 colours (preserving the full level range and the colour
    progression) so the vendored data renders directly. A palette that already
    meets or exceeds its bin count, a matplotlib colormap *name*, or a palette
    without explicit levels is returned unchanged.
    """
    if not isinstance(colors, list) or not isinstance(levels, list):
        return colors
    bins = len(levels) - 1
    if len(colors) >= bins or len(colors) < 2:
        return colors
    cmap = LinearSegmentedColormap.from_list("_fit", colors, N=bins)
    return [to_hex(cmap(i)) for i in range(bins)]


def fetch_reference(ref, stem):
    with _HTTP_ONLY_OPENER.open(REFERENCE_RAW.format(ref=ref, stem=stem)) as r:
        return yaml.safe_load(r.read().decode("utf-8"))


def build_reference(ref):
    """Fetch + resolve every curated reference parameter style -> (presets, skipped)."""
    presets, skipped = {}, []
    for stem, (short, label) in REFERENCE_PARAMS.items():
        doc = fetch_reference(ref, stem)
        opt = doc["styles"][doc["optimal"]]
        if opt.get("type") != "Style" or opt.get("colors") is None:
            skipped.append((short, stem))
            continue
        levels = parse_levels(opt.get("levels"))
        presets[short] = {
            "label": label,
            "colors": fit_colors_to_levels(clean_colors(opt["colors"]), levels),
            "levels": levels,
            "extend": opt.get("extend", "neither"),
            "units": opt.get("units"),
        }
    return presets, skipped


# --- merge --------------------------------------------------------------------


def transform_magics(raw_presets):
    """Magics raw records (``palette``/``opacity``/``magics_style``) -> the merged schema.

    Renames ``palette`` to ``colors``; records the discrete band count
    (``bands``, always ``len(colors)`` -- every Magics preset renders as flat
    equal-width bands); decodes ``magics_style``'s ``f<from>t<to>`` range into
    plain ``vmin``/``vmax`` when the name carries one, dropping the encoded
    token itself (its optional interval is decoded but never used downstream,
    same as before); drops the shortNames the reference source supersedes.
    """
    out = {}
    for key, rec in raw_presets.items():
        if key in REFERENCE_OVERRIDE_KEYS:
            continue
        colors = rec["palette"]
        merged = {
            "label": rec["label"],
            "colors": colors,
            "opacity": rec["opacity"],
            "bands": len(colors),
        }
        rng = decode_magics_range(rec.get("magics_style"))
        if rng is not None:
            merged["vmin"], merged["vmax"] = rng
            # A diverging field (range straddling zero) must not be a value-linked
            # "overlay": cleopatra ties opacity monotonically to value, so the
            # strong negative anomalies would render transparent while the quiet
            # centre stays opaque -- inverting the intent. Magics' own alpha ramp
            # for such fields is symmetric (faint near zero), which the single
            # monotonic overlay policy can't reproduce, so fall back to an opaque
            # field rather than silently hiding half the signal.
            if merged["opacity"] == "overlay" and rng[0] < 0 < rng[1]:
                merged["opacity"] = "opaque"
        out[key] = merged
    return out


#: reference shortNames whose field fades at the low end (value-linked opacity)
#: rather than rendering as a constant-alpha opaque layer -- e.g. precipitation,
#: which should be transparent over dry ground. These carry no constant `alpha`,
#: so cleopatra ties their opacity to value (see the `total_precipitation`
#: fade-at-low-end invariant in `tests/test_preset_schema.py`).
REFERENCE_FADE_KEYS = {"tp"}


def transform_reference(raw_presets):
    """Reference raw records -> the merged schema (adds the explicit opacity policy).

    Every reference preset renders as an opaque full field, except the
    `REFERENCE_FADE_KEYS` fields, which fade at the low end (a value-linked
    overlay carrying no constant `alpha`).
    """
    out = {}
    for key, rec in raw_presets.items():
        merged = dict(rec)
        merged["opacity"] = "overlay" if key in REFERENCE_FADE_KEYS else "opaque"
        out[key] = merged
    return out


#: GRIB shortName -> the descriptive `DATA_STYLES` key it ships under. Both
#: Magics and the reference source index their upstream data by shortName (needed to
#: join contours.json/palettes.json, or locate the right auto-style YAML), so
#: fetching stays shortName-keyed throughout `build_magics`/`build_reference`;
#: this is applied once, at the very end, to the merged asset's keys.
SHORTNAME_TO_NAME = {
    "10fg": "wind_gust_10m",
    "10fgi": "wind_gust_10m_index",
    "10si": "wind_speed_10m",
    "10u": "wind_u_10m",
    "10v": "wind_v_10m",
    "10wsi": "wind_speed_10m_index",
    "2d": "dewpoint_temperature_2m",
    "2t": "temperature_2m",
    "2ti": "temperature_2m_index",
    "2tp": "temperature_2m_probability",
    "aod550": "aerosol_optical_depth_550nm",
    "cape": "convective_available_potential_energy",
    "capei": "convective_available_potential_energy_index",
    "capes": "convective_available_potential_energy_shear",
    "capesi": "convective_available_potential_energy_shear_index",
    "cbh": "cloud_base_height",
    "ceil": "cloud_ceiling",
    "ch4_300": "methane_300hpa",
    "ch4_50": "methane_50hpa",
    "ch4_500": "methane_500hpa",
    "ch4_850": "methane_850hpa",
    "ch4sfc": "methane_surface",
    "ci": "sea_ice_cover",
    "cin": "convective_inhibition",
    "clbt": "cloudy_brightness_temperature",
    "co": "carbon_monoxide",
    "co_500": "carbon_monoxide_500hpa",
    "co_700": "carbon_monoxide_700hpa",
    "cp": "convective_precipitation",
    "crfrate": "convective_rainfall_rate",
    "d": "divergence",
    "d1000": "divergence_1000hpa",
    "dis": "river_discharge",
    "duaod550": "dust_aerosol_optical_depth_550nm",
    "frpfire": "wildfire_radiative_power",
    "fzra": "freezing_rain",
    "go3": "ozone",
    "gtco3": "total_column_ozone",
    "hcc": "high_cloud_cover",
    "kx": "k_index",
    "lcc": "low_cloud_cover",
    "lsp": "large_scale_precipitation",
    "lsrrate": "large_scale_rainfall_rate",
    "maxswh": "max_significant_wave_height",
    "maxswhi": "max_significant_wave_height_index",
    "mcc": "medium_cloud_cover",
    "mean10ws": "mean_wind_speed_10m",
    "mean2t": "mean_temperature_2m",
    "mn2t": "min_temperature_2m",
    "mn2ti": "min_temperature_2m_index",
    "mpts": "mean_period_total_swell",
    "mpww": "mean_period_wind_waves",
    "mslpp": "mean_sea_level_pressure_probability",
    "mwp": "mean_wave_period",
    "mx2t": "max_temperature_2m",
    "mx2ti": "max_temperature_2m_index",
    "no2": "nitrogen_dioxide",
    "ph": "hurricane_probability",
    "pm10": "particulate_matter_10um",
    "pm2p5": "particulate_matter_2p5um",
    "prate": "precipitation_rate",
    "pt": "potential_temperature",
    "ptd": "tropical_depression_probability",
    "pts": "tropical_storm_probability",
    "pv315k": "potential_vorticity_315k",
    "q": "specific_humidity",
    "rh1000": "relative_humidity_1000hpa",
    "sf": "snowfall",
    "sfi": "snowfall_index",
    "sh10": "significant_wave_height_over_10s_period",
    "shts": "significant_height_total_swell",
    "shww": "significant_height_wind_waves",
    "srweq": "snowfall_rate_water_equivalent",
    "ssr": "surface_net_solar_radiation",
    "stl1p": "soil_temperature_level1_probability",
    "suaod550": "sulphate_aerosol_optical_depth_550nm",
    "swe": "snow_water_equivalent",
    "swh": "significant_wave_height_combined",
    "swi": "soil_wetness_index",
    "t": "air_temperature",
    "t250": "temperature_250hpa",
    "t3": "temperature_3hpa",
    "t500": "temperature_500hpa",
    "t925": "temperature_925hpa",
    "tcch4": "ch4_column_mean_molar_fraction",
    "tcco": "total_column_carbon_monoxide",
    "tcco2": "co2_column_mean_molar_fraction",
    "tcso2": "total_column_sulphur_dioxide",
    "totalx": "total_totals_index",
    "tp": "total_precipitation",
    "tpi": "total_precipitation_index",
    "tpp": "total_precipitation_probability",
    "tprate": "total_precipitation_rate",
    "trwe": "total_runoff_water_equivalent",
    "uvbed": "uv_biologically_effective_dose",
    "uvbedcs": "uv_biologically_effective_dose_clear_sky",
    "uvi": "uv_index",
    "uvics": "uv_index_clear_sky",
    "vis": "visibility",
    "vo": "relative_vorticity",
    "vo700": "relative_vorticity_700hpa",
    "vo850": "relative_vorticity_850hpa",
    "w": "vertical_velocity",
    "w700": "vertical_velocity_700hpa",
    "wbpt850": "wet_bulb_potential_temperature_850hpa",
    "ws": "total_wind_speed",
    "ws200": "wind_speed_200hpa",
    "ws250": "wind_speed_250hpa",
    "ws300": "wind_speed_300hpa",
    "ws500": "wind_speed_500hpa",
    "ws800": "wind_speed_800hpa",
    "ws925": "wind_speed_925hpa",
}


def rename_to_descriptive_keys(merged_presets):
    """Rename a merged (shortName-keyed) preset dict to its `SHORTNAME_TO_NAME` keys.

    A shortName absent from `SHORTNAME_TO_NAME` (e.g. a new upstream parameter)
    keeps its raw code rather than crashing the build -- add it to the map and
    re-run once its descriptive name is chosen.
    """
    out = {}
    unmapped = []
    for key, rec in merged_presets.items():
        name = SHORTNAME_TO_NAME.get(key)
        if name is None:
            unmapped.append(key)
            name = key
        out[name] = rec
    return out, unmapped


def _safe_out_path(out_path):
    """Resolve `out_path` and confine it to the current working directory tree.

    This is a maintainer-only CLI whose destination comes from `argv`, so
    canonicalise the path and reject anything that resolves outside the invoking
    directory (the repo) before opening it -- a stray or `..`-laden argument must
    not escape the tree (path traversal, CWE-22).
    """
    base = Path.cwd().resolve()
    resolved = Path(out_path)
    resolved = (
        resolved.resolve() if resolved.is_absolute() else (base / resolved).resolve()
    )
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"refusing to write outside {base}: {out_path!r}")
    return resolved


def _canonical_layer(rec):
    """Build one canonical v2 layer dict from a merged old-format record.

    Chooses the `colormap` mode (a name string -> `named`; a banded/levelled list
    -> `listed`; a plain list -> `perceptual`) and copies the optional
    `units`/`bands`/`levels`/`extend`/`vmin`/`vmax` fields, plus a constant
    `alpha` for an opaque field.

    Args:
        rec: One merged old-format record.

    Returns:
        dict: The canonical layer.
    """
    colors = rec["colors"]
    if isinstance(colors, str):
        colormap = "named"
    elif rec.get("bands") is not None or rec.get("levels") is not None:
        colormap = "listed"
    else:
        colormap = "perceptual"
    layer = {"label": rec["label"], "colors": colors, "colormap": colormap}
    if rec.get("units") is not None:
        layer["units"] = rec["units"]
    if rec.get("bands") is not None:
        layer["bands"] = rec["bands"]
    if rec.get("levels") is not None:
        layer["levels"] = rec["levels"]
        if rec.get("extend") is not None:
            layer["extend"] = rec["extend"]
    if rec.get("vmin") is not None:
        layer["vmin"] = rec["vmin"]
    if rec.get("vmax") is not None:
        layer["vmax"] = rec["vmax"]
    if rec.get("opacity") == "opaque":
        layer["alpha"] = 1.0
    return layer


def _to_canonical(records):
    """Wrap the merged old-format records into the canonical v2 preset asset.

    Each record becomes a single-layer preset (see `_canonical_layer`).

    Args:
        records: The merged shortName -> old-format record mapping.

    Returns:
        dict: A `{version, source, license, presets}` v2 asset.
    """
    presets = {}
    for name, rec in sorted(records.items()):
        presets[name] = {"layers": {name: _canonical_layer(rec)}}
    return {"version": 1, "source": "magics", "license": "Apache-2.0", "presets": presets}


def main(out_path, magics_ref="develop", reference_ref="main"):
    magics_presets, magics_skipped, unresolved = build_magics(magics_ref)
    reference_presets, reference_skipped = build_reference(reference_ref)
    merged = {
        **transform_magics(magics_presets),
        **transform_reference(reference_presets),
    }
    renamed, unmapped = rename_to_descriptive_keys(merged)
    asset = _to_canonical(renamed)
    with open(_safe_out_path(out_path), "w", encoding="utf-8") as f:
        json.dump(asset, f, indent=1, ensure_ascii=False)
    if unmapped:
        print(
            f"WARNING: {len(unmapped)} shortName(s) have no descriptive name in "
            f"SHORTNAME_TO_NAME, kept as-is: {', '.join(sorted(unmapped))}"
        )
    print(
        f"wrote {len(merged)} weather presets to {out_path} "
        f"({len(magics_presets) - len(REFERENCE_OVERRIDE_KEYS)} Magics + "
        f"{len(reference_presets)} reference, {len(magics_skipped)} Magics + "
        f"{len(reference_skipped)} reference skipped)"
    )
    if unresolved:
        print(
            f"WARNING: {len(unresolved)} Magics colour name(s) still unresolved "
            f"(palettes truncated): {', '.join(unresolved)}"
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
