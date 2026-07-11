#!/usr/bin/env python3
"""Build cleopatra's cmocean preset asset (maintainer-side, offline).

This script is **not** part of the shipped package and is **not** run at
import or install time. It is the one-off, maintainer-machine tool that
derives ``src/cleopatra/data/cmocean_presets.json`` from the ``cmocean``
oceanography colormap collection (MIT, Copyright (c) 2015 Kristen M. Thyng;
see ``src/cleopatra/data/CMOCEAN_NOTICE.txt``).

Each curated cmocean colormap is sampled to hex control points and paired with
an ocean/hydrology/DEM variable label and an opacity policy (all opaque -- these
are full-field colormaps). The diverging maps (sea-level anomaly, vorticity) and
the land+sea topography map carry ``center=0`` so ``apply_data_style`` renders
them symmetric about zero -- placing the colormap midpoint (and, for topography,
the coastline) on 0.

Maintainer dependency: ``cmocean`` (install in a throwaway env, *not* in
cleopatra; it is not a runtime dependency).

Re-run (from the repo root)::

    python tools/build_cmocean_presets.py src/cleopatra/data/cmocean_presets.json
"""
import datetime as _dt
import json
import sys

import cmocean
import numpy as np
from matplotlib.colors import to_hex

#: How many control points to sample per colormap. cmocean maps are smooth and
#: perceptually uniform, so 64 points reproduce them faithfully for display.
N_POINTS = 64

#: (cmocean map, preset key, legend label, diverging center or None).
CURATED = [
    ("thermal", "sea_surface_temperature", "Sea surface temperature", None),
    ("haline",  "salinity",                "Salinity", None),
    ("deep",    "bathymetry",              "Ocean depth", None),
    ("topo",    "topography",              "Topography (land & sea)", 0.0),
    ("turbid",  "turbidity",               "Turbidity / sediment", None),
    ("speed",   "current_speed",           "Current speed", None),
    ("dense",   "water_density",           "Water density", None),
    ("algae",   "chlorophyll",             "Chlorophyll", None),
    ("oxy",     "dissolved_oxygen",        "Dissolved oxygen", None),
    ("ice",     "sea_ice",                 "Sea ice", None),
    ("solar",   "solar_radiation",         "Solar radiation", None),
    ("rain",    "rainfall",                "Rainfall", None),
    ("phase",   "phase",                   "Phase / direction (cyclic)", None),
    ("balance", "sea_level_anomaly",       "Sea-level anomaly", 0.0),
    ("curl",    "vorticity",               "Vorticity", 0.0),
]


def main(out_path):
    xs = np.linspace(0.0, 1.0, N_POINTS)
    presets = {}
    for cm_name, key, label, center in CURATED:
        cmap = getattr(cmocean.cm, cm_name)
        palette = [to_hex(cmap(float(x))) for x in xs]
        rec = {"label": label, "palette": palette, "opacity": "opaque", "cmocean_map": cm_name}
        if center is not None:
            rec["center"] = center
        presets[key] = rec

    asset = {
        "_meta": {
            "source": "cmocean",
            "source_version": cmocean.__version__,
            "license": "MIT",
            "copyright": "Copyright (c) 2015 Kristen M. Thyng",
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d"),
            "n_points": N_POINTS,
            "note": (
                "Colour data derived from the cmocean collection (MIT). Each "
                "curated colormap is sampled to hex control points and exposed as "
                "an ocean/hydrology/DEM preset that auto-ranges. Diverging maps and "
                "the land+sea topography map carry center=0."
            ),
        },
        "presets": dict(sorted(presets.items())),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asset, f, indent=1, ensure_ascii=False)
    print(f"wrote {len(presets)} cmocean presets to {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
