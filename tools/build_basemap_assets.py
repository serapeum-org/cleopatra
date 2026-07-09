#!/usr/bin/env python3
"""Build cleopatra's reference-basemap assets (maintainer-side, offline).

This script is **not** part of the shipped package and is **not** run at
import or install time. It is the one-off, maintainer-machine tool that
converts upstream public datasets into the lightweight, dependency-free
artifacts that ``cleopatra.reference`` downloads and reads at runtime:

* **Natural Earth vectors** -> gzipped GeoJSON
  (``ne_<res>_<stem>.geojson.gz``), read at runtime with stdlib ``json``.
  Conversion happens here, where heavy GIS deps (geopandas/GDAL) are fine;
  the published artifact carries none of them.
* **Hypsometric relief** -> plain RGB PNG
  (``ne_hypso_rgb_<W>x<H>.png``), read at runtime with Pillow only. The
  geotransform is intentionally dropped: every product is a global
  EPSG:4326 raster, so ``cleopatra.reference`` hardcodes the extent.

Outputs land in ``--out-dir`` with exactly the filenames the runtime
helpers expect, ready to attach to a cleopatra release
(``basemap-data-v1``).

Maintainer dependencies (install in a throwaway env, *not* in cleopatra)::

    pip install geopandas shapely pillow

Examples::

    # Build everything (downloads Natural Earth zips from naciscdn.org).
    python tools/build_basemap_assets.py --out-dir dist/basemap

    # Vectors only, just the 110m + 50m resolutions, more aggressive
    # simplification to shrink the artifacts.
    python tools/build_basemap_assets.py --out-dir dist/basemap \
        --skip-relief --resolutions 110m 50m --simplify 0.02

    # Relief only, from local source GeoTIFFs you already downloaded.
    python tools/build_basemap_assets.py --out-dir dist/basemap \
        --skip-vectors --relief-src ./tif
"""
from __future__ import annotations

import argparse
import gzip
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

# --- runtime-mirrored maps (keep in sync with cleopatra/reference.py) -------

#: layer -> (Natural Earth category, dataset stem). Geometry kind is
#: irrelevant here -- the runtime infers it; we only need the source path.
_LAYERS = {
    "coastline": ("physical", "coastline"),
    "land": ("physical", "land"),
    "ocean": ("physical", "ocean"),
    "rivers": ("physical", "rivers_lake_centerlines"),
    "lakes": ("physical", "lakes"),
    "borders": ("cultural", "admin_0_boundary_lines_land"),
}
_RESOLUTIONS = ("110m", "50m", "10m")

_NE_BASE_URL = "https://naciscdn.org/naturalearth"

#: Relief products: output PNG name -> source GeoTIFF name. The source TIFFs
#: currently live on the *pyramids* basemap-data-v1 release; this script
#: re-emits them as PNG so the cleopatra asset owns no GeoTIFF.
_RELIEF_PRODUCTS = {
    "ne_hypso_rgb_720x360.png": "ne_hypso_rgb_720x360.tif",
    "ne_hypso_rgb_1440x720.png": "ne_hypso_rgb_1440x720.tif",
}
_RELIEF_SRC_URL = (
    "https://github.com/serapeum-org/pyramids/releases/download/basemap-data-v1/"
)

#: Per-resolution simplification tolerance (degrees) and coordinate grid
#: (degrees) applied before export. Coarser resolutions need almost no
#: help; 10m is where the savings matter. Overridable via --simplify /
#: --precision.
_DEFAULT_SIMPLIFY = {"110m": 0.0, "50m": 0.005, "10m": 0.01}
_DEFAULT_PRECISION = {"110m": 0.001, "50m": 0.0005, "10m": 0.0001}

USER_AGENT = "cleopatra-asset-builder (+https://github.com/serapeum-org/cleopatra)"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` (skipping if already present). http(s)."""
    if dest.exists():
        _log(f"  cached  {dest.name}")
        return dest
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"Refusing non-http(s) URL: {url!r}")
    _log(f"  fetch   {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
    except (urllib.error.URLError, OSError) as e:
        raise ConnectionError(f"Failed to download {url!r}: {e}") from e
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


# --- vectors ----------------------------------------------------------------


def build_vectors(
    out_dir: Path,
    cache_dir: Path,
    resolutions: list[str],
    layers: list[str],
    simplify: dict[str, float],
    precision: dict[str, float],
) -> None:
    """Convert Natural Earth shapefiles to gzipped GeoJSON in ``out_dir``."""
    import geopandas as gpd
    from shapely import set_precision

    for resolution in resolutions:
        for layer in layers:
            category, stem = _LAYERS[layer]
            zip_name = f"ne_{resolution}_{stem}.zip"
            url = f"{_NE_BASE_URL}/{resolution}/{category}/{zip_name}"
            out_name = f"ne_{resolution}_{stem}.geojson.gz"
            _log(f"[vector] {resolution}/{layer} -> {out_name}")

            zip_path = _download(url, cache_dir / zip_name)
            gdf = gpd.read_file(f"zip://{zip_path}")

            # Keep geometry only; runtime needs no attributes. Reproject to
            # EPSG:4326 defensively (Natural Earth already is, but be safe).
            gdf = gdf[[gdf.geometry.name]]
            if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(4326)

            tol = simplify.get(resolution, 0.0)
            if tol > 0:
                gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)
            grid = precision.get(resolution, 0.0)
            if grid > 0:
                gdf["geometry"] = set_precision(gdf.geometry.values, grid)
                gdf = gdf[~gdf.geometry.is_empty]

            # to_json -> a GeoJSON FeatureCollection string. drop_id keeps it
            # small; the runtime only reads .features[].geometry.
            geojson = gdf.to_json(drop_id=True)
            out_path = out_dir / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            raw = geojson.encode("utf-8")
            with gzip.open(out_path, "wb", compresslevel=9) as fh:
                fh.write(raw)
            _log(
                f"         {len(gdf)} features, "
                f"{len(raw) / 1e6:.1f} MB -> {out_path.stat().st_size / 1e6:.2f} MB gz"
            )


# --- relief -----------------------------------------------------------------


def build_relief(out_dir: Path, cache_dir: Path, relief_src: Path | None) -> None:
    """Convert relief GeoTIFFs to RGB PNGs in ``out_dir``.

    ``relief_src`` is a directory of source ``.tif`` files; if ``None`` the
    sources are downloaded from the pyramids basemap-data-v1 release.
    """
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    for png_name, tif_name in _RELIEF_PRODUCTS.items():
        _log(f"[relief] {tif_name} -> {png_name}")
        if relief_src is not None:
            tif_path = relief_src / tif_name
            if not tif_path.exists():
                raise FileNotFoundError(f"Missing relief source: {tif_path}")
        else:
            tif_path = _download(_RELIEF_SRC_URL + tif_name, cache_dir / tif_name)

        # Pillow reads the raster pixels and ignores the geo metadata, which
        # is exactly what we want: a plain north-up RGB image.
        with Image.open(tif_path) as img:
            rgb = img.convert("RGB")
            out_path = out_dir / png_name
            rgb.save(out_path, format="PNG", optimize=True)
        _log(
            f"         {rgb.size[0]}x{rgb.size[1]} -> "
            f"{out_path.stat().st_size / 1e6:.2f} MB"
        )


# --- cli --------------------------------------------------------------------


def _parse_kv(values: list[str] | None, base: dict[str, float]) -> dict[str, float]:
    """Merge ``res=value`` overrides (or a single scalar) into ``base``."""
    if not values:
        return dict(base)
    merged = dict(base)
    for item in values:
        if "=" in item:
            res, val = item.split("=", 1)
            merged[res] = float(val)
        else:  # a bare scalar applies to every resolution
            merged = {res: float(item) for res in merged}
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Directory for built assets."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Where to cache downloaded sources (default: a temp dir).",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=list(_RESOLUTIONS),
        choices=_RESOLUTIONS,
        help="Natural Earth resolutions to build.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=list(_LAYERS),
        choices=list(_LAYERS),
        help="Natural Earth layers to build.",
    )
    parser.add_argument(
        "--simplify",
        nargs="+",
        default=None,
        metavar="RES=TOL|TOL",
        help="Simplify tolerance in degrees, per resolution or a single "
        "scalar (e.g. '10m=0.02' or '0.01').",
    )
    parser.add_argument(
        "--precision",
        nargs="+",
        default=None,
        metavar="RES=GRID|GRID",
        help="Coordinate grid size in degrees for quantization.",
    )
    parser.add_argument(
        "--relief-src",
        type=Path,
        default=None,
        help="Directory of source relief GeoTIFFs (default: download).",
    )
    parser.add_argument(
        "--skip-vectors", action="store_true", help="Do not build vector assets."
    )
    parser.add_argument(
        "--skip-relief", action="store_true", help="Do not build relief assets."
    )
    args = parser.parse_args(argv)

    simplify = _parse_kv(args.simplify, _DEFAULT_SIMPLIFY)
    precision = _parse_kv(args.precision, _DEFAULT_PRECISION)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = args.cache_dir if args.cache_dir is not None else Path(tmp)
        cache_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_vectors:
            build_vectors(
                out_dir,
                cache_dir,
                args.resolutions,
                args.layers,
                simplify,
                precision,
            )
        if not args.skip_relief:
            build_relief(out_dir, cache_dir, args.relief_src)

    _log(f"\nDone. Assets written to {out_dir.resolve()}")
    _log("Attach the contents to the cleopatra 'basemap-data-v1' release.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
