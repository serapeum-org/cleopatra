# Installation

## Dependencies

### Required dependencies

- Python (3.11 or later)
- [numpy](https://www.numpy.org/) (2.0.0 or later)
- [matplotlib](https://matplotlib.org/) (3.9 or later)
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python/) — for animation export to MP4 / MOV / AVI
- [hpc-utils](https://github.com/serapeum-org/hpc) (0.1.4 or later)

!!! note
    Writing animations to `mp4` / `mov` / `avi` also needs an **FFmpeg** binary on
    your `PATH`. GIF export uses matplotlib's `PillowWriter` and needs no FFmpeg.

### Optional dependencies — the `tiles` extra

The web-tile basemap helper (`cleopatra.tiles.add_tiles`) needs four extra packages.
They are bundled in the `cleopatra[tiles]` extra (pip) / the `cleopatra-tiles` package
(conda) and are otherwise not installed:

- [mercantile](https://github.com/mapbox/mercantile) (1.2.1 or later)
- [pillow](https://python-pillow.org/) (12.1.1 or later)
- [pyproj](https://pyproj4.github.io/pyproj/) (3.7.2 or later)
- [xyzservices](https://xyzservices.readthedocs.io/) (2026.3.0 or later)

It is recommended to install `cleopatra` into a virtual environment so its
requirements do not interfere with your system Python.

## conda

`cleopatra` is available in the [conda-forge](https://conda-forge.org/) channel:

```bash
conda install -c conda-forge cleopatra

# with the optional web-tile basemap support
conda install -c conda-forge cleopatra-tiles
```

The conda packages are built from the
[cleopatra-feedstock](https://github.com/conda-forge/cleopatra-feedstock); the
`cleopatra-tiles` output simply pulls in `cleopatra` plus the four `[tiles]`
dependencies.

## pip

```bash
pip install cleopatra

# with the optional web-tile basemap support (cleopatra.tiles.add_tiles)
pip install "cleopatra[tiles]"
```

## From sources

The sources for `cleopatra` are hosted on the
[GitHub repo](https://github.com/serapeum-org/cleopatra).

Install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/serapeum-org/cleopatra.git
```

Or clone and do an editable install for development:

```bash
git clone https://github.com/serapeum-org/cleopatra.git
cd cleopatra
pip install -e .
```

## Check the installation

To confirm the install succeeded, import the package and print its version:

```bash
python -c "import cleopatra; print(cleopatra.__version__)"
```

This should print the installed version without errors.

## Documentation

- Latest (development) docs: <https://serapeum-org.github.io/cleopatra/latest/>
- Versioned docs are published with [mike](https://github.com/jimporter/mike);
  use the version selector in the top bar to switch releases.
