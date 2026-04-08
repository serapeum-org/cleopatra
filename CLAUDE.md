# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cleopatra is a matplotlib utility package for visualizing 2D/3D numpy arrays, with support for plotting, animation, and statistical histograms. It targets scientific/research users working with geospatial and raster data. Licensed under GPLv3.

## Commands

### Install
```bash
pip install -e .[dev]
```

### Run all tests
```bash
pytest -vvv --cov=src/cleopatra --cov-report=term-missing
```

### Run a single test file or test
```bash
pytest tests/test_colors.py -v
pytest tests/test_array_glyph.py::TestClassName::test_method -v
```

### Run tests by marker
```bash
pytest -m "not plot"    # skip plot tests
pytest -m fast          # only fast tests
```

### Validate notebooks
```bash
pytest --nbval-lax
```

### Run doctests
```bash
python -c "import matplotlib; matplotlib.use('Agg'); import pytest; pytest.main(['--doctest-modules'])"
```

### Pre-commit hooks
```bash
pre-commit run --all-files
```

### Build docs
```bash
mkdocs serve   # local preview
mkdocs build   # build static site
```

## Architecture

**Layout**: `src/` layout with package at `src/cleopatra/`.

### Core Classes

- **ArrayGlyph** (`array_glyph.py`): Main class for 2D/3D array visualization. Handles plotting with colorbars, color scales (linear, power, sym-lognorm, boundary-norm, midpoint), cell value display, point overlays, and animation (gif/mp4/mov/avi via ffmpeg). Depends on `hpc-utils` for index operations.
- **Colors** (`colors.py`): Color format handling — converts between hex, RGB (0-255), and normalized RGB (0-1). Can extract color ramps from images and create custom matplotlib colormaps.
- **StatisticalGlyph** (`statistical_glyph.py`): Histogram visualization for 1D and 2D datasets.
- **Styles** (`styles.py`): Predefined line styles, marker styles, and `MidpointNormalize` (custom matplotlib norm used by ArrayGlyph).
- **Config** (`config.py`): Matplotlib backend auto-detection and configuration. Automatically called at package import.

### Key Patterns

- `DEFAULT_OPTIONS` dicts are defined per module and merged (`STYLE_DEFAULTS | DEFAULT_OPTIONS` in array_glyph.py) to compose configuration.
- `conftest.py` sets `Config.set_matplotlib_backend(backend="Agg")` for headless test rendering. All test fixtures are module-scoped and load data from `tests/data/`.
- Docstrings use **Google style** with embedded doctests using markdown fenced code blocks.

## Code Style

- **Formatter**: black (line length 88)
- **Import sorting**: isort
- **Linter**: flake8 (ignores E203, E266, E501, W503, D403, D414, C901, E731)
- **Security**: bandit (B101 skipped), gitleaks, detect-secrets, truffleHog, checkov
- **Commit messages**: Conventional commits enforced by commitizen (`type(scope): message`). Summary must be imperative, capitalized, no trailing punctuation.
- **Pre-commit** blocks direct commits to `main`.
- Python >=3.11 required.
