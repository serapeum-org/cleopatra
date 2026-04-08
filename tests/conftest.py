from __future__ import annotations

import numpy as np
import pytest

from cleopatra.config import Config

Config.set_matplotlib_backend(backend="Agg")


@pytest.fixture(scope="module")
def arr() -> np.ndarray:
    return np.load("tests/data/arr.npy")


@pytest.fixture(scope="module")
def no_data_value(arr: np.ndarray) -> float:
    return arr[0, 0]


@pytest.fixture(scope="module")
def rhine_dem_arr() -> np.ndarray:
    return np.load("tests/data/DEM5km_Rhine_burned_fill.npy")


@pytest.fixture(scope="module")
def rhine_no_data_val(rhine_dem_arr: np.ndarray) -> float:
    return rhine_dem_arr[0, 0]


@pytest.fixture(scope="module")
def cmap() -> str:
    return "terrain"


@pytest.fixture(scope="module")
def color_scale() -> list[str]:
    return ["linear", "power", "sym-lognorm", "boundary-norm", "midpoint"]


@pytest.fixture(scope="module")
def ticks_spacing() -> int:
    return 500


@pytest.fixture(scope="module")
def color_scale_2_gamma() -> float:
    return 0.5


@pytest.fixture(scope="module")
def color_scale_3_linscale() -> float:
    return 0.001


@pytest.fixture(scope="module")
def color_scale_3_linthresh() -> float:
    return 0.0001


@pytest.fixture(scope="module")
def bounds() -> list:
    return [-559, 0, 440, 940, 1440, 1940, 2440, 2940, 3500]


@pytest.fixture(scope="module")
def midpoint() -> int:
    return 20


@pytest.fixture(scope="module")
def display_cell_value() -> bool:
    return True


@pytest.fixture(scope="module")
def num_size() -> int:
    return 8


@pytest.fixture(scope="module")
def background_color_threshold():
    return None


@pytest.fixture(scope="module")
def gauge_size() -> int:
    return 100


@pytest.fixture(scope="module")
def gauge_color() -> str:
    return "blue"


@pytest.fixture(scope="module")
def points():
    return np.loadtxt("tests/data/points.csv", skiprows=1, delimiter=",")


@pytest.fixture(scope="module")
def id_size() -> int:
    return 20


@pytest.fixture(scope="module")
def id_color() -> str:
    return "green"


@pytest.fixture(scope="module")
def point_size() -> int:
    return 100


@pytest.fixture(scope="module")
def coello_data() -> np.ndarray:
    return np.load("tests/data/coello.npy")


@pytest.fixture(scope="module")
def animate_time_list() -> list:
    return list(range(1, 11))


@pytest.fixture(scope="module")
def sentinel_2() -> np.ndarray:
    return np.load("tests/data/s2a.npy")


@pytest.fixture(scope="module")
def color_ramp_image() -> str:
    return "tests/data/colors/color-ramp.png"
