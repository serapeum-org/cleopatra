"""Error-path rollback of the sticky `default_options` on the primitive glyphs.

A glyph's `plot()` merges grouped parameter objects (`color=`, `contour=`,
`classify=`) into its persistent `default_options` before rendering. If the
render then raises (e.g. an unsupported classification scheme), the merged
options must be rolled back so a later plain `plot()` on the same instance is
not bricked and does not silently render with a colour scale that was never
successfully applied. These tests pin that contract for the four primitive
glyphs, which gained the shared `rollback_options_on_error` guard.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cleopatra.glyphs.gridded.vector_glyph import VectorGlyph  # noqa: E402
from cleopatra.glyphs.primitives.flow_glyph import FlowGlyph  # noqa: E402
from cleopatra.glyphs.primitives.polygon_glyph import PolygonGlyph  # noqa: E402
from cleopatra.glyphs.primitives.scatter_glyph import ScatterGlyph  # noqa: E402
from cleopatra.styling.params import Classify  # noqa: E402
from cleopatra.styling.scaling import ColorScaling  # noqa: E402


def _flow() -> FlowGlyph:
    """A scalar-valued `FlowGlyph` (two paths, two magnitudes)."""
    paths = [np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[0.0, 1.0], [1.0, 0.0]])]
    return FlowGlyph(paths, values=np.array([1.0, 2.0]))


def _scatter() -> ScatterGlyph:
    """A scalar-valued `ScatterGlyph` (three points, three values)."""
    return ScatterGlyph(
        [0.0, 1.0, 2.0], [3.0, 4.0, 5.0], values=np.array([1.0, 2.0, 3.0])
    )


def _polygon() -> PolygonGlyph:
    """A scalar-valued `PolygonGlyph` (two triangles, two values)."""
    polygons = [
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
    ]
    return PolygonGlyph(polygons, values=np.array([10.0, 20.0]))


def _vector() -> VectorGlyph:
    """A `VectorGlyph` over a 3x3 field coloured by magnitude."""
    x, y = np.meshgrid(np.arange(3.0), np.arange(3.0))
    return VectorGlyph(x, y, np.full_like(x, 3.0), np.full_like(y, 4.0))


_FACTORIES = {
    "flow": _flow,
    "scatter": _scatter,
    "polygon": _polygon,
    "vector": _vector,
}


@pytest.mark.parametrize("name", list(_FACTORIES))
def test_failed_scheme_does_not_poison_default_options(name):
    """A rejected `classify` scheme rolls back so a later plain plot() works."""
    glyph = _FACTORIES[name]()
    before = glyph.default_options.get("scheme")
    bad = Classify(scheme="not_a_scheme")
    with pytest.raises(ValueError):
        glyph.plot(classify=bad)
    assert glyph.default_options.get("scheme") == before, (
        f"{name}: a failed classify scheme must roll back to its default"
    )
    glyph.plot()  # not bricked
    plt.close("all")


@pytest.mark.parametrize("name", list(_FACTORIES))
def test_failed_plot_rolls_back_co_passed_color(name):
    """A failed styled plot must not leak a co-passed color= into later plots."""
    glyph = _FACTORIES[name]()
    power = ColorScaling.power(gamma=0.7)
    bad = Classify(scheme="not_a_scheme")
    with pytest.raises(ValueError):
        glyph.plot(color=power, classify=bad)
    assert glyph.default_options["color_scale"] == "linear", (
        f"{name}: a failed plot must roll back the co-passed colour scale"
    )
    assert glyph.default_options["gamma"] == 0.5, (
        f"{name}: gamma must roll back to its default"
    )
    plt.close("all")
