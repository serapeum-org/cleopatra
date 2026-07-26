"""Tests for `cleopatra.perceptual` -- the numpy-only perceptual colour toolkit."""
import numpy as np
import pytest
from matplotlib.colors import Colormap, LinearSegmentedColormap

from cleopatra.perceptual import (
    interp_perceptual,
    lab_to_srgb,
    make_categorical,
    make_diverging,
    perceptual_colormap,
    perceptual_uniformity,
    srgb_to_lab,
)


class TestColorSpace:
    """The sRGB <-> CIELAB transform."""

    def test_white_maps_to_l100(self):
        """Pure white has lightness L*=100 and no chroma."""
        assert np.allclose(srgb_to_lab(np.array([1.0, 1.0, 1.0])), [100, 0, 0], atol=1e-3)

    def test_black_maps_to_l0(self):
        """Pure black has lightness L*=0."""
        assert np.allclose(srgb_to_lab(np.array([0.0, 0.0, 0.0])), [0, 0, 0], atol=1e-6)

    def test_round_trip_identity(self):
        """srgb -> lab -> srgb reproduces the input across the gamut."""
        rng = np.random.default_rng(0)
        rgb = rng.random((500, 3))
        assert np.allclose(lab_to_srgb(srgb_to_lab(rgb)), rgb, atol=1e-6)

    def test_out_of_gamut_lab_is_clipped(self):
        """A Lab colour outside the sRGB gamut clamps into [0, 1], never wraps."""
        out = lab_to_srgb(np.array([50.0, 200.0, -200.0]))
        assert np.all((out >= 0.0) & (out <= 1.0))

    def test_transform_is_vectorised(self):
        """The transform preserves the leading shape of the input."""
        assert srgb_to_lab(np.zeros((4, 5, 3))).shape == (4, 5, 3)


class TestInterpPerceptual:
    """Perceptual (Lab, arc-length) interpolation of anchors."""

    ANCHORS = ["#ffffff", "#ff5fc9", "#c400a0", "#200018"]

    def test_shape_and_range(self):
        """Returns an (n, 3) LUT with every channel in [0, 1]."""
        lut = interp_perceptual(self.ANCHORS, n=64)
        assert lut.shape == (64, 3)
        assert np.all((lut >= 0.0) & (lut <= 1.0))

    def test_endpoints_are_exact(self):
        """The first and last output colours equal the first/last anchors exactly."""
        lut = interp_perceptual(self.ANCHORS, n=32)
        assert np.allclose(lut[0], [1.0, 1.0, 1.0])
        assert np.allclose(lut[-1], [0.125, 0.0, 0.094], atol=0.02)

    def test_more_uniform_than_rgb(self):
        """Lab interpolation is more perceptually even than RGB interpolation."""
        lab_cv = perceptual_uniformity(interp_perceptual(self.ANCHORS))
        rgb_cv = perceptual_uniformity(
            LinearSegmentedColormap.from_list("r", self.ANCHORS)(np.linspace(0, 1, 256))[:, :3]
        )
        assert lab_cv < rgb_cv

    def test_identical_anchors_give_constant_ramp(self):
        """Two identical anchors produce a single-colour LUT without dividing by zero."""
        lut = interp_perceptual(["#336699", "#336699"], n=10)
        assert np.allclose(lut, lut[0])

    @pytest.mark.parametrize("bad", [(["#000000"], 8), (["#000", "#fff"], 1)])
    def test_invalid_input_raises(self, bad):
        """Fewer than two anchors, or n < 2, raise ValueError."""
        anchors, n = bad
        with pytest.raises(ValueError):
            interp_perceptual(anchors, n=n)


class TestPerceptualColormap:
    """The `perceptual_colormap` convenience wrapper."""

    def test_returns_named_colormap(self):
        """Returns a Colormap carrying the requested name."""
        cmap = perceptual_colormap("mine", ["#ffffff", "#004cff"])
        assert isinstance(cmap, Colormap)
        assert cmap.name == "mine"

    def test_endpoints_match_anchors(self):
        """cmap(0.0)/cmap(1.0) are the exact endpoint anchors."""
        cmap = perceptual_colormap("m", ["#ffffff", "#123456"])
        assert tuple(float(round(v, 6)) for v in cmap(0.0)[:3]) == (1.0, 1.0, 1.0)
        assert np.allclose(cmap(1.0)[:3], [0x12 / 255, 0x34 / 255, 0x56 / 255], atol=1e-6)


class TestMakeDiverging:
    """Natively-built diverging colormaps."""

    def test_returns_colormap_of_requested_size(self):
        """Returns a Colormap with the requested number of levels."""
        cmap = make_diverging("#762a83", "#1b7837", n=128)
        assert isinstance(cmap, Colormap)
        assert cmap.N == 128

    def test_centre_is_lighter_than_ends(self):
        """The neutral centre is lighter (higher L*) than either end."""
        cmap = make_diverging("#762a83", "#1b7837")
        l_lo = srgb_to_lab(cmap(0.0)[:3])[0]
        l_mid = srgb_to_lab(cmap(0.5)[:3])[0]
        l_hi = srgb_to_lab(cmap(1.0)[:3])[0]
        assert l_mid > l_lo and l_mid > l_hi

    def test_balance_equalises_endpoint_lightness(self):
        """With balance=True the two ends share (near) equal lightness."""
        cmap = make_diverging("#762a83", "#1b7837", balance=True)
        l_lo = srgb_to_lab(cmap(0.0)[:3])[0]
        l_hi = srgb_to_lab(cmap(1.0)[:3])[0]
        assert abs(l_lo - l_hi) < 2.0

    def test_unbalanced_can_differ(self):
        """Without balancing, hues of unequal natural lightness stay unequal."""
        cmap = make_diverging("#762a83", "#1b7837", balance=False)
        l_lo = srgb_to_lab(cmap(0.0)[:3])[0]
        l_hi = srgb_to_lab(cmap(1.0)[:3])[0]
        assert abs(l_lo - l_hi) > 2.0


class TestMakeCategorical:
    """glasbey-style categorical palette generation."""

    def test_count_and_distinct(self):
        """Returns exactly n distinct hex colours."""
        cols = make_categorical(10)
        assert len(cols) == 10
        assert len(set(cols)) == 10
        assert all(c.startswith("#") for c in cols)

    def test_deterministic(self):
        """The greedy selection is deterministic across separate calls."""
        first = make_categorical(12)
        second = make_categorical(12)
        assert first == second

    def test_colours_are_well_separated(self):
        """The minimum pairwise Lab distance clears a distinguishability floor."""
        lab = srgb_to_lab(np.array([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]
                                    for c in make_categorical(10)]))
        dists = [np.sqrt(((lab[i] - lab[j]) ** 2).sum())
                 for i in range(len(lab)) for j in range(i + 1, len(lab))]
        assert min(dists) > 20.0

    def test_invalid_count_raises(self):
        """n < 1 raises ValueError."""
        with pytest.raises(ValueError):
            make_categorical(0)


class TestUniformityMetric:
    """The perceptual-uniformity diagnostic."""

    def test_accepts_colormap_and_lut(self):
        """Works on both a Colormap and a raw (m, 3) LUT, returning a float."""
        cmap = perceptual_colormap("p", ["#ffffff", "#004cff", "#000033"])
        assert isinstance(perceptual_uniformity(cmap), float)
        assert isinstance(perceptual_uniformity(cmap(np.linspace(0, 1, 64))[:, :3]), float)

    def test_zero_for_even_ramp(self):
        """A Lab-uniform ramp scores near zero (very even steps)."""
        assert perceptual_uniformity(interp_perceptual(["#ffffff", "#000000"])) < 0.05
