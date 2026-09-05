"""Tests for `cleopatra.glyphs.base.compositing`."""

import numpy as np
import pytest

from cleopatra.glyphs.base.compositing import alpha_over


class TestAlphaOver:
    """Tests for the `alpha_over` Porter-Duff "over" primitive."""

    def test_rgb_background_blends_and_drops_alpha(self):
        """Half-transparent red over an opaque blue canvas blends 50/50 to a 3-band result.

        Test scenario:
            Foreground alpha 0.5 over an RGB background yields
            `fg_rgb * a + bg * (1 - a)` with no alpha channel.
        """
        fg = np.array([[[1.0, 0.0, 0.0, 0.5]]])
        bg = np.array([[[0.0, 0.0, 1.0]]])
        out = alpha_over(fg, bg)
        assert out.shape == (1, 1, 3), f"RGB background must give a 3-band result, got {out.shape}"
        assert np.allclose(out.ravel(), [0.5, 0.0, 0.5]), f"Unexpected blend: {out.ravel()}"

    def test_rgba_background_keeps_alpha(self):
        """The same mark over an RGBA canvas keeps a 4-band result with combined coverage.

        Test scenario:
            Foreground alpha 0.5 over an opaque RGBA background gives out_a == 1.0
            and the same colour blend, retaining the alpha channel.
        """
        fg = np.array([[[1.0, 0.0, 0.0, 0.5]]])
        bg = np.array([[[0.0, 0.0, 1.0, 1.0]]])
        out = alpha_over(fg, bg)
        assert out.shape == (1, 1, 4), f"RGBA background must give a 4-band result, got {out.shape}"
        assert np.allclose(out.ravel(), [0.5, 0.0, 0.5, 1.0]), f"Unexpected blend: {out.ravel()}"

    def test_opaque_foreground_covers_background(self):
        """A fully opaque foreground reproduces its own colour, ignoring the background.

        Test scenario:
            Foreground alpha 1.0 over any background returns the foreground RGB.
        """
        fg = np.array([[[0.2, 0.4, 0.6, 1.0]]])
        bg = np.array([[[1.0, 1.0, 1.0]]])
        out = alpha_over(fg, bg)
        assert np.allclose(out.ravel(), [0.2, 0.4, 0.6]), f"Opaque foreground should win, got {out.ravel()}"

    def test_transparent_foreground_shows_background(self):
        """A fully transparent foreground leaves an RGB background untouched.

        Test scenario:
            Foreground alpha 0.0 over an RGB background returns the background RGB.
        """
        fg = np.array([[[1.0, 1.0, 1.0, 0.0]]])
        bg = np.array([[[0.1, 0.2, 0.3]]])
        out = alpha_over(fg, bg)
        assert np.allclose(out.ravel(), [0.1, 0.2, 0.3]), f"Transparent foreground should reveal bg, got {out.ravel()}"

    def test_transparent_over_transparent_is_guarded(self):
        """Fully transparent over fully transparent stays black rather than dividing by zero.

        Test scenario:
            Both alphas 0 makes out_a == 0; the divide-by-zero guard keeps the
            pixel at all-zeros with no NaN.
        """
        out = alpha_over(np.zeros((1, 1, 4)), np.zeros((1, 1, 4)))
        assert np.allclose(out.ravel(), [0.0, 0.0, 0.0, 0.0]), f"Zero-coverage pixel should stay zero, got {out.ravel()}"
        assert not np.isnan(out).any(), "Guard must prevent NaN from a zero output alpha"

    def test_guard_only_touches_zero_coverage_pixels(self):
        """A mix of covered and uncovered pixels guards only the uncovered one.

        Test scenario:
            One pixel has zero foreground and background alpha (stays zero); the
            neighbour blends normally, proving the guard is per-pixel.
        """
        fg = np.array([[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.5]]])
        bg = np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
        out = alpha_over(fg, bg)
        assert np.allclose(out[0, 0], [0.0, 0.0, 0.0, 0.0]), f"Uncovered pixel should stay zero, got {out[0, 0]}"
        assert np.allclose(out[0, 1], [0.5, 0.0, 0.5, 1.0]), f"Covered pixel should blend, got {out[0, 1]}"

    def test_partial_over_partial_un_premultiplies(self):
        """A partial-alpha mark over a partial-alpha canvas un-premultiplies the colour.

        Test scenario:
            fg alpha 0.5 over bg alpha 0.5 gives out_a 0.75 and colours divided
            back out by that coverage, matching the hand-computed Porter-Duff result.
        """
        fg = np.array([[[1.0, 0.0, 0.0, 0.5]]])
        bg = np.array([[[0.0, 1.0, 0.0, 0.5]]])
        out = alpha_over(fg, bg)
        assert np.allclose(out.ravel(), [2 / 3, 1 / 3, 0.0, 0.75]), f"Un-premultiplied blend wrong: {out.ravel()}"

    def test_per_pixel_alpha_broadcasts_across_channels(self):
        """Different foreground alphas blend their own pixel across all three colours.

        Test scenario:
            A two-pixel RGB composite applies each pixel's alpha to its RGB triple
            independently, confirming the (H, W, 1) alpha broadcasts correctly.
        """
        fg = np.array([[[1.0, 0.0, 0.0, 0.25], [0.0, 1.0, 0.0, 0.75]]])
        bg = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        out = alpha_over(fg, bg)
        assert np.allclose(out[0, 0], [0.25, 0.0, 0.0]), f"First pixel alpha misapplied: {out[0, 0]}"
        assert np.allclose(out[0, 1], [0.0, 0.75, 0.0]), f"Second pixel alpha misapplied: {out[0, 1]}"

    def test_integer_inputs_are_cast_to_float(self):
        """Integer arrays are accepted and produce a float result.

        Test scenario:
            An opaque integer foreground over an integer RGB background returns a
            float array without integer truncation.
        """
        fg = np.array([[[1, 0, 0, 1]]], dtype=int)
        bg = np.array([[[0, 0, 1]]], dtype=int)
        out = alpha_over(fg, bg)
        assert out.dtype == np.dtype(float), f"Result should be float, got {out.dtype}"
        assert np.allclose(out.ravel(), [1.0, 0.0, 0.0]), f"Integer opaque foreground should win, got {out.ravel()}"

    def test_rgba_result_is_float(self):
        """The RGBA path returns a float array.

        Test scenario:
            An RGBA-over-RGBA composite yields a floating-point result regardless
            of input dtype.
        """
        out = alpha_over(np.zeros((2, 2, 4)), np.ones((2, 2, 4)))
        assert out.dtype == np.dtype(float), f"RGBA result should be float, got {out.dtype}"

    def test_float32_inputs_preserve_precision(self):
        """A float32 foreground and background composite to a float32 result.

        Test scenario:
            Floating-point inputs keep their own width, so a float32 pair stays
            float32 rather than being upcast to float64.
        """
        fg = np.zeros((2, 2, 4), dtype=np.float32)
        bg = np.ones((2, 2, 4), dtype=np.float32)
        out = alpha_over(fg, bg)
        assert out.dtype == np.float32, f"float32 inputs should stay float32, got {out.dtype}"

    def test_opaque_foreground_over_rgba_reports_full_coverage(self):
        """An opaque foreground over an RGBA canvas yields its colour at full alpha.

        Test scenario:
            Foreground alpha 1.0 over a partially transparent RGBA background gives
            out_a == 1.0 and the foreground RGB, exercising the RGBA branch at the
            alpha extreme where the background contributes nothing.
        """
        fg = np.array([[[0.2, 0.4, 0.6, 1.0]]])
        bg = np.array([[[0.9, 0.9, 0.9, 0.3]]])
        out = alpha_over(fg, bg)
        assert np.allclose(out.ravel(), [0.2, 0.4, 0.6, 1.0]), f"Opaque fg over RGBA should win at alpha 1, got {out.ravel()}"

    def test_does_not_mutate_inputs(self):
        """Compositing leaves both input arrays unchanged.

        Test scenario:
            alpha_over is pure -- neither the foreground nor the background array
            is modified in place by the blend.
        """
        fg = np.array([[[1.0, 0.0, 0.0, 0.5]]])
        bg = np.array([[[0.0, 0.0, 1.0, 1.0]]])
        fg_before = fg.copy()
        bg_before = bg.copy()
        alpha_over(fg, bg)
        assert np.array_equal(fg, fg_before), "Foreground array must not be mutated"
        assert np.array_equal(bg, bg_before), "Background array must not be mutated"

    def test_accepts_array_like_inputs(self):
        """Nested Python lists are coerced to arrays before compositing.

        Test scenario:
            Passing plain lists (not np.ndarray) is accepted via np.asarray and
            produces the same blend as equivalent array inputs.
        """
        out = alpha_over([[[1.0, 0.0, 0.0, 0.5]]], [[[0.0, 0.0, 1.0]]])
        assert np.allclose(out.ravel(), [0.5, 0.0, 0.5]), f"List inputs should blend like arrays, got {out.ravel()}"

    @pytest.mark.parametrize(
        "foreground, background, match",
        [
            (np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), "foreground must be"),
            (np.zeros((2, 2)), np.zeros((2, 2, 3)), "foreground must be"),
            (np.zeros((1, 1, 4)), np.zeros((1, 1, 2)), "background must be"),
            (np.zeros((1, 1, 4)), np.zeros((1, 1)), "background must be"),
            (np.zeros((2, 2, 4)), np.zeros((2, 3, 3)), "share the same"),
        ],
    )
    def test_invalid_shapes_raise(self, foreground, background, match):
        """Malformed foreground, background, or mismatched sizes raise ValueError.

        Args:
            foreground: The foreground array under test.
            background: The background array under test.
            match: A substring expected in the error message.

        Test scenario:
            Each shape violation (non-RGBA foreground, wrong-band background,
            differing height/width) raises ValueError naming the offending array.
        """
        with pytest.raises(ValueError, match=match):
            alpha_over(foreground, background)
