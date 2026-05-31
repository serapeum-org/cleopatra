"""Tests for cleopatra.animation glyph-independent save/embed helpers.

Covers `save_animation`, `to_gif`, and `embed_gif` operating on a plain
`matplotlib.animation.FuncAnimation` (no `Glyph` involved). A tiny 2-3 frame
animation is rendered on the Agg backend (set globally via `MPLBACKEND`).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import pytest
from matplotlib.animation import FuncAnimation

from cleopatra.animation import (
    SUPPORTED_VIDEO_FORMAT,
    embed_gif,
    save_animation,
    to_gif,
)


@pytest.fixture
def tiny_anim():
    """A 3-frame line animation built directly from matplotlib (no Glyph)."""
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0, 0])

    def update(i):
        line.set_ydata([0, i])
        return (line,)

    anim = FuncAnimation(fig, update, frames=3, blit=True)
    yield anim
    plt.close(fig)


class TestSaveAnimation:
    """Tests for `save_animation`."""

    def test_gif_round_trips(self, tiny_anim, tmp_path):
        """A GIF is written to disk and is non-empty."""
        path = tmp_path / "out.gif"
        returned = save_animation(tiny_anim, str(path), fps=2)

        assert returned == str(path), "should return the path it wrote"
        assert path.exists(), "GIF file was not created"
        assert path.stat().st_size > 0, "GIF file is empty"
        # GIF magic number — confirms a real GIF, not a stray file.
        assert path.read_bytes()[:6] in (b"GIF87a", b"GIF89a")

    def test_unsupported_format_raises(self, tiny_anim, tmp_path):
        """An unsupported extension raises a clear ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            save_animation(tiny_anim, str(tmp_path / "out.webm"))

    def test_ffmpeg_missing_raises_friendly_error(self, tmp_path):
        """A missing FFmpeg binary surfaces as `FileNotFoundError` with URL."""
        anim = MagicMock(spec=FuncAnimation)
        anim.save = MagicMock(side_effect=FileNotFoundError("ffmpeg not found"))

        with pytest.raises(FileNotFoundError, match="ffmpeg.org"):
            save_animation(anim, str(tmp_path / "movie.mp4"))


class TestToGif:
    """Tests for `to_gif`."""

    def test_returns_non_empty_gif_bytes(self, tiny_anim):
        """`to_gif` returns in-memory GIF bytes with the GIF magic number."""
        data = to_gif(tiny_anim, fps=2)

        assert isinstance(data, bytes), "expected raw bytes"
        assert len(data) > 0, "GIF bytes are empty"
        assert data[:6] in (b"GIF87a", b"GIF89a")

    def test_leaves_no_temp_file(self, tiny_anim, tmp_path, monkeypatch):
        """The temp file used for rendering is cleaned up afterwards."""
        monkeypatch.setattr(
            "tempfile.tempdir", str(tmp_path)
        )  # confine temp files here
        before = set(tmp_path.iterdir())
        to_gif(tiny_anim, fps=2)
        after = set(tmp_path.iterdir())

        assert before == after, "to_gif left a temp file behind"


class TestEmbedGif:
    """Tests for `embed_gif` (notebook inline display)."""

    def test_returns_ipython_image(self, tiny_anim):
        """`embed_gif` returns an `IPython.display.Image` wrapping the GIF."""
        Image = pytest.importorskip("IPython.display").Image

        result = embed_gif(tiny_anim, fps=2)

        assert isinstance(result, Image), "expected an IPython.display.Image"
        assert result.format == "gif"
        assert result.data[:6] in (b"GIF87a", b"GIF89a")


class TestSupportedVideoFormat:
    """Tests for the module-level `SUPPORTED_VIDEO_FORMAT` constant."""

    def test_contains_expected_formats(self):
        """The constant lists exactly the four supported container formats."""
        assert set(SUPPORTED_VIDEO_FORMAT) == {"gif", "mov", "avi", "mp4"}

    def test_is_single_source_of_truth(self):
        """`cleopatra.glyph` re-exports the same object, not a copy."""
        from cleopatra.glyph import SUPPORTED_VIDEO_FORMAT as glyph_constant

        assert glyph_constant is SUPPORTED_VIDEO_FORMAT, (
            "glyph should re-import the constant, not redefine it"
        )
