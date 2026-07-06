"""Tests for cleopatra.animation glyph-independent save/embed helpers.

Covers `save_animation`, `to_gif`, and `embed_gif` operating on a plain
`matplotlib.animation.FuncAnimation` (no `Glyph` involved). A tiny 2-3 frame
animation is rendered on the Agg backend (set globally via `MPLBACKEND`).
"""

from __future__ import annotations

import builtins
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import pytest
from matplotlib.animation import FuncAnimation

import cleopatra.animation as anim_mod
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

    def test_accepts_pathlib_path(self, tiny_anim, tmp_path):
        """A `pathlib.Path` output path is accepted, not just `str` (issue #180).

        Regression: the format was derived with `str.rsplit`, so a `Path`
        raised `AttributeError`. The path is now normalised via `os.fspath`.
        """
        path = tmp_path / "from_path.gif"
        returned = save_animation(tiny_anim, path, fps=2)  # Path, not str

        assert path.exists(), "GIF was not written from a pathlib.Path"
        assert path.read_bytes()[:6] in (b"GIF87a", b"GIF89a")
        assert returned == str(path), "should return the fspath string of the Path"

    def test_unsupported_format_raises(self, tiny_anim, tmp_path):
        """An unsupported extension raises a clear ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            save_animation(tiny_anim, str(tmp_path / "out.webm"))

    def test_unsupported_format_raises_with_path(self, tiny_anim, tmp_path):
        """An unsupported extension raises even when the path is a `Path`."""
        with pytest.raises(ValueError, match="not supported"):
            save_animation(tiny_anim, tmp_path / "out.webm")

    def test_extension_is_case_insensitive(self, tiny_anim, tmp_path):
        """An upper/mixed-case extension is matched the same as lower-case."""
        path = tmp_path / "out.GIF"
        save_animation(tiny_anim, str(path), fps=2)

        assert path.exists(), "upper-case .GIF was not written"
        assert path.read_bytes()[:6] in (b"GIF87a", b"GIF89a")

    def test_ffmpeg_missing_raises_friendly_error(self, tmp_path):
        """A missing FFmpeg binary surfaces as `FileNotFoundError` with URL."""
        anim = MagicMock(spec=FuncAnimation)
        anim.save = MagicMock(side_effect=FileNotFoundError("ffmpeg not found"))

        with pytest.raises(FileNotFoundError, match="ffmpeg.org"):
            save_animation(anim, str(tmp_path / "movie.mp4"))

    def test_routes_gif_to_pillow_writer(self, monkeypatch):
        """The `.gif` branch builds a `PillowWriter(fps=...)` and saves with it.

        Test scenario:
            A mocked animation and a patched `PillowWriter` confirm the GIF
            branch is taken, the writer receives the requested ``fps``, and
            ``anim.save`` is called once with that writer.
        """

        anim = MagicMock(spec=FuncAnimation)
        pillow = MagicMock(name="PillowWriter")
        monkeypatch.setattr(anim_mod, "PillowWriter", pillow)

        result = save_animation(anim, "clip.gif", fps=7)

        pillow.assert_called_once_with(fps=7)
        anim.save.assert_called_once_with("clip.gif", writer=pillow.return_value)
        assert result == "clip.gif", f"should return the path, got {result!r}"

    @pytest.mark.parametrize("ext", ["mov", "avi", "mp4"])
    def test_routes_video_to_ffmpeg_writer(self, ext, monkeypatch):
        """Non-GIF formats build an `FFMpegWriter(fps=, bitrate=1800)`.

        Args:
            ext: A supported video container extension.

        Test scenario:
            For each video extension the else-branch is taken, the writer is
            constructed with the expected ``fps``/``bitrate``, ``anim.save``
            is invoked with it, and the written path is returned. Exercises
            the video success path without requiring FFmpeg.
        """

        anim = MagicMock(spec=FuncAnimation)
        ffmpeg = MagicMock(name="FFMpegWriter")
        monkeypatch.setattr(anim_mod, "FFMpegWriter", ffmpeg)

        result = save_animation(anim, f"clip.{ext}", fps=5)

        ffmpeg.assert_called_once_with(fps=5, bitrate=1800)
        anim.save.assert_called_once_with(f"clip.{ext}", writer=ffmpeg.return_value)
        assert result == f"clip.{ext}", f"should return the path, got {result!r}"

    def test_no_extension_raises(self):
        """A path with no extension is rejected with a clear message.

        Test scenario:
            ``"noext"`` has no dot, so ``os.path.splitext`` yields an empty
            extension, which raises ``ValueError`` naming the path before any
            save is attempted.
        """
        anim = MagicMock(spec=FuncAnimation)
        with pytest.raises(ValueError, match="no file extension"):
            save_animation(anim, "noext")
        anim.save.assert_not_called()

    @pytest.mark.parametrize("path", [".gif", "dir/.gif", "gif", "mp4"])
    def test_dotfile_or_bare_name_rejected(self, path):
        """Dotfile-style / extension-less names have no real extension and raise.

        Test scenario:
            `os.path.splitext` treats a leading-dot basename (`.gif`) or a
            dot-less name (`gif`) as having no extension, so these are rejected
            rather than silently written — locking the intended behaviour.
        """
        anim = MagicMock(spec=FuncAnimation)
        with pytest.raises(ValueError, match="no file extension"):
            save_animation(anim, path)
        anim.save.assert_not_called()

    def test_multi_dot_filename_uses_last_segment(self, monkeypatch):
        """Only the final dot-segment is treated as the extension.

        Test scenario:
            ``"my.movie.v2.gif"`` resolves to ``gif`` (not ``v2``), so the
            GIF branch is taken and the full path is preserved on save.
        """

        anim = MagicMock(spec=FuncAnimation)
        monkeypatch.setattr(anim_mod, "PillowWriter", MagicMock())

        result = save_animation(anim, "my.movie.v2.gif", fps=2)

        anim.save.assert_called_once()
        assert result == "my.movie.v2.gif", f"path not preserved: {result!r}"

    def test_default_fps_is_two(self, monkeypatch):
        """Omitting ``fps`` defaults the writer to 2 frames per second.

        Test scenario:
            Calling without ``fps`` builds ``PillowWriter(fps=2)``.
        """

        pillow = MagicMock(name="PillowWriter")
        monkeypatch.setattr(anim_mod, "PillowWriter", pillow)

        save_animation(MagicMock(spec=FuncAnimation), "clip.gif")

        pillow.assert_called_once_with(fps=2)


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

    def test_removes_temp_file_on_save_failure(self, tmp_path, monkeypatch):
        """The temp file is removed even when rendering raises.

        Test scenario:
            ``save_animation`` is patched to raise; the original error must
            propagate and the ``finally`` block must still delete the temp
            file (no leak on the error path).
        """

        monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

        def boom(*args, **kwargs):
            raise RuntimeError("render failed")

        monkeypatch.setattr(anim_mod, "save_animation", boom)
        before = set(tmp_path.iterdir())

        with pytest.raises(RuntimeError, match="render failed"):
            to_gif(MagicMock(spec=FuncAnimation))

        assert set(tmp_path.iterdir()) == before, "temp file leaked on failure"

    def test_forwards_fps_to_save_animation(self, tmp_path, monkeypatch):
        """``fps`` is forwarded to ``save_animation`` and bytes are returned.

        Test scenario:
            ``save_animation`` is patched to record ``fps`` and write known
            bytes; ``to_gif`` must pass the requested ``fps`` through and
            return exactly those bytes.
        """

        monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
        captured = {}

        def fake_save(anim, path, fps):
            captured["fps"] = fps
            Path(path).write_bytes(b"GIF89a-data")
            return path

        monkeypatch.setattr(anim_mod, "save_animation", fake_save)

        data = to_gif(MagicMock(spec=FuncAnimation), fps=9)

        assert captured["fps"] == 9, f"fps not forwarded, got {captured.get('fps')!r}"
        assert data == b"GIF89a-data", f"unexpected bytes: {data!r}"


class TestEmbedGif:
    """Tests for `embed_gif` (notebook inline display)."""

    def test_returns_ipython_image(self, tiny_anim):
        """`embed_gif` returns an `IPython.display.Image` wrapping the GIF."""
        Image = pytest.importorskip("IPython.display").Image

        result = embed_gif(tiny_anim, fps=2)

        assert isinstance(result, Image), "expected an IPython.display.Image"
        assert result.format == "gif"
        assert result.data[:6] in (b"GIF87a", b"GIF89a")

    def test_delegates_to_to_gif_with_fps(self, monkeypatch):
        """`embed_gif` renders via `to_gif` (forwarding ``fps``) then wraps it.

        Test scenario:
            ``to_gif`` is patched to return known bytes; ``embed_gif`` must
            call it with the same animation and ``fps``, and wrap the result
            in an ``Image`` of format ``gif`` carrying those bytes. Avoids a
            real render for determinism.
        """
        pytest.importorskip("IPython.display")

        fake_to_gif = MagicMock(return_value=b"GIF89a-embed")
        monkeypatch.setattr(anim_mod, "to_gif", fake_to_gif)
        anim = MagicMock(spec=FuncAnimation)

        result = anim_mod.embed_gif(anim, fps=4)

        fake_to_gif.assert_called_once_with(anim, fps=4)
        assert result.format == "gif", f"expected gif, got {result.format!r}"
        assert result.data == b"GIF89a-embed", f"unexpected bytes: {result.data!r}"

    def test_missing_ipython_raises_friendly_error(self, monkeypatch):
        """Without IPython, a clear `ModuleNotFoundError` with a hint is raised.

        Test scenario:
            The ``IPython.display`` import is forced to fail; ``embed_gif``
            must surface an actionable message (``pip install ipython`` and a
            pointer to ``to_gif``) rather than a bare import error.
        """
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("IPython"):
                raise ModuleNotFoundError(
                    "No module named 'IPython'", name="IPython"
                )
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ModuleNotFoundError, match="pip install ipython"):
            embed_gif(MagicMock(spec=FuncAnimation))

    def test_missing_subdependency_is_not_remapped(self, monkeypatch):
        """A missing IPython *sub-dependency* surfaces unchanged, not remapped.

        Test scenario:
            IPython is importable, but one of its transitive imports raises
            ``ModuleNotFoundError`` for some other package. ``embed_gif`` must
            re-raise that original error rather than misattribute it to a
            missing IPython.
        """
        pytest.importorskip("IPython.display")
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "IPython.display":
                raise ModuleNotFoundError(
                    "No module named 'some_dep'", name="some_dep"
                )
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ModuleNotFoundError, match="some_dep"):
            embed_gif(MagicMock(spec=FuncAnimation))

    def test_image_not_imported_at_module_level(self):
        """IPython stays optional: ``Image`` is not bound at module import.

        Test scenario:
            The lazy import inside ``embed_gif`` means importing
            ``cleopatra.animation`` must not expose an ``Image`` attribute,
            so the package never hard-depends on IPython at load time.
        """

        assert not hasattr(anim_mod, "Image"), (
            "IPython Image must not be imported at module load time"
        )


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
