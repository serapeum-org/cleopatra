"""Tests for cleopatra.glyphs.base.animation glyph-independent save/embed helpers.

Covers `save_animation`, `to_gif`, and `embed_gif` operating on a plain
`matplotlib.animation.FuncAnimation` (no `Glyph` involved). A tiny 2-3 frame
animation is rendered on the Agg backend (set globally via `MPLBACKEND`).
"""

from __future__ import annotations

import builtins
import doctest
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation
from PIL import Image

import cleopatra.glyphs.base.animation as anim_mod
from cleopatra.glyphs.base.animation import (
    SUPPORTED_VIDEO_FORMAT,
    embed_gif,
    save_animation,
    to_bytes,
    to_gif,
    to_mp4,
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
        """The `.gif` branch builds an `_OptimizedPillowWriter` and saves with it.

        Test scenario:
            A mocked animation and a patched `_OptimizedPillowWriter` confirm
            the GIF branch is taken, the writer receives the requested ``fps``,
            and ``anim.save`` is called once with that writer.
        """

        anim = MagicMock(spec=FuncAnimation)
        pillow = MagicMock(name="_OptimizedPillowWriter")
        monkeypatch.setattr(anim_mod, "_OptimizedPillowWriter", pillow)

        result = save_animation(anim, "clip.gif", fps=7)

        pillow.assert_called_once_with(fps=7, optimize=True, loop=0)
        anim.save.assert_called_once_with("clip.gif", writer=pillow.return_value)
        assert result == "clip.gif", f"should return the path, got {result!r}"

    @pytest.mark.parametrize("ext", ["mov", "avi", "mp4"])
    def test_routes_video_to_ffmpeg_writer(self, ext, monkeypatch):
        """Non-GIF formats build an `FFMpegWriter` with the even-pad + pix_fmt args.

        Args:
            ext: A supported video container extension.

        Test scenario:
            For each video extension the else-branch is taken, ffmpeg
            availability is resolved, the writer is constructed with the
            expected ``fps``/``bitrate`` plus the odd-dimension pad filter and
            an explicit ``yuv420p`` pixel format, ``anim.save`` is invoked with
            it, and the written path is returned. Exercises the video success
            path without requiring a real FFmpeg run.
        """

        anim = MagicMock(spec=FuncAnimation)
        ffmpeg = MagicMock(name="FFMpegWriter")
        monkeypatch.setattr(anim_mod, "FFMpegWriter", ffmpeg)
        monkeypatch.setattr(anim_mod, "_ensure_ffmpeg_available", lambda: None)

        result = save_animation(anim, f"clip.{ext}", fps=5)

        ffmpeg.assert_called_once_with(
            fps=5,
            extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p"],
        )
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
        monkeypatch.setattr(anim_mod, "_OptimizedPillowWriter", MagicMock())

        result = save_animation(anim, "my.movie.v2.gif", fps=2)

        anim.save.assert_called_once()
        assert result == "my.movie.v2.gif", f"path not preserved: {result!r}"

    def test_default_fps_is_two(self, monkeypatch):
        """Omitting ``fps`` defaults the writer to 2 frames per second.

        Test scenario:
            Calling without ``fps`` builds ``PillowWriter(fps=2)``.
        """

        pillow = MagicMock(name="_OptimizedPillowWriter")
        monkeypatch.setattr(anim_mod, "_OptimizedPillowWriter", pillow)

        save_animation(MagicMock(spec=FuncAnimation), "clip.gif")

        pillow.assert_called_once_with(fps=2, optimize=True, loop=0)


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


class TestToBytes:
    """Tests for `to_bytes` (in-memory render in any supported format)."""

    def test_gif_bytes(self, tiny_anim):
        """A ``fmt="gif"`` render returns GIF-magic bytes."""
        data = to_bytes(tiny_anim, fmt="gif", fps=2)

        assert data[:6] in (b"GIF87a", b"GIF89a"), f"not GIF bytes: {data[:6]!r}"

    def test_webp_bytes(self, tiny_anim):
        """A ``fmt="webp"`` render returns RIFF/WEBP bytes."""
        data = to_bytes(tiny_anim, fmt="webp", fps=2)

        assert data[:4] == b'RIFF', 'not WebP bytes'
        assert data[8:12] == b'WEBP', 'not WebP bytes'

    def test_leading_dot_and_case_tolerated(self, tiny_anim):
        """``fmt`` accepts a leading dot and mixed case (e.g. ``".GIF"``)."""
        data = to_bytes(tiny_anim, fmt=".GIF")

        assert data[:6] in (b"GIF87a", b"GIF89a"), "dot/case fmt not normalised"

    def test_unsupported_format_raises(self):
        """An unsupported ``fmt`` raises a clear ValueError before rendering."""
        with pytest.raises(ValueError, match="not supported"):
            to_bytes(MagicMock(spec=FuncAnimation), fmt="webm")

    def test_forwards_kwargs_to_save_animation(self, tmp_path, monkeypatch):
        """``fps`` and extra kwargs are forwarded to ``save_animation``.

        Test scenario:
            ``save_animation`` is patched to record what it receives; ``to_bytes``
            must forward ``fps`` and any quality kwargs and return the written
            bytes.
        """
        monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
        captured = {}

        def fake_save(anim, path, fps, **kwargs):
            captured["fps"] = fps
            captured.update(kwargs)
            Path(path).write_bytes(b"payload")
            return path

        monkeypatch.setattr(anim_mod, "save_animation", fake_save)

        data = to_bytes(MagicMock(spec=FuncAnimation), fmt="mp4", fps=6, crf=24)

        assert captured == {"fps": 6, "crf": 24}, f"kwargs not forwarded: {captured}"
        assert data == b"payload", f"unexpected bytes: {data!r}"

    def test_leaves_no_temp_file(self, tiny_anim, tmp_path, monkeypatch):
        """The temp file used for rendering is cleaned up afterwards."""
        monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
        before = set(tmp_path.iterdir())

        to_bytes(tiny_anim, fmt="gif")

        assert set(tmp_path.iterdir()) == before, "to_bytes left a temp file behind"


class TestToMp4:
    """Tests for `to_mp4`."""

    def test_returns_mp4_bytes(self, tiny_anim):
        """`to_mp4` returns a non-empty ISO base-media (MP4) payload."""
        data = to_mp4(tiny_anim, fps=2)

        assert data[4:8] == b"ftyp", f"not an MP4/ISO-BMFF payload: {data[:12]!r}"
        assert len(data) > 0, "MP4 bytes are empty"

    def test_delegates_to_to_bytes(self, monkeypatch):
        """`to_mp4` delegates to `to_bytes` with ``fmt="mp4"`` and forwards kwargs.

        Test scenario:
            ``to_bytes`` is patched to return known bytes; ``to_mp4`` must call
            it with ``fmt="mp4"``, the same ``fps``, and any extra kwargs, and
            return its result.
        """
        spy = MagicMock(return_value=b"MP4-bytes")
        monkeypatch.setattr(anim_mod, "to_bytes", spy)
        anim = MagicMock(spec=FuncAnimation)

        result = anim_mod.to_mp4(anim, fps=5, crf=20)

        spy.assert_called_once_with(anim, fmt="mp4", fps=5, crf=20)
        assert result == b"MP4-bytes", f"unexpected bytes: {result!r}"


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
                raise ModuleNotFoundError("No module named 'IPython'", name="IPython")
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
                raise ModuleNotFoundError("No module named 'some_dep'", name="some_dep")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ModuleNotFoundError, match="some_dep"):
            embed_gif(MagicMock(spec=FuncAnimation))

    def test_image_not_imported_at_module_level(self):
        """IPython stays optional: ``Image`` is not bound at module import.

        Test scenario:
            The lazy import inside ``embed_gif`` means importing
            ``cleopatra.glyphs.base.animation`` must not expose an ``Image`` attribute,
            so the package never hard-depends on IPython at load time.
        """

        assert not hasattr(anim_mod, "Image"), (
            "IPython Image must not be imported at module load time"
        )


class TestOddDimensionAutoPad:
    """Regression tests for odd pixel dimensions crashing mp4 export (issue #185)."""

    def test_odd_dimension_mp4_encodes(self, tmp_path):
        """A figure with odd pixel width/height encodes to mp4 without crashing.

        Test scenario:
            libx264 rejects odd dimensions, so a 335x335 px figure previously
            died with "height not divisible by 2". The auto-pad video filter
            must let it encode to a real, non-empty mp4.
        """
        fig = plt.figure(figsize=(3.35, 3.35), dpi=100)
        ax = fig.add_subplot(111)
        (line,) = ax.plot([0, 1], [0, 0])
        width = int(round(fig.get_figwidth() * fig.dpi))
        height = int(round(fig.get_figheight() * fig.dpi))
        assert width % 2 == 1, (
            f'fixture must be odd-sized to exercise the pad, got {width}x{height}'
        )
        assert height % 2 == 1, (
            f'fixture must be odd-sized to exercise the pad, got {width}x{height}'
        )
        anim = FuncAnimation(fig, lambda i: (line,), frames=2)
        out = tmp_path / "odd.mp4"

        save_animation(anim, str(out), fps=2)
        plt.close(fig)

        assert out.exists(), "odd-dimension mp4 was not written"
        assert out.stat().st_size > 0, "odd-dimension mp4 is empty"


class TestWebP:
    """Tests for animated WebP output (issue #185)."""

    def test_writes_animated_webp(self, tiny_anim, tmp_path):
        """A ``.webp`` path is written by Pillow as a multi-frame WebP.

        Test scenario:
            WebP routes to the Pillow writer (not FFmpeg); the output has the
            RIFF/WEBP magic bytes and more than one frame.
        """
        out = tmp_path / "out.webp"

        returned = save_animation(tiny_anim, str(out), fps=3)

        assert returned == str(out), "should return the written path"
        raw = out.read_bytes()
        assert raw[:4] == b'RIFF', 'not a WebP file'
        assert raw[8:12] == b'WEBP', 'not a WebP file'
        assert getattr(Image.open(out), "n_frames", 1) > 1, "WebP is not animated"

    def test_webp_routes_to_pillow_writer(self, monkeypatch):
        """WebP uses `_OptimizedPillowWriter`, never the FFmpeg writer.

        Test scenario:
            The ``.webp`` branch builds the Pillow writer with the loop/optimize
            settings and does not touch ``FFMpegWriter``.
        """
        anim = MagicMock(spec=FuncAnimation)
        pillow = MagicMock(name="_OptimizedPillowWriter")
        ffmpeg = MagicMock(name="FFMpegWriter")
        monkeypatch.setattr(anim_mod, "_OptimizedPillowWriter", pillow)
        monkeypatch.setattr(anim_mod, "FFMpegWriter", ffmpeg)

        save_animation(anim, "clip.webp", fps=4, loop=1)

        pillow.assert_called_once_with(fps=4, optimize=True, loop=1)
        ffmpeg.assert_not_called()


class TestEnsureFfmpegAvailable:
    """Tests for `_ensure_ffmpeg_available` (ffmpeg binary resolution)."""

    def test_keeps_system_ffmpeg_on_path(self, monkeypatch):
        """A resolvable ffmpeg on PATH is kept and imageio-ffmpeg is not consulted.

        Test scenario:
            When ``shutil.which`` finds the configured binary, the rcParam is
            left untouched and the bundled-binary fallback is never invoked.
        """
        import imageio_ffmpeg

        monkeypatch.setitem(mpl.rcParams, "animation.ffmpeg_path", "ffmpeg")
        monkeypatch.setattr(anim_mod.shutil, "which", lambda name: "C:/bin/ffmpeg.exe")
        calls = {"n": 0}
        monkeypatch.setattr(
            imageio_ffmpeg,
            "get_ffmpeg_exe",
            lambda: calls.__setitem__("n", calls["n"] + 1),
        )

        anim_mod._ensure_ffmpeg_available()

        assert mpl.rcParams["animation.ffmpeg_path"] == "ffmpeg", (
            "system ffmpeg path should be left unchanged"
        )
        assert calls["n"] == 0, (
            "bundled binary must not be consulted when PATH resolves"
        )

    def test_falls_back_to_bundled_binary(self, monkeypatch):
        """With no system ffmpeg, the rcParam is pointed at imageio-ffmpeg's binary.

        Test scenario:
            When neither an absolute path nor a PATH lookup resolves, the
            resolver sets ``animation.ffmpeg_path`` to
            ``imageio_ffmpeg.get_ffmpeg_exe()`` so export still works.
        """
        import imageio_ffmpeg

        monkeypatch.setitem(mpl.rcParams, "animation.ffmpeg_path", "ffmpeg")
        monkeypatch.setattr(anim_mod.os.path, "isfile", lambda path: False)
        monkeypatch.setattr(anim_mod.shutil, "which", lambda name: None)
        monkeypatch.setattr(
            imageio_ffmpeg, "get_ffmpeg_exe", lambda: "C:/bundled/ffmpeg.exe"
        )

        anim_mod._ensure_ffmpeg_available()

        assert mpl.rcParams["animation.ffmpeg_path"] == "C:/bundled/ffmpeg.exe", (
            "resolver should fall back to the imageio-ffmpeg binary"
        )

    def test_warns_when_overriding_explicit_path(self, monkeypatch):
        """Overriding a non-default, unresolved ffmpeg_path emits a RuntimeWarning.

        Test scenario:
            A user who set an explicit path that no longer resolves should be
            told their choice was replaced by the bundled binary, not have it
            discarded silently.
        """
        import imageio_ffmpeg

        monkeypatch.setitem(
            mpl.rcParams, "animation.ffmpeg_path", "C:/nope/custom-ffmpeg.exe"
        )
        monkeypatch.setattr(anim_mod.os.path, "isfile", lambda path: False)
        monkeypatch.setattr(anim_mod.shutil, "which", lambda name: None)
        monkeypatch.setattr(
            imageio_ffmpeg, "get_ffmpeg_exe", lambda: "C:/bundled/ffmpeg.exe"
        )

        with pytest.warns(RuntimeWarning, match="custom-ffmpeg"):
            anim_mod._ensure_ffmpeg_available()

        assert mpl.rcParams["animation.ffmpeg_path"] == "C:/bundled/ffmpeg.exe", (
            "should still fall back after warning"
        )

    def test_default_path_override_is_silent(self, monkeypatch, recwarn):
        """Falling back from the default ``"ffmpeg"`` emits no warning.

        Test scenario:
            The unconfigured default is expected to be replaced quietly.
        """
        import imageio_ffmpeg

        monkeypatch.setitem(mpl.rcParams, "animation.ffmpeg_path", "ffmpeg")
        monkeypatch.setattr(anim_mod.os.path, "isfile", lambda path: False)
        monkeypatch.setattr(anim_mod.shutil, "which", lambda name: None)
        monkeypatch.setattr(
            imageio_ffmpeg, "get_ffmpeg_exe", lambda: "C:/bundled/ffmpeg.exe"
        )

        anim_mod._ensure_ffmpeg_available()

        assert len(recwarn) == 0, f"default fallback should not warn: {list(recwarn)}"

    def test_raises_when_no_ffmpeg_available(self, monkeypatch):
        """When neither system ffmpeg nor imageio-ffmpeg exist, raise FileNotFoundError.

        Test scenario:
            The bundled-binary import is forced to fail; the resolver must
            surface an actionable ``FileNotFoundError`` naming imageio-ffmpeg.
        """
        monkeypatch.setattr(anim_mod.os.path, "isfile", lambda path: False)
        monkeypatch.setattr(anim_mod.shutil, "which", lambda name: None)
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "imageio_ffmpeg":
                raise ModuleNotFoundError("no imageio_ffmpeg", name="imageio_ffmpeg")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(FileNotFoundError, match="imageio-ffmpeg"):
            anim_mod._ensure_ffmpeg_available()


class TestOptimizedPillowWriter:
    """Tests for `_OptimizedPillowWriter` (optimised, loopable GIF writer)."""

    def test_stores_optimize_and_loop(self):
        """The constructor records the ``optimize`` and ``loop`` settings.

        Test scenario:
            Non-default values are stored on the instance so ``finish`` can
            forward them to Pillow.
        """
        writer = anim_mod._OptimizedPillowWriter(fps=5, optimize=False, loop=3)

        assert writer._optimize is False, "optimize flag not stored"
        assert writer._loop == 3, "loop count not stored"

    def test_writes_gif_with_configured_loop(self, tiny_anim, tmp_path):
        """A finite loop count is embedded in the written GIF's metadata.

        Test scenario:
            Saving with ``loop=2`` yields a GIF whose Pillow ``loop`` info is 2.
        """
        out = tmp_path / "loop.gif"
        tiny_anim.save(str(out), writer=anim_mod._OptimizedPillowWriter(fps=3, loop=2))

        assert Image.open(out).info.get("loop") == 2, "loop count not written to GIF"

    def test_default_loops_forever(self, tiny_anim, tmp_path):
        """The default ``loop=0`` produces a GIF that loops forever.

        Test scenario:
            Saving without a loop override yields Pillow's forever-loop marker
            (``loop == 0``).
        """
        out = tmp_path / "forever.gif"
        tiny_anim.save(str(out), writer=anim_mod._OptimizedPillowWriter(fps=3))

        assert Image.open(out).info.get("loop") == 0, "default GIF should loop forever"

    def test_forwards_optimize_flag_to_pillow(self, tiny_anim, tmp_path, monkeypatch):
        """The ``optimize`` flag reaches ``PIL.Image.Image.save``.

        Test scenario:
            A save spy captures the keyword arguments Pillow receives; writing
            with ``optimize=False`` must forward that exact value.
        """
        captured = {}
        real_save = Image.Image.save

        def spy_save(self, fp, *args, **kwargs):
            captured.update(kwargs)
            return real_save(self, fp, *args, **kwargs)

        monkeypatch.setattr(Image.Image, "save", spy_save)
        out = tmp_path / "opt.gif"

        tiny_anim.save(
            str(out), writer=anim_mod._OptimizedPillowWriter(fps=3, optimize=False)
        )

        assert captured.get("optimize") is False, (
            f"optimize flag not forwarded to Pillow: {captured}"
        )

    def test_gif_shared_palette_keeps_constant_region_stable(self, tmp_path):
        """A region that never changes stays byte-identical across GIF frames.

        Test scenario:
            An animation whose top half is constant white and bottom half is
            random. A per-frame GIF palette would re-quantise/re-dither the
            white top and make it shimmer; the writer's shared palette keeps it
            byte-stable frame-to-frame.
        """
        rng = np.random.default_rng(0)
        fig, ax = plt.subplots()
        ax.set_position([0, 0, 1, 1])
        ax.set_axis_off()
        im = ax.imshow(np.zeros((40, 40, 3)))

        def update(_):
            frame = rng.random((40, 40, 3))
            frame[:20] = 1.0  # constant white top half
            im.set_data(frame)
            return (im,)

        anim = FuncAnimation(fig, update, frames=6, blit=False)
        out = tmp_path / "const.gif"
        save_animation(anim, str(out), fps=4)

        gif = Image.open(out)
        gif.seek(0)
        first = np.asarray(gif.convert("RGB")).copy()
        gif.seek(5)
        last = np.asarray(gif.convert("RGB")).copy()
        top = slice(0, first.shape[0] // 3)  # safely inside the constant white top
        changed = (
            np.abs(first[top].astype(int) - last[top].astype(int)).sum(2) > 8
        ).mean()
        plt.close("all")

        assert changed == 0.0, (
            f"shared palette should keep a constant region byte-stable, got {changed}"
        )

    def test_gif_reserves_black_for_crisp_overlays(self, tmp_path):
        """A pure-black overlay on a colourful field stays black in the GIF.

        Test scenario:
            A colourful (random-RGB) animated field with a constant pure-black
            block in the corner -- standing in for the date label / colourbar
            ticks drawn on the frame. With every palette slot spent on the
            photographic colours the block would quantise to a muddy dark grey
            and read as faded; the writer pins pure black into the palette so
            the block stays black.
        """
        rng = np.random.default_rng(1)
        fig, ax = plt.subplots()
        ax.set_position([0, 0, 1, 1])
        ax.set_axis_off()
        im = ax.imshow(np.zeros((60, 60, 3)), interpolation="nearest")

        def update(_):
            frame = rng.random((60, 60, 3))  # colourful -- fills the palette
            frame[:20, :30] = 0.0  # constant pure-black block (the "overlay")
            im.set_data(frame)
            return (im,)

        anim = FuncAnimation(fig, update, frames=6, blit=False)
        out = tmp_path / "black.gif"
        save_animation(anim, str(out), fps=4)

        gif = Image.open(out)
        gif.seek(3)
        arr = np.asarray(gif.convert("RGB")).copy()
        h, w, _ = arr.shape
        block = arr[: h // 6, : w // 4]  # safely inside the black block
        darkest = int(block.sum(2).min())
        plt.close("all")

        assert darkest <= 12, (
            "reserved-black palette should keep a pure-black overlay black, "
            f"got darkest pixel sum {darkest} (0 = pure black)"
        )


class TestQualityControls:
    """Tests for the crf/bitrate/codec/preset/dpi/gif controls of `save_animation`."""

    def _mock_ffmpeg(self, monkeypatch):
        """Patch the ffmpeg writer and availability check; return the writer mock."""
        ffmpeg = MagicMock(name="FFMpegWriter")
        monkeypatch.setattr(anim_mod, "FFMpegWriter", ffmpeg)
        monkeypatch.setattr(anim_mod, "_ensure_ffmpeg_available", lambda: None)
        return ffmpeg

    def test_crf_and_preset_reach_writer(self, monkeypatch):
        """`crf` and `preset` are appended to the ffmpeg extra_args.

        Test scenario:
            Requesting ``crf=26, preset="slow"`` puts ``-crf 26 -preset slow``
            at the tail of the writer's ``extra_args``.
        """
        ffmpeg = self._mock_ffmpeg(monkeypatch)

        save_animation(MagicMock(spec=FuncAnimation), "clip.mp4", crf=26, preset="slow")

        _, kwargs = ffmpeg.call_args
        assert kwargs["extra_args"][-4:] == [
            "-crf",
            "26",
            "-preset",
            "slow",
        ], f"crf/preset not in extra_args: {kwargs['extra_args']}"

    def test_bitrate_and_codec_reach_writer(self, monkeypatch):
        """`bitrate` and `codec` are forwarded to the FFMpegWriter constructor.

        Test scenario:
            ``bitrate=2500, codec="libx264"`` appear as constructor kwargs.
        """
        ffmpeg = self._mock_ffmpeg(monkeypatch)

        save_animation(
            MagicMock(spec=FuncAnimation), "clip.mp4", bitrate=2500, codec="libx264"
        )

        _, kwargs = ffmpeg.call_args
        assert kwargs["bitrate"] == 2500, f"bitrate not forwarded: {kwargs}"
        assert kwargs["codec"] == "libx264", f"codec not forwarded: {kwargs}"

    def test_crf_and_bitrate_together_raises(self):
        """Passing both `crf` and `bitrate` is rejected as competing modes.

        Test scenario:
            crf and bitrate are mutually exclusive rate-control modes, so
            supplying both raises ``ValueError`` before any encode.
        """
        with pytest.raises(ValueError, match="either crf or bitrate"):
            save_animation(
                MagicMock(spec=FuncAnimation), "clip.mp4", crf=20, bitrate=2000
            )

    def test_crf_and_bitrate_together_raises_for_gif(self):
        """crf+bitrate is rejected uniformly, even for the GIF/Pillow path.

        Test scenario:
            The mutual-exclusion check runs before the format branch, so a GIF
            with both raises rather than silently ignoring them.
        """
        with pytest.raises(ValueError, match="either crf or bitrate"):
            save_animation(
                MagicMock(spec=FuncAnimation), "clip.gif", crf=20, bitrate=2000
            )

    def test_ffmpeg_only_kwargs_ignored_for_gif(self, monkeypatch):
        """A lone ffmpeg-only kwarg is accepted (and ignored) on the GIF path.

        Test scenario:
            ``crf`` alone on a ``.gif`` does not raise and never touches the
            FFmpeg writer — GIF goes through the Pillow writer.
        """
        pillow = MagicMock(name="_OptimizedPillowWriter")
        ffmpeg = MagicMock(name="FFMpegWriter")
        monkeypatch.setattr(anim_mod, "_OptimizedPillowWriter", pillow)
        monkeypatch.setattr(anim_mod, "FFMpegWriter", ffmpeg)

        save_animation(MagicMock(spec=FuncAnimation), "clip.gif", crf=20)

        pillow.assert_called_once()
        ffmpeg.assert_not_called()

    def test_dpi_forwarded_to_save(self, monkeypatch):
        """`dpi` is forwarded to ``anim.save`` for the ffmpeg path.

        Test scenario:
            ``dpi=150`` reaches the underlying save call.
        """
        self._mock_ffmpeg(monkeypatch)
        anim = MagicMock(spec=FuncAnimation)

        save_animation(anim, "clip.mp4", dpi=150)

        _, kwargs = anim.save.call_args
        assert kwargs.get("dpi") == 150, f"dpi not forwarded: {kwargs}"

    def test_dpi_omitted_when_none(self, monkeypatch):
        """With no `dpi`, ``anim.save`` is called without a dpi kwarg.

        Test scenario:
            Backward compatibility — the default call must not inject a dpi.
        """
        self._mock_ffmpeg(monkeypatch)
        anim = MagicMock(spec=FuncAnimation)

        save_animation(anim, "clip.mp4")

        _, kwargs = anim.save.call_args
        assert "dpi" not in kwargs, f"dpi should be omitted when None: {kwargs}"

    def test_caller_vf_is_merged_with_pad(self, monkeypatch):
        """A caller ``-vf`` filter is merged into one chain, pad applied last.

        Test scenario:
            ``extra_args=["-vf", "scale=320:-1", "-tune", "film"]`` yields a
            single ``-vf scale=320:-1,pad=...`` chain and preserves the other
            flags.
        """
        ffmpeg = self._mock_ffmpeg(monkeypatch)

        save_animation(
            MagicMock(spec=FuncAnimation),
            "clip.mp4",
            extra_args=["-vf", "scale=320:-1", "-tune", "film"],
        )

        _, kwargs = ffmpeg.call_args
        args = kwargs["extra_args"]
        assert args[0] == "-vf", f"first flag should be -vf: {args}"
        assert args[1] == "scale=320:-1,pad=ceil(iw/2)*2:ceil(ih/2)*2", (
            f"caller filter not merged with pad: {args}"
        )
        assert args[-2:] == ["-tune", "film"], f"passthrough flags lost: {args}"

    def test_custom_pix_fmt_reaches_writer(self, monkeypatch):
        """A custom ``pix_fmt`` param replaces the default in the writer args.

        Test scenario:
            ``pix_fmt="yuv444p"`` is emitted as the single ``-pix_fmt`` value.
        """
        ffmpeg = self._mock_ffmpeg(monkeypatch)

        save_animation(MagicMock(spec=FuncAnimation), "clip.mp4", pix_fmt="yuv444p")

        args = ffmpeg.call_args.kwargs["extra_args"]
        assert args.count("-pix_fmt") == 1, f"expected one -pix_fmt: {args}"
        assert args[args.index("-pix_fmt") + 1] == "yuv444p", f"pix_fmt wrong: {args}"

    def test_caller_pix_fmt_in_extra_args_overrides_default(self, monkeypatch):
        """A ``-pix_fmt`` in ``extra_args`` overrides the default without duplication.

        Test scenario:
            ``extra_args=["-pix_fmt", "rgb24"]`` yields exactly one ``-pix_fmt``
            equal to ``rgb24`` (the forced ``yuv420p`` is not also emitted).
        """
        ffmpeg = self._mock_ffmpeg(monkeypatch)

        save_animation(
            MagicMock(spec=FuncAnimation), "clip.mp4", extra_args=["-pix_fmt", "rgb24"]
        )

        args = ffmpeg.call_args.kwargs["extra_args"]
        assert args.count("-pix_fmt") == 1, f"duplicate -pix_fmt emitted: {args}"
        assert args[args.index("-pix_fmt") + 1] == "rgb24", f"override lost: {args}"

    def test_explicit_empty_pix_fmt_override_is_honoured(self, monkeypatch):
        """An explicit (even empty) ``-pix_fmt`` in extra_args is authoritative.

        Test scenario:
            ``extra_args=["-pix_fmt", ""]`` yields an empty ``-pix_fmt`` value
            rather than silently reverting to the default (a present flag+value
            is treated as the caller's choice via an ``is not None`` sentinel).
        """
        ffmpeg = self._mock_ffmpeg(monkeypatch)

        save_animation(
            MagicMock(spec=FuncAnimation), "clip.mp4", extra_args=["-pix_fmt", ""]
        )

        args = ffmpeg.call_args.kwargs["extra_args"]
        assert args[args.index("-pix_fmt") + 1] == "", f"empty override dropped: {args}"

    @pytest.mark.parametrize("bad", [["-vf"], ["-crf", "20", "-pix_fmt"]])
    def test_dangling_flag_in_extra_args_raises(self, bad, monkeypatch):
        """A trailing valueless ``-vf``/``-pix_fmt`` raises instead of corrupting args.

        Args:
            bad: An ``extra_args`` list ending in a flag with no value.

        Test scenario:
            The merge helper rejects malformed input rather than appending a
            dangling flag that would break the ffmpeg command line.
        """
        self._mock_ffmpeg(monkeypatch)

        with pytest.raises(ValueError, match="must be followed by a value"):
            save_animation(MagicMock(spec=FuncAnimation), "clip.mp4", extra_args=bad)

    def test_gif_loop_and_optimize_forwarded(self, monkeypatch):
        """GIF `optimize` and `loop` reach the `_OptimizedPillowWriter`.

        Test scenario:
            ``optimize=False, loop=3`` are passed through to the GIF writer.
        """
        pillow = MagicMock(name="_OptimizedPillowWriter")
        monkeypatch.setattr(anim_mod, "_OptimizedPillowWriter", pillow)

        save_animation(
            MagicMock(spec=FuncAnimation), "clip.gif", optimize=False, loop=3
        )

        pillow.assert_called_once_with(fps=2, optimize=False, loop=3)


class TestSupportedVideoFormat:
    """Tests for the module-level `SUPPORTED_VIDEO_FORMAT` constant."""

    def test_contains_expected_formats(self):
        """The constant lists exactly the five supported container formats."""
        assert set(SUPPORTED_VIDEO_FORMAT) == {"gif", "mov", "avi", "mp4", "webp"}

    def test_is_single_source_of_truth(self):
        """`cleopatra.glyphs.base.glyph` re-exports the same object, not a copy."""
        from cleopatra.glyphs.base.glyph import SUPPORTED_VIDEO_FORMAT as glyph_constant

        assert glyph_constant is SUPPORTED_VIDEO_FORMAT, (
            "glyph should re-import the constant, not redefine it"
        )


def test_module_doctests_execute():
    """Run the module's docstring examples so they are exercised in CI.

    Pytest is not configured with ``--doctest-modules``, so docstring examples in
    ``src/`` would otherwise never run. This test executes them for
    ``cleopatra.glyphs.base.animation`` (including the ``to_bytes``/``to_mp4`` magic-byte checks
    and the ``_build_ffmpeg_extra_args`` flag-merge examples) and fails if any
    example's output no longer matches.
    """
    try:
        results = doctest.testmod(anim_mod, verbose=False)
    finally:
        plt.close("all")
    assert results.failed == 0, (
        f"{results.failed} doctest example(s) failed in animation"
    )
    assert results.attempted > 0, (
        "no doctest examples were collected from animation; the module's docstring "
        "examples may have been moved or removed, silently dropping this coverage"
    )


#: A textured background with four small, highly saturated marks that move each
#: frame -- the shape of a satellite-showcase clip in miniature. The texture owns
#: ~99% of the pixels, so it is exactly the case where a population-weighted
#: palette starves the marks.
_CLIP_W, _CLIP_H, _CLIP_FRAMES = 320, 180, 12
_MARK_COLORS = ((255, 0, 255), (0, 255, 255), (255, 255, 0), (255, 40, 0))
_MARK_ROWS = (40, 80, 120, 160)
_MARK_RADIUS = 2


def _clip_frames(radius=_MARK_RADIUS):
    """Build the texture-heavy clip.

    Args:
        radius: Half-width of each saturated mark, so a test can vary how many
            pixels a mark covers. ``0`` gives a single-pixel mark.

    Returns:
        list: One ``(frame, boxes)`` pair per frame, where `frame` is a float
        ``(H, W, 3)`` array in ``[0, 1]`` and `boxes` are the
        ``(y0, y1, x0, x1)`` slices holding each saturated mark.
    """
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:_CLIP_H, 0:_CLIP_W]
    base = np.clip(
        np.stack(
            [
                0.45 + 0.20 * np.sin(xx / 37.0) + 0.10 * np.cos(yy / 23.0),
                0.40 + 0.18 * np.cos(xx / 29.0) + 0.12 * np.sin(yy / 19.0),
                0.35 + 0.15 * np.sin((xx + yy) / 41.0),
            ],
            axis=-1,
        )
        + rng.normal(0, 0.035, (_CLIP_H, _CLIP_W, 3)),
        0,
        1,
    )
    frames = []
    for index in range(_CLIP_FRAMES):
        arr = base.copy()
        boxes = []
        for slot, rgb in enumerate(_MARK_COLORS):
            cy, cx = _MARK_ROWS[slot], 30 + index * 20
            y0, y1 = cy - radius, cy + radius + 1
            x0, x1 = cx - radius, cx + radius + 1
            arr[y0:y1, x0:x1] = np.array(rgb) / 255.0
            boxes.append((y0, y1, x0, x1))
        frames.append((arr, boxes))
    return frames


def _clip_animation(frames):
    """Wrap `_clip_frames` output in a pixel-exact `FuncAnimation`.

    Args:
        frames: The output of `_clip_frames`.

    Returns:
        tuple: The `Figure` and its `FuncAnimation`. The figure is sized so one
        array cell maps to one output pixel, letting a test read a mark's colour
        straight back out of the decoded GIF.
    """
    fig = plt.figure(figsize=(_CLIP_W / 100, _CLIP_H / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    image = ax.imshow(frames[0][0], interpolation="nearest")

    def update(i):
        image.set_data(frames[i][0])
        return (image,)

    return fig, FuncAnimation(fig, update, frames=len(frames))


def _decode(path):
    """Read every frame of an animated image back as an RGB array.

    Args:
        path: The animation to read.

    Returns:
        list: One ``(H, W, 3)`` ``uint8`` array per frame.
    """
    decoded = []
    with Image.open(path) as handle:
        try:
            while True:
                decoded.append(np.asarray(handle.convert("RGB"), dtype=np.uint8))
                handle.seek(handle.tell() + 1)
        except EOFError:
            pass
    return decoded


def _mark_distances(decoded, frames):
    """Mean RGB distance between each decoded mark and its intended colour.

    Args:
        decoded: Frames read back from the written file.
        frames: The source `_clip_frames` output the marks came from.

    Returns:
        list: One mean distance per mark, in the order of `_MARK_COLORS`.
    """
    distances = []
    for slot, rgb in enumerate(_MARK_COLORS):
        samples = []
        for index in range(min(len(decoded), len(frames))):
            y0, y1, x0, x1 = frames[index][1][slot]
            patch = decoded[index][y0:y1, x0:x1].reshape(-1, 3).astype(float)
            samples.append(patch.mean(axis=0))
        distances.append(
            float(np.linalg.norm(np.mean(samples, axis=0) - np.array(rgb, dtype=float)))
        )
    return distances


class TestClipPaletteQuality:
    """The shared GIF palette must not starve small saturated colours (#315)."""

    @pytest.mark.parametrize("radius, size", [(2, "5x5"), (1, "3x3"), (0, "1x1")])
    def test_small_saturated_marks_survive_quantisation(self, tmp_path, radius, size):
        """Saturated marks stay their own colour however few pixels they cover.

        Args:
            tmp_path: pytest temp directory.
            radius: Half-width of the marks under test.
            size: Human-readable mark size, for the failure message.

        Test scenario:
            The background owns ~99% of the pixels. Allocating palette slots by
            pixel population (median cut) hands nearly all of them to the
            texture and the marks decode ~100-180 away from the colours they
            were drawn in. The size sweep is the point: a palette built from a
            spatially downsampled clip passes at 5x5 and fails at 1x1, because
            the resize blends a one-pixel mark away before the quantiser sees
            it. Building it from the clip's distinct colours is independent of
            how large a mark is, so every size must hold -- and 1x1 is the size
            that actually matters for satellites, orbit paths and labels.
        """
        frames = _clip_frames(radius)
        fig, anim = _clip_animation(frames)
        out = tmp_path / "clip.gif"
        save_animation(anim, str(out), fps=12)
        plt.close(fig)

        distances = _mark_distances(_decode(out), frames)
        assert max(distances) < 40, (
            f"{size} saturated marks were quantised away; distances from the "
            f"intended colours were {[round(d, 1) for d in distances]}"
        )

    def test_palette_is_shared_across_the_clip(self, tmp_path):
        """Every frame draws from one 256-entry table, not its own.

        Test scenario:
            Independently quantised frames would between them use far more than
            256 distinct colours. The union across all frames must stay within
            a single palette.
        """
        frames = _clip_frames()
        fig, anim = _clip_animation(frames)
        out = tmp_path / "clip.gif"
        save_animation(anim, str(out), fps=12)
        plt.close(fig)

        decoded = _decode(out)
        union = set()
        for frame in decoded:
            union |= {tuple(pixel) for pixel in frame.reshape(-1, 3)}
        assert len(union) <= 256, (
            f"frames appear to be quantised independently: {len(union)} distinct "
            f"colours across {len(decoded)} frames"
        )


class TestGifFromVideo:
    """`gif_from_video` derives a GIF from an already-rendered video (#308)."""

    @pytest.fixture
    def clip_mp4(self, tmp_path_factory):
        """Render the texture clip once to a full-chroma MP4.

        Returns:
            tuple: The MP4 path and the `_clip_frames` output behind it.
        """
        frames = _clip_frames()
        fig, anim = _clip_animation(frames)
        path = tmp_path_factory.mktemp("video") / "clip.mp4"
        save_animation(anim, str(path), fps=12, crf=0, pix_fmt="yuv444p")
        plt.close(fig)
        return str(path), frames

    def test_derives_a_gif_from_an_mp4(self, clip_mp4, tmp_path):
        """A GIF is written from the video and the path comes back.

        Test scenario:
            The rendered MP4 is converted without touching the original
            animation, and the result carries the GIF magic bytes.
        """
        src, _ = clip_mp4
        out = tmp_path / "derived.gif"
        returned = anim_mod.gif_from_video(src, str(out), fps=12)
        assert returned == str(out), "the output path should be returned"
        assert out.read_bytes()[:6] in (b"GIF87a", b"GIF89a"), "not a GIF"

    def test_uses_one_clip_wide_palette(self, clip_mp4, tmp_path):
        """The derived GIF shares a single palette across its frames.

        Test scenario:
            `gif_from_video` runs the decoded frames through the same
            `build_clip_palette` the live-animation path uses, so the colour
            union across frames stays within one table.
        """
        src, _ = clip_mp4
        out = tmp_path / "derived.gif"
        anim_mod.gif_from_video(src, str(out), fps=12)
        union = set()
        for frame in _decode(out):
            union |= {tuple(pixel) for pixel in frame.reshape(-1, 3)}
        assert len(union) <= 256, f"palette is not clip-wide: {len(union)} colours"

    def test_saturated_marks_survive_the_round_trip(self, clip_mp4, tmp_path):
        """Marks survive decode + re-quantisation from a full-chroma source.

        Test scenario:
            Deriving from a video only pays off if the marks are still there
            afterwards. With a `yuv444p` source the round trip must land close
            to what rendering the GIF directly achieves.
        """
        src, frames = clip_mp4
        out = tmp_path / "derived.gif"
        anim_mod.gif_from_video(src, str(out), fps=12)
        distances = _mark_distances(_decode(out), frames)
        assert max(distances) < 40, (
            f"marks degraded through the video round trip: "
            f"{[round(d, 1) for d in distances]}"
        )

    def test_fps_controls_the_frame_count(self, clip_mp4, tmp_path):
        """Sampling at a lower fps yields proportionally fewer frames.

        Test scenario:
            The source is 12 frames at 12 fps (one second). Sampling at 6 fps
            halves the frame count.
        """
        src, _ = clip_mp4
        full = tmp_path / "full.gif"
        half = tmp_path / "half.gif"
        anim_mod.gif_from_video(src, str(full), fps=12)
        anim_mod.gif_from_video(src, str(half), fps=6)
        assert len(_decode(half)) < len(_decode(full)), (
            "a lower fps should sample fewer frames"
        )

    def test_width_scales_and_preserves_aspect(self, clip_mp4, tmp_path):
        """`width` resizes the output, keeping the source's aspect ratio.

        Test scenario:
            Asking for 160px from a 320x180 source gives a 160x90 GIF.
        """
        src, _ = clip_mp4
        out = tmp_path / "small.gif"
        anim_mod.gif_from_video(src, str(out), fps=12, width=160)
        with Image.open(out) as handle:
            assert handle.size == (160, 90), f"unexpected size {handle.size}"

    def test_warns_on_a_chroma_subsampled_source(self, tmp_path):
        """A 4:2:0 source warns that colour detail is already gone.

        Test scenario:
            `save_animation` writes `yuv420p` by default, which discards colour
            resolution before the GIF palette ever runs. Deriving a GIF from
            such a file must say so rather than silently under-delivering.
        """
        frames = _clip_frames()
        fig, anim = _clip_animation(frames)
        src = tmp_path / "subsampled.mp4"
        save_animation(anim, str(src), fps=12)
        plt.close(fig)

        with pytest.warns(UserWarning, match="yuv420p|reduced resolution"):
            anim_mod.gif_from_video(str(src), str(tmp_path / "out.gif"), fps=12)

    def test_full_chroma_source_does_not_warn(self, clip_mp4, tmp_path):
        """A `yuv444p` source is the recommended input and stays quiet.

        Test scenario:
            The warning must not fire for a source that kept its colour
            resolution, or it would be noise.
        """
        src, _ = clip_mp4
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            anim_mod.gif_from_video(src, str(tmp_path / "out.gif"), fps=12)

    def test_decoder_is_closed_when_iteration_stops_early(self, monkeypatch):
        """Abandoning the frame stream closes the decoder rather than leaking it.

        Args:
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            A caller that stops early -- or an exception raised mid-iteration --
            must not leave an ffmpeg child running. The decoder is replaced with
            a generator that records its own closure, and the frame iterator is
            abandoned after one frame.
        """
        closed = []

        def fake_reader(src, **kwargs):
            try:
                yield {"size": (2, 2), "pix_fmt": "yuv444p"}
                while True:
                    yield bytes(12)
            finally:
                closed.append(True)

        monkeypatch.setattr(anim_mod, "_read_video_frames", fake_reader)
        stream = anim_mod._iter_video_frames("ignored.mp4", 12, None)
        next(stream)
        stream.close()
        assert closed, "the decoder was not closed when the stream was abandoned"

    def test_missing_source_raises(self, tmp_path):
        """A source that does not exist raises `FileNotFoundError`.

        Test scenario:
            The failure surfaces up front, naming the path, rather than as a
            decoder error later.
        """
        with pytest.raises(FileNotFoundError, match="does not exist"):
            anim_mod.gif_from_video(
                str(tmp_path / "nope.mp4"), str(tmp_path / "out.gif")
            )

    @pytest.mark.parametrize("max_colors", [1, 0, 255, 300])
    def test_invalid_max_colors_raises(self, clip_mp4, tmp_path, max_colors):
        """A palette size outside ``2-254`` raises `ValueError`.

        Args:
            clip_mp4: The rendered source fixture.
            tmp_path: pytest temp directory.
            max_colors: The out-of-range palette size under test.

        Test scenario:
            Two entries are reserved for pure black and white, so 254 is the
            ceiling; fewer than two colours is not a palette.
        """
        src, _ = clip_mp4
        with pytest.raises(ValueError, match="max_colors must be"):
            anim_mod.gif_from_video(
                src, str(tmp_path / "out.gif"), max_colors=max_colors
            )

    def test_width_matching_the_source_skips_resizing(self, clip_mp4, tmp_path):
        """Asking for the source's own width leaves the frames untouched.

        Test scenario:
            The resize is skipped when it would be a no-op, and the output
            keeps the source dimensions.
        """
        src, _ = clip_mp4
        out = tmp_path / "same.gif"
        anim_mod.gif_from_video(src, str(out), fps=12, width=_CLIP_W)
        with Image.open(out) as handle:
            assert handle.size == (_CLIP_W, _CLIP_H), f"unexpected size {handle.size}"

    def test_source_yielding_no_frames_raises(self, clip_mp4, tmp_path, monkeypatch):
        """A video that decodes to nothing raises a clear `ValueError`.

        Args:
            clip_mp4: The rendered source fixture.
            tmp_path: pytest temp directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            A truncated or empty stream would otherwise fail deep inside Pillow
            with an IndexError; the decoder is stubbed to yield only metadata so
            the guard is exercised directly.
        """
        src, _ = clip_mp4
        import imageio_ffmpeg

        def only_meta(path, **kwargs):
            yield {"size": (4, 4), "pix_fmt": "yuv444p"}

        monkeypatch.setattr(imageio_ffmpeg, "read_frames", only_meta)
        with pytest.raises(ValueError, match="yielded no frames"):
            anim_mod.gif_from_video(src, str(tmp_path / "out.gif"))

    def test_max_colors_limits_the_palette(self, clip_mp4, tmp_path):
        """A small `max_colors` narrows the colours actually used.

        Test scenario:
            Quantising to 8 entries must yield far fewer distinct colours than
            the default 254, proving the argument reaches the palette builder
            rather than being ignored.
        """
        src, _ = clip_mp4
        narrow = tmp_path / "narrow.gif"
        anim_mod.gif_from_video(src, str(narrow), fps=6, max_colors=8)
        union = set()
        for frame in _decode(narrow):
            union |= {tuple(pixel) for pixel in frame.reshape(-1, 3)}
        assert len(union) <= 16, (
            f"max_colors=8 should keep the palette tiny, saw {len(union)} colours"
        )

    @pytest.mark.parametrize("kwargs", [{"fps": 0}, {"fps": -1}, {"width": 0}])
    def test_invalid_sampling_arguments_raise(self, clip_mp4, tmp_path, kwargs):
        """Non-positive `fps` or `width` raise `ValueError`.

        Args:
            clip_mp4: The rendered source fixture.
            tmp_path: pytest temp directory.
            kwargs: The invalid argument under test.

        Test scenario:
            Both are rejected before any decoding starts.
        """
        src, _ = clip_mp4
        with pytest.raises(ValueError, match="must be positive"):
            anim_mod.gif_from_video(src, str(tmp_path / "out.gif"), **kwargs)


class TestIsChromaSubsampled:
    """Tests for `_is_chroma_subsampled`."""

    @pytest.mark.parametrize(
        "pix_fmt, expected",
        [
            ("yuv420p", True),
            ("yuv422p", True),
            ("yuvj420p", True),
            ("YUV420P", True),
            ("nv12", True),
            ("nv21", True),
            ("yuv444p", False),
            ("yuvj444p", False),
            ("rgb24", False),
            ("gbrp", False),
            ("", False),
            (None, False),
        ],
    )
    def test_classifies_pixel_formats(self, pix_fmt, expected):
        """Only sub-4:4:4 YUV and the NV planar formats count as subsampled.

        Args:
            pix_fmt: The pixel format string under test.
            expected: Whether it should be reported as chroma-subsampled.

        Test scenario:
            RGB-family formats keep full colour resolution, `yuv444p` keeps it
            too, and an unreported format must not trigger a false warning.
            `nv12` / `nv21` are 4:2:0 despite not starting with ``yuv``.
        """
        result = anim_mod._is_chroma_subsampled(pix_fmt)
        assert result is expected, (
            f"{pix_fmt!r} should be {'subsampled' if expected else 'full chroma'}, got {result}"
        )


class TestBuildClipPalette:
    """Tests for `build_clip_palette`."""

    @staticmethod
    def _solid(color, size=(12, 12)):
        """Build a solid-colour RGB frame.

        Args:
            color: The ``(r, g, b)`` fill.
            size: The frame size.

        Returns:
            PIL.Image.Image: The filled frame.
        """
        return Image.new("RGB", size, color)

    def test_reserves_pure_black_and_white(self):
        """Entries 254 and 255 are pinned to black and white.

        Test scenario:
            A colourful clip would otherwise spend every slot on its own
            colours, leaving a single-colour overlay to snap to the nearest
            photographic neighbour. The last two entries are held back.
        """
        frames = [self._solid((200, 30, 30)), self._solid((30, 200, 30))]
        palette = anim_mod.build_clip_palette(frames)
        entries = palette.getpalette()
        assert entries[254 * 3 : 254 * 3 + 3] == [0, 0, 0], "entry 254 should be black"
        assert entries[255 * 3 : 255 * 3 + 3] == [255, 255, 255], (
            "entry 255 should be white"
        )

    def test_returns_a_palette_mode_image(self):
        """The result is a ``"P"``-mode image usable as a quantize palette.

        Test scenario:
            `Image.quantize(palette=...)` requires a palette-mode image, so the
            builder must hand one back rather than a raw list of entries.
        """
        palette = anim_mod.build_clip_palette([self._solid((10, 20, 30))])
        assert palette.mode == "P", f"expected a P-mode image, got {palette.mode}"
        assert len(palette.getpalette()) == 768, "palette should carry 256 RGB entries"

    def test_honours_the_colors_argument(self):
        """A smaller `colors` budget reserves black/white at that offset.

        Test scenario:
            Passing ``colors=16`` quantises to 16 entries and pins black and
            white immediately after them, so a caller asking for a small
            palette still gets the reserved pair.
        """
        frames = [self._solid((200, 30, 30)), self._solid((30, 30, 200))]
        entries = anim_mod.build_clip_palette(frames, colors=16).getpalette()
        assert entries[16 * 3 : 16 * 3 + 3] == [0, 0, 0], (
            "black should follow the budget"
        )
        assert entries[17 * 3 : 17 * 3 + 3] == [255, 255, 255], (
            "white should follow black"
        )

    def test_palette_spans_the_whole_clip_not_one_frame(self):
        """A colour that appears only in a late frame still reaches the palette.

        Test scenario:
            The palette is built from a montage of every frame. A distinctive
            colour introduced in the last frame must therefore be represented,
            which a first-frame-only palette would miss.
        """
        frames = [self._solid((0, 0, 0))] * 4 + [self._solid((255, 0, 255))]
        entries = anim_mod.build_clip_palette(frames).getpalette()
        triples = [tuple(entries[i : i + 3]) for i in range(0, 254 * 3, 3)]
        assert any(
            abs(r - 255) < 30 and g < 30 and abs(b - 255) < 30 for r, g, b in triples
        ), "the late frame's magenta is absent from the clip-wide palette"

    def test_single_frame_clip(self):
        """A one-frame clip builds a palette without special-casing.

        Test scenario:
            The montage degenerates to a single tile; the builder must still
            return a usable palette.
        """
        palette = anim_mod.build_clip_palette([self._solid((123, 45, 67))])
        assert palette.mode == "P", "a single-frame clip should still yield a palette"

    def test_tiny_frames_do_not_collapse_to_zero(self):
        """Frames smaller than the montage divisor still produce a tile.

        Test scenario:
            A 2x2 frame divided by the montage divisor floors to 0, which would
            be an invalid image size; the builder clamps each tile to at least
            one pixel.
        """
        palette = anim_mod.build_clip_palette([self._solid((1, 2, 3), size=(2, 2))])
        assert palette.mode == "P", "a sub-divisor frame should not break the montage"


class TestQuantizeToPalette:
    """Tests for `quantize_to_palette`."""

    def test_returns_one_palette_frame_per_input(self):
        """Every input frame comes back as a ``"P"``-mode image.

        Test scenario:
            The count is preserved and each frame is converted, so the writer
            can hand the list straight to Pillow.
        """
        frames = [Image.new("RGB", (8, 8), c) for c in ((255, 0, 0), (0, 255, 0))]
        palette = anim_mod.build_clip_palette(frames)
        result = anim_mod.quantize_to_palette(frames, palette)
        assert len(result) == len(frames), (
            f"expected {len(frames)} frames, got {len(result)}"
        )
        assert all(f.mode == "P" for f in result), "every frame should be palette-mode"

    def test_frames_share_one_palette(self):
        """All quantised frames carry the same colour table.

        Test scenario:
            A shared table is what keeps constant regions byte-stable between
            frames, so the palettes must be identical, not merely similar.
        """
        frames = [Image.new("RGB", (8, 8), c) for c in ((255, 0, 0), (0, 0, 255))]
        result = anim_mod.quantize_to_palette(
            frames, anim_mod.build_clip_palette(frames)
        )
        first = result[0].getpalette()
        assert all(f.getpalette() == first for f in result[1:]), (
            "frames do not share a single palette"
        )


class TestWritePillowAnimation:
    """Tests for `_write_pillow_animation`."""

    @pytest.fixture
    def palette_frames(self):
        """Two palette-mode frames sharing one table.

        Returns:
            list: The quantised frames.
        """
        frames = [Image.new("RGB", (8, 8), c) for c in ((255, 0, 0), (0, 0, 255))]
        return anim_mod.quantize_to_palette(frames, anim_mod.build_clip_palette(frames))

    @pytest.mark.parametrize("fps, expected", [(4, 250), (10, 100), (2, 500)])
    def test_duration_follows_fps(self, palette_frames, tmp_path, fps, expected):
        """Frame duration is the millisecond reciprocal of `fps`.

        Args:
            palette_frames: The quantised-frames fixture.
            tmp_path: pytest temp directory.
            fps: Playback rate under test.
            expected: The per-frame duration it implies.

        Test scenario:
            GIF stores a per-frame delay, so fps has to be converted; 4 fps is
            250 ms per frame.
        """
        out = tmp_path / "d.gif"
        anim_mod._write_pillow_animation(palette_frames, str(out), fps, 0, True)
        with Image.open(out) as handle:
            assert handle.info["duration"] == expected, (
                f"fps={fps} should give {expected}ms, got {handle.info['duration']}"
            )

    def test_loop_is_written(self, palette_frames, tmp_path):
        """A non-zero `loop` reaches the written file.

        Test scenario:
            `loop=3` must be recorded rather than silently defaulting to the
            forever-loop Pillow's writer hardcodes.
        """
        out = tmp_path / "l.gif"
        anim_mod._write_pillow_animation(palette_frames, str(out), 5, 3, True)
        with Image.open(out) as handle:
            assert handle.info.get("loop") == 3, (
                f"expected loop=3, got {handle.info.get('loop')}"
            )

    def test_all_frames_are_written(self, palette_frames, tmp_path):
        """Every frame ends up in the file, not just the first.

        Test scenario:
            The first frame is saved with the rest appended; a mistake there
            would silently produce a single-frame GIF.
        """
        out = tmp_path / "n.gif"
        anim_mod._write_pillow_animation(palette_frames, str(out), 5, 0, True)
        assert len(_decode(out)) == len(palette_frames), "frame count mismatch"
