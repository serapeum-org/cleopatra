"""Tests for cleopatra.animation glyph-independent save/embed helpers.

Covers `save_animation`, `to_gif`, and `embed_gif` operating on a plain
`matplotlib.animation.FuncAnimation` (no `Glyph` involved). A tiny 2-3 frame
animation is rendered on the Agg backend (set globally via `MPLBACKEND`).
"""

from __future__ import annotations

import builtins
import doctest
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from matplotlib.animation import FuncAnimation
from PIL import Image

import cleopatra.animation as anim_mod
from cleopatra.animation import (
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

        assert data[:4] == b"RIFF" and data[8:12] == b"WEBP", "not WebP bytes"

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
            ``cleopatra.animation`` must not expose an ``Image`` attribute,
            so the package never hard-depends on IPython at load time.
        """

        assert not hasattr(
            anim_mod, "Image"
        ), "IPython Image must not be imported at module load time"


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
        assert (
            width % 2 == 1 and height % 2 == 1
        ), f"fixture must be odd-sized to exercise the pad, got {width}x{height}"
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
        assert raw[:4] == b"RIFF" and raw[8:12] == b"WEBP", "not a WebP file"
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

        assert (
            mpl.rcParams["animation.ffmpeg_path"] == "ffmpeg"
        ), "system ffmpeg path should be left unchanged"
        assert (
            calls["n"] == 0
        ), "bundled binary must not be consulted when PATH resolves"

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

        assert (
            mpl.rcParams["animation.ffmpeg_path"] == "C:/bundled/ffmpeg.exe"
        ), "resolver should fall back to the imageio-ffmpeg binary"

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

        assert (
            mpl.rcParams["animation.ffmpeg_path"] == "C:/bundled/ffmpeg.exe"
        ), "should still fall back after warning"

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

        assert (
            captured.get("optimize") is False
        ), f"optimize flag not forwarded to Pillow: {captured}"


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
        assert (
            args[1] == "scale=320:-1,pad=ceil(iw/2)*2:ceil(ih/2)*2"
        ), f"caller filter not merged with pad: {args}"
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
        """`cleopatra.glyph` re-exports the same object, not a copy."""
        from cleopatra.glyph import SUPPORTED_VIDEO_FORMAT as glyph_constant

        assert (
            glyph_constant is SUPPORTED_VIDEO_FORMAT
        ), "glyph should re-import the constant, not redefine it"


def test_module_doctests_execute():
    """Run the module's docstring examples so they are exercised in CI.

    Pytest is not configured with ``--doctest-modules``, so docstring examples in
    ``src/`` would otherwise never run. This test executes them for
    ``cleopatra.animation`` (including the ``to_bytes``/``to_mp4`` magic-byte checks
    and the ``_build_ffmpeg_extra_args`` flag-merge examples) and fails if any
    example's output no longer matches.
    """
    try:
        results = doctest.testmod(anim_mod, verbose=False)
    finally:
        plt.close("all")
    assert (
        results.failed == 0
    ), f"{results.failed} doctest example(s) failed in animation"
    assert results.attempted > 0, (
        "no doctest examples were collected from animation; the module's docstring "
        "examples may have been moved or removed, silently dropping this coverage"
    )
