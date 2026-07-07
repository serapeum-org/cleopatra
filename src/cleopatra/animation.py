"""Save/embed helpers for matplotlib animations (glyph-independent).

These helpers operate on *any* `matplotlib.animation.FuncAnimation`, not
only on a `Glyph`'s internal `self.anim`. Saving or embedding an animation
is generic matplotlib machinery — it works on a sine wave, stock prices, or
a map — so it lives here alongside the glyph classes that produce
animations. Downstream packages that build their own `FuncAnimation` can
reuse cleopatra's writer/format handling instead of re-rolling temp-file +
writer + `IPython.display` glue.

`Glyph.save_animation` delegates to `save_animation` below, so the
writer/format logic has a single source of truth.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from typing import TYPE_CHECKING

import matplotlib as mpl
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

if TYPE_CHECKING:  # import only for type checkers; IPython stays optional
    from IPython.display import Image

#: Container formats `save_animation` can write. GIF and (animated) WebP use
#: Pillow (`_OptimizedPillowWriter`); mov/avi/mp4 require FFmpeg (`FFMpegWriter`).
#: WebP is typically 3-5x smaller than GIF for photographic/satellite frames.
SUPPORTED_VIDEO_FORMAT = ["gif", "mov", "avi", "mp4", "webp"]

#: Formats written by Pillow rather than FFmpeg.
_PILLOW_FORMATS = {"gif", "webp"}


def _ensure_ffmpeg_available() -> None:
    """Make sure matplotlib can find an ffmpeg binary to shell out to.

    matplotlib's `FFMpegWriter` runs the ffmpeg *binary* named by
    `matplotlib.rcParams["animation.ffmpeg_path"]` (default ``"ffmpeg"``,
    resolved on `PATH`). If that binary is not found, fall back to the static
    ffmpeg that `imageio-ffmpeg` bundles, so mp4/mov/avi export works with no
    separate system install. A system ffmpeg on `PATH` still takes precedence.
    If the rcParam was set to an explicit path that no longer resolves, a
    `RuntimeWarning` is emitted before falling back, so an overridden user
    choice is never discarded silently.

    Raises:
        FileNotFoundError: If neither a system ffmpeg nor `imageio-ffmpeg`'s
            bundled binary can be located.
    """
    configured = mpl.rcParams["animation.ffmpeg_path"]
    # Already usable: an absolute path that exists, or a name found on PATH.
    if os.path.isfile(configured) or shutil.which(configured):
        return
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as e:  # pragma: no cover - imageio-ffmpeg is a dep
        raise FileNotFoundError(
            "FFmpeg not found on PATH and imageio-ffmpeg is not installed. "
            "Install imageio-ffmpeg (ships a bundled ffmpeg) or download "
            "ffmpeg from https://ffmpeg.org/ and add it to your PATH."
        ) from e
    bundled = imageio_ffmpeg.get_ffmpeg_exe()
    # Only the matplotlib default (``"ffmpeg"``) is overridden silently; an
    # explicit path the user configured is theirs, so warn before replacing it.
    if configured not in ("ffmpeg", "ffmpeg.exe"):
        warnings.warn(
            f"Configured ffmpeg binary {configured!r} was not found; falling "
            f"back to the imageio-ffmpeg bundled binary at {bundled!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
    mpl.rcParams["animation.ffmpeg_path"] = bundled


class _OptimizedPillowWriter(PillowWriter):
    """`PillowWriter` that writes optimised GIFs with a configurable loop.

    matplotlib's stock `PillowWriter` hardcodes ``loop=0`` and never passes
    ``optimize`` to `PIL.Image.save`, so GIFs come out unoptimised — needlessly
    large for photographic/satellite frames. This subclass forwards ``optimize``
    and ``loop`` while reusing the parent's frame-grabbing logic.

    Args:
        optimize: Run Pillow's GIF optimisation pass (palette + delta frames).
        loop: Number of times the GIF loops; ``0`` means loop forever (Pillow's
            convention).
    """

    def __init__(self, *args, optimize: bool = True, loop: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._optimize = optimize
        self._loop = loop

    def finish(self):
        self._frames[0].save(
            self.outfile,
            save_all=True,
            append_images=self._frames[1:],
            duration=int(1000 / self.fps),
            loop=self._loop,
            optimize=self._optimize,
        )


#: ffmpeg video filter that rounds the frame up to an even width/height.
#: libx264 refuses odd dimensions, so this is always applied to video output.
_EVEN_PAD_FILTER = "pad=ceil(iw/2)*2:ceil(ih/2)*2"


def _build_ffmpeg_extra_args(
    pix_fmt: str,
    crf: int | None,
    preset: str | None,
    extra_args: list[str] | None,
) -> list[str]:
    """Assemble the ffmpeg ``extra_args`` list for a video export.

    Combines the mandatory even-dimension pad filter with an explicit pixel
    format and any caller-supplied CRF, preset, or raw ffmpeg flags. A caller
    ``-vf`` filter is merged into a single chain — ffmpeg honours only the last
    ``-vf`` — with the pad applied last so the frame ends up even whatever the
    caller's filters produce.

    Args:
        pix_fmt: Pixel format passed as ``-pix_fmt`` (e.g. ``"yuv420p"``).
        crf: Constant Rate Factor; appended as ``-crf`` when not ``None``.
        preset: libx264 speed/size preset; appended as ``-preset`` when set.
        extra_args: Extra ffmpeg flags. A ``-vf`` pair here is merged into the
            pad chain; everything else is passed through unchanged.

    Returns:
        The assembled argument list, always starting with the merged ``-vf``
        chain followed by ``-pix_fmt``.

    Examples:
        - Defaults produce just the pad filter and pixel format:
            ```python
            >>> from cleopatra.animation import _build_ffmpeg_extra_args
            >>> _build_ffmpeg_extra_args("yuv420p", None, None, None)
            ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p']

            ```
        - A CRF and preset are appended after the pixel format:
            ```python
            >>> from cleopatra.animation import _build_ffmpeg_extra_args
            >>> _build_ffmpeg_extra_args("yuv420p", 26, "slow", None)[4:]
            ['-crf', '26', '-preset', 'slow']

            ```
        - A caller ``-vf`` is merged into one chain with the pad applied last:
            ```python
            >>> from cleopatra.animation import _build_ffmpeg_extra_args
            >>> _build_ffmpeg_extra_args("yuv420p", None, None, ["-vf", "scale=320:-1"])[:2]
            ['-vf', 'scale=320:-1,pad=ceil(iw/2)*2:ceil(ih/2)*2']

            ```
    """
    user_args = list(extra_args) if extra_args else []
    vf_filters: list[str] = []
    passthrough: list[str] = []
    i = 0
    while i < len(user_args):
        if user_args[i] == "-vf" and i + 1 < len(user_args):
            vf_filters.append(user_args[i + 1])
            i += 2
        else:
            passthrough.append(user_args[i])
            i += 1
    vf_filters.append(_EVEN_PAD_FILTER)

    built = ["-vf", ",".join(vf_filters), "-pix_fmt", pix_fmt]
    if crf is not None:
        built += ["-crf", str(crf)]
    if preset is not None:
        built += ["-preset", preset]
    built += passthrough
    return built


def save_animation(
    anim: FuncAnimation,
    path: str | os.PathLike,
    fps: int = 2,
    *,
    crf: int | None = None,
    bitrate: int | None = None,
    codec: str | None = None,
    preset: str | None = None,
    pix_fmt: str = "yuv420p",
    dpi: int | None = None,
    optimize: bool = True,
    loop: int = 0,
    extra_args: list[str] | None = None,
) -> str:
    """Save any `FuncAnimation` to a file.

    The output format is determined by the file extension. GIF and animated
    WebP use an optimising Pillow writer; mov/avi/mp4 use FFmpeg. FFmpeg is
    located on `PATH` when present and otherwise falls back to the binary
    bundled with `imageio-ffmpeg`, so video export works with no separate
    install. WebP is typically 3-5x smaller than GIF for photographic frames.

    For the FFmpeg formats the frame is automatically padded up to an even
    width/height (libx264 rejects odd dimensions) and encoded with
    ``pix_fmt=yuv420p`` for universal playback. GIF/WebP output is written with
    Pillow's ``optimize`` pass enabled and loops forever.

    Args:
        anim: The animation to save.
        path: Output file path, as a `str` or `os.PathLike` (e.g. a
            `pathlib.Path`). Extension determines format.
            Supported: gif, mov, avi, mp4, webp.
        fps: Frames per second. Default is 2.
        crf: Constant Rate Factor for the ffmpeg formats (lower is higher
            quality/larger; ~18-28 is typical). Mutually exclusive with
            ``bitrate``. Ignored for GIF/WebP. ``None`` uses the encoder default.
        bitrate: Target bitrate in kbit/s for the ffmpeg formats. Mutually
            exclusive with ``crf``. Ignored for GIF/WebP. ``None`` lets the
            encoder choose.
        codec: ffmpeg codec (e.g. ``"libx264"``). ``None`` uses matplotlib's
            default. Ignored for GIF/WebP.
        preset: libx264 speed/size preset (e.g. ``"slow"``). Ignored for
            GIF/WebP.
        pix_fmt: Pixel format for the ffmpeg formats. Defaults to
            ``"yuv420p"`` for universal playback. Ignored for GIF/WebP.
        dpi: Resolution in dots per inch. ``None`` uses the figure's dpi.
        optimize: GIF/WebP only — run Pillow's optimisation pass. Default
            ``True``.
        loop: GIF/WebP only — number of times to loop; ``0`` loops forever.
        extra_args: Extra ffmpeg flags. A ``-vf`` filter here is merged with
            the automatic even-dimension pad. Ignored for GIF/WebP.

    Returns:
        The output path as a `str` (the `os.fspath` of `path`),
        convenient for chaining. Note a `pathlib.Path` argument comes
        back as its string form, not the original object.

    Raises:
        ValueError: If the file format is not supported, or if both ``crf``
            and ``bitrate`` are given (competing rate-control modes).
        FileNotFoundError: If a video format is requested but neither a system
            FFmpeg nor imageio-ffmpeg's bundled binary can be found.

    Examples:
        - Save a tiny animation to a GIF; the call returns the path it wrote:
            ```python
            >>> import os, shutil, tempfile, matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from pathlib import Path
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.animation import save_animation
            >>> tmp = tempfile.mkdtemp()
            >>> fig, ax = plt.subplots()
            >>> (line,) = ax.plot([0, 1], [0, 0])
            >>> anim = FuncAnimation(fig, lambda i: (line,), frames=2)
            >>> path = os.path.join(tmp, "wave.gif")
            >>> save_animation(anim, path) == path
            True
            >>> Path(path).read_bytes()[:6] in (b"GIF87a", b"GIF89a")
            True
            >>> plt.close(fig)
            >>> shutil.rmtree(tmp)

            ```
        - The extension is matched case-insensitively, so ``.GIF`` also works:
            ```python
            >>> import os, shutil, tempfile, matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.animation import save_animation
            >>> tmp = tempfile.mkdtemp()
            >>> fig, ax = plt.subplots()
            >>> (line,) = ax.plot([0, 1], [0, 0])
            >>> anim = FuncAnimation(fig, lambda i: (line,), frames=2)
            >>> save_animation(anim, os.path.join(tmp, "WAVE.GIF")).endswith("WAVE.GIF")
            True
            >>> plt.close(fig)
            >>> shutil.rmtree(tmp)

            ```
        - An unsupported extension raises ``ValueError`` before writing (here
          the animation is rendered once first, so nothing is left dangling):
            ```python
            >>> import os, shutil, tempfile, matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.animation import save_animation
            >>> tmp = tempfile.mkdtemp()
            >>> fig, ax = plt.subplots()
            >>> (line,) = ax.plot([0, 1], [0, 0])
            >>> anim = FuncAnimation(fig, lambda i: (line,), frames=2)
            >>> _ = save_animation(anim, os.path.join(tmp, "ok.gif"))
            >>> save_animation(anim, "movie.webm")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: ...not supported...
            >>> plt.close(fig)
            >>> shutil.rmtree(tmp)

            ```

    See Also:
        to_gif: Render an animation to in-memory GIF bytes instead of a file.
        embed_gif: Wrap an animation as an ``IPython.display.Image``.
    """
    # Accept str or os.PathLike (e.g. pathlib.Path): normalise to a string
    # once so the extension parse and the writers both get a plain path.
    path = os.fspath(path)
    video_format = os.path.splitext(path)[1].lstrip(".").lower()
    if not video_format:
        raise ValueError(
            f"The output path {path!r} has no file extension; the output "
            f"format is taken from the extension, so use one of "
            f"{SUPPORTED_VIDEO_FORMAT}."
        )
    if video_format not in SUPPORTED_VIDEO_FORMAT:
        raise ValueError(
            f"The given extension {video_format} implies a format that is "
            f"not supported, only {SUPPORTED_VIDEO_FORMAT} are supported"
        )

    save_kwargs = {} if dpi is None else {"dpi": dpi}

    if video_format in _PILLOW_FORMATS:
        anim.save(
            path,
            writer=_OptimizedPillowWriter(fps=fps, optimize=optimize, loop=loop),
            **save_kwargs,
        )
    else:
        if crf is not None and bitrate is not None:
            raise ValueError(
                "Pass either crf or bitrate, not both: they are competing "
                "rate-control modes for the encoder."
            )
        _ensure_ffmpeg_available()
        writer_kwargs = {
            "fps": fps,
            "extra_args": _build_ffmpeg_extra_args(pix_fmt, crf, preset, extra_args),
        }
        if bitrate is not None:
            writer_kwargs["bitrate"] = bitrate
        if codec is not None:
            writer_kwargs["codec"] = codec
        try:
            anim.save(path, writer=FFMpegWriter(**writer_kwargs), **save_kwargs)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "FFmpeg not found. Please visit https://ffmpeg.org/ "
                "and download a version compatible with your OS."
            ) from e
    return path


def to_bytes(anim: FuncAnimation, fmt: str = "gif", fps: int = 2, **kwargs) -> bytes:
    """Render a `FuncAnimation` to in-memory bytes in any supported format.

    Renders to a temporary file (the writers need a real path) and reads it
    back, leaving nothing on disk. Handy for embedding in a notebook or
    serving over HTTP.

    Args:
        anim: The animation to render.
        fmt: Output format — any member of ``SUPPORTED_VIDEO_FORMAT`` (e.g.
            ``"gif"``, ``"mp4"``, ``"webp"``). A leading dot is tolerated.
        fps: Frames per second. Default is 2.
        **kwargs: Extra keyword arguments forwarded to `save_animation`
            (e.g. ``crf``, ``codec``, ``loop``).

    Returns:
        The encoded bytes of the animation in the requested format.

    Raises:
        ValueError: If ``fmt`` is not a supported format.

    Examples:
        - Render to GIF bytes and inspect the payload:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.animation import to_bytes
            >>> fig, ax = plt.subplots()
            >>> (line,) = ax.plot([0, 1], [0, 0])
            >>> anim = FuncAnimation(fig, lambda i: (line,), frames=2)
            >>> data = to_bytes(anim, fmt="gif")
            >>> data[:6] in (b"GIF87a", b"GIF89a")
            True
            >>> plt.close(fig)

            ```
        - Render to animated WebP and confirm the container magic bytes:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.animation import to_bytes
            >>> fig, ax = plt.subplots()
            >>> (line,) = ax.plot([0, 1], [0, 0])
            >>> anim = FuncAnimation(fig, lambda i: (line,), frames=2)
            >>> data = to_bytes(anim, fmt="webp")
            >>> data[:4] == b"RIFF" and data[8:12] == b"WEBP"
            True
            >>> plt.close(fig)

            ```
        - An unsupported format raises ``ValueError``:
            ```python
            >>> from unittest.mock import MagicMock
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.animation import to_bytes
            >>> to_bytes(MagicMock(spec=FuncAnimation), fmt="webm")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: ...not supported...

            ```

    See Also:
        to_gif: Convenience wrapper for GIF bytes.
        to_mp4: Convenience wrapper for MP4 bytes.
        save_animation: Write an animation directly to a file path.
    """
    fmt = fmt.lstrip(".").lower()
    if fmt not in SUPPORTED_VIDEO_FORMAT:
        raise ValueError(
            f"The format {fmt!r} is not supported, only "
            f"{SUPPORTED_VIDEO_FORMAT} are supported"
        )
    # Close our handle immediately so the writer can reopen the path; this
    # makes the handle lifecycle explicit (no reliance on GC) and avoids a
    # PermissionError when reopening on Windows.
    fd, tmp = tempfile.mkstemp(suffix=f".{fmt}")
    os.close(fd)
    try:
        save_animation(anim, tmp, fps=fps, **kwargs)
        with open(tmp, "rb") as fh:
            return fh.read()
    finally:
        os.remove(tmp)


def to_gif(anim: FuncAnimation, fps: int = 2, **kwargs) -> bytes:
    """Render a `FuncAnimation` to in-memory GIF bytes.

    Handy for embedding in a notebook or serving over HTTP without leaving
    a file on disk. Thin wrapper around `to_bytes` with ``fmt="gif"``.

    Args:
        anim: The animation to render.
        fps: Frames per second. Default is 2.
        **kwargs: Extra keyword arguments forwarded to `save_animation`
            (e.g. ``optimize``, ``loop``).

    Returns:
        The GIF-encoded bytes of the animation.

    Examples:
        - Render an animation to GIF bytes and inspect the payload:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.animation import to_gif
            >>> fig, ax = plt.subplots()
            >>> (line,) = ax.plot([0, 1], [0, 0])
            >>> anim = FuncAnimation(fig, lambda i: (line,), frames=2)
            >>> data = to_gif(anim)
            >>> data[:6] in (b"GIF87a", b"GIF89a")
            True
            >>> len(data) > 0
            True
            >>> plt.close(fig)

            ```
        - A higher ``fps`` still yields self-contained bytes you can serve over
          HTTP or write yourself, without leaving a temp file behind:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.animation import to_gif
            >>> fig, ax = plt.subplots()
            >>> (line,) = ax.plot([0, 1], [0, 0])
            >>> anim = FuncAnimation(fig, lambda i: (line,), frames=3)
            >>> payload = to_gif(anim, fps=5)
            >>> payload.startswith((b"GIF87a", b"GIF89a"))
            True
            >>> plt.close(fig)

            ```

    See Also:
        to_bytes: Render to bytes in any supported format.
        save_animation: Write an animation directly to a file path.
        embed_gif: Wrap these bytes as an ``IPython.display.Image``.
    """
    return to_bytes(anim, fmt="gif", fps=fps, **kwargs)


def to_mp4(anim: FuncAnimation, fps: int = 2, **kwargs) -> bytes:
    """Render a `FuncAnimation` to in-memory MP4 (H.264) bytes.

    Handy for embedding a compact, universally-playable clip or serving it
    over HTTP without leaving a file on disk. Thin wrapper around `to_bytes`
    with ``fmt="mp4"``; the frame is auto-padded to even dimensions and
    encoded ``yuv420p`` like every other MP4 export.

    Args:
        anim: The animation to render.
        fps: Frames per second. Default is 2.
        **kwargs: Extra keyword arguments forwarded to `save_animation`
            (e.g. ``crf``, ``bitrate``, ``codec``, ``preset``).

    Returns:
        The MP4-encoded bytes of the animation.

    Raises:
        FileNotFoundError: If neither a system FFmpeg nor imageio-ffmpeg's
            bundled binary can be found.

    Examples:
        - Render to MP4 bytes and confirm the ISO base-media ``ftyp`` box:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.animation import to_mp4
            >>> fig, ax = plt.subplots()
            >>> (line,) = ax.plot([0, 1], [0, 0])
            >>> anim = FuncAnimation(fig, lambda i: (line,), frames=2)
            >>> data = to_mp4(anim)
            >>> data[4:8] == b"ftyp"
            True
            >>> plt.close(fig)

            ```
        - Trade size for quality with a CRF and confirm non-empty output:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.animation import to_mp4
            >>> fig, ax = plt.subplots()
            >>> (line,) = ax.plot([0, 1], [0, 0])
            >>> anim = FuncAnimation(fig, lambda i: (line,), frames=2)
            >>> data = to_mp4(anim, crf=30, preset="veryfast")
            >>> len(data) > 0
            True
            >>> plt.close(fig)

            ```

    See Also:
        to_bytes: Render to bytes in any supported format.
        to_gif: Render an animation to in-memory GIF bytes.
        save_animation: Write an animation directly to a file path.
    """
    return to_bytes(anim, fmt="mp4", fps=fps, **kwargs)


def embed_gif(anim: FuncAnimation, fps: int = 2) -> Image:
    """Return an `IPython.display.Image` of the animation for inline display.

    IPython is imported lazily, so importing cleopatra never requires it.
    IPython ships with Jupyter, so any notebook already has it; outside a
    notebook the returned `Image` is not renderable anyway — use `to_gif`
    for raw bytes with no IPython dependency.

    Args:
        anim: The animation to embed.
        fps: Frames per second. Default is 2.

    Returns:
        An `IPython.display.Image` wrapping the rendered GIF, ready to be
        returned as the last expression of a notebook cell.

    Raises:
        ModuleNotFoundError: If IPython is not installed, with a hint to
            `pip install ipython` (or to use `to_gif` instead).

    Examples:
        - Wrap an animation as an inline image and read back its payload:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.animation import embed_gif
            >>> fig, ax = plt.subplots()
            >>> (line,) = ax.plot([0, 1], [0, 0])
            >>> anim = FuncAnimation(fig, lambda i: (line,), frames=2)
            >>> img = embed_gif(anim)
            >>> img.format
            'gif'
            >>> img.data[:6] in (b"GIF87a", b"GIF89a")
            True
            >>> plt.close(fig)

            ```
        - Returning the image as a cell's last expression renders it inline;
          a custom ``fps`` controls playback speed:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.animation import embed_gif
            >>> fig, ax = plt.subplots()
            >>> (line,) = ax.plot([0, 1], [0, 0])
            >>> anim = FuncAnimation(fig, lambda i: (line,), frames=2)
            >>> img = embed_gif(anim, fps=3)
            >>> len(img.data) > 0
            True
            >>> plt.close(fig)

            ```

    See Also:
        to_gif: Produce the underlying GIF bytes without IPython.
        save_animation: Write the animation to a file path instead.
    """
    try:
        from IPython.display import Image
    except ModuleNotFoundError as e:
        # Only remap when IPython itself is missing; if a sub-dependency of
        # IPython failed to import, surface that original error unchanged.
        if e.name and e.name.split(".")[0] != "IPython":
            raise
        raise ModuleNotFoundError(
            "embed_gif requires IPython for inline display. Install it with "
            "`pip install ipython` (already present in any Jupyter/IPython "
            "environment). For raw GIF bytes without IPython, use to_gif()."
        ) from e

    return Image(data=to_gif(anim, fps=fps), format="gif")
