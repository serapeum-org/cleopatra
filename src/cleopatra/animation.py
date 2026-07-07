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
from typing import TYPE_CHECKING

import matplotlib as mpl
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

if TYPE_CHECKING:  # import only for type checkers; IPython stays optional
    from IPython.display import Image

#: Container formats `save_animation` can write. GIF uses `PillowWriter`;
#: the rest require FFmpeg (`FFMpegWriter`).
SUPPORTED_VIDEO_FORMAT = ["gif", "mov", "avi", "mp4"]


def _ensure_ffmpeg_available() -> None:
    """Make sure matplotlib can find an ffmpeg binary to shell out to.

    matplotlib's `FFMpegWriter` runs the ffmpeg *binary* named by
    `matplotlib.rcParams["animation.ffmpeg_path"]` (default ``"ffmpeg"``,
    resolved on `PATH`). If that binary is not found, fall back to the static
    ffmpeg that `imageio-ffmpeg` bundles, so mp4/mov/avi export works with no
    separate system install. A system ffmpeg on `PATH` still takes precedence.

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
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


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


def save_animation(
    anim: FuncAnimation, path: str | os.PathLike, fps: int = 2
) -> str:
    """Save any `FuncAnimation` to a file.

    The output format is determined by the file extension. GIF uses an
    optimising `PillowWriter`; mov/avi/mp4 use FFmpeg. FFmpeg is located on
    `PATH` when present and otherwise falls back to the binary bundled with
    `imageio-ffmpeg`, so video export works with no separate install.

    For the FFmpeg formats the frame is automatically padded up to an even
    width/height (libx264 rejects odd dimensions) and encoded with
    ``pix_fmt=yuv420p`` for universal playback. GIF output is written with
    Pillow's ``optimize`` pass enabled and loops forever.

    Args:
        anim: The animation to save.
        path: Output file path, as a `str` or `os.PathLike` (e.g. a
            `pathlib.Path`). Extension determines format.
            Supported: gif, mov, avi, mp4.
        fps: Frames per second. Default is 2.

    Returns:
        The output path as a `str` (the `os.fspath` of `path`),
        convenient for chaining. Note a `pathlib.Path` argument comes
        back as its string form, not the original object.

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If a video format is requested but FFmpeg is
            not installed.

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

    if video_format == "gif":
        anim.save(path, writer=_OptimizedPillowWriter(fps=fps))
    else:
        _ensure_ffmpeg_available()
        # libx264 requires even width/height, so a figure whose pixel size is
        # odd otherwise dies with "height not divisible by 2". Pad the frame up
        # to the next even size so any figure encodes. Set pix_fmt=yuv420p
        # explicitly so the output plays everywhere (browsers, QuickTime); it
        # is otherwise one matplotlib default away from an unplayable file.
        extra_args = [
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
        ]
        try:
            anim.save(
                path,
                writer=FFMpegWriter(fps=fps, bitrate=1800, extra_args=extra_args),
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "FFmpeg not found. Please visit https://ffmpeg.org/ "
                "and download a version compatible with your OS."
            ) from e
    return path


def to_gif(anim: FuncAnimation, fps: int = 2) -> bytes:
    """Render a `FuncAnimation` to in-memory GIF bytes.

    Handy for embedding in a notebook or serving over HTTP without leaving
    a file on disk.

    Args:
        anim: The animation to render.
        fps: Frames per second. Default is 2.

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
        save_animation: Write an animation directly to a file path.
        embed_gif: Wrap these bytes as an ``IPython.display.Image``.
    """
    # Close our handle immediately so the writer can reopen the path; this
    # makes the handle lifecycle explicit (no reliance on GC) and avoids a
    # PermissionError when reopening on Windows.
    fd, tmp = tempfile.mkstemp(suffix=".gif")
    os.close(fd)
    try:
        save_animation(anim, tmp, fps=fps)
        with open(tmp, "rb") as fh:
            return fh.read()
    finally:
        os.remove(tmp)


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
