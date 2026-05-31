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
import tempfile

from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

#: Container formats `save_animation` can write. GIF uses `PillowWriter`;
#: the rest require FFmpeg (`FFMpegWriter`).
SUPPORTED_VIDEO_FORMAT = ["gif", "mov", "avi", "mp4"]


def save_animation(anim: FuncAnimation, path: str, fps: int = 2) -> str:
    """Save any `FuncAnimation` to a file.

    The output format is determined by the file extension. GIF uses
    `PillowWriter`; mov/avi/mp4 require FFmpeg to be installed.

    Args:
        anim: The animation to save.
        path: Output file path. Extension determines format.
            Supported: gif, mov, avi, mp4.
        fps: Frames per second. Default is 2.

    Returns:
        The `path` that was written (convenient for chaining).

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If a video format is requested but FFmpeg is
            not installed.

    Examples:
        - Check the supported video formats:
            ```python
            >>> from cleopatra.animation import SUPPORTED_VIDEO_FORMAT
            >>> sorted(SUPPORTED_VIDEO_FORMAT)
            ['avi', 'gif', 'mov', 'mp4']

            ```
    """
    video_format = path.rsplit(".", 1)[-1].lower()
    if video_format not in SUPPORTED_VIDEO_FORMAT:
        raise ValueError(
            f"The given extension {video_format} implies a format that is "
            f"not supported, only {SUPPORTED_VIDEO_FORMAT} are supported"
        )

    if video_format == "gif":
        anim.save(path, writer=PillowWriter(fps=fps))
    else:
        try:
            anim.save(path, writer=FFMpegWriter(fps=fps, bitrate=1800))
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


def embed_gif(anim: FuncAnimation, fps: int = 2):
    """Return an `IPython.display.Image` of the animation for inline display.

    IPython is imported lazily so it stays an optional, notebook-only
    dependency.

    Args:
        anim: The animation to embed.
        fps: Frames per second. Default is 2.

    Returns:
        An `IPython.display.Image` wrapping the rendered GIF, ready to be
        returned as the last expression of a notebook cell.
    """
    from IPython.display import Image  # optional dep, only for inline display

    return Image(data=to_gif(anim, fps=fps), format="gif")
