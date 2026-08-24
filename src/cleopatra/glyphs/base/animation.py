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

import itertools
import os
import shutil
import tempfile
import warnings
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from PIL import Image as PILImage

if TYPE_CHECKING:  # import only for type checkers; IPython stays optional
    from IPython.display import Image

#: Container formats `save_animation` can write. GIF and (animated) WebP use
#: Pillow (`_OptimizedPillowWriter`); mov/avi/mp4 require FFmpeg (`FFMpegWriter`).
#: WebP is typically 3-5x smaller than GIF for photographic/satellite frames.
SUPPORTED_VIDEO_FORMAT = ["gif", "mov", "avi", "mp4", "webp"]

#: Formats written by Pillow rather than FFmpeg.
_PILLOW_FORMATS = {"gif", "webp"}

#: Palette entries a GIF clip's shared colour table is quantised to. Two of the
#: 256 are held back for pure black and white (see `build_clip_palette`).
_CLIP_PALETTE_COLORS = 254

#: Bits per channel kept when collecting the colours a clip contains. Six bits
#: resolves 262144 buckets -- far finer than a 254-entry palette can express --
#: and caps the census at that many pixels however long the clip runs.
_GAMUT_BITS = 6


def _clip_gamut(frames: list) -> "PILImage.Image":
    """Collect every colour the clip contains, once each, as a compact image.

    The palette is chosen for colour *coverage*, so what the quantiser needs is
    the set of colours present, not how many pixels each covers. Spatially
    downsampling the frames to save time -- the obvious way to build a cheap
    palette source -- destroys exactly the information this change exists to
    keep: an interpolating resize blends a one-pixel mark into its background
    before the quantiser ever sees it, and a nearest-neighbour resize drops it
    whenever it falls between samples. Counting distinct colours instead is
    independent of how large a feature is on screen, and a presence bitmap makes
    it a single linear pass with no sort.

    Args:
        frames: The clip's frames as RGB `PIL.Image.Image` objects. They need
            not share a size.

    Returns:
        PIL.Image.Image: An RGB image holding one real colour from each occupied
        bucket, exactly once.

    Raises:
        ValueError: If `frames` is empty.
    """
    if not frames:
        raise ValueError("a clip palette needs at least one frame, got none.")

    shift = 8 - _GAMUT_BITS
    seen = np.zeros(1 << (3 * _GAMUT_BITS), dtype=bool)
    # Each bucket keeps a colour that genuinely occurs in the clip rather than
    # its own centre, so an exact colour -- pure red, a brand hue -- reaches the
    # quantiser unshifted instead of arriving a bucket-width off.
    representative = np.zeros(1 << (3 * _GAMUT_BITS), dtype=np.uint32)
    for frame in frames:
        pixels = np.asarray(frame, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
        exact = (pixels[:, 0] << 16) | (pixels[:, 1] << 8) | pixels[:, 2]
        bucket = (
            ((pixels[:, 0] >> shift) << (2 * _GAMUT_BITS))
            | ((pixels[:, 1] >> shift) << _GAMUT_BITS)
            | (pixels[:, 2] >> shift)
        )
        seen[bucket] = True
        representative[bucket] = exact

    keys = representative[np.flatnonzero(seen)]
    colours = np.stack(
        [(keys >> 16) & 0xFF, (keys >> 8) & 0xFF, keys & 0xFF], axis=-1
    ).astype(np.uint8)

    side = int(np.ceil(np.sqrt(len(colours))))
    canvas = np.empty((side * side, 3), dtype=np.uint8)
    canvas[: len(colours)] = colours
    canvas[len(colours) :] = colours[-1]  # pad with a colour already present
    return PILImage.fromarray(canvas.reshape(side, side, 3), "RGB")


def build_clip_palette(frames: list, colors: int = _CLIP_PALETTE_COLORS):
    """Build one colour palette shared by every frame of a clip.

    Quantising each frame independently makes constant regions shimmer and lets
    the same colour drift between frames, so one table is derived from the whole
    clip and every frame is mapped through it.

    The table is chosen for colour *coverage* (Pillow's `MAXCOVERAGE`) over the
    set of colours the clip contains, gathered by `_clip_gamut`. Median cut, the
    obvious alternative, splits by pixel population instead: on a clip with a
    large textured area the background claims nearly every slot and small
    saturated marks -- overlay glyphs, thin paths, labels -- collapse to the
    nearest muddy neighbour. Because coverage is computed over distinct colours
    rather than pixels, a mark survives no matter how few pixels it covers; a
    single-pixel mark is kept as faithfully as a large one.

    Args:
        frames: The clip's frames as RGB `PIL.Image.Image` objects.
        colors: How many palette entries to quantise to. The remaining entries
            up to 256 are reserved -- pure black and white are pinned so
            single-colour overlays stay crisp.

    Returns:
        PIL.Image.Image: A ``"P"``-mode image carrying the shared palette,
        ready to pass to `Image.quantize(palette=...)`.

    Raises:
        ValueError: If `frames` is empty, or `colors` is outside ``2-254`` --
            above 254 the reserved black and white would displace chosen
            entries.

    Examples:
        - Pure black and white are held back at the top of the table, so a
          single-colour overlay drawn on the clip stays crisp:
            ```python
            >>> from PIL import Image
            >>> from cleopatra.glyphs.base.animation import build_clip_palette
            >>> frames = [
            ...     Image.new("RGB", (12, 12), (200, 30, 30)),
            ...     Image.new("RGB", (12, 12), (30, 30, 200)),
            ... ]
            >>> entries = build_clip_palette(frames).getpalette()
            >>> entries[254 * 3 : 254 * 3 + 3]
            [0, 0, 0]
            >>> entries[255 * 3 : 255 * 3 + 3]
            [255, 255, 255]

            ```
        - A smaller budget moves the reserved pair up behind it, so asking for
          16 colours still leaves black and white reachable:
            ```python
            >>> from PIL import Image
            >>> from cleopatra.glyphs.base.animation import build_clip_palette
            >>> frames = [
            ...     Image.new("RGB", (12, 12), (200, 30, 30)),
            ...     Image.new("RGB", (12, 12), (30, 30, 200)),
            ... ]
            >>> entries = build_clip_palette(frames, colors=16).getpalette()
            >>> entries[16 * 3 : 16 * 3 + 3]
            [0, 0, 0]

            ```
        - The table spans the whole clip, so a colour introduced only in the
          last frame is still represented:
            ```python
            >>> from PIL import Image
            >>> from cleopatra.glyphs.base.animation import build_clip_palette
            >>> frames = [Image.new("RGB", (9, 9), (0, 0, 0))] * 4
            >>> frames.append(Image.new("RGB", (9, 9), (255, 0, 255)))
            >>> entries = build_clip_palette(frames).getpalette()
            >>> triples = [tuple(entries[i : i + 3]) for i in range(0, 254 * 3, 3)]
            >>> any(r > 200 and g < 40 and b > 200 for r, g, b in triples)
            True

            ```

    See Also:
        quantize_to_palette: Map the frames onto the palette this returns.
        gif_from_video: Derives a GIF through this same palette.
    """
    if not 2 <= colors <= _CLIP_PALETTE_COLORS:
        # Above 254 the reserved black/white pair would overwrite chosen entries,
        # and Pillow rejects 256 outright with a bare "invalid palette size".
        raise ValueError(f"colors must be in 2-{_CLIP_PALETTE_COLORS}, got {colors!r}.")

    census = _clip_gamut(frames)
    base = census.quantize(colors=colors, method=PILImage.Quantize.MAXCOVERAGE)
    entries = (list(base.getpalette() or []) + [0] * 768)[:768]
    entries[colors * 3 : colors * 3 + 6] = [0, 0, 0, 255, 255, 255]
    palette = PILImage.new("P", (1, 1))
    palette.putpalette(entries)
    return palette


def quantize_to_palette(frames: list, palette) -> list:
    """Map every frame onto a shared palette, dithering the residual error.

    Args:
        frames: The clip's frames as RGB `PIL.Image.Image` objects.
        palette: A ``"P"``-mode image carrying the palette, from
            `build_clip_palette`.

    Returns:
        list: The frames as ``"P"``-mode images sharing `palette`.

    Examples:
        - Every frame comes back palette-mode, carrying the same table -- which
          is what keeps a constant region byte-stable from frame to frame:
            ```python
            >>> from PIL import Image
            >>> from cleopatra.glyphs.base.animation import (
            ...     build_clip_palette,
            ...     quantize_to_palette,
            ... )
            >>> frames = [
            ...     Image.new("RGB", (8, 8), (255, 0, 0)),
            ...     Image.new("RGB", (8, 8), (0, 0, 255)),
            ... ]
            >>> quantised = quantize_to_palette(frames, build_clip_palette(frames))
            >>> len(quantised)
            2
            >>> quantised[0].mode
            'P'
            >>> quantised[0].getpalette() == quantised[1].getpalette()
            True

            ```
        - A colour the palette holds exactly survives the round trip unchanged:
            ```python
            >>> from PIL import Image
            >>> from cleopatra.glyphs.base.animation import (
            ...     build_clip_palette,
            ...     quantize_to_palette,
            ... )
            >>> frames = [Image.new("RGB", (8, 8), (255, 0, 0))]
            >>> quantised = quantize_to_palette(frames, build_clip_palette(frames))
            >>> quantised[0].convert("RGB").getpixel((0, 0))
            (255, 0, 0)

            ```

    See Also:
        build_clip_palette: Builds the shared palette these frames map onto.
    """
    return [
        frame.quantize(palette=palette, dither=PILImage.Dither.FLOYDSTEINBERG)
        for frame in frames
    ]


def _validate_pillow_options(fps: float, loop: int) -> None:
    """Check the options Pillow cannot fail cleanly on itself.

    Args:
        fps: Playback rate; becomes a per-frame duration in milliseconds.
        loop: Loop count; `0` loops forever.

    Raises:
        ValueError: If `fps` is not positive (Pillow would surface a bare
            `ZeroDivisionError` from the duration conversion) or if `loop` is
            negative (it is written as an unsigned short, so Pillow would raise
            a bare `struct.error`).
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}.")
    if loop < 0:
        raise ValueError(f"loop must be zero or positive, got {loop!r}.")


def _write_pillow_animation(
    frames: list, path, fps: float, loop: int, optimize: bool
) -> None:
    """Write frames out as an animated GIF or WebP via Pillow.

    Args:
        frames: The clip's frames, as a sequence or a lazy iterable. For GIF
            these are ``"P"``-mode images sharing one palette; for WebP they are
            passed through as grabbed.
        path: Where to write the animation.
        fps: Playback rate, converted to a per-frame duration in
            milliseconds. GIF's delay field is in hundredths of a second, so
            rates above 100 fps are held at that format's 10 ms floor rather
            than rounding to a zero delay viewers discard; WebP keeps
            millisecond timing and only floors at 1 ms.
        loop: How many times to loop; `0` loops forever.
        optimize: Run Pillow's optimisation pass (a no-op for WebP).

    Raises:
        ValueError: If `fps` is not positive or `loop` is negative.
    """
    _validate_pillow_options(fps, loop)
    # GIF stores its frame delay in hundredths of a second, so anything under
    # 10 ms rounds to zero on disk and viewers fall back to their own default --
    # far slower than asked for. WebP keeps millisecond timing, so it only needs
    # to stay above zero.
    floor = 10 if str(path).lower().endswith(".gif") else 1
    duration = max(floor, int(1000 / fps))
    stream = iter(frames)
    first = next(stream)
    # Pillow consumes `append_images` lazily, so handing it the rest of the
    # iterator keeps a streamed clip from being materialised just to be written.
    first.save(
        path,
        save_all=True,
        append_images=stream,
        duration=duration,
        loop=loop,
        optimize=optimize,
    )


def _ensure_ffmpeg_available() -> None:
    """Make sure matplotlib can find an ffmpeg binary to shell out to.

    matplotlib's `FFMpegWriter` runs the ffmpeg *binary* named by
    `matplotlib.rcParams["animation.ffmpeg_path"]` (default `"ffmpeg"`,
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
    if configured not in ("ffmpeg", "ffmpeg.exe"):
        warnings.warn(
            f"Configured ffmpeg binary {configured!r} was not found; falling "
            f"back to the imageio-ffmpeg bundled binary at {bundled!r}.",
            RuntimeWarning,
            stacklevel=3,
        )
    mpl.rcParams["animation.ffmpeg_path"] = bundled


class _OptimizedPillowWriter(PillowWriter):
    """`PillowWriter` that writes optimised, loop-configurable GIF/WebP output.

    This is the writer for both Pillow-backed formats (`gif` and `webp`).
    matplotlib's stock `PillowWriter` hardcodes `loop=0` and never passes
    `optimize` to `PIL.Image.save`, so GIFs come out unoptimised — needlessly
    large for photographic/satellite frames. This subclass forwards `optimize`
    and `loop` while reusing the parent's frame-grabbing logic.

    Args:
        optimize: Run Pillow's optimisation pass. Effective for GIF (palette
            compression); the WebP encoder ignores it, so it is a no-op there.
        loop: Number of times the animation loops; `0` means loop forever
            (Pillow's convention). Honoured by both GIF and WebP.
    """

    def __init__(self, *args, optimize: bool = True, loop: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._optimize = optimize
        self._loop = loop

    def finish(self):
        frames = self._frames  # type: ignore[attr-defined]
        if str(self.outfile).lower().endswith(".gif") and len(frames) > 1:
            rgb = [f.convert("RGB") for f in frames]
            frames = quantize_to_palette(rgb, build_clip_palette(rgb))
        _write_pillow_animation(
            frames, self.outfile, self.fps, self._loop, self._optimize
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
    """Assemble the ffmpeg `extra_args` list for a video export.

    Combines the mandatory even-dimension pad filter with an explicit pixel
    format and any caller-supplied CRF, preset, or raw ffmpeg flags. A caller
    `-vf` filter is merged into a single chain — ffmpeg honours only the last
    `-vf` — with the pad applied last so the frame ends up even whatever the
    caller's filters produce.

    Args:
        pix_fmt: Pixel format passed as `-pix_fmt` (e.g. `"yuv420p"`).
        crf: Constant Rate Factor; appended as `-crf` when not `None`.
        preset: libx264 speed/size preset; appended as `-preset` when set.
        extra_args: Extra ffmpeg flags. A `-vf` pair here is merged into the
            pad chain and a `-pix_fmt` pair overrides `pix_fmt` (rather than
            duplicating the flag); everything else is passed through unchanged.

    Returns:
        The assembled argument list, always starting with the merged `-vf`
        chain followed by a single `-pix_fmt`.

    Raises:
        ValueError: If `extra_args` ends with a valueless `-vf` or
            `-pix_fmt` flag.

    Examples:
        - Defaults produce just the pad filter and pixel format:
            ```python
            >>> from cleopatra.glyphs.base.animation import _build_ffmpeg_extra_args
            >>> _build_ffmpeg_extra_args("yuv420p", None, None, None)
            ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p']

            ```
        - A CRF and preset are appended after the pixel format:
            ```python
            >>> from cleopatra.glyphs.base.animation import _build_ffmpeg_extra_args
            >>> _build_ffmpeg_extra_args("yuv420p", 26, "slow", None)[4:]
            ['-crf', '26', '-preset', 'slow']

            ```
        - A caller `-vf` is merged into one chain with the pad applied last:
            ```python
            >>> from cleopatra.glyphs.base.animation import _build_ffmpeg_extra_args
            >>> _build_ffmpeg_extra_args("yuv420p", None, None, ["-vf", "scale=320:-1"])[:2]
            ['-vf', 'scale=320:-1,pad=ceil(iw/2)*2:ceil(ih/2)*2']

            ```
    """
    user_args = list(extra_args) if extra_args else []
    vf_filters: list[str] = []
    passthrough: list[str] = []
    caller_pix_fmt: str | None = None
    i = 0
    while i < len(user_args):
        arg = user_args[i]
        if arg in ("-vf", "-pix_fmt"):
            if i + 1 >= len(user_args):
                raise ValueError(
                    f"Malformed extra_args: {arg!r} must be followed by a value."
                )
            if arg == "-vf":
                vf_filters.append(user_args[i + 1])
            else:
                caller_pix_fmt = user_args[i + 1]
            i += 2
        else:
            passthrough.append(arg)
            i += 1
    vf_filters.append(_EVEN_PAD_FILTER)

    chosen_pix_fmt = caller_pix_fmt if caller_pix_fmt is not None else pix_fmt
    built = ["-vf", ",".join(vf_filters), "-pix_fmt", chosen_pix_fmt]
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

    Note: when no system FFmpeg is found, the first video export sets
    matplotlib's global `rcParams["animation.ffmpeg_path"]` to the bundled
    binary — a process-wide side effect that then applies to any later
    matplotlib animation in the same process.

    For the FFmpeg formats the frame is automatically padded up to an even
    width/height (libx264 rejects odd dimensions) and encoded with
    `pix_fmt=yuv420p` for universal playback. By default no fixed bitrate is
    requested (unlike older versions, which forced 1800 kbit/s), so libx264
    uses its constant-quality default of roughly CRF 23 — pass `crf` or
    `bitrate` to trade size against quality. GIF output is written with
    Pillow's `optimize` pass enabled; both GIF and WebP loop forever by
    default.

    Args:
        anim: The animation to save.
        path: Output file path, as a `str` or `os.PathLike` (e.g. a
            `pathlib.Path`). Extension determines format.
            Supported: gif, mov, avi, mp4, webp.
        fps: Frames per second. Default is 2.
        crf: Constant Rate Factor for the ffmpeg formats (lower is higher
            quality/larger; ~18-28 is typical). Assumes an x264/x265-family
            `codec`. Mutually exclusive with `bitrate`. Ignored for
            GIF/WebP. `None` uses the encoder default.
        bitrate: Target bitrate in kbit/s for the ffmpeg formats. Mutually
            exclusive with `crf`. Ignored for GIF/WebP. `None` lets the
            encoder choose.
        codec: ffmpeg codec (e.g. `"libx264"`). `None` uses matplotlib's
            default. Ignored for GIF/WebP.
        preset: libx264/libx265 speed/size preset (e.g. `"slow"`); ignored by
            codecs that don't accept it. Ignored for GIF/WebP.
        pix_fmt: Pixel format for the ffmpeg formats. Defaults to
            `"yuv420p"` for universal playback. Ignored for GIF/WebP.
        dpi: Resolution in dots per inch. `None` uses the figure's dpi.
        optimize: GIF only — run Pillow's palette optimisation pass (a no-op
            for WebP, whose encoder ignores it). Default `True`.
        loop: GIF/WebP only — number of times to loop; `0` loops forever.
        extra_args: Extra ffmpeg flags. A `-vf` filter here is merged with
            the automatic even-dimension pad and a `-pix_fmt` overrides
            `pix_fmt`. Note these flags bypass the `crf`/`bitrate`
            exclusivity check, so don't smuggle a conflicting `-b:v`/`-crf`
            through here. Ignored for GIF/WebP.

    Returns:
        The output path as a `str` (the `os.fspath` of `path`),
        convenient for chaining. Note a `pathlib.Path` argument comes
        back as its string form, not the original object.

    Raises:
        ValueError: If the file format is not supported, if both `crf`
            and `bitrate` are given (competing rate-control modes), or -- for
            the Pillow formats -- if `fps` is not positive or `loop` is
            negative.
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
            >>> from cleopatra.glyphs.base.animation import save_animation
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
        - The extension is matched case-insensitively, so `.GIF` also works:
            ```python
            >>> import os, shutil, tempfile, matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.glyphs.base.animation import save_animation
            >>> tmp = tempfile.mkdtemp()
            >>> fig, ax = plt.subplots()
            >>> (line,) = ax.plot([0, 1], [0, 0])
            >>> anim = FuncAnimation(fig, lambda i: (line,), frames=2)
            >>> save_animation(anim, os.path.join(tmp, "WAVE.GIF")).endswith("WAVE.GIF")
            True
            >>> plt.close(fig)
            >>> shutil.rmtree(tmp)

            ```
        - An unsupported extension raises `ValueError` before writing (here
          the animation is rendered once first, so nothing is left dangling):
            ```python
            >>> import os, shutil, tempfile, matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.glyphs.base.animation import save_animation
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
        embed_gif: Wrap an animation as an `IPython.display.Image`.
    """
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

    if video_format in _PILLOW_FORMATS:
        _validate_pillow_options(fps, loop)

    if crf is not None and bitrate is not None:
        raise ValueError(
            "Pass either crf or bitrate, not both: they are competing "
            "rate-control modes for the encoder."
        )

    save_kwargs: dict[str, Any] = {} if dpi is None else {"dpi": dpi}

    if video_format in _PILLOW_FORMATS:
        anim.save(
            path,
            writer=_OptimizedPillowWriter(fps=fps, optimize=optimize, loop=loop),
            **save_kwargs,
        )
    else:
        _ensure_ffmpeg_available()
        writer_kwargs: dict[str, Any] = {
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
                "FFmpeg could not be run. imageio-ffmpeg's bundled binary "
                "normally makes this work out of the box; if you pinned a custom "
                "ffmpeg via matplotlib's animation.ffmpeg_path, make sure it still "
                "exists, or install a system ffmpeg from https://ffmpeg.org/."
            ) from e
    return path


def to_bytes(anim: FuncAnimation, fmt: str = "gif", fps: int = 2, **kwargs) -> bytes:
    """Render a `FuncAnimation` to in-memory bytes in any supported format.

    Renders to a temporary file (the writers need a real path) and reads it
    back, leaving nothing on disk. Handy for embedding in a notebook or
    serving over HTTP.

    Args:
        anim: The animation to render.
        fmt: Output format — any member of `SUPPORTED_VIDEO_FORMAT` (e.g.
            `"gif"`, `"mp4"`, `"webp"`). A leading dot is tolerated.
        fps: Frames per second. Default is 2.
        **kwargs: Extra keyword arguments forwarded to `save_animation`
            (e.g. `crf`, `codec`, `loop`).

    Returns:
        The encoded bytes of the animation in the requested format.

    Raises:
        ValueError: If `fmt` is not a supported format.

    Examples:
        - Render to GIF bytes and inspect the payload:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.glyphs.base.animation import to_bytes
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
            >>> from cleopatra.glyphs.base.animation import to_bytes
            >>> fig, ax = plt.subplots()
            >>> (line,) = ax.plot([0, 1], [0, 0])
            >>> anim = FuncAnimation(fig, lambda i: (line,), frames=2)
            >>> data = to_bytes(anim, fmt="webp")
            >>> data[:4] == b"RIFF" and data[8:12] == b"WEBP"
            True
            >>> plt.close(fig)

            ```
        - An unsupported format raises `ValueError`:
            ```python
            >>> from unittest.mock import MagicMock
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.glyphs.base.animation import to_bytes
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
    a file on disk. Thin wrapper around `to_bytes` with `fmt="gif"`.

    Args:
        anim: The animation to render.
        fps: Frames per second. Default is 2.
        **kwargs: Extra keyword arguments forwarded to `save_animation`
            (e.g. `optimize`, `loop`).

    Returns:
        The GIF-encoded bytes of the animation.

    Examples:
        - Render an animation to GIF bytes and inspect the payload:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.glyphs.base.animation import to_gif
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
        - A higher `fps` still yields self-contained bytes you can serve over
          HTTP or write yourself, without leaving a temp file behind:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.glyphs.base.animation import to_gif
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
        embed_gif: Wrap these bytes as an `IPython.display.Image`.
    """
    return to_bytes(anim, fmt="gif", fps=fps, **kwargs)


def to_mp4(anim: FuncAnimation, fps: int = 2, **kwargs) -> bytes:
    """Render a `FuncAnimation` to in-memory MP4 (H.264) bytes.

    Handy for embedding a compact, universally-playable clip or serving it
    over HTTP without leaving a file on disk. Thin wrapper around `to_bytes`
    with `fmt="mp4"`; the frame is auto-padded to even dimensions and
    encoded `yuv420p` like every other MP4 export.

    Args:
        anim: The animation to render.
        fps: Frames per second. Default is 2.
        **kwargs: Extra keyword arguments forwarded to `save_animation`
            (e.g. `crf`, `bitrate`, `codec`, `preset`).

    Returns:
        The MP4-encoded bytes of the animation.

    Raises:
        FileNotFoundError: If neither a system FFmpeg nor imageio-ffmpeg's
            bundled binary can be found.

    Examples:
        - Render to MP4 bytes and confirm the ISO base-media `ftyp` box:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.glyphs.base.animation import to_mp4
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
            >>> from cleopatra.glyphs.base.animation import to_mp4
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


def _is_chroma_subsampled(pix_fmt: str | None) -> bool:
    """Whether a pixel format throws away colour resolution.

    Args:
        pix_fmt: The source's pixel format as ffmpeg reports it, e.g.
            ``"yuv420p"``. `None` when the decoder did not report one.

    Returns:
        bool: `True` for a YUV format that is not 4:4:4 (or the NV planar
        formats, which are 4:2:0), `False` otherwise.
    """
    if not pix_fmt:
        return False
    fmt = pix_fmt.lower()
    if fmt.startswith("nv"):  # nv12 / nv21 are 4:2:0
        return True
    return fmt.startswith(("yuv", "yuvj")) and "444" not in fmt


def _read_video_frames(src: str, **kwargs):
    """Open a video for raw-frame reading via the bundled FFmpeg.

    Args:
        src: Path to the video.
        **kwargs: Passed through to `imageio_ffmpeg.read_frames` (e.g.
            `output_params`).

    Returns:
        Generator: Yields the decoder's metadata dict first, then each frame as
        raw ``rgb24`` bytes. Close it when done.

    Raises:
        FileNotFoundError: If imageio-ffmpeg is not installed, so no FFmpeg
            binary is available to decode with.
    """
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as e:  # pragma: no cover - imageio-ffmpeg is a dep
        raise FileNotFoundError(
            "Deriving a GIF from a video needs FFmpeg. Install imageio-ffmpeg "
            "(ships a bundled binary) or an ffmpeg on PATH."
        ) from e
    return imageio_ffmpeg.read_frames(src, **kwargs)


def _video_metadata(src: str) -> dict:
    """Read a video's header without decoding any of it.

    Args:
        src: Path to the video.

    Returns:
        dict: The decoder's metadata, including ``size`` and ``pix_fmt``.
    """
    reader = _read_video_frames(src)
    try:
        return next(reader)
    finally:
        reader.close()


def _iter_video_frames(src: str, fps: float, width: int | None):
    """Yield a video's frames as RGB images, closing the decoder afterwards.

    Frames are produced lazily so a clip never has to be held in memory all at
    once, and the decoder is closed on the way out however the caller stops --
    exhausted, `break`, or an exception -- rather than leaving an orphaned
    ffmpeg child behind.

    Args:
        src: Path to the video.
        fps: Rate to sample at; FFmpeg drops or duplicates frames to hit it.
        width: Width to scale to, preserving aspect, or `None` to keep the
            source size.

    Yields:
        PIL.Image.Image: Each sampled frame, as RGB.
    """
    reader = _read_video_frames(src, output_params=["-vf", f"fps={fps}"])
    try:
        frame_w, frame_h = next(reader)["size"]
        target = None
        if width is not None and width != frame_w:
            target = (width, max(1, round(frame_h * width / frame_w)))
        for buffer in reader:
            frame = PILImage.frombytes("RGB", (frame_w, frame_h), bytes(buffer))
            yield frame if target is None else frame.resize(target, PILImage.LANCZOS)
    finally:
        reader.close()


def gif_from_video(
    src: str | os.PathLike,
    path: str | os.PathLike,
    *,
    fps: float = 12,
    width: int | None = None,
    max_colors: int = _CLIP_PALETTE_COLORS,
    loop: int = 0,
    optimize: bool = True,
) -> str:
    """Derive a GIF from an existing video, without re-rendering the frames.

    Drawing is usually far more expensive than encoding -- hours, for a long
    scientific animation -- so a clip is best rendered once to a video and every
    other format derived from that file. `save_animation` needs a live
    `FuncAnimation` and would re-render; this reads the frames back off disk
    instead.

    The frames go through exactly the same clip-wide palette as
    `save_animation`'s GIF path (`build_clip_palette`), so a GIF derived from a
    video and one rendered straight from the animation quantise identically.

    The source is decoded twice and streamed both times -- once to learn the
    clip's colours, once to quantise and write -- so peak memory is flat in the
    clip's length rather than proportional to it, and a long master does not
    have to fit in RAM.

    Args:
        src: The source video. Any container the bundled FFmpeg can decode.
        path: Where to write the GIF.
        fps: Frames per second to sample the source at. Frames are dropped or
            duplicated by FFmpeg's `fps` filter as needed. Defaults to `12`.
        width: Scale the output to this width in pixels, preserving aspect.
            `None` (the default) keeps the source's own size.
        max_colors: Palette size, in ``2-254``. The rest of the 256 entries are
            reserved for pure black and white.
        loop: How many times the GIF loops; `0` loops forever.
        optimize: Run Pillow's optimisation pass.

    Returns:
        The output path as a `str`, convenient for chaining.

    Raises:
        FileNotFoundError: If `src` does not exist, or if no FFmpeg binary can
            be found.
        ValueError: If `max_colors` is outside ``2-254``, if `fps` is not
            positive, if `width` is not positive, if `loop` is negative, if
            `path` does not end in ``.gif``, or if `src` yields no frames.

    Warns:
        UserWarning: If `src` is chroma-subsampled (e.g. the ``yuv420p`` that
            `save_animation` writes by default). Colour resolution is already
            gone from such a file, which caps how well saturated detail can
            survive whatever the GIF palette then does -- render the
            intermediate with ``pix_fmt="yuv444p"`` and a low `crf` when the
            plan is to derive a GIF from it.

    Examples:
        - Render an animation once to MP4, then derive a GIF from that file:
            ```python
            >>> import os, shutil, tempfile, warnings, matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.glyphs.base.animation import gif_from_video, save_animation
            >>> tmp = tempfile.mkdtemp()
            >>> fig, ax = plt.subplots()
            >>> (line,) = ax.plot([0, 1], [0, 0])
            >>> anim = FuncAnimation(fig, lambda i: (line,), frames=4)
            >>> mp4 = save_animation(anim, os.path.join(tmp, "clip.mp4"), fps=4,
            ...                      pix_fmt="yuv444p")
            >>> gif = gif_from_video(mp4, os.path.join(tmp, "clip.gif"), fps=4)
            >>> open(gif, "rb").read()[:6] in (b"GIF87a", b"GIF89a")
            True
            >>> plt.close(fig)
            >>> shutil.rmtree(tmp)

            ```
        - `width` scales the output for a web copy, keeping the aspect ratio of
          the source and leaving the master untouched:
            ```python
            >>> import os, shutil, tempfile, matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from PIL import Image
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.glyphs.base.animation import gif_from_video, save_animation
            >>> tmp = tempfile.mkdtemp()
            >>> fig, ax = plt.subplots()
            >>> (line,) = ax.plot([0, 1], [0, 0])
            >>> anim = FuncAnimation(fig, lambda i: (line,), frames=4)
            >>> mp4 = save_animation(anim, os.path.join(tmp, "master.mp4"), fps=4,
            ...                      pix_fmt="yuv444p")
            >>> gif = gif_from_video(mp4, os.path.join(tmp, "web.gif"), fps=4, width=160)
            >>> with Image.open(gif) as web:
            ...     web.size
            (160, 120)
            >>> plt.close(fig)
            >>> shutil.rmtree(tmp)

            ```
        - A source that does not exist is reported up front, rather than
          failing later inside the decoder:
            ```python
            >>> from cleopatra.glyphs.base.animation import gif_from_video
            >>> gif_from_video("no-such-clip.mp4", "out.gif")
            Traceback (most recent call last):
                ...
            FileNotFoundError: The source video 'no-such-clip.mp4' does not exist.

            ```

    See Also:
        save_animation: Write a live `FuncAnimation` straight to a file.
        build_clip_palette: The shared palette both paths quantise through.
    """
    src = os.fspath(src)
    path = os.fspath(path)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"The source video {src!r} does not exist.")
    if not 2 <= max_colors <= _CLIP_PALETTE_COLORS:
        raise ValueError(
            f"max_colors must be in 2-{_CLIP_PALETTE_COLORS}, got {max_colors!r}."
        )
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}.")
    if width is not None and width <= 0:
        raise ValueError(f"width must be positive, got {width!r}.")
    # Pillow picks its encoder from the extension, so a stray one silently
    # writes a different format -- .png yields an APNG that no caller of a
    # function named gif_from_video is expecting.
    extension = os.path.splitext(path)[1].lstrip(".").lower()
    if extension != "gif":
        raise ValueError(
            f"gif_from_video writes GIFs, but {path!r} implies {extension or 'no'} "
            "format. Use a .gif extension."
        )
    _validate_pillow_options(fps, loop)

    meta = _video_metadata(src)
    if _is_chroma_subsampled(meta.get("pix_fmt")):
        warnings.warn(
            f"{src!r} is {meta.get('pix_fmt')}, which stores colour at reduced "
            "resolution; saturated detail is already degraded before the GIF "
            "palette sees it. Render the intermediate with pix_fmt='yuv444p' "
            "and a low crf when a GIF will be derived from it.",
            UserWarning,
            stacklevel=2,
        )

    # Two streamed passes rather than one buffered one: the palette needs to see
    # the whole clip before any frame can be quantised, but neither pass needs
    # to keep a frame once it has been read, so peak memory stays flat in the
    # clip's length. Decoding twice is far cheaper than holding it all in RAM --
    # a 10-minute 1080p master would otherwise need tens of gigabytes.
    survey = _iter_video_frames(src, fps, width)
    first = next(survey, None)
    if first is None:
        raise ValueError(f"The source video {src!r} yielded no frames.")
    palette = build_clip_palette(itertools.chain([first], survey), colors=max_colors)

    quantised = (
        frame.quantize(palette=palette, dither=PILImage.Dither.FLOYDSTEINBERG)
        for frame in _iter_video_frames(src, fps, width)
    )
    _write_pillow_animation(quantised, path, fps, loop, optimize)
    return path


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
            >>> from cleopatra.glyphs.base.animation import embed_gif
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
          a custom `fps` controls playback speed:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.animation import FuncAnimation
            >>> from cleopatra.glyphs.base.animation import embed_gif
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
        if e.name and e.name.split(".")[0] != "IPython":
            raise
        raise ModuleNotFoundError(
            "embed_gif requires IPython for inline display. Install it with "
            "`pip install ipython` (already present in any Jupyter/IPython "
            "environment). For raw GIF bytes without IPython, use to_gif()."
        ) from e

    return Image(data=to_gif(anim, fps=fps), format="gif")
