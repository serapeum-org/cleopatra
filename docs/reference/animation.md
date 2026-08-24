# Animation Module — Save / Embed Helpers for Any `FuncAnimation`

The `cleopatra.glyphs.base.animation` module exposes cleopatra's animation **save / inline-embed**
machinery as **glyph-independent** helpers. They operate on *any*
`matplotlib.animation.FuncAnimation` — a sine wave, stock prices, or a map — not only on a
`Glyph`'s internal `self.anim`:

- `save_animation(anim, path, fps=2, ...)` writes an animation to a file, choosing the
  writer from the extension: `gif` and `webp` via Pillow, `mov`/`avi`/`mp4` via FFmpeg. The
  extension is matched **case-insensitively**. Quality controls are keyword-only:
  `crf` / `bitrate` (mutually exclusive), `codec`, `preset`, `pix_fmt`, and `dpi` for the
  FFmpeg formats; `optimize` and `loop` for the Pillow (GIF/WebP) formats; plus `extra_args`
  passed straight through to the writer.
- `to_bytes(anim, fmt="gif", fps=2, ...)` renders to in-memory **bytes** in any supported
  format (temp file cleaned up afterwards); `to_gif(...)` and `to_mp4(...)` are thin
  wrappers for the two common formats.
- `embed_gif(anim, fps=2)` returns an `IPython.display.Image` for inline notebook display.
  IPython is imported **lazily** (and is bundled with Jupyter, so any notebook already has
  it); if it is absent, `embed_gif` raises a clear `ModuleNotFoundError` with a
  `pip install ipython` hint — or use `to_gif` for raw bytes with no IPython dependency.
- `gif_from_video(src, path, fps=12, width=None, max_colors=254, ...)` derives a GIF from a
  video **already on disk**, without re-rendering. Drawing is usually far more expensive than
  encoding, so a long clip is best rendered once to MP4 and every other format derived from
  that file.

## The GIF palette

Both GIF paths — `save_animation` and `gif_from_video` — quantise through one palette shared
by every frame, built by `build_clip_palette` from the colours the whole clip contains and
applied by `quantize_to_palette`. Both are public, so a downstream package writing its own
frames can reuse the same table rather than re-deriving one. Per-frame
palettes would make constant regions shimmer and let a colour drift between frames; two of
the 256 entries are pinned to pure black and white so single-colour overlays stay crisp.

The palette is chosen for colour **coverage**, not pixel population, over the set of colours the
clip contains. The distinction matters on exactly the clips this package produces: with a
population-weighted split (median cut) a large textured background claims nearly every palette
slot, and small saturated marks — overlay glyphs, thin paths, labels — collapse to the nearest
muddy neighbour. On a texture-heavy test clip those marks landed 100–180 away (in RGB distance)
from the colours they were drawn in; selecting for coverage reproduces them exactly.

Because coverage is computed over **distinct colours rather than pixels**, a mark survives no
matter how small it is: a one-pixel orbit path is kept as faithfully as a large glyph. Sampling
the frames spatially to build a cheaper palette source would undo that — an interpolating resize
blends a one-pixel mark into its background before the quantiser ever sees it.

The trade is a marginally coarser background: on the same clip, background RMSE moves from 7.5 to
8.3, because palette entries now go to colours the clip contains rather than to the colours it
contains *most of*. File size moves either way depending on the clip — it shrank on the
texture-heavy case and grew on a constant-background one.

`quantize_method` is the opt-out, on both `save_animation` and `gif_from_video`. It takes a key of
`QUANTIZE_METHODS` — `"coverage"` (the default), `"median"`, or `"octree"`. Reach for `"median"` on
a smooth photographic clip with no small marks at stake, where weighting by pixel population spends
the palette on the dominant colours and renders the background a little more finely:

```python
save_animation(anim, "clip.gif", fps=12, quantize_method="median")
```

!!! warning "Render the intermediate with `pix_fmt="yuv444p"` if a GIF will be derived from it"

    `save_animation` writes `yuv420p` by default — the right choice for playback compatibility,
    but it stores colour at half resolution in each direction. That loss happens *before* the
    GIF palette ever runs, and no quantiser can undo it: on the same test clip a `yuv420p`
    intermediate caps the derived GIF at ~50 RGB distance, against ~5 from a `yuv444p` one.
    `gif_from_video` emits a `UserWarning` when it is handed a subsampled source.

    ```python
    mp4 = save_animation(anim, "master.mp4", fps=12, crf=0, pix_fmt="yuv444p")
    gif_from_video(mp4, "web.gif", fps=12, width=720)
    ```

`SUPPORTED_VIDEO_FORMAT` is `["gif", "mov", "avi", "mp4", "webp"]`. `Glyph.save_animation`
delegates to `save_animation`, so the writer/format logic has a single source of truth.
Downstream packages that build their own `FuncAnimation` can reuse these helpers instead of
re-rolling temp-file + writer + `IPython.display` glue.

## Usage

```python
import matplotlib
matplotlib.use("Agg")  # any backend; Agg shown for headless rendering
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from cleopatra.glyphs.base.animation import embed_gif, save_animation, to_gif

# Build any FuncAnimation — no Glyph required.
fig, ax = plt.subplots()
(line,) = ax.plot([0, 1], [0, 0])


def update(i):
    line.set_ydata([0, i])
    return (line,)


anim = FuncAnimation(fig, update, frames=3, blit=True)

save_animation(anim, "wave.gif", fps=3)   # write to a file (gif/webp/mov/avi/mp4)
save_animation(anim, "wave.mp4", fps=3, crf=18)  # quality-controlled MP4
gif_bytes = to_gif(anim, fps=3)           # in-memory GIF bytes
embed_gif(anim, fps=3)                     # inline in a notebook cell
```

!!! note
    The output format is taken from the file extension and is matched case-insensitively
    (`out.GIF` works). `gif` / `webp` are written with Pillow (no FFmpeg needed). Video
    formats (`mov`/`avi`/`mp4`) use FFmpeg: cleopatra depends on `imageio-ffmpeg`, which
    **bundles a static FFmpeg binary**, so video export works out of the box — a system
    FFmpeg on the `PATH` is used in preference when present. A `FileNotFoundError` (pointing
    at <https://ffmpeg.org/>) is raised only if no FFmpeg — bundled or system — is available;
    an unsupported extension raises a `ValueError`. `embed_gif` imports IPython only when
    called; if IPython is absent it raises a `ModuleNotFoundError` with a `pip install
    ipython` hint (use `to_gif` to avoid IPython).

## Module Documentation

::: cleopatra.glyphs.base.animation
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
