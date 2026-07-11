# Animation Module — Save / Embed Helpers for Any `FuncAnimation`

The `cleopatra.animation` module exposes cleopatra's animation **save / inline-embed**
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

from cleopatra.animation import embed_gif, save_animation, to_gif

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

::: cleopatra.animation
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
