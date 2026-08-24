# Watermark — Stamp a Logo / Brand-Mark on a Figure

The `cleopatra.styling.watermark` module places a logo or watermark image onto a finished
matplotlib `Figure` with a single call, so anything you publish or share can carry a mark
without re-rolling the same inset-axes glue in every notebook.

The one entry point is `stamp_mark(fig, path, *, frac=0.11, corner="lower right", margin=0.025,
shadow=True, blur=0.065)`. Two things make it more than a one-liner over `imshow`:

- **Fraction-of-figure sizing.** The mark is drawn on a frameless inset axes in
  *figure-fraction* coordinates, so it stays the same proportion (and corner offset) no
  matter what dpi the figure is later saved at — the MP4 master, the smaller web copy, and
  the GIF all get a mark of the same relative size. `frac` sets the width relative to the
  figure width; the height is derived from the image and figure aspect ratios, so the image
  is never stretched. (This is the dpi-independent counterpart of `Figure.figimage`, which is
  pixel-based.)
- **Optional halo.** With `shadow=True` (the default) a gaussian-blurred black copy of the
  mark's alpha is composited *behind* it so the mark separates from a busy or dark canvas.
  The blur uses Pillow (already a cleopatra dependency), so no new dependency — and no
  SciPy — is pulled in.

  The halo is **centred, not offset**. A mark is composited over arbitrary imagery — night
  ocean, sunlit cloud, a bright limb — and a symmetric halo reads the same whichever way the
  background falls, where a down-right drop shadow implies a light direction nothing else in
  the frame has. `blur` is the halo's sigma as a fraction of the mark's own width.

It is a presentation helper, not a glyph: it takes whatever `Figure` you hand it and draws on
top. Single-image, corner-anchored marks only — text watermarks, tiled / repeated marks, and
any licensing / provenance semantics are out of scope.

`stamp_mark` accepts the mark either as a **file path** (any format Pillow can open, read as
RGBA) or as an in-memory `(H, W, 3)` / `(H, W, 4)` NumPy array (`uint8` `0-255` or float
`0-1`; RGB gains an opaque alpha). It returns the frameless inset `Axes` it drew on, so you
can adjust it further.

## Usage

```python
import matplotlib.pyplot as plt
import numpy as np
from cleopatra.styling.watermark import stamp_mark

fig = plt.figure(figsize=(12, 8))
fig.add_subplot(111).imshow(np.random.default_rng(0).random((60, 90)), cmap="magma")

# a file on disk...
stamp_mark(fig, "brand/logo.png", frac=0.12, corner="lower right")

# ...or an in-memory RGBA array, in a different corner, without the shadow
logo = np.zeros((80, 160, 4), dtype=np.uint8)
logo[..., :3] = 255
logo[..., 3] = 255
stamp_mark(fig, logo, frac=0.09, corner="upper left", shadow=False)

fig.savefig("figure.png", dpi=200)  # the mark keeps its proportion at any dpi
```

`corner` is one of `"lower right"` (default), `"lower left"`, `"upper right"`, or
`"upper left"`; anything else raises a `ValueError` naming the bad value. `margin` is the gap
between the **mark** and the figure edges as a fraction of the figure — either a scalar for
both axes or an `(x, y)` pair. The pair matters when a mark has to tuck hard into a corner on
one axis while keeping a gap on the other:

```python
stamp_mark(fig, "brand/logo.png", margin=(0.025, 0.0))  # flush with the bottom, inset from the right
```

!!! note "Call `stamp_mark` last, and save the whole figure"

    The mark is baked at stamp time from the figure's current size, so stamp **after** any
    `tight_layout()` / layout finalization and after the final `set_size_inches` (stamping first
    then calling `tight_layout()` emits a `UserWarning`). The fraction-of-figure sizing assumes the
    **whole** figure is saved: a plain `dpi=` save keeps the mark proportional, but
    `savefig(bbox_inches="tight")` crops surrounding whitespace and so changes the mark's relative
    margin and size.

!!! note "`frac` sizes the mark, not the halo canvas"

    The halo needs a transparent pad of three sigmas on each side to hold its own tail, which
    makes the composited canvas about 1.39x the mark's width at the default `blur`. `stamp_mark`
    grows the inset axes by exactly that factor, so the **visible mark** still measures `frac`.
    Sizing the padded canvas to `frac` instead would render the mark at roughly 72 % of the
    requested size — easy to miss, because the axes bounding box still looks correct. With
    `shadow=True` the returned axes' bbox therefore covers mark *and* halo, and is larger than
    `frac`; `margin` is still measured to the mark, so a halo beside a small margin is clipped at
    the figure edge (which is what you want when tucking a mark into a corner).

::: cleopatra.styling.watermark.stamp_mark
