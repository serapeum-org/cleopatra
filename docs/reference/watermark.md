# Watermark — Stamp a Logo / Brand-Mark on a Figure

The `cleopatra.styling.watermark` module places a logo or watermark image onto a finished
matplotlib `Figure` with a single call, so anything you publish or share can carry a mark
without re-rolling the same inset-axes glue in every notebook.

The one entry point is `stamp_mark(fig, path, *, frac=0.11, corner="lower right", margin=0.025,
shadow=True)`. Two things make it more than a one-liner over `imshow`:

- **Fraction-of-figure sizing.** The mark is drawn on a frameless inset axes in
  *figure-fraction* coordinates, so it stays the same proportion (and corner offset) no
  matter what dpi the figure is later saved at — the MP4 master, the smaller web copy, and
  the GIF all get a mark of the same relative size. `frac` sets the width relative to the
  figure width; the height is derived from the image and figure aspect ratios, so the image
  is never stretched. (This is the dpi-independent counterpart of `Figure.figimage`, which is
  pixel-based.)
- **Optional drop shadow.** With `shadow=True` (the default) a gaussian-blurred, black,
  slightly-offset copy of the mark's alpha is drawn beneath it so the mark separates from a
  busy or dark canvas. The blur uses Pillow (already a cleopatra dependency), so no new
  dependency — and no SciPy — is pulled in.

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
to the figure edges as a fraction of the figure.

::: cleopatra.styling.watermark.stamp_mark
