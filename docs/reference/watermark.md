# Watermark — Stamp a Logo or Brand Text on a Figure

The `cleopatra.styling.watermark` module puts a mark on a finished matplotlib `Figure` with a
single call, so anything you publish or share can carry one without re-rolling the same glue in
every notebook. Two entry points, designed to be used together:

- **`stamp_mark`** — a logo *image*, anchored in a corner.
- **`stamp_watermark`** — diagonal brand *text* across the middle, with an optional credit line
  along the bottom.

Both size by a fraction of the figure and position by a margin from its edge, so a mark keeps its
proportions across the several dpis a figure is exported at.

## `stamp_mark` — a logo image

`stamp_mark(fig, path, *, frac=0.11, corner="lower right", margin=0.025, shadow=True,
blur=0.065)`. Two things make it more than a one-liner over `imshow`:

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
top. Single-image, corner-anchored marks only — tiled / repeated marks and any licensing /
provenance semantics are out of scope.

`stamp_mark` accepts the mark either as a **file path** (any format Pillow can open, read as
RGBA) or as an in-memory `(H, W, 3)` / `(H, W, 4)` NumPy array (`uint8` `0-255` or float
`0-1`; RGB gains an opaque alpha). It returns the frameless inset `Axes` it drew on, so you
can adjust it further.

## `stamp_watermark` — brand text

`stamp_watermark(fig, text, *, frac=0.55, angle=30.0, alpha=0.65, color="white", credit=None,
credit_frac=0.28, credit_alpha=1.0, margin=0.014)` stamps translucent brand text across the middle
of the figure, and optionally a credit line along the bottom.

- **The fraction means the same thing for any text.** Scaling a point size off the figure width —
  the obvious shortcut — renders a short word small and a long one straight off the canvas,
  because how much of a frame a string covers depends on how many characters it has. `frac` is
  measured on what is actually rendered, so a two-letter brand and a twenty-character one both
  land at the fraction you asked for.
- **The credit line is placed by `margin`**, a fraction of the figure height above the bottom
  edge, and sized by `credit_frac` — the same shapes `stamp_mark` uses, rather than a hardcoded
  offset and point size.
- **Only the credit line is outlined.** That asymmetry is deliberate: an outline on the large
  diagonal text makes it read as a solid caption rather than a watermark, while the credit is
  small enough that it needs the stroke to stay legible against whatever the frame contains.

It returns `(brand_text, credit_text)` — the second is `None` when no `credit` was given.

```python
from cleopatra.styling.watermark import stamp_mark, stamp_watermark

stamp_mark(fig, LOGO, frac=0.18, corner="lower left")
stamp_watermark(fig, "earthlens", credit="github.com/serapeum-org/earthlens")
```

!!! note "Call both last"
    Like `stamp_mark`, the text size is baked from the figure's current size, so stamp **after**
    any `tight_layout()` and after the final `set_size_inches`. The proportion holds across dpi.
    It does not survive a later `set_size_inches`, and here the text differs from the mark: a mark
    lives on an inset axes in figure-fraction coordinates and keeps its share of a figure resized
    proportionally afterwards, whereas text is measured in points and keeps its *absolute* size,
    halving its share when the figure doubles.

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
