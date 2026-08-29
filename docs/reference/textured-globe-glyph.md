# TexturedGlobeGlyph Class

The `textured_globe_glyph` module provides the `TexturedGlobeGlyph` class — cleopatra's one
deliberate **3-D** glyph. It wraps an equirectangular (lon/lat) RGB(A) texture onto a tilted
sphere drawn on a matplotlib `Axes3D`, and can spin the globe frame-by-frame for animation.

It takes the same equirectangular layout as
[`cleopatra.basemap.reference.relief()`](reference-data.md) — an `(H, W, 3)` (or `(H, W, 4)`)
array with row 0 at the north pole (+90°) and column 0 at −180° — so you can drape a relief
raster (or any world texture) straight onto a globe. Like `HistogramGlyph`, it is a **standalone
class**, not a `Glyph` subclass, because the base class's 2-D figure/colorbar pipeline does not
apply to a sphere.

!!! note "Resolution is the cost driver"
    matplotlib's 3-D surface is drawn on the CPU as one polygon per mesh face, so render time
    grows with `n_lon × n_lat`. The default `180 × 90` (~16k faces) renders a recognisable globe
    in ~1.5 s; `360 × 180` is ~7.7 s and `720 × 360` ~27 s. Raise the resolution for a sharper
    still, lower it for a smooth animation.

## Class Documentation

::: cleopatra.glyphs.globe.textured_globe_glyph.TexturedGlobeGlyph
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Examples

### A still globe from a relief texture

```python
import matplotlib.pyplot as plt
from cleopatra.basemap.reference import relief
from cleopatra.glyphs.globe.textured_globe_glyph import TexturedGlobeGlyph

texture = relief("low")            # (360, 720, 3) equirectangular RGB, north-up
globe = TexturedGlobeGlyph(texture, tilt_deg=23.44, brightness=1.1)
fig, ax = globe.draw(spin=60.0, elev=20, background="black")
plt.show()
```

### A synthetic texture (no download)

```python
import numpy as np
import matplotlib.pyplot as plt
from cleopatra.glyphs.globe.textured_globe_glyph import TexturedGlobeGlyph

texture = np.zeros((180, 360, 3), dtype=np.uint8)
texture[:90] = (40, 90, 180)       # northern hemisphere blue
texture[90:] = (180, 120, 40)      # southern hemisphere ochre

fig, ax = TexturedGlobeGlyph(texture).draw(spin=45.0)
plt.show()
```

### A day/night terminator (directional lighting)

Pass a `sun` unit vector (in world space: `+z` is north/up, `+x` faces the viewer at `spin=0`) to
light the sphere from a direction. `ambient` is a floor so the night side stays legible. Lighting
is applied per frame from the already-rotated vertices — no texture re-sampling — so a fixed `sun`
with a spinning globe sweeps the terminator across the surface. `sun=None` (the default) renders
evenly.

```python
from cleopatra.basemap.reference import relief
from cleopatra.glyphs.globe.textured_globe_glyph import TexturedGlobeGlyph

globe = TexturedGlobeGlyph(relief("low"), sun=(0.0, 1.0, 0.3), ambient=0.13)
fig, ax = globe.draw(spin=40.0, background="black")   # side-lit: one half in daylight, the other in night
```

### A spinning animation

The texture is sampled once; each frame only rotates the pre-computed mesh, so use a modest
resolution for smooth playback. Add `sun=...` for a lit globe whose terminator moves as it turns.

```python
from cleopatra.basemap.reference import relief
from cleopatra.glyphs.globe.textured_globe_glyph import TexturedGlobeGlyph

globe = TexturedGlobeGlyph(relief("low"), n_lon=180, n_lat=90)
anim = globe.animate(n_frames=60, revolutions=1.0, interval=50, sun=(1.0, 0.0, 0.0))
# save with cleopatra.glyphs.base.animation.save_animation (or to_gif/to_mp4),
# or matplotlib's own writers:
# from cleopatra.glyphs.base.animation import save_animation
# save_animation(anim, "globe.gif")
```

### Aligning your own geometry with the globe (the tilt transform)

The glyph tilts the sphere about the world `x` axis by `tilt_deg`, then spins it about the polar axis.
To place your own scene geometry — a marker on the surface, a ring in the equatorial plane, an orbit
plane — so it sits consistently with the rendered globe, push it through the **same** transform with
`transform(points, spin=...)` (or grab the `(3, 3)` matrix with `rotation_matrix(spin)`). The body
frame is the unit sphere: `+z` at the north pole, so a surface point at `(lon, lat)` is
`[cos(lat)·cos(lon), cos(lat)·sin(lon), sin(lat)]` and the equatorial plane is `z = 0`.

```python
import numpy as np
from cleopatra.glyphs.globe.textured_globe_glyph import TexturedGlobeGlyph

globe = TexturedGlobeGlyph(relief("low"), tilt_deg=23.44)
fig, ax = globe.draw(spin=40.0)

# a geostationary ring in the equatorial plane, tilted+spun to match the globe
theta = np.linspace(0, 2 * np.pi, 200)
ring = np.column_stack([1.3 * np.cos(theta), 1.3 * np.sin(theta), np.zeros_like(theta)])
ring = globe.transform(ring, spin=40.0)
ax.plot(ring[:, 0], ring[:, 1], ring[:, 2])
```
