# MeshGlyph Class

The `MeshGlyph` class provides visualization for UGRID-style unstructured mesh data
using matplotlib triangulation. It supports face-centered and node-centered plotting,
wireframe rendering, all 5 color scale types, and time-series animation.

## Class Documentation

::: cleopatra.glyphs.gridded.mesh_glyph.MeshGlyph
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Examples

### Basic Face-Centered Plot

```python
import numpy as np
import matplotlib.tri as mtri
from cleopatra.styling.colorbar import ColorBar
from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph

# Create a triangular mesh from random points
rng = np.random.default_rng(42)
node_x = rng.uniform(0, 10, 50)
node_y = rng.uniform(0, 8, 50)
tri = mtri.Triangulation(node_x, node_y)

mg = MeshGlyph(node_x, node_y, tri.triangles)

# Synthetic face data
cx = node_x[tri.triangles].mean(axis=1)
cy = node_y[tri.triangles].mean(axis=1)
face_data = np.sin(cx * 0.5) * np.cos(cy * 0.4) + 2

fig, ax = mg.plot(face_data, cmap="RdYlBu_r", title="Face-Centered Data")
```

### Node-Centered Contour Plot

```python
# Node data produces smooth interpolated contours
node_data = np.sin(node_x * 0.5) * np.cos(node_y * 0.4) * 3

fig, ax = mg.plot(
    node_data,
    location="node",
    cmap="terrain",
    levels=15,
    title="Node-Centered Contour",
)
```

### Wireframe Outline

```python
# Render mesh edges as a wireframe
fig, ax = mg.plot_outline(color="steelblue", linewidth=0.5)
```

### Overlay Data with Wireframe

```python
# Plot face data, then overlay wireframe on the same axes
mg2 = MeshGlyph(node_x, node_y, tri.triangles)
fig, ax = mg2.plot(face_data, cmap="Blues", title="Data + Wireframe")
mg2.plot_outline(color="black", linewidth=0.2)
```

### Mixed-Element Mesh (Quads + Triangles)

```python
# Mixed meshes use fill_value=-1 for padding
node_x = np.array([0, 1, 2, 0, 1, 2], dtype=float)
node_y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
faces = np.array([
    [0, 1, 4, 3],   # quad
    [1, 2, 5, -1],  # triangle (padded with -1)
    [1, 5, 4, -1],  # triangle
])

mg = MeshGlyph(node_x, node_y, faces, fill_value=-1)
fig, ax = mg.plot(np.array([1.0, 2.0, 3.0]), edgecolor="black")
```

### Color Scales

All color-scale types are chosen via the `color=ColorScaling.…` group object:

```python
from cleopatra.styling.scaling import ColorScaling

mg = MeshGlyph(node_x, node_y, faces, fill_value=-1)

# Power scale (emphasize low values)
fig, ax = mg.plot(data, color=ColorScaling.power(gamma=0.3))

# Symmetrical log scale
fig, ax = mg.plot(data, color=ColorScaling.sym_log())

# Discrete boundary scale
fig, ax = mg.plot(data, color=ColorScaling.boundary(bounds=[0, 2, 4, 6]))

# Midpoint scale (split at a value)
fig, ax = mg.plot(data, color=ColorScaling.midpoint(at=3.0), cmap="RdBu_r")
```

### Colorbar Customization

```python
mg = MeshGlyph(node_x, node_y, faces, fill_value=-1)
fig, ax = mg.plot(
    data,
    colorbar=ColorBar(
        label="Water Depth [m]",
        orientation="horizontal",
        length=0.6,
        label_size=14,
    ),
)
```

### Animation

```python
# Animate time-varying face data on a fixed mesh
mg = MeshGlyph(node_x, node_y, tri.triangles)

# frames: (n_timesteps, n_faces) array
frames = np.array([face_data * (1 + 0.2 * t) for t in range(10)])
time_labels = [f"t={t}" for t in range(10)]

anim = mg.animate(frames, time=time_labels, cmap="plasma", interval=300)
mg.save_animation("mesh_animation.gif", fps=3)
```

### Explicit Edge Connectivity

When edge-node connectivity is available (e.g. from UGRID NetCDF files),
pass it for faster wireframe rendering:

```python
edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
mg = MeshGlyph(node_x, node_y, faces, edge_node_connectivity=edges)
fig, ax = mg.plot_outline()
```
