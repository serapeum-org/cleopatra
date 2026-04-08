# Unstructured Mesh Visualization Guide

This guide explains what unstructured meshes are, their components,
how they differ from regular grids, and how to visualize them with
Cleopatra's `MeshGlyph` class.

## What Is an Unstructured Mesh?

A **regular (structured) grid** divides space into a uniform array of
rectangles. Every cell has the same shape and size, and cell positions
are defined implicitly by row and column indices. This is what
`ArrayGlyph` visualizes.

An **unstructured mesh** divides space into arbitrary polygons --
typically triangles, quadrilaterals, or a mix of both. Cell shapes
and sizes vary freely across the domain, which allows:

- **Higher resolution** where it matters (around structures, along
  coastlines, in steep gradients) without wasting cells in uniform
  regions.
- **Flexible geometry** that conforms to irregular boundaries like
  river banks, coastlines, or building footprints.
- **Mixed elements** combining triangles and quads in the same mesh.

Unstructured meshes are widely used in computational fluid dynamics
(CFD), finite element analysis (FEA), and hydrodynamic modeling
(Delft3D FM, MIKE FM, SCHISM, ADCIRC).

```
Structured Grid              Unstructured Mesh
+---+---+---+---+            +-------+
|   |   |   |   |           / \     / \
+---+---+---+---+          /   \   /   \
|   |   |   |   |         +-----+-----+
+---+---+---+---+          \   / \ . / \
|   |   |   |   |           \ / . .\/ . \
+---+---+---+---+            +-----+-----+
```

## Mesh Components

An unstructured mesh is defined by three types of topological elements:

### Nodes (Vertices)

The fundamental building blocks. Each node is a point in 2D space
with an `(x, y)` coordinate.

```python
node_x = np.array([0.0, 1.0, 0.5, 1.5])
node_y = np.array([0.0, 0.0, 1.0, 1.0])
```

In `MeshGlyph`, nodes are passed as two separate 1D arrays
(`node_x` and `node_y`), each of length `n_nodes`.

### Faces (Cells / Elements)

Polygonal regions bounded by edges. Each face is defined by an
ordered list of node indices that form its vertices.

```python
# Two triangular faces referencing 4 nodes
face_node_connectivity = np.array([
    [0, 1, 2],   # triangle: nodes 0, 1, 2
    [1, 3, 2],   # triangle: nodes 1, 3, 2
])
```

This is the **face-node connectivity** array, shaped
`(n_faces, max_nodes_per_face)`. For mixed meshes where faces have
different numbers of vertices, shorter rows are padded with a
**fill value** (typically `-1`):

```python
# Mixed mesh: 1 quad + 1 triangle
face_node_connectivity = np.array([
    [0, 1, 4, 3],   # quad: 4 nodes
    [1, 2, 5, -1],  # triangle: 3 nodes + padding
])
```

Faces are where simulation results are most commonly defined --
water depth, velocity magnitude, pollutant concentration, etc.

### Edges

Line segments connecting two nodes. Each edge is shared by at most
two faces. Edges are defined by a 2-column array of node index pairs:

```python
edge_node_connectivity = np.array([
    [0, 1],  # edge between node 0 and node 1
    [1, 2],
    [2, 0],
    [1, 3],
    [3, 2],
])
```

Edges are **optional** in `MeshGlyph`. If not provided, they are
derived automatically from the face-node connectivity (with
deduplication). Providing them explicitly is faster for large meshes
since it avoids the derivation step.

## Data Locations

Simulation data can be defined at different mesh locations:

### Face-Centered Data

One value per face. This is the most common output from finite volume
models. Each face gets a single flat color when plotted.

```python
water_depth = np.array([1.5, 2.3])  # one value per face
mg.plot(water_depth, location="face")
```

`MeshGlyph` uses matplotlib's `tripcolor` for face-centered rendering.
Internally, polygonal faces are decomposed into triangles via **fan
triangulation** (a quad becomes 2 triangles, a pentagon becomes 3,
etc.), and each sub-triangle receives its parent face's value.

### Node-Centered Data

One value per node. Used when data is defined at vertices (e.g.,
finite element solutions). Produces smooth interpolated contours.

```python
elevation = np.array([0.0, 1.0, 2.0, 3.0])  # one value per node
mg.plot(elevation, location="node")
```

`MeshGlyph` uses matplotlib's `tricontourf` for node-centered
rendering, which interpolates between node values to produce smooth
filled contour plots.

## The UGRID Convention

[UGRID](https://ugrid-conventions.github.io/ugrid-conventions/) is a
standard for storing unstructured mesh data in NetCDF files. It defines
how mesh topology (nodes, edges, faces, connectivity arrays) and data
variables are organized.

A typical UGRID file contains:

| Variable | Shape | Description |
|----------|-------|-------------|
| `mesh2d_node_x` | `(n_nodes,)` | Node x-coordinates |
| `mesh2d_node_y` | `(n_nodes,)` | Node y-coordinates |
| `mesh2d_face_nodes` | `(n_faces, max_nodes)` | Face-node connectivity |
| `mesh2d_edge_nodes` | `(n_edges, 2)` | Edge-node connectivity |
| `mesh2d_waterdepth` | `(n_faces,)` | Face-centered data |
| `mesh2d_s1` | `(n_nodes,)` | Node-centered data |

`MeshGlyph` maps directly to these arrays:

```python
import netCDF4 as nc

ds = nc.Dataset("output.nc")
mg = MeshGlyph(
    node_x=ds["mesh2d_node_x"][:],
    node_y=ds["mesh2d_node_y"][:],
    face_node_connectivity=ds["mesh2d_face_nodes"][:],
    fill_value=ds["mesh2d_face_nodes"]._FillValue,
    edge_node_connectivity=ds["mesh2d_edge_nodes"][:],
)
fig, ax = mg.plot(ds["mesh2d_waterdepth"][:], title="Water Depth")
```

## Fan Triangulation

Matplotlib can only render triangles natively. When a mesh contains
quads, pentagons, or other polygons, `MeshGlyph` decomposes them into
triangles using **fan triangulation**:

```
Fan triangulation of a quad:

    3 ---- 2          3 ---- 2        3 ---- 2
    |      |          |    / |        | \    |
    |      |   -->    |   /  |   or   |  \   |
    |      |          |  /   |        |   \  |
    0 ---- 1          0 ---- 1        0 ---- 1

    Original          Fan from        Fan from
    quad              node 0          node 3
```

`MeshGlyph` fans from the first vertex of each face. A face with
`N` valid nodes produces `N - 2` triangles. All sub-triangles inherit
the same data value from their parent face.

For pure-triangle meshes (the most common case), this step is skipped
entirely -- the face-node connectivity is already the triangle array.

## Visualization Modes

### Face Plot (`location="face"`)

Each face is colored by its data value using flat shading. Good for
showing discrete per-cell results from finite volume models.

```python
fig, ax = mg.plot(face_data, location="face", cmap="viridis")
```

### Node Contour Plot (`location="node"`)

Smooth filled contours interpolated between node values. Good for
showing continuous fields from finite element models.

```python
fig, ax = mg.plot(node_data, location="node", cmap="terrain", levels=20)
```

### Wireframe Outline

Renders mesh edges only, without data. Useful for inspecting mesh
quality, resolution distribution, or overlaying on a data plot.

```python
fig, ax = mg.plot_outline(color="black", linewidth=0.3)
```

Since `MeshGlyph` stores `fig`/`ax` as instance state, you can overlay
a wireframe on a data plot by calling both methods in sequence:

```python
mg = MeshGlyph(node_x, node_y, faces)
mg.plot(data, cmap="Blues")
mg.plot_outline(color="white", linewidth=0.1)
```

### Animation

For time-varying data on a fixed mesh (e.g., a hydrodynamic
simulation), `animate()` creates a frame-by-frame animation:

```python
# frames: shape (n_timesteps, n_faces)
mg.animate(frames, time=time_labels, cmap="plasma")
mg.save_animation("simulation.gif", fps=5)
```

## Color Scales

`MeshGlyph` supports all 5 color scale types inherited from the
`Glyph` base class:

| Scale | Use case | Key parameters |
|-------|----------|----------------|
| `linear` | Default. Uniform mapping from data to color. | `vmin`, `vmax` |
| `power` | Emphasize low or high values. | `gamma` (< 1 emphasizes low) |
| `sym-lognorm` | Data spanning many orders of magnitude with a zero crossing. | `line_threshold`, `line_scale` |
| `boundary-norm` | Discrete color bins at specific thresholds. | `bounds` (list of boundaries) |
| `midpoint` | Split the colormap at a meaningful value (e.g., zero for difference plots). | `midpoint` |

```python
# Example: midpoint scale for a difference plot
mg.plot(
    difference_data,
    color_scale="midpoint",
    midpoint=0.0,
    cmap="RdBu_r",
    cbar_label="Change [m]",
)
```

## Comparison: ArrayGlyph vs MeshGlyph

| Aspect | ArrayGlyph | MeshGlyph |
|--------|-----------|-----------|
| **Data type** | Regular 2D/3D numpy arrays | Unstructured mesh (nodes + connectivity) |
| **Rendering** | `imshow` / `matshow` | `tripcolor` / `tricontourf` |
| **Cell shape** | Uniform rectangles | Arbitrary polygons (triangles, quads, mixed) |
| **Resolution** | Uniform everywhere | Variable -- dense where needed |
| **Wireframe** | N/A | `plot_outline()` |
| **Face vs Node** | N/A (always grid cells) | `location="face"` or `location="node"` |
| **Color scales** | All 5 types | All 5 types |
| **Animation** | `animate()` over 3D array slices | `animate()` over list of data arrays |
| **RGB support** | Yes | No |
| **Shared base** | `Glyph` | `Glyph` |
