"""Unstructured mesh visualization.

Provides `MeshGlyph` for plotting UGRID-style unstructured mesh data
using matplotlib triangulation (tripcolor, tricontourf) and wireframe
rendering via LineCollection. Accepts raw numpy arrays of node
coordinates and face-node connectivity. Also integrates with
pyramids-gis `Mesh2d` objects for geospatial workflows.

Examples:
    - Plot face-centered data on a triangular mesh:
        ```python
        >>> import numpy as np
        >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
        >>> node_x = np.array([0.0, 1.0, 0.5, 1.5])
        >>> node_y = np.array([0.0, 0.0, 1.0, 1.0])
        >>> face_nodes = np.array([[0, 1, 2], [1, 3, 2]])
        >>> face_data = np.array([10.0, 20.0])
        >>> mg = MeshGlyph(node_x, node_y, face_nodes)
        >>> fig, ax = mg.plot(face_data, location="face", title="Water Level")

        ```
    - Plot a wireframe outline:
        ```python
        >>> fig, ax = mg.plot_outline(color="blue", linewidth=0.5)

        ```
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colorbar import Colorbar
from matplotlib.colors import BoundaryNorm, ListedColormap

from cleopatra.styling.colorbar import ColorBar, _resolve_colorbar, _warn_deprecated_cbar_kwargs
from cleopatra.styling.colors import (
    category_boundaries,
    resolve_colormap,
    resolve_single_layer_style,
    resolve_style_norm,
)
from cleopatra.basemap.geo import GeoMixin
from cleopatra.glyphs.base.glyph import (
    Glyph,
    _clear_prior_render_artists,
    _clear_projection_frame,
    _mark_render_artists,
    _restore_flat_axes,
    _stash_projection_frame,
)
from cleopatra.glyphs.base.hillshade import resolve_hillshade, shade_faces
from cleopatra.basemap.projection import apply_projection_style_mesh, projection_draws_frame
from cleopatra.styling.params import Contour, DataStyle
from cleopatra.styling.scaling import ColorScaling
from cleopatra.styling.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS
from cleopatra.styling.styles import disjoint_legend

MESH_DEFAULT_OPTIONS = {
    "vmin": None,
    "vmax": None,
    "labels": False,
    "label_kw": None,
    "hillshade": False,
    "style": None,
    "projection": None,
}
MESH_DEFAULT_OPTIONS = STYLE_DEFAULTS | MESH_DEFAULT_OPTIONS

#: Sentinel distinguishing "hillshade not forwarded" from an explicit
#: `hillshade=None` in `apply_style`, so an unset value keeps any sticky
#: relief shading rather than clearing it.
_UNSET_HILLSHADE = object()


class MeshGlyph(GeoMixin, Glyph):
    """Visualization class for unstructured mesh data.

    Wraps matplotlib's triangulation-based rendering to plot data on
    UGRID-style unstructured meshes (triangles, quads, mixed polygons).
    Handles fan triangulation for mixed meshes and maps face-centered
    values to individual triangles.

    Args:
        node_x: 1D array of node x-coordinates (n_nodes,).
        node_y: 1D array of node y-coordinates (n_nodes,).
        face_node_connectivity: 2D array of node indices per face
            (n_faces, max_nodes_per_face). Use `fill_value` to pad
            rows for faces with fewer nodes.
        fill_value: Padding value in `face_node_connectivity` for
            mixed meshes. Default is -1.
        edge_node_connectivity: 2D array of node indices per edge
            (n_edges, 2). If provided, used for efficient wireframe
            rendering. If None, edges are derived from face
            connectivity. Default is None.

    Attributes:
        node_x: Node x-coordinates.
        node_y: Node y-coordinates.
        n_faces: Number of faces in the mesh.
        n_nodes: Number of nodes in the mesh.
        n_edges: Number of edges (0 if edge connectivity not provided).
        contour_labels: The inline contour-label `Text` artists from the
            most recent `plot(location="node", filled=False, labels=True)`,
            or `None` when labelling was not requested (the default, and
            for `tripcolor`/`tricontourf`). A labelled line tricontour with
            no isolines (e.g. a constant-value field) yields an empty list.

    Examples:
        - Create a MeshGlyph and inspect its topology:
            ```python
            >>> import numpy as np
            >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
            >>> node_x = np.array([0.0, 1.0, 0.5])
            >>> node_y = np.array([0.0, 0.0, 1.0])
            >>> faces = np.array([[0, 1, 2]])
            >>> mg = MeshGlyph(node_x, node_y, faces)
            >>> mg.n_faces
            1
            >>> mg.n_nodes
            3

            ```
    """

    #: Option keys this glyph accepts (see `Glyph.option_keys`/`filter_kwargs`).
    DEFAULT_OPTIONS = MESH_DEFAULT_OPTIONS

    def __init__(
        self,
        node_x: np.ndarray,
        node_y: np.ndarray,
        face_node_connectivity: np.ndarray,
        fill_value: int = -1,
        edge_node_connectivity: np.ndarray | None = None,
        fig=None,
        ax=None,
        **kwargs,
    ):
        super().__init__(default_options=MESH_DEFAULT_OPTIONS, fig=fig, ax=ax, **kwargs)
        self._node_x = np.asarray(node_x, dtype=np.float64)
        self._node_y = np.asarray(node_y, dtype=np.float64)
        self._face_nodes = np.asarray(face_node_connectivity, dtype=np.intp)
        self._fill_value = fill_value
        self._edge_nodes = (
            np.asarray(edge_node_connectivity, dtype=np.intp)
            if edge_node_connectivity is not None
            else None
        )

        if self._node_x.ndim != 1:
            raise ValueError(f"node_x must be 1D, got {self._node_x.ndim}D.")
        if self._node_x.shape != self._node_y.shape:
            raise ValueError(
                f"node_x and node_y must have the same shape, "
                f"got {self._node_x.shape} and {self._node_y.shape}."
            )
        if self._face_nodes.ndim != 2:
            raise ValueError(
                f"face_node_connectivity must be 2D, got {self._face_nodes.ndim}D."
            )
        valid_indices = self._face_nodes[self._face_nodes != self._fill_value]
        if len(valid_indices) > 0:
            if valid_indices.min() < 0 or valid_indices.max() >= self.n_nodes:
                raise ValueError(
                    f"face_node_connectivity indices must be in "
                    f"[0, {self.n_nodes}), got range "
                    f"[{valid_indices.min()}, {valid_indices.max()}]."
                )
        if self._edge_nodes is not None:
            if self._edge_nodes.ndim != 2 or self._edge_nodes.shape[1] != 2:
                raise ValueError(
                    f"edge_node_connectivity must have shape (n_edges, 2), "
                    f"got {self._edge_nodes.shape}."
                )

        self._cached_triangulation: mtri.Triangulation | None = None
        self._cached_tri_array: np.ndarray | None = None
        self._cached_nodes_per_face: np.ndarray | None = None
        self._cbar: Colorbar | None = None
        #: Colour-mapped artist from the most recent `plot` call (the
        #: `tripcolor`/`tricontour(f)` mappable); `None` before first render.
        self.im: Any = None
        #: Per-frame time-label `Text` artist from the most recent
        #: `animate` call, if any; `None` before any `animate` call.
        self._day_text = None
        #: Inline contour-label `Text` artists from the most recent
        #: `plot(location="node", filled=False, labels=True)`, or `None`
        #: when labelling was not requested (the default, and for
        #: `tripcolor`/`tricontourf`); an empty list when the line
        #: tricontour has no isolines.
        self.contour_labels = None
        #: `hillshade` set at construction. `plot()` resets `default_options`
        #: to the class defaults on each call, so this is restored there when
        #: `hillshade` is not overridden at `plot()` time -- keeping the option
        #: honoured at construction, consistent with `ArrayGlyph`/`KDEGlyph`.
        self._construct_hillshade = self.default_options.get("hillshade", False)
        #: The sticky `style` preset. `plot()` resets `default_options` each
        #: call, so this tracks the current preset (updated whenever `plot()` /
        #: `apply_style` passes `style`, including `None` to clear) and is
        #: restored after the reset -- so a style survives a later plain
        #: `plot(data)` (sticky + clearable, like `ArrayGlyph`).
        self._style_state = self.default_options.get("style")
        #: Sticky `projection` preset (like `_style_state`): restored after the
        #: `plot()` options reset so a constructor-time `projection=` survives a
        #: later plain `plot(data)`.
        self._projection_state = self.default_options.get("projection")
        #: Last `(data, location)` rendered, so `apply_style` can restyle in
        #: place without the caller re-supplying the mesh data.
        self._last_data: np.ndarray | None = None
        self._last_location = "face"

    @property
    def node_x(self) -> np.ndarray:
        """Node x-coordinates.

        Returns:
            np.ndarray: 1D float array of node x-coordinates, in node
                order (length ``n_nodes``).

        Examples:
            - Read back the x-coordinates and pick out a single node:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> mg = MeshGlyph(
                ...     np.array([0.0, 1.0, 0.5]),
                ...     np.array([0.0, 0.0, 1.0]),
                ...     np.array([[0, 1, 2]]),
                ... )
                >>> mg.node_x
                array([0. , 1. , 0.5])
                >>> float(mg.node_x[1])
                1.0

                ```
        """
        return self._node_x

    @property
    def node_y(self) -> np.ndarray:
        """Node y-coordinates.

        Returns:
            np.ndarray: 1D float array of node y-coordinates, in node
                order (length ``n_nodes``).

        Examples:
            - Read back the y-coordinates and take their maximum:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> mg = MeshGlyph(
                ...     np.array([0.0, 1.0, 0.5]),
                ...     np.array([0.0, 0.0, 1.0]),
                ...     np.array([[0, 1, 2]]),
                ... )
                >>> mg.node_y
                array([0., 0., 1.])
                >>> float(mg.node_y.max())
                1.0

                ```
        """
        return self._node_y

    @property
    def n_faces(self) -> int:
        """Number of faces in the mesh.

        Returns:
            int: Count of faces (rows of the face-node connectivity),
                regardless of how many nodes each face has.

        Examples:
            - A two-face mesh reports two faces, one row per face:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> mg = MeshGlyph(
                ...     np.array([0.0, 1.0, 0.5, 1.5]),
                ...     np.array([0.0, 0.0, 1.0, 1.0]),
                ...     np.array([[0, 1, 2], [1, 3, 2]]),
                ... )
                >>> mg.n_faces
                2

                ```
        """
        return int(self._face_nodes.shape[0])

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the mesh.

        Returns:
            int: Count of nodes, i.e. the length of the coordinate
                arrays ``node_x``/``node_y``.

        Examples:
            - The node count matches the coordinate array length:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> mg = MeshGlyph(
                ...     np.array([0.0, 1.0, 0.5, 1.5]),
                ...     np.array([0.0, 0.0, 1.0, 1.0]),
                ...     np.array([[0, 1, 2], [1, 3, 2]]),
                ... )
                >>> mg.n_nodes
                4

                ```
        """
        return len(self._node_x)

    @property
    def n_edges(self) -> int:
        """Number of edges (0 if edge connectivity not provided).

        Edges are only counted when explicit ``edge_node_connectivity``
        was supplied at construction; otherwise this is 0 even though the
        mesh has implicit polygon edges (which ``plot_outline`` derives on
        demand).

        Returns:
            int: Number of rows in ``edge_node_connectivity``, or 0 when
                no edge connectivity was given.

        Examples:
            - Without explicit edges the count is 0:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> mg = MeshGlyph(
                ...     np.array([0.0, 1.0, 0.5]),
                ...     np.array([0.0, 0.0, 1.0]),
                ...     np.array([[0, 1, 2]]),
                ... )
                >>> mg.n_edges
                0

                ```
            - Supplying ``edge_node_connectivity`` makes the count match
                the number of edge rows:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> mg = MeshGlyph(
                ...     np.array([0.0, 1.0, 1.0, 0.0]),
                ...     np.array([0.0, 0.0, 1.0, 1.0]),
                ...     np.array([[0, 1, 2, 3]]),
                ...     edge_node_connectivity=np.array(
                ...         [[0, 1], [1, 2], [2, 3], [3, 0]]
                ...     ),
                ... )
                >>> mg.n_edges
                4

                ```
        """
        return self._edge_nodes.shape[0] if self._edge_nodes is not None else 0

    @property
    def nodes_per_face(self) -> np.ndarray:
        """Number of valid nodes per face (excluding fill values).

        Returns:
            np.ndarray: 1D integer array of length n_faces.

        Examples:
            - Pure triangular mesh returns all 3s:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> mg = MeshGlyph(
                ...     np.array([0.0, 1.0, 0.5, 1.5]),
                ...     np.array([0.0, 0.0, 1.0, 1.0]),
                ...     np.array([[0, 1, 2], [1, 3, 2]]),
                ... )
                >>> mg.nodes_per_face
                array([3, 3])

                ```
            - Mixed mesh with quads and triangles:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> mg = MeshGlyph(
                ...     np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
                ...     np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
                ...     np.array([[0, 1, 4, 3], [1, 2, 5, -1]]),
                ...     fill_value=-1,
                ... )
                >>> mg.nodes_per_face
                array([4, 3])

                ```
        """
        if self._cached_nodes_per_face is None:
            self._cached_nodes_per_face = np.sum(
                self._face_nodes != self._fill_value, axis=1
            ).astype(np.intp)
        return self._cached_nodes_per_face

    @property
    def triangulation(self) -> mtri.Triangulation:
        """Matplotlib Triangulation built via fan decomposition.

        Each face with N valid nodes is decomposed into (N-2)
        triangles by fanning from the first vertex. Faces with
        fewer than 3 valid nodes are skipped.

        Returns:
            matplotlib.tri.Triangulation: Triangulation ready for
                tripcolor/tricontourf.

        Raises:
            ValueError: If no faces have 3 or more valid nodes.

        Examples:
            - Build a triangulation and check its shape:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> mg = MeshGlyph(
                ...     np.array([0.0, 1.0, 0.5]),
                ...     np.array([0.0, 0.0, 1.0]),
                ...     np.array([[0, 1, 2]]),
                ... )
                >>> tri = mg.triangulation
                >>> tri.triangles.shape
                (1, 3)

                ```
        """
        if self._cached_triangulation is None:
            tri_array = self._fan_triangles()
            self._cached_triangulation = mtri.Triangulation(
                self._node_x, self._node_y, tri_array
            )
        return self._cached_triangulation

    def _fan_triangles(self) -> np.ndarray:
        """Compute fan triangulation for mixed-element meshes.

        Each face with N valid nodes is decomposed into (N-2) triangles
        using fan decomposition from the first vertex. Pure-triangle
        meshes use a fast path that returns the connectivity directly.

        Returns:
            np.ndarray: (n_triangles, 3) array of node indices.

        Raises:
            ValueError: If no valid triangles can be formed.

        Examples:
            - A single quad fans into two triangles from its first vertex:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> mg = MeshGlyph(
                ...     np.array([0.0, 1.0, 1.0, 0.0]),
                ...     np.array([0.0, 0.0, 1.0, 1.0]),
                ...     np.array([[0, 1, 2, 3]]),
                ... )
                >>> mg._fan_triangles()
                array([[0, 1, 2],
                       [0, 2, 3]])

                ```
            - A mixed mesh (quad + triangle) keeps faces in order; the
                quad's two triangles come first, then the triangle:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> mg = MeshGlyph(
                ...     np.array([0.0, 1.0, 2.0, 0.0, 1.0]),
                ...     np.array([0.0, 0.0, 0.0, 1.0, 1.0]),
                ...     np.array([[0, 1, 4, 3], [1, 2, 4, -1]]),
                ...     fill_value=-1,
                ... )
                >>> mg._fan_triangles()
                array([[0, 1, 4],
                       [0, 4, 3],
                       [1, 2, 4]])

                ```
        """
        if self._cached_tri_array is not None:
            return self._cached_tri_array

        counts = self.nodes_per_face

        if not np.any(counts >= 3):
            raise ValueError("Cannot create triangulation: no faces with 3+ nodes.")

        if self._face_nodes.shape[1] == 3 and np.all(counts == 3):
            self._cached_tri_array = self._face_nodes.copy()
            return self._cached_tri_array

        flat_nodes = self._face_nodes[self._face_nodes != self._fill_value]
        face_start = np.cumsum(counts) - counts

        # A face with c valid nodes produces (c - 2) fan triangles.
        valid = counts >= 3
        base = np.repeat(face_start[valid], counts[valid] - 2)
        t = self._grouped_arange(counts[valid] - 2)

        # Triangle (v0, v_{t+1}, v_{t+2}) fanning from each face's first vertex.
        first = flat_nodes[base]
        second = flat_nodes[base + 1 + t]
        third = flat_nodes[base + 2 + t]

        self._cached_tri_array = np.stack([first, second, third], axis=1)
        return self._cached_tri_array

    @staticmethod
    def _grouped_arange(sizes: np.ndarray) -> np.ndarray:
        """Concatenated per-group ranges: ``[0..s0-1, 0..s1-1, ...]``.

        Vectorized equivalent of
        ``np.concatenate([np.arange(s) for s in sizes])``. Zero-size groups
        contribute nothing and are handled correctly.

        Args:
            sizes: 1D array of non-negative group sizes.

        Returns:
            np.ndarray: 1D intp array of length ``sizes.sum()``.

        Examples:
            - Each group ``i`` contributes the range ``0..sizes[i]-1``,
                concatenated in order:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> MeshGlyph._grouped_arange(np.array([2, 3, 1]))
                array([0, 1, 0, 1, 2, 0])

                ```
            - Zero-size groups contribute nothing and do not shift the
                counter of later groups:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> MeshGlyph._grouped_arange(np.array([2, 0, 3]))
                array([0, 1, 0, 1, 2])

                ```
        """
        sizes = np.asarray(sizes, dtype=np.intp)
        total = int(sizes.sum())
        if total == 0:
            return np.empty(0, dtype=np.intp)
        group_start = np.cumsum(sizes) - sizes
        return np.asarray(
            np.arange(total, dtype=np.intp) - np.repeat(group_start, sizes)
        )

    def _map_face_to_triangle_values(self, face_values: np.ndarray) -> np.ndarray:
        """Map per-face values to per-triangle values.

        Each original face may produce multiple triangles via fan
        decomposition. All triangles from the same face receive
        the same data value.

        Args:
            face_values: 1D array of values, one per face.

        Returns:
            np.ndarray: 1D array of values, one per triangle.

        Examples:
            - Quad face produces 2 triangles with the same value:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> mg = MeshGlyph(
                ...     np.array([0.0, 1.0, 1.0, 0.0]),
                ...     np.array([0.0, 0.0, 1.0, 1.0]),
                ...     np.array([[0, 1, 2, 3]]),
                ... )
                >>> mg._map_face_to_triangle_values(np.array([42.0]))
                array([42., 42.])

                ```
        """
        counts = self.nodes_per_face
        valid = counts >= 3
        return np.repeat(face_values[valid], counts[valid] - 2)

    def _validate_location_and_data(self, data: np.ndarray, location: str) -> None:
        """Validate location string and data length."""
        if location not in ("face", "node"):
            raise ValueError(
                f"Plotting not supported for location='{location}'. "
                f"Use 'face' or 'node'."
            )
        expected = self.n_faces if location == "face" else self.n_nodes
        if len(data) != expected:
            raise ValueError(
                f"data length ({len(data)}) does not match n_{location}s ({expected})."
            )

    def _apply_projection(self) -> None:
        """Reproject the mesh onto the `projection` preset and frame the axes.

        Replaces the cached triangulation with one on the reprojected (e.g.
        orthographic-globe) node coordinates -- far-hemisphere triangles masked
        -- and draws the globe boundary + graticule. Must run after the axes is
        cleared and before the mesh is drawn, so the render uses the reprojected
        triangulation.

        Owns the projection frame's lifecycle: every call first removes the
        boundary/graticule drawn on a previous render (they are not
        `_mark_render_artists`-tracked, so a replot would otherwise stack a
        duplicate frame). When `projection` is cleared to a falsy value, it drops
        the reprojected triangulation cache so the `triangulation` property
        rebuilds the flat mesh, and -- if a globe frame was present -- restores
        the flat axes view (the globe froze the limits/axis-off), otherwise a
        `plot(projection=None)` after a globe would silently render the flat mesh
        into a frozen, axis-off view as an invisible speck.
        """
        had_frame = _clear_projection_frame(self.ax)
        projection = self.default_options.get("projection")
        if not projection_draws_frame(projection):
            self._cached_triangulation = None
            if had_frame:
                _restore_flat_axes(
                    self.ax,
                    float(self._node_x.min()),
                    float(self._node_x.max()),
                    float(self._node_y.min()),
                    float(self._node_y.max()),
                    aspect="equal",
                )
            return
        before = set(map(id, self.ax.patches)) | set(map(id, self.ax.lines))
        self._cached_triangulation = apply_projection_style_mesh(
            self.ax, self._node_x, self._node_y, self._fan_triangles(), style=projection
        )
        _stash_projection_frame(
            self.ax,
            [a for a in (*self.ax.patches, *self.ax.lines) if id(a) not in before],
        )

    def _render_mesh(
        self,
        ax,
        data: np.ndarray,
        location: str,
        edgecolor: str = "none",
        norm=None,
        filled: bool = True,
        **render_kwargs,
    ):
        """Render mesh data on axes and return the mappable.

        Args:
            ax: Matplotlib axes.
            data: 1D data array.
            location: `"face"` or `"node"`.
            edgecolor: Edge color for face rendering.
            norm: Color normalization.
            filled: For node data, whether to draw filled contours
                (`tricontourf`, the default) or line contours
                (`tricontour`). Ignored for face data, which always uses
                `tripcolor`.
            **render_kwargs: Passed to tripcolor, tricontourf, or
                tricontour.

        Returns:
            ScalarMappable: The tripcolor, tricontourf, or tricontour
                result.
        """
        tri = self.triangulation
        cmap = resolve_colormap(self.default_options["cmap"])
        vmin = self.default_options["vmin"]
        vmax = self.default_options["vmax"]

        if location == "face":
            tri_values = self._map_face_to_triangle_values(data)
            kw: dict[str, Any] = {"cmap": cmap, "edgecolors": edgecolor}
            if norm is not None:
                kw["norm"] = norm
            else:
                kw["vmin"] = vmin
                kw["vmax"] = vmax
            kw.update(render_kwargs)
            return ax.tripcolor(tri, facecolors=tri_values, **kw)

        contour_kw: dict[str, Any] = {"cmap": cmap, "levels": 20}
        if norm is not None:
            contour_kw["norm"] = norm
        else:
            if vmin is not None:
                contour_kw["vmin"] = vmin
            if vmax is not None:
                contour_kw["vmax"] = vmax
        contour_kw.update(render_kwargs)
        if filled:
            return ax.tricontourf(tri, data, **contour_kw)
        return ax.tricontour(tri, data, **contour_kw)

    def _render_shaded_relief(
        self,
        ax: Any,
        data: np.ndarray,
        edgecolor: str,
        norm: Any,
        hillshade: dict[str, Any],
        **render_kwargs: Any,
    ) -> Any:
        """Render the mesh as a relief-shaded terrain surface (node elevation).

        Colours each triangle by its mean node elevation, then blends the
        triangle-normal hillshade into those colours via
        `cleopatra.glyphs.base.hillshade.shade_faces`, so a wide-range terrain mesh reads
        by form. The returned `tripcolor` mappable keeps its cmap/norm, so the
        colorbar spans the **node** elevation range (`vmin`/`vmax` from the
        per-node `data`). Note that faces are coloured by each triangle's
        *mean* node elevation, whose range is narrower than the node range on
        any mesh with within-triangle variation, so the colorbar's extreme
        colours may not appear on the surface — the bar reflects the input
        data range, not the drawn per-face means. Requires node-centered
        `data` (the surface's per-node elevation).

        Note:
            Hillshade is intended for native (flat) coordinates. Under
            `projection="globe"` the triangulation is in orthographic **metres**
            (~1e6) while elevations stay in metres (~1e3), so the surface reads
            as nearly flat in the xy frame and the relief washes out. Combine
            hillshade with the plain (flat) mesh, not the globe.

        Args:
            ax: Axes to draw on.
            data: Node-centered elevation (one value per mesh node).
            edgecolor: Triangle edge colour.
            norm: Colour normalization, or `None` to use `vmin`/`vmax`.
            hillshade: Resolved hillshade settings.
            **render_kwargs: Forwarded to `tripcolor`.

        Returns:
            The `tripcolor` mappable, with per-face colours set to the shaded
            RGBA.
        """
        tri = self.triangulation
        z_nodes = np.asarray(data, dtype=float)
        tri_faces = tri.triangles
        tri_z = z_nodes[tri_faces].mean(axis=1)

        kw: dict[str, Any] = {
            "cmap": resolve_colormap(self.default_options["cmap"]),
            "edgecolors": edgecolor,
        }
        if norm is not None:
            kw["norm"] = norm
        else:
            kw["vmin"] = self.default_options["vmin"]
            kw["vmax"] = self.default_options["vmax"]
        kw.update(render_kwargs)
        tpc = ax.tripcolor(tri, facecolors=tri_z, **kw)

        node_xy = np.column_stack([tri.x, tri.y])
        base_rgba = tpc.to_rgba(tri_z)
        shaded = shade_faces(node_xy, tri_faces, z_nodes, base_rgba, **hillshade)
        nan_faces = ~np.isfinite(z_nodes[tri_faces]).all(axis=1)
        shaded[nan_faces] = 0.0
        tpc.set_array(None)
        tpc.set_alpha(None)
        tpc.set_facecolor(shaded)
        return tpc

    @property
    def style(self) -> str | None:
        """Name of the `DATA_STYLES` preset currently applied, or `None`.

        Reads back the preset set via the `style` constructor kwarg, a
        `plot(style=...)` call, or `apply_style`.
        """
        return self.default_options.get("style")

    def apply_style(
        self, style: str, data: np.ndarray | None = None, **kwargs: Any
    ) -> tuple[plt.Figure, plt.Axes]:
        """Apply a `DATA_STYLES` preset by name, re-rendering the mesh in place.

        A discoverable wrapper over `plot(style=...)` for restyling an
        already-built glyph. It redraws **in place** on the glyph's own axes
        (taking full ownership -- do not use on a shared axes), or on a fresh
        figure if the glyph was never plotted or its figure was closed. It
        reuses the last-plotted mesh data (and location) so the caller need not
        re-supply it; pass `data=` when the glyph has not been plotted yet. The
        applied style is **sticky** (survives a later plain `plot(data)`);
        `plot(data, style=None)` clears it. Extra keyword arguments (e.g.
        `location`, `hillshade`, `edgecolor`) are forwarded to `plot`.

        Args:
            style: A `cleopatra.styling.colors.DATA_STYLES` preset name.
            data: Mesh data to render; defaults to the last-plotted data.
            **kwargs: Forwarded to `plot` (e.g. `location`, `hillshade`).

        Returns:
            tuple[Figure, Axes]: The figure and axes drawn on.

        Raises:
            ValueError: If `style` is unknown (raised by `plot`), or no data is
                available (never plotted and none passed).
        """
        resolve_single_layer_style(style)
        if data is None:
            data = self._last_data
            if data is None:
                raise ValueError(
                    "apply_style needs mesh data: call plot(data, ...) first, "
                    "or pass data= explicitly."
                )
        location = kwargs.pop("location", self._last_location)
        self._reset_axes_for_restyle()
        # Fold style (and an optional forwarded hillshade) into the grouped
        # data_style object; leaving hillshade unset keeps any sticky value.
        hillshade = kwargs.pop("hillshade", _UNSET_HILLSHADE)
        data_style = (
            DataStyle(style=style)
            if hillshade is _UNSET_HILLSHADE
            else DataStyle(style=style, hillshade=hillshade)
        )
        return self.plot(
            data, location=location, ax=self.ax, data_style=data_style, **kwargs
        )

    def plot(
        self,
        data: np.ndarray,
        location: str = "face",
        ax: Any = None,
        edgecolor: str = "none",
        colorbar: bool | ColorBar | None = True,
        title: str | None = None,
        filled: bool = True,
        color: ColorScaling | None = None,
        contour: Contour | None = None,
        data_style: DataStyle | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot mesh data using matplotlib triangulation.

        For face-centered data, uses `tripcolor` where each triangle
        is colored by the value of its parent face. For node-centered
        data, uses `tricontourf` for smooth interpolated filled
        contours, or `tricontour` for line contours when
        `filled=False`.

        Supports all 5 color scale types from `default_options`:
        linear, power, sym-lognorm, boundary-norm, and midpoint.

        Args:
            data: 1D data array. Length must match face count
                (location="face") or node count (location="node").
            location: Mesh element location: `"face"` or `"node"`.
                Default is `"face"`.
            ax: Axes to plot on. If None, uses stored axes or creates
                new.
            edgecolor: Edge color for face rendering. Default is
                `"none"`.
            colorbar: Draw a colorbar, by default `True`. Accepts a typed
                `ColorBar` spec (placement / caption / sizing) or `True`/`None`
                to draw a default one; `False` suppresses it.
            title: Plot title. Overrides `default_options["title"]`.
            filled: For node data, draw filled contours (`tricontourf`,
                the default) or line contours (`tricontour`) when
                `False`. Ignored for face data. Default is True.
            **kwargs: Override any key in `default_options` (cmap,
                vmin, vmax, color_scale, gamma, midpoint, bounds,
                figsize, etc.) or pass extra rendering kwargs (levels for
                tricontourf / tricontour). The loose `ticks_spacing` /
                `cbar_*` keys still work but are deprecated -- pass
                `colorbar=ColorBar(...)` instead. Two label options are
                honoured **only** for line tricontours
                (`location="node"`, `filled=False`):

                - `labels` (bool, default `False`): when truthy, draw
                  inline numeric labels on the isolines via `ax.clabel`
                  and store the resulting `Text` artists on
                  `self.contour_labels`. A documented no-op for
                  `tripcolor` (face data) and `tricontourf`
                  (`filled=True`), which leave `contour_labels` as `None`.
                - `label_kw` (dict): forwarded to `ax.clabel`, merged
                  over cleopatra's defaults (`inline=True`, `fontsize=8`,
                  `fmt="%g"`) so user keys (`fmt`, `fontsize`, `colors`,
                  `inline_spacing`, …) win on collision.

                One relief option is honoured **only** for node data
                (`location="node"`):

                - `hillshade` (bool | dict, default `False`): render the
                  mesh as a relief-shaded terrain surface. Each triangle is
                  coloured by its *mean* node elevation and blended with a
                  triangle-normal hillshade. The colorbar spans the **node**
                  elevation range, so — because faces use per-triangle means
                  — its extreme colours may not appear on the surface; the
                  bar reflects the input data range, not the drawn per-face
                  colours. Faces touching a non-finite (nodata) node render
                  transparent. Passing `hillshade` with `location="face"`
                  raises `ValueError`.

                A data-style preset option:

                - `style` (str, default `None`): name of a
                  `cleopatra.styling.colors.DATA_STYLES` preset (valid names:
                  `sorted(cleopatra.styling.colors.DATA_STYLES)`). A continuous preset
                  overrides the cmap + norm (and composes with `hillshade`); a
                  categorical preset builds a discrete colormap, masks
                  out-of-range codes transparent, and draws a legend instead of
                  the colorbar. Takes precedence over `cmap`/`color_scale`.

        Returns:
            tuple[Figure, Axes]: The matplotlib Figure and Axes objects.
                When no axes exist, a new figure is created. Call
                `plt.close(fig)` after saving to avoid memory leaks
                in batch processing.

        Raises:
            ValueError: If `location` is not `"face"` or `"node"`,
                or if `data` length does not match the expected mesh
                dimension.

        Examples:
            - Plot face-centered data:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> node_x = np.array([0.0, 1.0, 0.5, 1.5])
                >>> node_y = np.array([0.0, 0.0, 1.0, 1.0])
                >>> faces = np.array([[0, 1, 2], [1, 3, 2]])
                >>> mg = MeshGlyph(node_x, node_y, faces)
                >>> fig, ax = mg.plot(np.array([1.0, 2.0]))

                ```
            - Plot node-centered data:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> node_x = np.array([0.0, 1.0, 0.5, 1.5])
                >>> node_y = np.array([0.0, 0.0, 1.0, 1.0])
                >>> faces = np.array([[0, 1, 2], [1, 3, 2]])
                >>> mg = MeshGlyph(node_x, node_y, faces)
                >>> fig, ax = mg.plot(
                ...     np.array([0.0, 1.0, 2.0, 3.0]),
                ...     location="node",
                ... )

                ```
            - Plot node-centered data as line contours
                (`tricontour`) instead of filled:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> node_x = np.array([0.0, 1.0, 0.5, 1.5])
                >>> node_y = np.array([0.0, 0.0, 1.0, 1.0])
                >>> faces = np.array([[0, 1, 2], [1, 3, 2]])
                >>> mg = MeshGlyph(node_x, node_y, faces)
                >>> fig, ax = mg.plot(
                ...     np.array([0.0, 1.0, 2.0, 3.0]),
                ...     location="node",
                ...     filled=False,
                ... )

                ```
            - Label the line tricontours inline (`labels=True`); the
                `Text` artists are exposed on `glyph.contour_labels`:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> node_x = np.array([0.0, 1.0, 0.5, 1.5])
                >>> node_y = np.array([0.0, 0.0, 1.0, 1.0])
                >>> faces = np.array([[0, 1, 2], [1, 3, 2]])
                >>> mg = MeshGlyph(node_x, node_y, faces)
                >>> fig, ax = mg.plot(
                ...     np.array([0.0, 1.0, 2.0, 3.0]),
                ...     location="node",
                ...     filled=False,
                ...     contour=Contour(labels=True, label_kw={"fmt": "%.1f"}),
                ... )
                >>> isinstance(mg.contour_labels, list)
                True

                ```
            - Plot with power color scale:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> node_x = np.array([0.0, 1.0, 0.5, 1.5])
                >>> node_y = np.array([0.0, 0.0, 1.0, 1.0])
                >>> faces = np.array([[0, 1, 2], [1, 3, 2]])
                >>> mg = MeshGlyph(node_x, node_y, faces)
                >>> fig, ax = mg.plot(
                ...     np.array([1.0, 2.0]),
                ...     color=ColorScaling.power(gamma=0.5),
                ...     cmap="coolwarm",
                ... )

                ```
        """
        self._validate_location_and_data(data, location)

        if np.all(np.isnan(data)):
            raise ValueError("data is entirely NaN, cannot determine color range.")

        self._default_options = MESH_DEFAULT_OPTIONS.copy()

        render_kwargs: dict[str, Any] = {}
        option_kwargs: dict[str, Any] = {}
        for key, val in kwargs.items():
            if key in self.default_options:
                option_kwargs[key] = val
            else:
                render_kwargs[key] = val
        self._merge_kwargs(option_kwargs)
        self._merge_group_params(color, contour, data_style)
        _warn_deprecated_cbar_kwargs(kwargs)
        resolved_colorbar = (
            _resolve_colorbar(colorbar) if isinstance(colorbar, ColorBar) else {}
        )
        self.default_options.update(resolved_colorbar)
        if colorbar is not False:
            colorbar = True

        # `style`/`hillshade` now arrive via the `data_style` group object;
        # detect whether this call provided each so the sticky-state logic
        # (a preset persists across later plain plots) still applies.
        ds_opts = data_style.to_options() if data_style is not None else {}
        if "hillshade" not in ds_opts:
            self.default_options["hillshade"] = self._construct_hillshade
        if "style" in ds_opts:
            new_style = self.default_options["style"]
            if new_style is not None:
                try:
                    resolve_single_layer_style(new_style)
                except ValueError:
                    self.default_options["style"] = self._style_state
                    raise
            self._style_state = new_style
        else:
            self.default_options["style"] = self._style_state
        if "projection" in option_kwargs:
            self._projection_state = self.default_options["projection"]
        else:
            self.default_options["projection"] = self._projection_state

        self._last_data = (
            np.ma.copy(data) if np.ma.isMaskedArray(data) else np.array(data, copy=True)
        )
        self._last_location = location

        if "vmin" not in option_kwargs:
            self.default_options["vmin"] = float(np.nanmin(data))
        if "vmax" not in option_kwargs:
            self.default_options["vmax"] = float(np.nanmax(data))
        self._vmin = self.default_options["vmin"]
        self._vmax = self.default_options["vmax"]

        if (
            "ticks_spacing" not in option_kwargs
            and "ticks_spacing" not in resolved_colorbar
        ):
            spacing = (self._vmax - self._vmin) / 10
            self.default_options["ticks_spacing"] = max(spacing, 1e-10)
        self.ticks_spacing = self.default_options["ticks_spacing"]

        if title is not None:
            self.default_options["title"] = title

        if ax is not None:
            self.ax = ax
            self.fig = ax.get_figure()
        elif self.fig is None:
            self.fig, self.ax = self.create_figure_axes()

        ticks = self.get_ticks()
        norm, cbar_kw = self._create_norm_and_cbar_kw(ticks)

        style = self.default_options.get("style")
        style_legend = None
        if style is not None:
            _, cfg = resolve_single_layer_style(style)
            data_f = np.asarray(data, dtype=float)
            categories = cfg.get("categories")
            if categories is not None:
                cats = sorted(categories, key=lambda c: c[0])
                cat_values = np.array([float(c[0]) for c in cats])
                cat_colors = [c[1] for c in cats]
                cat_labels = [c[2] for c in cats]
                self.default_options["cmap"] = ListedColormap(cat_colors)
                norm = BoundaryNorm(
                    category_boundaries(list(cat_values)), len(cat_colors)
                )
                data = np.where(np.isin(data_f, cat_values), data_f, np.nan)
                colorbar = False
                self.default_options["hillshade"] = False
                style_legend = (cat_colors, cat_labels, cfg["label"])
                if location == "node":
                    warnings.warn(
                        "a categorical data-style preset with location='node' "
                        "interpolates discrete class codes via tricontourf; use "
                        "location='face' for correct per-cell class colours.",
                        stacklevel=2,
                    )
            else:
                self.default_options["cmap"] = cfg["cmap"]
                norm, _, _ = resolve_style_norm(data_f, cfg)
                cbar_kw.pop("ticks", None)

        self.contour_labels = None

        hillshade = resolve_hillshade(self.default_options.get("hillshade"))
        if hillshade is not None:
            if location != "node":
                raise ValueError(
                    "hillshade needs node-centered elevation; pass location='node'"
                )
            _clear_prior_render_artists(self.ax)
            self.im = None
            self._cbar = None
            self._apply_projection()
            tpc = self._render_shaded_relief(
                self.ax, data, edgecolor, norm, hillshade, **render_kwargs
            )
        else:
            _clear_prior_render_artists(self.ax)
            self.im = None
            self._cbar = None
            self._apply_projection()
            tpc = self._render_mesh(
                self.ax,
                data,
                location,
                edgecolor=edgecolor,
                norm=norm,
                filled=filled,
                **render_kwargs,
            )
        self.im = tpc

        if (
            location == "node"
            and not filled
            and hillshade is None
            and self.default_options.get("labels")
        ):
            label_kw = {
                "inline": True,
                "fontsize": 8,
                "fmt": "%g",
                **(self.default_options.get("label_kw") or {}),
            }
            self.contour_labels = self.ax.clabel(tpc, **label_kw)

        if colorbar:
            self._cbar = self.create_color_bar(self.ax, tpc, cbar_kw)

        if style_legend is not None:
            cat_colors, cat_labels, cat_title = style_legend
            disjoint_legend(
                self.ax, cat_colors, cat_labels, title=cat_title, loc="upper right"
            )

        if self.default_options["title"]:
            self.ax.set_title(
                self.default_options["title"],
                fontsize=self.default_options["title_size"],
            )
        self.ax.set_aspect("equal")

        _mark_render_artists(self.ax, self._cbar, self.im)
        return self.fig, self.ax

    def animate(
        self,
        data: np.ndarray | list[np.ndarray],
        time: list[Any],
        location: str = "face",
        edgecolor: str = "none",
        interval: int = 200,
        text_loc: list | None = None,
        colorbar: bool | ColorBar | None = None,
        color: ColorScaling | None = None,
        contour: Contour | None = None,
        data_style: DataStyle | None = None,
        **kwargs: Any,
    ) -> FuncAnimation:
        """Create an animation from time-varying mesh data.

        Iterates over the first dimension of `data` (or elements of a
        list), rendering each frame on the fixed mesh topology.

        Args:
            data: Sequence of data arrays. If a 2D ndarray of shape
                `(n_frames, n_elements)`, each row is one frame.
                If a list, each element is a 1D array for one frame.
            time: Labels for each frame (timestamps, strings, etc.).
                Length must match the number of frames.
            location: `"face"` or `"node"`. Default is `"face"`.
            edgecolor: Edge color for face rendering. Default is
                `"none"`.
            interval: Milliseconds between frames. Default is 200.
            text_loc: `[x, y]` position for the time label text.
                Default is `[0.1, 0.2]`.
            colorbar: Typed `ColorBar` spec (placement / caption / sizing) for
                the animation's colorbar, or `True`/`None` to draw a default one;
                `False` suppresses it. Default `None` (draw).
            **kwargs: Override any key in `default_options` (cmap,
                vmin, vmax, color_scale, gamma, midpoint, figsize,
                title, etc.). The loose `ticks_spacing` / `cbar_*` keys
                still work but are deprecated -- pass
                `colorbar=ColorBar(...)` instead.

        Returns:
            FuncAnimation: The animation object. Use
                `save_animation()` to export.

        Raises:
            ValueError: If `data` frames don't match mesh topology
                or `time` length doesn't match frame count.

        Notes:
            An animation draws no inline contour labels, so this clears
            `contour_labels` back to `None`; any label artists left by a
            previous `plot(filled=False, labels=True)` call do not leak
            into the animation state.

        Examples:
            - Animate face data over 3 time steps:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> node_x = np.array([0.0, 1.0, 0.5, 1.5])
                >>> node_y = np.array([0.0, 0.0, 1.0, 1.0])
                >>> faces = np.array([[0, 1, 2], [1, 3, 2]])
                >>> mg = MeshGlyph(node_x, node_y, faces)
                >>> frames = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                >>> anim = mg.animate(frames, time=["t0", "t1", "t2"])

                ```
        """
        if text_loc is None:
            text_loc = [0.1, 0.2]

        if isinstance(data, np.ndarray) and data.ndim == 2:
            frames = [data[i] for i in range(data.shape[0])]
        else:
            frames = list(data)

        n_frames = len(frames)
        if len(time) != n_frames:
            raise ValueError(
                f"time length ({len(time)}) does not match frame count ({n_frames})."
            )
        expected = self.n_faces if location == "face" else self.n_nodes
        for i, frame in enumerate(frames):
            if len(frame) != expected:
                raise ValueError(
                    f"Frame {i}: data length ({len(frame)}) does not "
                    f"match n_{location}s ({expected})."
                )

        self._default_options = MESH_DEFAULT_OPTIONS.copy()
        self._merge_kwargs(kwargs)
        self._merge_group_params(color, contour, data_style)
        _warn_deprecated_cbar_kwargs(kwargs)
        resolved_colorbar = (
            _resolve_colorbar(colorbar) if isinstance(colorbar, ColorBar) else {}
        )
        self.default_options.update(resolved_colorbar)

        if "vmin" not in kwargs:
            global_min = min(float(np.nanmin(f)) for f in frames)
            self.default_options["vmin"] = global_min
        if "vmax" not in kwargs:
            global_max = max(float(np.nanmax(f)) for f in frames)
            self.default_options["vmax"] = global_max
        self._vmin = self.default_options["vmin"]
        self._vmax = self.default_options["vmax"]

        if "ticks_spacing" not in kwargs and "ticks_spacing" not in resolved_colorbar:
            spacing = (self._vmax - self._vmin) / 10
            self.default_options["ticks_spacing"] = max(spacing, 1e-10)
        self.ticks_spacing = self.default_options["ticks_spacing"]

        if self.fig is None:
            self.fig, self.ax = self.create_figure_axes()
        fig, ax = self.fig, self.ax

        ticks = self.get_ticks()
        norm, cbar_kw = self._create_norm_and_cbar_kw(ticks)

        self.contour_labels = None

        _clear_prior_render_artists(ax)
        self.im = None
        self._cbar = None

        tpc = self._render_mesh(
            ax,
            frames[0],
            location,
            edgecolor=edgecolor,
            norm=norm,
        )
        self.im = tpc
        if colorbar is not False:
            self._cbar = self.create_color_bar(ax, tpc, cbar_kw)

        if self.default_options["title"]:
            ax.set_title(
                self.default_options["title"],
                fontsize=self.default_options["title_size"],
            )
        ax.set_aspect("equal")

        day_text = ax.text(
            text_loc[0],
            text_loc[1],
            " ",
            fontsize=self.default_options["cbar_label_size"],
            transform=ax.transAxes,
        )
        self._day_text = day_text

        current_mappable = [tpc]
        _mark_render_artists(ax, self._cbar, self.im, self._day_text)

        def _update(i):
            """Update the plot for frame i."""
            prev = current_mappable[0]
            if hasattr(prev, "collections"):
                for coll in prev.collections:
                    coll.remove()
            elif hasattr(prev, "remove"):
                prev.remove()
            current_mappable[0] = self._render_mesh(
                ax,
                frames[i],
                location,
                edgecolor=edgecolor,
                norm=norm,
            )
            day_text.set_text(str(time[i]))
            self.im = current_mappable[0]
            _mark_render_artists(ax, self._cbar, self.im, self._day_text)

        plt.tight_layout()
        anim = FuncAnimation(
            fig,
            _update,
            frames=n_frames,
            interval=interval,
            blit=False,
        )
        self._anim = anim
        return anim

    def plot_outline(
        self,
        ax: Any = None,
        color: str = "black",
        linewidth: float = 0.3,
        figsize: tuple[int, int] = (10, 8),
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot mesh edges as a wireframe.

        Uses `matplotlib.collections.LineCollection` for efficient
        rendering of thousands of edges.

        Args:
            ax: Axes to plot on. If None, uses stored axes or creates
                new.
            color: Edge color. Default is `"black"`.
            linewidth: Edge line width. Default is `0.3`.
            figsize: Figure size in inches. Default is `(10, 8)`.
            **kwargs: Additional keyword arguments passed to
                `LineCollection`.

        Returns:
            tuple[Figure, Axes]: The matplotlib Figure and Axes objects.
                When `ax` is None, a new figure is created. Call
                `plt.close(fig)` after saving to avoid memory leaks
                in batch processing.

        Notes:
            An outline carries no scalar mapping, so this resets `self.im`
            to None (clearing any colour-mapped artist left by a prior
            `plot()` call).

        Examples:
            - Render a triangular mesh wireframe:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> mg = MeshGlyph(
                ...     np.array([0.0, 1.0, 0.5]),
                ...     np.array([0.0, 0.0, 1.0]),
                ...     np.array([[0, 1, 2]]),
                ... )
                >>> fig, ax = mg.plot_outline(color="blue")

                ```
        """
        if ax is not None:
            self.ax = ax
            self.fig = ax.get_figure()
        elif self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)

        _clear_prior_render_artists(self.ax)
        self.im = None
        self._cbar = None

        segments = self._build_edge_segments()

        lc = mcoll.LineCollection(
            list(segments), colors=color, linewidths=linewidth, **kwargs
        )
        self.ax.add_collection(lc)
        self.ax.autoscale()
        self.ax.set_aspect("equal")

        _mark_render_artists(self.ax, lc)

        return self.fig, self.ax

    def _build_edge_segments(self) -> np.ndarray:
        """Build line segments for wireframe rendering.

        Uses edge_node_connectivity if available, otherwise derives the
        unique polygon edges from face_node_connectivity by walking each
        face boundary (with wrap-around) and deduplicating undirected
        edges via a sort. Both paths are fully vectorized.

        Returns:
            np.ndarray: Array of shape (n_segments, 2, 2) where each
                segment is `[[x1, y1], [x2, y2]]`. Returns an empty
                array with shape (0, 2, 2) if no edges can be derived.

        Examples:
            - A single triangle yields its three boundary segments, and the
                first segment connects the two lowest-indexed nodes:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> mg = MeshGlyph(
                ...     np.array([0.0, 1.0, 0.5]),
                ...     np.array([0.0, 0.0, 1.0]),
                ...     np.array([[0, 1, 2]]),
                ... )
                >>> segs = mg._build_edge_segments()
                >>> segs.shape
                (3, 2, 2)
                >>> segs[0]
                array([[0., 0.],
                       [1., 0.]])

                ```
            - An edge shared by two faces is emitted only once, so two
                triangles sharing one edge produce five segments, not six:
                ```python
                >>> import numpy as np
                >>> from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
                >>> mg = MeshGlyph(
                ...     np.array([0.0, 1.0, 0.5, 1.5]),
                ...     np.array([0.0, 0.0, 1.0, 1.0]),
                ...     np.array([[0, 1, 2], [1, 3, 2]]),
                ... )
                >>> mg._build_edge_segments().shape
                (5, 2, 2)

                ```
        """
        if self._edge_nodes is not None:
            n1 = self._edge_nodes[:, 0]
            n2 = self._edge_nodes[:, 1]
            starts = np.column_stack([self._node_x[n1], self._node_y[n1]])
            ends = np.column_stack([self._node_x[n2], self._node_y[n2]])
            return np.stack([starts, ends], axis=1)

        counts = self.nodes_per_face
        flat_nodes = self._face_nodes[self._face_nodes != self._fill_value]
        if flat_nodes.size == 0:
            return np.empty((0, 2, 2), dtype=np.float64)

        face_start = np.cumsum(counts) - counts
        next_pos = np.arange(flat_nodes.size, dtype=np.intp) + 1
        nonempty = counts >= 1
        last_pos = face_start[nonempty] + counts[nonempty] - 1
        next_pos[last_pos] = face_start[nonempty]

        a = flat_nodes
        b = flat_nodes[next_pos]
        n_nodes = np.int64(self.n_nodes)
        lo = np.minimum(a, b).astype(np.int64)
        hi = np.maximum(a, b).astype(np.int64)
        sorted_keys = np.sort(lo * n_nodes + hi)
        keys = sorted_keys[
            np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1]))
        ]
        n1, n2 = keys // n_nodes, keys % n_nodes
        starts = np.column_stack([self._node_x[n1], self._node_y[n1]])
        ends = np.column_stack([self._node_x[n2], self._node_y[n2]])
        return np.stack([starts, ends], axis=1)
