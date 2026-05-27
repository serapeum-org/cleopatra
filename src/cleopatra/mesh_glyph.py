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
        >>> from cleopatra.mesh_glyph import MeshGlyph
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

from typing import Any

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.animation import FuncAnimation

from cleopatra.glyph import Glyph
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS

MESH_DEFAULT_OPTIONS = {
    "vmin": None,
    "vmax": None,
}
MESH_DEFAULT_OPTIONS = STYLE_DEFAULTS | MESH_DEFAULT_OPTIONS


class MeshGlyph(Glyph):
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

    Examples:
        - Create a MeshGlyph and inspect its topology:
            ```python
            >>> import numpy as np
            >>> from cleopatra.mesh_glyph import MeshGlyph
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
        self._cbar = None
        #: Colour-mapped artist from the most recent `plot` call (the
        #: `tripcolor`/`tricontour(f)` mappable); `None` before first render.
        self.im = None

    @property
    def node_x(self) -> np.ndarray:
        """Node x-coordinates."""
        return self._node_x

    @property
    def node_y(self) -> np.ndarray:
        """Node y-coordinates."""
        return self._node_y

    @property
    def n_faces(self) -> int:
        """Number of faces in the mesh."""
        return self._face_nodes.shape[0]

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the mesh."""
        return len(self._node_x)

    @property
    def n_edges(self) -> int:
        """Number of edges (0 if edge connectivity not provided)."""
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
                >>> from cleopatra.mesh_glyph import MeshGlyph
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
                >>> from cleopatra.mesh_glyph import MeshGlyph
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
                >>> from cleopatra.mesh_glyph import MeshGlyph
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
                >>> from cleopatra.mesh_glyph import MeshGlyph
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
                >>> from cleopatra.mesh_glyph import MeshGlyph
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

        # Fast path only when the connectivity is already a clean (n, 3)
        # array; a wider padded array where every face happens to be a
        # triangle still needs fill values stripped below.
        if self._face_nodes.shape[1] == 3 and np.all(counts == 3):
            self._cached_tri_array = self._face_nodes.copy()
            return self._cached_tri_array

        # Vectorized fan decomposition for mixed-element meshes. Valid node
        # indices are compacted in face/row order, so face i occupies the
        # slice flat_nodes[face_start[i] : face_start[i] + counts[i]].
        # _face_nodes is already np.intp (set in the constructor), so boolean
        # masking preserves the dtype without an explicit cast.
        flat_nodes = self._face_nodes[self._face_nodes != self._fill_value]
        face_start = np.cumsum(counts) - counts

        # A face with c valid nodes produces (c - 2) fan triangles.
        valid = counts >= 3
        base = np.repeat(face_start[valid], counts[valid] - 2)
        # Local triangle index t in [0, c-3] for each output triangle, built
        # as a concatenated per-face arange without a Python loop.
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
                >>> from cleopatra.mesh_glyph import MeshGlyph
                >>> MeshGlyph._grouped_arange(np.array([2, 3, 1]))
                array([0, 1, 0, 1, 2, 0])

                ```
            - Zero-size groups contribute nothing and do not shift the
                counter of later groups:
                ```python
                >>> import numpy as np
                >>> from cleopatra.mesh_glyph import MeshGlyph
                >>> MeshGlyph._grouped_arange(np.array([2, 0, 3]))
                array([0, 1, 0, 1, 2])

                ```
        """
        sizes = np.asarray(sizes, dtype=np.intp)
        total = int(sizes.sum())
        if total == 0:
            return np.empty(0, dtype=np.intp)
        # Subtract each element's group-start offset from a global arange so
        # the counter restarts at 0 within every group. np.repeat emits
        # nothing for zero-size groups, so empty groups are handled naturally.
        group_start = np.cumsum(sizes) - sizes
        return np.arange(total, dtype=np.intp) - np.repeat(group_start, sizes)

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
                >>> from cleopatra.mesh_glyph import MeshGlyph
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
                f"data length ({len(data)}) does not match "
                f"n_{location}s ({expected})."
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
        cmap = self.default_options["cmap"]
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

    def plot(
        self,
        data: np.ndarray,
        location: str = "face",
        ax: Any = None,
        edgecolor: str = "none",
        colorbar: bool = True,
        title: str | None = None,
        filled: bool = True,
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
            colorbar: Whether to add a colorbar. Default is True.
            title: Plot title. Overrides `default_options["title"]`.
            filled: For node data, draw filled contours (`tricontourf`,
                the default) or line contours (`tricontour`) when
                `False`. Ignored for face data. Default is True.
            **kwargs: Override any key in `default_options` (cmap,
                vmin, vmax, color_scale, gamma, midpoint, bounds,
                ticks_spacing, cbar_orientation, cbar_label, figsize,
                etc.) or pass extra rendering kwargs (levels for
                tricontourf / tricontour).

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
                >>> from cleopatra.mesh_glyph import MeshGlyph
                >>> node_x = np.array([0.0, 1.0, 0.5, 1.5])
                >>> node_y = np.array([0.0, 0.0, 1.0, 1.0])
                >>> faces = np.array([[0, 1, 2], [1, 3, 2]])
                >>> mg = MeshGlyph(node_x, node_y, faces)
                >>> fig, ax = mg.plot(np.array([1.0, 2.0]))

                ```
            - Plot node-centered data:
                ```python
                >>> import numpy as np
                >>> from cleopatra.mesh_glyph import MeshGlyph
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
                >>> from cleopatra.mesh_glyph import MeshGlyph
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
            - Plot with power color scale:
                ```python
                >>> import numpy as np
                >>> from cleopatra.mesh_glyph import MeshGlyph
                >>> node_x = np.array([0.0, 1.0, 0.5, 1.5])
                >>> node_y = np.array([0.0, 0.0, 1.0, 1.0])
                >>> faces = np.array([[0, 1, 2], [1, 3, 2]])
                >>> mg = MeshGlyph(node_x, node_y, faces)
                >>> fig, ax = mg.plot(
                ...     np.array([1.0, 2.0]),
                ...     color_scale="power",
                ...     gamma=0.5,
                ...     cmap="coolwarm",
                ... )

                ```
        """
        self._validate_location_and_data(data, location)

        # Guard against all-NaN data.
        if np.all(np.isnan(data)):
            raise ValueError(
                "data is entirely NaN, cannot determine color range."
            )

        # Reset default_options to a fresh copy so repeated plot() calls
        # on the same instance don't accumulate stale overrides.
        self._default_options = MESH_DEFAULT_OPTIONS.copy()

        # Separate rendering kwargs (e.g. levels) from default_options kwargs.
        render_kwargs: dict[str, Any] = {}
        option_kwargs: dict[str, Any] = {}
        for key, val in kwargs.items():
            if key in self.default_options:
                option_kwargs[key] = val
            else:
                render_kwargs[key] = val
        self._merge_kwargs(option_kwargs)

        # Recompute vmin/vmax from data unless user explicitly passed them.
        if "vmin" not in option_kwargs:
            self.default_options["vmin"] = float(np.nanmin(data))
        if "vmax" not in option_kwargs:
            self.default_options["vmax"] = float(np.nanmax(data))
        self._vmin = self.default_options["vmin"]
        self._vmax = self.default_options["vmax"]

        # Compute ticks_spacing and write it to default_options for get_ticks().
        if "ticks_spacing" not in option_kwargs:
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

        tpc = self._render_mesh(
            self.ax,
            data,
            location,
            edgecolor=edgecolor,
            norm=norm,
            filled=filled,
            **render_kwargs,
        )
        # Expose the colour-mapped artist (the `PolyCollection` from
        # `tripcolor`, or the `TriContourSet` from `tricontour(f)`) so a
        # caller can attach a colorbar/register the layer without scraping
        # `ax.collections` (mirrors `ArrayGlyph.im` / the other glyphs).
        self.im = tpc

        # Remove previous colorbar before adding a new one.
        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None

        if colorbar:
            self._cbar = self.create_color_bar(self.ax, tpc, cbar_kw)

        if self.default_options["title"]:
            self.ax.set_title(
                self.default_options["title"],
                fontsize=self.default_options["title_size"],
            )
        self.ax.set_aspect("equal")

        return self.fig, self.ax

    def animate(
        self,
        data: np.ndarray | list[np.ndarray],
        time: list[Any],
        location: str = "face",
        edgecolor: str = "none",
        interval: int = 200,
        text_loc: list | None = None,
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
            **kwargs: Override any key in `default_options` (cmap,
                vmin, vmax, color_scale, gamma, midpoint,
                ticks_spacing, cbar_label, cbar_orientation, figsize,
                title, etc.).

        Returns:
            FuncAnimation: The animation object. Use
                `save_animation()` to export.

        Raises:
            ValueError: If `data` frames don't match mesh topology
                or `time` length doesn't match frame count.

        Examples:
            - Animate face data over 3 time steps:
                ```python
                >>> import numpy as np
                >>> from cleopatra.mesh_glyph import MeshGlyph
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

        # Normalize data to a list of 1D arrays.
        if isinstance(data, np.ndarray) and data.ndim == 2:
            frames = [data[i] for i in range(data.shape[0])]
        else:
            frames = list(data)

        n_frames = len(frames)
        if len(time) != n_frames:
            raise ValueError(
                f"time length ({len(time)}) does not match "
                f"frame count ({n_frames})."
            )
        expected = self.n_faces if location == "face" else self.n_nodes
        for i, frame in enumerate(frames):
            if len(frame) != expected:
                raise ValueError(
                    f"Frame {i}: data length ({len(frame)}) does not "
                    f"match n_{location}s ({expected})."
                )

        # Reset default_options to a fresh copy.
        self._default_options = MESH_DEFAULT_OPTIONS.copy()
        self._merge_kwargs(kwargs)

        # Compute global vmin/vmax across all frames unless user set them.
        if "vmin" not in kwargs:
            global_min = min(float(np.nanmin(f)) for f in frames)
            self.default_options["vmin"] = global_min
        if "vmax" not in kwargs:
            global_max = max(float(np.nanmax(f)) for f in frames)
            self.default_options["vmax"] = global_max
        self._vmin = self.default_options["vmin"]
        self._vmax = self.default_options["vmax"]

        # Compute ticks_spacing and write it to default_options for get_ticks().
        if "ticks_spacing" not in kwargs:
            spacing = (self._vmax - self._vmin) / 10
            self.default_options["ticks_spacing"] = max(spacing, 1e-10)
        self.ticks_spacing = self.default_options["ticks_spacing"]

        if self.fig is None:
            self.fig, self.ax = self.create_figure_axes()
        fig, ax = self.fig, self.ax

        ticks = self.get_ticks()
        norm, cbar_kw = self._create_norm_and_cbar_kw(ticks)

        # Render the first frame.
        tpc = self._render_mesh(
            ax,
            frames[0],
            location,
            edgecolor=edgecolor,
            norm=norm,
        )
        self.create_color_bar(ax, tpc, cbar_kw)

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

        # Track the current mappable so we can remove it cleanly.
        current_mappable = [tpc]

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

        Examples:
            - Render a triangular mesh wireframe:
                ```python
                >>> import numpy as np
                >>> from cleopatra.mesh_glyph import MeshGlyph
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

        segments = self._build_edge_segments()

        lc = mcoll.LineCollection(
            segments, colors=color, linewidths=linewidth, **kwargs
        )
        self.ax.add_collection(lc)
        self.ax.autoscale()
        self.ax.set_aspect("equal")

        # An outline carries no scalar mapping, so clear any colour-mapped
        # artist left on `self.im` by a previous `plot()` call.
        self.im = None

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
                >>> from cleopatra.mesh_glyph import MeshGlyph
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
                >>> from cleopatra.mesh_glyph import MeshGlyph
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
        # Compacted valid node indices in face/row order; face i occupies the
        # slice flat_nodes[face_start[i] : face_start[i] + counts[i]].
        # _face_nodes is already np.intp (set in the constructor), so boolean
        # masking preserves the dtype without an explicit cast.
        flat_nodes = self._face_nodes[self._face_nodes != self._fill_value]
        if flat_nodes.size == 0:
            return np.empty((0, 2, 2), dtype=np.float64)

        face_start = np.cumsum(counts) - counts
        # Each valid node starts one polygon edge to the next node within the
        # same face, wrapping from the last node back to the first.
        next_pos = np.arange(flat_nodes.size, dtype=np.intp) + 1
        nonempty = counts >= 1
        last_pos = face_start[nonempty] + counts[nonempty] - 1
        next_pos[last_pos] = face_start[nonempty]

        a = flat_nodes
        b = flat_nodes[next_pos]
        # Undirected edges: order each endpoint pair, then drop duplicates by
        # encoding the pair as a single key (lo * n_nodes + hi) and applying a
        # 1-D sort + adjacent-diff dedup (cheaper than a row-wise lexsort). The
        # key stays exact in int64 for any realistic mesh; it would only
        # overflow when n_nodes exceeds ~3e9.
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
