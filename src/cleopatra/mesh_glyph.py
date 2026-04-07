"""Unstructured mesh visualization.

Provides `MeshGlyph` for plotting UGRID-style unstructured mesh data
using matplotlib triangulation (tripcolor, tricontourf) and wireframe
rendering via LineCollection. Designed for use with pyramids' Mesh2d
objects but accepts raw arrays for standalone use.

Examples
--------
Plot face-centered data on a triangular mesh:

    >>> import numpy as np
    >>> from cleopatra.mesh_glyph import MeshGlyph
    >>> node_x = np.array([0.0, 1.0, 0.5, 1.5])
    >>> node_y = np.array([0.0, 0.0, 1.0, 1.0])
    >>> face_nodes = np.array([[0, 1, 2], [1, 3, 2]])
    >>> face_data = np.array([10.0, 20.0])
    >>> mg = MeshGlyph(node_x, node_y, face_nodes)
    >>> fig, ax = mg.plot(face_data, location="face", title="Water Level")

Plot a wireframe outline:

    >>> fig, ax = mg.plot_outline(color="blue", linewidth=0.5)
"""

from __future__ import annotations

from typing import Any

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


class MeshGlyph:
    """Visualization class for unstructured mesh data.

    Wraps matplotlib's triangulation-based rendering to plot data on
    UGRID-style unstructured meshes (triangles, quads, mixed polygons).
    Handles fan triangulation for mixed meshes and maps face-centered
    values to individual triangles.

    Parameters
    ----------
    node_x : np.ndarray
        1D array of node x-coordinates (n_nodes,).
    node_y : np.ndarray
        1D array of node y-coordinates (n_nodes,).
    face_node_connectivity : np.ndarray
        2D array of node indices per face (n_faces, max_nodes_per_face).
        Use ``fill_value`` to pad rows for faces with fewer nodes.
    fill_value : int, optional
        Padding value in ``face_node_connectivity`` for mixed meshes.
        Default is -1.
    edge_node_connectivity : np.ndarray or None, optional
        2D array of node indices per edge (n_edges, 2). If provided,
        used for efficient wireframe rendering. If None, edges are
        derived from face connectivity. Default is None.

    Attributes
    ----------
    node_x : np.ndarray
        Node x-coordinates.
    node_y : np.ndarray
        Node y-coordinates.
    n_faces : int
        Number of faces in the mesh.
    n_nodes : int
        Number of nodes in the mesh.
    n_edges : int
        Number of edges (0 if edge connectivity not provided).

    Examples
    --------
    Create a MeshGlyph from a simple triangular mesh:

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
    """

    def __init__(
        self,
        node_x: np.ndarray,
        node_y: np.ndarray,
        face_node_connectivity: np.ndarray,
        fill_value: int = -1,
        edge_node_connectivity: np.ndarray | None = None,
    ):
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
        result = self._edge_nodes.shape[0] if self._edge_nodes is not None else 0
        return result

    @property
    def nodes_per_face(self) -> np.ndarray:
        """Number of valid nodes per face (excluding fill values).

        Returns
        -------
        np.ndarray
            1D integer array of length n_faces.
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

        Returns
        -------
        matplotlib.tri.Triangulation
            Triangulation ready for tripcolor/tricontourf.

        Raises
        ------
        ValueError
            If no faces have 3 or more valid nodes.

        Examples
        --------
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
        """
        if self._cached_triangulation is None:
            tri_array = self._fan_triangles()
            self._cached_triangulation = mtri.Triangulation(
                self._node_x, self._node_y, tri_array
            )
        return self._cached_triangulation

    def _fan_triangles(self) -> np.ndarray:
        """Compute fan triangulation for mixed-element meshes.

        Returns
        -------
        np.ndarray
            (n_triangles, 3) array of node indices.

        Raises
        ------
        ValueError
            If no valid triangles can be formed.
        """
        if self._cached_tri_array is not None:
            return self._cached_tri_array

        triangles: list[list[int]] = []

        for i in range(self.n_faces):
            row = self._face_nodes[i]
            nodes = row[row != self._fill_value]
            n = len(nodes)
            if n < 3:
                continue
            for j in range(1, n - 1):
                triangles.append([int(nodes[0]), int(nodes[j]), int(nodes[j + 1])])

        if not triangles:
            raise ValueError("Cannot create triangulation: no faces with 3+ nodes.")

        self._cached_tri_array = np.array(triangles, dtype=np.intp)
        return self._cached_tri_array

    def _map_face_to_triangle_values(self, face_values: np.ndarray) -> np.ndarray:
        """Map per-face values to per-triangle values.

        Each original face may produce multiple triangles via fan
        decomposition. All triangles from the same face receive
        the same data value.

        Parameters
        ----------
        face_values : np.ndarray
            1D array of values, one per face.

        Returns
        -------
        np.ndarray
            1D array of values, one per triangle.

        Examples
        --------
            >>> import numpy as np
            >>> from cleopatra.mesh_glyph import MeshGlyph
            >>> mg = MeshGlyph(
            ...     np.array([0.0, 1.0, 1.0, 0.0]),
            ...     np.array([0.0, 0.0, 1.0, 1.0]),
            ...     np.array([[0, 1, 2, 3]]),
            ... )
            >>> mg._map_face_to_triangle_values(np.array([42.0]))
            array([42., 42.])
        """
        counts = self.nodes_per_face
        valid = counts >= 3
        n_triangles = int(np.sum(counts[valid] - 2))
        tri_values = np.empty(n_triangles)

        tri_idx = 0
        for face_idx in range(self.n_faces):
            if counts[face_idx] < 3:
                continue
            n_tris = counts[face_idx] - 2
            tri_values[tri_idx : tri_idx + n_tris] = face_values[face_idx]
            tri_idx += n_tris

        return tri_values

    def plot(
        self,
        data: np.ndarray,
        location: str = "face",
        ax: Any = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        edgecolor: str = "none",
        colorbar: bool = True,
        title: str | None = None,
        figsize: tuple[int, int] = (10, 8),
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot mesh data using matplotlib triangulation.

        For face-centered data, uses ``tripcolor`` where each triangle
        is colored by the value of its parent face. For node-centered
        data, uses ``tricontourf`` for smooth interpolated contours.

        Parameters
        ----------
        data : np.ndarray
            1D data array. Length must match face count (location="face")
            or node count (location="node").
        location : str, optional
            Mesh element location: ``"face"`` or ``"node"``.
            Default is ``"face"``.
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot on. If None, a new figure is created.
        cmap : str, optional
            Matplotlib colormap name. Default is ``"viridis"``.
        vmin : float or None, optional
            Minimum color scale value.
        vmax : float or None, optional
            Maximum color scale value.
        edgecolor : str, optional
            Edge color for face rendering. Default is ``"none"``.
        colorbar : bool, optional
            Whether to add a colorbar. Default is True.
        title : str or None, optional
            Plot title.
        figsize : tuple[int, int], optional
            Figure size in inches. Default is ``(10, 8)``.
        **kwargs
            Additional keyword arguments passed to ``tripcolor`` or
            ``tricontourf``. Do not pass ``cmap``, ``levels``,
            ``vmin``, or ``vmax`` here — use the dedicated parameters.

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib Figure and Axes objects.

        Raises
        ------
        ValueError
            If ``location`` is not ``"face"`` or ``"node"``, or if
            ``data`` length does not match the expected mesh dimension.

        Examples
        --------
        Plot face-centered data:

            >>> import numpy as np
            >>> from cleopatra.mesh_glyph import MeshGlyph
            >>> node_x = np.array([0.0, 1.0, 0.5, 1.5])
            >>> node_y = np.array([0.0, 0.0, 1.0, 1.0])
            >>> faces = np.array([[0, 1, 2], [1, 3, 2]])
            >>> mg = MeshGlyph(node_x, node_y, faces)
            >>> fig, ax = mg.plot(np.array([1.0, 2.0]))

        Plot node-centered data:

            >>> fig, ax = mg.plot(
            ...     np.array([0.0, 1.0, 2.0, 3.0]),
            ...     location="node",
            ... )
        """
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

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.get_figure()

        tri = self.triangulation

        if location == "face":
            tri_values = self._map_face_to_triangle_values(data)
            tpc = ax.tripcolor(
                tri,
                facecolors=tri_values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                edgecolors=edgecolor,
                **kwargs,
            )
        else:
            contour_kw: dict[str, Any] = {"cmap": cmap, "levels": 20}
            if vmin is not None:
                contour_kw["vmin"] = vmin
            if vmax is not None:
                contour_kw["vmax"] = vmax
            tpc = ax.tricontourf(tri, data, **contour_kw, **kwargs)

        if colorbar:
            plt.colorbar(tpc, ax=ax)
        if title:
            ax.set_title(title)
        ax.set_aspect("equal")

        result = (fig, ax)
        return result

    def plot_outline(
        self,
        ax: Any = None,
        color: str = "black",
        linewidth: float = 0.3,
        figsize: tuple[int, int] = (10, 8),
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot mesh edges as a wireframe.

        Uses ``matplotlib.collections.LineCollection`` for efficient
        rendering of thousands of edges.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot on. If None, a new figure is created.
        color : str, optional
            Edge color. Default is ``"black"``.
        linewidth : float, optional
            Edge line width. Default is ``0.3``.
        figsize : tuple[int, int], optional
            Figure size in inches. Default is ``(10, 8)``.
        **kwargs
            Additional keyword arguments passed to ``LineCollection``.

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib Figure and Axes objects.

        Examples
        --------
            >>> import numpy as np
            >>> from cleopatra.mesh_glyph import MeshGlyph
            >>> mg = MeshGlyph(
            ...     np.array([0.0, 1.0, 0.5]),
            ...     np.array([0.0, 0.0, 1.0]),
            ...     np.array([[0, 1, 2]]),
            ... )
            >>> fig, ax = mg.plot_outline(color="blue")
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.get_figure()

        segments = self._build_edge_segments()

        lc = mcoll.LineCollection(
            segments, colors=color, linewidths=linewidth, **kwargs
        )
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect("equal")

        result = (fig, ax)
        return result

    def _build_edge_segments(
        self,
    ) -> list[list[tuple[float, float]]]:
        """Build line segments for wireframe rendering.

        Uses edge_node_connectivity if available, otherwise derives
        unique edges from face_node_connectivity.

        Returns
        -------
        list[list[tuple[float, float]]]
            List of [(x1, y1), (x2, y2)] segments.
        """
        segments: list[list[tuple[float, float]]] = []

        if self._edge_nodes is not None:
            for i in range(self._edge_nodes.shape[0]):
                n1 = int(self._edge_nodes[i, 0])
                n2 = int(self._edge_nodes[i, 1])
                segments.append(
                    [
                        (self._node_x[n1], self._node_y[n1]),
                        (self._node_x[n2], self._node_y[n2]),
                    ]
                )
        else:
            seen: set[tuple[int, int]] = set()
            for i in range(self.n_faces):
                row = self._face_nodes[i]
                nodes = row[row != self._fill_value]
                n = len(nodes)
                for j in range(n):
                    n1, n2 = int(nodes[j]), int(nodes[(j + 1) % n])
                    key = (min(n1, n2), max(n1, n2))
                    if key not in seen:
                        seen.add(key)
                        segments.append(
                            [
                                (self._node_x[n1], self._node_y[n1]),
                                (self._node_x[n2], self._node_y[n2]),
                            ]
                        )

        return segments
