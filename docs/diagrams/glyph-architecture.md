# Cleopatra glyph architecture

Architecture and class diagrams for cleopatra's glyph subsystem and the shared
colour / scale pipeline, including the geoplot-upstream additions
(classification `scheme`, value→size scaling, `FlowGlyph`, `KDEGlyph`).

All diagrams use [Mermaid](https://mermaid.js.org/) (rendered by
`mkdocs-mermaid2` in the docs site and by GitHub).

## Legend

| Style | Meaning |
|-------|---------|
| **base** (blue) | `Glyph` base class — shared figure/axes, colour, colorbar, ticks |
| **glyph** (green) | concrete user-facing glyph subclasses |
| **standalone** (grey) | classes that do **not** subclass `Glyph` |
| **helper** (amber) | `styles` functions / classes (no glyph instance needed) |

---

## 1. Class hierarchy

`ArrayGlyph`, `MeshGlyph`, `ScatterGlyph`, `PolygonGlyph`, `VectorGlyph`,
`FlowGlyph`, and `KDEGlyph` subclass `Glyph`. `StatisticalGlyph` is independent.

```mermaid
classDiagram
    class Glyph {
        +dict default_options
        +Figure fig
        +Axes ax
        +vmin
        +vmax
        +option_keys() set
        +filter_kwargs(kwargs) dict
        +create_figure_axes()
        +get_ticks() ndarray
        #_resolve_limits(values)
        #_prepare_scalar_mapping(values)
        #_prepare_classified_mapping(values, scheme)
        #_create_norm_and_cbar_kw(ticks)
        #_levels_to_bounds(levels, vmin, vmax)
        +create_color_bar(ax, im, cbar_kw)
    }

    class ArrayGlyph {
        +plot()
        +animate()
        #_resolve_color_limits()
    }
    class MeshGlyph {
        +plot()
        #_render_mesh()
    }
    class ScatterGlyph {
        +x
        +y
        +values
        +sizes
        +plot()
        #_resolve_marker_area()
        #_draw_size_legend()
    }
    class PolygonGlyph {
        +polygons
        +values
        +plot(outline_only)
    }
    class VectorGlyph {
        +x
        +y
        +u
        +v
        +magnitude
        +plot(kind)
        +add_key()
    }
    class FlowGlyph {
        +paths
        +values
        +widths
        +plot()
        #_resolve_linewidths()
        #_draw_width_legend()
    }
    class KDEGlyph {
        +x
        +y
        +clip_path
        +evaluate()
        +plot()
        #_bandwidth()
        #_resolve_levels()
        #_apply_clip()
    }
    class StatisticalGlyph {
        +plot()
    }

    Glyph <|-- ArrayGlyph
    Glyph <|-- MeshGlyph
    Glyph <|-- ScatterGlyph
    Glyph <|-- PolygonGlyph
    Glyph <|-- VectorGlyph
    Glyph <|-- FlowGlyph
    Glyph <|-- KDEGlyph

    note for StatisticalGlyph "Stands alone — 1-D/2-D histograms,\ndoes not subclass Glyph"

    style Glyph fill:#cfe2ff,stroke:#3D59AB,color:#000
    style ArrayGlyph fill:#d1e7dd,stroke:#0f5132,color:#000
    style MeshGlyph fill:#d1e7dd,stroke:#0f5132,color:#000
    style ScatterGlyph fill:#d1e7dd,stroke:#0f5132,color:#000
    style PolygonGlyph fill:#d1e7dd,stroke:#0f5132,color:#000
    style VectorGlyph fill:#d1e7dd,stroke:#0f5132,color:#000
    style FlowGlyph fill:#d1e7dd,stroke:#0f5132,color:#000
    style KDEGlyph fill:#d1e7dd,stroke:#0f5132,color:#000
    style StatisticalGlyph fill:#e2e3e5,stroke:#41464b,color:#000
```

---

## 2. Shared colour / scale pipeline

Every colour-by-value glyph that routes through `_prepare_scalar_mapping`
(`ScatterGlyph`, `PolygonGlyph`, `VectorGlyph`, `FlowGlyph`) shares one
contract. Since the geoplot-upstream work, a `scheme` option short-circuits to
classified (discrete) colouring; otherwise the continuous `color_scale` /
`levels` path runs. `ArrayGlyph` / `MeshGlyph` build their norm directly via
`_create_norm_and_cbar_kw` and therefore do **not** accept `scheme`.

```mermaid
flowchart TD
    A["glyph.plot(values)"] --> B["_prepare_scalar_mapping(values)"]
    B --> C["_resolve_limits(values)\nvmin / vmax / ticks_spacing"]
    C --> D{"scheme set?"}

    D -- "yes" --> E["_prepare_classified_mapping(values, scheme)"]
    E --> F["styles.classify(values, scheme, k)"]
    F --> G["bin_edges + BoundaryNorm\n(discrete classes)"]
    E -. "warn if color_scale/levels\nalso set (ignored)" .-> G
    G --> K["create_color_bar(ax, im, cbar_kw)\nstepped colorbar"]

    D -- "no" --> H["get_ticks()"]
    H --> I["_create_norm_and_cbar_kw(ticks)"]
    I --> J["norm (linear / power / sym-log /\nboundary / midpoint) or None"]
    J --> K

    subgraph bypass["ArrayGlyph / MeshGlyph (bypass)"]
        M["plot()"] --> N["_resolve_color_limits / get_ticks"]
        N --> I
    end

    style B fill:#cfe2ff,stroke:#3D59AB,color:#000
    style E fill:#cfe2ff,stroke:#3D59AB,color:#000
    style I fill:#cfe2ff,stroke:#3D59AB,color:#000
    style K fill:#cfe2ff,stroke:#3D59AB,color:#000
    style F fill:#fff3cd,stroke:#997404,color:#000
    style G fill:#fff3cd,stroke:#997404,color:#000
```

---

## 3. `styles` helpers

Stateless helpers used by the glyphs (and usable standalone). `scheme`/`k`
live in `CLASSIFY_OPTIONS`, mixed only into the glyphs whose colouring is
driven purely by the norm.

```mermaid
classDiagram
    class classify {
        <<function>>
        +classify(values, scheme, k) tuple~edges,BoundaryNorm~
    }
    class resolve_sizes {
        <<function>>
        +resolve_sizes(values, out_min, out_max, scale) ndarray
    }
    class size_legend {
        <<function>>
        +size_legend(ax, marker_sizes, labels) Legend
    }
    class width_legend {
        <<function>>
        +width_legend(ax, linewidths, labels) Legend
    }
    class Scale {
        +rescale(v, omin, omax, nmin, nmax)$
        +log_scale(v)$
        +power_scale(min_val)$
        +identity_scale(min_val, max_val)$
    }
    class MidpointNormalize {
        +midpoint
        +__call__(value)
    }
    class ColorScale {
        <<enum>>
        LINEAR
        POWER
        SYM_LOGNORM
        BOUNDARY_NORM
        MIDPOINT
    }
    class CLASSIFY_OPTIONS {
        <<dict>>
        scheme = None
        k = 5
    }

    Normalize <|-- MidpointNormalize
    classify ..> BoundaryNorm : returns
    resolve_sizes ..> Scale : uses rescale / log_scale
    ScatterGlyph ..> resolve_sizes : marker area
    ScatterGlyph ..> size_legend
    FlowGlyph ..> resolve_sizes : line width
    FlowGlyph ..> width_legend
    Glyph ..> classify : via _prepare_classified_mapping

    style classify fill:#fff3cd,stroke:#997404,color:#000
    style resolve_sizes fill:#fff3cd,stroke:#997404,color:#000
    style size_legend fill:#fff3cd,stroke:#997404,color:#000
    style width_legend fill:#fff3cd,stroke:#997404,color:#000
    style Scale fill:#fff3cd,stroke:#997404,color:#000
    style MidpointNormalize fill:#fff3cd,stroke:#997404,color:#000
```

---

## 4. Sequence — classified choropleth (`scheme`)

A `PolygonGlyph(..., scheme="quantiles", k=5).plot()` call, end to end.

```mermaid
sequenceDiagram
    actor User
    participant PG as PolygonGlyph
    participant G as Glyph (base)
    participant ST as styles.classify
    participant MPL as matplotlib

    User->>PG: plot()
    PG->>G: _prepare_scalar_mapping(values)
    G->>G: _resolve_limits(values)
    G->>G: scheme set -> _prepare_classified_mapping
    G->>ST: classify(values, "quantiles", k=5)
    ST-->>G: (bin_edges, BoundaryNorm)
    G-->>PG: (norm, cbar_kw, edges)
    PG->>MPL: PolyCollection(array=values, norm=BoundaryNorm)
    PG->>G: create_color_bar(ax, pc, cbar_kw)
    G->>MPL: fig.colorbar(...) (stepped)
    PG-->>User: (fig, ax, PolyCollection)
```

---

## 5. Sequence — value→size scatter

A `ScatterGlyph(x, y, values=v, sizes=w, size_legend=True).plot()` call: colour
and size are resolved independently.

```mermaid
sequenceDiagram
    actor User
    participant SG as ScatterGlyph
    participant G as Glyph (base)
    participant RS as styles.resolve_sizes
    participant SL as styles.size_legend
    participant MPL as matplotlib

    User->>SG: plot()
    SG->>SG: _resolve_marker_area()
    SG->>RS: resolve_sizes(sizes, *size_limits, scale)
    RS-->>SG: per-point areas
    SG->>G: _prepare_scalar_mapping(values)
    G-->>SG: (norm, cbar_kw, ticks)
    SG->>MPL: ax.scatter(c=values, s=areas, norm=norm)
    SG->>G: create_color_bar(ax, paths, cbar_kw)
    SG->>SL: size_legend(ax, repr_areas, labels)
    SG-->>User: (fig, ax, PathCollection)
```

---

## Notes / design patterns

- **Template method** — `Glyph` owns the colour/colorbar/ticks contract; each
  subclass implements only its `plot()` rendering and reuses
  `_prepare_scalar_mapping` + `create_color_bar`.
- **Strategy** — `color_scale` (`ColorScale`) and `scheme` select the
  normalization strategy; `size_scale` / `width_scale` select the value→size
  transform inside `resolve_sizes`.
- **Shared-but-scoped options** — `scheme`/`k` are *not* in the shared
  `DEFAULT_OPTIONS`; they live in `CLASSIFY_OPTIONS` and are mixed only into the
  glyphs whose renderer is driven purely by the norm, so `ArrayGlyph` /
  `MeshGlyph` / `KDEGlyph` reject `scheme` instead of silently ignoring it.
- **No new dependency at all** — classification (including the Fisher-Jenks
  natural-breaks optimisation), KDE, and rescaling are pure numpy + matplotlib.
