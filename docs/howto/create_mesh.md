(howto-mesh)=
# Create a mesh

A **mesh** is a discrete representation of the simulation domain.
Meshes are produced by {py:class}`~EasyFEA.FEM.Mesher` (which wraps [Gmsh](https://gmsh.info/)) and
are accessible in the {py:mod}`EasyFEA.FEM` module.

```{note}
To import an existing mesh file instead of creating one from scratch, see
{ref}`howto-import-mesh`.
```

---

## Geometry objects

Mesh generation relies on manipulating the geometric objects listed below, which are available in the {py:class}`EasyFEA.Geoms` namespace:

| Class | Description |
|---|---|
| {py:class}`~EasyFEA.Geoms.Point` | Single point; optional fillet radius `r` |
| {py:class}`~EasyFEA.Geoms.Line` | Straight segment between two points |
| {py:class}`~EasyFEA.Geoms.Circle` | Full circle defined by center and diameter |
| {py:class}`~EasyFEA.Geoms.CircleArc` | Circular arc (from radius, center, or on-circle point) |
| {py:class}`~EasyFEA.Geoms.Points` | Closed polygon / spline from a list of points |
| {py:class}`~EasyFEA.Geoms.Contour` | Closed loop assembled from `Line`, `CircleArc`, and/or `Points` |
| {py:class}`~EasyFEA.Geoms.Domain` | Axis-aligned rectangle or box defined by two corners |

`meshSize` on any geometry object sets the local target element size.
`isHollow=True` (the default) means the geometry defines a hole
or boundary only;
`isHollow=False` means it defines a filled region.

---

```{tip}
All examples below use `Display` (matplotlib) for inline output.
For interactive 3D visualization, replace `Display.Plot_Mesh(mesh)` with
`PyVista.Plot_Mesh(mesh)` — see {ref}`howto-postprocess`.
```

## 2D mesh

{py:meth}`~EasyFEA.Geoms._Geom.Mesh_2D` meshes the surface of the geometry,
with optional inclusions (holes, filled regions), cracks, and local
refinement.

### Simple rectangle

```{eval-rst}
.. jupyter-execute::

    from EasyFEA import Display, ElemType
    from EasyFEA.Geoms import Domain

    domain = Domain((0, 0), (100, 20), meshSize=5.0)
    mesh = domain.Mesh_2D([], ElemType.TRI3)
    Display.Plot_Mesh(mesh)
    Display.Plot_Tags(mesh)
```

### Rectangle with a circular hole

```{eval-rst}
.. jupyter-execute::

    from EasyFEA import Display, ElemType
    from EasyFEA.Geoms import Domain, Circle

    domain = Domain((0, 0), (100, 50), meshSize=5.0)
    hole   = Circle(center=(50, 25), diam=20, meshSize=1.0, isHollow=True)

    mesh = domain.Mesh_2D([hole], ElemType.TRI3)
    Display.Plot_Mesh(mesh)
```

### Structured quad mesh

By setting `isOrganised=True`, you obtain a structured mesh that requires a structurable polygon consisting of four or three segments:

```{eval-rst}
.. jupyter-execute::

    from EasyFEA import Display, ElemType
    from EasyFEA.Geoms import Domain

    domain = Domain((0, 0), (100, 20), meshSize=5.0)
    mesh = domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    Display.Plot_Mesh(mesh)
```

---

## 3D mesh by extrusion

{py:meth}`~EasyFEA.Geoms._Geom.Mesh_Extrude` creates a 3D volume by
extruding a 2D surface along a direction vector.

```{eval-rst}
.. jupyter-execute::

    from EasyFEA import Display, ElemType
    from EasyFEA.Geoms import Domain

    domain = Domain((0, 0), (100, 20), meshSize=5.0)
    mesh = domain.Mesh_Extrude(
        extrude=(0, 0, 10),   # extrusion direction and length
        layers=[3],            # number of layers along the extrusion
        elemType=ElemType.HEXA8,
        isOrganised=True,
    )
    Display.Plot_Mesh(mesh)
```

---

## 3D mesh by revolution

{py:meth}`~EasyFEA.Geoms._Geom.Mesh_Revolve` sweeps a 2D cross-section
around an axis to create an axisymmetric volume.

```{eval-rst}
.. jupyter-execute::

    from EasyFEA import Display, ElemType
    from EasyFEA.Geoms import Domain, Line, Point

    cross_section = Domain((5, 0), (8, 8), meshSize=2.0)

    # revolution axis: Y-axis
    axis = Line((0, 0), (0, 1))

    mesh = cross_section.Mesh_Revolve(
        axis=axis,
        angle=270,
        layers=[30],    # number of angular divisions
        elemType=ElemType.PRISM6,
    )
    Display.Plot_Mesh(mesh)
```

---

## Element types

The types of isoparametric geometric elements that can be used to discretize the domain are available in the {py:class}`~EasyFEA.FEM.ElemType` class.