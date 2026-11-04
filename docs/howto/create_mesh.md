(howto-mesh)=

# Create a mesh

A **mesh** is a discrete representation of the continuous simulation domain / geometry.
Meshes are produced by {py:class}`~EasyFEA.FEM.Mesher` (which wraps
[Gmsh](https://gmsh.info/)) and are accessible in the {py:mod}`EasyFEA.FEM` module.

```{note}
To import an existing mesh file instead of creating one from scratch, see
{ref}`howto-import-mesh`.
```

```{seealso}
- {ref}`FEM` API
```

______________________________________________________________________

## Geometry objects

Mesh generation relies on manipulating the geometric objects listed below, which are
available in the {py:class}`EasyFEA.Geoms` namespace:

| Class                                | Description                                                     |
| ------------------------------------ | --------------------------------------------------------------- |
| {py:class}`~EasyFEA.Geoms.Point`     | Single point; optional fillet radius `r`                        |
| {py:class}`~EasyFEA.Geoms.Line`      | Straight segment between two points                             |
| {py:class}`~EasyFEA.Geoms.Circle`    | Full circle defined by center and diameter                      |
| {py:class}`~EasyFEA.Geoms.CircleArc` | Circular arc (from radius, center, or on-circle point)          |
| {py:class}`~EasyFEA.Geoms.Points`    | Closed polygon / spline from a list of points                   |
| {py:class}`~EasyFEA.Geoms.Contour`   | Closed loop assembled from `Line`, `CircleArc`, and/or `Points` |
| {py:class}`~EasyFEA.Geoms.Domain`    | Axis-aligned rectangle or box defined by two corners            |

`meshSize` on any geometry object sets the local target element size.

______________________________________________________________________

```{tip}
All examples below use `Matplotlib` (matplotlib) for inline output.
For interactive 3D visualization, replace `Matplotlib.Plot_Mesh(mesh)` with
`PyVista.Plot_Mesh(mesh)` — see {ref}`howto-postprocess`.
```

## 2D mesh

{py:meth}`~EasyFEA.Geoms._Geom.Mesh_2D` meshes the surface of the geometry, with
optional inclusions (holes, filled regions), cracks, and local refinement.

### Simple rectangle

```{eval-rst}
.. jupyter-execute::

    from EasyFEA import Matplotlib, ElemType
    from EasyFEA.Geoms import Domain

    domain = Domain((0, 0), (100, 20), meshSize=5.0)
    mesh = domain.Mesh_2D([], ElemType.TRI3)
    Matplotlib.Plot_Mesh(mesh)
    Matplotlib.Plot_Tags(mesh)
```

### Rectangle with a circular hole

When the geometry is not used as a contour in `Mesh_*` functions `isFilled=False` (the
default) means the geometry defines a hole or boundary only; `isFilled=True` means it
defines a filled region.

```{eval-rst}
.. jupyter-execute::

    from EasyFEA import Matplotlib, ElemType
    from EasyFEA.Geoms import Domain, Circle

    domain = Domain((0, 0), (100, 50), meshSize=5.0)
    hole   = Circle(center=(50, 25), diam=20, meshSize=1.0)

    mesh = domain.Mesh_2D([hole], ElemType.TRI6)
    Matplotlib.Plot_Mesh(mesh, title="hole.isFilled=False by default")

    hole.isFilled=True
    mesh = domain.Mesh_2D([hole], ElemType.TRI6)
    Matplotlib.Plot_Mesh(mesh, title="hole.isFilled=True")
```

### Structured quad mesh

By setting `isOrganised=True`, you obtain a structured mesh that requires a structurable
polygon consisting of four or three segments:

```{eval-rst}
.. jupyter-execute::

    from EasyFEA import Matplotlib, ElemType
    from EasyFEA.Geoms import Domain

    domain = Domain((0, 0), (100, 20), meshSize=5.0)
    mesh = domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    Matplotlib.Plot_Mesh(mesh)
```

______________________________________________________________________

## 3D mesh by extrusion

{py:meth}`~EasyFEA.Geoms._Geom.Mesh_Extrude` creates a 3D volume by extruding a 2D
surface along a direction vector.

```{eval-rst}
.. jupyter-execute::

    from EasyFEA import Matplotlib, ElemType
    from EasyFEA.Geoms import Domain

    domain = Domain((0, 0), (100, 20), meshSize=5.0)
    mesh = domain.Mesh_Extrude(
        extrude=(0, 0, 10),   # extrusion direction and length
        layers=[3],            # number of layers along the extrusion
        elemType=ElemType.HEXA8,
        isOrganised=True,
    )
    Matplotlib.Plot_Mesh(mesh)
```

______________________________________________________________________

## 3D mesh by revolution

{py:meth}`~EasyFEA.Geoms._Geom.Mesh_Revolve` sweeps a 2D cross-section around an axis to
create an axisymmetric volume.

```{eval-rst}
.. jupyter-execute::

    from EasyFEA import Matplotlib, ElemType
    from EasyFEA.Geoms import Domain, Line, Point

    cross_section = Domain((5, 0), (8, 8), meshSize=2.0)

    mesh = cross_section.Mesh_Revolve(
        [],
        (0, 0),
        (0, 1),
        angle=270,
        layers=[30],    # number of angular divisions
        elemType=ElemType.PRISM6,
    )
    Matplotlib.Plot_Mesh(mesh)
```

______________________________________________________________________

## Element types

The types of isoparametric geometric elements that can be used to discretize the domain
are available in the {py:class}`~EasyFEA.FEM.ElemType` class.

(howto-function-spaces)=

## Non-isoparametric fields

By default a field uses the mesh's elements: it is isoparametric.
`Mesh._With_FunctionSpace` returns a new mesh on which one field uses the complete
Lagrange elements of another order, on the same geometry. "P2 on TRI3" below means a
quadratic field on a linear mesh, "P1 on TRI6" the opposite.

```python
from EasyFEA import ElemType, Models, Simulations
from EasyFEA.Geoms import Domain

contour = Domain((0, 0), (1, 1), 0.1)
mesh = contour.Mesh_2D([], ElemType.TRI3)
# u on TRI6, the geometry on TRI3
meshP2 = mesh._With_FunctionSpace(Simulations.Elastic.U, 2)

material = Models.Elastic.Isotropic(2, E=210000, v=0.3)
simu = Simulations.Elastic(meshP2, material)
```

The space belongs to one field. The displacement is one problem type in every
simulation: `Simulations.InElastic.U` is `Simulations.Elastic.U`. Other fields have
their own, such as `Simulations.Thermal.T`: a space set for `T` leaves the displacement
isoparametric, without an error.

### Select the nodes of a field

On a non-isoparametric space, pass `problemType=` to the `Nodes_*` functions:

```python
U = Simulations.Elastic.U
nodesClamped = meshP2.Nodes_Conditions(lambda x, y, z: x == 0, problemType=U)
simu.add_dirichlet(nodesClamped, [0, 0], ["x", "y"])
```

Without it they select the geometry's nodes. Under P2 on TRI3 the nodes the space adds
are then left out and, in a Dirichlet condition, stay free. The first such selection
prints a warning. The figures select `x <= 0.5` both ways:

```{eval-rst}
.. jupyter-execute::

    from EasyFEA import Matplotlib, ElemType, Simulations
    from EasyFEA.Geoms import Domain

    U = Simulations.Elastic.U
    domain = Domain((0, 0), (1, 1), meshSize=0.25)
    mesh = domain.Mesh_2D([], ElemType.TRI3)._With_FunctionSpace(U, 2)
    meshU = mesh._Get_FunctionSpace(U).Get_mesh()

    nodes = mesh.Nodes_Conditions(lambda x, y, z: x <= 0.5)
    ax = Matplotlib.Plot_Mesh(mesh, alpha=0)
    Matplotlib.Plot_Nodes(meshU, nodes, title="without problemType", ax=ax)

    nodes = mesh.Nodes_Conditions(lambda x, y, z: x <= 0.5, problemType=U)
    ax = Matplotlib.Plot_Mesh(mesh, alpha=0)
    Matplotlib.Plot_Nodes(meshU, nodes, title="with problemType", ax=ax)
```

Two things to know about a selection made with `problemType=`:

- Its ids can go past `mesh.Nn`, so `mesh.coord[nodes]` raises `IndexError`. Read the
  coordinates with `mesh._Get_FunctionSpace(U).coord[nodes]`.
- It can serve another field: a condition skips the nodes that carry none of that
  field's DOFs.

### Curved boundaries

With P2 on TRI3 an edge stays straight, so its mid-edge node lies on the chord, not on
the circle, and a test on the radius misses it. So when a space adds nodes,
`Nodes_Conditions`, `Nodes_Circle` and `Nodes_Cylinder` with `problemType=` also take
every node of a boundary edge or face whose vertices are all selected. On a partitioned
mesh these calls are then collective under MPI: make them on every rank.

The figure selects the outer radius of a thick cylinder, mid-edge nodes included:

```{eval-rst}
.. jupyter-execute::

    import numpy as np
    from EasyFEA import Matplotlib, ElemType, Simulations
    from EasyFEA.Geoms import Point, Points

    a, b = 100.0, 200.0  # inner and outer radius
    meshSize = (b - a) / 3
    # r > 0 rounds the corner (b, b), r < 0 cuts an arc centred on the origin
    contour = Points(
        [
            Point(0, 0, r=-a),
            (b, 0),
            Point(b, b, r=b),
            (0, b),
        ],
        meshSize,
    ).Get_Contour()

    U = Simulations.Elastic.U
    mesh = contour.Mesh_2D([], ElemType.TRI3)._With_FunctionSpace(U, 2)

    meshU = mesh._Get_FunctionSpace(U).Get_mesh()
    nodes = mesh.Nodes_Conditions(
        lambda x, y, z: np.isclose(np.hypot(x, y), b), problemType=U
    )
    ax = Matplotlib.Plot_Mesh(mesh, alpha=0)
    Matplotlib.Plot_Geoms(
        contour,
        ax=ax,
        plotPoints=False,
        plotLegend=False,
        showGrid=False,
    )
    Matplotlib.Plot_Nodes(meshU, nodes, title="the rim", ax=ax)
```

This rule only looks at vertices, so one call spanning two faces can take an edge
between them: selecting the top and bottom of a strip one element thick also takes its
end sides' mid-edge nodes ({ref}`see the figure <howto-check-selection>`). To avoid
this, select one face per call, or by tag: here `mesh.Nodes_Tags("L2", problemType=U)`
takes exactly the nodes of the outer arc. Check what was picked with
{py:func}`~EasyFEA.Viz.Matplotlib.Plot_BoundaryConditions`, which draws every selected
node.

The geometry does not change, which sets the limits on a curved boundary:

- Above the geometry's order (P2 on TRI3), edges stay straight. To follow the curve,
  mesh with quadratic elements instead (TRI6).
- Below it (P1 on TRI6), the field integrates on the chord, loads included.

### Limits

These do not support another order:

| Feature                                                          | With a non-isoparametric displacement        |
| ---------------------------------------------------------------- | -------------------------------------------- |
| {py:class}`~EasyFEA.Simulations.Beam`                            | raises at construction: it rebuilds the mesh |
| External variables of {py:class}`~EasyFEA.Simulations.InElastic` | raises: they are set on the mesh's nodes     |

{py:class}`~EasyFEA.Simulations.PhaseField` needs one order for both fields: give
`Simulations.PhaseField.U` and `Simulations.PhaseField.D` the same, or
{py:func}`~EasyFEA.Simulations.PhaseField.Solve` raises.
