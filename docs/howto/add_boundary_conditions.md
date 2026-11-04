(howto-boundary-conditions)=

# Apply boundary conditions

**Boundary conditions** prescribe loads and constraints on the degrees of freedom of a
{py:class}`~EasyFEA.Simulations._Simu`. They are applied to **node sets** via `add_*`
methods on the simulation object and are accessible in the {py:mod}`EasyFEA.Simulations`
namespace. Node sets come from coordinate conditions or from mesh tags — see
{ref}`howto-import-mesh-tags` for the tag-based approach.

______________________________________________________________________

## Query available unknowns

Each simulation exposes the DOF names it solves for via
{py:meth}`~EasyFEA.Simulations._Simu.Get_unknowns`:

```python
print(simu.Get_unknowns())
# Elastic 2D:       ['x', 'y']
# Elastic 3D:       ['x', 'y', 'z']
# HyperElastic 2D:  ['x', 'y']
# Thermal:           ['t']
# Beam (planar):     ['x', 'y', 'rz']
# PhaseField:        ['x', 'y'] for elastic sub-problem, ['d'] for damage
```

Pass these strings as the `unknowns` argument to every `add_*` method. You can also pass
a **subset** — for example, fix only `["x"]` to constrain horizontal motion while
leaving the vertical DOF free.

______________________________________________________________________

## Summary of BC methods

| Method                                                 | Physical meaning                | Integration             |
| ------------------------------------------------------ | ------------------------------- | ----------------------- |
| {py:meth}`~EasyFEA.Simulations._Simu.add_dirichlet`    | Prescribed DOF value            | —                       |
| {py:meth}`~EasyFEA.Simulations._Simu.add_neumann`      | Concentrated nodal force / flux | Point                   |
| {py:meth}`~EasyFEA.Simulations._Simu.add_lineLoad`     | Force per unit length           | Along a line            |
| {py:meth}`~EasyFEA.Simulations._Simu.add_surfLoad`     | Force per unit area             | Over a surface          |
| {py:meth}`~EasyFEA.Simulations._Simu.add_volumeLoad`   | Body force per unit volume      | Over a volume           |
| {py:meth}`~EasyFEA.Simulations._Simu.add_pressureLoad` | Normal pressure                 | Along a line or surface |

______________________________________________________________________

## Dirichlet conditions with {py:meth}`~EasyFEA.Simulations._Simu.add_dirichlet`

Prescribes the value of one or more DOFs on a set of nodes.

On a non-isoparametric space (P2 on TETRA4: a quadratic field on a linear mesh), select
with `problemType=`: `mesh.Nodes_Conditions(func, problemType=U)`. Without it the
selection holds the geometry's nodes only, and the nodes the space adds stay free. See
{ref}`howto-function-spaces`.

```python
# fix x an y displacement DOFs to zero (Elastic simulation)
nodes = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
simu.add_dirichlet(nodes, [0, 0], ["x", "y"])

# impose temperature 100 °C on the top surface (Thermal simulation)
nodes_top = mesh.Nodes_Conditions(lambda x, y, z: y == H)
simu.add_dirichlet(nodes_top, [100.0], ["t"])

# prescribed y displacement that varies along x
simu.add_dirichlet(nodes, [lambda x, y, z: 0.01 * x], ["y"])
```

On a plate, the left side clamped and the right side moved along y. The figures of this
page draw the conditions with
{py:func}`~EasyFEA.Viz.Matplotlib.Plot_BoundaryConditions`: a non-zero value is an arrow
scaled by the largest value, outside the material: it starts on its node when it points
out of the material, and ends there otherwise. A zero value is a marker:

```{eval-rst}
.. jupyter-execute::

    from EasyFEA import Matplotlib, ElemType, Models, Simulations
    from EasyFEA.Geoms import Circle, Domain, Point

    L, H = 2.0, 1.0
    domain = Domain((0, 0), (L, H), meshSize=0.25)
    mesh = domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    material = Models.Elastic.Isotropic(2)
    simu = Simulations.Elastic(mesh, material)

    nodesLeft = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesRight = mesh.Nodes_Conditions(lambda x, y, z: x == L)
    simu.add_dirichlet(nodesLeft, [0, 0], ["x", "y"])
    simu.add_dirichlet(nodesRight, [0.1], ["y"])
    Matplotlib.Plot_BoundaryConditions(simu, title="add_dirichlet")
```

______________________________________________________________________

## Concentrated force with {py:meth}`~EasyFEA.Simulations._Simu.add_neumann`

Applies a concentrated force (or flux) directly on the selected nodes. No integration is
performed — the value is added as-is to the right-hand side. Useful for point loads on
beams or reactions at a single node.

```python
# point load at the beam tip (Elastic simulation)
nodes_tip = mesh.Nodes_Point((L, h / 2))
simu.add_neumann(nodes_tip, [0, -500], ["x", "y"])
```

On the plate's top right corner:

```{eval-rst}
.. jupyter-execute::

    simu.Bc_Init()
    nodesCorner = mesh.Nodes_Point((L, H))
    simu.add_neumann(nodesCorner, [-500], ["y"])
    Matplotlib.Plot_BoundaryConditions(simu, title="add_neumann")
```

______________________________________________________________________

## Distributed loads

On a non-isoparametric space, these integrate on the field's elements, with or without
`problemType=` on the selection. The selection must hold whole edges or faces: one with
no complete element raises.

### Line load with {py:meth}`~EasyFEA.Simulations._Simu.add_lineLoad`

Force per unit **length** integrated along the selected boundary edges. Used in 2D for
loads applied along a line, or in 3D for edge loads.

```python
# uniform downward load of −1000 N/m along the top edge
nodes_top = mesh.Nodes_Conditions(lambda x, y, z: y == H)
simu.add_lineLoad(nodes_top, [-1000], ["y"])

# spatially varying line load
simu.add_lineLoad(nodes_top, [lambda x, y, z: -500 * (1 + x / L)], ["y"])
```

```{eval-rst}
.. jupyter-execute::

    simu.Bc_Init()
    nodesTop = mesh.Nodes_Conditions(lambda x, y, z: y == H)
    simu.add_lineLoad(nodesTop, [-1000], ["y"])
    Matplotlib.Plot_BoundaryConditions(simu, title="add_lineLoad")
```

### Surface load with {py:meth}`~EasyFEA.Simulations._Simu.add_surfLoad`

Force per unit **area** integrated over the selected boundary faces. Typical for
pressure loads in 3D, or traction/pressure in 2D (force per unit area × thickness).

```python
# uniform pressure of −800 Pa along x on the right face
nodes_right = mesh.Nodes_Conditions(lambda x, y, z: x == L)
simu.add_surfLoad(nodes_right, [-800], ["x"])

# traction vector on a 3D face
simu.add_surfLoad(nodes_right, [0, 0, -1e6], ["x", "y", "z"])
```

```{eval-rst}
.. jupyter-execute::

    simu.Bc_Init()
    simu.add_surfLoad(nodesRight, [-800], ["x"])
    Matplotlib.Plot_BoundaryConditions(simu, title="add_surfLoad")
```

### Volume load with {py:meth}`~EasyFEA.Simulations._Simu.add_volumeLoad`

Body force per unit **volume** integrated over the selected elements. Typical use is
gravity.

```python
# gravity in −y, ρ = 7800 kg/m³
simu.add_volumeLoad(mesh.nodes, [-7800 * 9.81], ["y"])
```

A condition on every node is drawn once, at the center of the mesh:

```{eval-rst}
.. jupyter-execute::

    simu.Bc_Init()
    simu.add_volumeLoad(mesh.nodes, [-7800 * 9.81], ["y"])
    Matplotlib.Plot_BoundaryConditions(simu, title="add_volumeLoad")
```

### Pressure load with {py:meth}`~EasyFEA.Simulations._Simu.add_pressureLoad`

Applies a **normal** pressure of given magnitude on a boundary surface. Its sign follows
the orientation of the boundary, not the material: a positive magnitude pushes into the
material on some boundaries and pulls on others, and it can differ between 2D and 3D.
Check the direction with `Plot_BoundaryConditions`, and flip the sign if needed.

```python
# normal pressure of 1 MPa on a cylinder inner surface
simu.add_pressureLoad(inner_surface_nodes, magnitude=-1e6)
```

On a hole in the plate, where a negative magnitude pushes into the material:

```{eval-rst}
.. jupyter-execute::

    hole = Circle(Point(L / 2, H / 2), H / 2, meshSize=0.1)
    meshHole = domain.Mesh_2D([hole], ElemType.TRI3)
    simuHole = Simulations.Elastic(meshHole, material)
    nodesHole = meshHole.Nodes_Circle(hole)
    simuHole.add_pressureLoad(nodesHole, magnitude=-1e6)
    Matplotlib.Plot_BoundaryConditions(simuHole, title="add_pressureLoad")
```

______________________________________________________________________

## Spatially varying values

All `values` arguments accept:

- a **float** — uniform value applied to all selected nodes,
- a **NumPy array** of length `Nn` — one value per node,
- a **lambda function** `lambda x, y, z: ...` — evaluated at node (or integration-point)
  coordinates.

```python
# temperature that increases linearly along x
simu.add_dirichlet(nodes, [lambda x, y, z: 20 + 80 * x / L], ["t"])

# load proportional to distance from center
simu.add_surfLoad(nodes, [lambda x, y, z: -100 * (x - L/2)**2], ["x"])
```

Functions always receive three arguments `x, y, z` regardless of the problem dimension.

An array needs one value per selected node, or it raises. A function has no such
constraint, whichever nodes the selection holds.

______________________________________________________________________

(howto-check-selection)=

## Visualize applied conditions

```python
from EasyFEA import Matplotlib, PyVista

Matplotlib.Plot_BoundaryConditions(simu)   # matplotlib
PyVista.Plot_BoundaryConditions(simu).show()   # interactive 3D
```

These draw every selected node, those a space adds included, so they show a node picked
by mistake. Only a condition on every node is drawn once, at the center. On a strip one
element thick with P2 on TRI3, selecting the top and the bottom in one call also takes
the mid-edge nodes of the end sides, whose two vertices are selected (see
{ref}`howto-function-spaces`). One call per face does not:

```{eval-rst}
.. jupyter-execute::

    from EasyFEA import Matplotlib, ElemType, Models, Simulations
    from EasyFEA.Geoms import Domain

    U = Simulations.Elastic.U
    domain = Domain((0, 0), (3, 1), meshSize=1.0)
    mesh = domain.Mesh_2D([], ElemType.TRI3)._With_FunctionSpace(U, 2)
    material = Models.Elastic.Isotropic(2)

    simu = Simulations.Elastic(mesh, material)
    nodes = mesh.Nodes_Conditions(lambda x, y, z: (y == 0) | (y == 1), problemType=U)
    simu.add_dirichlet(nodes, [0, 0], ["x", "y"])
    Matplotlib.Plot_BoundaryConditions(simu, title="one call", plotLegend=False)

    simu = Simulations.Elastic(mesh, material)
    nodesBottom = mesh.Nodes_Conditions(lambda x, y, z: y == 0, problemType=U)
    nodesTop = mesh.Nodes_Conditions(lambda x, y, z: y == 1, problemType=U)
    simu.add_dirichlet(nodesBottom, [0, 0], ["x", "y"])
    simu.add_dirichlet(nodesTop, [0, 0], ["x", "y"])
    Matplotlib.Plot_BoundaryConditions(simu, title="two calls", plotLegend=False)
```

______________________________________________________________________

## Reset boundary conditions

Call {py:meth}`~EasyFEA.Simulations._Simu.Bc_Init` to clear all previously defined
conditions (e.g., between load steps):

```python
simu.Bc_Init()
```

(howto-import-mesh-tags)=

## Use tags to apply boundary conditions

Tags are named node and element sets attached to the mesh — they correspond to physical
groups defined in the meshing tool (Gmsh, Medit, …). Once a mesh is imported with its
physical groups, tags are the most reliable way to identify boundaries without
hard-coding coordinates.

### List available tags

```python
# List all tag names on the main element group
print(mesh.groupElem.nodeTags)
# e.g. ['bottom', 'top', 'left', 'right', 'inlet', 'wall']
```

### Retrieve nodes or elements by tag

```python
# Nodes associated with a tag
nodes_bottom = mesh.Nodes_Tags("bottom")

# Combine multiple tags into one node set
nodes_fixed = mesh.Nodes_Tags("left", "right")

# On a non-isoparametric space, the field's nodes on a tag
nodes_bottom = mesh.Nodes_Tags("bottom", problemType=U)

# Elements associated with a tag
elems_inlet = mesh.Elements_Tags("inlet")
```

### Assign a custom tag

You can also define your own tags from a coordinate condition and reuse them later:

```python
nodes = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
mesh.Set_Tag(nodes, "clamped_end")

# later in the script
simu.add_dirichlet(mesh.Nodes_Tags("clamped_end"), [0] * dim, simu.Get_unknowns())
```

### Apply boundary conditions from tags

Node sets retrieved from tags can be passed directly to `add_dirichlet` or
`add_neumann`, exactly like node sets obtained from coordinate conditions:

```python
from EasyFEA import Simulations, Models

# fixed displacement on the "fixed" boundary
simu.add_dirichlet(mesh.Nodes_Tags("fixed"), [0, 0], ["x", "y"])

# prescribed displacement on the "top" surface
simu.add_dirichlet(mesh.Nodes_Tags("top"), [-0.1], ["y"])

# uniform pressure on the "inlet" surface
simu.add_neumann(mesh.Nodes_Tags("inlet"), [-100], ["x"])
```
