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

| Simulation      | Unknowns                                                 |
| --------------- | -------------------------------------------------------- |
| Elastic 2D      | `['x', 'y']`                                             |
| Elastic 3D      | `['x', 'y', 'z']`                                        |
| HyperElastic 2D | `['x', 'y']`                                             |
| Thermal         | `['t']`                                                  |
| Beam (planar)   | `['x', 'y', 'rz']`                                       |
| PhaseField      | `['x', 'y']` for elastic sub-problem, `['d']` for damage |

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

A scalar unknown has no direction, so it stays a marker whatever its value, here 100 °C
on the top of a thermal simulation:

```{eval-rst}
.. jupyter-execute::

    nodesTop = mesh.Nodes_Conditions(lambda x, y, z: y == H)
    simuThermal = Simulations.Thermal(mesh, Models.Thermal(k=1.0, c=1.0))
    simuThermal.add_dirichlet(nodesTop, [100.0], ["t"])
    Matplotlib.Plot_BoundaryConditions(simuThermal, title="temperature")
```

______________________________________________________________________

## Concentrated force with {py:meth}`~EasyFEA.Simulations._Simu.add_neumann`

Applies a concentrated force (or flux) directly on the selected nodes. No integration is
performed — the value is added as-is to the right-hand side. Useful for point loads on
beams or reactions at a single node.

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
loads applied along a line, or in 3D for edge loads. Here −1000 N/m downward along the
top edge:

```{eval-rst}
.. jupyter-execute::

    simu.Bc_Init()
    simu.add_lineLoad(nodesTop, [-1000], ["y"])
    Matplotlib.Plot_BoundaryConditions(simu, title="add_lineLoad")
```

### Surface load with {py:meth}`~EasyFEA.Simulations._Simu.add_surfLoad`

Force per unit **area** integrated over the selected boundary faces. Typical for
pressure loads in 3D, or traction/pressure in 2D (force per unit area × thickness). Here
−800 Pa along x on the right side:

```{eval-rst}
.. jupyter-execute::

    simu.Bc_Init()
    simu.add_surfLoad(nodesRight, [-800], ["x"])
    Matplotlib.Plot_BoundaryConditions(simu, title="add_surfLoad")
```

### Volume load with {py:meth}`~EasyFEA.Simulations._Simu.add_volumeLoad`

Body force per unit **volume** integrated over the selected elements. Typical use is
gravity, here in −y with ρ = 7800 kg/m³. A condition on every node is drawn once, at the
center of the mesh:

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

Functions always receive three arguments `x, y, z` regardless of the problem dimension.

An array needs one value per selected node, or it raises. A function has no such
constraint, whichever nodes the selection holds.

```{eval-rst}
.. jupyter-execute::

    simu.Bc_Init()
    simu.add_lineLoad(nodesTop, [lambda x, y, z: -500 * (1 + x / L)], ["y"])
    Matplotlib.Plot_BoundaryConditions(simu, title="a function")

    simu.Bc_Init()
    simu.add_dirichlet(nodesTop, [0.1 * mesh.coord[nodesTop, 0]], ["y"])
    Matplotlib.Plot_BoundaryConditions(simu, title="an array")
```

______________________________________________________________________

(howto-check-selection)=

## Visualize applied conditions

{py:func}`~EasyFEA.Viz.Matplotlib.Plot_BoundaryConditions`, or
{py:func}`~EasyFEA.Viz.PyVista.Plot_BoundaryConditions` for an interactive 3D view, draw
every selected node, those a space adds included, so they show a node picked by mistake.
Only a condition on every node is drawn once, at the center. On a strip one element
thick with P2 on TRI3, selecting the top and the bottom in one call also takes the
mid-edge nodes of the end sides, whose two vertices are selected (see
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
conditions (e.g., between load steps), as the figures above do.

(howto-import-mesh-tags)=

## Use tags to apply boundary conditions

Tags are named node and element sets attached to the mesh — they correspond to physical
groups defined in the meshing tool (Gmsh, Medit, …). Once a mesh is imported with its
physical groups, tags are the most reliable way to identify boundaries without
hard-coding coordinates.

### Show the available tags

{py:func}`~EasyFEA.Viz.Matplotlib.Plot_Tags` draws the tags of a mesh with their names,
as does {py:func}`~EasyFEA.Viz.PyVista.Plot_Tags`. A mesh built from a geometry names
its points, lines, surfaces and volumes `P0`, `L0`, `S0`, `V0`, and so on; an imported
mesh keeps the names of its physical groups. By default it names the mesh's two highest
dimensions: surfaces and lines in 2D, volumes and surfaces in 3D. `dimElems` picks
others, and the lines are always drawn. In 3D everything is see-through.

```{eval-rst}
.. jupyter-execute::

    circle = Circle(Point(L / 2, H / 2), 0.3, meshSize=0.1, isFilled=True)
    domain = Domain((0, 0), (L, H), meshSize=0.1)
    mesh = domain.Mesh_2D([circle], ElemType.TRI3)

    Matplotlib.Plot_Tags(mesh, title="by default")
    Matplotlib.Plot_Tags(mesh, useColorCycler=True, title="useColorCycler=True")
    Matplotlib.Plot_Tags(mesh, dimElems=[1, 0], title="dimElems=[1, 0]")
    Matplotlib.Plot_Tags(mesh, showId=False, title="showId=False")
```

Both viewers take the same options:

| Option              | Effect                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------ |
| `dimElems`          | the dimensions to draw and name, by default the mesh's two highest; lines are always drawn |
| `useColorCycler`    | one colour per tag and its name on the highest surfaces or volumes drawn                   |
| `showId`            | writes the names, on by default                                                            |
| `alpha`             | opacity of the surfaces and volumes                                                        |
| `linewidth`         | width of the tagged lines                                                                  |
| `deformFactor`      | draws a simulation's mesh deformed                                                         |
| `bounds`            | fixed view box `(xmin, xmax, ymin, ymax, zmin, zmax)`                                      |
| `title`, `showGrid` | title and grid                                                                             |

Matplotlib also takes `ax` and `folder`, PyVista `plotter`.

Each element group holds the tags of its own dimension:

```{eval-rst}
.. jupyter-execute::

    print(mesh.groupElem.nodeTags)
    print(mesh.Get_list_groupElem(mesh.dim - 1)[0].nodeTags)
```

### Apply boundary conditions from tags

{py:meth}`~EasyFEA.FEM.Mesh.Nodes_Tags` returns the nodes of one or several tags, which
go to any `add_*` method like nodes from a coordinate condition. On a non-isoparametric
space, `problemType=` returns the field's nodes on them.
{py:meth}`~EasyFEA.FEM.Mesh.Elements_Tags` returns the elements of a tag.

```{eval-rst}
.. jupyter-execute::

    simu = Simulations.Elastic(mesh, material)
    nodesLeft = mesh.Nodes_Tags("L3")
    nodesRollers = mesh.Nodes_Tags("L0", "L2")
    nodesRight = mesh.Nodes_Tags("L1")
    simu.add_dirichlet(nodesLeft, [0, 0], ["x", "y"])
    simu.add_dirichlet(nodesRollers, [0], ["y"])
    simu.add_lineLoad(nodesRight, [1000], ["x"])
    Matplotlib.Plot_BoundaryConditions(simu, title="from tags")
```

### Assign a custom tag

{py:meth}`~EasyFEA.FEM.Mesh.Set_Tag` names the elements a selection holds, here the
upper half of the left side, and {py:func}`~EasyFEA.Viz.Matplotlib.Plot_Tags` then draws
it:

```{eval-rst}
.. jupyter-execute::

    nodes = mesh.Nodes_Conditions(lambda x, y, z: (x == 0) & (y >= H / 2))
    mesh.Set_Tag(nodes, "clamped")
    Matplotlib.Plot_Tags(mesh, title="Set_Tag")

    simu.Bc_Init()
    nodesClamped = mesh.Nodes_Tags("clamped")
    simu.add_dirichlet(nodesClamped, [0, 0], ["x", "y"])
    Matplotlib.Plot_BoundaryConditions(simu, title="clamped")
```
