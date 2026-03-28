(howto-boundary-conditions)=
# Apply boundary conditions

**Boundary conditions** prescribe loads and constraints on the degrees of
freedom of a {py:class}`~EasyFEA.Simulations._Simu`. They are applied to
**node sets** via `add_*` methods on the simulation object and are accessible
in the {py:mod}`EasyFEA.Simulations` namespace. Node sets come from coordinate
conditions or from mesh tags — see {ref}`howto-import-mesh-tags` for the
tag-based approach.

---

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

Pass these strings as the `unknowns` argument to every `add_*` method.
You can also pass a **subset** — for example, fix only `["x"]` to constrain
horizontal motion while leaving the vertical DOF free.

---

## Summary of BC methods

| Method | Physical meaning | Integration |
|---|---|---|
| {py:meth}`~EasyFEA.Simulations._Simu.add_dirichlet` | Prescribed DOF value (displacement, temperature, …) | — |
| {py:meth}`~EasyFEA.Simulations._Simu.add_neumann` | Concentrated nodal force / flux | Point |
| {py:meth}`~EasyFEA.Simulations._Simu.add_lineLoad` | Force per unit length | Along a line |
| {py:meth}`~EasyFEA.Simulations._Simu.add_surfLoad` | Force per unit area (pressure in 2D, traction in 3D) | Over a surface |
| {py:meth}`~EasyFEA.Simulations._Simu.add_volumeLoad` | Body force per unit volume | Over a volume |
| {py:meth}`~EasyFEA.Simulations._Simu.add_pressureLoad` | Normal pressure (outward positive) | Over a boundary surface |

---

## Dirichlet conditions ({py:meth}`~EasyFEA.Simulations._Simu.add_dirichlet`)

Prescribes the value of one or more DOFs on a set of nodes.

```python
# fix all displacement DOFs to zero — clamped wall (Elastic simulation)
nodes = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
simu.add_dirichlet(nodes, [0, 0], ["x", "y"])

# impose temperature 100 °C on the top surface (Thermal simulation)
nodes_top = mesh.Nodes_Conditions(lambda x, y, z: y == H)
simu.add_dirichlet(nodes_top, [100.0], ["t"])

# prescribed displacement that varies along x
simu.add_dirichlet(nodes, [lambda x, y, z: 0.01 * x, 0], ["x", "y"])
```

---

## Concentrated force ({py:meth}`~EasyFEA.Simulations._Simu.add_neumann`)

Applies a concentrated force (or flux) directly on the selected nodes. No
integration is performed — the value is added as-is to the right-hand side.
Useful for point loads on beams or reactions at a single node.

```python
# point load at the beam tip (Elastic simulation)
nodes_tip = mesh.Nodes_Point((L, h / 2))
simu.add_neumann(nodes_tip, [0, -500], ["x", "y"])
```

---

## Distributed loads

### Line load ({py:meth}`~EasyFEA.Simulations._Simu.add_lineLoad`)

Force per unit **length** integrated along the selected boundary edges. Used
in 2D for loads applied along a line, or in 3D for edge loads.

```python
# uniform downward load of −1000 N/m along the top edge
nodes_top = mesh.Nodes_Conditions(lambda x, y, z: y == H)
simu.add_lineLoad(nodes_top, [-1000], ["y"])

# spatially varying line load
simu.add_lineLoad(nodes_top, [lambda x, y, z: -500 * (1 + x / L)], ["y"])
```

### Surface load ({py:meth}`~EasyFEA.Simulations._Simu.add_surfLoad`)

Force per unit **area** integrated over the selected boundary faces. Typical
for pressure loads in 3D, or traction/pressure in 2D (force per unit area
× thickness).

```python
# uniform pressure of −800 Pa in x on the right face
nodes_right = mesh.Nodes_Conditions(lambda x, y, z: x == L)
simu.add_surfLoad(nodes_right, [-800], ["x"])

# traction vector on a 3D face
simu.add_surfLoad(nodes_right, [-1e6], ["z"])
```

### Volume load ({py:meth}`~EasyFEA.Simulations._Simu.add_volumeLoad`)

Body force per unit **volume** integrated over the selected elements. Typical
use is gravity.

```python
# gravity in −y, ρ = 7800 kg/m³
simu.add_volumeLoad(mesh.nodes, [-7800 * 9.81], ["y"])
```

### Pressure load ({py:meth}`~EasyFEA.Simulations._Simu.add_pressureLoad`)

Applies a **normal** pressure of given magnitude on a boundary surface.
The direction is automatically computed from the outward normal at each
boundary face.

```python
# internal pressure of 1 MPa on a cylinder inner surface
simu.add_pressureLoad(inner_surface_nodes, magnitude=1e6)
```

---

## Spatially varying values

All `values` arguments accept:

- a **float** — uniform value applied to all selected nodes,
- a **NumPy array** of length `Nn` — one value per node,
- a **lambda function** `lambda x, y, z: ...` — evaluated at node (or
  integration-point) coordinates.

```python
# temperature that increases linearly along x
simu.add_dirichlet(nodes, [lambda x, y, z: 20 + 80 * x / L], ["t"])

# load proportional to distance from center
simu.add_surfLoad(nodes, [lambda x, y, z: -100 * (x - L/2)**2], ["x"])
```

Functions always receive three arguments `x, y, z` regardless of the problem
dimension.

---

## Visualize applied conditions

```python
from EasyFEA import Display, PyVista

Display.Plot_BoundaryConditions(simu)   # matplotlib
PyVista.Plot_BoundaryConditions(simu)   # interactive 3D
```

---

## Reset boundary conditions

Call {py:meth}`~EasyFEA.Simulations._Simu.Bc_Init` to clear all previously defined conditions (e.g., between load steps):

```python
simu.Bc_Init()
```

(howto-import-mesh-tags)=
## Use tags to apply boundary conditions

Tags are named node and element sets attached to the mesh — they correspond to physical groups defined in the meshing tool (Gmsh, Medit, …).
Once a mesh is imported with its physical groups, tags are the most reliable way to identify boundaries without hard-coding coordinates.

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

# Elements associated with a tag
elems_inlet = mesh.Elements_Tags("inlet")
```

### Assign a custom tag

You can also define your own tags from a coordinate condition and reuse them
later:

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
