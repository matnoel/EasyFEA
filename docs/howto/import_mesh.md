(howto-import-mesh)=
# Import a mesh

A **mesh** can be loaded from an external file instead of being generated from geometry.
All import paths return a {py:class}`~EasyFEA.FEM.Mesh` that can be used
directly in any {py:class}`~EasyFEA.Simulations._Simu`.
Import utilities are available in the {py:class}`~EasyFEA.FEM.Mesher` class and in the {py:mod}`EasyFEA.Utilities.MeshIO` namespace.

When a mesh file contains **physical groups** (named regions, boundaries), EasyFEA preserves them as **tags**.
Tags are the primary way to identify node and elements sets to apply boundary conditions — see {ref}`howto-import-mesh-tags`.

```{note}
A mesh currently supports **only one element group of dimension `mesh.dim`**
— the dominant volumetric (or surface) elements. Mixing element types at the
same dimension (e.g. `TETRA4` and `HEXA8` in the same 3D mesh) is not
supported. Lower-dimension groups (boundary faces, edges) can coexist and are
used for tags and Neumann conditions. Lifting this restriction would require
non-trivial changes to the simulation assembly layer.
```

---

## From a Gmsh `.msh` file

Use {py:meth}`~EasyFEA.FEM.Mesher.Mesh_Import_mesh` to load an existing Gmsh
mesh file:

```python
from EasyFEA import Display, ElemType
from EasyFEA.FEM import Mesher
from EasyFEA.Geoms import Domain

mesher = Mesher()
mesh = mesher.Mesh_Import_mesh("path/to/mesh.msh")
```

The `coef` argument scales node coordinates — useful to convert units (e.g.
`m` to `mm`):

```python
mesh = mesher.Mesh_Import_mesh("path/to/mesh.msh", coef=1000.0)
```

Set `setPhysicalGroups=True` to automatically create physical groups from the
Gmsh entities, which lets you retrieve node sets by tag afterwards:

```python
mesh = mesher.Mesh_Import_mesh("path/to/mesh.msh", setPhysicalGroups=True)
```

After importing, use {py:func}`EasyFEA.Utilities.Display.Plot_Tags` or
{py:func}`EasyFEA.Utilities.PyVista.Plot_Tags` to visualize the available tags
on the mesh:

```python
Display.Plot_Tags(mesh)
```

---

## From a CAD file (`.stp` / `.igs`)

Use {py:meth}`~EasyFEA.FEM.Mesher.Mesh_Import_part` to mesh a STEP or IGES
file directly:

```python
mesh = mesher.Mesh_Import_part("part.stp", dim=3, meshSize=5.0)
```

Only triangular (`TRI3`) or tetrahedral (`TETRA4`) elements are supported for
CAD import. Use `refineGeoms` to locally refine the mesh around specific
geometric regions:

```python
refine = Domain((0, 0, 0), (10, 10, 10), meshSize=1.0)
mesh = mesher.Mesh_Import_part("part.stp", dim=3, meshSize=5.0, refineGeoms=[refine])
```

---

## From a meshio-compatible format

```{note}
The `MeshIO` module requires the `io` optional dependency:

~~~bash
pip install EasyFEA[io]
~~~
```

Use {py:func}`~EasyFEA.Utilities.MeshIO.Gmsh_to_EasyFEA` after loading with `meshio` directly, or use the dedicated converters:

```python
from EasyFEA.Utilities import MeshIO

# From a Medit .mesh file
mesh = MeshIO.Medit_to_EasyFEA("path/to/mesh.mesh")

# From a Gmsh .msh file (alternative to Mesher.Mesh_Import_mesh)
mesh = MeshIO.Gmsh_to_EasyFEA("path/to/mesh.msh")
```

---

## From PyVista

Use {py:func}`~EasyFEA.Utilities.MeshIO.PyVista_to_EasyFEA` to convert a
PyVista `UnstructuredGrid` or `MultiBlock`:

```python
import pyvista as pv
from EasyFEA.Utilities import MeshIO

pv_mesh = pv.read("path/to/mesh.vtk")
mesh = MeshIO.PyVista_to_EasyFEA(pv_mesh)
```

---

## From an Ensight file

```python
from EasyFEA.Utilities import MeshIO

mesh = MeshIO.Ensight_to_EasyFEA("path/to/mesh.geo")
```

---

## Export a mesh

EasyFEA meshes can also be exported to external formats for use in other tools:

```python
from EasyFEA.Utilities import MeshIO

# To Gmsh .msh
MeshIO.EasyFEA_to_Gmsh(mesh, folder="output/", name="mesh")

# To Medit .mesh
MeshIO.EasyFEA_to_Medit(mesh, folder="output/", name="mesh")

# To Ensight .geo
MeshIO.EasyFEA_to_Ensight(mesh, folder="output/", name="mesh")

# To PyVista UnstructuredGrid
pv_mesh = MeshIO.EasyFEA_to_PyVista(mesh)
```

---

## Geometric operations

EasyFEA meshes support in-place geometric transformations directly on the
{py:class}`~EasyFEA.FEM.Mesh` object:

```python
# Translate by (dx, dy, dz)
mesh.Translate(dx=10.0, dy=0.0, dz=0.0)

# Rotate by theta degrees around an axis
mesh.Rotate(theta=90, center=(0, 0, 0), direction=(0, 0, 1))  # 90° around z-axis

# Symmetry with respect to a plane defined by a point and its normal
mesh.Symmetry(point=(0, 0, 0), n=(1, 0, 0))  # mirror across the y-z plane
```

All three operations modify the mesh **in place** and update every element
group consistently. They are useful when composing assemblies: transform
individual parts before merging them.

---

## Merge meshes

To combine several meshes into one (e.g. assemblies with multiple parts):

```python
merged = MeshIO.Merge([mesh1, mesh2, mesh3])
```

Set `constructUniqueElements=False` to skip deduplication of shared elements if the meshes are already conforming.