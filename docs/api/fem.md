(fem)=
# FEM

The {py:mod}`EasyFEA.FEM` module in EasyFEA provides essential tools for creating and managing finite element meshes, which are crucial for numerical simulations using the [Finite Element Method](https://en.wikipedia.org/wiki/Finite_element_method) (FEM).

---

## What is a mesh in EasyFEA?

A {py:class}`~EasyFEA.FEM.Mesh` object in EasyFEA represents a collection of {py:class}`~EasyFEA.FEM.ElemType` used to define the geometry and structure for finite element analysis. It contains multiple {py:class}`~EasyFEA.FEM._GroupElem` instances, which are groups of {py:class}`~EasyFEA.FEM.ElemType` that collectively define the spatial discretization of the domain for numerical simulations.

For example, a {py:class}`~EasyFEA.FEM.Elems.HEXA8` mesh includes the following element types:

- {py:class}`~EasyFEA.FEM.Elems.POINT` (0D element)
- {py:class}`~EasyFEA.FEM.Elems.SEG2` (1D element)
- {py:class}`~EasyFEA.FEM.Elems.QUAD4` (2D element)
- {py:class}`~EasyFEA.FEM.Elems.HEXA8` (3D element)

All implemented element types, along with their corresponding shape functions and derivatives, are defined in the {py:mod}`EasyFEA.FEM.Elems` module.
The Gauss point quadratures are implemented in the {py:class}`~EasyFEA.FEM.Gauss` class.

---

## Creating or importing a Mesh

To construct a {py:class}`~EasyFEA.FEM.Mesh` using the {py:class}`~EasyFEA.FEM.Mesher`, you must first create {py:class}`~EasyFEA.Geoms._Geom` objects (see {ref}`geoms` for some examples). The {py:class}`~EasyFEA.FEM.Mesher` class serves as an interface to [Gmsh](https://gmsh.info/), a powerful meshing tool, and includes the following primary functions for mesh generation:

- {py:meth}`~EasyFEA.FEM.Mesher.Mesh_2D`: Generates a 2D mesh.
- {py:meth}`~EasyFEA.FEM.Mesher.Mesh_Extrude`: Creates a mesh by extruding a 2D shape.
- {py:meth}`~EasyFEA.FEM.Mesher.Mesh_Revolve`: Generates a mesh by revolving a 2D shape around an axis.
- {py:meth}`~EasyFEA.FEM.Mesher.Mesh_Import_part`: Imports a cad (e.g. .stp) part to create a mesh.
- {py:meth}`~EasyFEA.FEM.Mesher.Mesh_Import_mesh`: Imports an existing gmsh mesh. EasyFEA is also linked to meshio and can be used through the
- {py:meth}`~EasyFEA.Utilities.MeshIO.Medit_to_EasyFEA`: Imports medit mesh.
- {py:meth}`~EasyFEA.Utilities.MeshIO.Gmsh_to_EasyFEA`: Imports gmsh mesh.
- {py:meth}`~EasyFEA.Utilities.MeshIO.PyVista_to_EasyFEA`: Imports pyvista mesh (UnstructuredGrid or MultiBlock).
- {py:meth}`~EasyFEA.Utilities.MeshIO.Ensight_to_EasyFEA`: Imports ensight mesh.

Several examples are available in {doc}`../examples/Meshes/index`.

## FEM API

```{eval-rst}
.. automodule:: EasyFEA.FEM
.. automodule:: EasyFEA.FEM.Elems
```