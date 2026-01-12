(fem)=
# fem

The [`EasyFEA/fem/`](https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/fem) module in EasyFEA provides essential tools for creating and managing finite element meshes, which are crucial for numerical simulations using the [Finite Element Method](https://en.wikipedia.org/wiki/Finite_element_method) (FEM).

---

## What is a mesh in EasyFEA?

A {py:class}`~EasyFEA.fem._mesh.Mesh` object in EasyFEA represents a collection of {py:class}`~EasyFEA.fem._utils.ElemType` used to define the geometry and structure for finite element analysis. It contains multiple {py:class}`~EasyFEA.fem._group_elem._GroupElem` instances, which are groups of {py:class}`~EasyFEA.fem._utils.ElemType` that collectively define the spatial discretization of the domain for numerical simulations.

For example, a {py:class}`~EasyFEA.fem.elems._hexa.HEXA8` mesh includes the following element types:

- {py:class}`~EasyFEA.fem.elems._point.POINT` (0D element)
- {py:class}`~EasyFEA.fem.elems._seg.SEG2` (1D element)
- {py:class}`~EasyFEA.fem.elems._quad.QUAD4` (2D element)
- {py:class}`~EasyFEA.fem.elems._hexa.HEXA8` (3D element)

All implemented element types, along with their corresponding shape functions and derivatives, are defined in the [`EasyFEA/fem/elems/`](https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/fem/elems) directory.
The Gauss point quadratures are implemented in the [`EasyFEA/fem/_gauss.py`](https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/fem/_gauss.py) module.

---

## Creating or importing a Mesh

To construct a {py:class}`~EasyFEA.fem._mesh.Mesh` using the {py:class}`~EasyFEA.fem._gmsh.Mesher`, you must first create {py:class}`~EasyFEA.geoms._Geom` objects (see {ref}`geoms` for some examples). The {py:class}`~EasyFEA.fem._gmsh.Mesher` class serves as an interface to [Gmsh](https://gmsh.info/), a powerful meshing tool, and includes the following primary functions for mesh generation:

- {py:meth}`~EasyFEA.fem._gmsh.Mesher.Mesh_2D`: Generates a 2D mesh.
- {py:meth}`~EasyFEA.fem._gmsh.Mesher.Mesh_Extrude`: Creates a mesh by extruding a 2D shape.
- {py:meth}`~EasyFEA.fem._gmsh.Mesher.Mesh_Revolve`: Generates a mesh by revolving a 2D shape around an axis.
- {py:meth}`~EasyFEA.fem._gmsh.Mesher.Mesh_Import_part`: Imports a cad (e.g. .stp) part to create a mesh.
- {py:meth}`~EasyFEA.fem._gmsh.Mesher.Mesh_Import_mesh`: Imports an existing gmsh mesh. EasyFEA is also linked to meshio and can be used through the
- {py:meth}`~EasyFEA.utilities.MeshIO.Medit_to_EasyFEA`: Imports medit mesh.
- {py:meth}`~EasyFEA.utilities.MeshIO.Gmsh_to_EasyFEA`: Imports gmsh mesh.
- {py:meth}`~EasyFEA.utilities.MeshIO.PyVista_to_EasyFEA`: Imports pyvista mesh (UnstructuredGrid or MultiBlock).
- {py:meth}`~EasyFEA.utilities.MeshIO.Ensight_to_EasyFEA`: Imports ensight mesh.

Several examples are available in {doc}`../examples/Meshes/index`.

## Detailed fem API

```{eval-rst}
.. automodule:: EasyFEA.fem.elems
    :members:
    :private-members:
    :imported-members:
    :show-inheritance:

.. automodule:: EasyFEA.fem._boundary_conditions
    :members: 
    :private-members:
    :show-inheritance:

.. automodule:: EasyFEA.fem._field
    :members: 
    :private-members:
    :show-inheritance:

.. automodule:: EasyFEA.fem._forms
    :members: 
    :private-members:
    :show-inheritance:

.. automodule:: EasyFEA.fem._gauss
    :members: 
    :private-members:
    :show-inheritance:

.. automodule:: EasyFEA.fem._gmsh
    :members: 
    :private-members:
    :show-inheritance:

.. automodule:: EasyFEA.fem._group_elem
    :members: 
    :private-members:
    :show-inheritance:
    :exclude-members: _GroupElem__Set_Elements_Tag, _GroupElem__Set_Nodes_Tag

.. automodule:: EasyFEA.fem._linalg
    :members: 
    :private-members:
    :show-inheritance:

.. automodule:: EasyFEA.fem._mesh
    :members: 
    :private-members:
    :show-inheritance:

.. automodule:: EasyFEA.fem._utils
    :members: 
    :private-members:
    :show-inheritance:
```
