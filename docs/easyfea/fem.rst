.. _easyfea-api-fem:

fem
===

The `EasyFEA/fem/ <https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/fem>`_ module in EasyFEA is designed to provide essential tools for creating and managing finite element meshes, which are crucial for numerical simulations using the `Finite Element Method <https://en.wikipedia.org/wiki/Finite_element_method>`_ (FEM).

What is a mesh in EasyFEA ?
---------------------------

A :py:class:`~EasyFEA.fem.Mesh` object in EasyFEA represents a collection of :py:class:`~EasyFEA.fem.ElemType` used to define the geometry and structure for finite element analysis. It contains multiple :py:class:`~EasyFEA.fem._GroupElem` instances, which are groups of :py:class:`~EasyFEA.fem.ElemType` that collectively define the spatial discretization of the domain for numerical simulations.

For example, a :py:class:`~EasyFEA.fem.elems._hexa.HEXA8` mesh includes the following element types:

- :py:class:`~EasyFEA.fem.elems._point.POINT` (0D element)
- :py:class:`~EasyFEA.fem.elems._seg.SEG2` (1D element)
- :py:class:`~EasyFEA.fem.elems._quad.QUAD4` (2D element)
- :py:class:`~EasyFEA.fem.elems._hexa.HEXA8` (3D element)

All implemented element types, along with their shape functions and derivatives, are defined in the `EasyFEA/fem/elems/ <https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/fem/elems>`_ directory. The Gauss point quadratures are implemented in the `EasyFEA/fem/_gauss.py <https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/fem/_gauss.py>`_ module.

Creating or importing a Mesh
----------------------------

To construct a :py:class:`~EasyFEA.fem.Mesh` using the :py:class:`~EasyFEA.fem.Mesher`, you must first create :py:class:`~EasyFEA.geoms._Geom` objects. The :py:class:`~EasyFEA.fem.Mesher` class serves as an interface to `Gmsh <https://gmsh.info/>`_, a powerful meshing tool, and includes the following primary functions for mesh generation:

- :py:meth:`~EasyFEA.fem.Mesher.Mesh_2D`: Generates a 2D mesh.
- :py:meth:`~EasyFEA.fem.Mesher.Mesh_Extrude`: Creates a mesh by extruding a 2D shape.
- :py:meth:`~EasyFEA.fem.Mesher.Mesh_Revolve`: Generates a mesh by revolving a 2D shape around an axis.
- :py:meth:`~EasyFEA.fem.Mesher.Mesh_Import_part`: Imports a cad (e.g. .stp) part to create a mesh.
- :py:meth:`~EasyFEA.fem.Mesher.Mesh_Import_mesh`: Imports an existing gmsh mesh. EasyFEA is also linked to meshio and can be used threw the 
- :py:meth:`~EasyFEA.utilities.MeshIO._Meshio_to_EasyFEA`: Imports any kind of mesh using `meshio <https://pypi.org/project/meshio/>`_.


Detailed fem api
----------------

.. automodule:: EasyFEA.fem    
    :members:
    :private-members:
    :undoc-members:
    :imported-members:
    :show-inheritance:
   
.. automodule:: EasyFEA.fem.elems
    :members:
    :private-members:
    :undoc-members:
    :imported-members:
    :show-inheritance:
   