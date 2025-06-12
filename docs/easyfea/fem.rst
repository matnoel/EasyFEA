.. _easyfea-api-fem:

fem
===

Within the `EasyFEA/fem/ <https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/fem>`_ module, you will find the following key components:

.. autosummary::

   ~EasyFEA.fem.ElemType
   ~EasyFEA.fem._GroupElem
   ~EasyFEA.fem.Mesher
   ~EasyFEA.fem.Mesh

A :py:class:`~EasyFEA.fem.Mesh` object contains multiple :py:class:`~EasyFEA.fem._GroupElem` instances. For example, a :py:class:`~EasyFEA.fem.elems._hexa.HEXA8` mesh includes the following element types:

- :py:class:`~EasyFEA.fem.elems._point.POINT` (0D element)
- :py:class:`~EasyFEA.fem.elems._seg.SEG2` (1D element)
- :py:class:`~EasyFEA.fem.elems._quad.QUAD4` (2D element)
- :py:class:`~EasyFEA.fem.elems._hexa.HEXA8` (3D element)

All implemented element types, along with their shape functions and derivatives, are defined in the `EasyFEA/fem/elems/ <https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/fem/elems>`_ directory. The Gauss point quadratures are implemented in the `EasyFEA/fem/_gauss.py <https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/fem/_gauss.py>`_ module.

To construct a :py:class:`~EasyFEA.fem.Mesh` using the :py:class:`~EasyFEA.fem.Mesher`, you must first create :py:class:`~EasyFEA.geoms._Geom` objects. The :py:class:`~EasyFEA.fem.Mesher` serves as a Gmsh wrapper that operates on :py:class:`~EasyFEA.geoms._Geom` objects, with the following main functions:

- :py:meth:`~EasyFEA.fem.Mesher.Mesh_2D`
- :py:meth:`~EasyFEA.fem.Mesher.Mesh_Extrude`
- :py:meth:`~EasyFEA.fem.Mesher.Mesh_Revolve`
- :py:meth:`~EasyFEA.fem.Mesher.Mesh_Import_part`
- :py:meth:`~EasyFEA.fem.Mesher.Mesh_Import_mesh`
 


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
   