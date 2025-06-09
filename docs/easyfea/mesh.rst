.. _easyfea-api-mesh:

Mesh
====

A :py:class:`~EasyFEA.fem.Mesh` object contains multiple :py:class:`~EasyFEA.fem._GroupElem` instances. For example, a :py:class:`~EasyFEA.fem.elems._hexa.HEXA8` mesh includes the following element types:

+ :py:class:`~EasyFEA.fem.elems._hexa.POINT` (dimension 0)
+ :py:class:`~EasyFEA.fem.elems._hexa.SEG2` (dimension 1)
+ :py:class:`~EasyFEA.fem.elems._hexa.QUAD4` (dimension 2)
+ :py:class:`~EasyFEA.fem.elems._hexa.HEXA8` (dimension 3)

.. autosummary::

   ~EasyFEA.fem.ElemType
   ~EasyFEA.fem._GroupElem
   ~EasyFEA.fem.Mesher
   ~EasyFEA.fem.Mesh

All implemented element types, along with their respective shape functions and derivative shape functions, are available in the `EasyFEA/fem/elems/ <https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/fem/elems>`_ folder. The Gauss point quadratures are defined in the `EasyFEA/fem/_gauss.py <https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/fem/_gauss.py>`_ Python script.


**Detailed API Reference**

.. autoclass:: EasyFEA.fem.ElemType
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: EasyFEA.fem._GroupElem
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: EasyFEA.fem.Mesher
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: EasyFEA.fem.Mesh
   :members:
   :undoc-members:
   :show-inheritance: