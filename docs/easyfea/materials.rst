.. _easyfea-api-materials:

materials
=========

The `EasyFEA/materials/ <https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/materials>`_ module in EasyFEA is designed to provide essential tools for creating and managing material objects. These material models are used to construct :py:class:`~EasyFEA.simulations._Simu` instances and primarily contain material parameters.

With this module, you can construct:

+ Linear elastic materials, such as :py:class:`~EasyFEA.materials.Elas_Isot`, :py:class:`~EasyFEA.materials.Elas_IsotTrans`, and :py:class:`~EasyFEA.materials.Elas_Anisot`.
+ Nonlinear hyperelastic materials, such as :py:class:`~EasyFEA.materials.NeoHookean`, :py:class:`~EasyFEA.materials.MooneyRivlin`, and :py:class:`~EasyFEA.materials.SaintVenantKirchhoff`.
+ Elastic beams with :py:class:`~EasyFEA.materials.Beam_Elas_Isot`.
+ Phase-field materials with :py:class:`~EasyFEA.materials.PhaseField`.
+ Thermal materials with :py:class:`~EasyFEA.materials.Thermal`.


Detailed materials api
----------------------

.. automodule:: EasyFEA.materials
   :members:
   :private-members: _IModel
   :undoc-members:
   :imported-members:
   :show-inheritance:
   