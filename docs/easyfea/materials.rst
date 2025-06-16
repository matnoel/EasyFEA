.. _easyfea-api-materials:

materials
=========

The `EasyFEA/materials/ <https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/materials>`_ module in EasyFEA provides essential tools for creating and managing materials.
These models are used to build :py:class:`~EasyFEA.simulations._Simu` instances and mainly contain material parameters.


With this module, you can construct:

+ Linear elastic materials, such as :py:class:`~EasyFEA.materials.ElasIsot`, :py:class:`~EasyFEA.materials.ElasIsotTrans`, and :py:class:`~EasyFEA.materials.ElasAnisot`.
+ Nonlinear hyperelastic materials, such as :py:class:`~EasyFEA.materials.NeoHookean`, :py:class:`~EasyFEA.materials.MooneyRivlin`, and :py:class:`~EasyFEA.materials.SaintVenantKirchhoff`.
+ Elastic beams with :py:class:`~EasyFEA.materials.BeamElasIsot`.
+ Phase-field materials with :py:class:`~EasyFEA.materials.PhaseField`.
+ Thermal materials with :py:class:`~EasyFEA.materials.Thermal`.


Detailed materials API
----------------------

.. automodule:: EasyFEA.Materials
    
.. automodule:: EasyFEA.materials
   :members:
   :private-members: _IModel
   :undoc-members:
   :imported-members:
   :show-inheritance:

