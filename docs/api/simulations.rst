.. _easyfea-api-simulations:

simulations
===========

The `EasyFEA/simulations/ <https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/simulations>`_ module in EasyFEA provides essential tools for creating and managing simulations.
These simulations are built using a :py:class:`~EasyFEA.fem.Mesh` and a :py:class:`~EasyFEA.Models._IModel` (material).

With this module, you can construct:

+ Linear elastic simulations with :py:class:`~EasyFEA.Simulations.ElasticSimu`.
+ Nonlinear hyperelastic simulations with :py:class:`~EasyFEA.Simulations.HyperElasticSimu`.
+ Euler-Bernoulli beam simulations with :py:class:`~EasyFEA.Simulations.BeamSimu`.
+ PhaseField damage simulations for quasi-static brittle fracture with :py:class:`~EasyFEA.Simulations.PhaseFieldSimu`.
+ Thermal simulations with :py:class:`~EasyFEA.Simulations.ThermalSimu`.

How to create new simulations in EasyFEA ?
------------------------------------------

To create new simulation classes, you can take inspiration from existing implementations.  
Make sure to follow the :py:class:`~EasyFEA.simulations._Simu` interface.  
The :py:class:`~EasyFEA.Simulations.ThermalSimu` class is relatively simple and can serve as a good starting point.  
See `EasyFEA/simulations/_thermal.py <https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/simulations/_thermal.py>`_ for more details.

Detailed simulations API
------------------------

.. automodule:: EasyFEA.simulations
   :members:
   :private-members: _Simu
   :undoc-members:
   :imported-members:
   :show-inheritance:

.. automodule:: EasyFEA.Simulations
    :members: ElasticSimu, HyperElasticSimu, BeamSimu, PhaseFieldSimu, ThermalSimu

   