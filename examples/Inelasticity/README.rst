.. _easyfea-examples-inelasticity:

Inelastic simulations
=====================

Scripts that demonstrate small-strain materials whose stress depends on the history of strain:
plasticity, viscoplasticity and viscoelasticity.

Each material is a :py:class:`~EasyFEA.Models.InElastic._Behavior`, written as one ``Update`` and
one ``Stress`` at one 3D material point and driven by :py:class:`~EasyFEA.Simulations.InElastic`, which derives the
tangent and handles 2D. The shipped ones:

.. list-table::
    :header-rows: 1

    * - Behavior
      - What it models
    * - :py:class:`~EasyFEA.Models.InElastic.Plasticity`
      - plasticity, on any surface from :py:mod:`~EasyFEA.Models.InElastic.Yield` (von Mises,
        Hill, Drucker-Prager) with any hardening from
        :py:mod:`~EasyFEA.Models.InElastic.IsotropicHardening` (linear, Voce, Swift)
    * - :py:class:`~EasyFEA.Models.InElastic.Norton`
      - it creeps and relaxes once yielded
    * - :py:class:`~EasyFEA.Models.InElastic.Chaboche`
      - the surface moves, giving the Bauschinger effect — Prager, Armstrong-Frederick and their
        superposition
    * - :py:class:`~EasyFEA.Models.InElastic.Maxwell`
      - viscoelastic relaxation

Several scripts use :py:class:`~EasyFEA.Models.InElastic.MaterialPoint`, which drives a behavior at a
single point with no mesh and no solver.
