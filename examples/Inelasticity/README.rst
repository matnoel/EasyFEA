.. _easyfea-examples-behaviour:

Materials with history
======================

Scripts that demonstrate small-strain materials whose stress depends on the history of strain:
plasticity, viscoplasticity and viscoelasticity.

:py:class:`~EasyFEA.Models.Behaviour` is assembled from independent pieces rather than chosen
from a list, and driven by :py:class:`~EasyFEA.Simulations.Behaviour`. Each piece is optional and
turns on one behaviour:

.. list-table::
    :header-rows: 1

    * - Piece
      - What it adds
    * - *(none)*
      - linear elasticity
    * - :py:class:`Models.Yield <EasyFEA.Models.Yield>`
      - plasticity — von Mises, Hill, Drucker-Prager
    * - :py:class:`Models.IsotropicHardening <EasyFEA.Models.IsotropicHardening>`
      - the surface grows — linear, Voce, Swift
    * - :py:class:`Models.KinematicHardening <EasyFEA.Models.KinematicHardening>`
      - the surface moves, giving the Bauschinger effect — Prager, Armstrong-Frederick, Chaboche
    * - :py:class:`Models.ViscoPlastic <EasyFEA.Models.ViscoPlastic>`
      - it creeps and relaxes once yielded — Norton, Perzyna
    * - :py:class:`Models.ViscoElastic <EasyFEA.Models.ViscoElastic>`
      - it relaxes without ever yielding — Maxwell branches

Because hardening lives in the free energy rather than inside the yield surface, any hardening
law composes with any surface. Damage is out of scope and
:py:class:`~EasyFEA.Simulations.PhaseField` handles it; so is finite strain.

Several scripts use :py:class:`~EasyFEA.Models.MaterialPoint`, which drives a behaviour at a
single Gauss point with no mesh and no solver. The two solvers behind it are described in
:ref:`simulations-behaviour`.
