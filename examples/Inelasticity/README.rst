.. _easyfea-examples-inelasticity:

Materials with history
======================

Scripts that demonstrate small-strain materials whose stress depends on the history of strain:
plasticity, viscoplasticity and viscoelasticity.

:py:class:`~EasyFEA.Models.InElastic.Behavior` is assembled from independent pieces rather than chosen
from a list, and driven by :py:class:`~EasyFEA.Simulations.InElastic`. Each piece is optional and
turns on one behavior:

.. list-table::
    :header-rows: 1

    * - Piece
      - What it adds
    * - *(none)*
      - linear elasticity
    * - :py:class:`Models.InElastic.Yield <EasyFEA.Models.InElastic.Yield>`
      - plasticity — von Mises, Hill, Drucker-Prager
    * - :py:class:`Models.InElastic.IsotropicHardening <EasyFEA.Models.InElastic.IsotropicHardening>`
      - the surface grows — linear, Voce, Swift
    * - :py:class:`Models.InElastic.KinematicHardening <EasyFEA.Models.InElastic.KinematicHardening>`
      - the surface moves, giving the Bauschinger effect — Prager, Armstrong-Frederick, Chaboche
    * - :py:class:`Models.InElastic.ViscoPlastic <EasyFEA.Models.InElastic.ViscoPlastic>`
      - it creeps and relaxes once yielded — Norton, Perzyna
    * - :py:class:`Models.InElastic.ViscoElastic <EasyFEA.Models.InElastic.ViscoElastic>`
      - it relaxes without ever yielding — Maxwell branches

Several scripts use :py:class:`~EasyFEA.Models.InElastic.MaterialPoint`, which drives a behaviour at a
single Gauss point with no mesh and no solver. The two solvers behind it are described in
:ref:`simulations-inelastic`.
