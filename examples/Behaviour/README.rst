.. _easyfea-examples-behaviour:

Materials with history
======================

``Models.Behaviour`` is assembled from independent pieces rather than chosen from a list:

.. code-block:: python

    Models.Behaviour(
        3,
        elastic      = Isotropic(3, E=210000, v=0.3),
        yieldSurface = Models.Yield.VonMises(250.0),
        hardening    = Models.IsotropicHardening.Voce(150.0, 30.0),
        kinematic    = Models.KinematicHardening.Chaboche((60000, 500), (2000, 0)),
        rate         = Models.ViscoPlastic.Norton(1e-2),
        branches     = [Models.ViscoElastic.Maxwell(g=0.3, tau=1.0)],
    )

Every piece is optional. With none of them it is linear elasticity; with a yield surface it is
plasticity; add a rate law and it creeps; add Maxwell branches and it relaxes without yielding.
Damage is deliberately absent: local softening is mesh-dependent past the peak, which is what
``Simulations.PhaseField`` exists for.

Because hardening lives in the **free energy** rather than inside the yield surface, any
hardening law composes with any surface: three hardening laws and three surfaces are six
objects, not nine.

Two solvers produce the stress and the consistent tangent, and the material picks between them.
A quadratic yield surface reduces the local problem to a single scalar unknown, solved in the
material eigenspace; everything else goes to an implicit solve over the full state. They agree
to machine precision wherever both apply, which is a test. ``solver="newton"`` forces the
general one.

- ``StressStrain`` — uniaxial curves for each hardening law, each against its closed form, and
  one hardening law on three different yield surfaces. Uses ``Models.MaterialPoint``, which
  drives a behaviour at a single Gauss point with no mesh and no solver — the same code path
  assembly uses, so it is the real behaviour and not a second implementation of it.
- ``ThickCylinder`` — a **verification**: the plastic zone spreading from the bore of a
  pressurised cylinder, against Hill's closed form, with a mesh-convergence study.
- ``BeamBending`` — a second **verification**: moment-curvature of a rectangular section against
  Chakrabarty, and the shape factor 3/2.
- ``TensileTest`` — a tensile specimen pulled past yield in 3D, and a **verification** of a
  different kind: rotated into a general orientation it must give the identical answer, which
  tests frame indifference of the strain computation, the return mapping, the tangent and
  assembly all at once. Use quadratic elements: plastic flow is incompressible, and fully
  integrated linear hexahedra lock under it.
