(models)=
# Models

The {py:mod}`EasyFEA.Models` module provides essential tools for creating and managing models.
These models are used to build {py:class}`~EasyFEA.Simulations._Simu` instances and mainly contain material parameters.

In the simulation workflow, `Models` is the **third step**: once the mesh exists, a model encapsulates the physics and material constants (Young's modulus, thermal conductivity, fracture toughness, …) before being passed to the simulation constructor.


With this module, you can construct:

(models-elastic)=
+ Linear elastic materials, such as {py:class}`~EasyFEA.Models.Elastic.Isotropic`, {py:class}`~EasyFEA.Models.Elastic.TransverselyIsotropic`, {py:class}`~EasyFEA.Models.Elastic.Orthotropic`, and {py:class}`~EasyFEA.Models.Elastic.Anisotropic`, in {py:class}`Models.Elastic <EasyFEA.Models.Elastic>`.
(models-hyperelastic)=
+ Nonlinear hyperelastic materials, such as {py:class}`~EasyFEA.Models.HyperElastic.NeoHookean`, {py:class}`~EasyFEA.Models.HyperElastic.CiarletGeymonat`, {py:class}`~EasyFEA.Models.HyperElastic.MooneyRivlin`, {py:class}`~EasyFEA.Models.HyperElastic.SaintVenantKirchhoff`, and {py:class}`~EasyFEA.Models.HyperElastic.HolzapfelOgden`, in {py:class}`Models.HyperElastic <EasyFEA.Models.HyperElastic>`.
(models-behaviour)=
+ Small-strain materials whose stress depends on the *history* of strain — plasticity, viscoplasticity and viscoelasticity — with {py:class}`~EasyFEA.Models.Behaviour`. It is assembled from independent pieces rather than chosen from a list: an elastic law, plus any of a yield surface from {py:class}`Models.Yield <EasyFEA.Models.Yield>`, isotropic hardening from {py:class}`Models.IsotropicHardening <EasyFEA.Models.IsotropicHardening>`, a back-stress from {py:class}`Models.KinematicHardening <EasyFEA.Models.KinematicHardening>`, a rate law from {py:class}`Models.ViscoPlastic <EasyFEA.Models.ViscoPlastic>`, and Maxwell branches from {py:class}`Models.ViscoElastic <EasyFEA.Models.ViscoElastic>`. With none of them it is linear elasticity. {py:class}`~EasyFEA.Models.MaterialPoint` drives one of these at a single Gauss point, with no mesh and no solver.
(models-beam)=
+ Elastic beams with {py:class}`~EasyFEA.Models.Beam.Isotropic`, {py:class}`~EasyFEA.Models.Beam.BeamStructure`, in {py:class}`Models.Beam <EasyFEA.Models.Beam>`.
+ Phase-field materials with {py:class}`~EasyFEA.Models.PhaseField`.
+ Thermal materials with {py:class}`~EasyFEA.Models.Thermal`.
+ Weak forms with {py:class}`~EasyFEA.Models.WeakForms`.

```{seealso}
- {ref}`howto-models`
```

## Models API

```{eval-rst}
.. automodule:: EasyFEA.Models
.. automodule:: EasyFEA.Models.Elastic
.. automodule:: EasyFEA.Models.HyperElastic
.. automodule:: EasyFEA.Models.Beam
.. automodule:: EasyFEA.Models.Yield
.. automodule:: EasyFEA.Models.IsotropicHardening
.. automodule:: EasyFEA.Models.KinematicHardening
.. automodule:: EasyFEA.Models.ViscoPlastic
.. automodule:: EasyFEA.Models.ViscoElastic
```