(models)=

# Models

The {py:mod}`EasyFEA.Models` module provides essential tools for creating and managing
models. These models are used to build {py:class}`~EasyFEA.Simulations._Simu` instances
and mainly contain material parameters.

In the simulation workflow, `Models` is the **third step**: once the mesh exists, a
model encapsulates the physics and material constants (Young's modulus, thermal
conductivity, fracture toughness, …) before being passed to the simulation constructor.

With this module, you can construct:

(models-elastic)=

- Linear elastic materials, such as {py:class}`~EasyFEA.Models.Elastic.Isotropic`,
  {py:class}`~EasyFEA.Models.Elastic.TransverselyIsotropic`,
  {py:class}`~EasyFEA.Models.Elastic.Orthotropic`, and
  {py:class}`~EasyFEA.Models.Elastic.Anisotropic`, in
  {py:class}`Models.Elastic <EasyFEA.Models.Elastic>`.

(models-hyperelastic)=

- Nonlinear hyperelastic materials, such as
  {py:class}`~EasyFEA.Models.HyperElastic.NeoHookean`,
  {py:class}`~EasyFEA.Models.HyperElastic.CiarletGeymonat`,
  {py:class}`~EasyFEA.Models.HyperElastic.MooneyRivlin`,
  {py:class}`~EasyFEA.Models.HyperElastic.SaintVenantKirchhoff`, and
  {py:class}`~EasyFEA.Models.HyperElastic.HolzapfelOgden`, in
  {py:class}`Models.HyperElastic <EasyFEA.Models.HyperElastic>`.

(models-inelastic)=

- Small-strain materials whose stress depends on the *history* of strain — plasticity,
  viscoplasticity and viscoelasticity — as a
  {py:class}`~EasyFEA.Models.InElastic._Behavior`: one `Update` and one `Stress` written
  at one 3D point, in `jax.numpy`, from which EasyFEA derives the tangent and handles
  2D. Shipped: {py:class}`~EasyFEA.Models.InElastic.Plasticity` on any surface from
  {py:class}`~EasyFEA.Models.InElastic.Yield` with any hardening from
  {py:class}`~EasyFEA.Models.InElastic.IsotropicHardening`,
  {py:class}`~EasyFEA.Models.InElastic.Norton`,
  {py:class}`~EasyFEA.Models.InElastic.Chaboche` and
  {py:class}`~EasyFEA.Models.InElastic.Maxwell`.
  {py:class}`~EasyFEA.Models.InElastic.MaterialPoint` drives one at a single point, with
  no mesh and no solver.

(models-beam)=

- Elastic beams with {py:class}`~EasyFEA.Models.Beam.Isotropic`,
  {py:class}`~EasyFEA.Models.Beam.BeamStructure`, in
  {py:class}`Models.Beam <EasyFEA.Models.Beam>`.
- Phase-field materials with {py:class}`~EasyFEA.Models.PhaseField`.
- Thermal materials with {py:class}`~EasyFEA.Models.Thermal`.
- Weak forms with {py:class}`~EasyFEA.Models.WeakForms`.

```{seealso}
- {ref}`howto-models`
```

## Models API

```{eval-rst}
.. automodule:: EasyFEA.Models
    :imported-members:
.. automodule:: EasyFEA.Models.Elastic
    :imported-members:
.. automodule:: EasyFEA.Models.HyperElastic
    :imported-members:
.. automodule:: EasyFEA.Models.InElastic
    :imported-members:
.. automodule:: EasyFEA.Models.InElastic.Yield
.. automodule:: EasyFEA.Models.InElastic.IsotropicHardening
.. automodule:: EasyFEA.Models.Beam
    :imported-members:
```
