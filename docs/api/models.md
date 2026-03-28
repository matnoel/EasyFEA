(models)=
# Models

The {py:mod}`EasyFEA.Models` module provides essential tools for creating and managing models.
These models are used to build {py:class}`~EasyFEA.Simulations._Simu` instances and mainly contain material parameters.

In the simulation workflow, `Models` is the **third step**: once the mesh exists, a model encapsulates the physics and material constants (Young's modulus, thermal conductivity, fracture toughness, …) before being passed to the simulation constructor. See {ref}`howto-models` for practical examples with constructor signatures and printed outputs.


With this module, you can construct:

(models-elastic)=
+ Linear elastic materials, such as {py:class}`~EasyFEA.Models.Elastic.Isotropic`, {py:class}`~EasyFEA.Models.Elastic.TransverselyIsotropic`, {py:class}`~EasyFEA.Models.Elastic.Orthotropic`, and {py:class}`~EasyFEA.Models.Elastic.Anisotropic`, in {py:class}`Models.Elastic <EasyFEA.Models.Elastic>`.
(models-hyperelastic)=
+ Nonlinear hyperelastic materials, such as {py:class}`~EasyFEA.Models.HyperElastic.NeoHookean`, {py:class}`~EasyFEA.Models.HyperElastic.MooneyRivlin`, {py:class}`~EasyFEA.Models.HyperElastic.SaintVenantKirchhoff`, and {py:class}`~EasyFEA.Models.HyperElastic.HolzapfelOgden`, in {py:class}`Models.HyperElastic <EasyFEA.Models.HyperElastic>`.
(models-beam)=
+ Elastic beams with {py:class}`~EasyFEA.Models.Beam.Isotropic`, {py:class}`~EasyFEA.Models.Beam.BeamStructure`, in {py:class}`Models.Beam <EasyFEA.Models.Beam>`.
+ Phase-field materials with {py:class}`~EasyFEA.Models.PhaseField`.
+ Thermal materials with {py:class}`~EasyFEA.Models.Thermal`.
+ Weak forms with {py:class}`~EasyFEA.Models.WeakForms`.

```{seealso}
{ref}`howto-models` — choose and construct a material model, with constructor
signatures and printed outputs for every model family.
```

## Models API

```{eval-rst}
.. automodule:: EasyFEA.Models
.. automodule:: EasyFEA.Models.Elastic
.. automodule:: EasyFEA.Models.HyperElastic
.. automodule:: EasyFEA.Models.Beam
```