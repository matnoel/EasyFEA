(models)=
# models

The [EasyFEA/models/](https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/models) module in EasyFEA provides essential tools for creating and managing models.
These models are used to build {py:class}`~EasyFEA.simulations._Simu` instances and mainly contain material parameters.


With this module, you can construct:

+ Linear elastic materials, such as {py:class}`~EasyFEA.Models.Elastic.Isotropic`, {py:class}`~EasyFEA.Models.Elastic.TransverselyIsotropic`, {py:class}`~EasyFEA.Models.Elastic.Orthotropic`, and {py:class}`~EasyFEA.Models.Elastic.Anisotropic` (see the {py:class}`~EasyFEA.Models.Elastic` class).
+ Nonlinear hyperelastic materials, such as {py:class}`~EasyFEA.Models.HyperElastic.NeoHookean`, {py:class}`~EasyFEA.Models.HyperElastic.MooneyRivlin`, {py:class}`~EasyFEA.Models.HyperElastic.SaintVenantKirchhoff`, and {py:class}`~EasyFEA.Models.HyperElastic.HolzapfelOgden` (see the {py:class}`~EasyFEA.Models.HyperElastic` class).
+ Elastic beams with {py:class}`~EasyFEA.Models.Beam.Isotropic` (see the {py:class}`~EasyFEA.Models.Beam` class).
+ Phase-field materials with {py:class}`~EasyFEA.Models.PhaseField`.
+ Thermal materials with {py:class}`~EasyFEA.Models.Thermal`.
+ Weark forms with {py:class}`~EasyFEA.Models.WeakForms`.


## Detailed materials API

```{eval-rst}
.. automodule:: EasyFEA.models
   :private-members: _IModel

.. automodule:: EasyFEA.Models
   :members:
   :undoc-members:
   :imported-members:
```