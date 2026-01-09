(models)=
# models

The [EasyFEA/models/](https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/models) module in EasyFEA provides essential tools for creating and managing models.
These models are used to build {py:class}`~EasyFEA.simulations._Simu` instances and mainly contain material parameters.


With this module, you can construct:

+ Linear elastic materials, such as {py:class}`~EasyFEA.models.ElasIsot`, {py:class}`~EasyFEA.models.ElasIsotTrans`, {py:class}`~EasyFEA.models.ElasOrthotropic`, and {py:class}`~EasyFEA.models.ElasAnisot`.
+ Nonlinear hyperelastic materials, such as {py:class}`~EasyFEA.models.NeoHookean`, {py:class}`~EasyFEA.models.MooneyRivlin`, {py:class}`~EasyFEA.models.SaintVenantKirchhoff`, and {py:class}`~EasyFEA.models.HolzapfelOgden`.
+ Elastic beams with {py:class}`~EasyFEA.models.BeamElasIsot`.
+ Phase-field materials with {py:class}`~EasyFEA.models.PhaseField`.
+ Thermal materials with {py:class}`~EasyFEA.models.Thermal`.
+ Weark forms with {py:class}`~EasyFEA.models.WeakForms`.


## Detailed materials API

```{eval-rst}
.. automodule:: EasyFEA.Models
    
.. automodule:: EasyFEA.models
   :members:
   :private-members: _IModel
   :undoc-members:
   :imported-members:
   :show-inheritance:
```