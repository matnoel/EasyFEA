(models)=
# models

The [EasyFEA/models/](https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/models) module in EasyFEA provides essential tools for creating and managing models.
These models are used to build {py:class}`~EasyFEA.simulations._Simu` instances and mainly contain material parameters.


With this module, you can construct:

(models-elastic)=
+ Linear elastic materials, such as {py:class}`~EasyFEA.Models.Elastic.Isotropic`, {py:class}`~EasyFEA.Models.Elastic.TransverselyIsotropic`, {py:class}`~EasyFEA.Models.Elastic.Orthotropic`, and {py:class}`~EasyFEA.Models.Elastic.Anisotropic`, in `Models.Elastic`.
(models-hyperelastic)=
+ Nonlinear hyperelastic materials, such as {py:class}`~EasyFEA.Models.HyperElastic.NeoHookean`, {py:class}`~EasyFEA.Models.HyperElastic.MooneyRivlin`, {py:class}`~EasyFEA.Models.HyperElastic.SaintVenantKirchhoff`, and {py:class}`~EasyFEA.Models.HyperElastic.HolzapfelOgden`, in `Models.HyperElastic`.
(models-beam)=
+ Elastic beams with {py:class}`~EasyFEA.Models.Beam.Isotropic`, {py:class}`~EasyFEA.Models.Beam.BeamStructure`, in `Models.Beam`.
+ Phase-field materials with {py:class}`~EasyFEA.Models.PhaseField`.
+ Thermal materials with {py:class}`~EasyFEA.Models.Thermal`.
+ Weark forms with {py:class}`~EasyFEA.Models.WeakForms`.


## Detailed models API

```{eval-rst}
.. automodule:: EasyFEA.models
    :private-members: _IModel

.. automodule:: EasyFEA.Models.Elastic
    :members:
    :special-members: __init__
    :private-members: _Elastic
    :imported-members:
    :show-inheritance:

.. automodule:: EasyFEA.Models.HyperElastic
    :members:
    :special-members: __init__
    :private-members: _HyperElastic
    :imported-members:
    :show-inheritance:

.. automodule:: EasyFEA.Models.Beam
    :members:
    :special-members: __init__
    :private-members: _Beam
    :imported-members:
    :show-inheritance:

.. autoclass:: EasyFEA.Models.PhaseField
    :members:
    :special-members: __init__
    :show-inheritance:

.. autoclass:: EasyFEA.Models.Thermal
    :members:
    :special-members: __init__
    :show-inheritance:

.. autoclass:: EasyFEA.Models.WeakForms
    :members:
    :special-members: __init__
    :show-inheritance:

.. automodule:: EasyFEA.models._utils
    :members:
```