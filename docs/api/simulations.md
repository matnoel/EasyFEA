(simulations)=
# simulations

The [EasyFEA/simulations/](https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/simulations) module in EasyFEA provides essential tools for creating and managing simulations.
These simulations are built using a {py:class}`~EasyFEA.fem.Mesh` and a {py:class}`~EasyFEA.models._IModel` (material).

With this module, you can construct:

+ Linear elastic simulations with {py:class}`~EasyFEA.Simulations.ElasticSimu`.
+ Nonlinear hyperelastic simulations with {py:class}`~EasyFEA.Simulations.HyperElasticSimu`.
+ Euler-Bernoulli beam simulations with {py:class}`~EasyFEA.Simulations.BeamSimu`.
+ PhaseField damage simulations for quasi-static brittle fracture with {py:class}`~EasyFEA.Simulations.PhaseFieldSimu`.
+ Thermal simulations with {py:class}`~EasyFEA.Simulations.ThermalSimu`.
+ Weak form simulations with {py:class}`~EasyFEA.Simulations.WeakFormSimu`.

## Matrix System Solvers

EasyFEA handles the resolution of matrix systems automatically, allowing developers to focus solely on constructing local matrices.

### Elliptic
```{math}
:label: elliptic
\Krm \, \urm = \Frm
```

### Parabolic
```{math}
:label: parabolic
\Krm \, \urm + \Crm \, \vrm = \Frm
```

### Hyperbolic
- **Methods:** Newmark, HHT, Midpoint

```{math}
:label: hyperbolic
\Krm \, \urm + \Crm \, \vrm + \Mrm \, \arm = \Frm
```

## How to create new simulations in EasyFEA ?

To create new simulation classes, you can take inspiration from existing implementations.  
Make sure to follow the {py:class}`~EasyFEA.simulations._Simu` interface.  
The {py:class}`~EasyFEA.Simulations.ThermalSimu` class is relatively simple and can serve as a good starting point.  
See [EasyFEA/simulations/_thermal.py](https://github.com/matnoel/EasyFEA/tree/main/EasyFEA/simulations/_thermal.py) for more details.

## Detailed simulations API

```{eval-rst}
.. automodule:: EasyFEA.simulations
   :members:
   :private-members: _Simu
   :undoc-members:
   :imported-members:
   :show-inheritance:

.. automodule:: EasyFEA.Simulations
    :members: ElasticSimu, HyperElasticSimu, BeamSimu, PhaseFieldSimu, ThermalSimu, WeakFormSimu
```