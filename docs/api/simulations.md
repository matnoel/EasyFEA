(simulations)=
# Simulations

The {py:mod}`EasyFEA.Simulations` module in EasyFEA provides essential tools for creating and managing simulations.
These simulations are built using a {py:class}`~EasyFEA.FEM.Mesh` and a {py:class}`~EasyFEA.Models._IModel` (material).

With this module, you can construct:

+ Linear elastic simulations with {py:class}`~EasyFEA.Simulations.Elastic`.
+ Nonlinear hyperelastic simulations with {py:class}`~EasyFEA.Simulations.HyperElastic`.
+ Euler-Bernoulli beam simulations with {py:class}`~EasyFEA.Simulations.Beam`.
+ PhaseField damage simulations for quasi-static brittle fracture with {py:class}`~EasyFEA.Simulations.PhaseField`.
+ Thermal simulations with {py:class}`~EasyFEA.Simulations.Thermal`.
+ Weak form simulations with {py:class}`~EasyFEA.Simulations.WeakForms`.

## Matrix System Solvers

EasyFEA automatically manages the resolution of `elliptic`, `parabolic` and `hyperbolic` matrix systems, allowing developers to focus exclusively on constructing local matrices via the `Construct_local_matrix_system` method.

### Elliptic
$$
\Krm \, \urm = \Frm
$$ (elliptic)

### Parabolic
$$
\Krm \, \urm + \Crm \, \vrm = \Frm
$$ (parabolic)

### Hyperbolic
- **Methods:** Newmark, HHT, Midpoint

$$
\Krm \, \urm + \Crm \, \vrm + \Mrm \, \arm = \Frm
$$ (hyperbolic)

## How to create new simulations in EasyFEA ?

To create new simulation classes, you can take inspiration from existing implementations.  
Make sure to follow the {py:class}`~EasyFEA.Simulations._Simu` interface.  
The {py:class}`~EasyFEA.Simulations.Thermal` class is relatively simple and can serve as a good starting point.  

## Simulations API

```{eval-rst}
.. automodule:: EasyFEA.Simulations
```