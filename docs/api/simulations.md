(simulations)=
# Simulations

The {py:mod}`EasyFEA.Simulations` module provides essential tools for creating and managing simulations.
These simulations are built using a {py:class}`~EasyFEA.FEM.Mesh` and a {py:class}`~EasyFEA.Models._IModel` (see {ref}`models`).

In the simulation workflow, `Simulations` is the **central step**: it takes a mesh and a model, exposes boundary-condition methods (`add_dirichlet`, `add_surfLoad`, …), drives the linear solver, and stores the solution history.

With this module, you can construct:

+ Linear elastic simulations with {py:class}`~EasyFEA.Simulations.Elastic`.
+ Nonlinear hyperelastic simulations with {py:class}`~EasyFEA.Simulations.HyperElastic`.
+ Euler-Bernoulli and Timoshenko beam simulations with {py:class}`~EasyFEA.Simulations.Beam` (`useTimoshenko=True` to switch).
+ PhaseField damage simulations for quasi-static brittle fracture with {py:class}`~EasyFEA.Simulations.PhaseField`.
+ Thermal simulations with {py:class}`~EasyFEA.Simulations.Thermal`.
+ Weak form simulations with {py:class}`~EasyFEA.Simulations.WeakForms`.

```{seealso}
- {ref}`howto-boundary-conditions` 
- {ref}`howto-pipeline`
```

## Matrix System Solvers

EasyFEA automatically manages the resolution of `elliptic`, `parabolic`, and `hyperbolic` matrix systems, allowing developers to focus exclusively on constructing local matrices via the `Construct_local_matrix_system` method.

### Elliptic
$$
\Krm \, \urm = \Frm
$$ (elliptic)

### Parabolic
$$
\Krm \, \urm^{n+\alpha} + \Crm \, \vrm^{n+\alpha} = \Frm^{n+\alpha}
$$ (parabolic)

Set with `simu.Solver_Set_Parabolic_Algorithm(dt, alpha=0.5)`.

| Method | α | Order | Stability |
|--------|---|-------|-----------|
| Forward Euler | 0 | 1st | Conditionally stable |
| Crank–Nicolson | 0.5 | 2nd | Unconditionally stable |
| Backward Euler | 1 | 1st | Unconditionally stable |

### Hyperbolic
$$
\Krm \, \urm + \Crm \, \vrm + \Mrm \, \arm = \Frm
$$ (hyperbolic)

Set with `simu.Solver_Set_Hyperbolic_Algorithm(dt, algo=AlgoType.newmark)`.

| Method | `AlgoType` | Order | Stability | Notes |
|--------|------------|-------|-----------|-------|
| Newmark β | {py:attr}`~EasyFEA.Simulations.Solvers.AlgoType.newmark` | 2nd | Unconditionally stable | Default; energy-conserving (β=1/4, γ=1/2) |
| Midpoint | {py:attr}`~EasyFEA.Simulations.Solvers.AlgoType.midpoint` | 2nd | Unconditionally stable | Energy-conserving |
| HHT-α | {py:attr}`~EasyFEA.Simulations.Solvers.AlgoType.hht` | 2nd | Unconditionally stable | Numerical damping (α ∈ [0, 1[) |
| Euler implicit | {py:attr}`~EasyFEA.Simulations.Solvers.AlgoType.euler_implicit` | 1st | Unconditionally stable | Dissipative |
| Euler explicit | {py:attr}`~EasyFEA.Simulations.Solvers.AlgoType.euler_explicit` | 1st | Conditionally stable (dt < h_e/c) | Linear only |

## How to Create New Simulations in EasyFEA?

To create new simulation classes, you can take inspiration from existing implementations.
Make sure to follow the {py:class}`~EasyFEA.Simulations._Simu` interface.
The {py:class}`~EasyFEA.Simulations.Thermal` class is relatively simple and can serve as a good starting point.

```{seealso}
- {ref}`howto-new-simulation`
- [EasyFEA/Simulations/_thermal.py](https://github.com/matnoel/EasyFEA/blob/main/EasyFEA/Simulations/_thermal.py) source code
- {ref}`howto-pipeline`
```

## Simulations API

```{eval-rst}
.. automodule:: EasyFEA.Simulations
```

## Solvers API

```{eval-rst}
.. automodule:: EasyFEA.Simulations.Solvers
   :no-members:

.. autoclass:: EasyFEA.Simulations.Solvers.AlgoType
   :members:
   :undoc-members:

.. autoclass:: EasyFEA.Simulations.Solvers.ResolType
   :members:
   :undoc-members:
   
.. autoclass:: EasyFEA.Simulations.Solvers.SolverType
   :members:
   :undoc-members:

```