(simulations)=

# Simulations

The {py:mod}`EasyFEA.Simulations` module provides essential tools for creating and
managing simulations. These simulations are built using a {py:class}`~EasyFEA.FEM.Mesh`
and a {py:class}`~EasyFEA.Models._IModel` (see {ref}`models`).

In the simulation workflow, {py:class}`~EasyFEA.Simulations._Simu` is the **central
step**: it takes a mesh and a model, exposes boundary-condition methods
({py:func}`~EasyFEA.Simulations._Simu.add_dirichlet`,
{py:func}`~EasyFEA.Simulations._Simu.add_surfLoad`, …), drives the linear solver, and
stores the solution history.

With this module, you can construct:

- Linear elastic simulations with {py:class}`~EasyFEA.Simulations.Elastic`.
- Nonlinear hyperelastic simulations with {py:class}`~EasyFEA.Simulations.HyperElastic`.
- Small-strain materials whose stress depends on the history of strain — plasticity,
  viscoplasticity and viscoelasticity — with {py:class}`~EasyFEA.Simulations.InElastic`
  (see {ref}`simulations-inelastic`).
- Euler-Bernoulli and Timoshenko beam simulations with
  {py:class}`~EasyFEA.Simulations.Beam` (`useTimoshenko=True` to switch).
- PhaseField damage simulations for quasi-static brittle fracture with
  {py:class}`~EasyFEA.Simulations.PhaseField`.
- Thermal simulations with {py:class}`~EasyFEA.Simulations.Thermal`.
- Weak form simulations with {py:class}`~EasyFEA.Simulations.WeakForms`.

```{seealso}
- {ref}`howto-boundary-conditions` 
- {ref}`howto-pipeline`
```

## Matrix System Solvers

EasyFEA automatically manages the resolution of `elliptic`, `parabolic`, and
`hyperbolic` matrix systems, allowing developers to focus exclusively on declaring the
local matrix system via the {py:func}`~EasyFEA.Simulations._Simu.Get_terms` method.

### Elliptic

$$
\Krm \, \urm = \Frm
$$ (elliptic)

### Parabolic

$$
\Krm \, \urm^{n+\alpha} + \Crm \, \vrm^{n+\alpha} = \Frm^{n+\alpha}
$$ (parabolic)

Set with {py:func}`~EasyFEA.Simulations._Simu.Solver_Set_Parabolic_Algorithm`.

| Method         | α   | Order | Stability              |
| -------------- | --- | ----- | ---------------------- |
| Forward Euler  | 0   | 1st   | Conditionally stable   |
| Crank–Nicolson | 0.5 | 2nd   | Unconditionally stable |
| Backward Euler | 1   | 1st   | Unconditionally stable |

### Hyperbolic

$$
\Krm \, \urm + \Crm \, \vrm + \Mrm \, \arm = \Frm
$$ (hyperbolic)

Set with {py:func}`~EasyFEA.Simulations._Simu.Solver_Set_Hyperbolic_Algorithm`.

| Method         | `AlgoType`                                                      | Order | Stability                         | Notes                                                             |
| -------------- | --------------------------------------------------------------- | ----- | --------------------------------- | ----------------------------------------------------------------- |
| Newmark β      | {py:attr}`~EasyFEA.Simulations.Solvers.AlgoType.newmark`        | 2nd   | Unconditionally stable            | Default; energy-conserving for **linear** problems (β=1/4, γ=1/2) |
| Midpoint       | {py:attr}`~EasyFEA.Simulations.Solvers.AlgoType.midpoint`       | 2nd   | Unconditionally stable            | Energy-conserving for **linear** problems                         |
| HHT-α          | {py:attr}`~EasyFEA.Simulations.Solvers.AlgoType.hht`            | 2nd   | Unconditionally stable            | Numerical damping (α ∈ \[0, 1\[)                                  |
| Euler implicit | {py:attr}`~EasyFEA.Simulations.Solvers.AlgoType.euler_implicit` | 1st   | Unconditionally stable            | Dissipative                                                       |
| Euler explicit | {py:attr}`~EasyFEA.Simulations.Solvers.AlgoType.euler_explicit` | 1st   | Conditionally stable (dt < h_e/c) | Linear only                                                       |

### Hyperelastic stress

For a **nonlinear** problem the time scheme alone does not conserve energy: with a
hyperelastic law, `newmark` and `midpoint` both drift (a few % of $E_0$ on a free
vibration). What decides conservation is the stress used in the internal force, selected
independently of the scheme with
`simu.Solver_Set_Stress(HyperElastic.StressType.gonzalez)`.

| Stress     | `HyperElastic.StressType`                                          | Energy                   | Cost / step               | Notes                                                                              |
| ---------- | ------------------------------------------------------------------ | ------------------------ | ------------------------- | ---------------------------------------------------------------------------------- |
| Pointwise  | {py:attr}`~EasyFEA.Simulations.HyperElastic.StressType.pointwise`  | Drifts                   | 1 stress evaluation       | Default; $\Srm(\eb(\ub^t))$ at the scheme's evaluation state                       |
| Gonzalez   | {py:attr}`~EasyFEA.Simulations.HyperElastic.StressType.gonzalez`   | Exact, any law           | 1 evaluation + correction | Discrete gradient $\bar{\Srm} + \alpha \Delta \eb$                                 |
| Quadrature | {py:attr}`~EasyFEA.Simulations.HyperElastic.StressType.quadrature` | Exact as `nPoints` grows | `nPoints` evaluations     | Strain-path average; spectral convergence, exact at every rule for a quadratic $W$ |

Both non-default stresses rest on the identity
$\Delta \eb = \Brm(\bar{\ub}) \cdot \Delta \ub$ and therefore require
{py:attr}`~EasyFEA.Simulations.Solvers.AlgoType.midpoint`; call `Solver_Set_Stress`
**after** `Solver_Set_Hyperbolic_Algorithm`. Calling it with no argument returns to
`pointwise`. See {ref}`fem-operators` for the operators that assemble them, and
`examples/Hyperelasticity/Hyperelas5.py` for a side-by-side comparison.

(simulations-inelastic)=

### InElastic

{py:class}`~EasyFEA.Simulations.InElastic` covers materials whose stress depends on the
*history* of strain — plasticity, viscoplasticity, viscoelasticity — so it carries
internal variables `z` from one converged step to the next. It solves **two nested
problems**.

*Globally*, a Newton-Raphson on the displacement, exactly as any nonlinear simulation:
assemble $\Krm_T = \int \Brm^T \Crm_{alg} \Brm$ and the residual $-\int \Brm^T \Sig$,
solve, repeat.

*Locally*, at every Gauss point, the material is a
{py:class}`~EasyFEA.Models.InElastic._Behavior`: one `Update(eps, z, dt)` written at one
3D point in `jax.numpy`, returning the stress and the new state. EasyFEA lifts it to
every Gauss point (`vmap`), derives $\Crm_{alg}$ by `jacfwd` through `Update`, pads
plane strain, and under plane stress solves $\varepsilon_{zz}$ at each point so that
$\sigma_{zz} = 0$, the tangent going through that solve too.

A return mapping is a {py:func}`~EasyFEA.Models.InElastic.Newton` on the state
increments, differentiable through its root, so the consistent tangent needs no
hand-written Jacobian. {py:class}`~EasyFEA.Models.InElastic.Plasticity` writes

$$
\begin{aligned}
r_p &= \Delta\Eps^p - \Delta p\, \Nrm, \qquad \Nrm = \dpartial{f}{\Sig} \\
r_f &= f(\Sig, R(p_n + \Delta p)) / \sigma_y
\end{aligned}
$$

with $\Nrm$ from `jax.grad` of the surface; {py:class}`~EasyFEA.Models.InElastic.Norton`
subtracts the overstress its creep rate sustains from $r_f$, and
{py:class}`~EasyFEA.Models.InElastic.Chaboche` reads the surface at $\Sig - \Xrm$ and
adds one row per back-stress. A point on the surface flows, so that it gets the loading
tangent whatever the roundoff in $f$.

A converged {py:func}`~EasyFEA.Simulations.InElastic.Solve` is a step: it commits the
internal variables, which the next step integrates from and results read, the stress
being `Stress` at the committed ones. {py:func}`~EasyFEA.Simulations._Simu.Save_Iter`
only records them, so it can be skipped on some steps;
{py:func}`~EasyFEA.Simulations._Simu.Set_Iter` restarts from a saved one. The kernel is
compiled on the first step, which costs about a second.

```{seealso}
- {ref}`easyfea-examples-inelasticity` — nine examples, each checked against a closed form
- {ref}`Inelastic models <models-inelastic>` — the shipped behaviors and how to write one
```

## How to Create New Simulations in EasyFEA?

To create new simulation classes, you can take inspiration from existing
implementations. Make sure to follow the {py:class}`~EasyFEA.Simulations._Simu`
interface. The {py:class}`~EasyFEA.Simulations.Thermal` class is relatively simple and
can serve as a good starting point.

```{seealso}
- {ref}`howto-new-simulation`
- [EasyFEA/Simulations/_thermal.py](https://github.com/matnoel/EasyFEA/blob/main/EasyFEA/Simulations/_thermal.py) source code
- {ref}`howto-pipeline`
```

## Simulations API

```{eval-rst}
.. automodule:: EasyFEA.Simulations
    :imported-members:

.. autoclass:: EasyFEA.Simulations.HyperElastic.StressType
   :members:
   :undoc-members:
   :no-index:
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
