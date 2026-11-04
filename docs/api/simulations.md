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
- Perfectly incompressible hyperelastic simulations, solved for displacement and
  pressure, with {py:class}`~EasyFEA.Simulations.IncompressibleHyperElastic` (see
  {ref}`simulations-incompressible`).
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
3D point in `jax.numpy`, returning the stress and the new internal state. EasyFEA lifts
it to every Gauss point (`vmap`), derives $\Crm_{alg}$ by `jacfwd` through `Update`,
pads plane strain, and under plane stress solves $\varepsilon_{zz}$ at each point so
that $\sigma_{zz} = 0$, the tangent going through that solve too.

A return mapping is a {py:func}`~EasyFEA.Models.InElastic.Newton` on the internal state
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

External variables a behavior declares, such as a temperature, are given by
{py:func}`~EasyFEA.Simulations.InElastic.Set_external` as scalars or nodal fields, e.g.
`simu.Set_external(T=thermal.thermal)`, and kept until changed; a converged `Solve`
commits them with the internal variables, so results read those of the last step.

The displacement can use another order than the mesh (see {ref}`howto-function-spaces`):
the internal variables then live at the Gauss points of the displacement's elements.

```{seealso}
- {ref}`easyfea-examples-inelasticity`
- {ref}`Inelastic models <models-inelastic>` — the shipped behaviors
- {ref}`howto-models-inelastic-own`
```

(simulations-incompressible)=

### IncompressibleHyperElastic

{py:class}`~EasyFEA.Simulations.IncompressibleHyperElastic` enforces $J = 1$ exactly:
one Newton solves the displacement $\ub$ and the pressure $p$ together, as one system
`[u | p]`.

$$
\Sb = \dpartial{W}{\eb} - p \, J \, \Cb^{-1}, \qquad
\int_{\Omega_0} q \, (1 - J) \, \dO = 0 \quad \forall \, q
$$

$p$ is a Lagrange multiplier, positive in compression. It is the hydrostatic pressure
only with a decoupled law. It is computed, never prescribed, so
{py:func}`~EasyFEA.Simulations._Simu.add_dirichlet` on it raises.

When the whole boundary has prescribed normal displacement, nothing sets the level of
$p$: the simulation pins its constant mode, then shifts $p$ to zero mean. That mean is
taken on the straight sides of the elements, so on a curved mesh $p$, and the total
stress through it, are off by a constant that shrinks as the mesh is refined. $\ub$ and
$J$ are not affected.

The pressure space comes from the displacement's elements, on their vertices:

| $\ub$ | TRI6 | QUAD9 | TETRA10 | HEXA27 | PRISM18 |
| ----- | ---- | ----- | ------- | ------ | ------- |
| $p$   | TRI3 | QUAD4 | TETRA4  | HEXA8  | PRISM6  |

Any other displacement element raises. On a linear mesh, raise the displacement's order
first (see {ref}`howto-function-spaces`):

```python
from EasyFEA import ElemType, Models, Simulations
from EasyFEA.Geoms import Domain

contour = Domain((0, 0), (1, 1), 0.1)
mesh = contour.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
meshQ2 = mesh._With_FunctionSpace(Simulations.IncompressibleHyperElastic.U, 2)

# here K is a shear coefficient, not a bulk modulus: the pressure enforces J = 1
material = Models.HyperElastic.NeoHookean(2, K=0.5)
simu = Simulations.IncompressibleHyperElastic(meshQ2, material)

U = Simulations.IncompressibleHyperElastic.U
nodesLeft = meshQ2.Nodes_Conditions(lambda x, y, z: x == 0, problemType=U)
nodesRight = meshQ2.Nodes_Conditions(lambda x, y, z: x == 1, problemType=U)
simu.add_dirichlet(nodesLeft, [0, 0], ["x", "y"])
simu.add_surfLoad(nodesRight, [0.01], ["y"])
simu.Solve()

p_n = simu.Result("p")  # at the nodes
J_e = simu.Result("J", nodeValues=False)  # on the elements
```

- A law with a non-zero bulk term is refused, because it locks against the pressure: set
  its bulk modulus `K` to 0. `NeoHookean` has none: its `K` is a shear coefficient. A
  viscosity or an active stress is refused too: the simulation is static and elastic.
- The system is a saddle point, so it needs a direct solver: iterative solvers raise.
  Under MPI it defaults to PETSc with MUMPS (see {ref}`howto-mpi`).
- The simulation works on a copy of the given mesh that carries the pressure space,
  `simu.mesh`: node ids are the same, but moving the given mesh does not reach it.
- `simu.pressure` is the vector of pressure DOFs. `simu.U` and `simu.P` name the two
  problem types, and `simu.Assembly((simu.U, simu.P))` assembles the coupled system; one
  field alone raises.

```{seealso}
- {ref}`easyfea-examples-hyperelasticity` — `Hyperelas5` and `Hyperelas6`
- {ref}`howto-function-spaces`
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
