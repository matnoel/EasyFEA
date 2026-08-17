(simulations)=
# Simulations

The {py:mod}`EasyFEA.Simulations` module provides essential tools for creating and managing simulations.
These simulations are built using a {py:class}`~EasyFEA.FEM.Mesh` and a {py:class}`~EasyFEA.Models._IModel` (see {ref}`models`).

In the simulation workflow, `Simulations` is the **central step**: it takes a mesh and a model, exposes boundary-condition methods (`add_dirichlet`, `add_surfLoad`, …), drives the linear solver, and stores the solution history.

With this module, you can construct:

+ Linear elastic simulations with {py:class}`~EasyFEA.Simulations.Elastic`.
+ Nonlinear hyperelastic simulations with {py:class}`~EasyFEA.Simulations.HyperElastic`.
+ Small-strain materials whose stress depends on the history of strain — plasticity, viscoplasticity and viscoelasticity — with {py:class}`~EasyFEA.Simulations.Behaviour` (see {ref}`simulations-behaviour`).
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
| Newmark β | {py:attr}`~EasyFEA.Simulations.Solvers.AlgoType.newmark` | 2nd | Unconditionally stable | Default; energy-conserving for **linear** problems (β=1/4, γ=1/2) |
| Midpoint | {py:attr}`~EasyFEA.Simulations.Solvers.AlgoType.midpoint` | 2nd | Unconditionally stable | Energy-conserving for **linear** problems |
| HHT-α | {py:attr}`~EasyFEA.Simulations.Solvers.AlgoType.hht` | 2nd | Unconditionally stable | Numerical damping (α ∈ [0, 1[) |
| Euler implicit | {py:attr}`~EasyFEA.Simulations.Solvers.AlgoType.euler_implicit` | 1st | Unconditionally stable | Dissipative |
| Euler explicit | {py:attr}`~EasyFEA.Simulations.Solvers.AlgoType.euler_explicit` | 1st | Conditionally stable (dt < h_e/c) | Linear only |

### Hyperelastic stress

For a **nonlinear** problem the time scheme alone does not conserve energy: with a hyperelastic
law, `newmark` and `midpoint` both drift (a few % of $E_0$ on a free vibration). What decides
conservation is the stress used in the internal force, selected independently of the scheme with
`simu.Solver_Set_Stress(HyperElastic.StressType.gonzalez)`.

| Stress | `HyperElastic.StressType` | Energy | Cost / step | Notes |
|--------|---------------------------|--------|-------------|-------|
| Pointwise | {py:attr}`~EasyFEA.Simulations.HyperElastic.StressType.pointwise` | Drifts | 1 stress evaluation | Default; $\Srm(\eb(\ub^t))$ at the scheme's evaluation state |
| Gonzalez | {py:attr}`~EasyFEA.Simulations.HyperElastic.StressType.gonzalez` | Exact, any law | 1 evaluation + correction | Discrete gradient $\bar{\Srm} + \alpha \Delta \eb$ |
| Quadrature | {py:attr}`~EasyFEA.Simulations.HyperElastic.StressType.quadrature` | Exact as `nPoints` grows | `nPoints` evaluations | Strain-path average; spectral convergence, exact at every rule for a quadratic $W$ |

Both non-default stresses rest on the identity $\Delta \eb = \Brm(\bar{\ub}) \cdot \Delta \ub$ and
therefore require {py:attr}`~EasyFEA.Simulations.Solvers.AlgoType.midpoint`; call
`Solver_Set_Stress` **after** `Solver_Set_Hyperbolic_Algorithm`. Calling it with no argument
returns to `pointwise`. See {ref}`fem-operators` for the operators that assemble them, and
`examples/Hyperelasticity/Hyperelas5.py` for a side-by-side comparison.

(simulations-behaviour)=
### Materials with history

{py:class}`~EasyFEA.Simulations.Behaviour` covers materials whose stress depends on the *history*
of strain — plasticity, viscoplasticity, viscoelasticity — so it carries internal variables `z`
from one converged step to the next. It solves **two nested problems**.

*Globally*, a Newton-Raphson on the displacement, exactly as any nonlinear simulation: assemble
$\Krm_T = \int \Brm^T \Crm_{alg} \Brm$ and the residual $-\int \Brm^T \Sig$, solve, repeat.

*Locally*, at every Gauss point independently, the material is asked for the stress and the
consistent tangent given the strain increment. Two routes are available; the material picks, and
`solver="newton"` forces the general one.

| Route | Used when | Unknowns |
|-------|-----------|----------|
| Implicit solve | Always applicable | the state increments $\Delta z$ and $\Delta\gamma$ |
| Spectral return | The yield surface is quadratic, $\phi^2 = \Sig:\Prm:\Sig$ — von Mises, Hill — with homogeneous $\Crm$, no kinematic hardening and no viscous branches | one scalar |

#### The implicit solve

Every internal variable is advanced by backward Euler, which gives one algebraic system per Gauss
point. The unknowns are the **increments**, as in MFront's implicit DSL, so each row is a change
and the committed state never appears on both sides of a subtraction:

$$
\begin{aligned}
r_{v,i} &= \Delta\Eps^v_i - \frac{\dt}{\tau_i}\left(\Eps^e - \Eps^v_i\right)
  &&\text{one per Maxwell branch} \\
r_p &= \Delta\Eps^p - \Delta\gamma\, \Nrm
  &&\text{the flow rule} \\
r_\alpha &= \Delta\alpha - \Delta\gamma
  &&\text{accumulated plastic strain} \\
r_{X_j} &= \Delta\boldsymbol{\alpha}_j - \Delta\gamma\left(\Nrm - \gamma_j\boldsymbol{\alpha}_j\right)
  &&\text{one per back-stress} \\
r_f &= f(\Sig, R) - \phi^{-1}(\Delta\gamma/\dt)
  &&\text{consistency, or the rate law}
\end{aligned}
$$

Laws are evaluated at the values $z_n + \Delta z$. The flow direction $\Nrm$ is read at the stress
shifted by the back-stress, $\boldsymbol{\xi} = \Sig - \Xrm$, which is what makes kinematic hardening move
the surface's centre rather than grow it.

Which rows exist depends on the pieces given. With no rate law the last term of $r_f$ drops and it
becomes the consistency condition $f = 0$. With no yield surface only the $r_{v,i}$ remain, the
system is linear, and it converges in a single iteration — pure viscoelastic relaxation needs no
iteration at all.

The system is small and there is one of it per Gauss point. Its size $n_u$ is set by the pieces
given — six rows for each tensor variable, one for the accumulated plastic strain, one for
$\Delta\gamma$:

| Configuration | Rows |
|---------------|------|
| Maxwell branches only, no yield surface | 6 per branch |
| von Mises + isotropic hardening | 8 |
| \+ one Armstrong-Frederick back-stress | 14 |
| \+ Chaboche with three components | 26 |
| \+ three components and two branches | 38 |

Newton starts from $\Delta z = 0$, and the **whole mesh is advanced together**. `J` is assembled as
a FeArray of shape `(Ne, nPg, nu, nu)` and the residual as `(Ne, nPg, nu)`, so

```python
u = Bound(u - np.linalg.solve(J, r[..., None])[..., 0])
```

is one call that solves `Ne × nPg` independent $n_u \times n_u$ systems: `np.linalg.solve` reads the
last two axes as the matrix and broadcasts over the leading ones, so `r[..., None]` supplies one
column vector per point. There is no Python loop over points, which is why the local solve costs
about the same as an assembly pass rather than dominating it.

The tangent reuses the same shape. `D` is $\partial r/\partial\Eps$ with shape `(Ne, nPg, nu, 6)`,
so `np.linalg.solve(J, D)` solves the *same* matrices against **six right-hand sides** — one per
Kelvin-Mandel strain component — and returns $\partial u/\partial\Eps$ in one call.

$\Jrm = \partial r/\partial u$ is written analytically, never finite-differenced. Every
dependence runs through the stress and the shifted stress,

$$\Sig = \Crm:(\Eps - \Eps^p) - \sum_i g_i \Crm : \Eps^v_i,
\qquad \boldsymbol{\xi} = \Sig - \sum_j k_j\boldsymbol{\alpha}_j$$

so the sensitivities are read straight off them — $\partial\Sig/\partial\Delta\Eps^p = -\Crm$,
$\partial\Sig/\partial\Delta\Eps^v_i = -g_i\Crm$,
$\partial\boldsymbol{\xi}/\partial\Delta\boldsymbol{\alpha}_j = -k_j$ — and the chain rule fills
the blocks:

| Block | Value | From |
|-------|-------|------|
| $\partial r_p/\partial\Delta\Eps^p$ | $\Irm + \Delta\gamma\,\dfrac{\partial \Nrm}{\partial\Sig}\Crm$ | the surface's own `dNdSig` |
| $\partial r_p/\partial\Delta\gamma$ | $-\Nrm$ | |
| $\partial r_f/\partial\Delta\Eps^p$ | $-\Nrm\Crm$ | $\partial f/\partial\Sig = \Nrm$ |
| $\partial r_f/\partial\Delta\alpha$ | $-R'$ | the hardening's own `dR` |
| $\partial r_f/\partial\Delta\gamma$ | $-\phi^{-1\prime}/\dt$ | the rate law's own `dinverse` |
| $\partial r_{v,i}/\partial\Delta\Eps^v_i$ | $(1 + \dt/\tau_i)\,\Irm$ | |
| $\partial r_{X_j}/\partial\Delta\boldsymbol{\alpha}_j$ | $(1 + \Delta\gamma\,\gamma_j)\,\Irm + \ldots$ | plus a full coupling block, since every back-stress shifts $\boldsymbol{\xi}$ and so moves $\Nrm$ for all the others |

The same pass builds $\partial r/\partial\Eps$, which the tangent needs. What makes this
extensible is that no block knows what law it belongs to: a yield surface supplies $\Nrm$ and
$\partial\Nrm/\partial\Sig$, a hardening law $R$ and $R'$, a rate law its inverse and derivative,
and the assembly is generic. Adding a mechanism means adding its rows and its blocks, not
touching the solver.

Three details matter for robustness:

- **Not every point flows.** A point whose trial state is inside the surface has its $\Delta\gamma$
  row replaced by $\Delta\gamma = 0$, so elastic points cannot be dragged into flowing by the
  shared solve. Viscous branches keep evolving either way, since relaxation needs no yield surface.
- **$\Delta\gamma \ge 0$** is enforced after each update; a negative multiplier has no meaning.
- **Rows are scaled before they are compared.** Most rows are strains and $r_f$ is a stress, so
  $r_f$ is divided by the surface scale. Left unscaled the stress row dominates by orders of
  magnitude and no step looks like an improvement.

With a rate law $\Delta\gamma$ is seeded from an explicit rate estimate rather than zero, because
the inverse rate law has unbounded derivative at zero flow and Newton would not move from there.

The **consistent tangent** costs no extra stress evaluations: the Jacobian
$\Jrm = \partial r/\partial u$ is built from quantities the converged state already carries, and
one further linear solve gives the tangent. Differentiating $r = 0$ with respect to the strain,

$$\frac{\partial u}{\partial \Eps} = -\Jrm^{-1}\frac{\partial r}{\partial \Eps},
\qquad
\Crm_{alg} = \Crm - \Crm\frac{\partial \Eps^p}{\partial \Eps}
- \sum_i g_i\,\Crm\frac{\partial \Eps^v_i}{\partial \Eps}$$

This is $\partial\Sig/\partial\Eps$ of the *algorithm*, not of the continuous law. Substituting the
elastic $\Crm$ still converges to the same answer, but costs the global Newton its quadratic rate.

#### The spectral return

A quadratic surface makes the update linear in the stress, which removes the need to iterate on a
vector at all. The flow is $\Delta\Eps^p = \Delta\gamma\,\Prm\Sig/\phi$, so with
$\theta = \Delta\gamma/\phi$,

$$(\Irm + \theta\,\Crm\Prm)\,\Sig = \Sig_{tr}$$

Diagonalising $\Crm^{1/2}\Prm\Crm^{1/2} = \Qrm\Lambda\Qrm^T$ — once per material, not per Gauss
point — makes that inverse diagonal, and consistency becomes an explicit scalar function of
$\theta$, monotone decreasing, solved by one safeguarded Newton. It applies to *any* linear
elasticity, not only isotropic. With isotropic $\Crm$ and von Mises the eigenvalue is repeated, the
scalar equation becomes linear, and it reduces to the classical radial return.

Measured over 20000 × 4 Gauss points, this is 2.6× faster than the implicit solve for von Mises
and 10.9× for Hill; its cost does not depend on the anisotropy. Where both routes apply they
agree to machine precision on stress, state and tangent, and the test suite checks it.

```{seealso}
- {ref}`easyfea-examples-behaviour` — ten examples, each checked against a closed form
- {ref}`models-behaviour` — the pieces a behaviour is assembled from
```

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