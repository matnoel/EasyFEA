(howto-pipeline)=
# Understand the solve pipeline

The **solve pipeline** is the internal call chain executed each time {py:meth}`~EasyFEA.Simulations._Simu.Solve` is called.
Every {py:class}`~EasyFEA.Simulations._Simu` in the {py:mod}`EasyFEA.Simulations` namespace runs the same pipeline, which lives in {py:class}`EasyFEA.Simulations._Simu`.
This guide traces it from the moment you call `Solve()` to the moment the solution is stored, aimed at advanced or curious users who want to understand the internals without reading the full source.

```{note}
The internals described here live in `EasyFEA/Simulations/_simu.py`,
`EasyFEA/Simulations/Solvers.py`, and `EasyFEA/FEM/_forms.py`. The
single-underscore methods (e.g. `_Solver_Apply_Dirichlet`) are advanced API;
the double-underscore ones are private and should never be called directly.
```

---

## Overview

Every call to {py:meth}`~EasyFEA.Simulations._Simu.Solve` performs the same three high-level operations:

1. **Build** — assemble the global sparse matrices ($\Krm, \Crm, \Mrm, \Frm$) from element
   integrals (skipped if nothing changed since the last solve).
2. **Apply BCs** — add Neumann contributions to the right-hand side, then
   enforce Dirichlet constraints to constrain the system $\Arm \, \xrm = \brm$.
3. **Solve** — pass $\Arm \, \xrm = \brm$ to the linear algebra backend (scipy, PETSc,
   pypardiso) and store the solution.

For **non-linear** problems (`simu.isNonLinear = True`), steps 1–3 are wrapped in a Newton–Raphson loop that repeats until the residual converges — see {ref}`howto-pipeline-newton`.

---

(howto-pipeline-operators)=
## From element operators to the global system

**Build** turns the mesh + model into four global sparse matrices. Each {py:class}`~EasyFEA.Simulations._Simu` implements `Construct_local_matrix_system`, which loops over the mesh element groups and returns, per group, the element tuple $(\Krm_e, \Crm_e, \Mrm_e, \Frm_e)$ — stiffness (the tangent, for non-linear laws), damping, mass, and the force vector. For a linear problem $\Frm_e$ is the load alone and the time scheme moves $\urm^n, \vrm^n, \arm^n$ to the right-hand side; for a non-linear one it is the complete residual $-\Rrm_e$, as described {ref}`below <howto-pipeline-nonlinear-operators>`.

These element matrices come from **operators** in {py:mod}`EasyFEA.FEM.Operators` — small functions that integrate a form over the Gauss points: {py:mod}`~EasyFEA.FEM.Operators.Bilinear` (e.g. $\int \Brm^\top \, \bf{C} \, \Brm \, \dO$), {py:mod}`~EasyFEA.FEM.Operators.Linear`, and {py:mod}`~EasyFEA.FEM.Operators.NonLinear`. The simulation assembles them into the global $\Krm, \Crm, \Mrm, \Frm$ and combines $\Krm, \Crm, \Mrm$ into the system matrix according to the time scheme:

$$\Arm = \alpha_\Krm\,\Krm + \alpha_\Crm\,\Crm + \alpha_\Mrm\,\Mrm$$

where $(\alpha_\Krm, \alpha_\Crm, \alpha_\Mrm)$ are the scheme-dependent weights for the active `AlgoType` (Newmark, HHT, midpoint, …); a static problem uses $\alpha_\Crm = \alpha_\Mrm = 0$.

(howto-pipeline-nonlinear-operators)=
### Non-linear assembly

A non-linear problem solves for the **increment**, not for $\urm^{n+1}$:

$$\Arm(\urm) \, \Delta \urm = - \Rrm(\urm)$$

That single difference sets the whole contract. Because the unknown is $\Delta \urm$, the time-scheme history terms of the previous section do not apply — they move $\urm^n, \vrm^n, \arm^n$ to the right-hand side of a solve for $\urm^{n+1}$ — so the solver adds nothing of its own. $\Frm_e$ must therefore be the **complete** residual $-\Rrm_e$, inertia and damping included.

Every operator in {py:mod}`~EasyFEA.FEM.Operators.NonLinear` is built for this and returns its own $(\text{tangent}, \text{residual})$ pair, evaluated at the current iterate. Assembly is then the same two moves for each of them, whatever the physics:

- the **residual** is subtracted from $\Frm_e$;
- the **tangent** goes to the slot carrying the weight it needs — $\Krm_e$ for a derivative with respect to $\urm$, $\Crm_e$ with respect to $\vrm$, $\Mrm_e$ with respect to $\arm$ — so that the single $\Arm = \alpha_\Krm\,\Krm + \alpha_\Crm\,\Crm + \alpha_\Mrm\,\Mrm$ combination weights it correctly, with no time-scheme special-casing.

| Operator | Contributes |
|---|---|
| {py:func}`~EasyFEA.FEM.Operators.NonLinear.SecondPiolaKirchhoffStressTensor` | consistent tangent (material + geometric) and internal residual $\int \Brm^\top \boldsymbol{\Sigma} \, \dO$ |
| {py:func}`~EasyFEA.FEM.Operators.NonLinear.GonzalezStressTensor` | the same pair, from an energy-conserving discrete gradient |
| {py:func}`~EasyFEA.FEM.Operators.NonLinear.TimeQuadratureStressTensor` | the same pair, from a strain-path average (also reports the quadrature point count used) |
| {py:func}`~EasyFEA.FEM.Operators.NonLinear.ActiveStressTensor` | fiber active stress $\tau \, (\hat{\Tb} \otimes \hat{\Tb})$ — geometric tangent and residual, no material tangent |
| {py:func}`~EasyFEA.FEM.Operators.NonLinear.KelvinVoigtDamping` | viscous stress $\boldsymbol{\Sigma}_{\text{visco}} = \eta \dot{\Erm}$ — configuration tangent and residual, plus the damping matrix for the $\Crm$ slot |
| {py:func}`~EasyFEA.FEM.Operators.NonLinear.FollowingPressure` | pressure tracking the deformed normal — non-symmetric tangent and follower load |
| {py:func}`~EasyFEA.FEM.Operators.NonLinear.PenaltyContact` | penalty tangent on the active contact set, and the force pushing the body out of the obstacle |

---


(howto-pipeline-newton)=
## Non-linear convergence

{py:meth}`~EasyFEA.Simulations._Simu._Solver_Solve_Newton_Raphson` drives the iteration. It starts from the last converged solution and repeats three moves — reassemble at the current iterate, solve for the correction, add it:

```python
u = self._Get_u_n(problemType)                  # a copy, so `u += ...` is safe

while not converged and newtonIter < maxIter:
    self.Need_Update()                          # rebuild A and R at the current u
    self.__Solver_Set_Newton_Raphson_current_solution(u)

    delta_u, norm = Solve_simu(self, self.problemType)
    relNorm = norm / list_norm[0]
    u += delta_u

    converged = (norm < absTol) or (relNorm < relTol) or (norm_delta_u < incTol)

assert converged, "..."
```

`Need_Update()` is what makes this Newton rather than a fixed-point iteration: it invalidates the cached matrices, so `Construct_local_matrix_system` re-evaluates the tangent **and** the residual at the new iterate. Setting the current solution just before is what lets both the assembly and the boundary conditions see it.

### What the norm measures

Not $\|\Delta \urm\|$, and not the raw right-hand side: `norm` is the norm of the **reduced** right-hand side, over the free dofs, after the prescribed ones have been eliminated. Restricting to free dofs is deliberate — at a prescribed dof the residual is the reaction, which is not an equilibrium error and never vanishes. Under MPI the per-rank contributions are reduced across ranks.

Because the unknown is $\Delta \urm$, the prescribed values are incremental too: {py:meth}`~EasyFEA.Simulations._Simu._Solver_Apply_Dirichlet` subtracts the current iterate and imposes $\urm_{\text{target}} - \urm$. The full boundary jump is therefore applied on the first iteration and decays to zero as $\urm$ reaches the target — which is why the first residual is large, and why $\text{relNorm}$ reads as a fraction of the load step's initial imbalance.

### Stopping

Three criteria, any one of which is enough:

| Criterion | Default | Meaning |
|---|---|---|
| `norm < absTol` | `1e-6` | absolute residual |
| `relNorm < relTol` | `1e-10` | residual relative to the step's first |
| `norm_delta_u < incTol` | `1e-11` | the correction has stopped moving |

The defaults are {py:class}`~EasyFEA.Simulations.HyperElastic`'s and are set per simulation. `absTol` compares a force in the model's own units, so it is scale-dependent: on a large or stiff model it may never fire and `relTol` does the work. `incTol` is the backstop for a residual that stalls on a floor — {py:func}`~EasyFEA.FEM.Operators.NonLinear.GonzalezStressTensor` has one, from the cancellation in its $\Delta W - \bar{\Srm} : \Delta \erm$ term. Exhausting `maxIter` raises an `AssertionError`: a partially converged result is never returned.

```{note}
Convergence is tested on the residual at the iterate *before* `u += delta_u`, so on exit `u` has taken one more step than the last reported norm describes. The returned solution is slightly better than the printed value suggests.
```

### Reading the residual history

The per-iteration norms are saved with the step, so the *shape* of the convergence can be inspected afterwards:

```python
simu.Get_results(-1)["newtonIter"]    # 4
simu.Get_results(-1)["list_norm_r"]   # [3.67e+04, 1.85e+02, 2.72e-01, 6.13e-07]
```

That shape is a diagnostic. Newton is quadratic only if the tangent is exactly $\dpartial{\Rrm}{\urm}$ for the residual being assembled, in which case each norm is roughly the square of the previous one (as above). A tangent that does *not* match its residual still converges — geometrically, at a constant ratio, and to a slightly different answer:

```text
consistent    1.00e+00  2.09e-03  9.54e-09  2.81e-13     each ~ previous²
inconsistent  1.00e+00  3.84e-03  1.62e-05  8.66e-08     constant ratio ~4e-3
```

Both reach the tolerance. If a non-linear model converges but converges *linearly*, the tangent and the residual disagree — check that every operator's residual reached $\Frm_e$ exactly once, and that each tangent went to the slot matching the variable it differentiates.

---
## Use EasyFEA with a Python debugger and an IDE

To step through the solve pipeline with a debugger, install EasyFEA in editable mode so that the source files are used directly (no compiled copies):

```bash
git clone https://github.com/matnoel/EasyFEA.git
cd EasyFEA
python -m pip install -e .
```

With an editable install, breakpoints set inside `EasyFEA/Simulations/_simu.py`, `EasyFEA/FEM/_forms.py`, or any other source file will be hit normally.

```{note}
In editable mode, code-completion may stop working in some IDEs because the
package is not placed in `site-packages`. To restore it:

- **VS Code / Pylance**: add the repository root (the folder containing `EasyFEA/`) to
  *Python › Analysis: Extra Paths* in the Pylance extension settings,
  or add `<repo>/EasyFEA/` to your `PYTHONPATH`.
- **PyCharm**: mark the repository root as a *Sources Root*
  (*right-click → Mark Directory as → Sources Root*).
```

You could start by debugging the {ref}`HelloWorld` example.

## EasyFEA beyond forward solves

The FEM infrastructure — `groupElem.Get_*` functions, `FeArray` arithmetic, Gauss-point integration — is not restricted to `_Simu` subclasses.
You can use it directly to evaluate arbitrary integrals or construct custom operators
over a mesh.

{py:class}`~EasyFEA.Simulations.DIC` (Digital Image Correlation) is the canonical example: it is a full analysis class built on the same mesh and integration machinery, but it never solves a linear system in the traditional sense.
Instead it assembles correlation operators directly from `mesh.Get_*` functions and minimises a correlation functional.
See {doc}`../examples/DIC/index` for worked examples.

This means EasyFEA can serve as a general-purpose FEM toolkit for any computation that benefits from structured Gauss-point integration over a mesh.
