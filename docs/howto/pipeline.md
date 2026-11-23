(howto-pipeline)=

# Understand the solve pipeline

The **solve pipeline** is the internal call chain executed each time
{py:meth}`~EasyFEA.Simulations._Simu.Solve` is called. Every
{py:class}`~EasyFEA.Simulations._Simu` in the {py:mod}`EasyFEA.Simulations` namespace
runs the same pipeline, which lives in {py:class}`EasyFEA.Simulations._Simu`. This guide
traces it from the moment you call `Solve()` to the moment the solution is stored, aimed
at advanced or curious users who want to understand the internals without reading the
full source.

```{note}
The internals described here live in `EasyFEA/Simulations/_simu.py`,
`EasyFEA/Simulations/Problem/` (terms, assembly, dof layout, `Problem`),
`EasyFEA/Solvers/`, and `EasyFEA/FEM/Operators/`. The
single-underscore methods (e.g. `_Solver_Get_A_b`) are advanced API;
the double-underscore ones are private and should never be called directly.
```

______________________________________________________________________

## Overview

A simulation solves one or several **problems**: each is its fields solved together as
one system $[\urm \mid p \mid \dots]$ (a single field by default, `[u | p]` for
{py:class}`~EasyFEA.Simulations.IncompressibleHyperElastic`). For each of them
{py:meth}`~EasyFEA.Simulations._Simu.Solve` runs four steps:

1. **Integrate** — the simulation's {py:class}`~EasyFEA.Simulations.Term` objects
   (`Get_terms`) are integrated over the mesh groups into **contributions**: one element
   array per term, group and slot, kept unsummed.
2. **Assemble** — the contributions are summed straight into the system matrix
   $\Arm = \alpha_\Krm\,\Krm + \alpha_\Crm\,\Crm + \alpha_\Mrm\,\Mrm$ and the right-hand
   side $\brm$ of $\Arm \, \xrm = \brm$. No global $\Krm$, $\Crm$ or $\Mrm$ is built.
3. **Constrain and solve** — {py:meth}`~EasyFEA.Solvers.System.Solve` applies the
   Dirichlet and Lagrange conditions and passes the system to the linear algebra backend
   (scipy, PETSc, pypardiso).
4. **Store** — the solution is committed, ready for `Save_Iter`.

A **linear** problem keeps its contributions and $\Arm$ until something marks them
stale, so steps 1–2 are skipped on the next solve. A **non-linear** one (a strategy set
with `simu.Solver_Set_Nonlinear_Strategy(Newton())`, then `simu.isNonLinear` is `True`)
integrates again at each iterate: the strategy repeats steps 1–3 until the residual
converges, see {ref}`howto-pipeline-newton`.

______________________________________________________________________

(howto-pipeline-operators)=

## From element operators to the global system

### Terms and contributions

Each {py:class}`~EasyFEA.Simulations._Simu` implements `Get_terms`, which returns a list
of {py:class}`~EasyFEA.Simulations.Term`: an **operator**, the fields it couples and a
scale. {py:func}`~EasyFEA.Simulations.Integrate_terms` evaluates each operator on every
mesh group and returns one **contribution** per slot it fills:

| Slot | Element array                                          | Reaches                             |
| ---- | ------------------------------------------------------ | ----------------------------------- |
| `K`  | $\Krm_e$, stiffness (the tangent, for non-linear laws) | $\Arm$, and $\brm$ through $\urm_t$ |
| `C`  | $\Crm_e$, damping                                      | $\Arm$, and $\brm$ through $\vrm_t$ |
| `M`  | $\Mrm_e$, mass                                         | $\Arm$, and $\brm$ through $\arm_t$ |
| `F`  | $\Frm_e$, an external load (a source)                  | $\brm$, added                       |
| `R`  | $\Rrm_e$, an internal force, non-linear terms only     | $\brm$, subtracted                  |

The operators are small functions in {py:mod}`EasyFEA.FEM.Operators` that integrate a
form over the Gauss points: {py:mod}`~EasyFEA.FEM.Operators.Bilinear` (e.g.
$\int \Brm^\top \, \bf{C} \, \Brm \, \dO$), {py:mod}`~EasyFEA.FEM.Operators.Linear` and
{py:mod}`~EasyFEA.FEM.Operators.NonLinear`. A contribution carries its slot, its block
`(rows, cols)` of fields, its mesh group and the element array times the thickness and
the term's scale.

### Assembly

The `Problem` of a monolithic group of fields numbers its dofs end to end (its **dof
layout**) and sums the contributions on a sparse pattern it builds once per set of
contributing blocks. For the time scheme's weights
$(\alpha_\Krm, \alpha_\Crm, \alpha_\Mrm)$, its `coefs` (Newmark, HHT, midpoint, …; a
static problem has $\alpha_\Crm = \alpha_\Mrm = 0$):

$$\Arm = \alpha_\Krm\,\Krm + \alpha_\Crm\,\Crm + \alpha_\Mrm\,\Mrm$$

The right-hand side is the external load (the Neumann conditions plus the `F`
contributions) minus every internal force at the iterate. Every `K`, `C` or `M`
contribution is a linear term: it contributes $\Krm_e \, \urm_t + \Crm_e \, \vrm_t +
\Mrm_e \, \arm_t$, the fields the time scheme evaluates at the unknown. An `R`
contribution is a non-linear term's own internal force, as described
{ref}`below <howto-pipeline-nonlinear-operators>`.

The separate global matrices remain available for post-processing:
{py:meth}`~EasyFEA.Simulations._Simu.Assembly` returns $\Krm, \Crm, \Mrm, \Frm$ of a
problem, built on each call and never read by a solve.

### Constraints

Dirichlet conditions are not written into $\Arm$ and $\brm$.
{py:meth}`~EasyFEA.Solvers.System.Solve` receives them as
{py:class}`~EasyFEA.Solvers.Constraints`, the known dofs with their values plus the
Lagrange constraints, and applies them to the system it solves: the known dofs leave the
unknowns and their values are lifted into $\brm$ (`ResolType.r1`); a Lagrange constraint
(`r2`) adds one row; the penalty (`r3`) puts 1 on the diagonal of the known rows.

(howto-pipeline-nonlinear-operators)=

### Non-linear assembly

A non-linear problem solves for the **increment**, not for $\urm^{n+1}$:

$$\Arm(\urm) \, \Delta \urm = - \Rrm(\urm)$$

The right-hand side is the load minus every internal force at the current iterate: a
linear term's $\Krm_e \, \urm_t + \Crm_e \, \vrm_t + \Mrm_e \, \arm_t$, a non-linear
term's own $\Rrm_e$. A non-linear term's matrices go to $\Arm$ only, so its $\Rrm_e$
must be **complete**, damping included.

Every operator in {py:mod}`~EasyFEA.FEM.Operators.NonLinear` is built for this and
returns its own $(\text{tangent}, \text{residual})$ pair, evaluated at the current
iterate. Assembly is then the same two moves for each of them, whatever the physics:

- the **residual** goes to the $\Rrm$ slot, subtracted from the load;
- the **tangent** goes to the slot carrying the weight it needs — $\Krm_e$ for a
  derivative with respect to $\urm$, $\Crm_e$ with respect to $\vrm$, $\Mrm_e$ with
  respect to $\arm$ — so that the single
  $\Arm = \alpha_\Krm\,\Krm + \alpha_\Crm\,\Crm + \alpha_\Mrm\,\Mrm$ combination weights
  it correctly, with no time-scheme special-casing.

| Operator                                                                     | Contributes                                                                                                                                             |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| {py:func}`~EasyFEA.FEM.Operators.NonLinear.SecondPiolaKirchhoffStressTensor` | consistent tangent (material + geometric) and internal residual $\int \Brm^\top \boldsymbol{\Sigma} \, \dO$                                             |
| {py:func}`~EasyFEA.FEM.Operators.NonLinear.ActiveStressTensor`               | fiber active stress $\tau \, (\hat{\Tb} \otimes \hat{\Tb})$ — geometric tangent and residual, no material tangent                                       |
| {py:func}`~EasyFEA.FEM.Operators.NonLinear.KelvinVoigtDamping`               | viscous stress $\boldsymbol{\Sigma}_{\text{visco}} = \eta \dot{\Erm}$ — configuration tangent and residual, plus the damping matrix for the $\Crm$ slot |
| {py:func}`~EasyFEA.FEM.Operators.NonLinear.FollowingPressure`                | pressure tracking the deformed normal — non-symmetric tangent and follower load                                                                         |
| {py:func}`~EasyFEA.FEM.Operators.NonLinear.PenaltyContact`                   | penalty tangent on the active contact set, and the force pushing the body out of the obstacle                                                           |

______________________________________________________________________

(howto-pipeline-newton)=

## Non-linear convergence

{py:meth}`~EasyFEA.Simulations._Simu._Solver_Solve_problemType` solves a linear problem
with one {py:meth}`~EasyFEA.Solvers.System.Solve` of $\Arm \, \mathrm{x} = \brm$, its
known dofs at the Dirichlet values. A nonlinear one goes through its strategy, set by
`Solver_Set_Nonlinear_Strategy`, e.g. {py:class}`~EasyFEA.Solvers.Newton`, given the
problem (`problem.Get_A_b(x)`, `problem.Constraints(x)`); it iterates from the last
converged solution, or from `simu.Solve(x0=...)`, repeating three moves — reassemble at
the current iterate, solve for the correction, add it:

```python
for _ in range(maxIter):
    A, b = problem.Get_A_b(x)                   # integrated at x
    dx, norm = system.Solve(
        A,
        b,
        x,
        problem.Constraints(x),
        backend,
        comm,
    )
    x += dx
    if norm < absTol or relNorm < relTol or system.Norm(dx, comm) < incTol:
        return x, norms
```

Integrating at x is what makes this Newton rather than a fixed-point iteration: a
nonlinear problem keeps no contributions, so each `Get_A_b(x)` re-evaluates the terms,
giving the tangent **and** the residual at the new iterate, and a reader after the solve
integrates again at the committed state. A linear problem keeps its contributions and
$\Arm$ until a change marks them stale or the time-scheme coefficients change. The
iterate reaches them as the trial solution,
{py:data}`~EasyFEA.Simulations.UnknownValues`: a dict of each field's value, passed to
{py:meth}`~EasyFEA.Simulations._Simu.Get_terms` — the solved fields at `x`, the others
committed. {py:meth}`~EasyFEA.Simulations._Simu.Get_u_v_a` evaluates a field of it at
the time scheme's point, as node values, the layout the operators read.

### What the norm measures

Not $\|\Delta \urm\|$, and not the raw right-hand side: `norm` is the norm of the
**reduced** right-hand side, over the free dofs, after the prescribed ones have been
eliminated. Restricting to free dofs is deliberate — at a prescribed dof the residual is
the reaction, which is not an equilibrium error and never vanishes. Under MPI the
per-rank contributions are reduced across ranks.

Because the unknown is $\Delta \urm$, the prescribed values are incremental too: the
solve subtracts the current iterate and imposes $\urm_{\text{target}} - \urm$. The full
boundary jump is therefore applied on the first iteration and decays to zero as $\urm$
reaches the target — which is why the first residual is large, and why $\text{relNorm}$
reads as a fraction of the load step's initial imbalance.

### Stopping

Three criteria, any one of which is enough:

| Criterion               | Default | Meaning                               |
| ----------------------- | ------- | ------------------------------------- |
| `norm < absTol`         | `1e-6`  | absolute residual                     |
| `relNorm < relTol`      | `1e-10` | residual relative to the step's first |
| `norm_delta_u < incTol` | `1e-11` | the correction has stopped moving     |

The defaults are {py:class}`~EasyFEA.Simulations.HyperElastic`'s and are set per
simulation. `absTol` compares a force in the model's own units, so it is
scale-dependent: on a large or stiff model it may never fire and `relTol` does the work.
`incTol` is the backstop for a residual that stalls on a round-off floor. Exhausting
`maxIter` raises {py:class}`~EasyFEA.Solvers.NotConvergedError`, an `AssertionError`: a
partially converged result is never returned.

```{note}
Convergence is tested on the residual at the iterate *before* `u += delta_u`, so on exit `u` has taken one more step than the last reported norm describes. The returned solution is slightly better than the printed value suggests.
```

### Reading the residual history

The per-iteration norms are saved with the step, so the *shape* of the convergence can
be inspected afterwards:

```python
simu.Get_results(-1)["newtonIter"]    # 4
simu.Get_results(-1)["list_norm_r"]   # [3.67e+04, 1.85e+02, 2.72e-01, 6.13e-07]
```

That shape is a diagnostic. Newton is quadratic only if the tangent is exactly
$\dpartial{\Rrm}{\urm}$ for the residual being assembled, in which case each norm is
roughly the square of the previous one (as above). A tangent that does *not* match its
residual still converges — geometrically, at a constant ratio, and to a slightly
different answer:

```text
consistent    1.00e+00  2.09e-03  9.54e-09  2.81e-13     each ~ previous²
inconsistent  1.00e+00  3.84e-03  1.62e-05  8.66e-08     constant ratio ~4e-3
```

Both reach the tolerance. If a non-linear model converges but converges *linearly*, the
tangent and the residual disagree — check that every operator's residual reached
$\Frm_e$ exactly once, and that each tangent went to the slot matching the variable it
differentiates.

______________________________________________________________________

## Use EasyFEA with a Python debugger and an IDE

To step through the solve pipeline with a debugger, install EasyFEA in editable mode so
that the source files are used directly (no compiled copies):

```bash
git clone https://github.com/matnoel/EasyFEA.git
cd EasyFEA
python -m pip install -e .
```

With an editable install, breakpoints set inside `EasyFEA/Simulations/_simu.py`,
`EasyFEA/FEM/_forms.py`, or any other source file will be hit normally.

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

The FEM infrastructure — `groupElem.Get_*` functions, `FeArray` arithmetic, Gauss-point
integration — is not restricted to `_Simu` subclasses. You can use it directly to
evaluate arbitrary integrals or construct custom operators over a mesh.

{py:class}`~EasyFEA.Simulations.DIC` (Digital Image Correlation) is the canonical
example: it is a full analysis class built on the same mesh and integration machinery,
but it never solves a linear system in the traditional sense. Instead it assembles
correlation operators directly from `mesh.Get_*` functions and minimises a correlation
functional. See {doc}`../examples/DIC/index` for worked examples.

This means EasyFEA can serve as a general-purpose FEM toolkit for any computation that
benefits from structured Gauss-point integration over a mesh.
