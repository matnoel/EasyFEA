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

For **non-linear** problems (`simu.isNonLinear = True`), steps 1–3 are wrapped in a Newton–Raphson loop that repeats until the residual converges — see {py:meth}`~EasyFEA.Simulations._Simu._Solver_Solve_Newton_Raphson` for more information.

---

(howto-pipeline-operators)=
## From element operators to the global system

**Build** turns the mesh + model into four global sparse matrices. Each {py:class}`~EasyFEA.Simulations._Simu` implements `Construct_local_matrix_system`, which loops over the mesh element groups and returns, per group, the element tuple $(\Krm_e, \Crm_e, \Mrm_e, \Frm_e)$ — stiffness (the tangent, for non-linear laws), damping, mass, and the load / internal-force vector.

These element matrices come from **operators** in {py:mod}`EasyFEA.FEM.Operators` — small functions that integrate a form over the Gauss points: {py:mod}`~EasyFEA.FEM.Operators.Bilinear` (e.g. $\int \Brm^\top \, \bf{C} \, \Brm \, \dO$), {py:mod}`~EasyFEA.FEM.Operators.Linear`, and {py:mod}`~EasyFEA.FEM.Operators.NonLinear`. The simulation assembles them into the global $\Krm, \Crm, \Mrm, \Frm$ and combines $\Krm, \Crm, \Mrm$ into the system matrix according to the time scheme:

$$\Arm = \alpha_\Krm\,\Krm + \alpha_\Crm\,\Crm + \alpha_\Mrm\,\Mrm$$

where $(\alpha_\Krm, \alpha_\Crm, \alpha_\Mrm)$ are the scheme-dependent weights for the active `AlgoType` (Newmark, HHT, midpoint, …); a static problem uses $\alpha_\Crm = \alpha_\Mrm = 0$.

(howto-pipeline-hyperelastic-operators)=
### Hyperelastic (non-linear) operators

For {py:class}`~EasyFEA.Simulations.HyperElastic`, each Newton iterate builds the element tuple from two operators evaluated at the current displacement:

- {py:func}`~EasyFEA.FEM.Operators.NonLinear.SecondPiolaKirchhoffStressTensor` returns the consistent tangent $\Krm_e$ (material + geometric) and the internal residual $\Frm_e$.
- {py:func}`~EasyFEA.FEM.Operators.NonLinear.KelvinVoigtDamping` returns the large-strain Kelvin–Voigt viscous contributions $(\Crm_e, \Krm_e^{\text{geo}})$ when `material.eta > 0` and a velocity field is available (otherwise `(None, None)`):
    - $\Crm_e = \mathrm{thickness} \cdot \eta \int \Brm^\top \Brm \, \dO$ — the **damping matrix**, placed in slot 2. It carries the viscous residual ($\brm \mathrel{-}= \Crm \, \vrm_t$) and rides $\alpha_\Crm$.
    - $\Krm_e^{\text{geo}}$ — the **configuration tangent** $\dpartial{(\Crm \vrm)}{\urm}$ at fixed velocity (geometric stiffening from the viscous second Piola-Kirchhoff stress $\boldsymbol{\Sigma}_{\text{visco}} = \eta \dot{\Erm}$ plus the $\dpartial{\dot{\Erm}}{\urm}$ term). It is added to $\Krm_e$, so it rides $\alpha_\Krm$.

Splitting the viscous tangent this way is what lets the damping (which must ride $\alpha_\Crm$) and its configuration derivative (which must ride $\alpha_\Krm$) be weighted correctly by the single $\alpha_\Krm\,\Krm + \alpha_\Crm\,\Crm + \alpha_\Mrm\,\Mrm$ assembly, with no time-scheme special-casing.

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
