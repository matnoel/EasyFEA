(howto-pipeline)=
# Understand the solve pipeline

The **solve pipeline** is the internal call chain executed each time
`simu.Solve()` is called. Every {py:class}`~EasyFEA.Simulations._Simu` in the
{py:mod}`EasyFEA.Simulations` namespace runs the same pipeline, which lives in
`EasyFEA/Simulations/_simu.py`. This guide traces it from the moment you call
`Solve()` to the moment the solution is stored, aimed at advanced or curious
users who want to understand the internals without reading the full source.

```{note}
The internals described here live in `EasyFEA/Simulations/_simu.py`,
`EasyFEA/Simulations/Solvers.py`, and `EasyFEA/FEM/_forms.py`. The
single-underscore methods (e.g. `_Solver_Apply_Dirichlet`) are advanced API;
the double-underscore ones are private and should never be called directly.
```

---

## Overview

Every call to `simu.Solve()` performs the same three high-level operations:

1. **Build** — assemble the global sparse matrices $(K, C, M, F)$ from element
   integrals (skipped if nothing changed since the last solve).
2. **Apply BCs** — add Neumann contributions to the right-hand side, then
   enforce Dirichlet constraints to form the system $A \, x = b$.
3. **Solve** — pass $A \, x = b$ to the linear algebra backend (scipy, PETSc,
   pypardiso) and store the solution.

```
simu.Solve()
    │
    ├── 1. BUILD ──────────────────────────────────────────────────────────
    │       Get_K_C_M_F()              (cached — rebuilds only if needUpdate)
    │           └─ Assembly()
    │                └─ Construct_local_matrix_system()   ← your code
    │                       element integrals → K_e, C_e, M_e, F_e
    │                   COO scatter → global sparse K, C, M, F
    │
    ├── 2. APPLY BCs ──────────────────────────────────────────────────────
    │       _Solver_Apply_Neumann()    add flux/load contributions to F
    │       _Solver_Apply_Dirichlet()  form A from K/C/M, enforce prescribed DOFs
    │
    └── 3. SOLVE ──────────────────────────────────────────────────────────
            _Solve_Axb()              scipy / PETSc / pypardiso
            _Solver_Update_solutions() recover u, v, a from time scheme
            _Set_solutions()          store solution on the simulation object
```

For **non-linear** problems (`simu.isNonLinear = True`), steps 1–3 are wrapped
in a Newton–Raphson loop that repeats until the residual converges — each
iteration rebuilds the tangent stiffness and solves for the increment $\Delta u$.

---

## Step 1 — Build: matrix assembly

### Caching (`Get_K_C_M_F`)

`Get_K_C_M_F(problemType)` returns the global matrices $(K, C, M, F)$.
It maintains a **dirty flag** (`needUpdate`) and only calls `Assembly()`
when the flag is set. The flag is raised automatically when:

- the mesh geometry changes,
- model material properties change,
- a non-linear iteration updates the solution state.

When nothing has changed, the cached matrices are returned immediately — so
repeated solves with the same system cost almost nothing extra.

```python
# force a manual rebuild on the next Solve() (rarely needed)
simu.Need_Update()
```

### Global assembly (`Assembly`)

`Assembly(problemType)` turns element arrays into global sparse matrices:

1. Calls `Construct_local_matrix_system()` to get element arrays `(K_e, C_e, M_e, F_e)`, each of shape `(Ne, nPe·dof, nPe·dof)`.
2. Retrieves the global DOF index maps (`rows_e`, `columns_e`) from the element group.
3. Scatters all element contributions into global sparse CSR matrices.

### Local integration (`Construct_local_matrix_system`)

This is the method you implement when subclassing
{py:class}`~EasyFEA.Simulations._Simu`. It returns the element-level arrays
for one physics. When using {py:class}`~EasyFEA.Simulations.WeakForms`,
EasyFEA generates it automatically from `@BiLinearForm` / `@LinearForm`
decorators — but you can always implement it directly for full control.

Each `BiLinearForm` evaluates the weak-form integrand at all Gauss points
simultaneously (vectorized over all `Ne` elements) — no Python loops.

---

## Step 2 — Apply boundary conditions

### Neumann (natural) BCs

`_Solver_Apply_Neumann(problemType)` adds all Neumann loads (surface tractions,
line loads, point forces) to the right-hand side vector $F$. These were
registered with `add_neumann`, `add_surfLoad`, etc.

### Dirichlet (essential) BCs

`_Solver_Apply_Dirichlet(problemType, b, resolution)` constructs the effective
system matrix $A$ from the assembled $K$, $C$, $M$ weighted by the
time-integration scheme, then eliminates prescribed DOFs:

| Problem type | System matrix $A$ |
|---|---|
| Elliptic (static) | $A = K$ |
| Parabolic (heat, diffusion) | $A = K + \frac{1}{\alpha \Delta t} C$ |
| Newmark / Midpoint / HHT | $A = c_K K + c_C C + c_M M$ |

For non-linear problems, $A$ is built from the **tangent stiffness** $K_t$,
and prescribed values are converted to incremental form
$\Delta u_D = u_D - u_D^{\text{current}}$ so that the Newton correction is
consistent.

---

## Step 3 — Solve

### Linear solve (`_Solve_Axb`)

`_Solve_Axb` selects a **resolution strategy** based on the BCs present, then
dispatches to the active backend:

| Strategy | When | Approach |
|---|---|---|
| `r1` (default) | Standard Dirichlet | Partition into known/unknown DOFs; solve reduced system |
| `r2` | Lagrange multiplier BCs | Augment system with Lagrange multipliers |

| Backend | When active |
|---|---|
| `scipy` (default) | Serial, no MPI |
| `pypardiso` | Serial, Intel MKL available |
| `petsc4py` | MPI (`MPI_SIZE > 1`) or set manually |

In parallel, PETSc solves the distributed system and EasyFEA performs an
`Allreduce` so that every rank holds the same complete DOF solution — no
manual gather needed inside the solve loop.

### Recover and store (`_Solver_Update_solutions` + `_Set_solutions`)

`_Solver_Update_solutions` applies the active time-integration scheme to
recover all solution fields from the solved $u$:

| Scheme | Recovers |
|---|---|
| Elliptic | $u$ only |
| Parabolic | $u$, $\dot{u}$ (via backward Euler / Crank–Nicolson) |
| Newmark / HHT / Midpoint | $u$, $\dot{u}$, $\ddot{u}$ |

`_Set_solutions` then stores all fields on the simulation object
(`simu.u`, `simu.v`, `simu.a`).

---

## Non-linear problems: Newton–Raphson loop

When `isNonLinear = True`, steps 1–3 are wrapped in a loop. The loop solves
for an **increment** $\Delta u$ at each iteration and accumulates it:

```
u ← Get_x0()           start from the current solution state

REPEAT:
    Need_Update()           mark K as dirty so it is rebuilt with current u
    Δu = Solve_simu(...)    solve  Kt Δu = −R(u)
    u += Δu                 accumulate the Newton correction
    check ‖R‖ < tolConv     convergence on residual norm
      OR  ‖Δu‖ < 1e-11      convergence on increment norm
UNTIL converged OR maxIter reached
```

`Construct_local_matrix_system` must return:

- `K_e` — the **tangent stiffness** $K_t = \partial R / \partial u$
- `F_e` — the **residual** $R(u)$

---

## Iteration management (`Save_Iter` / `Set_Iter`)

After each time step or load increment, call
{py:meth}`~EasyFEA.Simulations._Simu.Save_Iter` to snapshot the current
solution:

```python
simu.Save_Iter()   # stores u (and v, a if present) for this step
```

Derived quantities (stress, strain, …) are **not** stored — they are
recomputed on demand by {py:meth}`~EasyFEA.Simulations._Simu.Result`.

To restore a previous state (e.g. for post-processing or animation):

```python
simu.Set_Iter(i)   # restores u, v, a to the values saved at step i
```

```{important}
If the simulation uses **different meshes across iterations** (adaptive
remeshing, crack propagation, phase-field with refinement), `Set_Iter`
also restores the mesh that was active at step `i`. No manual mesh switching
is needed when replaying history or generating animations.
```

---

## EasyFEA beyond forward solves

The FEM infrastructure — `mesh.Get_*` functions, `FeArray` arithmetic,
Gauss-point integration — is not restricted to `_Simu` subclasses. You can
use it directly to evaluate arbitrary integrals or construct custom operators
over a mesh, without ever calling `Solve()`.

{py:class}`~EasyFEA.Simulations.DIC` (Digital Image Correlation) is the
canonical example: it is a full analysis class built on the same mesh and
integration machinery, but it never solves a linear system in the traditional
sense. Instead it assembles correlation operators directly from `mesh.Get_*`
functions and minimises a correlation functional.
See {doc}`../examples/DIC/index` for worked examples.

This means EasyFEA can serve as a general-purpose FEM toolkit for any
computation that benefits from structured Gauss-point integration over a mesh,
not only for forward PDE solves.
