(howto-new-simulation)=

# Create a custom simulation

A **simulation** drives the full assembly–solve–store pipeline for a given physics.
Every simulation inherits from {py:class}`~EasyFEA.Simulations._Simu` and is accessible
in the {py:mod}`EasyFEA.Simulations` namespace.

```{tip}
**Most users will never need this guide.** EasyFEA ships with ready-to-use
simulation classes that cover the vast majority of common problems:

| Class | Physics |
|---|---|
| {py:class}`~EasyFEA.Simulations.Elastic` | Linear elasticity (static, dynamic, contact) |
| {py:class}`~EasyFEA.Simulations.Thermal` | Heat conduction (static, transient) |
| {py:class}`~EasyFEA.Simulations.PhaseField` | Phase-field brittle fracture |
| {py:class}`~EasyFEA.Simulations.Beam` | Euler–Bernoulli and Timoshenko beams |
| {py:class}`~EasyFEA.Simulations.DIC` | Digital Image Correlation |
| {py:class}`~EasyFEA.Simulations.HyperElastic` | Hyperelasticity (static, dynamic) |
| {py:class}`~EasyFEA.Simulations.IncompressibleHyperElastic` | Perfectly incompressible hyperelasticity, `[u \| p]` (static) |

If your physics is not listed above, consider
[opening an issue](https://github.com/matnoel/EasyFEA/issues) to propose it
before writing custom code — new physics contributions are very welcome.
```

For problems outside the built-in classes, EasyFEA provides three extension points, from
easiest to most flexible:

1. **{py:class}`~EasyFEA.Simulations.WeakForms`** — define any PDE in variational form
   with a few lines of Python. Covers scalar problems (Poisson), vector problems
   (elasticity), and transient or non-linear problems. No FEM assembly knowledge
   required.
2. **Extend an existing simulation** — when a built-in class already covers most of your
   physics and you only need to add extra terms (a boundary contribution, a penalty, a
   coupling), subclass it and compose them into
   {py:meth}`~EasyFEA.Simulations._Simu.Get_terms`. See
   {ref}`howto-new-simulation-extend`.
3. **Subclass {py:class}`~EasyFEA.Simulations._Simu`** — provides full control over the
   assembly at the element level for problems that are difficult to model in
   {py:class}`~EasyFEA.Simulations.WeakForms`, or to improve performance. Knowledge of
   finite element methods is required.

EasyFEA supports multi-physics problems such as phase-field fracture simulations, which
couple an elastic sub-problem with a damage sub-problem via a staggered algorithm: each
sub-problem is solved in turn with the other held fixed, and the two are iterated to
convergence within each load step. This pattern is already implemented in
{py:class}`~EasyFEA.Simulations.PhaseField`. Monolithic coupling—assembling all physics
into a single global system—is not currently implemented, but there is no fundamental
limitation preventing it.

```{note}
{py:class}`~EasyFEA.Simulations.WeakForms` is still evolving. Contributions
are welcome to extend its capabilities — in particular, support for contact
mechanics and elastoplastic simulations would benefit greatly from community
involvement. See the
[Contributing Guide](https://github.com/matnoel/EasyFEA/blob/main/CONTRIBUTING.md)
if you would like to help.
```

______________________________________________________________________

## The weak form approach

A weak form is defined by:

- a **bilinear form** $a(u, v)$ that produces the stiffness matrix $\Krm$,
- a **linear form** $\ell(v)$ that produces the load vector $\Frm$.

EasyFEA assembles $\Krm$ and $\Frm$ automatically from these forms using Gauss
quadrature over the mesh elements.

### Available operators

{py:class}`~EasyFEA.FEM.Field` represents the unknown and test fields. The following
operators from `EasyFEA.FEM` act on `Field` objects and return
{py:class}`~EasyFEA.FEM.FeArray` tensors:

| Operator           | Description                                                               |
| ------------------ | ------------------------------------------------------------------------- |
| `u.grad`           | Gradient $\nabla u$ — shape `(Ne, pg, dof_n, dim)`                        |
| `Sym_Grad(u)`      | Symmetric gradient $\frac{1}{2}(\nabla u + \nabla u^\top)$                |
| `Trace(A)`         | Trace of a square `FeArray`                                               |
| `Transpose(A)`     | Transpose of a `FeArray`                                                  |
| `Det(A)`           | Determinant of a square `FeArray`                                         |
| `Inv(A)`           | Inverse of a square `FeArray`                                             |
| `TensorProd(a, b)` | Tensor (outer) product $a \otimes b$                                      |
| `Norm(a)`          | Euclidean norm                                                            |
| `A.dot(B)`         | Contracted product (inner product for vectors, matrix–vector for tensors) |
| `A.ddot(B)`        | Double contraction $A : B$ — used for stress–strain products              |

All operators act element- and Gauss-point-wise over arrays of shape `(Ne, pg, ...)`, so
**no Python loops are needed** over elements or integration points.

All weak-form-based simulations are available in {ref}`easyfea-examples-weak-forms`.

______________________________________________________________________

(howto-new-simulation-extend)=

## Extend an existing simulation

When a built-in simulation already covers most of your physics, subclass it and add
{py:class}`~EasyFEA.Simulations.Term` objects to
{py:meth}`~EasyFEA.Simulations._Simu.Get_terms`. The list is rebuilt at every assembly,
so a value that changes between steps is a parameter of the subclass, read there. As in
the cardiac examples'
[utils.py](https://github.com/matnoel/EasyFEA/blob/main/examples/CardiacElastoDynamics/utils.py),
a {py:class}`~EasyFEA.Simulations.HyperElastic` plus a surface spring and a follower
pressure:

```python
from EasyFEA import MatrixType, Simulations
from EasyFEA.Simulations import ProblemType, Term, UnknownValues
from EasyFEA.FEM import Operators
from EasyFEA.Utilities import _params

class Ventricle(Simulations.HyperElastic):

    pressure = _params.ScalarParameter()   # a write marks the matrices outdated

    def __init__(self, mesh, material):
        super().__init__(mesh, material)
        self.pressure = 0.0   # a parameter has no default: reading it unset raises

    def Get_terms(
        self, x: UnknownValues, problemType: ProblemType | None = None
    ) -> list[Term]:
        # `dim=2` picks the surface groups, `tag` the subset within them;
        # `constant=True` builds the contribution once and reuses it while its arguments are unchanged
        epi = Term("K", Operators.Bilinear.MassAlongNormal, dim=2, tag="epi", constant=True)
        # the iterate, by node, at the time scheme's evaluation point
        u = self.Get_u_v_a(problemType, x)[0]
        return super().Get_terms(x, problemType) + [
            epi.Scaled(1e8),          # stiffness
            epi.Scaled(5e3, "C"),     # damping, from the same integration
            # `groupElem` and, since `tag` is set, `elements` come from the integration
            Term(
                "KR",
                Operators.NonLinear.FollowingPressure,
                dim=2,
                tag="endo",
                u=u,
                pressure=self.pressure,
                matrixType=MatrixType.mass,
            ),
        ]

simu = Ventricle(mesh, material)
for t in times:
    simu.pressure = pressure_at(t)
    simu.Solve()
```

Ready-made operators live in {py:mod}`EasyFEA.FEM.Operators` (`Bilinear`, `Linear`,
`NonLinear`) — see {ref}`fem-operators`. The contact example extends
{py:class}`~EasyFEA.Simulations.Elastic` the same way:

```python
from EasyFEA.Solvers import Newton

class RigidContact(Simulations.Elastic):

    penalty = _params.ScalarParameter()

    def __init__(self, mesh, model, penalty):
        super().__init__(mesh, model)
        # contact makes the problem nonlinear: Newton-Raphson, so the `R` slot is used
        self.Solver_Set_Nonlinear_Strategy(Newton(absTol=1e-5, maxIter=50))
        self.penalty = penalty

    def Get_terms(
        self, x: UnknownValues, problemType: ProblemType | None = None
    ) -> list[Term]:
        u = self.Get_u_v_a(problemType, x)[0]
        return super().Get_terms(x, problemType) + [
            Term("KR", self.__Contact, dim=self.dim - 1, u=u)
        ]

    def __Contact(self, groupElem, u):
        gap_e_pg, normal_e_pg = self.__Gap_and_normal(groupElem, u)
        return Operators.NonLinear.PenaltyContact(
            groupElem, self.penalty, gap_e_pg, normal_e_pg
        )
```

`__Contact` shows the shape to use when an operator's arguments depend on the element
group — a hyperelastic state, a contact projection: a named method
`(groupElem, ...) -> array`, or a tuple of arrays.

______________________________________________________________________

(howto-new-simulation-subclass)=

## Subclass `_Simu` for a fully custom simulation

If your problem requires custom assembly logic that cannot be expressed as a weak form,
subclass {py:class}`~EasyFEA.Simulations._Simu` directly.
{py:class}`~EasyFEA.Simulations.Thermal` is the simplest existing subclass and is a good
starting point; consult
[\_thermal.py](https://github.com/matnoel/EasyFEA/blob/main/EasyFEA/Simulations/_thermal.py)
for implementation details.

The complete interface to implement (abstract unless marked optional; `Get_dof_n`
defaults to the number of unknowns):

```python
import numpy as np
from EasyFEA.Simulations import _Simu, ProblemType, Term, UnknownValues

class MySimulation(_Simu):

    U = ProblemType("my_problem")
    problemTypes = (U,)  # the first is the default, `problemType`
    dofFields = {U: ("u", "v", "a")}  # see below

    def __init__(self, mesh, model, folder="", verbosity=False):
        super().__init__(mesh, model, folder, verbosity)

    # --- problem definition ---

    def Get_unknowns(self, problemType=None) -> list[str]:
        ...  # Get_dof_n is their number

    # --- physics (see below for details) ---

    def Get_terms(self, x: UnknownValues, problemType=None) -> list[Term]:
        ...

    # --- post-processing ---

    def Results_nodeFields_elementFields(self, details=False):
        ...

    # --- optional: the dof-field results of `dofFields` by default, extend them ---

    def Results_Available(self) -> list[str]:
        return [*super().Results_Available(), "my_result"]

    def _Result(self, result: str, nodeValues=True):
        if result == "my_result":
            ...
        return super()._Result(result, nodeValues)

    # --- optional: no summary, no energy, zero displacement by default ---

    def Results_Iter_Summary(self):
        ...

    def Results_dict_Energy(self) -> dict[str, float]:
        ...

    def Results_displacement_matrix(self):
        ...
```

### Declaring `problemTypes` and `dofFields`

`problemTypes` lists the problems the simulation solves, the first being its default
`problemType`. `dofFields` names, for each of them, its fields in a saved iteration, by
role: u, then v (velocity) and a (acceleration) when the simulation has them:

```python
problemTypes = (U,)
dofFields = {U: ("displacement", "speed", "accel")}
```

`Save_Iter` stores the roles the algorithm integrates (u when elliptic, u and v when
parabolic, u, v and a when hyperbolic) and `Set_Iter` restores them, a role not saved as
zero: neither is yours to write. A problem without time derivatives declares u alone,
e.g. `{U: ("displacement",), P: ("pressure",)}`. The names are the keys of
`simu.Get_results(i)`, and the base results: each name the scheme integrates, then
`<name>_norm` and one component per unknown, `<role><unknown>` (`ux`, `vy`), when a node
carries several dofs.

### Internal and external variables

A simulation whose state is more than its dof fields declares it, and `_Simu` saves,
restores and commits it:

```python
def _Trial_internals(self) -> dict[_GroupElem, dict[str, FeArray]]:
    # PhaseField: the history field H, the running max of psi+ with the committed H
    internals = {}
    for groupElem in self._Get_FunctionSpace().Get_list_groupElem():
        psiP = self.__Calc_psiPlus_e_pg(groupElem)  # of the current displacement
        H = self._variables.internals.get(groupElem, {}).get("H")  # none before a commit
        internals[groupElem] = {"H": psiP if H is None else np.maximum(psiP, H)}
    return internals

@property
def _externalNames(self) -> tuple[str, ...]:
    # e.g. a temperature, given by simu.Set_external(T=...) and read as self.external
    return ("T",)
```

At the end of a converged `Solve`, `_Commit_Internal_and_External_states()` makes the
trial internal variables and the external variables set the committed
`self._variables.internals` and `.externals`, which the next solve integrates from and
results read; a `Solve` override calls `_Commit_Internal_and_External_states()` once
converged. `Save_Iter` stores the committed ones, by kind, beside the dof fields and the
scalars (`simu.Get_results(i)` shows them as `internal/<elemType>/<name>` and
`external/<name>`), `Set_Iter` restores copies and drops every problem's contributions.

Key the internal variables by the groups of the default problem's space, on which
`Set_Iter` restores them. {py:class}`~EasyFEA.Simulations.PhaseField` is the full
example: beside the `_Trial_internals` above, its staggered `Solve` records its loop
with `_Iter_Record(...)`, stored by the next `Save_Iter` and restored by `Set_Iter`, and
calls `_Commit_Internal_and_External_states()` once converged.

The scalars of a saved iteration (the record, `indexMesh`) are written with it and kept
in memory too: `Results_Iter_Summary` reads them with `_Get_results_scalars(i)`, which
opens no result file; PhaseField records `damageMax` for its summary.

### Reading a past iteration

`_Result` reads the current state only. `Result` checks the name and, given `iter`,
reads inside `_Iter_Visit()`: `Set_Iter` moves the simulation, and on exit every
attribute is bound back to what it was. Rebind any state the subclass derives
(`self.__K = ...`), never write it in place: an in-place write outlives the read.

### Implementing `Get_terms`

`Get_terms` is the only method where you provide physics-specific data: it returns the
list of {py:class}`~EasyFEA.Simulations.Term` objects that make up the local matrix
system, at the trial solution `x` ({py:data}`~EasyFEA.Simulations.UnknownValues`: each
field's value, the iterate during a Newton solve, the committed state otherwise). From
the element-level matrices they produce, `_Simu` automatically:

1. assembles the global sparse system $\Krm$, $\Crm$, $\Mrm$, $\Frm$,
2. applies boundary conditions,
3. selects the appropriate combination of $\Krm$, $\Crm$, $\Mrm$ for the active
   time-integration algorithm,
4. solves the resulting linear system $\Krm \urm + \Crm \vrm + \Mrm \arm = \Frm$ (where
   $\vrm$ and $\arm$ are the velocity and acceleration computed by the time scheme).

**Your only responsibility is to declare the right terms.** Everything else is handled
by `_Simu` internally.

```{seealso}
- {ref}`howto-pipeline`
```

A {py:class}`~EasyFEA.Simulations.Term` names one {ref}`operator <fem-operators>` and,
with a string of slot letters, where each array it returns belongs — one letter per
array:

| Letter | Goes to | Meaning                                        |
| ------ | ------- | ---------------------------------------------- |
| `K`    | $\Krm$  | stiffness, or the tangent of a non-linear term |
| `C`    | $\Crm$  | damping (parabolic / hyperbolic)               |
| `M`    | $\Mrm$  | mass (hyperbolic only)                         |
| `F`    | $\Frm$  | an external load, added as-is                  |
| `R`    | $\Frm$  | an **internal** force, so it is subtracted     |

So `Term("K", op)` is a stiffness, `Term("KR", op)` an operator returning a tangent
*and* an internal force, `Term("KF", op)` a tangent and a load.

{py:func}`~EasyFEA.Simulations.Integrate_terms` takes care of the rest: contributions
**accumulate** when several terms target one element group, the 2D `thickness` is
applied once, and a term filling a **single** slot also gets its residual —
$-\Krm_e \urm_t$, $-\Crm_e \vrm_t$, $-\Mrm_e \arm_t$ — contracted for it, since one slot
means the contribution is linear in that field. Add `constant=True` when a contribution
does not depend on the solution: it is then built once and reused across Newton
iterations and time steps, until one of its arguments changes. So a constant term must
get everything it reads through its arguments — a plain operator such as `Bilinear.UV`
with `coef=self.rho`, never a bound method, which `Term` refuses.

To inspect what it built, filter its flat list of contributions, one per term, element
group and slot, unsummed:

```python
from EasyFEA.Simulations import Integrate_terms

x = {problemType: simu._Get_u_n(problemType)}  # the trial solution to integrate at
list_contribution = Integrate_terms(simu, simu.Get_terms(x, problemType), problemType)
K_es = [
    contribution.array
    for contribution in list_contribution
    if contribution.slot == "K" and contribution.block == (rows, cols)
]
coefs = [
    contribution.coef for contribution in list_contribution if contribution.slot == "K"
]
```

`rows` and `cols` are both `problemType` unless the term's `problemTypes` says
otherwise; `contribution.coef` (thickness times the term's scale) is not applied to
`contribution.array`.

```{warning}
A **non-linear** problem needs no special handling here: put the tangent in `K`
and the internal force in `R`, and the residual follows. See
[`HyperElastic.Get_terms`](https://github.com/matnoel/EasyFEA/blob/main/EasyFEA/Simulations/_hyperelastic.py)
for a worked finite-deformation example,
{ref}`howto-pipeline-nonlinear-operators` for how the tangent and damping terms
are weighted into the time scheme, and
{ref}`easyfea-examples-hyperelasticity` for the corresponding examples.
```

#### The `groupElem.Get_*` interface

Writing an operator of your own — the integration hands it one `groupElem` at a time —
means reading its integration data through three functions, each taking a
{py:class}`~EasyFEA.FEM.MatrixType`:

```python
from EasyFEA.FEM import MatrixType

# weighted Jacobians: shape (Ne, pg)
wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)

# shape function gradients: shape (Ne, pg, dim, nPe)
dN_e_pg = groupElem.Get_dN_e_pg(matrixType)

# shape functions (reaction term): shape (Ne, pg, nPe, nPe)
N_e_pg  = groupElem.Get_ReactionPart_e_pg(matrixType)
```

The choice of `MatrixType` must match the **integrand form**:

| Integrand                 | `MatrixType`      | Typical use                                 |
| ------------------------- | ----------------- | ------------------------------------------- |
| $\nabla N \cdot \nabla N$ | `MatrixType.rigi` | Stiffness, damping ($\Krm$, $\Crm$)         |
| $N \cdot N$               | `MatrixType.mass` | Mass, reaction, capacity ($\Mrm$, $\Crm_t$) |

**Using the wrong matrix type causes under- or over-integration.** `rigi` and `mass`
select different Gauss quadrature rules: `rigi` uses a rule exact for the polynomial
degree of $\nabla N \nabla N$, while `mass` uses a higher rule suited to $N N$. Applying
`rigi` to an $N N$ form **under-integrates** it (too few Gauss points, quadrature
error); applying `mass` to a $\nabla N \nabla N$ form **over-integrates** it
(unnecessary cost but also potential numerical issues with some element types). Either
mistake silently produces wrong matrices.

These arrays are {py:class}`~EasyFEA.FEM.FeArray` objects: they cover all `Ne` elements
and `pg` Gauss points simultaneously. **No Python loops** are needed. Intermediate
quantities such as `jacobian_e_pg` are computed once and cached on the group element
object via a cache decorator, so repeated calls are free.

#### Example: thermal stiffness and capacity matrices

The following example assembles both the conductivity matrix $K_t$
($\int_\Omega k \, \nabla t \cdot \nabla \delta t \, \dO$ — a $\nabla N \nabla N$ form)
and the heat capacity matrix $C_t$ ($\int_\Omega \rho c \, t \, \delta t \, \dO$ — an
$N N$ form).

Both forms already exist in {py:mod}`~EasyFEA.FEM.Operators.Bilinear`, so neither the
quadrature nor the Gauss-point summation has to be written by hand — this is essentially
all of {py:class}`~EasyFEA.Simulations.Thermal`'s assembly:

```python
from EasyFEA.FEM import Operators
from EasyFEA.Simulations import Term

def Get_terms(self, x, problemType=None):
    model = self.thermalModel

    return [
        # conductivity — ∫ k ∇t·∇δt dΩ  (∇N·∇N form)
        Term("K", Operators.Bilinear.GradUGradV, coef=model.k),
        # capacity — ∫ ρc t·δt dΩ  (N·N form, one dof per node)
        Term("C", Operators.Bilinear.UV, coef=self.rho * model.c, dof_n=1),
    ]
```

No mass term: the thermal problem has no inertia. No load term either — volumetric
sources are applied as Neumann boundary conditions. For structural dynamics you would
add `Term("M", Operators.Bilinear.UV, coef=self.rho, dof_n=self.dim)`.

```{warning}
Neither term passes `dim=`, so each is integrated over the groups of the mesh's own
dimension. Do not pass `dim=self.dim`: that is the *field* dimension (dofs per node,
`1` for a thermal problem), not the mesh dimension, so it would silently assemble over
the `SEG2` boundary edges of a 2D mesh instead of its `QUAD4` cells.
```

Each operator returns the element matrix already integrated, of shape
`(Ne, nPe·dof_n, nPe·dof_n)`, and picks the `MatrixType` its form requires — `rigi` for
{py:func}`~EasyFEA.FEM.Operators.Bilinear.GradUGradV`, `mass` for
{py:func}`~EasyFEA.FEM.Operators.Bilinear.UV` — so the under- and over-integration
mistake described above cannot be made by accident. Pass `matrixType=` explicitly only
to override that choice deliberately.

`coef` is not restricted to a scalar: it broadcasts from `(Ne,)`, `(nPg,)` or
`(Ne, nPg)`, which is how a spatially varying conductivity or a temperature-dependent
capacity is supplied.

```{tip}
Reach for an existing operator first — the catalogue of
{py:mod}`~EasyFEA.FEM.Operators.Bilinear`,
{py:mod}`~EasyFEA.FEM.Operators.Linear` and
{py:mod}`~EasyFEA.FEM.Operators.NonLinear` forms is in {ref}`fem-operators`.
Write the quadrature by hand only for a form the module does not cover, and
prefer adding it there over inlining it in a simulation, so the next
simulation needing it gets it too.
```

```{note}
Declare only the terms the formulation actually has: a parabolic problem
(heat equation) needs a `C` term, a hyperbolic one (structural dynamics) both
`C` and `M`. A slot with no term is simply absent from the system.
```

```{note}
Implementing `Get_terms` requires familiarity with FEM
formulations.  For most new physics, the weak-form approach described above
is simpler and should be preferred.
```
