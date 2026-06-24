(howto-new-simulation)=
# Create a custom simulation

A **simulation** drives the full assembly–solve–store pipeline for a given
physics.
Every simulation inherits from {py:class}`~EasyFEA.Simulations._Simu`
and is accessible in the {py:mod}`EasyFEA.Simulations` namespace.

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

If your physics is not listed above, consider
[opening an issue](https://github.com/matnoel/EasyFEA/issues) to propose it
before writing custom code — new physics contributions are very welcome.
```

For problems outside the built-in classes, EasyFEA provides three extension
points, from easiest to most flexible:

1. **{py:class}`~EasyFEA.Simulations.WeakForms`** — define any PDE in
   variational form with a few lines of Python. Covers scalar problems
   (Poisson), vector problems (elasticity), and transient or non-linear
   problems. No FEM assembly knowledge required.
2. **Extend an existing simulation** — when a built-in class already covers
   most of your physics and you only need to add extra terms (a boundary
   contribution, a penalty, a coupling), subclass it and override
   {py:meth}`~EasyFEA.Simulations._Simu.Construct_local_matrix_system`: call
   `super().Construct_local_matrix_system(...)` for the base contributions,
   then add your own before returning. See {ref}`howto-new-simulation-extend`.
3. **Subclass {py:class}`~EasyFEA.Simulations._Simu`** — provides full control over the assembly at the element level for problems that are difficult to model in {py:class}`~EasyFEA.Simulations.WeakForms`, or to improve performance. Knowledge of finite element methods is required.

EasyFEA supports multi-physics problems such as phase-field fracture simulations, which couple an elastic sub-problem with a damage sub-problem via a staggered algorithm: each sub-problem is solved in turn with the other held fixed, and the two are iterated to convergence within each load step.
This pattern is already implemented in {py:class}`~EasyFEA.Simulations.PhaseField`.
Monolithic coupling—assembling all physics into a single global system—is not currently implemented, but there is no fundamental limitation preventing it.

```{note}
{py:class}`~EasyFEA.Simulations.WeakForms` is still evolving. Contributions
are welcome to extend its capabilities — in particular, support for contact
mechanics and elastoplastic simulations would benefit greatly from community
involvement. See the
[Contributing Guide](https://github.com/matnoel/EasyFEA/blob/main/CONTRIBUTING.md)
if you would like to help.
```

---

## The weak form approach

A weak form is defined by:

- a **bilinear form** $a(u, v)$ that produces the stiffness matrix $\Krm$,
- a **linear form** $\ell(v)$ that produces the load vector $\Frm$.

EasyFEA assembles $\Krm$ and $\Frm$ automatically from these forms using Gauss quadrature over the mesh elements.

### Available operators

{py:class}`~EasyFEA.FEM.Field` represents the unknown and test fields. The
following operators from `EasyFEA.FEM` act on `Field` objects and return
{py:class}`~EasyFEA.FEM.FeArray` tensors:

| Operator | Description |
|---|---|
| `u.grad` | Gradient $\nabla u$ — shape `(Ne, pg, dof_n, dim)` |
| `Sym_Grad(u)` | Symmetric gradient $\frac{1}{2}(\nabla u + \nabla u^\top)$ |
| `Trace(A)` | Trace of a square `FeArray` |
| `Transpose(A)` | Transpose of a `FeArray` |
| `Det(A)` | Determinant of a square `FeArray` |
| `Inv(A)` | Inverse of a square `FeArray` |
| `TensorProd(a, b)` | Tensor (outer) product $a \otimes b$ |
| `Norm(a)` | Euclidean norm |
| `A.dot(B)` | Contracted product (inner product for vectors, matrix–vector for tensors) |
| `A.ddot(B)` | Double contraction $A : B$ — used for stress–strain products |

All operators act element- and Gauss-point-wise over arrays of shape `(Ne, pg, ...)`, so **no Python loops are needed** over elements or integration points.

All weak-form-based simulations are available in {ref}`easyfea-examples-weak-forms`.

---

(howto-new-simulation-extend)=
## Extend an existing simulation

When a built-in simulation already covers most of your physics, you rarely need to reimplement assembly from scratch. Subclass the existing class and override {py:meth}`~EasyFEA.Simulations._Simu.Construct_local_matrix_system` to **add** contributions on top of the base ones: call `super().Construct_local_matrix_system(...)` to obtain the base `{groupElem: (K_e, C_e, M_e, F_e)}` dict, then add your custom terms before returning it.

The `MonoVentricular` example
([CardiacElastoDynamics/MonoVentricular.py](https://github.com/matnoel/EasyFEA/blob/main/examples/CardiacElastoDynamics/MonoVentricular.py))
does exactly this: it subclasses {py:class}`~EasyFEA.Simulations.HyperElastic`
and augments the hyperelastic tangent/residual with a following pressure on the
endocardium and Robin-type surface penalties on the `top` and `epi` boundaries.

```python
from EasyFEA import MatrixType, Simulations
from EasyFEA.FEM import Operators

class CardiacElastoDynamics(Simulations.HyperElastic):

    def Construct_local_matrix_system(self, problemType):
        # base hyperelastic contributions: {groupElem: (K_e, C_e, M_e, F_e)}
        results = super().Construct_local_matrix_system(problemType)

        displacement = self._Solver_Get_Newton_Raphson_current_solution()

        # add surface terms on the (dim-1) boundary element groups
        for groupElem in self.mesh.Get_list_groupElem(self.dim - 1):
            tangent_e, residual_e = Operators.NonLinear.FollowingPressure(
                groupElem, displacement, self.pressure,
                groupElem.Get_Elements_Tag("endo"), MatrixType.mass,
            )
            results[groupElem] = (tangent_e, None, None, residual_e)

        return results
```

Two things to note: the boundary loop iterates the `self.dim - 1` (surface) element groups, and each new contribution is written into the dict returned by the base class. The {py:mod}`EasyFEA.FEM.Operators` module (`Bilinear`, `Linear`, `NonLinear`) provides ready-made element operators — see {ref}`fem-operators` for the full list — so this path rarely requires hand-writing the integration described in the next section.

---

(howto-new-simulation-subclass)=
## Subclass `_Simu` for a fully custom simulation

If your problem requires custom assembly logic that cannot be expressed as a
weak form, subclass {py:class}`~EasyFEA.Simulations._Simu` directly.
{py:class}`~EasyFEA.Simulations.Thermal` is the simplest existing subclass
and is a good starting point; consult [_thermal.py](https://github.com/matnoel/EasyFEA/blob/main/EasyFEA/Simulations/_thermal.py) for implementation details.

The complete interface to implement (all methods are abstract):

```python
import numpy as np
from EasyFEA.Simulations import _Simu
from EasyFEA.Models import ModelType

class MySimulation(_Simu):

    def __init__(self, mesh, model, folder="", verbosity=False):
        super().__init__(mesh, model, folder, verbosity)

    # --- problem definition ---

    def Get_problemTypes(self) -> list[ModelType]:
        ...

    def Get_unknowns(self, problemType=None) -> list[str]:
        ...

    def Get_dof_n(self, problemType=None) -> int:
        ...

    def Get_x0(self, problemType=None):
        ...

    # --- assembly (see below for details) ---

    def Construct_local_matrix_system(self, problemType):
        ...

    # --- iteration management ---

    def Save_Iter(self, iter={}):
        ...

    def Set_Iter(self, iter=-1, resetAll=False):
        ...

    # --- post-processing ---

    def Results_Available(self) -> list[str]:
        ...

    def Result(self, result: str, nodeValues=True, iter=None):
        ...

    def Results_Iter_Summary(self):
        ...

    def Results_dict_Energy(self) -> dict[str, float]:
        ...

    def Results_displacement_matrix(self):
        ...

    def Results_nodeFields_elementFields(self, details=False):
        ...
```

### Implementing `Construct_local_matrix_system`

`Construct_local_matrix_system` is the only method where you provide
physics-specific data. From those element-level matrices, `_Simu`
automatically:

1. assembles the global sparse system $\Krm$, $\Crm$, $\Mrm$, $\Frm$,
2. applies boundary conditions,
3. selects the appropriate combination of $\Krm$, $\Crm$, $\Mrm$ for the
   active time-integration algorithm,
4. solves the resulting linear system
   $\Krm \urm + \Crm \vrm + \Mrm \arm = \Frm$
   (where $\vrm$ and $\arm$ are the velocity and acceleration computed by
   the time scheme).

**Your main responsibility is to return the correct element-level matrices.**
Everything else is handled by `_Simu` internally.

```{seealso}
- {ref}`howto-pipeline`
```

Since v2.0.0 a mesh can hold **several element groups of the same dimension** (mixed-element meshes), so the method returns a **dict** mapping each contributing {py:class}`~EasyFEA.FEM._GroupElem` to its 4-tuple `(K_e, C_e, M_e, F_e)`:

```python
{groupElem: (K_e, C_e, M_e, F_e), ...}
```

Build the dict by looping over `self.mesh.Get_list_groupElem(self.dim)` and computing the element matrices for each group. Within each tuple the order is strict — swapping `M_e` and `F_e`, for instance, would silently produce a wrong system. Each term is a `np.ndarray` or `None` (`Ne` and `nPe` below are per-group):

| Tuple position | Symbol | Role | Shape |
|---|---|---|---|
| 1st | `K_e` | Stiffness (linear) or **tangent** (non-linear) | `(Ne, nPe·dof_n, nPe·dof_n)` |
| 2nd | `C_e` | Damping matrix (parabolic / hyperbolic) | same, or `None` |
| 3rd | `M_e` | Mass matrix (hyperbolic only) | same as `K_e`, or `None` |
| 4th | `F_e` | Load vector (linear) or **residual** (non-linear) | `(Ne, nPe·dof_n, 1)`, or `None` |

```{warning}
In a **non-linear** problem, `K_e` must contain the **tangent stiffness**
matrix $\Krm_t = \partial \Rrm / \partial \urm$ evaluated at the current
solution state, and `F_e` must contain the **residual** $\Rrm(\urm)$ rather
than the linear load vector. `_Simu` passes the current solution through
`Get_x0` so that the assembly can depend on it.

See the
[`Hyperelastic.Construct_local_matrix_system`](https://github.com/matnoel/EasyFEA/blob/main/EasyFEA/Simulations/_hyperelastic.py#L129-L179)
source for a concrete example of how tangent stiffness and residual are
assembled in a non-linear finite deformation setting,
{ref}`howto-pipeline-hyperelastic-operators` for how those tangent / damping
terms are weighted into the time-scheme assembly, and
{ref}`easyfea-examples-hyperelasticity` for the corresponding worked
examples.
```

#### The `groupElem.Get_*` interface

All integration data is accessed through each group-element object (the items yielded by `mesh.Get_list_groupElem(self.dim)`) using three key functions that accept a {py:class}`~EasyFEA.FEM.MatrixType` argument:

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

| Integrand | `MatrixType` | Typical use |
|---|---|---|
| $\nabla N \cdot \nabla N$ | `MatrixType.rigi` | Stiffness, damping ($\Krm$, $\Crm$) |
| $N \cdot N$ | `MatrixType.mass` | Mass, reaction, capacity ($\Mrm$, $\Crm_t$) |

**Using the wrong matrix type causes under- or over-integration.** `rigi`
and `mass` select different Gauss quadrature rules: `rigi` uses a rule exact
for the polynomial degree of $\nabla N \nabla N$, while `mass` uses a higher
rule suited to $N N$.
Applying `rigi` to an $N N$ form
**under-integrates** it (too few Gauss points, quadrature error); applying
`mass` to a $\nabla N \nabla N$ form **over-integrates** it (unnecessary cost
but also potential numerical issues with some element types). Either mistake
silently produces wrong matrices.

These arrays are {py:class}`~EasyFEA.FEM.FeArray` objects: they cover all
`Ne` elements and `pg` Gauss points simultaneously.
**No Python loops** are needed.
Intermediate quantities such as `jacobian_e_pg` are computed once
and cached on the group element object via a cache decorator, so repeated
calls are free.

#### Example: thermal stiffness and capacity matrices

The following example assembles both the conductivity matrix $K_t$
($\int_\Omega k \, \nabla t \cdot \nabla \delta t \, \dO$ — a
$\nabla N \nabla N$ form) and the heat capacity matrix $C_t$
($\int_\Omega \rho c \, t \, \delta t \, \dO$ — an $N N$ form):

```python
from EasyFEA.FEM import MatrixType

def Construct_local_matrix_system(self, problemType):
    model = self.thermalModel
    out = {}

    for groupElem in self.mesh.Get_list_groupElem(self.dim):

        # --- stiffness: ∇N·∇N form → MatrixType.rigi ---
        matrixType = MatrixType.rigi
        wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)  # (Ne, pg)
        dN_e_pg = groupElem.Get_dN_e_pg(matrixType)                # (Ne, pg, dim, nPe)

        # (Ne, pg, nPe, nPe) -> sum over pg -> (Ne, nPe, nPe)
        Kt_e = (model.k * wJ_e_pg * dN_e_pg.T @ dN_e_pg).sum(axis=1)

        # --- capacity: N·N form → MatrixType.mass ---
        matrixType = MatrixType.mass
        wJ_e_pg      = groupElem.Get_weightedJacobian_e_pg(matrixType)  # (Ne, pg)
        reactionPart = groupElem.Get_ReactionPart_e_pg(matrixType)      # (Ne, pg, nPe, nPe)

        # (Ne, pg, nPe, nPe) -> sum over pg -> (Ne, nPe, nPe)
        Ct_e = (self.rho * model.c * reactionPart).sum(axis=1)

        # order: (K_e, C_e, M_e, F_e)
        # M_e is None: no inertia term in the thermal problem.
        # F_e is None: volumetric sources are handled as Neumann BCs, not here.
        # For structural dynamics, M_e would also be assembled (MatrixType.mass)
        # and returned in the 3rd position.
        out[groupElem] = (Kt_e, Ct_e, None, None)

    return out
```

```{tip}
This hand-written assembly is shown to expose the underlying mechanics. In practice the built-in {py:class}`~EasyFEA.Simulations.Thermal` builds the same matrices with the {py:mod}`EasyFEA.FEM.Operators` module — `Operators.Bilinear.GradUGradV(groupElem, coef=model.k)` and `Operators.Bilinear.UV(groupElem, coef=self.rho * model.c, dof_n=1)` — which is the recommended way to write new operators. The complete catalogue of {py:mod}`~EasyFEA.FEM.Operators.Bilinear`, {py:mod}`~EasyFEA.FEM.Operators.Linear`, and {py:mod}`~EasyFEA.FEM.Operators.NonLinear` operators is in {ref}`fem-operators`.
```

The `.sum(axis=1)` call sums the contribution of each Gauss point along
`axis=1` (the `pg` axis), producing the assembled element matrix of shape
`(Ne, nPe, nPe)`.

```{note}
`None` means the corresponding term is absent from the system — not that it
is always zero. For a parabolic problem (e.g. heat equation) `C_e` is
non-`None`; for a hyperbolic problem (e.g. structural dynamics) both `C_e`
and `M_e` must be assembled and returned. Only terms that are genuinely
absent from the formulation should be returned as `None`.
```

```{note}
Implementing `Construct_local_matrix_system` requires familiarity with FEM
formulations.  For most new physics, the weak-form approach described above
is simpler and should be preferred.
```