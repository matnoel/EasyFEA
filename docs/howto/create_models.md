(howto-models)=
# Create a model

A **model** encapsulates the material parameters and physics of a simulation ({py:class}`~EasyFEA.Simulations._Simu`) and is accessible in the {py:mod}`EasyFEA.Models` namespace.

---

## Linear elastic models

All elastic models expose a `.C` property: the **stiffness tensor in Kelvin–Mandel notation**, a symmetric positive-definite matrix that maps strains $\boldsymbol{\varepsilon}$ to stresses $\boldsymbol{\sigma}$:

$$
\boldsymbol{\sigma} = \Crm \, \boldsymbol{\varepsilon}
$$

- In 2D: $\Crm$ is $(3 \times 3)$, with components
  $[\sigma_{xx},\, \sigma_{yy},\, \sqrt{2}\,\sigma_{xy}]$.
- In 3D: $\Crm$ is $(6 \times 6)$, with components
  $[\sigma_{xx},\, \sigma_{yy},\, \sigma_{zz},\, \sqrt{2}\,\sigma_{yz},\, \sqrt{2}\,\sigma_{xz},\, \sqrt{2}\,\sigma_{xy}]$.

### {py:class}`~EasyFEA.Models.Elastic.Isotropic`

Defined by Young's modulus `E` and Poisson's ratio `v`:

```python
from EasyFEA import Models

# 2D plane stress (default), steel
mat = Models.Elastic.Isotropic(dim=2, E=210000, v=0.3, planeStress=True, thickness=1.0)

# 3D
mat3d = Models.Elastic.Isotropic(dim=3, E=210000, v=0.3)
```

`planeStress=True` (default) applies the plane-stress assumption in 2D.
Set `planeStress=False` for plane strain.

### {py:class}`~EasyFEA.Models.Elastic.TransverselyIsotropic`

One isotropic plane (T, R) with a distinct longitudinal direction `L`:

```python
from EasyFEA import Models

mat = Models.Elastic.TransverselyIsotropic(
    dim=2,
    El=12000,    # longitudinal Young's modulus
    Et=800,      # transverse Young's modulus
    Gl=500,      # longitudinal shear modulus
    vl=0.3,      # longitudinal Poisson ratio
    vt=0.4,      # transverse Poisson ratio
    axis_l=(1, 0, 0),
    axis_t=(0, 1, 0),
)
```

### {py:class}`~EasyFEA.Models.Elastic.Orthotropic`

Three distinct material axes, each with its own Young's modulus, shear
modulus, and Poisson ratio:

```python
from EasyFEA import Models

mat = Models.Elastic.Orthotropic(
    dim=3,
    E1=12000, E2=800, E3=500,
    G23=200, G13=300, G12=400,
    v23=0.3, v13=0.25, v12=0.4,
    axis_1=(1, 0, 0),
    axis_2=(0, 1, 0),
)
```

### {py:class}`~EasyFEA.Models.Elastic.Anisotropic`

Provide the full stiffness matrix `C` directly in the material basis:

```python
import numpy as np
from EasyFEA import Models

C = np.eye(3) * 210000   # example — replace with your actual matrix
mat = Models.Elastic.Anisotropic(
    dim=2,
    C=C,
    useVoigtNotation=False,  # True if C is expressed in Voigt notation
    axis1=(1, 0, 0),
    axis2=(0, 1, 0),
)
```

---

## Hyperelastic models

### {py:class}`~EasyFEA.Models.HyperElastic.NeoHookean`

```python
from EasyFEA import Models

mat = Models.HyperElastic.NeoHookean(dim=2, K=100000, thickness=1.0)
```

### {py:class}`~EasyFEA.Models.HyperElastic.MooneyRivlin`

```python
from EasyFEA import Models

mat = Models.HyperElastic.MooneyRivlin(dim=2, K1=80000, K2=20000, K=0.0)
```

### {py:class}`~EasyFEA.Models.HyperElastic.SaintVenantKirchhoff`

```python
from EasyFEA import Models

E, nu = 210000.0, 0.3
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
mu    = E / (2 * (1 + nu))

mat = Models.HyperElastic.SaintVenantKirchhoff(dim=3, lmbda=lmbda, mu=mu)
```

### {py:class}`~EasyFEA.Models.HyperElastic.HolzapfelOgden`

For fiber-reinforced soft tissues with two fiber families:

```python
import numpy as np
from EasyFEA import Models

mat = Models.HyperElastic.HolzapfelOgden(
    dim=3,
    C0=0.059, C1=8.0, C2=0.0,
    C3=18.6, C4=16.5, C5=0.0,
    C6=0.0, C7=0.0,
    K=100.0,
    Mu1=0.059, Mu2=0.059,
    T1=np.array([1, 0, 0]),   # fiber direction 1
    T2=np.array([0, 1, 0]),   # fiber direction 2
)
```

---

## Thermal model

A {py:class}`~EasyFEA.Models.Thermal` model is defined by thermal conductivity `k` and, for transient problems, heat
capacity `c`:

```python
from EasyFEA import Models

# static (k only)
mat = Models.Thermal(k=1.0)

# transient (k + c)
mat_transient = Models.Thermal(k=1.0, c=500.0, thickness=1.0)
```

---

## Phase-field model

{py:class}`~EasyFEA.Models.PhaseField` wraps an {py:class}`~EasyFEA.Models.Elastic` model and adds the
fracture parameters:

```python
from EasyFEA import Models

mat_elas = Models.Elastic.Isotropic(dim=2, E=210000, v=0.3)

mat = Models.PhaseField(
    material=mat_elas,
    split=Models.PhaseField.SplitType.Bourdin,   # energy split
    regularization=Models.PhaseField.ReguType.AT2,
    Gc=0.07,   # critical energy release rate [J/m²]
    l0=0.01,   # half crack width [m]
)
```

---

## WeakForms model

{py:class}`~EasyFEA.Models.WeakForms` takes the bilinear and linear form
functions defined with `@BiLinearForm` / `@LinearForm` decorators:

```python
from EasyFEA.FEM import Field, BiLinearForm

field = Field(mesh.groupElem, dof_n=1)

@BiLinearForm
def a(u: Field, v: Field):
    return u.grad.dot(v.grad)

model = Models.WeakForms(field, computeK=a)
```