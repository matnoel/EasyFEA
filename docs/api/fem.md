(fem)=
# FEM

The {py:mod}`EasyFEA.FEM` module provides essential tools for creating and managing finite element meshes, which are crucial for numerical simulations using the [Finite Element Method](https://en.wikipedia.org/wiki/Finite_element_method) (FEM).

In the simulation workflow, `FEM` is the **second step**: geometry objects from {py:mod}`~EasyFEA.Geoms` are passed to {py:class}`~EasyFEA.FEM.Mesher` to produce the {py:class}`~EasyFEA.FEM.Mesh`.

```{seealso}
- {ref}`howto-mesh`
- {ref}`howto-import-mesh`
- {ref}`easyfea-examples-meshes` examples
```

---

## What is a mesh in EasyFEA?

A {py:class}`~EasyFEA.FEM.Mesh` object in EasyFEA represents a collection of elements used to define the geometry and structure for finite element analysis. It contains multiple {py:class}`~EasyFEA.FEM._GroupElem` instances, which are groups of {py:class}`~EasyFEA.FEM.ElemType` that collectively define the spatial discretization of the domain for numerical simulations.

For example, a {py:class}`~EasyFEA.FEM.Elems.HEXA8` mesh includes the following element types:

- {py:class}`~EasyFEA.FEM.Elems.POINT` (0D element)
- {py:class}`~EasyFEA.FEM.Elems.SEG2` (1D element)
- {py:class}`~EasyFEA.FEM.Elems.QUAD4` (2D element)
- {py:class}`~EasyFEA.FEM.Elems.HEXA8` (3D element)

All implemented element types, along with their corresponding shape functions and derivatives, are defined in the {py:mod}`EasyFEA.FEM.Elems` module and were defined in {ref}`examples-meshes-shape-functions`.
The Gauss point quadratures are implemented in the {py:class}`~EasyFEA.FEM.Gauss` class.

---

## Creating or importing a Mesh

To construct a {py:class}`~EasyFEA.FEM.Mesh` using the {py:class}`~EasyFEA.FEM.Mesher`, you must first create {py:class}`~EasyFEA.Geoms._Geom` objects (see {ref}`geoms` for some examples). The {py:class}`~EasyFEA.FEM.Mesher` class serves as an interface to [Gmsh](https://gmsh.info/), a powerful meshing tool, and includes the following primary functions for mesh generation:

- {py:meth}`~EasyFEA.FEM.Mesher.Mesh_2D`: Generates a 2D mesh.
- {py:meth}`~EasyFEA.FEM.Mesher.Mesh_Extrude`: Creates a mesh by extruding a 2D shape.
- {py:meth}`~EasyFEA.FEM.Mesher.Mesh_Revolve`: Generates a mesh by revolving a 2D shape around an axis.
- {py:meth}`~EasyFEA.FEM.Mesher.Mesh_Import_part`: Imports a CAD part (e.g., .stp) to create a mesh.
- {py:meth}`~EasyFEA.FEM.Mesher.Mesh_Import_mesh`: Imports an existing Gmsh mesh. EasyFEA is also linked to meshio and can be used through the following functions:
- {py:meth}`~EasyFEA.Utilities.MeshIO.Medit_to_EasyFEA`: Imports a Medit mesh.
- {py:meth}`~EasyFEA.Utilities.MeshIO.Gmsh_to_EasyFEA`: Imports a Gmsh mesh.
- {py:meth}`~EasyFEA.Utilities.MeshIO.PyVista_to_EasyFEA`: Imports a PyVista mesh (UnstructuredGrid or MultiBlock).
- {py:meth}`~EasyFEA.Utilities.MeshIO.Ensight_to_EasyFEA`: Imports an EnSight mesh.

```{seealso}
- {ref}`howto-geom`
- {ref}`howto-mesh`
- {ref}`howto-import-mesh`
- {ref}`easyfea-examples-meshes` examples
```

---

(fem-operators)=
## Operators

The {py:mod}`EasyFEA.FEM.Operators` module provides the element-level operators that integrate a form over the Gauss points and produce the element matrices/vectors assembled into the global system (see {ref}`howto-pipeline`). Below, $c$ is a scalar/field coefficient, $\Brm$ the strain-displacement operator and $\Nrm$ the shape functions.

### {py:mod}`~EasyFEA.FEM.Operators.Bilinear` — element matrices

- {py:func}`~EasyFEA.FEM.Operators.Bilinear.UV` — $\int_\Omega c \, u \, v \, \dO$ (mass).
- {py:func}`~EasyFEA.FEM.Operators.Bilinear.GradUGradV` — $\int_\Omega c \, \grad u \cdot \grad v \, \dO$ (diffusion / Laplacian).
- {py:func}`~EasyFEA.FEM.Operators.Bilinear.GradU_A_GradV` — $\int_\Omega c \, \grad u \cdot \Abf \cdot \grad v \, \dO$ (anisotropic diffusion).
- {py:func}`~EasyFEA.FEM.Operators.Bilinear.LinearizedElasticity` — $\int_\Omega \Eps(u) : \Cbf : \Eps(v) \, \dO$ (small-strain stiffness).
- {py:func}`~EasyFEA.FEM.Operators.Bilinear.MassAlongNormal` — $\int_\Gamma c \, (u \cdot \nbf)(v \cdot \nbf) \, \dS$ (surface mass projected on the normal; Robin terms).
- {py:func}`~EasyFEA.FEM.Operators.Bilinear.BeamBending` — $\int_e \Brm^\top \Dbf_{\text{bend}} \, \Brm \, dx$ (axial + bending + torsion).
- {py:func}`~EasyFEA.FEM.Operators.Bilinear.BeamShear` — $\int_e \Brm^\top \Dbf_{\text{shear}} \, \Brm \, dx$ (transverse shear, selective reduced integration).
- {py:func}`~EasyFEA.FEM.Operators.Bilinear.BeamStiffness` — $\Krm_e = \text{BeamBending} + \text{BeamShear}$.
- {py:func}`~EasyFEA.FEM.Operators.Bilinear.BeamMass` — $\int_e c \, \Nrm^\top \Mbf \, \Nrm \, dx$ (consistent beam mass).

### {py:mod}`~EasyFEA.FEM.Operators.Linear` — element load vectors

- {py:func}`~EasyFEA.FEM.Operators.Linear.V` — $\int_\Omega \fbf \cdot v \, \dO$ (body / surface load).

### {py:mod}`~EasyFEA.FEM.Operators.NonLinear` — tangent + residual

- {py:func}`~EasyFEA.FEM.Operators.NonLinear.SecondPiolaKirchhoffStressTensor` — residual $\Frm_e = \int \Brm^\top \boldsymbol{\Sigma} \, \dO$ and consistent tangent $\Krm_e = \int \Brm^\top \dpartial{\boldsymbol{\Sigma}}{\Erm} \Brm \, \dO + \int \grad^\top \boldsymbol{\Sigma} \, \grad \, \dO$ (material + geometric), with $\boldsymbol{\Sigma}$ the second Piola-Kirchhoff stress.
- {py:func}`~EasyFEA.FEM.Operators.NonLinear.KelvinVoigtDamping` — damping $\Crm_e = c \, \eta \int \Brm^\top \Brm \, \dO$ and the configuration tangent $\Krm_e^{\text{geo}} = \dpartial{(\Crm \vrm)}{\urm}$ (large-strain Kelvin–Voigt).
- {py:func}`~EasyFEA.FEM.Operators.NonLinear.FollowingPressure` — following-pressure load $\int_\Gamma p \, v \cdot \nbf(u) \, \dS$ with a deformation-dependent normal, plus its (non-symmetric) tangent.
- {py:func}`~EasyFEA.FEM.Operators.NonLinear.PenaltyContact` — penalty contact on a surface group: force $\Frm_e = \varepsilon_n \int_\Gamma \langle -g_n \rangle \, (v \cdot \nbf) \, \dS$ and tangent $\Krm_e = \varepsilon_n \int_{\Gamma_c} (u \cdot \nbf)(v \cdot \nbf) \, \dS$ — the normal-projected penalty on the active contact zone $\Gamma_c$ where the signed normal gap $g_n < 0$ (penetration); cf. {py:func}`~EasyFEA.FEM.Operators.Bilinear.MassAlongNormal`. $\langle\cdot\rangle$ is the Macaulay bracket, $\nbf$ the outward obstacle normal, and $g_n,\nbf$ are supplied at the contact-surface Gauss points (e.g. from {py:meth}`~EasyFEA.FEM._GroupElem._Get_gap_and_normal`).

## FEM API

```{eval-rst}
.. automodule:: EasyFEA.FEM
.. automodule:: EasyFEA.FEM.Elems
.. automodule:: EasyFEA.FEM.Operators.Bilinear
   :exclude-members: FeArray, TensorProd, MatrixType
.. automodule:: EasyFEA.FEM.Operators.Linear
   :exclude-members: FeArray, MatrixType
.. automodule:: EasyFEA.FEM.Operators.NonLinear
   :exclude-members: FeArray, MatrixType, Project_vector_to_matrix
```