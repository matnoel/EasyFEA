![EasyFEA_banner](https://github.com/matnoel/EasyFEA/blob/main/docs/_static/EasyFEA_banner.jpg?raw=True)

<!-- Don't do this otherwise PyPi could lose access to the image -->
<!-- ![EasyFEA_banner](docs/_static/EasyFEA_banner.jpg?raw=True) -->

[![PyPI version](https://img.shields.io/pypi/v/easyfea.svg)](https://pypi.org/project/easyfea/)
[![Python Version](https://img.shields.io/pypi/pyversions/easyfea)](https://pypi.org/project/easyfea/)
[![Documentation Status](https://readthedocs.org/projects/easyfea/badge/?version=latest)](https://easyfea.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/matnoel/EasyFEA/blob/main/LICENSE.txt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)
[![PyPI Downloads](https://img.shields.io/pypi/dm/easyfea)](https://pypi.org/project/easyfea/)
[![Tests](https://github.com/matnoel/EasyFEA/actions/workflows/tests.yaml/badge.svg)](https://github.com/matnoel/EasyFEA/actions/workflows/tests.yaml)

## 🧭 Overview

**EasyFEA** is a user-friendly Python library that simplifies finite element analysis. It is flexible and supports different types of simulations without requiring users to handle complex PDE formulations. You will find below the finite element analysis that you can conduct using EasyFEA:

1. **Linear Elastostatic and Elastodynamics simulations** 
    - [examples/Elastic](https://easyfea.readthedocs.io/en/latest/examples/Elastic/index.html)
    - [examples/Dynamic](https://easyfea.readthedocs.io/en/latest/examples/Dynamic/index.html)
    - [examples/Homog](https://easyfea.readthedocs.io/en/latest/examples/Homog/index.html)
    - [examples/Contact](https://easyfea.readthedocs.io/en/latest/examples/Contact/index.html)
2. **Nonlinear Hyperelastic Static simulations**
    - [examples/HyperElastic](https://easyfea.readthedocs.io/en/latest/examples/HyperElastic/index.html)
3. **Euler-Bernoulli beam simulations**
    - [examples/Beam](https://easyfea.readthedocs.io/en/latest/examples/Beam/index.html)
4. **Thermal simulations**
    - [examples/Thermal](https://easyfea.readthedocs.io/en/latest/examples/Thermal/index.html)
5. **Phase-field damage simulations for quasi-static brittle fracture** 
    - [examples/PhaseField](https://easyfea.readthedocs.io/en/latest/examples/PhaseField/index.html)
6. **Weak forms simulations** 
    - [examples/WeakForms](https://easyfea.readthedocs.io/en/latest/examples/WeakForms/index.html)
7. **Digital Image Correlation (DIC) and Parameter identification** 
    - [Material parameters of a phase-field model used to simulate brittle fracture of spruce specimens.](https://gitlab.univ-eiffel.fr/collaboration-msme-fcba/spruce-params)
8. **Stochastic phase-field simulations**
    - [Stochastic phase-field modeling of spruce wood specimens under compression.](https://gitlab.univ-eiffel.fr/collaboration-msme-fcba/spruce-stochastic)

For each simulation, users create a **mesh** and a **model**. Once the simulation has been set up, defining the boundary conditions, solving the problem and visualizing the results are straightforward.

Numerous examples of mesh creation are available in the [Meshes](https://easyfea.readthedocs.io/en/latest/examples/Meshes/index.html) gallery.

The simplest and quickest introduction is shown below and is available in the [Beginner’s Guide](https://easyfea.readthedocs.io/en/latest/begin.html).

```python
from EasyFEA import Display, ElemType, Models, Simulations
from EasyFEA.Geoms import Domain

# ----------------------------------------------
# Mesh
# ----------------------------------------------
L = 120  # mm
h = 13

domain = Domain((0, 0), (L, h), h / 3)
mesh = domain.Mesh_2D([], ElemType.QUAD9, isOrganised=True)

# ----------------------------------------------
# Simulation
# ----------------------------------------------
E = 210000  # MPa
v = 0.3
F = -800  # N

mat = Models.ElasIsot(2, E, v, planeStress=True, thickness=h)

simu = Simulations.ElasticSimu(mesh, mat)

nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

simu.add_dirichlet(nodesX0, [0, 0], ["x", "y"])
simu.add_surfLoad(nodesXL, [F / h / h], ["y"])

simu.Solve()

# ----------------------------------------------
# Results
# ----------------------------------------------
Display.Plot_Mesh(simu, deformFactor=10)
Display.Plot_BoundaryConditions(simu)
Display.Plot_Result(simu, "uy", plotMesh=True)
Display.Plot_Result(simu, "Svm", plotMesh=True, ncolors=11)

Display.plt.show()
```

## ⚖️ License

Copyright (C) 2021-2025 Université Gustave Eiffel.

EasyFEA is distributed under the terms of the [GNU General Public License v3.0 only](https://spdx.org/licenses/GPL-3.0-only.html), see [LICENSE.txt](https://github.com/matnoel/EasyFEA/blob/main/LICENSE.txt) and [CREDITS.md](https://github.com/matnoel/EasyFEA/blob/main/CREDITS.md) for more information.

## 📚 Documentation

Refer to the [documentation](https://easyfea.readthedocs.io/en/latest/index.html) for detailed installation and usage details.

## 💻  Installation

EasyFEA can be easily installed from [PyPI](https://pypi.org/project/EasyFEA/) using pip, compatible with Python versions 3.9 through 3.13:

```
pip install EasyFEA
```

You can also install EasyFEA with the [source code](https://github.com/matnoel/EasyFEA) using the `pip install .` command in the downloaded or cloned EasyFEA `folder`.

### 📦 Dependencies

EasyFEA uses several libraries, such as NumPy and Gmsh - as such, the following projects are required dependencies of EasyFEA:

+ [`numpy`](https://pypi.org/project/numpy/) - Fundamental package for scientific computing with Python.
+ [`gmsh`](https://pypi.org/project/gmsh/) (>= 4.12) - Three-dimensional finite element mesh generator.
+ [`scipy`](https://pypi.org/project/scipy/) - Fundamental package for scientific computing in Python.
+ [`matplotlib`](https://pypi.org/project/matplotlib/) - Plotting package.
+ [`pyvista`](https://pypi.org/project/pyvista/) - Plotting package.
+ [`numba`](https://pypi.org/project/numba/) - Compiling Python code using LLVM.
+ [`pandas`](https://pypi.org/project/pandas/) (3.9 <= Python <= 3.12) - Powerful data structures for data analysis.
+ [`imageio`](https://pypi.org/project/imageio/) and [`imageio[ffmpeg]`](https://pypi.org/project/imageio-ffmpeg/) - Library for reading and writing a wide range of image, video, scientific, and volumetric data formats.
+ [`meshio`](https://github.com/matnoel/meshio/tree/medit_higher_order_elements) - I/O for many mesh formats.

### 🧪 Optional Dependencies

EasyFEA includes a few optional dependencies for reducing resolution time or for performing DIC:

+ [`pypardiso`](https://pypi.org/project/pypardiso/) (Python > 3.8 & Intel oneAPI)  - Library for solving large systems of sparse linear equations.
+ [`petsc`](https://pypi.org/project/petsc/) and [`petsc4py`](https://pypi.org/project/petsc4py/) - Python bindings for PETSc.
+ [`opencv-python`](https://pypi.org/project/opencv-python/) - Computer Vision package.

## 🔤 Naming conventions

**EasyFEA** uses Object-Oriented Programming ([OOP](https://en.wikipedia.org/wiki/Object-oriented_programming)) with the following naming conventions:
+ `PascalCasing` for classes
+ `camelCasing` for properties
+ `Snake_Casing` or `Snake_casing` for functions/methods

In this library, objects can contain both **public** and **private** properties or functions.

**Private** parameters or functions are designated by a double underscore, such as `__privateParam`. In addition, parameters or functions beginning with an underscore, such as `_My_Function` are accessible to advanced users, but should be used with caution.

## ✍️ Citing EasyFEA

If you are using EasyFEA as part of your scientific research, please contribute to the scientific visibility of the project by citing it as follows.

> Noel M., *EasyFEA: a user-friendly Python library that simplifies finite element analysis*, https://hal.science/hal-04571962

Bibtex:

```
@softwareversion{noel:hal-04571962v1,
  TITLE = {{EasyFEA: a user-friendly Python library that simplifies finite element analysis}},
  AUTHOR = {Noel, Matthieu},
  URL = {https://hal.science/hal-04571962},
  NOTE = {},
  INSTITUTION = {{Universit{\'e} Gustave Eiffel}},
  YEAR = {2024},
  MONTH = Apr,
  SWHID = {swh:1:dir:ffb0e56fe2ce8a344ed27df7baf8f5f1b58700b5;origin=https://github.com/matnoel/EasyFEA;visit=swh:1:snp:88527adbdb363d97ebaee858943a02d98fc5c23c;anchor=swh:1:rev:ee2a09258bfd7fd60886ad9334b0893f4989cf35},
  REPOSITORY = {https://github.com/matnoel/EasyFEA},
  LICENSE = {GNU General Public License v3.0},
  KEYWORDS = {Finite element analyses ; Computational Mechanics ; Numerical Simulation ; Phase field modeling of brittle fracture ; Linear elasticity ; Euler-Bernoulli beam ; DIC - Digital Image Correlation ; User friendly ; Object oriented programming ; Mesh Generation},
  HAL_ID = {hal-04571962},
  HAL_VERSION = {v1},
}
```

## 📘 Projects and Publications

### 📝 Scientific Publications

- Noel M. et al.,  *Parameter identification for phase-field modeling of brittle fracture in spruce wood* - Engineering Fracture Mechanics, https://doi.org/10.1016/j.engfracmech.2025.111304

### 🧪 Research Projects

- [Material parameters of a phase-field model used to simulate brittle fracture of spruce specimens.](https://gitlab.univ-eiffel.fr/collaboration-msme-fcba/spruce-params)
- [Stochastic phase-field modeling of spruce wood specimens under compression.](https://gitlab.univ-eiffel.fr/collaboration-msme-fcba/spruce-stochastic)

## 🤝 Contributing

**EasyFEA** is an emerging project with a strong commitment to growth and improvement. Your input and ideas are invaluable to me. I welcome your comments and advice with open arms, encouraging a culture of respect and kindness in our collaborative journey towards improvement.

To learn more about contributing to EasyFEA, please consult the [Contributing Guide](https://github.com/matnoel/EasyFEA/blob/main/CONTRIBUTING.md).