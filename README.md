# EasyFEA - Easy Finite Element Analysis

## Overview

**EasyFEA** is a user-friendly Python library that simplifies finite element analysis. It is flexible and supports different types of simulation without requiring users to handle complex PDE formulations. The library currently supports four specific simulation types:

1. **ElasticSimu** (static and dynamic): Examples in `/examples/Elastic`, `/examples/Dynamic` and `/examples/Contact`.
2. **BeamSimu** (static Euler-Bernoulli): Examples in `/examples/Beam`.
3. **ThermalSimu** (stationary and transient): Examples in `/examples/Thermal`.
4. **PhaseFieldSimu:** (quasi-static phase field) Examples in `/examples/PhaseField`.

All examples are available [here](https://github.com/matnoel/EasyFEA/tree/master/examples).

For each simulation, users create a **mesh** and a **model**. Once the simulation has been set up, defining the boundary conditions, solving the problem and visualizing the results is straightforward.

Numerous examples of mesh creation are available in the `examples/Meshes` folder.

The simplest and quickest introduction is shown below and is available in `examples/HelloWorld.py`.

```python
from EasyFEA import (Display, Mesher, ElemType,
                     Materials, Simulations)
from EasyFEA.Geoms import Point, Domain

# ----------------------------------------------
# Mesh
# ----------------------------------------------
L = 120 # mm
h = 13

domain = Domain(Point(), Point(L,h), h/5*2)
mesh = Mesher().Mesh_2D(domain, [], ElemType.QUAD4, isOrganised=True)

# ----------------------------------------------
# Simulation
# ----------------------------------------------
E = 210000 # MPa
v = .3
F = -800 # N

mat = Materials.Elas_Isot(2, E, v, planeStress=True, thickness=h)

simu = Simulations.ElasticSimu(mesh, mat)

nodesX0 = mesh.Nodes_Conditions(lambda x,y,z: x==0)
nodesXL = mesh.Nodes_Conditions(lambda x,y,z: x==L)

simu.add_dirichlet(nodesX0, [0]*2, ["x","y"])
simu.add_surfLoad(nodesXL, [F/h/h], ["y"])

simu.Solve()

# ----------------------------------------------
# Results
# ----------------------------------------------
Display.Plot_Mesh(mesh)
Display.Plot_BoundaryConditions(simu)
Display.Plot_Result(simu, 'uy', plotMesh=True)
Display.Plot_Result(simu, 'Svm', plotMesh=True, ncolors=11)

Display.plt.show()
```

## Installation

EasyFEA can be easily installed from [PyPI](https://pypi.org/project/EasyFEA/) using pip, compatible with Python versions 3.9 through 3.12:

```
pip install EasyFEA
```

### Dependencies

EasyFEA uses several libraries such as NumPy and Gmsh - as such, the following projects are required dependencies of EasyFEA:

+ [`numpy`](https://pypi.org/project/numpy/) - Fundamental package for scientific computing with Python.
+ [`gmsh`](https://pypi.org/project/gmsh/) (>= 4.12) - Three-dimensional finite element mesh generator.
+ [`scipy`](https://pypi.org/project/scipy/) - Fundamental package for scientific computing in Python.
+ [`matplotlib`](https://pypi.org/project/matplotlib/) - Plotting package.
+ [`pyvista`](https://pypi.org/project/pyvista/) - Plotting package.
+ [`numba`](https://pypi.org/project/numba/) - Compiling Python code using LLVM.
+ [`pandas`](https://pypi.org/project/pandas/) (3.9 <= Python <= 3.12) - Powerful data structures for data analysis.
+ [`imageio`](https://pypi.org/project/imageio/) and [`imageio[ffmpeg]`](https://pypi.org/project/imageio-ffmpeg/) - Library for reading and writing a wide range of image, video, scientific, and volumetric data formats.

For detailed information on installing [`numba`]((https://pypi.org/project/numba/)), refer to the [Numba Installation Guide](https://numba.readthedocs.io/en/stable/user/installing.html#numba-support-info).

### Optional Dependencies

EasyFEA includes a few optional dependencies for reducing resolution time or for performing DIC:

+ [`pypardiso`](https://pypi.org/project/pypardiso/) (Python > 3.8 & Intel oneAPI)  - Library for solving large systems of sparse linear equations.
+ [`petsc`](https://pypi.org/project/petsc/) and [`petsc4py`](https://pypi.org/project/petsc4py/) - Python bindings for PETSc.
+ [`opencv-python`](https://pypi.org/project/opencv-python/) - Computer Vision package.

## Naming conventions

**EasyFEA** uses Object-Oriented Programming ([OOP](https://en.wikipedia.org/wiki/Object-oriented_programming)) with the following naming conventions:
+ `PascalCasing` for classes
+ `camelCasing` for properties
+ `Snake_Casing` or `Snake_casing` for functions/methods

In this library, objects can contain both **public** and **private** properties or functions.

**Private** parameters or functions are designated by a double underscore, such as `__privateParam`. In addition, parameters or functions beginning with an underscore, such as `_My_Function` are accessible to advanced users, but should be used with caution.

## Contributing

**EasyFEA** is an emerging project with a strong commitment to growth and improvement. Your input and ideas are invaluable to me. I welcome your comments and advice with open arms, encouraging a culture of respect and kindness in our collaborative journey towards improvement.

To learn more about contributing to EasyFEA, please consult the [Contributing Guide](https://github.com/matnoel/EasyFEA/blob/main/CONTRIBUTING.md).

## Citing EasyFEA

If you are using EasyFEA as part of your scientific research, please contribute to the scientific visibility of the project by citing it as follows.

> Matthieu Noel. EasyFEA: a user-friendly Python library that simplifies finite element analysis. 2024, https://hal.science/hal-04571962

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
  LICENSE = {GNU General Public License v3.0 or later},
  KEYWORDS = {Finite element analyses ; Computational Mechanics ; Numerical Simulation ; Phase field modeling of brittle fracture ; Linear elasticity ; Euler-Bernoulli beam ; DIC - Digital Image Correlation ; User friendly ; Object oriented programming ; Mesh Generation},
  HAL_ID = {hal-04571962},
  HAL_VERSION = {v1},
}
```

## License

Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.

EasyFEA is distributed under the terms of the [GNU General Public License v3.0 or later](https://spdx.org/licenses/GPL-3.0-or-later.html), see [LICENSE.txt](https://github.com/matnoel/EasyFEA/blob/main/LICENSE.txt) and [CREDITS.md](https://github.com/matnoel/EasyFEA/blob/main/CREDITS.md) for more information.