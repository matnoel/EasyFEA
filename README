# PythonEF - Finite Element Simulation Library

## Overview

PythonEF is a free Python library designed to simplify finite element simulations, providing a versatile platform for various simulation types. The library currently supports four simulation types:

1. Elastic (static and dynamic) - Examples in `/codes/Elasticity` and `/codes/Dynamic`.
2. Beam structure (static Euler-Bernoulli) - Examples in `/codes/Beam`.
3. Thermal (stationary and transient) - Examples in `/codes/Thermal`.
4. Damage (quasi-static phase field) - Examples in `/codes/PhaseField`.

For each simulation, users are required to build a mesh and a model. Once the simulation is set up, you can easily define boundary conditions, solve the problem, and visualize the results.

## Installation

### Step 1 - Install Python Libraries:

Ensure that you have Python version 3.9 to 3.12 installed. The following libraries are required:

- `numpy`
- `scipy`
- `matplotlib`
- `gmsh` >= 4.12.0
- `pandas` (3.9 <= Python <= 3.12)

Optional Libraries (Recommended for improved resolution time):

- `pypardiso` (Python > 3.8 & Intel oneAPI)
- `petsc` and `petsc4py`
- `numba` (3.5.x <= Python < 3.12)
- `mumps` and `mpi4py`
- `scikit-umfpack` (Python >= 3.9)
- `scikits-sparse` (3.6 <= Python <= 3.12)

For more information on installing `numba`, refer to [Numba Installation Guide](https://numba.readthedocs.io/en/stable/user/installing.html#numba-support-info).

### Step 2 - Add the PythonEF Modules to Python Path:

#### On macOS:

```bash
export PYTHONPATH="$PYTHONPATH:folder/PythonEF/modules"  # Add this to .zprofile
```

#### On Linux:

```bash
export PYTHONPATH=$PYTHONPATH:folder/PythonEF/modules  # Add this to .bash_aliases
```

#### On Windows:

- Modify system variables -> environment variables -> user variables
- Add `folder/PythonEF/modules` to PYTHONPATH.

Note: Replace "folder" with the directory where PythonEF is installed.

### Optional: Build Gmsh Yourself

If you need to build Gmsh yourself, follow these steps:

1. Install OpenCascade: `sudo port install opencascade`.
2. Install FLTK: `sudo port install fltk`.
3. Clone the Gmsh project from [https://gitlab.onelab.info/gmsh/gmsh](https://gitlab.onelab.info/gmsh/gmsh).
4. Build Gmsh:

```bash
cd gmsh
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/Applications/gmsh-install -DENABLE_BUILD_DYNAMIC=1 ..
make
make install
```

5. Add the path to `gmsh.py`:

```bash
export PYTHONPATH="$PYTHONPATH:/Applications/gmsh-install/lib"
```

This completes the installation and setup process for PythonEF.