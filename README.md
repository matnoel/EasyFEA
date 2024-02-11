# EasyFEA - Easy Finite Element Analysis

## Overview

EasyFEA is a user-friendly Python library that simplifies finite element analysis. It is flexible and supports different types of simulations without requiring users to handle complex PDE formulations. The library currently supports four specific types of simulation:

1. **Elastic (static and dynamic):** Examples in `/codes/Elasticity` and `/codes/Dynamic`.
2. **Beam structure (static Euler-Bernoulli):** Examples in `/codes/Beam`.
3. **Thermal (stationary and transient):** Examples in `/codes/Thermal`.
4. **Damage (quasi-static phase field):** Examples in `/codes/PhaseField`.

For each simulation, users create a mesh and a model. Once the simulation is set up, defining boundary conditions, solving the problem, and visualizing the results is straightforward.

## Installation

### Step 1 - Install Python Libraries:

Ensure that your Python version is between 3.9 and 3.11.

The following libraries are required:

- `numpy`
- `scipy`
- `matplotlib`
- `gmsh` >= 4.12.0
- `numba` (3.5.x <= Python < 3.12)
- `pandas` (3.9 <= Python <= 3.12)

For detailed information on installing `numba`, refer to the [Numba Installation Guide](https://numba.readthedocs.io/en/stable/user/installing.html#numba-support-info).

Optional Libraries (Recommended for improved resolution time):

- `pypardiso` (Python > 3.8 & Intel oneAPI)
- `petsc` and `petsc4py`
- `mumps` and `mpi4py`
- `scikit-umfpack` (Python >= 3.9)
- `scikits-sparse` (3.6 <= Python <= 3.12)

### Step 2 - Add the EasyFEA Modules to Python Path:

#### On macOS:

```bash
export PYTHONPATH="$PYTHONPATH:folder/EasyFEA/modules"  # Add this to .zprofile
```

#### On Linux:

```bash
export PYTHONPATH=$PYTHONPATH:folder/EasyFEA/modules  # Add this to .bash_aliases
```

#### On Windows:

- Modify system variables -> environment variables -> user variables
- Add `folder/EasyFEA/modules` to PYTHONPATH.

Note: Replace "folder" with the directory where EasyFEA is installed.

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