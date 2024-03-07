# EasyFEA - Easy Finite Element Analysis

## Overview

EasyFEA is a user-friendly Python library that simplifies finite element analysis. It is flexible and supports different types of simulation without requiring users to handle complex PDE formulations. The library currently supports four specific simulation types:

1. **Displacement (static and dynamic):** Examples in `/examples/Elasticity`, `/examples/Dynamic` and `/examples/Contact`.
2. **Beam (static Euler-Bernoulli):** Examples in `/examples/Beam`.
3. **Thermal (stationary and transient):** Examples in `/examples/Thermal`.
4. **PhaseField (quasi-static phase field):** Examples in `/examples/PhaseField`.

For each simulation, users create a mesh and a model. Once the simulation has been set up, defining the boundary conditions, solving the problem and visualizing the results is straightforward.

Numerous examples of mesh creation are available in the `examples/Meshes` folder.

The simplest and quickest introduction is shown below and is available in `examples/HelloWorld.py`.

```python
import Display
from Geoms import Point, Domain
from Gmsh_Interface import Mesher, ElemType
import Materials
import Simulations

# ----------------------------------------------
# Mesh
# ----------------------------------------------
L = 120 # mm
h = 13
F = -800 # N

domain = Domain(Point(), Point(L,h), h/5)
mesh = Mesher().Mesh_2D(domain, [], ElemType.QUAD4, isOrganised=True)

# ----------------------------------------------
# Simulation
# ----------------------------------------------
E = 210000 # MPa
v = .3
mat = Materials.Elas_Isot(2, E, v, planeStress=True, thickness=h)

simu = Simulations.Displacement(mesh, mat)

nodesX0 = mesh.Nodes_Conditions(lambda x,y,z: x==0)
nodesXL = mesh.Nodes_Conditions(lambda x,y,z: x==L)

simu.add_dirichlet(nodesX0, [0]*2, ["x","y"])
simu.add_surfLoad(nodesXL, [F/h/h], ["y"])

simu.Solve()

# ----------------------------------------------
# Results
# ----------------------------------------------
Display.Plot_Mesh(mesh)
Display.Plot_Result(simu, 'uy', plotMesh=True)
Display.Plot_Result(simu, 'Svm', plotMesh=True)

Display.plt.show()
```

## Installation

### Step 1 - Install Python Libraries:

Make sure your Python version is between 3.9 and 3.11.

The following libraries are required:

- `numpy`
- `scipy`
- `matplotlib`
- `pyvista` (`imageio`, `imageio[ffmpeg]` for gif or mp4 videos)
- `gmsh` >= 4.12.0
- `numba` (3.5.x <= Python < 3.12)
- `pandas` (3.9 <= Python <= 3.12)

For detailed information on installing `numba`, refer to the [Numba Installation Guide](https://numba.readthedocs.io/en/stable/user/installing.html#numba-support-info).

Optional libraries (recommended to speed up resolution time):

- `pypardiso` (Python > 3.8 & Intel oneAPI)
- `petsc` and `petsc4py`

### Step 2 - Add the EasyFEA Modules to Python Path:

#### On macOS:

```bash
export PYTHONPATH="$PYTHONPATH:folder/EasyFEA/EasyFEA"  # Add this to .zprofile
```

#### On Linux:

```bash
export PYTHONPATH=$PYTHONPATH:folder/EasyFEA/EasyFEA  # Add this to .bash_aliases
```

#### On Windows:

- Modify system variables -> environment variables -> user variables
- Add `folder/EasyFEA/EasyFEA` to PYTHONPATH.

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