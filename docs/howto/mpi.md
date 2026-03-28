(howto-mpi)=
# Run simulations in parallel with MPI

EasyFEA supports distributed-memory parallelism through
[MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) via
[`mpi4py`](https://mpi4py.readthedocs.io) and
[`petsc4py`](https://petsc4py.readthedocs.io).
When running with more than one rank, EasyFEA partitions the mesh across ranks
at the element level, assembles the distributed linear system, and solves it
with [PETSc](https://petsc.org).

```{note}
`petsc4py` is **required** for parallel execution. EasyFEA raises an assertion
error at simulation construction time if it is not available.
```

---

## Install petsc4py

The simplest method uses [conda-forge](https://conda-forge.org), which
distributes pre-built binaries for Linux, macOS, and Windows:

```bash
conda install -c conda-forge petsc petsc4py mpi4py
```

This installs a complete, self-consistent stack — an MPI implementation
(OpenMPI or MPICH depending on the platform), PETSc, and Python bindings for
both — with no compiler or system library required.

Verify the installation:

```bash
python -c "from petsc4py import PETSc; print(PETSc.Sys.getVersion())"
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
```

For HPC clusters where MPI is provided by the system (e.g. via `module load`),
PETSc and `petsc4py` must be compiled against that MPI to ensure binary
compatibility. Refer to the
[PETSc installation guide](https://petsc.org/release/install/) and the
[petsc4py documentation](https://petsc4py.readthedocs.io/en/stable/install.html)
for instructions.

---

## Run a script in parallel

Any EasyFEA script runs in parallel without modification. Use `mpirun` (or
`mpiexec`) with the desired number of processes:

```bash
mpirun -n 4 python my_simulation.py
```

```{tip}
Start with a small number of ranks (2–4) to verify correctness before scaling
up. Partitioning overhead dominates on coarse meshes.
```

```{note}
Each rank prints timing and progress output independently. To avoid cluttered
terminal output, set `verbosity=False` at simulation construction time (the
default) and use `simu.Results_Set_Iteration_Summary(...)` to control progress
reporting.
```

---

## How parallelism works in EasyFEA

EasyFEA uses **element-level domain decomposition**: each rank owns a disjoint
subset of elements, plus a layer of ghost elements at partition boundaries
required for consistent matrix assembly. The node coordinate array is **not**
distributed — all ranks hold the full node array.

The parallel execution proceeds as follows for each solve:

1. Each rank assembles only its owned (and ghost) elements into a local
   stiffness matrix and load vector.
2. PETSc solves the distributed system $\Arm \, \xrm = \brm$ using a Krylov
   method (default: CG with GAMG preconditioner).
3. An `Allreduce` over the disjoint owned DOF sets of each rank reconstructs
   the full solution vector on **every** rank. Consequently, all ranks hold the
   same, complete DOF result after each solve.

The partition data for each element group can be inspected via
{py:meth}`~EasyFEA.FEM._GroupElem._Get_partitioned_data`.

### Iteration saving

{py:meth}`~EasyFEA.Simulations._Simu.Save_Iter` stores only the primary
unknowns per iteration (e.g. displacement for elastic simulations, displacement
and damage for phase-field). Derived quantities such as stress and strain are
not stored — they are recomputed on demand by `Result()`. Because all ranks
hold the same DOF vector after the solve, only rank 0 writes the result file.

Iteration results are kept in **memory** by default. To write them to **disk**,
provide a folder at construction time or set it afterwards:

```python
# at construction time
simu = Simulations.Elastic(mesh, mat, folder="path/to/results")

# or after construction
simu.folder = "path/to/results"
```

### Phase-field convergence

For phase-field simulations, energy-based convergence criteria
(`convOption=1` or `convOption=2`) reduce partial energy contributions across
all ranks via `Allreduce` before evaluating the stopping criterion. The
convergence flag is therefore consistent across ranks and produces results
identical to a serial run.

---

## Tune the linear solver

The default solver (`cg` + `gamg`) is a good general-purpose choice for both
serial and parallel execution. For large meshes or specific problem types, it
can be tuned via the advanced method
{py:meth}`~EasyFEA.Simulations._Simu._Solver_Set_PETSc4Py_Options`.

```{note}
{py:meth}`~EasyFEA.Simulations._Simu._Solver_Set_PETSc4Py_Options` is an
advanced API (single-underscore prefix). Use it when the default solver is too
slow or fails to converge.
```

**Symmetric positive-definite systems (most FEM problems):**

```python
# Direct solve — best for small to medium meshes; MPI-compatible via MUMPS
simu._Solver_Set_PETSc4Py_Options("preonly", "lu", "mumps")

# Iterative — scales to large meshes
simu._Solver_Set_PETSc4Py_Options("cg", "hypre")  # BoomerAMG, best scalability
simu._Solver_Set_PETSc4Py_Options("cg", "gamg")   # PETSc built-in AMG (default)
```

**Non-symmetric systems:**

```python
simu._Solver_Set_PETSc4Py_Options("gmres", "bjacobi")  # block Jacobi, MPI-compatible
simu._Solver_Set_PETSc4Py_Options("bcgs", "bjacobi")   # BiCGSTAB, cheaper per iteration
```

The optional `problemType` argument restricts the configuration to a specific
sub-problem (e.g. `elastic` or `damage` in a phase-field simulation):

```python
from EasyFEA.Models import ModelType

simu._Solver_Set_PETSc4Py_Options("preonly", "lu", "mumps", problemType=ModelType.elastic)
simu._Solver_Set_PETSc4Py_Options("gmres",   "bjacobi",     problemType=ModelType.damage)
```

---

## Post-process results

### Plotting and in-memory post-processing

Each rank holds only its local mesh partition. To post-process or visualize
results on the complete global mesh, call
{py:meth}`~EasyFEA.Simulations._Simu._Gather` after the solve loop:

```python
simu._Gather()  # assembles the full mesh on rank 0; no-op on other ranks

if MPI_RANK == 0:
    Display.Plot_Result(simu, "uy")
```

{py:meth}`~EasyFEA.Simulations._Simu._Gather` is called automatically by
{py:meth}`~EasyFEA.Simulations._Simu.Save`, which writes one pickle file per
rank (`simulation_rank{N}.pickle`) since each rank holds a different mesh
partition. `Load_Simu` reloads the appropriate file per rank transparently.
Explicit calls to `_Gather` are only necessary for in-memory post-processing
within the script.

### Export to ParaView

{py:func}`~EasyFEA.Utilities.Paraview.Save_simu` supports fully parallel
export. Each rank writes its own `.vtu` piece; rank 0 additionally writes the
`.pvtu` parallel descriptor and the `.pvd` timeline. ParaView reads the `.pvd`
and assembles all pieces automatically.

```python
from EasyFEA import Paraview

# Call directly after the solve loop — do NOT call _Gather() first.
Paraview.Save_simu(simu, folder, N=200)
```

```{warning}
Do **not** call {py:meth}`~EasyFEA.Simulations._Simu._Gather` before
`Paraview.Save_simu` in MPI mode. After `_Gather`, rank 0 holds the full
global mesh while other ranks still hold their partitions. `Save_simu` would
then write the full mesh once (rank 0) and each partition again (other ranks),
producing duplicate elements and corrupted fields in ParaView. `Save_simu`
raises an exception if `simu.isGathered` is `True` to prevent this.
```

---

## Known limitations

- **Serial-only solvers.** The `scipy` and `pypardiso` solvers are not
  available in MPI mode. `petsc4py` is the only supported solver when
  `MPI_SIZE > 1`.

- **Beam simulations.** {py:class}`~EasyFEA.Simulations.Beam` simulations are
  not yet supported in MPI mode. The {py:class}`~EasyFEA.Models.Beam.Isotropic`
  material requires a 2D cross-section mesh to compute section properties
  (area, second moments of area $I_y$, $I_z$) by integrating over all section
  elements. Partitioning the section across ranks would yield incorrect
  properties.

- **DIC analyses.** {py:class}`~EasyFEA.Simulations.DIC` analyses are not yet
  supported in MPI mode.

- **Topology optimisation.** The mesh-independence sensitivity filter
  (Sigmund, 1998) builds a full $N_e \times N_e$ element-distance matrix from
  all element centroids at once. With domain decomposition, each rank holds
  only a subset of centroids, making the global filter construction impossible
  without an explicit gather step.

- **Interactive visualization.** PyVista interactive windows and `plt.show()`
  calls must be guarded by `if MPI_RANK == 0:` after calling
  {py:meth}`~EasyFEA.Simulations._Simu._Gather`, to avoid spawning multiple
  figures or deadlocking non-root ranks.
