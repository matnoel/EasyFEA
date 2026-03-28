(howto-postprocess)=
# Post-process simulation results

**Post-processing** tools visualize and export simulation results.
They are accessible in the {py:mod}`EasyFEA.Utilities` namespace: {py:mod}`~EasyFEA.Utilities.Display` for static matplotlib figures, {py:mod}`~EasyFEA.Utilities.PyVista` for interactive 3D views, and {py:mod}`~EasyFEA.Utilities.Paraview` / {py:mod}`~EasyFEA.Utilities.Vizir` / {py:mod}`~EasyFEA.Utilities.GLTF` / {py:mod}`~EasyFEA.Utilities.USD` for external export.

```{eval-rst}
.. autosummary::
    ~EasyFEA.Utilities.Display
    ~EasyFEA.Utilities.PyVista
    ~EasyFEA.Utilities.Paraview
    ~EasyFEA.Utilities.Vizir
    ~EasyFEA.Utilities.USD
    ~EasyFEA.Utilities.GLTF
```

---

## Query available results

Each simulation exposes the list of computable result fields via
{py:meth}`~EasyFEA.Simulations._Simu.Results_Available`:

```python
print(simu.Results_Available())
# e.g. ['ux', 'uy', 'uz', 'displacement_norm', 'Exx', 'Eyy', 'Sxx', 'Syy', 'Svm', ...]
```

A scalar or vector field can then be retrieved as a NumPy array via
{py:meth}`~EasyFEA.Simulations._Simu.Result`:

```python
uy  = simu.Result("uy")                        # nodal values, shape (Nn,)
Svm = simu.Result("Svm", nodeValues=False)     # element values, shape (Ne,)
```

---

## Plot with matplotlib (Display)

{py:mod}`~EasyFEA.Utilities.Display` uses matplotlib and is the primary tool
for 2D and 3D result visualization.

### Plot a scalar field

{py:func}`~EasyFEA.Utilities.Display.Plot_Result` plots any result field on the
mesh:

```python
from EasyFEA import Display

Display.Plot_Result(simu, "uy")
Display.Plot_Result(simu, "Svm", plotMesh=True, ncolors=11)
```

Key options:

| Parameter | Description |
|---|---|
| `deformFactor` | Scale factor to display deformed geometry (`0` = undeformed) |
| `nodeValues` | `True` for nodal interpolation, `False` for element-constant |
| `plotMesh` | Overlay the mesh edges |
| `ncolors` | Number of discrete color levels in the colorbar |
| `clim` | Fix colorbar range, e.g. `clim=(0, 1)` |
| `coef` | Multiply the result by a constant (e.g. unit conversion) |
| `folder` / `filename` | Save the figure to disk |

### Plot the mesh

{py:func}`~EasyFEA.Utilities.Display.Plot_Mesh` plots the mesh:

```python
Display.Plot_Mesh(simu)                   # current state
Display.Plot_Mesh(simu, deformFactor=10)  # amplified deformation
Display.Plot_Mesh(mesh)                   # mesh object directly
```

### Plot boundary conditions

{py:func}`~EasyFEA.Utilities.Display.Plot_BoundaryConditions` visualizes the
applied loads and constraints:

```python
Display.Plot_BoundaryConditions(simu)
```

### Plot tags

{py:func}`~EasyFEA.Utilities.Display.Plot_Tags` shows the physical groups and
tags defined on the mesh:

```python
Display.Plot_Tags(mesh)
```

### Plot energy and iteration history

```python
Display.Plot_Energy(simu, folder=folder_save)
Display.Plot_Iter_Summary(simu, folder=folder_save)
```

### Save a figure

{py:func}`~EasyFEA.Utilities.Display.Save_fig` saves the current matplotlib
figure to disk:

```python
Display.Save_fig(folder_save, "my_figure")
```

### Create an animation

{py:func}`~EasyFEA.Utilities.Display.Movie_Simu` generates an animation
directly from a named result field:

```python
Display.Movie_Simu(simu, "uy", folder=folder_save, filename="animation.gif")
```

For custom frame content, use
{py:func}`~EasyFEA.Utilities.Display.Movie_func` with a user-defined function.
The function receives the matplotlib figure and the frame index `i`:

```python
import numpy as np

iterations = np.arange(0, simu.Niter, max(1, simu.Niter // 20))
fig = Display.Init_Axes()

def Func(fig, i):
    fig.clear()
    ax = fig.add_subplot(111)
    simu.Set_Iter(iterations[i])
    Display.Plot_Result(simu, "uy", ax=ax)

Display.Movie_func(Func, fig, iterations.size, folder_save, "animation.gif")
```

---

## Interactive 3D visualization (PyVista)

{py:mod}`~EasyFEA.Utilities.PyVista` provides interactive 3D rendering powered
by [PyVista](https://pyvista.org).

{py:func}`~EasyFEA.Utilities.PyVista.Plot` renders a result field in an
interactive window:

```python
from EasyFEA import PyVista

PyVista.Plot(simu, "uy")
PyVista.Plot(simu, "Svm", plotMesh=True, clim=(0, 500))
PyVista.Plot_Mesh(simu)
PyVista.Plot_BoundaryConditions(simu)
PyVista.Plot_Tags(simu)
```

### Create an animation

{py:func}`~EasyFEA.Utilities.PyVista.Movie_simu` generates an animation
directly from a named result field:

```python
PyVista.Movie_simu(simu, "uy", folder=folder_save, filename="animation.gif")
```

For custom frame content, use
{py:func}`~EasyFEA.Utilities.PyVista.Movie_func` with a user-defined function.
The function receives the PyVista plotter and the frame index `i`:

```python
import numpy as np

iterations = np.arange(0, simu.Niter, max(1, simu.Niter // 20))

def Func(plotter, i):
    simu.Set_Iter(iterations[i])
    PyVista.Plot(simu, "damage", plotter=plotter, clim=(0, 1))

PyVista.Movie_func(Func, iterations.size, folder_save, "damage.gif")
```

---

## Export to ParaView

{py:func}`~EasyFEA.Utilities.Paraview.Save_simu` generates a `.pvd` timeline
and `.vtu` files that ParaView reads directly:

```python
from EasyFEA import Paraview

Paraview.Save_simu(simu, folder_save, N=200)
```

`N` controls the maximum number of iterations exported — EasyFEA selects up
to `N` equally-spaced snapshots from the full iteration history. Open the
resulting `Paraview/simulation.pvd` file in ParaView to browse the timeline.

See {ref}`howto-mpi` for parallel ParaView export across MPI ranks.

---

## Export to Vizir

{py:func}`~EasyFEA.Utilities.Vizir.Save_simu` exports results to the
[Vizir](https://pyamg.saclay.inria.fr/vizir4.html) format, a high-order FEM
visualization tool developed by INRIA:

```python
from EasyFEA import Vizir

command = Vizir.Save_simu(simu, results=["uy", "Svm"], types=[1, 1], folder=folder_save)
print(command)  # prints the vizir command to run for visualization
```

---

## Export to glTF (web / interactive gallery)

{py:func}`~EasyFEA.Utilities.GLTF.Save_simu` exports results as a
[glTF](https://www.khronos.org/gltf/) file for use in web-based 3D viewers.
This is the format used to generate the interactive {doc}`../gallery/index` —
each model displayed there was exported with this function.

```python
from EasyFEA import GLTF

GLTF.Save_simu(simu, ["uy", "Svm"], folder_save)
```

To export a mesh without simulation results, use
{py:func}`~EasyFEA.Utilities.GLTF.Save_mesh`. It also accepts optional
displacement matrices and nodal value arrays for custom animations:

```python
GLTF.Save_mesh(mesh, folder_save)
```

---

## Export to USD (Pixar Universal Scene Description)

{py:func}`~EasyFEA.Utilities.USD.Save_simu` exports to the
[USD](https://openusd.org) format, compatible with Omniverse, USD Composer, and
other DCC tools:

```python
from EasyFEA import USD

USD.Save_simu(simu, ["uy", "Svm"], folder_save)
```

---

## Save and reload a simulation

A completed simulation (including all iteration history) can be saved to disk
and reloaded later without re-running via
{py:meth}`~EasyFEA.Simulations._Simu.Save` and
{py:func}`~EasyFEA.Simulations.Load_Simu`:

```python
from EasyFEA import Simulations

# Save
simu.Save(folder_save)

# Reload
simu = Simulations.Load_Simu(folder_save)
```

Iteration results are stored in `Results/results{N}.pickle` files when
`simu.folder` is set. Only primary unknowns are stored (e.g. displacement,
damage); derived quantities are recomputed on demand.