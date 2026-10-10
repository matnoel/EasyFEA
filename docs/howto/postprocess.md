(howto-postprocess)=

# Post-process simulation results

**Post-processing** tools visualize and export simulation results. The viewers live in
{py:mod}`EasyFEA.Viz`: {py:mod}`~EasyFEA.Viz.Matplotlib` for static matplotlib figures
and {py:mod}`~EasyFEA.Viz.PyVista` for interactive 3D views. Exports live in
{py:mod}`EasyFEA.IO`: {py:mod}`~EasyFEA.IO.Paraview` / {py:mod}`~EasyFEA.IO.Vizir` /
{py:mod}`~EasyFEA.IO.GLTF` / {py:mod}`~EasyFEA.IO.USD` / {py:mod}`~EasyFEA.IO.Gmsh`.

```{eval-rst}
.. autosummary::
    ~EasyFEA.Viz.Matplotlib
    ~EasyFEA.Viz.PyVista
    ~EasyFEA.IO.Paraview
    ~EasyFEA.IO.Vizir
    ~EasyFEA.IO.USD
    ~EasyFEA.IO.GLTF
    ~EasyFEA.IO.Gmsh
```

______________________________________________________________________

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

______________________________________________________________________

## Plot with matplotlib (Matplotlib)

{py:mod}`~EasyFEA.Viz.Matplotlib` uses matplotlib and is the primary tool for 2D and 3D
result visualization.

### Plot a scalar field

{py:func}`~EasyFEA.Viz.Matplotlib.Plot` plots any result field on the mesh:

```python
from EasyFEA import Matplotlib

Matplotlib.Plot(simu, "uy")
Matplotlib.Plot(simu, "Svm", plotMesh=True, nColors=11)
```

Matplotlib and PyVista share their function names and options, so a call switches viewer
by changing the module. The shared options:

| Group      | Options                                                                                     |
| ---------- | ------------------------------------------------------------------------------------------- |
| Object     | `deformFactor` (`0` = undeformed)                                                           |
| Field      | `result`, `coef` (e.g. unit conversion), `nodeValues` (`False` for element-constant)        |
| Colorbar   | `cmap`, `nColors`, `clim=(min, max)`, `colorbarTitle`, `plotColorbar`, `verticalColorbar`   |
| Style      | `color`, `edgecolor`, `linewidth`, `alpha`, `plotMesh`, `plotNodes`, `nodeSize`             |
| Annotation | `title`, `label`, `showId`, `showGrid`, `plotLegend`, `bounds=(xmin, xmax, ..., zmax)`      |

The view covers everything drawn in the figure; `bounds` fixes it instead. Draw into an
existing figure with `ax=` (Matplotlib) or `plotter=` (PyVista). Other keywords go to the
backend's draw call, except its own spelling of a shared option (`lw`, `line_width`, ...),
which raises `ValueError`. `Matplotlib.Plot` also saves the figure with `folder` /
`filename`.

### Plot the mesh

{py:func}`~EasyFEA.Viz.Matplotlib.Plot_Mesh` plots the mesh:

```python
Matplotlib.Plot_Mesh(simu)                   # current state
Matplotlib.Plot_Mesh(simu, deformFactor=10)  # amplified deformation
Matplotlib.Plot_Mesh(mesh)                   # mesh object directly
```

### Plot boundary conditions

{py:func}`~EasyFEA.Viz.Matplotlib.Plot_BoundaryConditions` visualizes the applied loads
and constraints:

```python
Matplotlib.Plot_BoundaryConditions(simu)
```

### Plot tags

{py:func}`~EasyFEA.Viz.Matplotlib.Plot_Tags` shows the physical groups and tags defined
on the mesh:

```python
Matplotlib.Plot_Tags(mesh)
```

### Plot energy and iteration history

```python
Matplotlib.Plot_Energy(simu, folder=folder_save)
Matplotlib.Plot_Iter_Summary(simu, folder=folder_save)
```

### Save a figure

{py:func}`~EasyFEA.Viz.Matplotlib.Save_fig` saves the current matplotlib figure to disk:

```python
Matplotlib.Save_fig(folder_save, "my_figure")
```

### Create an animation

{py:func}`~EasyFEA.Viz.Matplotlib.Movie_simu` generates an animation directly from a
named result field:

```python
Matplotlib.Movie_simu(simu, "uy", folder=folder_save, filename="animation.gif")
```

For custom frame content, use {py:func}`~EasyFEA.Viz.Matplotlib.Movie_func` with a
user-defined function. The function receives the matplotlib figure and the frame index
`i`:

```python
import numpy as np

iterations = np.arange(0, simu.Niter, max(1, simu.Niter // 20))

def Func(fig, i):
    fig.clear()
    ax = fig.add_subplot(111)
    simu.Set_Iter(iterations[i])
    Matplotlib.Plot(simu, "uy", ax=ax)

Matplotlib.Movie_func(Func, iterations.size, folder_save, "animation.gif")
```

______________________________________________________________________

## Interactive 3D visualization (PyVista)

{py:mod}`~EasyFEA.Viz.PyVista` provides interactive 3D rendering powered by
[PyVista](https://pyvista.org).

{py:func}`~EasyFEA.Viz.PyVista.Plot` renders a result field in an interactive window:

```python
from EasyFEA import PyVista

PyVista.Plot(simu, "uy")
PyVista.Plot(simu, "Svm", plotMesh=True, clim=(0, 500))
PyVista.Plot_Mesh(simu)
PyVista.Plot_BoundaryConditions(simu)
PyVista.Plot_Tags(simu)
```

### Create an animation

{py:func}`~EasyFEA.Viz.PyVista.Movie_simu` generates an animation directly from a named
result field:

```python
PyVista.Movie_simu(simu, "uy", folder=folder_save, filename="animation.gif")
```

For custom frame content, use {py:func}`~EasyFEA.Viz.PyVista.Movie_func` with a
user-defined function. The function receives the PyVista plotter and the frame index
`i`:

```python
import numpy as np

iterations = np.arange(0, simu.Niter, max(1, simu.Niter // 20))

def Func(plotter, i):
    simu.Set_Iter(iterations[i])
    PyVista.Plot(simu, "damage", plotter=plotter, clim=(0, 1))

PyVista.Movie_func(Func, iterations.size, folder_save, "damage.gif")
```

______________________________________________________________________

## Export to ParaView

{py:func}`~EasyFEA.IO.Paraview.Save_simu` generates a `.pvd` timeline and `.vtu` files
that ParaView reads directly:

```python
from EasyFEA import IO

IO.Paraview.Save_simu(simu, folder_save, N=200)
```

`N` controls the maximum number of iterations exported — EasyFEA selects up to `N`
equally-spaced snapshots from the full iteration history. Open the resulting
`Paraview/simulation.pvd` file in ParaView to browse the timeline.

See {ref}`howto-mpi` for parallel ParaView export across MPI ranks.

______________________________________________________________________

## Export to Vizir

{py:func}`~EasyFEA.IO.Vizir.Save_simu` exports results to the
[Vizir](https://pyamg.saclay.inria.fr/vizir4.html) format, a high-order FEM
visualization tool developed by INRIA:

```python
from EasyFEA import IO

command = IO.Vizir.Save_simu(simu, folder_save, results=["uy", "Svm"], types=[1, 1])
print(command)  # prints the vizir command to run for visualization
```

______________________________________________________________________

## Export to glTF (web / interactive gallery)

{py:func}`~EasyFEA.IO.GLTF.Save_simu` exports results as a
[glTF](https://www.khronos.org/gltf/) file for use in web-based 3D viewers. This is the
format used to generate the interactive {doc}`../gallery/index` — each model displayed
there was exported with this function.

```python
from EasyFEA import IO

IO.GLTF.Save_simu(simu, folder_save, results=["uy", "Svm"])
```

To export a mesh without simulation results, use {py:func}`~EasyFEA.IO.GLTF.Save_mesh`.
It also accepts optional displacement matrices and nodal value arrays for custom
animations:

```python
IO.GLTF.Save_mesh(mesh, folder_save)
```

______________________________________________________________________

## Export to USD (Pixar Universal Scene Description)

{py:func}`~EasyFEA.IO.USD.Save_simu` exports to the [USD](https://openusd.org) format,
compatible with Omniverse, USD Composer, and other DCC tools:

```python
from EasyFEA import IO

IO.USD.Save_simu(simu, folder_save, results=["uy", "Svm"])
```

______________________________________________________________________

## Save and reload a simulation

A completed simulation (including all iteration history) can be saved to disk and
reloaded later without re-running via {py:meth}`~EasyFEA.Simulations._Simu.Save` and
{py:func}`~EasyFEA.Simulations.Load_Simu`:

```python
from EasyFEA import Simulations

# Save
simu.Save(folder_save)

# Reload
simu = Simulations.Load_Simu(folder_save)
```

Iteration results are stored in `Results/results{N}.pickle` files when `simu.folder` is
set. Only primary unknowns are stored (e.g. displacement, damage); derived quantities
are recomputed on demand.
