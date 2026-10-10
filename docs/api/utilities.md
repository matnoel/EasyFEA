(utilities)=

# Utilities, Viz and IO

Once `simu.Solve()` has run, {py:mod}`EasyFEA.Viz` draws the results,
{py:mod}`EasyFEA.IO` reads and writes meshes and simulations, and
{py:mod}`EasyFEA.Utilities` holds the helpers every layer uses (console, folders,
timing).

- {py:mod}`~EasyFEA.Viz.Matplotlib` and {py:mod}`~EasyFEA.Viz.PyVista` are the two
  viewers: `from EasyFEA import Matplotlib, PyVista`.
- {py:mod}`EasyFEA.IO` has one module per format family, read as
  `IO.Gmsh.Load_mesh(...)`: {py:mod}`~EasyFEA.IO.Gmsh`, {py:mod}`~EasyFEA.IO.Medit`,
  {py:mod}`~EasyFEA.IO.Ensight`, {py:mod}`~EasyFEA.IO.PyVista`,
  {py:mod}`~EasyFEA.IO.Paraview`, {py:mod}`~EasyFEA.IO.Vizir`, {py:mod}`~EasyFEA.IO.USD`
  and {py:mod}`~EasyFEA.IO.GLTF`.

```{eval-rst}
.. autosummary::
    ~EasyFEA.Viz.Matplotlib
    ~EasyFEA.Viz.PyVista
    ~EasyFEA.IO.Gmsh
    ~EasyFEA.IO.Medit
    ~EasyFEA.IO.Ensight
    ~EasyFEA.IO.PyVista
    ~EasyFEA.IO.Paraview
    ~EasyFEA.IO.Vizir
    ~EasyFEA.IO.USD
    ~EasyFEA.IO.GLTF
    ~EasyFEA.Utilities.Terminal
    ~EasyFEA.Utilities.Folder
```

```{seealso}
- {ref}`howto-postprocess`
- {ref}`howto-import-mesh`
```

## Viz API

```{eval-rst}
.. automodule:: EasyFEA.Viz.Matplotlib
.. automodule:: EasyFEA.Viz.PyVista
```

## IO API

```{eval-rst}
.. automodule:: EasyFEA.IO.Gmsh
.. automodule:: EasyFEA.IO.Medit
.. automodule:: EasyFEA.IO.Ensight
.. automodule:: EasyFEA.IO.PyVista
.. automodule:: EasyFEA.IO.Paraview
.. automodule:: EasyFEA.IO.Vizir
.. automodule:: EasyFEA.IO.USD
.. automodule:: EasyFEA.IO.GLTF
```

## Utilities API

```{eval-rst}
.. automodule:: EasyFEA.Utilities
    :imported-members:
.. automodule:: EasyFEA.Utilities.Terminal
.. automodule:: EasyFEA.Utilities.Folder
```
