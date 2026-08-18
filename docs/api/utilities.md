(utilities)=
# Utilities

The {py:mod}`EasyFEA.Utilities` module provides essential tools for post-processing.

In the simulation workflow, `Utilities` is the **final step**: once `simu.Solve()` has run, these tools visualize results, export to external formats, and manage files. {py:mod}`~EasyFEA.Utilities.Matplotlib` and {py:mod}`~EasyFEA.Utilities.PyVista` cover interactive visualization; {py:mod}`~EasyFEA.Utilities.Terminal` provides console helpers; {py:mod}`~EasyFEA.Utilities.Paraview`, {py:mod}`~EasyFEA.Utilities.GLTF`, and {py:mod}`~EasyFEA.Utilities.USD` handle external export.

```{eval-rst}
.. autosummary::
    ~EasyFEA.Utilities.Matplotlib
    ~EasyFEA.Utilities.Terminal
    ~EasyFEA.Utilities.Folder
    ~EasyFEA.Utilities.MeshIO
    ~EasyFEA.Utilities.Paraview
    ~EasyFEA.Utilities.PyVista
    ~EasyFEA.Utilities.Vizir
    ~EasyFEA.Utilities.USD
    ~EasyFEA.Utilities.GLTF
```

```{seealso}
- {ref}`howto-postprocess`
- {ref}`howto-import-mesh`
```

## Utilities API

```{eval-rst}
.. automodule:: EasyFEA.Utilities
    :imported-members:
.. automodule:: EasyFEA.Utilities.Matplotlib
.. automodule:: EasyFEA.Utilities.Terminal
.. automodule:: EasyFEA.Utilities.Folder
.. automodule:: EasyFEA.Utilities.MeshIO
.. automodule:: EasyFEA.Utilities.Paraview
.. automodule:: EasyFEA.Utilities.PyVista
.. automodule:: EasyFEA.Utilities.Vizir
.. automodule:: EasyFEA.Utilities.USD
.. automodule:: EasyFEA.Utilities.GLTF
```