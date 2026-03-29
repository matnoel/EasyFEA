(utilities)=
# Utilities

The {py:mod}`EasyFEA.Utilities` module provides essential tools for post-processing.

In the simulation workflow, `Utilities` is the **final step**: once `simu.Solve()` has run, these tools visualize results, export to external formats, and manage files. {py:mod}`~EasyFEA.Utilities.Display` and {py:mod}`~EasyFEA.Utilities.PyVista` cover interactive visualization; {py:mod}`~EasyFEA.Utilities.Paraview`, {py:mod}`~EasyFEA.Utilities.GLTF`, and {py:mod}`~EasyFEA.Utilities.USD` handle external export.

```{eval-rst}
.. autosummary::
    ~EasyFEA.Utilities.Display
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
.. automodule:: EasyFEA.Utilities.Display
    :exclude-members: Mesh, Tic, ElemType, GroupElemFactory
.. automodule:: EasyFEA.Utilities.Folder
.. automodule:: EasyFEA.Utilities.MeshIO
    :exclude-members: Mesh, Tic, ElemType, GroupElemFactory, _GroupElem    
.. automodule:: EasyFEA.Utilities.Paraview
    :exclude-members: Mesh, Tic, ElemType, GroupElemFactory    
.. automodule:: EasyFEA.Utilities.PyVista
    :exclude-members: Mesh, Tic, ElemType, GroupElemFactory    
.. automodule:: EasyFEA.Utilities.Vizir
    :exclude-members: Mesh, Tic, ElemType, GroupElemFactory
.. automodule:: EasyFEA.Utilities.USD
    :exclude-members: Mesh, Tic, ElemType, GroupElemFactory, _Simu
.. automodule:: EasyFEA.Utilities.GLTF
    :exclude-members: Mesh, Tic, ElemType, GroupElemFactory, _Simu
```