(geoms)=
# Geoms

The {py:mod}`EasyFEA.Geoms` module provides essential tools for creating and managing {py:class}`~EasyFEA.Geoms._geom._Geom` objects. These geometric objects are used to construct {py:class}`~EasyFEA.FEM.Mesh` using the {py:class}`~EasyFEA.FEM.Mesher`.

In the simulation workflow, `Geoms` is the **first step**: you describe the domain shape here before passing it to the mesher.

```{seealso}
- {ref}`howto-geom`
```

With this module, you can construct:

```{eval-rst}
.. autosummary::
    ~EasyFEA.Geoms.Point
    ~EasyFEA.Geoms.Points
    ~EasyFEA.Geoms.Domain
    ~EasyFEA.Geoms.Line
    ~EasyFEA.Geoms.Circle
    ~EasyFEA.Geoms.CircleArc
    ~EasyFEA.Geoms.Contour
```

Once the geometric objects are created, you can manipulate them using {py:meth}`~EasyFEA.Geoms._Geom.copy`, {py:meth}`~EasyFEA.Geoms.Translate`, {py:meth}`~EasyFEA.Geoms.Rotate`, or {py:meth}`~EasyFEA.Geoms.Symmetry` (see the {ref}`examples <manipulate-example-section>` for details).

## Geoms API

```{eval-rst}
.. automodule:: EasyFEA.Geoms
```