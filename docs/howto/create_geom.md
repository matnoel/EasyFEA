(howto-geom)=
# Create a geometric object

A **geometric object** (accessible in the {py:mod}`EasyFEA.Geoms` namespace) describes the shape of the domain before its discretization using the meshing methods {py:meth}`~EasyFEA.Geoms._Geom.Mesh_2D`, {py:meth}`~EasyFEA.Geoms._Geom.Mesh_Extrude`, and {py:meth}`~EasyFEA.Geoms._Geom.Mesh_Revolve`.

```{seealso}
- {ref}`howto-mesh`
- {ref}`geoms` API
```

---
## Basic shapes

### Line

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Line

    line = Line((0, 0), (1, 1))
    line.Plot()
```

### Rectangle / box (`Domain`)

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Domain

    domain = Domain((0, 0), (2, 1))
    domain.Plot()
```

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Domain

    box = Domain((0, 0, 0), (2, 1, 0.5))
    box.Plot()
```

### Circle

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Circle

    circle = Circle(center=(0, 0), diam=1.0)
    circle.Plot()
```

`isFilled=False` (default) defines the circle as a boundary (hole or outer contour).
`isFilled=True` defines it as a filled inclusion.

A circle can also be oriented in 3D by specifying a normal vector `n`:

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Circle

    circle = Circle((0, 0), diam=1.0, n=(0.5, 0.5, 0.5))
    circle.Plot()
```

### Circular arc (`CircleArc`)

Three construction modes are available:

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import CircleArc

    # from two end points and a center
    arc = CircleArc((1, 0), (0, 1), center=(0, 0))
    arc.Plot()
```

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import CircleArc

    # from two end points and a radius
    arc = CircleArc((1, 0), (0, 1), R=0.5)
    arc.Plot()
```

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import CircleArc

    # from two end points and a point on the arc
    arc = CircleArc((1, 0), (0, 1), P=(0.8, 0.8))
    arc.Plot()
```

---

## Polygons and contours

### Polygon from a list of points (`Points`)

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Points

    contour = Points([(0, 0), (1, 0), (1, 1), (0, 1)]).Get_Contour()
    contour.Plot()
```

### Add fillets at corners

Assign a fillet radius `r` on individual `Point` objects. Positive `r`
rounds the corner outward; negative `r` rounds it inward:

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Point, Points

    contour = Points([Point(0, 0, r=0.2), (1, 0), (1, 1), (0, 1)]).Get_Contour()
    contour.Plot()
```

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Point, Points

    contour = Points([Point(0, 0, r=-0.2), (1, 0), (1, 1), (0, 1)]).Get_Contour()
    contour.Plot()
```

### Composite contour (`Contour`)

Assemble a closed loop from any mix of `Line`, `CircleArc`, and `Points`:

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Line, CircleArc, Points, Contour

    line       = Line((0, 0), (1, 0))
    points     = Points([(1, 0), (1.5, 0.5), (1, 1)])
    circle_arc = CircleArc((1, 1), (0, 0), center=(1, 0))
    contour    = Contour([line, points, circle_arc])
    contour.Plot()
```

---

(manipulate-example-section)=
## Geometric transformations

All geometry objects support {py:meth}`~EasyFEA.Geoms._Geom.copy`, {py:meth}`~EasyFEA.Geoms._Geom.Translate`, {py:meth}`~EasyFEA.Geoms._Geom.Rotate`, and {py:meth}`~EasyFEA.Geoms._Geom.Symmetry`.
These operations modify the object **in place**; use `copy()` first to preserve the original.

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Points

    contour1 = Points([(0, 0), (1, 0), (1, 1), (0, 1)]).Get_Contour()
    contour2 = contour1.copy(); contour2.Translate(dx=2)
    contour3 = contour2.copy(); contour3.Rotate(90, center=(0, 0), direction=(0, 0, 1))
    contour4 = contour3.copy(); contour4.Symmetry(point=(0, 0), n=(0, 1, 0))

    ax = contour1.Plot_Geoms(
        [contour1, contour2, contour3, contour4], plotPoints=False
    )
    ax.legend(["original", "translated", "rotated", "symmetry"])
```