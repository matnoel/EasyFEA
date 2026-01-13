(geoms)=
# Geoms

The {py:mod}`EasyFEA.Geoms` module in EasyFEA provides essential tools for creating and managing {py:class}`~EasyFEA.Geoms._geom._Geom` objects. These geometric objects are used to construct {py:class}`~EasyFEA.FEM.Mesh` using the {py:class}`~EasyFEA.FEM.Mesher`.

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

Once the geometric objects are created, you can manipulate them using {py:meth}`~EasyFEA.Geoms._Geom.copy`, {py:meth}`~EasyFEA.Geoms.Translate`, {py:meth}`~EasyFEA.Geoms.Rotate`, or {py:meth}`~EasyFEA.Geoms.Symmetry` (see the {ref}`example <manipulate-example-section>` for details).


## Creating a {py:class}`~EasyFEA.Geoms.Line`

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Line
    
    line = Line((0,0), (1,1))
    line.Plot()
```

## Creating a {py:class}`~EasyFEA.Geoms.Domain`

### 2D Domain

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Domain
    
    domain = Domain((0, 0), (1, 1))
    domain.Plot()
```

### 3D Domain

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Domain
    
    domain = Domain((0, 0, 0), (1, 1, 1))
    domain.Plot()
```

## Creating a {py:class}`~EasyFEA.Geoms.Circle`

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Circle
    
    circle = Circle((0, 0), 1.0)    
    circle.Plot()
```


### Using an axis normal to the circle

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Circle
    
    circle = Circle((0, 0), 1.0, n=(0.5, 0.5, 0.5))
    circle.Plot()
```

## Creating a {py:class}`~EasyFEA.Geoms.CircleArc`

### From 2 {py:class}`~EasyFEA.Geoms.Point` and a Center

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import CircleArc
    
    circleArc = CircleArc((1, 0), (0, 1), center=(0, 0))    
    circleArc.Plot()
```

### From 2 {py:class}`~EasyFEA.Geoms.Point` and a Radius

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import CircleArc
    
    circleArc = CircleArc((1, 0), (0, 1), R=0.5)
    circleArc.Plot()
```


### From 2 {py:class}`~EasyFEA.Geoms.Point` and a Point

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import CircleArc
    
    circleArc = CircleArc((1, 0), (0, 1), P=(0.8, 0.8))
    circleArc.Plot()
```

## Creating a {py:class}`~EasyFEA.Geoms.Contour` from {py:class}`~EasyFEA.Geoms.Points`

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Points
    
    contour = Points([(0, 0), (1,0), (1,1), (0,1)]).Get_Contour()
    contour.Plot()
```

### Add a fillet

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Point, Points
    
    contour = Points([Point(0, 0, r=0.5), (1,0), (1,1), (0,1)]).Get_Contour()
    contour.Plot()
```

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Point, Points
    
    contour = Points([Point(0, 0, r=-0.5), (1,0), (1,1), (0,1)]).Get_Contour()
    contour.Plot()
```

## Creating a {py:class}`~EasyFEA.Geoms.Contour` with {py:class}`~EasyFEA.Geoms.Line`, {py:class}`~EasyFEA.Geoms.CircleArc` and {py:class}`~EasyFEA.Geoms.Points`

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Line, CircleArc, Points, Contour
    
    line = Line((0, 0), (1, 0))
    points = Points([(1,0), (1.5, 0.5), (1, 1)])
    circleArc = CircleArc((1,1), (0,0), center=(1,0))
    contour = Contour([line, points, circleArc])
    contour.Plot()
```

(manipulate-example-section)=

## Manipulate a {py:class}`~EasyFEA.Geoms._Geom` object using the {py:meth}`~EasyFEA.Geoms._Geom.copy`, {py:meth}`~EasyFEA.Geoms.Translate`, {py:meth}`~EasyFEA.Geoms.Rotate`, and {py:meth}`~EasyFEA.Geoms.Symmetry` functions

```{eval-rst}
.. jupyter-execute::

    from EasyFEA.Geoms import Points

    contour1 = Points([(0,0), (1,0), (1,1), (0,1)]).Get_Contour()
    contour2 = contour1.copy(); contour2.Translate(dx=4)
    contour3 = contour2.copy(); contour3.Rotate(90, (0,0), (0,0,1))
    contour4 = contour3.copy(); contour4.Symmetry((0,0), (0,1))
    
    ax = contour1.Plot_Geoms([contour1, contour2, contour3, contour4], plotPoints=False)
    ax.legend(["contour1", "contour2", "contour3", "contour4"])
```

## Geoms API

```{eval-rst}
.. automodule:: EasyFEA.Geoms
```