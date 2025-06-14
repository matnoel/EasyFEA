.. _easyfea-api-geoms:

geoms
=====

To construct a :py:class:`~EasyFEA.fem.Mesh` using the :py:class:`~EasyFEA.fem.Mesher`, you must first define one or more :py:class:`~EasyFEA.geoms._Geom` objects.

.. autosummary::    
    ~EasyFEA.geoms.Point
    ~EasyFEA.geoms._Geom
    ~EasyFEA.geoms.Points
    ~EasyFEA.geoms.Domain
    ~EasyFEA.geoms.Line
    ~EasyFEA.geoms.Circle
    ~EasyFEA.geoms.CircleArc
    ~EasyFEA.geoms.Contour 

Creating a :py:class:`~EasyFEA.geoms.Contour` from :py:class:`~EasyFEA.geoms.Points`
------------------------------------------------------------------------------------

.. plot::
    :include-source:

    from EasyFEA.Geoms import Points
    
    contour = Points([(0, 0), (1,0), (1,1), (0,1)]).Get_Contour()
    contour.Plot()

Add a fillet
^^^^^^^^^^^^

.. plot::
    :include-source:

    from EasyFEA.Geoms import Point, Points
    
    contour = Points([Point(0, 0, r=0.5), (1,0), (1,1), (0,1)]).Get_Contour()
    contour.Plot()

.. plot::
    :include-source:

    from EasyFEA.Geoms import Point, Points
    
    contour = Points([Point(0, 0, r=-0.5), (1,0), (1,1), (0,1)]).Get_Contour()
    contour.Plot()

Creating a :py:class:`~EasyFEA.geoms.Domain`/Box
------------------------------------------------

2D Domain
^^^^^^^^^

.. plot::
    :include-source:

    from EasyFEA.Geoms import Domain
    
    domain = Domain((0, 0), (1, 1))
    domain.Plot()

3D Domain
^^^^^^^^^

.. plot::
    :include-source:

    from EasyFEA.Geoms import Domain
    
    domain = Domain((0, 0, 0), (1, 1, 1))
    domain.Plot()

Creating a :py:class:`~EasyFEA.geoms.Line`
------------------------------------------

.. plot::
    :include-source:

    from EasyFEA.Geoms import Line
    
    line = Line((0,0), (1,1))
    line.Plot()


Creating a :py:class:`~EasyFEA.geoms.Circle`
--------------------------------------------

.. plot::
    :include-source:

    from EasyFEA.Geoms import Circle
    
    circle = Circle((0, 0), 1.0)    
    circle.Plot()

Using an axis normal to the circle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. plot::
    :include-source:

    from EasyFEA.Geoms import Circle
    
    circle = Circle((0, 0), 1.0, n=(0.5, 0.5, 0.5))
    circle.Plot()

Creating a :py:class:`~EasyFEA.geoms.CircleArc`
-----------------------------------------------

From 2 :py:class:`~EasyFEA.geoms.Point` and a Center
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. plot::
    :include-source:

    from EasyFEA.Geoms import CircleArc
    
    circleArc = CircleArc((1, 0), (0, 1), center=(0, 0))    
    circleArc.Plot()

From 2 :py:class:`~EasyFEA.geoms.Point` and a Radius
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. plot::
    :include-source:

    from EasyFEA.Geoms import CircleArc
    
    circleArc = CircleArc((1, 0), (0, 1), R=0.5)
    circleArc.Plot()

From 2 :py:class:`~EasyFEA.geoms.Point` and a Point
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. plot::
    :include-source:

    from EasyFEA.Geoms import CircleArc
    
    circleArc = CircleArc((1, 0), (0, 1), P=(0.8, 0.8))
    circleArc.Plot()

Creating a :py:class:`~EasyFEA.geoms.Contour` with :py:class:`~EasyFEA.geoms.Line`, :py:class:`~EasyFEA.geoms.CircleArc` and :py:class:`~EasyFEA.geoms.Points`
--------------------------------------------------------------------------------------------------------------------------------------------------------------

.. plot::
    :include-source:

    from EasyFEA.Geoms import Line, CircleArc, Points, Contour
    
    line = Line((0, 0), (1, 0))
    points = Points([(1,0), (1.5, 0.5), (1, 1)])
    circleArc = CircleArc((1,1), (0,0), center=(1,0))
    contour = Contour([line, points, circleArc])
    contour.Plot()

Detailed geoms api
------------------

.. automodule:: EasyFEA.geoms
   :members:
   :private-members:
   :undoc-members:
   :imported-members:
   :show-inheritance:
   