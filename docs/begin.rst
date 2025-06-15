.. _begin:

Beginner's Guide
================

Like every python script, you first start by importing modules contained within the python package.

.. jupyter-execute::

    from EasyFEA import Display, Mesher, ElemType, Materials, Simulations
    from EasyFEA.Geoms import Point, Domain

----

Most EasyFEA simulations require few modules, this script requires just a few:

.. autosummary::
    ~EasyFEA.utilities.Display 
    ~EasyFEA.fem.Mesher 
    ~EasyFEA.fem.ElemType 
    ~EasyFEA.Materials
    ~EasyFEA.Simulations
    ~EasyFEA.Geoms

Once everything is imported, you can now create a :py:class:`~EasyFEA.fem.Mesh`:

.. jupyter-execute::

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    L = 120  # mm
    h = 13

    domain = Domain(Point(), Point(L, h), h / 3)
    mesh = Mesher().Mesh_2D(domain, [], ElemType.QUAD9, isOrganised=True)
    Display.Plot_Mesh(mesh)
    
----

Once the :py:class:`~EasyFEA.fem.Mesh` is created you can you can create a Material and Simulation:

.. jupyter-execute::

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    E = 210000  # MPa
    v = 0.3
    F = -800  # N

    mat = Materials.ElasIsot(2, E, v, planeStress=True, thickness=h)

    simu = Simulations.ElasticSimu(mesh, mat)
    
----

Once the simulation has been set up, defining the boundary conditions, solving the problem and visualizing the results is straightforward.

.. jupyter-execute::
    
    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

    simu.add_dirichlet(nodesX0, [0, 0], ["x", "y"])
    simu.add_surfLoad(nodesXL, [F / h / h], ["y"])

    simu.Solve()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    Display.Plot_Mesh(simu, deformFactor=10)
    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Result(simu, "uy", plotMesh=True)
    Display.Plot_Result(simu, "Svm", plotMesh=True, ncolors=11)
    
----

This script is available in the :doc:`HelloWorld example <examples/HelloWorld>`.

For additional details, please refer to either the :doc:`EasyFEA API documentation <easyfea>` or the comprehensive collection of :doc:`Examples <examples/index>`.

