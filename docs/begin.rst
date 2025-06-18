.. _begin:

Beginner's Guide
================

Like any Python script, you should start by importing the core modules from the EasyFEA package:

.. jupyter-execute::

    from EasyFEA import Display, ElemType, Materials, Simulations
    from EasyFEA.Geoms import Domain

----

The most commonly used modules in EasyFEA are:

.. autosummary::
    ~EasyFEA.utilities.Display 
    ~EasyFEA.fem.ElemType 
    ~EasyFEA.Materials
    ~EasyFEA.Simulations
    ~EasyFEA.Geoms

Let's now create a 2D :py:class:`~EasyFEA.fem.Mesh` using a simple rectangular domain:

.. jupyter-execute::

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    L = 120  # mm
    h = 13

    domain = Domain((0, 0), (L, h), h / 3)
    mesh = domain.Mesh_2D([], ElemType.QUAD9, isOrganised=True)
    Display.Plot_Mesh(mesh)
    
----

Next, define a linear :py:class:`~EasyFEA.materials.ElasIsot` material and set up the :py:class:`~EasyFEA.Simulations.ElasticSimu`  simulation:

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

