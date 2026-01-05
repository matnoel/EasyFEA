# Changelog

This document describes the changes made to the project.

## 1.6.4 (January 5, 2026):

- Updated sphinx documentation.
- Updated `ElasOrthotropic` implementation.
- Defined `ElasOrthotropic.Walpole_Decomposition` method.
- Updated how `Walpole_Decomposition` functions are checked during unitests.
- Create `mesh.Calc_regulation_projector`.
- Updated mpi4py and numba as optional dependencies.

## 1.6.3 (January 2, 2026):

- Updated the `Result_in_Strain_or_Stress_field` function and its calls to replace `field_e` with `field_e_pg`.
- Added `Field.Evaluate_e` and `Field.Evaluate_n` methods to evaluate functions on elements or nodes.
- Added the `von_mises_stress` function in `examples/WeakForms/LinearElasticity1.py`.
- Fixed a bug in the `Display.Plot_Result` function when `nodeValues=False`.
- Clarified the `Mesher._Mesh_Get_Mesh` function and moved the `_gmsh_interface.py` module to `_gmsh.py`.
- Began work on issue [#26](https://github.com/matnoel/EasyFEA/issues/26): Solve simulations using petsc4py and MPI.
- Added the `Gauss._Gauss_factory_nPg` function and refactored Gauss point construction.
- Updated the `Paraview.__Make_pvd` function to use relative paths for accessing VTU files.
- Improved the `MeshIO` module.
- Renamed `mesh.coordGlob` to `mesh.coord`.
- Created `examples/LinearizedElasticity/Homog4.py`.
- Implemented the `ElasOrthotropic` model.
- Fixed issue [#27](https://github.com/matnoel/EasyFEA/issues/27): Corrected bug in `PhaseFieldSimu._Calc_Psi_Ext` method.
- Updated iteration printing in Display, Paraview, and PyVista functions.
- Addressed issue [#29](https://github.com/matnoel/EasyFEA/issues/29): Clarified the `Simu.add_pressureLoad()` function.
- Updated the `simu.add_neumann` function.
- Resolved issue [#28](https://github.com/matnoel/EasyFEA/issues/28): Added topology optimization example.
- Enhanced the Sphinx documentation.

## 1.6.2 (November 20, 2025):

- Updated homogenization examples.
- Improved documentation.
- Removed pandas dependency.
- Enhanced `Tic.Plot_History` functions.
- Updated the `mesh.Evaluate_dofsValues_at_coordinates` function.
- Refactored mesh functions in `_gmsh_interface.py`.
- Updated functions in `EasyFEA/fem/_forms.py` used by `EasyFEA/simulations/_weak_forms.py`.

## 1.6.1 (November 9, 2025):

- Updated the thermal simulation description.
- Refactored the use of time schemes for nonlinear problems.
- Clarified the names and descriptions of the simulation solver functions.

## 1.6.0 (November 8, 2025):

- Updated the `Display` and `PyVista` functions.
- Updated the construction of `dict_tags` in the `_Meshio_to_EasyFEA` function.
- Improved the `hyperelastic` simulation to handle `hyperbolic` problems.
- Organized and enhanced the documentation scripts.
- Improved `hyperelastic` models for `2D` simulations under the plane strain assumption.

## 1.5.5 (November 5, 2025):

- Updated the solver parameter in the `phasefield` model.
- Added an assertion in `_Simu.Get_K_C_M_F` to detect multiple problem types.
- Updated `PositiveParameter` to `PositiveScalarParameter` in some models [#23](https://github.com/matnoel/EasyFEA/issues/23).
- Renamed `_pyVistaMesh` to `_pvMesh` in `EasyFEA/utilities/PyVista.py`.
- Updated `PyVista._setCameraPosition` with its docstrings and modified how the function is called.
- Added the `I6` and `I8` invariants in `EasyFEA/models/_hyperelastic.py`.
- Added `VectorParameter` and `_CheckIsVector` in `EasyFEA/utilities/_params.py`.
- Updated the behavior of `TensorProd` and `Project_Kelvin` functions for `FeArray`.
- Updated vector checks in `EasyFEA/models/_hyperelastic.py`.
- Created the `HolzapfelOgden` hyperelastic law in `EasyFEA/models/_hyperelastic_laws.py`.
- Updated FEM solvers:
    - Updated the `Newton_Raphson` algorithm.
    - Removed the `simu.solverIsIncremental` option.
    - Updated `newmark`, `midpoint`, and `hht` acceleration formulations to displacement formulations.
    - Updated `Set_Rayleigh_Damping_Coefs` to ensure the matrix is updated in `EasyFEA/simulations/_elastic.py`.
    - Revised the use of the `Newton_Raphson` algorithm in nonlinear simulations.
    - Updated the `Newton_Raphson` algorithm to print the residual L2 norm (with applied boundary conditions).
- Updated the `TensorProd` function in `EasyFEA/fem/_linalg.py`.
- Updated hyperelastic chain rules to ensure cross-term derivatives used in `d2W` utilize tensor products.
- Refactored the `HyperElastic` static class into a `HyperElasticState` class for improved performance and code readability.
- Applied the `cache_computed_values` decorator to replace outdated caching methods in `EasyFEA/fem/_group_elems.py` and `EasyFEA/models/_hyperelastic.py`.
- Fixed a bug in the first derivatives of anisotropic invariants `I4`, `I6`, and `I8`.
- Fixed a bug in the penalization solver.
- Added the bulk modulus term to the `MooneyRivlin` hyperelastic law.
- Added the bulk modulus term to the `Saint-Venant-Kirchhoff` hyperelastic law.

## 1.5.4 (October 23, 2025):

- Fixed issue [#23](https://github.com/matnoel/EasyFEA/issues/23): Descriptors are now used to simplify property creation in classes.
- Fixed issue [#24](https://github.com/matnoel/EasyFEA/issues/24): Introduced the `Construct_matrix_system` function and removed all `Assembly` functions. This reduces the amount of code required for each simulation.
- Fixed a bug in the `_Create_Lines` dispatch methods.
- Updated the `_Write_solution_file` function in `utilities.Vizir.py`.
- Fixed a bug in `Get_pointsInElem` caused by `nan` values in normalized arrays.
- Added the `mesh.Evaluate_dofsValues_at_coordinates` function.
- Added the `HolzapfelOgden` hyperelastic law in the `examples/HyperElastic/HyperElasticLaws.py` script.
- Removed the `useNumba` property from `simu` and `models`.
- Removed the `dim` argument from the `Thermal` model.
- Updated functions in `EasyFEA/utilities/_params.py`.
- Clarified tests for `phasefield`.

## 1.5.3 (October 1, 2025):

- Added the `I8` invariant to the `examples/HyperElastic/HyperElasticInvariants.py` script.
- Fixed issue [#22](https://github.com/matnoel/EasyFEA/issues/22): Implemented `singledispatch` methods in `_gmsh_interface.py`, `geoms/_utils.py`, `_simu.py` and `PyVista.py` to improve code readability.
- Introduced the `BUILDING_GALLERY` option and updated the `Display.Clear` function to ensure the `Tic` history is initialized when `BUILDING_GALLERY = True`.

## 1.5.2 (September 11, 2025):

- Fixed a bug in the `Mesh.Translate`, `Mesh.Rotate`, and `Mesh.Symmetry` functions.
- Added the `MeshIO.EasyFEA_to_Ensight` and `MeshIO.Ensight_to_EasyFEA` functions.
- Fixed a bug in the matrix order in the `Paraview.Save_simu` function.
- Added a `depth` option to the `Folder.Dir` function.
- Updated the docstrings for weak forms.
- Fixed a bug in the `Vizir.__Write_HOSolAt_Solution` function.
- Added a `concatenate` option to the `Field.Get_coords` function.
- Added the `groupElem.Get_normals_e_pg` function.
- Updated `mesh.Get_normals` to use the `groupElem.Get_normals_e_pg` function.
- Fixed the path used in the Sphinx documentation.
- Fixed a bug in `Mesh.Nodes_Tags`.

## 1.5.1 (July 23, 2025):

- Improved documentation by adding more animated GIFs
- Updated `MeshIO.Surface_reconstruction`.
- Updated `Material.WeakForms` docstrings.
- Fixed issue [#21](https://github.com/matnoel/EasyFEA/issues/21):
    - Clarified `Display.py` and `PyVista.py` modules.
    - Improved `Plot_Tags` functions.
- Improved mesh and element group tagging.
- Fixed bug in `_Additional_Points()`.
- Switched `GroupElemFactory` `_Create` and `Create`.

## 1.5.0 (July 16, 2025):

- Updated Python version in README.md.
- Updated Display module according to modifications in [#19](https://github.com/matnoel/EasyFEA/issues/19).
- Updated Vizir module.
- Updated MeshIO docstrings.
- Fixed mypy and ruff type issues.
- Renamed `linesVector_e` to `rowsVector_e` and `linesScalar_e` to `rowsScalar_e` in simulations and group of elements.
- Fixed issue [#20](https://github.com/matnoel/EasyFEA/issues/20): Users can now perform finite element analysis by simply providing weak form functions.
  - Renamed Materials to Models.
  - Created `Field`, `BilinearForm`, `LinearForm`, `Materials.WeakForms`, and `Simulations.WeakFormSimu`.
  - Created 2 Poisson and 2 linear elasticity examples with tests.
- Improved the documentation by providing as many animated GIFs as possible.

## 1.4.8 (July 7, 2025):

- Updated meshio's fork dependency in `pyproject.toml` [dev].
- Fixed mypy types.

## 1.4.7 (July 7, 2025):

- Updated visualization modules: `Vizir.py`, `PyVista`, and `Display.py`.
- Added new topological information (`Nedge`, `Nvolume`) for each element group.
- Added `surfaces` and `edges` data for each element group.
- Created `MeshIO.Surface_reconstruction()` method.
- Added meshio tests covering surface reconstruction and import functionality.
- Updated meshio's fork dependencies in:
  - `.github/workflows/tests.yaml`
  - `docs/requirements.txt`
  - `pyproject.toml` [dev]
- Added MeshIO import functions in FEM API documentation.
- Improved convergence criteria in `_Solver_Solve_NewtonRaphson`.
- Updated mesh used in `test_PhaseField` function.
- Moved `ModelType.Is_Non_Linear` to `simu._Solver_problemType_is_non_linear`.

## 1.4.6 (July 2, 2025):

- Updated links in `docs/index.rst` and `README.md`.
- Renamed `weightedJacobian_e_pg` to `wJ_e_pg` in scripts.
- Updated `Solvers.py` dependencies.
- Reordered tensor ordering in `Paraview.py`.
- Fixed issue [#17](https://github.com/matnoel/EasyFEA/issues/17): Fixed Newton-Raphson algorithm and added a new hyperelastic example.
- Fixed issue [#18](https://github.com/matnoel/EasyFEA/issues/18): Updated `MeshIO.py` to import `EnSight` meshes.

## 1.4.5 (June 25, 2025):

- Fixed issue [#16](https://github.com/matnoel/EasyFEA/issues/16): Enabled Paraview functionality without a simulation object.
- Updated documentation links to reference external research projects.

## 1.4.4 (June 21, 2025):

- Fixed issue [#15](https://github.com/matnoel/EasyFEA/issues/15): You can now create meshes simply by using geometric objects. 
- Enhanced docstrings, documentation and examples.

## 1.4.3 (June 16, 2025):

- Added an interface to Vizir (see issue [#9](https://github.com/matnoel/EasyFEA/issues/9)).
- Applied `black` code formatting across the codebase (see issue #10).
- Integrated continuous integration using GitHub Actions (see issues [#11](https://github.com/matnoel/EasyFEA/issues/11), [#13](https://github.com/matnoel/EasyFEA/issues/13), and [#14](https://github.com/matnoel/EasyFEA/issues/14)).
- Added comprehensive documentation (see issue [#5](https://github.com/matnoel/EasyFEA/issues/5)).
- Fixed tkinter issue in CI for py3.12 on windows. (see: https://github.com/matnoel/EasyFEA/actions/runs/15673958144/job/44150031408)

## 1.4.2 (June 7, 2025):

- Direct dependency on meshio@ git+https://github.com/matnoel/meshio.git cannot be included in PyPI.

## 1.4.1 (June 7, 2025):

- Updated project dependency to https://github.com/matnoel/meshio.git
- Fixed issue [#9](https://github.com/matnoel/EasyFEA/issues/9): add vizir output format
- Fixed issue [#10](https://github.com/matnoel/EasyFEA/issues/10): format the code with black
- Fixed bug in gauss quadrature for prisms
- Fixed issue [#11](https://github.com/matnoel/EasyFEA/issues/11): add continuous integration with github-actions
- Fixed issue [#14](https://github.com/matnoel/EasyFEA/issues/14): test types with mypy.
- Added new badges in the readme file.

## 1.4.0 (April 24, 2025):

- Fixed issues [#6](https://github.com/matnoel/EasyFEA/issues/6) and [#7](https://github.com/matnoel/EasyFEA/issues/7).
- Organized the `tests/` directory.
- Updated hyperbolic solvers (`hht`, `newmark`, `midpoint`).
- Created a linear algebra module for the `Trace`, `Det`, `Inv`, `TensorProd`, `Transpose`, and `Norm` functions.
- Updated the `MeshIO` interface. Removed unnecessary node reordering, which is now handled by the https://github.com/matnoel/meshio fork.
- Replaced `simu.Get_directions()` with `simu.Get_unknowns()`.
- Clarified the `groupElem.Get_F_e_pg()` function.
- Created `simu` functions to access Neumann boundary condition values.

## 1.3.4 (March 28, 2025):

- Moved `Display._Init_obj()` to `_simu._Init_obj()`.
- Updated the `Display.Plot_Result()` and `PyVista.Plot()` functions.
- Updated the `Modal1.py` and `Modal2.py` examples.
- Clarified the `_GroupElem.Get_F_e_pg()` and `_GroupElem.Get_invF_e_pg()` functions.

## 1.3.3 (March 22, 2025):

- Created the MeshIO interface.
- Updated the Geoms module.
- Created the params check functions.

## 1.3.2 (March 16, 2025):

- Updated pyproject.toml (name = "easyfea") to comply with PyPI distribution format specifications.
- Enhanced Gmsh_Interface to support linked surface creation by adding pointTags to the addSurfaceFilling function in Gmsh.

## 1.3.1 (February 28, 2025):

- Updated Folder functions (New_File -> Join(mkdir=True), Get_Path() -> Dir())
- Removed colors in Display.Plot_Tags()
- Updated the method for setting up a tag in a mesh (_Set_Nodes_Tag and _Set_Elements_Tag).
- Removed the old trick to generate the mesh with gmsh recombine
- Updated Gmsh_Interface tests (test_mesh_isOrganised).
- Enhanced examples.

## 1.3.0 (February 24, 2025):

- Implemented new element types: QUAD9, SEG5, TRI15, HEXA27, PRISM18.
- Enhanced Gmsh_Interface for QUAD and HEXA elements.
- Standardized shape functions.
- Updated Paraview_Interface and PyVista_Interface.
- Updated Gauss points quadrature.
- Migrated from unittest to pytest.
- Enhanced examples.

## 1.2.7 (February 7, 2025):

- Enhanced Gmsh_Interface.
- Finite element shape functions were renamed to improve code readability.
- NEW YEAR.

## 1.2.6 (November 7, 2024):

- Enhanced the Digital Image Correlation (DIC) analysis module.
- Enhanced examples.
- Updated docstrings.
- Updated Display functions.

## 1.2.5 (September 27, 2024):

- Updated phase field solver.

## 1.2.4 (September 22, 2024):

- Updated docstrings.
- Updated Display functions.
- Updated PyVista_Interface functions.
- Added _Elas.Get_sqrt_C_S() function.
- Updated He split for heterogeneous material properties.
- Added Save_pickle() and Load_pickle() functions.
- Updated Gmsh_Interface for cracks.
- Updated Folder functions.

## 1.2.3 (September 11, 2024):

- Ensures compatibility with python 3.9 and 3.10.

## 1.2.2 (September 10, 2024):

- Updated docstrings.
- Updated simulations/_phasefield.py solver.
- Updated Bc Config.
- Updated tests.
- Updated _Additional_Points().
- Updated Mesh_Beams().

## 1.2.1 (August 17, 2024):

- Updated docstrings.

## 1.2.0 (August 14, 2024):

- Added the resetAll option in Set_Iter() to simplify the update process after iteration activation.
- Enhanced clarity in phase field functions, including both simulation and material aspects.
- Improved display options for geometric objects.
- Improved display functions.
- Provided clearer functionality in mesh and group element.
- Updated the interface with Gmsh.
- Improved phasefield examples.
- Enhanced the Digital Image Correlation (DIC) analysis module.

## 1.1.0 (June 29, 2024):

- Updated gmsh interface functions.
- Added controls when calling certain functions.
- Updated geometric objects to modify parameters after creation.
- Added a banner to the project.
- Added CT.py damage simulation.

## 1.0.3 (May 16, 2024):

- Minor adjustments to README.md

## 1.0.2 (May 14, 2024):

- Minor adjustments to object printing.
- Reorganization of save functions in the simulation recording and loading process.
- Modification of the PETSc interface for the new version.
- Implemented minor refinements in Solvers.py to ensure correct canonical values of matrix A in Ax=b equations, thus avoiding potential bugs when using PETSc or pypardiso.
- Renamed functions in fem/_gauss.py to improve clarity and consistency.
- Updated function names in Display.py to improve readability.
- Updated copyright information to reflect the latest changes.
- Added a contribution guide to facilitate community participation and collaboration.
- Creation of a citation file.

## 1.0.1 (April 13, 2024):

- Update library dependencies.

## 1.0.0 (April 13, 2024):

- First version.