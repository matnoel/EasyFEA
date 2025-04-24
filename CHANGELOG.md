# Changelog

This document describes the changes made to the project.

## 1.4.0 (April 24, 2025):

- Fixed issues #6 and #7.
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