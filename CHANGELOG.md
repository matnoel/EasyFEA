# Changelog

This document describes the changes made to the project.

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