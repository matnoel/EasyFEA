.. include:: links.rst

Installation
============

EasyFEA can be easily installed from `PyPI`_ using pip, compatible with Python versions 3.9 through 3.12:

.. code-block:: console

   pip install EasyFEA

You can also install EasyFEA with the `source code <GitHub_>`_ using the ``pip install .`` command in the downloaded or cloned EasyFEA *folder*.

Dependencies
------------

EasyFEA uses several libraries such as NumPy and Gmsh - as such, the following projects are required dependencies of EasyFEA:

+ `numpy <https://pypi.org/project/numpy/>`_ - Fundamental package for scientific computing with Python.
+ `gmsh <https://pypi.org/project/gmsh/>`_ (>= 4.12) - Three-dimensional finite element mesh generator.
+ `scipy <https://pypi.org/project/scipy/>`_ - Fundamental package for scientific computing in Python.
+ `matplotlib <https://pypi.org/project/matplotlib/>`_ - Plotting package.
+ `pyvista <https://pypi.org/project/pyvista/>`_ - Plotting package.
+ `numba <https://pypi.org/project/numba/>`_ - Compiling Python code using LLVM.
+ `pandas <https://pypi.org/project/pandas/>`_ (3.9 <= Python <= 3.12) - Powerful data structures for data analysis.
+ `imageio <https://pypi.org/project/imageio/>`_ and `imageio[ffmpeg] <https://pypi.org/project/imageio-ffmpeg/>`_ - Library for reading and writing a wide range of image, video, scientific, and volumetric data formats.
+ `meshio <https://pypi.org/project/meshio/>`_ - I/O for many mesh formats.

Optional Dependencies
---------------------

EasyFEA includes a few optional dependencies for reducing resolution time or for performing DIC:

+ `pypardiso <https://pypi.org/project/pypardiso/>`_ (Python > 3.8 & Intel oneAPI)  - Library for solving large systems of sparse linear equations.
+ `petsc <https://pypi.org/project/petsc/>`_ and `petsc4py <https://pypi.org/project/petsc4py/>`_ - Python bindings for PETSc.
+ `opencv-python <https://pypi.org/project/opencv-python/>`_ - Computer Vision package.