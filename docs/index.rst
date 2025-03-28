.. include:: links.rst

EasyFEA documentation
=====================

.. image:: _static/EasyFEA_banner.jpg

Overview
--------

**EasyFEA** is a user-friendly Python library that simplifies finite element analysis. It is flexible and supports different types of simulation without requiring users to handle complex PDE formulations. The library currently supports **four** specific simulation types:

1. **ElasticSimu** (static and dynamic): See examples at `/examples/Elastic`, `/examples/Dynamic` and `/examples/Contact`.
2. **BeamSimu** (static Euler-Bernoulli):  See examples at `/examples/Beam`.
3. **ThermalSimu** (stationary and transient):  See examples at `/examples/Thermal`.
4. **PhaseFieldSimu:** (quasi-static phase field) See examples at `/examples/PhaseField`.

All examples are available `here <GitHubExamples_>`_.

For each simulation, users create a **mesh** and a **model**. Once the simulation has been set up, defining the boundary conditions, solving the problem and visualizing the results is straightforward.

Numerous examples of mesh creation are available in the `examples/Meshes` folder.

The simplest and quickest introduction is available in the :ref:`begin`.

License
-------

Copyright (C) 2021-2025 Université Gustave Eiffel.

EasyFEA is distributed under the terms of the `GNU General Public License v3.0 or later <GNU_>`_, see `LICENSE.txt <LICENSE_>`_ and `CREDITS.md <CREDITS_>`_ for more information.


Check out the :doc:`install` section for further information on installing EasyFEA.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   begin
   examples/index
   howto
   easyfea