# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""
ThermoElasticity
================

A temperature field read by a behavior, after the `weak-coupling tour of J. Bleyer
<https://bleyerj.github.io/comet-fenicsx/tours/linear_problems/thermoelasticity_weak/thermoelasticity_weak.html>`_.

A slab clamped on both sides is heated on top. The steady temperature is solved first, then given
to the mechanics as an external variable of a behavior written here,

.. math::
    \Sig = \Crm : (\Eps - \alpha\,\Delta T\,\Irm)

first alone, then with the self-weight.
"""

# sphinx_gallery_thumbnail_number = -1

from typing import NamedTuple

from EasyFEA import ElemType, Models, PyVista, Simulations
from EasyFEA.Geoms import Point, Line, Contour
from EasyFEA.Models.InElastic import _Behavior, ONE
from EasyFEA.Utilities import _params

# ----------------------------------------------
# Configuration
# ----------------------------------------------

L, H = 5.0, 0.3
E, v = 50e3, 0.2
alpha = 1e-5
rho_g = 2400 * 9.81e-6
dT = 20.0  # on top

# ----------------------------------------------
# Mesh
# ----------------------------------------------
p0, p1, p2, p3 = Point(0, 0), Point(L, 0), Point(L, H), Point(0, H)
contour = Contour(
    [
        Line(p0, p1, L / 20),
        Line(p1, p2, H / 5),
        Line(p2, p3, L / 20),
        Line(p3, p0, H / 5),
    ]
)
mesh = contour.Mesh_2D([], ElemType.QUAD9, isOrganised=True)

nodesSides = mesh.Nodes_Conditions(lambda x, y, z: (x == 0) | (x == L))
nodesCold = mesh.Nodes_Conditions(lambda x, y, z: (x == 0) | (x == L) | (y == 0))
nodesHot = mesh.Nodes_Conditions(lambda x, y, z: (y == H) & (0 < x) & (x < L))


# ----------------------------------------------
# Thermal problem
# ----------------------------------------------
thermal = Simulations.Thermal(mesh, Models.Thermal(k=1.0))
thermal.add_dirichlet(nodesCold, [0], ["t"])
thermal.add_dirichlet(nodesHot, [dT], ["t"])
thermal.Solve()

# ----------------------------------------------
# Mechanical problem
# ----------------------------------------------


class ThermoElastic(_Behavior):
    """Linear thermoelasticity, ``T`` being the temperature change."""

    class Externals(NamedTuple):
        T: float

    alpha: float = _params.PositiveScalarParameter()
    """thermal expansion coefficient"""

    def __init__(
        self,
        elastic: Models.Elastic.Isotropic,
        alpha: float,
    ):
        super().__init__(elastic)
        self.alpha = alpha

    def Update(self, eps, z, dt, T):
        return self.Stress(eps, z, T=T), z

    def Stress(self, eps, z, T):
        return self.C @ (eps - self.alpha * T * ONE)


# plane strain
material = ThermoElastic(
    Models.Elastic.Isotropic(
        2,
        E=E,
        v=v,
        planeStress=False,
    ),
    alpha,
)
simu = Simulations.InElastic(mesh, material)
simu.Set_external(T=thermal.thermal)
simu.add_dirichlet(nodesSides, [0, 0], ["x", "y"])

simu.Solve()
simu.Save_Iter()

simu.add_volumeLoad(mesh.nodes, [-rho_g], ["y"])
simu.Solve()
simu.Save_Iter()

# ----------------------------------------------
# Results
# ----------------------------------------------
plotter = PyVista.Plot(simu, "T", nColors=11, verticalColorbar=False)
plotter.show()

for i in range(2):
    simu.Set_Iter(i)
    plotter = PyVista.Plot(
        simu,
        "uy",
        deformFactor=1000,
        nColors=11,
        verticalColorbar=False,
    )
    plotter.show()
