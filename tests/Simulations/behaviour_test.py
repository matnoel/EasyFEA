# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""A Behaviour with no internal variables must reproduce Simulations.Elastic exactly."""

import numpy as np
import pytest

from EasyFEA import ElemType, Models, Simulations
from EasyFEA.Geoms import Domain, Point
from EasyFEA.Models.Elastic._laws import (
    Anisotropic,
    Isotropic,
    Orthotropic,
    TransverselyIsotropic,
)

L, H = 120.0, 13.0
E, nu = 210000.0, 0.3


def _laws_3d() -> dict:
    isot = Isotropic(3, E=E, v=nu)
    return {
        "Isotropic": isot,
        "TransverselyIsotropic": TransverselyIsotropic(
            3, El=E, Et=E / 2, Gl=E / 3, vl=0.3, vt=0.2
        ),
        "Orthotropic": Orthotropic(
            3,
            E1=E,
            E2=E / 2,
            E3=E / 3,
            G12=E / 4,
            G13=E / 5,
            G23=E / 6,
            v12=0.3,
            v13=0.2,
            v23=0.1,
        ),
        "Anisotropic": Anisotropic(3, isot.C, useVoigtNotation=False),
    }


def _solve(mesh, model, dim: int) -> np.ndarray:
    """Same BCs either way: clamped at x=0, pulled at x=L."""
    if isinstance(model, Models.Behaviour):
        simu = Simulations.Behaviour(mesh, model)
    else:
        simu = Simulations.Elastic(mesh, model)

    nodes0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesL = mesh.Nodes_Conditions(lambda x, y, z: x == L)
    simu.add_dirichlet(nodes0, [0] * dim, simu.Get_unknowns())
    simu.add_dirichlet(nodesL, [1.0], ["x"])
    simu.Solve()

    return simu.displacement


@pytest.fixture(scope="module")
def mesh2D():
    return Domain(Point(0, 0), Point(L, H), H / 2).Mesh_2D([], ElemType.QUAD4)


@pytest.fixture(scope="module")
def mesh3D():
    return Domain(Point(0, 0), Point(L, H), H).Mesh_Extrude(
        [], [0, 0, H], [1], ElemType.HEXA8
    )


@pytest.mark.parametrize("law", list(_laws_3d()))
def test_3d_matches_elastic(mesh3D, law: str):
    """3D, every shipped elastic law."""
    elastic = _laws_3d()[law]

    u_behaviour = _solve(mesh3D, Models.Behaviour(3, elastic), 3)
    u_elastic = _solve(mesh3D, elastic, 3)

    assert np.linalg.norm(u_behaviour - u_elastic) / np.linalg.norm(u_elastic) < 1e-12


@pytest.mark.parametrize("planeStress", [False, True])
def test_2d_matches_elastic(mesh2D, planeStress: bool):
    """2D plane strain and plane stress, against the elastic law's own 2D form."""
    behaviour = Models.Behaviour(
        2, Isotropic(3, E=E, v=nu), thickness=H, planeStress=planeStress
    )
    elastic2d = Isotropic(2, E=E, v=nu, planeStress=planeStress, thickness=H)

    u_behaviour = _solve(mesh2D, behaviour, 2)
    u_elastic = _solve(mesh2D, elastic2d, 2)

    assert np.linalg.norm(u_behaviour - u_elastic) / np.linalg.norm(u_elastic) < 1e-12


def test_stored_energy_matches_elastic(mesh2D):
    """The free energy integrated over the domain equals Simulations.Elastic's Wdef."""
    behaviour = Models.Behaviour(2, Isotropic(3, E=E, v=nu), thickness=H)
    elastic2d = Isotropic(2, E=E, v=nu, planeStress=False, thickness=H)

    simu = Simulations.Behaviour(mesh2D, behaviour)
    nodes0 = mesh2D.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesL = mesh2D.Nodes_Conditions(lambda x, y, z: x == L)
    simu.add_dirichlet(nodes0, [0, 0], ["x", "y"])
    simu.add_dirichlet(nodesL, [1.0], ["x"])
    simu.Solve()

    simuRef = Simulations.Elastic(mesh2D, elastic2d)
    simuRef.add_dirichlet(nodes0, [0, 0], ["x", "y"])
    simuRef.add_dirichlet(nodesL, [1.0], ["x"])
    simuRef.Solve()

    psi = simu.Results_dict_Energy()[r"$\Psi$"]
    assert np.isclose(psi, simuRef.Result("Wdef"), rtol=1e-10)


def test_plastic_bar_matches_the_closed_form(mesh3D):
    """A bar pulled past yield: uniform uniaxial stress, so the closed form applies everywhere.

    Proves the global Newton drives a flowing material, not just that the local solve is right.
    """
    sigma_y, Hm = 250.0, 2000.0
    behaviour = Models.Behaviour(
        3,
        Isotropic(3, E=E, v=nu),
        hardening=Models.IsotropicHardening.Linear(Hm),
        yieldSurface=Models.Yield.VonMises(sigma_y),
    )
    simu = Simulations.Behaviour(mesh3D, behaviour)

    nodes0 = mesh3D.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesL = mesh3D.Nodes_Conditions(lambda x, y, z: x == L)
    eps_target = 5 * sigma_y / E

    # statically determinate: x fixed on the whole face, y and z pinned on one edge each,
    # so the bar contracts freely and the stress state stays uniaxial
    nodesY = mesh3D.Nodes_Conditions(lambda x, y, z: (x == 0) & (y == 0))
    nodesZ = mesh3D.Nodes_Conditions(lambda x, y, z: (x == 0) & (z == 0))

    for eps_xx in np.linspace(eps_target / 10, eps_target, 10):
        simu.Bc_Init()
        simu.add_dirichlet(nodes0, [0], ["x"])
        simu.add_dirichlet(nodesY, [0], ["y"])
        simu.add_dirichlet(nodesZ, [0], ["z"])
        simu.add_dirichlet(nodesL, [eps_xx * L], ["x"])
        simu.Solve()
        simu.Save_Iter()

    sxx = simu.Result("Sxx", nodeValues=False)
    expected = E * (sigma_y + Hm * eps_target) / (E + Hm)

    assert np.allclose(sxx, expected, rtol=1e-6)
    assert expected > sigma_y  # the bar really did yield


def test_relaxation_through_the_simulation(mesh3D):
    """Hold the displacement and step time: the stress relaxes, so simu.dt reaches the material."""
    sigma_y = 250.0
    behaviour = Models.Behaviour(
        3,
        Isotropic(3, E=E, v=nu),
        hardening=Models.IsotropicHardening.Linear(2000.0),
        yieldSurface=Models.Yield.VonMises(sigma_y),
        rate=Models.ViscoPlastic.Norton(1e-2, 1.0, sigma_y),
    )
    simu = Simulations.Behaviour(mesh3D, behaviour)
    simu.dt = 1.0

    nodes0 = mesh3D.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesL = mesh3D.Nodes_Conditions(lambda x, y, z: x == L)
    nodesY = mesh3D.Nodes_Conditions(lambda x, y, z: (x == 0) & (y == 0))
    nodesZ = mesh3D.Nodes_Conditions(lambda x, y, z: (x == 0) & (z == 0))

    history = []
    for _ in range(6):
        simu.Bc_Init()
        simu.add_dirichlet(nodes0, [0], ["x"])
        simu.add_dirichlet(nodesY, [0], ["y"])
        simu.add_dirichlet(nodesZ, [0], ["z"])
        simu.add_dirichlet(nodesL, [5 * sigma_y / E * L], ["x"])
        simu.Solve()
        simu.Save_Iter()
        history.append(float(np.mean(simu.Result("Sxx", nodeValues=False))))

    assert history[-1] < history[0]
    assert np.all(np.diff(history) <= 1e-9)


def test_state_stays_empty_without_internal_variables(mesh2D):
    """No yield surface means nothing to store, and the solve is one Newton iteration."""
    behaviour = Models.Behaviour(2, Isotropic(3, E=E, v=nu), thickness=H)
    simu = Simulations.Behaviour(mesh2D, behaviour)

    nodes0 = mesh2D.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesL = mesh2D.Nodes_Conditions(lambda x, y, z: x == L)
    simu.add_dirichlet(nodes0, [0, 0], ["x", "y"])
    simu.add_dirichlet(nodesL, [1.0], ["x"])
    simu.Solve()
    simu.Save_Iter()

    assert behaviour.layout.n == 0
    assert simu.Set_Iter(-1) is not None
