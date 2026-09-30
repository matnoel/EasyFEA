# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Simulations.InElastic, driven by behaviors written on the contract. Skipped whole when jax is absent."""

import numpy as np
import pytest

from EasyFEA import ElemType, Mesh, Simulations
from EasyFEA.Geoms import Domain, Point
from EasyFEA.Models.Elastic._laws import Isotropic
from EasyFEA.Models.InElastic import _Behavior, Maxwell, Norton, Plasticity
from EasyFEA.Models.InElastic.IsotropicHardening import Linear as LinearHardening
from EasyFEA.Models.InElastic.Yield import VonMises

pytest.importorskip("jax")

L, H = 120.0, 13.0
E, nu = 210000.0, 0.3
ELASTIC = Isotropic(3, E=E, v=nu)
G, TAU, DT = [0.3, 0.2], [1.0, 10.0], 0.5


class Linear(_Behavior):
    def Update(
        self,
        eps,
        z,
        dt,
        **external,
    ):
        return ELASTIC.C @ eps, z


@pytest.fixture(scope="module")
def mesh2D():
    return Domain(Point(0, 0), Point(L, H), H / 2).Mesh_2D([], ElemType.QUAD4)


@pytest.fixture(scope="module")
def mesh3D():
    return Domain(Point(0, 0), Point(L, H), H).Mesh_Extrude(
        [], [0, 0, H], [1], ElemType.HEXA8
    )


def _pull(simu, mesh: Mesh) -> None:
    """Clamped at x=0, pulled at x=L."""
    simu.Bc_Init()
    nodes0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    simu.add_dirichlet(nodes0, [0] * mesh.dim, simu.Get_unknowns())
    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x, y, z: x == L), [1.0], ["x"])
    simu.Solve()


def _rel(a, b) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _relax(mesh: Mesh, nstep: int, planeStress: bool = True):
    behavior = Maxwell(ELASTIC, G, TAU, dim=2, planeStress=planeStress, thickness=H)
    simu = Simulations.InElastic(mesh, behavior)
    simu.dt = DT
    for _ in range(nstep):
        _pull(simu, mesh)
        simu.Save_Iter()
    return simu


@pytest.mark.parametrize("planeStress", [False, True])
def test_2d_matches_elastic(mesh2D: Mesh, planeStress: bool):
    simu = Simulations.InElastic(
        mesh2D, Linear(dim=2, planeStress=planeStress, thickness=H)
    )
    ref = Simulations.Elastic(
        mesh2D, Isotropic(2, E=E, v=nu, planeStress=planeStress, thickness=H)
    )
    _pull(simu, mesh2D)
    _pull(ref, mesh2D)

    assert _rel(simu.displacement, ref.displacement) < 1e-12
    assert _rel(simu.Result("Svm"), ref.Result("Svm")) < 1e-12


def test_3d_matches_elastic(mesh3D: Mesh):
    simu = Simulations.InElastic(mesh3D, Linear())
    ref = Simulations.Elastic(mesh3D, ELASTIC)
    _pull(simu, mesh3D)
    _pull(ref, mesh3D)

    assert _rel(simu.displacement, ref.displacement) < 1e-12


@pytest.mark.parametrize("planeStress", [False, True])
def test_maxwell_relaxes_the_whole_field_by_one_scalar(mesh2D: Mesh, planeStress):
    """Every branch carries a fraction of the same C, so the field scales by R(n) at held displacement."""
    nstep = 4
    simu = _relax(mesh2D, nstep, planeStress)
    g, tau = np.array(G), np.array(TAU)

    def R(n):
        return 1 - g.sum() + g @ (1 + DT / tau) ** -n

    sxx = [simu.Result("Sxx", nodeValues=False, iter=i) for i in range(nstep)]
    for n in range(1, nstep):
        assert np.allclose(sxx[n], R(n + 1) / R(1) * sxx[0], rtol=1e-10, atol=1e-8)


def test_the_stress_is_the_same_before_and_after_save_iter(mesh2D: Mesh):
    simu = _relax(mesh2D, 1)
    _pull(simu, mesh2D)
    trial = simu.Result("Sxx", nodeValues=False)
    simu.Save_Iter()

    assert np.allclose(simu.Result("Sxx", nodeValues=False), trial, rtol=1e-12)
    # asking again does not step the material on
    assert np.allclose(simu.Result("Sxx", nodeValues=False), trial, rtol=1e-12)


def test_state_is_committed_at_save_iter(mesh2D: Mesh):
    """Stepping on from a restored iteration reproduces the history."""
    simu = _relax(mesh2D, 3)
    last = simu.Result("Sxx", nodeValues=False)

    simu.Set_Iter(1)
    _pull(simu, mesh2D)
    simu.Save_Iter()

    assert np.allclose(simu.Result("Sxx", nodeValues=False), last, rtol=1e-12)


def test_simulation_survives_save_and_load(mesh2D: Mesh, tmp_path):
    simu = _relax(mesh2D, 2)
    simu.Save(str(tmp_path))

    loaded = Simulations.Load_Simu(str(tmp_path))
    for s in (loaded, simu):
        _pull(s, mesh2D)
        s.Save_Iter()

    assert np.allclose(loaded.Result("Sxx"), simu.Result("Sxx"), rtol=1e-12)


SIGMA_Y, HM = 250.0, 2000.0


def _pull_bar(simu, mesh: Mesh, eps_xx: float) -> None:
    """Statically determinate: x fixed on the face, y and z pinned on one edge each, so the stress stays uniaxial."""
    simu.Bc_Init()
    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x, y, z: x == 0), [0], ["x"])
    simu.add_dirichlet(
        mesh.Nodes_Conditions(lambda x, y, z: (x == 0) & (y == 0)), [0], ["y"]
    )
    simu.add_dirichlet(
        mesh.Nodes_Conditions(lambda x, y, z: (x == 0) & (z == 0)), [0], ["z"]
    )
    simu.add_dirichlet(
        mesh.Nodes_Conditions(lambda x, y, z: x == L), [eps_xx * L], ["x"]
    )
    simu.Solve()
    simu.Save_Iter()


def test_plastic_bar_matches_the_closed_form(mesh3D: Mesh):
    simu = Simulations.InElastic(
        mesh3D, Plasticity(ELASTIC, VonMises(SIGMA_Y), LinearHardening(HM))
    )
    eps_target = 5 * SIGMA_Y / E
    for eps_xx in np.linspace(eps_target / 10, eps_target, 10):
        _pull_bar(simu, mesh3D, eps_xx)

    expected = E * (SIGMA_Y + HM * eps_target) / (E + HM)
    assert np.allclose(simu.Result("Sxx", nodeValues=False), expected, rtol=1e-9)
    assert np.all(simu.Result("p", nodeValues=False) > 0)


def test_norton_relaxes_through_the_simulation(mesh3D: Mesh):
    """simu.dt reaches the material: held displacement, falling stress."""
    behavior = Norton(
        ELASTIC, VonMises(SIGMA_Y), LinearHardening(HM), A=1e-2, n=1.0, sigma_0=SIGMA_Y
    )
    simu = Simulations.InElastic(mesh3D, behavior)
    simu.dt = 1.0
    history = []
    for _ in range(6):
        _pull_bar(simu, mesh3D, 5 * SIGMA_Y / E)
        history.append(float(np.mean(simu.Result("Sxx", nodeValues=False))))

    assert history[-1] < history[0]
    assert np.all(np.diff(history) <= 1e-9)
