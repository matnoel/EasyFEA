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
from EasyFEA.Models.InElastic import (
    _Behavior,
    MaterialPoint,
    Maxwell,
    Norton,
    Plasticity,
)
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
        return self.Stress(eps, z), z

    def Stress(self, eps, z, **external):
        return self.C @ eps


@pytest.fixture(scope="module")
def mesh2D():
    return Domain(Point(0, 0), Point(L, H), H / 2).Mesh_2D([], ElemType.QUAD4)


@pytest.fixture(scope="module")
def meshFine():
    return Domain(Point(0, 0), Point(L, H), H / 4).Mesh_2D([], ElemType.QUAD4)


@pytest.fixture(scope="module")
def mesh3D():
    return Domain(Point(0, 0), Point(L, H), H).Mesh_Extrude(
        [], [0, 0, H], [1], ElemType.HEXA8
    )


def _pull(simu, mesh: Mesh, u: float = 1.0) -> None:
    """Clamped at x=0, pulled at x=L."""
    simu.Bc_Init()
    nodes0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    simu.add_dirichlet(nodes0, [0] * mesh.dim, simu.Get_unknowns())
    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x, y, z: x == L), [u], ["x"])
    simu.Solve()


def _rel(a, b) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _relax(mesh: Mesh, nstep: int, planeStress: bool = True):
    behavior = Maxwell(
        Isotropic(2, E=E, v=nu, planeStress=planeStress, thickness=H), G, TAU
    )
    simu = Simulations.InElastic(mesh, behavior)
    simu.dt = DT
    for _ in range(nstep):
        _pull(simu, mesh)
        simu.Save_Iter()
    return simu


@pytest.mark.parametrize("planeStress", [False, True])
def test_2d_matches_elastic(mesh2D: Mesh, planeStress: bool):
    simu = Simulations.InElastic(
        mesh2D, Linear(Isotropic(2, E=E, v=nu, planeStress=planeStress, thickness=H))
    )
    ref = Simulations.Elastic(
        mesh2D, Isotropic(2, E=E, v=nu, planeStress=planeStress, thickness=H)
    )
    _pull(simu, mesh2D)
    _pull(ref, mesh2D)

    assert _rel(simu.displacement, ref.displacement) < 1e-12
    assert _rel(simu.Result("Svm"), ref.Result("Svm")) < 1e-12


def test_a_heterogeneous_elastic_model_matches_elastic():
    """One group, so that E can be given per element."""
    mesh = Domain(Point(0, 0), Point(L, H), H / 2).Mesh_2D(
        [], ElemType.QUAD4, isOrganised=True
    )
    group = mesh.groupElem
    E_e = np.where(group.coord[group.connect].mean(1)[:, 0] < L / 2, E / 3, E)
    simu = Simulations.InElastic(mesh, Linear(Isotropic(2, E=E_e, v=nu, thickness=H)))
    ref = Simulations.Elastic(mesh, Isotropic(2, E=E_e, v=nu, thickness=H))
    _pull(simu, mesh)
    _pull(ref, mesh)

    assert _rel(simu.displacement, ref.displacement) < 1e-12
    assert _rel(simu.Result("Svm"), ref.Result("Svm")) < 1e-12


def test_3d_matches_elastic(mesh3D: Mesh):
    simu = Simulations.InElastic(mesh3D, Linear(ELASTIC))
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


@pytest.mark.parametrize("mesh", ["mesh2D", "mesh3D"])
def test_asking_for_a_result_leaves_the_stress_alone(mesh: str, request):
    """The shear components carry a sqrt(2) that reading them must not strip from the committed stress."""
    mesh = request.getfixturevalue(mesh)
    simu = Simulations.InElastic(
        mesh, Linear(Isotropic(mesh.dim, E=E, v=nu, thickness=H))
    )
    _pull(simu, mesh)
    simu.Save_Iter()
    first = simu.Result("Sxy", nodeValues=False)
    simu.Result("Svm", nodeValues=False)

    assert np.allclose(simu.Result("Sxy", nodeValues=False), first, rtol=1e-12)


def test_stepping_on_from_a_restored_iteration_reproduces_the_history(mesh2D: Mesh):
    simu = _relax(mesh2D, 3)
    last = simu.Result("Sxx", nodeValues=False)

    simu.Set_Iter(1)
    _pull(simu, mesh2D)
    simu.Save_Iter()

    assert np.allclose(simu.Result("Sxx", nodeValues=False), last, rtol=1e-12)


def test_no_internal_variable_saves_none(mesh2D: Mesh):
    """Plane strain, since plane stress keeps eps_zz."""
    simu = Simulations.InElastic(
        mesh2D, Linear(Isotropic(2, E=E, v=nu, planeStress=False, thickness=H))
    )
    _pull(simu, mesh2D)
    simu.Save_Iter()

    results = simu.Set_Iter(-1)
    assert results is not None
    assert not any(key.startswith("internal") for key in results)


def test_simulation_survives_save_and_load(mesh2D: Mesh, tmp_path):
    simu = _relax(mesh2D, 2)
    simu.Save(str(tmp_path))

    loaded = Simulations.Load_Simu(str(tmp_path))
    for s in (loaded, simu):
        _pull(s, mesh2D)
        s.Save_Iter()

    assert np.allclose(loaded.Result("Sxx"), simu.Result("Sxx"), rtol=1e-12)


SIGMA_Y, HM = 250.0, 2000.0


def _pull_bar(simu, mesh: Mesh, eps_xx: float, save: bool = True) -> None:
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
    if save:
        simu.Save_Iter()


def _plastic(elastic=ELASTIC) -> Plasticity:
    return Plasticity(elastic, VonMises(SIGMA_Y), LinearHardening(HM))


@pytest.mark.parametrize("save", [True, False])
def test_plastic_bar_matches_the_closed_form(mesh3D: Mesh, save: bool):
    """Without Save_Iter too: each Solve commits."""
    simu = Simulations.InElastic(mesh3D, _plastic())
    eps_target = 5 * SIGMA_Y / E
    for eps_xx in np.linspace(eps_target / 10, eps_target, 10):
        _pull_bar(simu, mesh3D, eps_xx, save)

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


# ----------------------------------------------
# Commit at Solve
# ----------------------------------------------


def test_a_second_solve_under_the_same_load_is_one_more_step(mesh3D: Mesh):
    """Rate-independent: nothing more flows. Viscous: one more dt of relaxation, as at a material point."""
    eps_xx = 5 * SIGMA_Y / E
    plastic = Simulations.InElastic(mesh3D, _plastic())
    _pull_bar(plastic, mesh3D, eps_xx, save=False)
    p = plastic.Result("p", nodeValues=False)
    _pull_bar(plastic, mesh3D, eps_xx, save=False)

    assert np.all(p > 0)
    assert np.allclose(plastic.Result("p", nodeValues=False), p, rtol=1e-12)

    eps_xx = 1e-3
    viscous = Simulations.InElastic(mesh3D, Maxwell(ELASTIC, 0.5, 1.0))
    viscous.dt = DT
    sxx = []
    for _ in range(2):
        _pull_bar(viscous, mesh3D, eps_xx, save=False)
        sxx.append(viscous.Result("Sxx", nodeValues=False))
    ref = MaterialPoint(Maxwell(ELASTIC, 0.5, 1.0)).Run(
        strain={"xx": np.full(2, eps_xx)}, dt=DT
    )

    assert np.all(sxx[1] < sxx[0])
    for n in range(2):
        assert np.allclose(sxx[n], ref["stress"][n, 0], rtol=1e-9)


def test_an_unconverged_solve_commits_nothing(mesh2D: Mesh):
    """On the clamped plate: the uniaxial bar converges in one iteration whatever the load."""
    simu = Simulations.InElastic(
        mesh2D, _plastic(Isotropic(2, E=E, v=nu, planeStress=False, thickness=H))
    )
    _pull(simu, mesh2D, 0.5)
    p = simu.Result("p", nodeValues=False)
    simu._Solver_Set_Newton_Raphson_Algorithm(1e-6, 1e-10, 1e-11, maxIter=2)

    with pytest.raises(AssertionError, match="did not converged"):
        _pull(simu, mesh2D, 2.0)
    assert p.max() > 0
    assert np.allclose(simu.Result("p", nodeValues=False), p, rtol=1e-12)


def test_internal_variables_do_not_follow_a_mesh_change(mesh2D: Mesh, meshFine: Mesh):
    simu = Simulations.InElastic(
        mesh2D, _plastic(Isotropic(2, E=E, v=nu, planeStress=False, thickness=H))
    )
    _pull(simu, mesh2D)
    simu.mesh = meshFine

    with pytest.raises(AssertionError, match="internal variables cannot follow"):
        _pull(simu, meshFine)


def test_every_assembly_integrates_from_the_last_committed_state(
    mesh2D: Mesh, monkeypatch
):
    """The internal state z handed to the behavior is the one the last Solve committed, never a Newton iterate's."""
    behavior = _plastic(Isotropic(2, E=E, v=nu, planeStress=False, thickness=H))
    simu = Simulations.InElastic(mesh2D, behavior)
    nGroups = len(mesh2D.Get_list_groupElem(2))

    calls: list[tuple[dict, dict]] = []
    Integrate = behavior.Integrate

    def Spy(eps, z, dt=0.0, **external):
        out = Integrate(eps, z, dt, **external)
        calls.append(
            (
                {k: np.array(v) for k, v in z.items()},
                {k: np.array(v) for k, v in out[2].items()},
            )
        )
        return out

    monkeypatch.setattr(behavior, "Integrate", Spy)

    # by element count, since the mesh has several groups
    committed: dict[int, dict] = {}
    for u in (0.2, 0.4, 0.6):
        calls.clear()
        _pull(simu, mesh2D, u)

        assert len(calls) > nGroups  # assemblies, then the commit
        for z, _ in calls:
            Ne = z["p"].shape[0]
            start = committed.get(Ne, {k: np.zeros_like(v) for k, v in z.items()})
            assert all(np.array_equal(z[k], start[k]) for k in z)
        committed = {new["p"].shape[0]: new for _, new in calls[-nGroups:]}

    # the steps were not vacuous
    assert max(new["p"].max() for new in committed.values()) > 0


def test_plane_stress_plastic_plate_matches_a_material_point(mesh2D: Mesh):
    """Statically determinate, so the plate is in uniaxial stress, sigma_zz held at zero by the eps_zz solve."""
    simu = Simulations.InElastic(
        mesh2D, _plastic(Isotropic(2, E=E, v=nu, planeStress=True, thickness=H))
    )
    path = np.linspace(1, 5, 5) * SIGMA_Y / E
    sxx, p = [], []
    for eps_xx in path:
        simu.Bc_Init()
        simu.add_dirichlet(mesh2D.Nodes_Conditions(lambda x, y, z: x == 0), [0], ["x"])
        simu.add_dirichlet(
            mesh2D.Nodes_Conditions(lambda x, y, z: (x == 0) & (y == 0)), [0], ["y"]
        )
        simu.add_dirichlet(
            mesh2D.Nodes_Conditions(lambda x, y, z: x == L), [eps_xx * L], ["x"]
        )
        simu.Solve()
        sxx.append(simu.Result("Sxx", nodeValues=False))
        p.append(simu.Result("p", nodeValues=False))
    ref = MaterialPoint(_plastic()).Run(strain={"xx": path})

    assert ref["p"][-1] > 0
    for n in range(path.size):
        assert np.allclose(sxx[n], ref["stress"][n, 0], rtol=1e-8)
        assert np.allclose(p[n], ref["p"][n], rtol=1e-8, atol=1e-14)
