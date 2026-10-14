# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Simulations.InElastic, driven by behaviors written on the contract. Skipped whole when jax is absent."""

from typing import NamedTuple

import numpy as np
import pytest

from EasyFEA import ElemType, Mesh, Models, Simulations
from EasyFEA.FEM import MatrixType
from EasyFEA.Geoms import Domain
from EasyFEA.Models.Elastic import Isotropic
from EasyFEA.Models.InElastic import (
    ONE,
    _Behavior,
    MaterialPoint,
    Maxwell,
    Norton,
    Plasticity,
)
from EasyFEA.Models.InElastic.IsotropicHardening import Linear as LinearHardening
from EasyFEA.Models.InElastic.Yield import VonMises

from _numerics import rel_error
from . import _reference
from ._fields import bar_mesh, component

pytest.importorskip("jax")

L, H = 120.0, 13.0
E, nu = 210000.0, 0.3
ELASTIC = Isotropic(3, E=E, v=nu)
G, TAU, DT = [0.3, 0.2], [1.0, 10.0], 0.5
ALPHA = 1e-5


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
    """4 QUAD4 in a row."""
    domain = Domain((0, 0), (L, H), L / 4)
    return domain.Mesh_2D([], ElemType.QUAD4)


@pytest.fixture(scope="module")
def meshFine():
    domain = Domain((0, 0), (L, H), L / 8)
    return domain.Mesh_2D([], ElemType.QUAD4)


@pytest.fixture(scope="module")
def mesh3D():
    """4 HEXA8 in a row."""
    domain = Domain((0, 0), (L, H), L / 4)
    return domain.Mesh_Extrude([], [0, 0, H], [1], ElemType.HEXA8)


def _pull(simu, mesh: Mesh, u: float = 1.0) -> None:
    """Clamped at x=0, pulled at x=L."""
    simu.Bc_Init()
    nodesClamped = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesPulled = mesh.Nodes_Conditions(lambda x, y, z: x == L)
    simu.add_dirichlet(nodesClamped, [0] * mesh.dim, simu.Get_unknowns())
    simu.add_dirichlet(nodesPulled, [u], ["x"])
    simu.Solve()


def _relax(mesh: Mesh, nstep: int, planeStress: bool = True):
    behavior = Maxwell(
        Isotropic(
            2,
            E=E,
            v=nu,
            planeStress=planeStress,
            thickness=H,
        ),
        G,
        TAU,
    )
    simu = Simulations.InElastic(mesh, behavior)
    simu.dt = DT
    for _ in range(nstep):
        _pull(simu, mesh)
        simu.Save_Iter()
    return simu


def test_InElastic_linear_behavior_matches_Elastic_in_2D(subtests, mesh2D: Mesh):
    for planeStress in (False, True):
        with subtests.test(planeStress=planeStress):
            elastic = Isotropic(
                2,
                E=E,
                v=nu,
                planeStress=planeStress,
                thickness=H,
            )
            simu = Simulations.InElastic(mesh2D, Linear(elastic))
            ref = Simulations.Elastic(mesh2D, elastic)
            _pull(simu, mesh2D)
            _pull(ref, mesh2D)

            assert rel_error(simu.displacement, ref.displacement) < 1e-12
            assert rel_error(simu.Result("Svm"), ref.Result("Svm")) < 1e-12


def test_InElastic_heterogeneous_linear_behavior_matches_Elastic():
    """One group, so that E can be given per element."""
    domain = Domain((0, 0), (L, H), L / 4)
    mesh = domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    group = mesh.groupElem
    E_e = np.where(group.coord[group.connect].mean(1)[:, 0] < L / 2, E / 3, E)
    simu = Simulations.InElastic(mesh, Linear(Isotropic(2, E=E_e, v=nu, thickness=H)))
    ref = Simulations.Elastic(mesh, Isotropic(2, E=E_e, v=nu, thickness=H))
    _pull(simu, mesh)
    _pull(ref, mesh)

    assert rel_error(simu.displacement, ref.displacement) < 1e-12
    assert rel_error(simu.Result("Svm"), ref.Result("Svm")) < 1e-12


def test_InElastic_linear_behavior_matches_Elastic_in_3D(mesh3D: Mesh):
    simu = Simulations.InElastic(mesh3D, Linear(ELASTIC))
    ref = Simulations.Elastic(mesh3D, ELASTIC)
    _pull(simu, mesh3D)
    _pull(ref, mesh3D)

    assert rel_error(simu.displacement, ref.displacement) < 1e-12


def test_InElastic_dt_relaxes_maxwell_by_one_scalar(subtests, mesh2D: Mesh):
    """Every branch carries a fraction of the same C, so the field scales by R(n) at held displacement."""
    nstep = 4
    g, tau = np.array(G), np.array(TAU)

    def R(n):
        return 1 - g.sum() + g @ (1 + DT / tau) ** -n

    for planeStress in (False, True):
        with subtests.test(planeStress=planeStress):
            simu = _relax(mesh2D, nstep, planeStress)

            sxx = [simu.Result("Sxx", nodeValues=False, iter=i) for i in range(nstep)]
            for n in range(1, nstep):
                expected = R(n + 1) / R(1) * sxx[0]
                assert np.allclose(sxx[n], expected, rtol=1e-10, atol=1e-8)


def test_InElastic_stress_is_the_same_before_and_after_Save_Iter(mesh2D: Mesh):
    simu = _relax(mesh2D, 1)
    _pull(simu, mesh2D)
    trial = simu.Result("Sxx", nodeValues=False)
    simu.Save_Iter()

    assert np.allclose(simu.Result("Sxx", nodeValues=False), trial, rtol=1e-12)
    # asking again does not step the material on
    assert np.allclose(simu.Result("Sxx", nodeValues=False), trial, rtol=1e-12)


def test_InElastic_reading_a_result_leaves_the_stress_alone(
    subtests, mesh2D: Mesh, mesh3D: Mesh
):
    """The shear components carry a sqrt(2) that reading them must not strip from the committed stress."""
    for mesh in (mesh2D, mesh3D):
        with subtests.test(dim=mesh.dim):
            behavior = Linear(Isotropic(mesh.dim, E=E, v=nu, thickness=H))
            simu = Simulations.InElastic(mesh, behavior)
            _pull(simu, mesh)
            simu.Save_Iter()
            first = simu.Result("Sxy", nodeValues=False)
            simu.Result("Svm", nodeValues=False)

            assert np.allclose(simu.Result("Sxy", nodeValues=False), first, rtol=1e-12)


def test_InElastic_stepping_on_from_a_restored_iteration_reproduces_the_history(
    mesh2D: Mesh,
):
    simu = _relax(mesh2D, 3)
    last = simu.Result("Sxx", nodeValues=False)

    simu.Set_Iter(1)
    _pull(simu, mesh2D)
    simu.Save_Iter()

    assert np.allclose(simu.Result("Sxx", nodeValues=False), last, rtol=1e-12)


def test_InElastic_without_internal_variables_saves_none(mesh2D: Mesh):
    """Plane strain, since plane stress keeps eps_zz."""
    simu = Simulations.InElastic(
        mesh2D,
        Linear(
            Isotropic(
                2,
                E=E,
                v=nu,
                planeStress=False,
                thickness=H,
            )
        ),
    )
    _pull(simu, mesh2D)
    simu.Save_Iter()

    results = simu.Set_Iter(-1)
    assert results is not None
    assert not any(key.startswith("internal") for key in results)


def test_InElastic_survives_Save_and_Load_Simu(mesh2D: Mesh, tmp_path):
    simu = _relax(mesh2D, 2)
    simu.Save(str(tmp_path))

    loaded = Simulations.Load_Simu(str(tmp_path))
    for s in (loaded, simu):
        _pull(s, mesh2D)
        s.Save_Iter()

    assert np.allclose(loaded.Result("Sxx"), simu.Result("Sxx"), rtol=1e-12)


SIGMA_Y, HM = 250.0, 2000.0


def _pull_bar(simu, eps_xx: float, save: bool = True) -> None:
    """On symmetry rollers, so the stress stays uniaxial."""
    simu.Bc_Init()
    _reference.pull_bar(simu, eps_xx * L)
    simu.Solve()
    if save:
        simu.Save_Iter()


def _plastic(elastic=ELASTIC) -> Plasticity:
    return Plasticity(elastic, VonMises(SIGMA_Y), LinearHardening(HM))


def test_InElastic_plastic_bar_matches_the_closed_form(subtests, mesh3D: Mesh):
    """Without Save_Iter too: each Solve commits."""
    eps_target = 5 * SIGMA_Y / E
    expected = E * (SIGMA_Y + HM * eps_target) / (E + HM)
    for save in (True, False):
        with subtests.test(save=save):
            simu = Simulations.InElastic(mesh3D, _plastic())
            for eps_xx in np.linspace(eps_target / 10, eps_target, 10):
                _pull_bar(simu, eps_xx, save)

            Sxx = simu.Result("Sxx", nodeValues=False)
            assert np.allclose(Sxx, expected, rtol=1e-9)
            assert np.all(simu.Result("p", nodeValues=False) > 0)


def test_InElastic_dt_relaxes_norton(mesh3D: Mesh):
    """simu.dt reaches the material: held displacement, falling stress."""
    behavior = Norton(
        ELASTIC, VonMises(SIGMA_Y), LinearHardening(HM), A=1e-2, n=1.0, sigma_0=SIGMA_Y
    )
    simu = Simulations.InElastic(mesh3D, behavior)
    simu.dt = 1.0
    history = []
    for _ in range(6):
        _pull_bar(simu, 5 * SIGMA_Y / E)
        history.append(float(np.mean(simu.Result("Sxx", nodeValues=False))))

    assert history[-1] < history[0]
    assert np.all(np.diff(history) <= 1e-9)


# ----------------------------------------------
# Commit at Solve
# ----------------------------------------------


def test_InElastic_a_second_solve_under_the_same_load_is_one_more_step(mesh3D: Mesh):
    """Rate-independent: nothing more flows. Viscous: one more dt of relaxation, as at a material point."""
    eps_xx = 5 * SIGMA_Y / E
    plastic = Simulations.InElastic(mesh3D, _plastic())
    _pull_bar(plastic, eps_xx, save=False)
    p = plastic.Result("p", nodeValues=False)
    _pull_bar(plastic, eps_xx, save=False)

    assert np.all(p > 0)
    assert np.allclose(plastic.Result("p", nodeValues=False), p, rtol=1e-12)

    eps_xx = 1e-3
    viscous = Simulations.InElastic(mesh3D, Maxwell(ELASTIC, 0.5, 1.0))
    viscous.dt = DT
    sxx = []
    for _ in range(2):
        _pull_bar(viscous, eps_xx, save=False)
        sxx.append(viscous.Result("Sxx", nodeValues=False))
    ref = MaterialPoint(Maxwell(ELASTIC, 0.5, 1.0)).Run(
        strain={"xx": np.full(2, eps_xx)}, dt=DT
    )

    assert np.all(sxx[1] < sxx[0])
    for n in range(2):
        assert np.allclose(sxx[n], ref["stress"][n, 0], rtol=1e-9)


def test_InElastic_an_unconverged_solve_commits_nothing(mesh2D: Mesh):
    """On the clamped plate: the uniaxial bar converges in one iteration whatever the load."""
    simu = Simulations.InElastic(
        mesh2D,
        _plastic(
            Isotropic(
                2,
                E=E,
                v=nu,
                planeStress=False,
                thickness=H,
            )
        ),
    )
    _pull(simu, mesh2D, 0.5)
    p = simu.Result("p", nodeValues=False)
    simu._Solver_Set_Newton_Raphson_Algorithm(1e-6, 1e-10, 1e-11, maxIter=2)

    with pytest.raises(AssertionError, match="did not converged"):
        _pull(simu, mesh2D, 2.0)
    assert p.max() > 0
    assert np.allclose(simu.Result("p", nodeValues=False), p, rtol=1e-12)


NEW_MESHES = {
    "finer": (L / 8, ElemType.QUAD4),
    "other element type": (L / 4, ElemType.TRI3),
    "same mesh rebuilt": (L / 4, ElemType.QUAD4),
}


def test_InElastic_internal_variables_do_not_follow_a_mesh_change(
    subtests, mesh2D: Mesh
):
    elastic = Isotropic(
        2,
        E=E,
        v=nu,
        planeStress=False,
        thickness=H,
    )
    for name, (size, elemType) in NEW_MESHES.items():
        with subtests.test(newMesh=name):
            simu = Simulations.InElastic(mesh2D, _plastic(elastic))
            _pull(simu, mesh2D)
            domain = Domain((0, 0), (L, H), size)
            newMesh = domain.Mesh_2D([], elemType)
            simu.mesh = newMesh

            with pytest.raises(
                AssertionError, match="internal variables cannot follow"
            ):
                _pull(simu, newMesh)


def test_InElastic_internal_variables_follow_the_same_mesh(mesh2D: Mesh):
    simu = Simulations.InElastic(
        mesh2D,
        _plastic(
            Isotropic(
                2,
                E=E,
                v=nu,
                planeStress=False,
                thickness=H,
            )
        ),
    )
    _pull(simu, mesh2D, 0.5)
    p = simu.Result("p", nodeValues=False)
    simu.mesh = mesh2D
    _pull(simu, mesh2D, 0.5)

    assert p.max() > 0
    assert np.allclose(simu.Result("p", nodeValues=False), p, rtol=1e-12)


def test_InElastic_every_assembly_integrates_from_the_last_committed_state(
    mesh2D: Mesh, monkeypatch
):
    """The internal state z handed to the behavior is the one the last Solve committed, never a Newton iterate's."""
    behavior = _plastic(
        Isotropic(
            2,
            E=E,
            v=nu,
            planeStress=False,
            thickness=H,
        )
    )
    simu = Simulations.InElastic(mesh2D, behavior)
    nGroups = len(mesh2D.Get_list_groupElem(2))

    calls: list[tuple[dict, dict]] = []
    Integrate = behavior.Integrate

    def Spy(kinematics, z, dt=0.0, **external):
        out = Integrate(kinematics, z, dt, **external)
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


# ----------------------------------------------
# External variables
# ----------------------------------------------


class ThermoElastic(_Behavior):
    """Reads the temperature change T."""

    class Externals(NamedTuple):
        T: float

    def Update(self, eps, z, dt, T):
        return self.Stress(eps, z, T=T), z

    def Stress(self, eps, z, T):
        return self.C @ (eps - ALPHA * T * ONE)


def _thermo(mesh: Mesh, planeStress: bool = False):
    return Simulations.InElastic(
        mesh,
        ThermoElastic(
            Isotropic(
                2,
                E=E,
                v=nu,
                planeStress=planeStress,
                thickness=H,
            )
        ),
    )


def _heat(simu) -> None:
    """Free to expand on symmetry rollers."""
    simu.Bc_Init()
    _reference.add_bar_supports(simu)
    simu.Solve()


def test_InElastic_Set_external_refuses_what_the_behavior_does_not_read(mesh2D: Mesh):
    simu = _thermo(mesh2D)
    with pytest.raises(AssertionError, match="reads no 'P'"):
        simu.Set_external(P=1.0)
    with pytest.raises(AssertionError, match="scalar or a"):
        simu.Set_external(T=np.ones(mesh2D.Nn + 1))


def test_InElastic_Set_external_scalar_is_a_uniform_nodal_field(mesh2D: Mesh):
    scalar, nodal = _thermo(mesh2D), _thermo(mesh2D)
    scalar.Set_external(T=20.0)
    nodal.Set_external(T=np.full(mesh2D.Nn, 20.0))
    _heat(scalar)
    _heat(nodal)

    assert np.allclose(scalar.displacement, nodal.displacement, rtol=1e-12)
    assert np.allclose(scalar.Result("T"), 20.0)


def test_InElastic_uniform_heating_expands_freely(subtests, mesh2D: Mesh):
    """No in-plane stress; eps = alpha T in plane stress, (1 + nu) alpha T in plane strain, where eps_zz = 0."""
    T = 20.0
    for planeStress in (False, True):
        with subtests.test(planeStress=planeStress):
            simu = _thermo(mesh2D, planeStress)
            simu.Set_external(T=T)
            _heat(simu)

            eps = ALPHA * T * (1 if planeStress else 1 + nu)
            assert np.isclose(simu.Result("ux").max(), eps * L, rtol=1e-10)
            assert np.isclose(simu.Result("uy").max(), eps * H, rtol=1e-10)
            for result in ("Sxx", "Syy", "Sxy"):
                stress = simu.Result(result, nodeValues=False)
                assert np.allclose(stress, 0.0, atol=1e-8)


def test_InElastic_external_is_for_the_next_solve_results_for_the_last(mesh2D: Mesh):
    """Clamped, so that heating stresses it."""
    simu = _thermo(mesh2D, planeStress=True)
    simu.Set_external(T=10.0)
    _pull(simu, mesh2D, 0.0)
    sxx = simu.Result("Sxx", nodeValues=False)

    simu.Set_external(T=20.0)
    assert np.allclose(simu.Result("Sxx", nodeValues=False), sxx, rtol=1e-12)
    assert np.allclose(simu.Result("T"), 10.0)
    assert np.allclose(simu.external["T"], 20.0)
    simu.Save_Iter()
    assert np.allclose(simu.Set_Iter(0)["external/T"], 10.0)


def test_InElastic_Save_Iter_needs_the_external_variables(mesh2D: Mesh):
    with pytest.raises(AssertionError, match="before saving"):
        _thermo(mesh2D).Save_Iter()


def test_InElastic_external_variables_do_not_follow_a_mesh_change(
    mesh2D: Mesh, meshFine: Mesh
):
    simu = _thermo(mesh2D)
    simu.Set_external(T=np.full(mesh2D.Nn, 20.0))
    simu.mesh = meshFine

    with pytest.raises(AssertionError, match="mesh change"):
        _heat(simu)


def test_InElastic_external_variables_are_saved_with_each_iteration(
    mesh2D: Mesh, tmp_path
):
    simu = _thermo(mesh2D)
    for T in (10.0, 20.0):
        simu.Set_external(T=T)
        _heat(simu)
        simu.Save_Iter()
    simu.Save(str(tmp_path))
    loaded = Simulations.Load_Simu(str(tmp_path))

    for s in (simu, loaded):
        for i, T in enumerate((10.0, 20.0)):
            s.Set_Iter(i)
            assert np.allclose(s.external["T"], T)
            assert np.isclose(s.Result("ux").max(), (1 + nu) * ALPHA * T * L)


# ----------------------------------------------
# Members
# ----------------------------------------------


def test_InElastic_rejects_a_model_that_is_not_a_behavior(mesh2D: Mesh):
    with pytest.raises(AssertionError, match="_Behavior"):
        Simulations.InElastic(mesh2D, Isotropic(2))


def test_InElastic_dt_is_zero_until_set_and_never_negative(mesh2D: Mesh):
    simu = Simulations.InElastic(mesh2D, Linear(Isotropic(2, E=E, v=nu)))
    assert simu.dt == 0.0

    simu.dt = DT

    assert simu.dt == DT
    with pytest.raises(AssertionError, match="dt"):
        simu.dt = -1.0


def test_InElastic_external_is_a_copy(mesh2D: Mesh):
    simu = _thermo(mesh2D)
    simu.Set_external(T=10.0)

    simu.external["T"][:] = 0.0

    assert np.all(simu.external["T"] == 10.0)


def test_InElastic_material_is_the_given_behavior(mesh3D: Mesh):
    behavior = _plastic()

    assert Simulations.InElastic(mesh3D, behavior).material is behavior


STRAIN = 1e-4  # below the plastic yield strain


def _stretched_plastic(mesh: Mesh) -> Simulations.InElastic:
    """The bar's exact elastic displacement at STRAIN injected, nothing committed: never solved. Plane strain in 2D, since plane stress commits eps_zz."""
    elastic = Models.Elastic.Isotropic(mesh.dim, E, nu, planeStress=False)
    simu = Simulations.InElastic(mesh, _plastic(elastic))
    lateral = nu / (1 - nu) if mesh.dim == 2 else nu
    u = _reference.bar_displacement(
        mesh.coord,
        L,
        STRAIN * L,
        lateral,
        mesh.dim,
    )
    simu._Set_solutions(simu.problemType, u.ravel())
    return simu


def _Sxx(mesh: Mesh) -> float:
    """Uniaxial stress of that bar: E·eps, E/(1 - nu²)·eps in plane strain."""
    return E * STRAIN / (1 - nu**2 if mesh.dim == 2 else 1)


def test_InElastic_displacement_is_the_dof_vector(mesh3D: Mesh):
    simu = _stretched_plastic(mesh3D)

    ux = simu.displacement.reshape(mesh3D.Nn, 3)[:, 0]

    assert rel_error(ux, STRAIN * mesh3D.coord[:, 0]) < 1e-12
    assert np.array_equal(simu.Result("ux"), ux)


def test_InElastic__Calc_Epsilon_is_per_group_and_matches_the_bar(subtests):
    for meshName in ("QUAD4+TRI3", "HEXA8+PRISM6"):
        with subtests.test(mesh=meshName):
            mesh = bar_mesh(meshName)
            simu = _stretched_plastic(mesh)

            Exx = component(simu._Calc_Epsilon(), mesh, 0)

            assert rel_error(Exx, STRAIN) < 1e-10


def test_InElastic__Calc_Sigma_is_per_group_and_uniaxial(subtests):
    for meshName in ("QUAD4+TRI3", "HEXA8+PRISM6"):
        with subtests.test(mesh=meshName):
            mesh = bar_mesh(meshName)
            simu = _stretched_plastic(mesh)

            stress = simu._Calc_Sigma()

            Sxx = _Sxx(mesh)
            assert rel_error(component(stress, mesh, 0), Sxx) < 1e-10
            assert rel_error(component(stress, mesh, 1), 0.0, scale=Sxx) < 1e-10


def test_InElastic__Calc_Sigma_lives_at_the_rigi_points(mesh2D: Mesh):
    simu = Simulations.InElastic(mesh2D, _plastic(Isotropic(2, E=E, v=nu)))

    with pytest.raises(AssertionError, match="rigi"):
        simu._Calc_Sigma(MatrixType.mass)
