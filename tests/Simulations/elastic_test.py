# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Callable

import numpy as np
import pytest

from EasyFEA import AlgoType, ElemType, Mesh, Models, Simulations
from EasyFEA.Geoms import Domain

from _numerics import rel_error
from . import _reference
from ._fields import BAR_MESHES, bar_mesh, component
from ._simus import measure, pulled_bar

E, v = 210e3, 0.3
UD = 1e-3
L, H = 120.0, 13.0


def _quad4_bar(make_mesh_2D) -> Mesh:
    return make_mesh_2D(elemType=ElemType.QUAD4, isOrganised=True)


def _pulled_bar(mesh: Mesh) -> Simulations.Elastic:
    """`pulled_bar` stretched by UD at x = L."""
    return pulled_bar(mesh, [E * UD / mesh.coord[:, 0].max()])


def test_Elastic_stiffness_holds_the_rigid_motions(make_mesh_2D):
    mesh = _quad4_bar(make_mesh_2D)
    simu = Simulations.Elastic(mesh, Models.Elastic.Isotropic(2, E, v))
    x, y = mesh.coord[:, 0], mesh.coord[:, 1]
    rigid = {
        "x": np.column_stack([np.ones_like(x), 0 * x]),
        "y": np.column_stack([0 * x, np.ones_like(x)]),
        "rotation": np.column_stack([-y, x]),
    }

    K = simu.Get_K_C_M_F()[0]

    scale = abs(K).max() * np.abs(mesh.coord).max()
    for u in rigid.values():
        assert rel_error(K @ u.ravel(), 0.0, scale=scale) < 1e-12


def test_Elastic_stiffness_stores_half_E_eps2_V_in_a_pulled_bar(make_mesh_2D):
    simu = _pulled_bar(_quad4_bar(make_mesh_2D))
    length = simu.mesh.coord[:, 0].max()
    K = simu.Get_K_C_M_F()[0]

    energy = simu.displacement @ K @ simu.displacement / 2

    expected = E * (UD / length) ** 2 / 2 * measure(simu.mesh)
    assert rel_error(energy, expected) < 1e-12


def test_Elastic_rejects_a_model_that_is_not_elastic(make_mesh_2D):
    with pytest.raises(AssertionError, match="elastic model"):
        Simulations.Elastic(make_mesh_2D(), Models.Thermal(1.0))


def test_Elastic_displacement_of_the_pulled_bar_gives_its_exact_stress(make_mesh_2D):
    mesh = _quad4_bar(make_mesh_2D)
    simu = _pulled_bar(mesh)
    length = mesh.coord[:, 0].max()

    Sxx = simu.Result("Sxx", nodeValues=False)
    Syy = simu.Result("Syy", nodeValues=False)

    ux = simu.displacement.reshape(mesh.Nn, 2)[:, 0]
    assert np.array_equal(ux, UD / length * mesh.coord[:, 0])
    assert rel_error(Sxx, _reference.bar_stress(E, length, UD)) < 1e-12
    assert rel_error(Syy, 0.0, scale=E * UD) < 1e-12


def _dynamic(make_mesh_2D) -> tuple[Simulations.Elastic, np.ndarray]:
    """A Newmark simulation with u, 2u and 3u injected; returns it and u."""
    mesh = _quad4_bar(make_mesh_2D)
    simu = Simulations.Elastic(mesh, Models.Elastic.Isotropic(2, E, v))
    simu.Solver_Set_Hyperbolic_Algorithm(0.1)
    u = np.linspace(0, 1, mesh.Nn * 2)
    simu._Set_solutions(simu.problemType, u, 2 * u, 3 * u)
    return simu, u


def test_Elastic_speed_is_the_velocity_dof_vector(make_mesh_2D):
    simu, u = _dynamic(make_mesh_2D)

    assert np.array_equal(simu.speed, 2 * u)


def test_Elastic_accel_is_the_acceleration_dof_vector(make_mesh_2D):
    simu, u = _dynamic(make_mesh_2D)

    assert np.array_equal(simu.accel, 3 * u)


def test_Elastic_material_is_the_given_model(make_mesh_2D):
    material = Models.Elastic.Isotropic(2)

    simu = Simulations.Elastic(make_mesh_2D(), material)

    assert simu.material is material


def _isotropic() -> Models.Elastic.Isotropic:
    return Models.Elastic.Isotropic(2)


def _transversely_isotropic() -> Models.Elastic.TransverselyIsotropic:
    return Models.Elastic.TransverselyIsotropic(
        2,
        El=10,
        Et=10,
        Gl=10,
        vl=0.1,
        vt=0.1,
    )


def _anisotropic() -> Models.Elastic.Anisotropic:
    P = Models.Get_Pmat(np.array([0.0, 1.0]), np.array([-1.0, 0.0]))
    C = Models.Apply_Pmat(P, _transversely_isotropic().C)
    return Models.Elastic.Anisotropic(2, C, False)


def _toggle_plane_stress(material) -> None:
    material.planeStress = not material.planeStress


CHANGES: dict[str, tuple[Callable, Callable]] = {
    "isotropic E": (_isotropic, lambda m: setattr(m, "E", 2 * m.E)),
    "isotropic v": (_isotropic, lambda m: setattr(m, "v", 0.2)),
    "isotropic planeStress": (_isotropic, _toggle_plane_stress),
    "transversely isotropic El": (
        _transversely_isotropic,
        lambda m: setattr(m, "El", 2 * m.El),
    ),
    "transversely isotropic Et": (
        _transversely_isotropic,
        lambda m: setattr(m, "Et", 2 * m.Et),
    ),
    "transversely isotropic Gl": (
        _transversely_isotropic,
        lambda m: setattr(m, "Gl", 2 * m.Gl),
    ),
    "transversely isotropic vl": (
        _transversely_isotropic,
        lambda m: setattr(m, "vl", 0.2),
    ),
    "transversely isotropic vt": (
        _transversely_isotropic,
        lambda m: setattr(m, "vt", 0.4),
    ),
    "transversely isotropic planeStress": (
        _transversely_isotropic,
        _toggle_plane_stress,
    ),
    "anisotropic C": (
        _anisotropic,
        lambda m: m.Set_C(_isotropic().C, False),
    ),
}


def test_Elastic_material_change_flags_reassembly(
    subtests, make_mesh_2D, assert_change_flags_reassembly
):
    mesh = make_mesh_2D()
    for name, (make, change) in CHANGES.items():
        with subtests.test(change=name):
            material = make()
            simu = Simulations.Elastic(mesh, material)

            assert_change_flags_reassembly(simu, lambda: change(material))


INVALID = {
    "negative E": ("E", -10),
    "v above 0.5": ("v", 10),
    "planeStress not a bool": ("planeStress", 10),
}


def test_Elastic_material_rejected_change_does_not_flag_reassembly(
    subtests, make_mesh_2D
):
    mesh = make_mesh_2D()
    for case, (name, value) in INVALID.items():
        with subtests.test(change=case):
            material = _isotropic()
            simu = Simulations.Elastic(mesh, material)
            simu.Get_K_C_M_F()
            assert not simu.needUpdate

            with pytest.raises(AssertionError):
                setattr(material, name, value)

            assert not simu.needUpdate


def test_Elastic_Set_Rayleigh_Damping_Coefs_builds_C_from_K_and_M(make_mesh_2D):
    simu = Simulations.Elastic(make_mesh_2D(), Models.Elastic.Isotropic(2))
    simu.rho = 7.8
    K, C, M, _ = simu.Get_K_C_M_F()
    assert C.nnz == 0

    simu.Set_Rayleigh_Damping_Coefs(coefM=0.3, coefK=2e-3)
    K, C, M, _ = simu.Get_K_C_M_F()

    expected = 2e-3 * K + 0.3 * M
    assert rel_error(C.toarray(), expected.toarray()) < 1e-14


def test_Elastic__Calc_Psi_Elas_of_a_pulled_bar_is_half_E_eps2_V(make_mesh_2D):
    mesh = _quad4_bar(make_mesh_2D)
    simu = _pulled_bar(mesh)
    length = mesh.coord[:, 0].max()
    expected = E * (UD / length) ** 2 / 2 * measure(mesh)

    W_e = simu._Calc_Psi_Elas(returnScalar=False)

    assert rel_error(simu._Calc_Psi_Elas(), expected) < 1e-10
    assert W_e.shape == (mesh.Ne,)
    assert rel_error(W_e.sum(), expected) < 1e-10


def test_Elastic__Calc_ZZ1_of_a_uniform_stress_is_zero(make_mesh_2D):
    mesh = _quad4_bar(make_mesh_2D)
    simu = _pulled_bar(mesh)

    error, error_e = simu._Calc_ZZ1()

    assert error < 1e-10
    assert error_e.shape == (mesh.Ne,)
    assert np.abs(error_e).max() < 1e-10


def test_Elastic__Calc_Epsilon_is_per_group_and_uniaxial(subtests):
    for meshName in BAR_MESHES:
        with subtests.test(mesh=meshName):
            mesh = bar_mesh(meshName)
            simu = _pulled_bar(mesh)

            strain = simu._Calc_Epsilon()

            assert rel_error(component(strain, mesh, 0), UD / L) < 1e-10
            assert rel_error(component(strain, mesh, 1), -v * UD / L) < 1e-10
            assert rel_error(component(strain, mesh, -1), 0.0, scale=UD / L) < 1e-10


def test_Elastic__Calc_Sigma_is_per_group_and_uniaxial(subtests):
    for meshName in BAR_MESHES:
        with subtests.test(mesh=meshName):
            mesh = bar_mesh(meshName)
            simu = _pulled_bar(mesh)

            stress = simu._Calc_Sigma()

            Sxx = E * UD / L
            assert rel_error(component(stress, mesh, 0), Sxx) < 1e-10
            assert rel_error(component(stress, mesh, 1), 0.0, scale=Sxx) < 1e-10


def test_Elastic_Results_Get_Iteration_Summary_reports_the_energy_and_extremes(
    make_mesh_3D,
):
    simu = _pulled_bar(make_mesh_3D(elemType=ElemType.HEXA8, isOrganised=True))

    summary = simu.Results_Get_Iteration_Summary()

    assert f"W def = {simu._Calc_Psi_Elas():.2f}" in summary
    assert f"Ux max = {UD:.2e}" in summary
    assert "Uz max" in summary


def test_Elastic_Results_nodeFields_elementFields_adds_speed_and_accel_when_dynamic(
    make_mesh_2D,
):
    simu = Simulations.Elastic(make_mesh_2D(), Models.Elastic.Isotropic(2))
    static = simu.Results_nodeFields_elementFields(details=True)

    simu.Solver_Set_Hyperbolic_Algorithm(0.1, AlgoType.midpoint)

    assert static == (["displacement"], ["Svm", "Stress", "Strain"])
    assert simu.Results_nodeFields_elementFields()[0] == [
        "displacement",
        "speed",
        "accel",
    ]


def test_Mesh_Optim_ZZ1_rejects_another_simulation(tmp_path):
    def DoSimu(refineGeom: str | None):
        domain = Domain((0, 0), (1, 1))
        return Simulations.Thermal(domain.Mesh_2D(), Models.Thermal(1.0))

    with pytest.raises(AssertionError, match="Displacement simulation"):
        Simulations.Mesh_Optim_ZZ1(DoSimu, str(tmp_path))
