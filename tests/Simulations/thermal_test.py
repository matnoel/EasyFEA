# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np

from EasyFEA import ElemType, Mesh, Models, Simulations
from EasyFEA.Geoms import Domain

from _numerics import rel_error
from . import _reference

T0, TL = 10.0, 40.0
L, H = 2.0, 1.0


def _slab_mesh() -> Mesh:
    domain = Domain((0, 0), (L, H), 0.25)
    return domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)


def _held_slab(k: float = 1.0, thickness: float = 1.0) -> Simulations.Thermal:
    """Held at T0 on x=0 and TL on x=L, its steady temperature injected: never solved."""
    mesh = _slab_mesh()
    simu = Simulations.Thermal(mesh, Models.Thermal(k, 1.0, thickness=thickness))
    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)
    simu.add_dirichlet(nodesX0, [T0], ["t"])
    simu.add_dirichlet(nodesXL, [TL], ["t"])
    t = _reference.slab_temperature(mesh.coord[:, 0], L, T0, TL)
    simu._Set_solutions(simu.problemType, t, np.zeros_like(t))
    return simu


def _free(simu: Simulations.Thermal) -> np.ndarray:
    return simu.Bc_dofs_known_unknown(simu.problemType)[1]


def test_Thermal_conduction_of_a_slab_stores_half_k_gradient2_V():
    k = 3.0
    simu = _held_slab(k)
    t = simu.thermal
    K = simu.Get_K_C_M_F()[0]

    energy = t @ K @ t / 2

    assert rel_error(K @ np.ones_like(t), 0.0, scale=abs(K).max()) < 1e-12
    assert rel_error(energy, k * ((TL - T0) / L) ** 2 / 2 * L * H) < 1e-12


def test_Thermal_flux_through_a_slab_does_not_depend_on_the_thickness(subtests):
    """Held at 0 on x=0, a flux q enters at x=L: t = q x / k balances K t = q, whatever the thickness."""
    k, q = 3.0, 5.0
    mesh = _slab_mesh()
    t = q * mesh.coord[:, 0] / k
    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)
    for thickness in (1.0, 2.5):
        with subtests.test(thickness=thickness):
            simu = Simulations.Thermal(mesh, Models.Thermal(k, thickness=thickness))
            simu.add_dirichlet(nodesX0, [0], ["t"])
            simu.add_surfLoad(nodesXL, [q], ["t"])
            K = simu.Get_K_C_M_F()[0]

            residual = K @ t - simu.Bc_vector_Neumann()

            free = _free(simu)
            assert rel_error(residual[free], 0.0, scale=q * H * thickness) < 1e-12


def test_Thermal_thermal_of_the_held_slab_is_its_result():
    simu = _held_slab()

    expected = _reference.slab_temperature(simu.mesh.coord[:, 0], L, T0, TL)
    assert simu.thermal.shape == (simu.mesh.Nn,)
    assert np.array_equal(simu.Result("thermal"), expected)
    assert np.array_equal(simu.thermal, expected)


def test_Thermal_thermalDot_vanishes_on_a_steady_slab():
    """The steady temperature balances the parabolic system with a zero rate."""
    simu = _held_slab()
    simu.Solver_Set_Parabolic_Algorithm(dt=0.1)
    K, C, _, F = simu.Get_K_C_M_F()

    residual = K @ simu.thermal + C @ simu.thermalDot - F.toarray().ravel()

    assert np.all(simu.thermalDot == 0)
    assert rel_error(residual[_free(simu)], 0.0, scale=TL - T0) < 1e-12


def test_Thermal_thermalDot_of_an_insulated_heated_body_is_r_over_rho_c():
    """c·ṫ = r: a uniform rate r / (rho c) balances the source."""
    rho, c, r = 2.0, 3.0, 12.0
    mesh = _slab_mesh()
    simu = Simulations.Thermal(mesh, Models.Thermal(1.0, c))
    simu.rho = rho
    simu.Solver_Set_Parabolic_Algorithm(0.1, alpha=1.0)
    simu.add_volumeLoad(mesh.nodes, [r], ["t"])
    rate = np.full(mesh.Nn, r / (rho * c))
    simu._Set_solutions(simu.problemType, np.zeros(mesh.Nn), rate)
    C = simu.Get_K_C_M_F()[1]

    assert rel_error(C @ simu.thermalDot, simu.Bc_vector_Neumann()) < 1e-12


def test_Thermal_thermalModel_is_the_given_model():
    domain = Domain((0, 0), (1, 1))
    model = Models.Thermal(1, 1)

    simu = Simulations.Thermal(domain.Mesh_2D(), model)

    assert simu.thermalModel is model


def test_Thermal_thermalModel_change_flags_reassembly(
    subtests, make_mesh_2D, assert_change_flags_reassembly
):
    mesh = make_mesh_2D()
    for name, factor in (("k", 2.0), ("c", 0.2)):
        with subtests.test(change=name):
            model = Models.Thermal(1, 1)
            simu = Simulations.Thermal(mesh, model)

            assert_change_flags_reassembly(
                simu, lambda: setattr(model, name, factor * getattr(model, name))
            )
