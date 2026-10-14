# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import os

import numpy as np
import pytest

from EasyFEA import ElemType, Mesh, Models, Simulations

from _numerics import rel_error
from ._simus import measure
from . import _reference
from ._fields import bar_mesh, component

E, v = 210e3, 0.3
GC, L0 = 2.7, 0.1
STRAIN = 0.02  # past the AT1 threshold, so both regularizations damage


def _simu(
    mesh: Mesh,
    split: str = "Bourdin",
    regularization: str = "AT2",
    d: float = 0.0,
    strain: float = STRAIN,
) -> Simulations.PhaseField:
    """A bar pulled by `strain`, its exact displacement and a uniform damage `d` injected: never solved."""
    material = Models.Elastic.Isotropic(mesh.dim, E, v, planeStress=True)
    pfm = Models.PhaseField(
        material,
        split,
        regularization,
        GC,
        L0,
    )
    simu = Simulations.PhaseField(mesh, pfm)
    length = mesh.coord[:, 0].max()
    u = _reference.bar_displacement(
        mesh.coord,
        length,
        strain * length,
        v,
        mesh.dim,
    )
    simu._Set_solutions(simu.ProblemTypes.elastic, u.ravel())
    simu._Set_solutions(simu.ProblemTypes.damage, np.full(mesh.Nn, d))
    return simu


def _quad4(make_mesh_2D) -> Mesh:
    return make_mesh_2D(elemType=ElemType.QUAD4, isOrganised=True)


def test_PhaseField_uniform_history_drives_a_uniform_damage(subtests, make_mesh_2D):
    """Unsplit, the history is E·eps²/2; d = f / r then balances the damage system, its gradient term vanishing."""
    mesh = _quad4(make_mesh_2D)
    psiP = E * STRAIN**2 / 2
    for regularization in Models.PhaseField.ReguType:
        with subtests.test(regularization=str(regularization)):
            simu = _simu(mesh, regularization=regularization)
            Kd, _, _, Fd = simu.Get_K_C_M_F(simu.ProblemTypes.damage)

            d = _reference.homogeneous_damage(psiP, GC, L0, regularization)
            residual = Kd @ np.full(mesh.Nn, d) - Fd.toarray().ravel()

            history = simu.Result("psiP", nodeValues=False)
            assert rel_error(history, psiP) < 1e-12
            assert rel_error(residual, 0.0, scale=np.abs(Fd).max()) < 1e-12


def test_PhaseField_rejects_a_model_that_is_not_a_phase_field(make_mesh_2D):
    with pytest.raises(AssertionError, match="phase field"):
        Simulations.PhaseField(make_mesh_2D(), Models.Elastic.Isotropic(2))


def test_PhaseField_uz_is_the_z_displacement(make_mesh_3D):
    mesh = make_mesh_3D(elemType=ElemType.HEXA8, isOrganised=True)
    simu = _simu(mesh, strain=1e-4)

    uz = simu.Result("uz")

    assert rel_error(uz, -v * 1e-4 * mesh.coord[:, 2]) < 1e-12


def test_PhaseField_displacement_is_the_elastic_dof_vector(make_mesh_2D):
    mesh = _quad4(make_mesh_2D)
    simu = _simu(mesh)

    ux = simu.displacement.reshape(mesh.Nn, 2)[:, 0]

    assert rel_error(ux, STRAIN * mesh.coord[:, 0]) < 1e-12
    assert np.array_equal(simu.Result("ux"), ux)


def test_PhaseField_damage_is_the_nodal_damage(make_mesh_2D):
    simu = _simu(_quad4(make_mesh_2D), d=0.3)

    assert simu.damage.shape == (simu.mesh.Nn,)
    assert np.all(simu.damage == 0.3)
    assert np.array_equal(simu.Result("damage"), simu.damage)


def test_PhaseField_Save_Iter_before_any_solve_saves_no_convergence(make_mesh_2D):
    simu = _simu(_quad4(make_mesh_2D), d=0.3)

    simu.Save_Iter()

    results = simu.Get_results(0)
    assert np.all(results["damage"] == 0.3)
    assert (results["Niter"], results["convIter"], results["timeIter"]) == (0, 0, 0)


def test_PhaseField_phaseFieldModel_is_the_given_model(make_mesh_2D):
    pfm = Models.PhaseField(Models.Elastic.Isotropic(2), "He", "AT1", 1, 0.01)

    assert Simulations.PhaseField(make_mesh_2D(), pfm).phaseFieldModel is pfm


CHANGES = {
    "E": lambda pfm: setattr(pfm.material, "E", 2 * pfm.material.E),
    "v": lambda pfm: setattr(pfm.material, "v", 0.1),
    "split": lambda pfm: setattr(pfm, "split", "Miehe"),
    "regularization": lambda pfm: setattr(pfm, "regularization", "AT2"),
    "Gc": lambda pfm: setattr(pfm, "Gc", 10),
    "l0": lambda pfm: setattr(pfm, "l0", 1),
    "solver": lambda pfm: setattr(pfm, "solver", pfm.SolverType.BoundConstrain),
    "A": lambda pfm: setattr(pfm, "A", np.eye(2) * 3),
}


def test_PhaseField_phaseFieldModel_change_flags_reassembly(
    subtests, make_mesh_2D, assert_change_flags_reassembly
):
    mesh = make_mesh_2D()
    for name, change in CHANGES.items():
        with subtests.test(change=name):
            pfm = Models.PhaseField(
                Models.Elastic.Isotropic(2),
                "He",
                "AT1",
                1,
                0.01,
            )
            simu = Simulations.PhaseField(mesh, pfm)

            assert_change_flags_reassembly(simu, lambda: change(pfm))


def test_PhaseField_needUpdate_until_both_problems_are_assembled(make_mesh_2D):
    pfm = Models.PhaseField(Models.Elastic.Isotropic(2), "He", "AT1", 1, 0.01)
    simu = Simulations.PhaseField(make_mesh_2D(), pfm)

    simu.Get_K_C_M_F("elastic")
    assert simu.needUpdate
    simu.Get_K_C_M_F("damage")
    assert not simu.needUpdate


def test_PhaseField_Detect_Damage_compares_the_nodes_to_the_threshold(make_mesh_2D):
    simu = _simu(_quad4(make_mesh_2D), d=0.4)
    nodes = simu.mesh.nodes[:3]

    assert simu.Detect_Damage(nodes, threshold=0.2)
    assert not simu.Detect_Damage(nodes)
    assert not simu.Detect_Damage(np.array([], dtype=int), threshold=0.0)


def test_PhaseField__Calc_Psi_Elas_of_an_unsplit_bar_is_the_degraded_energy(
    make_mesh_2D,
):
    mesh = _quad4(make_mesh_2D)
    d = 0.3
    simu = _simu(mesh, d=d)
    psi = E * STRAIN**2 / 2

    assert rel_error(simu._Calc_Psi_Elas(), (1 - d) ** 2 * psi * measure(mesh)) < 1e-9


def test_PhaseField__Calc_Psi_Crack_of_a_uniform_damage_has_no_gradient_term(
    make_mesh_2D,
):
    """1/2 dᵀ Kd d, whose reaction is Gc/l0 + 2 psi+ for AT2."""
    mesh = _quad4(make_mesh_2D)
    d = 0.3
    simu = _simu(mesh, d=d)
    r = GC / L0 + E * STRAIN**2

    assert rel_error(simu._Calc_Psi_Crack(), r * d**2 / 2 * measure(mesh)) < 1e-12


def test_PhaseField__Calc_Psi_Ext_is_the_work_of_the_given_forces(make_mesh_2D):
    simu = _simu(_quad4(make_mesh_2D))
    f = np.linspace(-1, 1, simu.displacement.size)

    assert np.isclose(simu._Calc_Psi_Ext(f), simu.displacement @ f, rtol=1e-14)
    assert simu._Calc_Psi_Ext(np.zeros_like(f)) == 0
    with pytest.raises(AssertionError, match="f_n"):
        simu._Calc_Psi_Ext(f[:-1])


def test_PhaseField__Calc_Epsilon_is_per_group_and_matches_the_bar():
    mesh = bar_mesh("QUAD4+TRI3")
    simu = _simu(mesh, "Amor", strain=1e-4)

    Exx = component(simu._Calc_Epsilon(), mesh, 0)

    assert rel_error(Exx, 1e-4) < 1e-10


def test_PhaseField__Calc_Sigma_is_per_group_and_uniaxial():
    mesh = bar_mesh("QUAD4+TRI3")
    simu = _simu(mesh, "Amor", strain=1e-4)

    stress = simu._Calc_Sigma()

    assert rel_error(component(stress, mesh, 0), E * 1e-4) < 1e-10
    assert rel_error(component(stress, mesh, 1), 0.0, scale=E * 1e-4) < 1e-10


def test_PhaseField__Calc_Sigma_of_a_uniaxial_3d_strain_is_finite_with_miehe(
    make_mesh_3D,
):
    mesh = make_mesh_3D(elemType=ElemType.HEXA8, isOrganised=True)
    simu = _simu(mesh, "Miehe", d=0.3)

    assert np.isfinite(np.asarray(simu._Calc_Sigma())).all()


def test_PhaseField_Folder_names_the_run_after_its_options(tmp_path):
    folder = str(tmp_path / "PhaseField")

    path = Simulations.PhaseField.Folder(
        folder,
        "Steel",
        "Miehe",
        "AT2",
        "CP",
        1e-3,
        "HistoryDamage",
        test=True,
        optimMesh=True,
        closeCrack=True,
        nL=100,
        theta=30.0,
    )

    name = "Steel_Miehe_AT2_CP_closeCrack_optimMesh_HistoryDamage_conv0.001"
    assert path == os.path.join(folder, "Test", f"{name} theta=30.0 nL=100")
    assert os.path.isdir(folder)


def test_PhaseField_Folder_omits_the_defaults(tmp_path):
    path = Simulations.PhaseField.Folder(
        str(tmp_path),
        "",
        "He",
        "AT1",
        "",
        1.0,
        "History",
        test=False,
        nL=2.5,
    )

    assert path == os.path.join(str(tmp_path), "He_AT1 nL=2.50")
