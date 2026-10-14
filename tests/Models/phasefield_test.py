# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Every phase-field split decomposes the elastic stiffness and energy, under an affine u."""

import numpy as np
import pytest

from EasyFEA.FEM import FeArray
from EasyFEA.Models.Elastic import (
    _Elastic,
    Isotropic,
    TransverselyIsotropic,
    Anisotropic,
)
from EasyFEA.Models import PhaseField

from _numerics import rel_error
from .conftest import affine_kinematics

TOL = 1e-12

C_COUPLED = np.array([[60, 20, 5], [20, 120, 10], [5, 10, 30]])
TI = dict(El=11580, Et=500, Gl=450, vl=0.02, vt=0.44)
DIAGONAL = 1 / np.sqrt(2)

MATERIALS = {
    "Isot-2D-stress": lambda: Isotropic(2, E=210e9, v=0.3, planeStress=True),
    "Isot-2D-strain": lambda: Isotropic(2, E=210e9, v=0.3, planeStress=False),
    "Isot-3D": lambda: Isotropic(3, E=210e9, v=0.3),
    "Anisot-2D-coupled": lambda: Anisotropic(2, C_COUPLED, useVoigtNotation=True),
    "TI-3D-rotated": lambda: TransverselyIsotropic(
        3,
        **TI,
        axis_l=[DIAGONAL, DIAGONAL, 0],
        axis_t=[DIAGONAL, -DIAGONAL, 0],
    ),
}

TRACE_BASED = [
    PhaseField.SplitType.Amor,
    PhaseField.SplitType.Miehe,
    PhaseField.SplitType.Stress,
]
"""Isotropic-only splits, branching on the sign of a trace."""


def _Cases():
    """(name, phase-field model, scale) per material × applicable split; `-u` too on trace-based splits."""
    for name, material in MATERIALS.items():
        isotropic = isinstance(material(), Isotropic)
        for split in PhaseField.Get_splits():
            if split in TRACE_BASED and not isotropic:
                continue
            pfm = PhaseField(
                material(),
                split,
                PhaseField.ReguType.AT2,
                1,
                1,
            )
            for scale in (1.0, -1.0) if split in TRACE_BASED else (1.0,):
                yield f"{name}-{split}", pfm, scale


def test_PhaseField__Split_C_sums_to_C(subtests):
    for name, pfm, scale in _Cases():
        with subtests.test(case=name, scale=scale):
            kinematics = affine_kinematics(pfm.dim, scale=scale).kinematics
            cP, cM = pfm._Split_C(kinematics, verif=True)
            C = FeArray.broadcast(
                pfm.material.C, *cP.shape[:2], tensor_shape=cP.shape[2:]
            )
            assert rel_error(cP + cM, C) < TOL


def test_PhaseField_Compute_psi_sums_to_the_elastic_energy(subtests):
    for name, pfm, scale in _Cases():
        with subtests.test(case=name, scale=scale):
            affine = affine_kinematics(pfm.dim, scale=scale)
            psiP, psiM = pfm.Compute_psi(affine.kinematics)
            material: _Elastic = pfm.material
            psi = affine.epsilon @ material.C @ affine.epsilon / 2
            assert rel_error(psiP + psiM, psi) < TOL


GC, L0 = 2.0, 0.5
PSI = FeArray.asfearray(np.array([[0.0, 0.5, 3.0]]))
AT1, AT2 = PhaseField.ReguType.AT1, PhaseField.ReguType.AT2


def _isotropic(regularization=AT2, split=PhaseField.SplitType.Miehe) -> PhaseField:
    return PhaseField(Isotropic(2, planeStress=False), split, regularization, GC, L0)


def test_PhaseField_Get_regularizations_are_AT1_and_AT2():
    assert PhaseField.Get_regularizations() == [AT1, AT2]


def test_PhaseField_Get_splits_lists_every_split():
    assert PhaseField.Get_splits() == list(PhaseField.SplitType)
    assert len(PhaseField.Get_splits()) == 14


def test_PhaseField_Get_solvers_lists_every_solver():
    assert [str(s) for s in PhaseField.Get_solvers()] == [
        "History",
        "HistoryDamage",
        "BoundConstrain",
    ]


def test_PhaseField_rejects_an_isotropic_split_on_an_anisotropic_material():
    with pytest.raises(AssertionError, match="only implemented"):
        PhaseField(
            MATERIALS["Anisot-2D-coupled"](),
            PhaseField.SplitType.Amor,
            AT2,
            GC,
            L0,
        )


def test_PhaseField_material_is_the_given_one():
    material = Isotropic(2)

    assert (
        PhaseField(material, PhaseField.SplitType.Bourdin, AT2, GC, L0).material
        is material
    )


def test_PhaseField_k_is_the_diffusion_of_its_regularization():
    assert np.isclose(_isotropic(AT1).k, 3 / 4 * GC * L0)
    assert np.isclose(_isotropic(AT2).k, GC * L0)


def test_PhaseField_c_w_is_the_dissipation_scale():
    assert _isotropic(AT1).c_w == 8 / 3
    assert _isotropic(AT2).c_w == 2


def test_PhaseField_Get_r_e_pg_is_the_reaction():
    assert rel_error(_isotropic(AT1).Get_r_e_pg(PSI), 2 * PSI) < TOL
    assert rel_error(_isotropic(AT2).Get_r_e_pg(PSI), 2 * PSI + GC / L0) < TOL


def test_PhaseField_Get_f_e_pg_is_the_source():
    f_AT1 = np.maximum(2 * PSI - 3 * GC / (8 * L0), 0.0)
    assert rel_error(_isotropic(AT1).Get_f_e_pg(PSI), f_AT1) < TOL
    assert rel_error(_isotropic(AT2).Get_f_e_pg(PSI), 2 * PSI) < TOL


DAMAGES = (0.0, 0.5, 1.0)


def test_PhaseField_Get_g_e_pg_is_the_quadratic_degradation(subtests):
    kinematics = affine_kinematics(2).kinematics
    for d in DAMAGES:
        with subtests.test(d=d):
            d_n = np.full(kinematics.groupElem.Nn, d)
            g = _isotropic().Get_g_e_pg(kinematics, d_n, k_res=1e-6)
            assert rel_error(g, np.full(g.shape, (1 - d) ** 2 + 1e-6)) < TOL


def test_PhaseField_Compute_Sigma_degrades_only_the_tensile_part(subtests):
    kinematics = affine_kinematics(2).kinematics
    pfm = _isotropic()
    sigP, sigM = pfm._Split_Sigma(kinematics)
    for d in DAMAGES:
        with subtests.test(d=d):
            d_n = np.full(kinematics.groupElem.Nn, d)
            expected = ((1 - d) ** 2 + 1e-12) * sigP + sigM
            assert rel_error(pfm.Compute_Sigma(kinematics, d_n), expected) < TOL


def test_PhaseField_Compute_C_degrades_only_the_tensile_part(subtests):
    kinematics = affine_kinematics(2).kinematics
    pfm = _isotropic()
    cP, cM = pfm._Split_C(kinematics)
    for d in DAMAGES:
        with subtests.test(d=d):
            d_n = np.full(kinematics.groupElem.Nn, d)
            expected = ((1 - d) ** 2 + 1e-12) * cP + cM
            assert rel_error(pfm.Compute_C(kinematics, d_n), expected) < TOL


def test_PhaseField__Split_Sigma_sums_to_C_epsilon(subtests):
    for name, pfm, scale in _Cases():
        with subtests.test(case=name, scale=scale):
            affine = affine_kinematics(pfm.dim, scale=scale)
            sigP, sigM = pfm._Split_Sigma(affine.kinematics)
            sigma = pfm.material.C @ affine.epsilon
            assert rel_error(sigP + sigM, np.broadcast_to(sigma, sigP.shape)) < TOL


def test_PhaseField__Eigen_values_vectors_projectors_rebuild_the_tensor(subtests):
    for dim in (2, 3):
        with subtests.test(dim=dim):
            affine = affine_kinematics(dim)
            pfm = PhaseField(
                Isotropic(dim),
                PhaseField.SplitType.Miehe,
                AT2,
                GC,
                L0,
            )
            eps = affine.kinematics.Compute_Epsilon()
            eigs, list_m, _ = pfm._Eigen_values_vectors_projectors(eps, verif=True)
            identity = np.array([1.0] * dim + [0.0] * (eps.shape[-1] - dim))

            rebuilt = sum(eigs[..., i, None] * m for i, m in enumerate(list_m))
            assert rel_error(rebuilt, eps) < TOL
            assert rel_error(sum(list_m), np.broadcast_to(identity, eps.shape)) < TOL
