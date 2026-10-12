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

C_ORTHO = np.array([[60, 20, 0], [20, 120, 0], [0, 0, 30]])
C_COUPLED = np.array([[60, 20, 5], [20, 120, 10], [5, 10, 30]])
TI = dict(El=11580, Et=500, Gl=450, vl=0.02, vt=0.44)
c = np.sqrt(2) / 2

MATERIALS = {
    "Isot-2D-stress": lambda: Isotropic(2, E=210e9, v=0.3, planeStress=True),
    "Isot-2D-strain": lambda: Isotropic(2, E=210e9, v=0.3, planeStress=False),
    "Isot-3D": lambda: Isotropic(3, E=210e9, v=0.3),
    "TI-3D-rotated": lambda: TransverselyIsotropic(
        3, **TI, axis_l=[c, c, 0], axis_t=[c, -c, 0]
    ),
    "TI-3D-l_along_y": lambda: TransverselyIsotropic(
        3, **TI, axis_l=[0, 1, 0], axis_t=[1, 0, 0]
    ),
    "TI-2D-stress": lambda: TransverselyIsotropic(2, **TI, planeStress=True),
    "TI-2D-strain": lambda: TransverselyIsotropic(2, **TI, planeStress=False),
    "Anisot-2D-voigt": lambda: Anisotropic(2, C_ORTHO, True),
    "Anisot-2D-kelvin": lambda: Anisotropic(2, C_ORTHO, False),
    "Anisot-2D-coupled": lambda: Anisotropic(2, C_COUPLED, True),
}

ISOTROPIC_ONLY = [
    PhaseField.SplitType.Amor,
    PhaseField.SplitType.Miehe,
    PhaseField.SplitType.Stress,
]


def _Cases():
    for name, material in MATERIALS.items():
        for split in PhaseField.Get_splits():
            if not name.startswith("Isot") and split in ISOTROPIC_ONLY:
                continue
            yield pytest.param((material, split), id=f"{name}-{split}")


@pytest.fixture(params=list(_Cases()))
def pfm(request) -> PhaseField:
    material, split = request.param
    return PhaseField(material(), split, PhaseField.ReguType.AT2, 1, 1)


def test_split_stiffnesses_sum_to_C(pfm: PhaseField):
    kinematics = affine_kinematics(pfm.dim).kinematics
    cP, cM = pfm._Split_C(kinematics, verif=True)
    C = FeArray.broadcast(pfm.material.C, *cP.shape[:2], tensor_shape=cP.shape[2:])
    assert rel_error(cP + cM, C) < TOL


def test_split_energies_sum_to_the_elastic_energy(pfm: PhaseField):
    affine = affine_kinematics(pfm.dim)
    psiP, psiM = pfm.Compute_psi(affine.kinematics)
    material: _Elastic = pfm.material
    psi = affine.epsilon @ material.C @ affine.epsilon / 2
    assert rel_error(psiP + psiM, psi) < TOL
