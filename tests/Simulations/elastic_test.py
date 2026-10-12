# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np
import pytest

from EasyFEA import Mesh, Models, Simulations, SolverType
from EasyFEA.Geoms import Domain

from _numerics import rel_error
from . import _reference

E, v = 210e3, 0.3
UD = 1e-3


def _pulled_bar(mesh: Mesh) -> Simulations.Elastic:
    material = Models.Elastic.Isotropic(mesh.dim, E, v, planeStress=True)
    simu = Simulations.Elastic(mesh, material)
    simu.solver = SolverType.scipy
    _reference.pull_bar(simu, UD)
    simu.Solve()
    return simu


def test_bar_in_tension_is_solved_exactly(box_mesh: Mesh):
    simu = _pulled_bar(box_mesh)
    L = box_mesh.coord[:, 0].max()

    expected = _reference.bar_displacement(
        box_mesh.coord,
        L,
        UD,
        v,
        box_mesh.dim,
    )
    assert rel_error(simu.displacement, expected.ravel()) < 1e-10
    Sxx = simu.Result("Sxx", nodeValues=False)
    assert rel_error(Sxx, _reference.bar_stress(E, L, UD)) < 1e-10


def test_bar_strain_is_uniaxial_with_poisson_contraction(box_mesh: Mesh):
    simu = _pulled_bar(box_mesh)
    strain = UD / box_mesh.coord[:, 0].max()

    assert rel_error(simu.Result("Exx", nodeValues=False), strain) < 1e-10
    Eyy = simu.Result("Eyy", nodeValues=False)
    assert rel_error(Eyy, -v * strain, scale=strain) < 1e-10
    assert rel_error(simu.Result("Exy", nodeValues=False), 0.0, scale=strain) < 1e-10


def test_an_equilibrated_bar_stays_at_rest_under_newmark(box_mesh: Mesh):
    simu = _pulled_bar(box_mesh)
    static = simu.displacement.copy()

    simu.Solver_Set_Hyperbolic_Algorithm(dt=0.1)
    simu.Solve()

    assert rel_error(simu.displacement, static) < 1e-10


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


CHANGES = {
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


def _assembled(material) -> Simulations.Elastic:
    domain = Domain((0, 0), (1, 1))
    mesh = domain.Mesh_2D()
    simu = Simulations.Elastic(mesh, material)
    simu.Get_K_C_M_F()
    assert not simu.needUpdate
    return simu


@pytest.mark.parametrize("make, change", CHANGES.values(), ids=CHANGES.keys())
def test_a_material_change_flags_the_simulation_for_reassembly(make, change):
    material = make()
    simu = _assembled(material)

    change(material)

    assert simu.needUpdate


INVALID = {
    "negative E": ("E", -10),
    "v above 0.5": ("v", 10),
    "planeStress not a bool": ("planeStress", 10),
}


@pytest.mark.parametrize("name, value", INVALID.values(), ids=INVALID.keys())
def test_a_rejected_material_change_does_not_flag_an_update(name: str, value):
    material = _isotropic()
    simu = _assembled(material)

    with pytest.raises(AssertionError):
        setattr(material, name, value)

    assert not simu.needUpdate
