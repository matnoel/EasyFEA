# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np
import pytest

from EasyFEA import ElemType, Mesh, Models, Simulations
from EasyFEA.Geoms import Domain

from _numerics import rel_error
from . import _reference

E, v = 210e3, 0.3
GC, L0 = 2.7, 0.1
STRAIN = 0.02  # past the AT1 threshold, so both regularizations damage


def _pull(simu: Simulations.PhaseField) -> np.ndarray:
    """One loading step; returns the history it leaves."""
    simu.Bc_Init()
    _reference.pull_bar(simu, STRAIN * simu.mesh.coord[:, 0].max())
    simu.Solve()
    simu.Save_Iter()
    return simu.Result("psiP", nodeValues=False)


def _pulled_twice(mesh: Mesh, split: str, regularization: str):
    """Returns the simulation and the history of the first step, which drives the second step's damage."""
    material = Models.Elastic.Isotropic(mesh.dim, E, v, planeStress=True)
    pfm = Models.PhaseField(material, split, regularization, GC, L0)
    simu = Simulations.PhaseField(mesh, pfm)
    # direct: the default cg + gamg stops at ~1e-5
    simu._Solver_Set_PETSc4Py_Options(
        "preonly", "lu", "petsc", simu.ProblemTypes.elastic
    )
    psiP = _pull(simu)
    # only now: with no history yet, the AT1 damage system is singular
    simu._Solver_Set_PETSc4Py_Options(
        "preonly", "lu", "petsc", simu.ProblemTypes.damage
    )
    _pull(simu)
    return simu, psiP


@pytest.mark.parametrize("regularization", list(Models.PhaseField.ReguType), ids=str)
@pytest.mark.parametrize("split", list(Models.PhaseField.SplitType), ids=str)
def test_a_uniform_history_drives_a_uniform_damage(
    make_mesh_2D, split: str, regularization: str
):
    mesh = make_mesh_2D(elemType=ElemType.QUAD4, isOrganised=True)
    simu, psiP = _pulled_twice(mesh, split, regularization)

    assert rel_error(psiP, psiP.mean()) < 1e-12
    expected = _reference.homogeneous_damage(psiP.mean(), GC, L0, regularization)
    assert rel_error(simu.damage, expected) < 1e-12


def test_an_unsplit_bar_damages_under_its_elastic_energy(box_mesh: Mesh):
    """Uniform degradation of a displacement-driven bar leaves the bar displacement unchanged."""
    simu, psiP = _pulled_twice(box_mesh, "Bourdin", "AT2")
    L = box_mesh.coord[:, 0].max()

    assert rel_error(psiP, E * STRAIN**2 / 2) < 1e-10
    expected = _reference.homogeneous_damage(E * STRAIN**2 / 2, GC, L0, "AT2")
    assert rel_error(simu.damage, expected) < 1e-10
    u = _reference.bar_displacement(box_mesh.coord, L, STRAIN * L, v, box_mesh.dim)
    assert rel_error(simu.displacement, u.ravel()) < 1e-10


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


@pytest.mark.parametrize("change", CHANGES.values(), ids=CHANGES.keys())
def test_a_model_change_flags_the_simulation_for_reassembly(change):
    domain = Domain((0, 0), (1, 1))
    mesh = domain.Mesh_2D()
    pfm = Models.PhaseField(Models.Elastic.Isotropic(2), "He", "AT1", 1, 0.01)
    simu = Simulations.PhaseField(mesh, pfm)
    simu.Get_K_C_M_F("elastic")
    assert simu.needUpdate, "clean only once both problems are assembled"
    simu.Get_K_C_M_F("damage")
    assert not simu.needUpdate

    change(pfm)

    assert simu.needUpdate


def test_each_group_keeps_its_history_after_unloading():
    """On a mixed QUAD4 + TRI3 mesh."""
    domain = Domain((0, 0), (120, 13), 13 / 2)
    mesh = domain.Mesh_2D([], ElemType.QUAD4)
    assert len(mesh.Get_list_groupElem()) == 2

    material = Models.Elastic.Isotropic(2, E=210000, v=0.3)
    pfm = Models.PhaseField(material, "Miehe", "AT2", 2700, 13)
    simu = Simulations.PhaseField(mesh, pfm)

    nodes_0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodes_a = mesh.Nodes_Conditions(lambda x, y, z: x == 120)

    def load(ud: float) -> np.ndarray:
        simu.Bc_Init()
        simu.add_dirichlet(nodes_0, [0, 0], ["x", "y"])
        simu.add_dirichlet(nodes_a, [ud], ["x"])
        simu.Solve()
        simu.Save_Iter()
        return simu.Result("psiP", nodeValues=False)

    loaded = load(1e-2)
    unloaded = load(0.0)

    assert np.all(loaded > 0)
    np.testing.assert_allclose(unloaded, loaded)


def test_get_lb_ub_keeps_the_solver():
    domain = Domain((0, 0), (1, 1))
    mesh = domain.Mesh_2D()
    material = Models.Elastic.Isotropic(2)
    pfm = Models.PhaseField(material, "He", "AT1", 1, 0.1)
    pfm.solver = pfm.SolverType.BoundConstrain
    simu = Simulations.PhaseField(mesh, pfm)
    solver = simu.solver

    lb, ub = simu.Get_lb_ub(simu.ProblemTypes.damage)

    assert lb.size == ub.size == mesh.Nn
    assert simu.solver == solver
