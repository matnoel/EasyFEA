# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Simulations shared by these tests and by IO and Viz, which save and draw them."""

import numpy as np

from EasyFEA import ElemType, Mesh, Models, Simulations
from EasyFEA.FEM import BiLinearForm, Field
from EasyFEA.Geoms import Domain

from . import _reference

LOADS = (10.0, 20.0, 30.0)
"""σxx of the saved `pulled_bar` states."""


def pulled_bar(mesh: Mesh, loads=LOADS, withBCs: bool = False) -> Simulations.Elastic:
    """Isotropic bar (E = 210000, v = 0.3, plane stress) under uniaxial σxx = `load`, its exact displacement injected and saved once per load; `withBCs` also declares the clamp and pull, never solved."""
    material = Models.Elastic.Isotropic(mesh.dim)
    simu = Simulations.Elastic(mesh, material)
    length = mesh.coord[:, 0].max()
    nodesClamped = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesPulled = mesh.Nodes_Conditions(lambda x, y, z: x == length)
    for load in loads:
        if withBCs:
            simu.Bc_Init()
            simu.add_dirichlet(nodesClamped, [0] * mesh.dim, simu.Get_unknowns())
            simu.add_surfLoad(nodesPulled, [load], ["x"])
        ud = load / material.E * length
        u = _reference.bar_displacement(mesh.coord, length, ud, material.v, mesh.dim)
        simu._Set_solutions(simu.problemType, u.ravel())
        simu.Save_Iter()
    return simu


def laplacian_form() -> BiLinearForm:
    """∫ ∇u · ∇v."""

    @BiLinearForm
    def laplacian(u: Field, v: Field):
        return u.grad.dot(v.grad)

    return laplacian


def laplacian(mesh: Mesh, dof_n: int = 1) -> Models.WeakForms:
    """`laplacian_form` on a `dof_n` field of the mesh's main group."""
    return Models.WeakForms(Field(mesh.groupElem, dof_n), laplacian_form())


def measure(mesh: Mesh) -> float:
    """Area in 2D, volume in 3D."""
    return mesh.area if mesh.dim == 2 else mesh.volume


def coupled_phasefield(
    *damageTerms: Simulations.Term, mesh: Mesh | None = None
) -> Simulations.PhaseField:
    """PhaseField whose damage equation also folds `damageTerms`: two fields of different dof_n, so a block between them is rectangular."""

    class Coupled(Simulations.PhaseField):
        def Get_terms(self, x, problemType=None):
            terms = super().Get_terms(x, problemType)
            if problemType == self.D:
                terms = terms + list(damageTerms)
            return terms

    if mesh is None:
        domain = Domain((0, 0), (1, 1), 0.5)
        mesh = domain.Mesh_2D([], ElemType.TRI3)
    pfm = Models.PhaseField(
        Models.Elastic.Isotropic(2),
        "He",
        "AT1",
        1,
        0.01,
    )
    return Coupled(mesh, pfm)


def coupling(groupElem, *fieldGroups) -> np.ndarray:
    """A (damage, elastic) block on an isoparametric 2D group: nPe rows, 2·nPe columns, each entry distinct."""
    Ne, nPe = groupElem.Ne, groupElem.nPe
    return np.arange(Ne * nPe * 2 * nPe, dtype=float).reshape(Ne, nPe, 2 * nPe)


PRESSURE = Simulations.IncompressibleHyperElastic.P


def incompressible(
    mesh: Mesh, K: float = 1.0
) -> Simulations.IncompressibleHyperElastic:
    """NeoHookean (`K` = μ/2) u/p simulation of `mesh`."""
    material = Models.HyperElastic.NeoHookean(mesh.dim, K)
    return Simulations.IncompressibleHyperElastic(mesh, material)


def moved_on(simu: Simulations._Simu) -> Simulations._Simu:
    """`simu` past its last saved iteration: twice its u, v = 2u, a = 3u, unsaved."""
    u = 2 * simu._Get_u_n(simu.problemType)
    simu._Set_solutions(simu.problemType, u, 2 * u, 3 * u)
    return simu


def live_state(simu: Simulations._Simu) -> dict[tuple, np.ndarray]:
    """u, v, a of every problem and every result, at nodes and elements."""
    state = {
        (problemType, role): np.copy(get(problemType))
        for problemType in simu.problemTypes
        for role, get in (
            ("u", simu._Get_u_n),
            ("v", simu._Get_v_n),
            ("a", simu._Get_a_n),
        )
    }
    for result in simu.Results_Available():
        for nodeValues in (True, False):
            state[(result, nodeValues)] = np.asarray(simu.Result(result, nodeValues))
    return state


def assert_same_state(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert np.array_equal(actual[key], value, equal_nan=True), key
