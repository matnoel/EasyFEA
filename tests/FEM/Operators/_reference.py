# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Small meshes and the element-restriction check of the operator tests."""

from typing import Callable

import numpy as np

from EasyFEA.FEM import ElemType, Mesh, _GroupElem
from EasyFEA.Geoms import Domain

# a field's space is chosen per problem type, which only Simulations names (rule 7 exception)
from EasyFEA import Simulations

from _numerics import rel_error


def element_mesh(dim: int, elemType: ElemType, layers: int = 1) -> Mesh:
    """Organised unit square or cube in `layers`² or `layers`³ cells, few elements to keep finite differences cheap."""
    contour = Domain((0, 0), (1, 1), 1 / layers)
    if dim == 2:
        return contour.Mesh_2D([], elemType, isOrganised=True)
    return contour.Mesh_Extrude([], [0, 0, 1], [layers], elemType, isOrganised=True)


def assert_restricts(operator: Callable, elements: np.ndarray) -> None:
    """`operator(elements)` keeps the rows of `operator(None)` on `elements` and zeros every other."""
    full, restricted = operator(None), operator(elements)
    assert rel_error(restricted[elements], full[elements]) < 1e-14
    assert not np.any(np.delete(restricted, elements, axis=0))


HYPERELASTIC = Simulations.HyperElastic.problemType


def p2_faces() -> tuple[_GroupElem, _GroupElem, np.ndarray]:
    """Surface group of P2 u on a 2-layer TETRA4 cube (TRI6 over the space's points), that of the straight TETRA10 cube, and the DOF (dof_n = 3) of the second at each DOF of the first."""
    mesh = element_mesh(3, ElemType.TETRA4, layers=2)
    meshP2 = mesh._With_FunctionSpace(HYPERELASTIC, 2)
    space = meshP2._Get_FunctionSpace(HYPERELASTIC)
    faces = space.Get_list_groupElem(2)[0]
    iso = element_mesh(3, ElemType.TETRA10, layers=2)
    distance = np.linalg.norm(space.coord[:, None] - iso.coord, axis=2)
    rows = np.argmin(distance, axis=1)
    return (
        faces,
        iso.Get_list_groupElem(2)[0],
        (rows[:, None] * 3 + np.arange(3)).ravel(),
    )


def assembled(groupElem: _GroupElem, A_e: np.ndarray, size: int) -> np.ndarray:
    """Element vectors (Ne, ndof) or matrices (Ne, ndof, ndof) summed into a global (size,) or (size, size) array."""
    dofs = groupElem.Get_assembly_e(A_e.shape[-1] // groupElem.nPe)
    if A_e.ndim == 2:
        out = np.zeros(size)
        np.add.at(out, dofs, A_e)
    else:
        out = np.zeros((size, size))
        np.add.at(out, (dofs[:, :, None], dofs[:, None, :]), A_e)
    return out


PRESSURE = Simulations.ProblemType("pressure")


def up_groups(
    dim: int, elemType: ElemType, order: int | None = None
) -> tuple[_GroupElem, _GroupElem, np.ndarray, np.ndarray]:
    """u's and p's groups on one organised element mesh, and their DOFs (dof_n = dim, 1): Taylor–Hood, p on the vertices of `elemType`, or, given `order`, u at that order over a linear `elemType` and p on its nodes."""
    mesh = element_mesh(dim, elemType)
    if order is None:
        mesh = mesh._With_FunctionSpace(PRESSURE, 1)
    else:
        mesh = mesh._With_FunctionSpace(HYPERELASTIC, order)
    groupElem = mesh.groupElem
    space_u = mesh._Get_FunctionSpace(HYPERELASTIC)
    space_p = mesh._Get_FunctionSpace(PRESSURE)
    return (
        space_u.Get_list_groupElem()[0],
        space_p.Get_list_groupElem()[0],
        space_u.Get_assembly_e(groupElem, dim),
        space_p.Get_assembly_e(groupElem, 1),
    )
