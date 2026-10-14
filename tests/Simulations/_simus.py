# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Simulations shared by these tests and by IO and Viz, which save and draw them."""

from EasyFEA import Mesh, Models, Simulations
from EasyFEA.FEM import BiLinearForm, Field

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
