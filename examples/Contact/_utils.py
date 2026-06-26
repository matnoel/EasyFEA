# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np

from EasyFEA import Simulations
from EasyFEA.FEM import Operators, Mesh, MatrixType, LagrangeCondition
from EasyFEA.Utilities import _params


class RigidContact(Simulations.Elastic):
    """Small-strain elasticity + rigid penalty contact, solved with Newton-Raphson.

    Subclasses :class:`Simulations.Elastic` (so all stress/strain results stay available) but switches the solver to Newton: each volume group contributes the constant elastic tangent ``K`` and the internal-force residual ``-K·u``; the body's contact surface adds the penalty contact tangent/residual, with the gap/normal obtained against the rigid obstacle mesh ``_contactMesh``.
    """

    penalty = _params.ScalarParameter()

    def __init__(self, mesh, model, penalty, **kwargs):
        super().__init__(mesh, model, **kwargs)
        self._Solver_Set_Newton_Raphson_Algorithm(tolConv=1e-5, maxIter=50)
        self.penalty = penalty
        self._contactMesh: Mesh = None

    def Construct_local_matrix_system(self, problemType):
        u = self._Solver_Get_Newton_Raphson_current_solution()
        thickness = self.material.thickness if self.dim == 2 else 1.0
        out = {}

        # bulk: elastic tangent K and internal-force residual -K·u (Newton: A Δu = -R)
        for indenterGroup in self.mesh.Get_list_groupElem(self.dim):
            K_e = thickness * Operators.Bilinear.LinearizedElasticity(
                groupElem=indenterGroup,
                C=self.material.C,
            )
            u_e = u[indenterGroup.Get_assembly_e(self.dim)]
            F_e = -np.einsum("eij,ej->ei", K_e, u_e, optimize=True)
            out[indenterGroup] = (K_e, None, None, F_e)

        # penalty contact: integrate over the body's "contact" surface (so it assembles onto the body dofs) with the gap/normal obtained by projecting its deformed Gauss points onto the rigid obstacle surface `_contactMesh`.
        indenter: Mesh = self._contactMesh
        matrixType = MatrixType.mass
        for indenterGroup in indenter.Get_list_groupElem(indenter.dim - 1):
            elements = (
                indenterGroup.Get_Elements_Tag("contact")
                if "contact" in indenterGroup.elementTags
                else None
            )
            for groupElem in self.mesh.Get_list_groupElem(self.dim - 1):

                # deformed contact-surface Gauss coordinates x = X + u
                N_pg = groupElem.Get_N_pg(matrixType)[:, 0, :]
                X_e_pg = groupElem.Get_GaussCoordinates_e_pg(matrixType)
                u_e = u.reshape(-1, self.dim)[groupElem.connect]
                x_e_pg = X_e_pg.copy()
                x_e_pg[..., : self.dim] += np.einsum("pn,enc->epc", N_pg, u_e)

                # project onto the obstacle surface -> outward normal + signed gap
                gap_e_pg, normal_e_pg = indenterGroup._Get_gap_and_normal(
                    x_e_pg,
                    elements=elements,
                    coord=indenter.center,
                    matrixType=matrixType,
                )

                Kc_e, Fc_e = Operators.NonLinear.PenaltyContact(
                    groupElem=groupElem,
                    penalty=self.penalty,
                    gap_e_pg=gap_e_pg,
                    normal_e_pg=normal_e_pg,
                    matrixType=matrixType,
                )
                out[groupElem] = (thickness * Kc_e, None, None, thickness * Fc_e)

        return out


class MutliBodyContact(Simulations.Elastic):
    """Two-deformable-body frictionless contact via Lagrange-multiplier ties.

    Each contact node of the *contactor* surface is tied (node-to-segment, along the
    contact normal) to the *target* surface, enforcing non-penetration EXACTLY through
    the bordered saddle-point system: no penalty parameter, no penetration, and the two
    bodies' dofs are coupled exactly (unlike the block-diagonal penalty operator, whose
    missing cross-term wrecks Newton convergence). The problem stays linear elastic.

    Call :meth:`Set_contact` once, then :meth:`Add_contact_conditions` after each
    ``Bc_Init`` (which clears the Lagrange conditions). The ties are bilateral, so this
    suits (near-)full, compressive contact (e.g. a flat punch); releasing on separation
    would need an active-set loop on top.
    """

    def __init__(self, mesh, model, **kwargs):
        super().__init__(mesh, model, **kwargs)
        self._ties: list[tuple] = []  # (contactor node, t1, t2, N1, N2)
        self._direction = "y"

    def Set_contact(self, contactorTag: str, targetTag: str, direction: str = "y"):
        """Pair each contactor contact-node with the target segment beneath it.

        ``contactorTag`` / ``targetTag`` are the two body tags (e.g. "indenter" / "body");
        ``direction`` is the contact normal ("y" for a horizontal interface, 2D).
        """
        self._direction = direction
        tangent = 0 if direction != "x" else 1  # in-plane span axis (2D)
        seg = self.mesh.Get_list_groupElem(self.dim - 1)[0]
        coord = self.mesh.coord

        in_contact = np.zeros(seg.Ne, dtype=bool)
        in_contact[seg.Get_Elements_Tag("contact")] = True
        on_target = np.isin(seg.connect, seg.Get_Nodes_Tag(targetTag)).all(axis=1)
        target_edges = seg.connect[np.where(in_contact & on_target)[0]]  # (Ne_t, 2)

        contact_nodes = set(seg.Get_Nodes_Tag("contact").tolist())
        contactor_nodes = [
            n for n in seg.Get_Nodes_Tag(contactorTag).tolist() if n in contact_nodes
        ]

        self._ties = []
        for c in contactor_nodes:
            sc = coord[c, tangent]
            for t1, t2 in target_edges:
                s1, s2 = coord[t1, tangent], coord[t2, tangent]
                if s1 == s2:
                    continue
                lo, hi = sorted((s1, s2))
                if lo - 1e-9 <= sc <= hi + 1e-9:
                    N2 = (sc - s1) / (s2 - s1)  # linear shape-function weight of t2
                    self._ties.append((c, int(t1), int(t2), 1.0 - N2, N2))
                    break

    def Add_contact_conditions(self):
        """(Re)add the contact Lagrange ties; call after ``Bc_Init`` each load step."""
        for c, t1, t2, N1, N2 in self._ties:
            nodes = np.array([c, t1, t2])
            dofs = self.Bc_dofs_nodes(nodes, [self._direction])
            # u_c·n - N1 u_t1·n - N2 u_t2·n = 0  (non-penetration along the normal)
            self._Bc_Add_Lagrange(
                LagrangeCondition(
                    self.problemType,
                    nodes,
                    dofs,
                    [self._direction],
                    np.array([0.0]),
                    np.array([1.0, -N1, -N2]),
                )
            )
