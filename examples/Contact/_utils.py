# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np

from EasyFEA import Simulations
from EasyFEA.FEM import Operators, Mesh, MatrixType
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
        for groupElem in self.mesh.Get_list_groupElem(self.dim):
            K_e = thickness * Operators.Bilinear.LinearizedElasticity(
                groupElem=groupElem,
                C=self.material.C,
            )
            u_e = u[groupElem.Get_assembly_e(self.dim)]
            F_e = -np.einsum("eij,ej->ei", K_e, u_e, optimize=True)
            out[groupElem] = (K_e, None, None, F_e)

        # penalty contact: integrate over the body's "contact" surface (so it assembles onto the body dofs) with the gap/normal obtained by projecting its deformed Gauss points onto the rigid obstacle surface `_contactMesh`.
        obstacle: Mesh = self._contactMesh
        matrixType = MatrixType.mass
        for obstacleGroup in obstacle.Get_list_groupElem(obstacle.dim - 1):
            obstacleElems = (
                obstacleGroup.Get_Elements_Tag("contact")
                if "contact" in obstacleGroup.elementTags
                else None
            )
            for bodyGroup in self.mesh.Get_list_groupElem(self.dim - 1):

                # deformed contact-surface Gauss coordinates x = X + u
                N_pg = bodyGroup.Get_N_pg(matrixType)[:, 0, :]
                X_e_pg = bodyGroup.Get_GaussCoordinates_e_pg(matrixType)
                u_e = u.reshape(-1, self.dim)[bodyGroup.connect]
                x_e_pg = X_e_pg.copy()
                x_e_pg[..., : self.dim] += np.einsum("pn,enc->epc", N_pg, u_e)

                # project onto the obstacle surface -> outward normal + signed gap
                gap_e_pg, normal_e_pg = obstacleGroup._Get_gap_and_normal(
                    x_e_pg,
                    elements=obstacleElems,
                    coord=obstacle.center,
                    matrixType=matrixType,
                )

                Kc_e, Fc_e = Operators.NonLinear.PenaltyContact(
                    groupElem=bodyGroup,
                    penalty=self.penalty,
                    gap_e_pg=gap_e_pg,
                    normal_e_pg=normal_e_pg,
                    matrixType=matrixType,
                )
                out[bodyGroup] = (thickness * Kc_e, None, None, thickness * Fc_e)

        return out
        # each pass onto its own dofs. Both surfaces live in the merged SEG2 group,
        # split by the "body"/"indenter" tags. Half penalty -> combined stiffness ~ penalty.
        seg = self.mesh.Get_list_groupElem(dim - 1)[0]
        matrixType = MatrixType.mass

        # nodal displacement (Nn, 3) used to deform the projected-onto surface
        U = np.zeros((self.mesh.Nn, 3))
        U[:, :dim] = u.reshape(self.mesh.Nn, dim)

        # split the "contact" interface edges into the two bodies' sides
        connect = seg.connect
        in_contact = np.zeros(seg.Ne, dtype=bool)
        in_contact[seg.Get_Elements_Tag("contact")] = True
        on_body = np.isin(connect, seg.Get_Nodes_Tag("body")).all(axis=1)
        on_indenter = np.isin(connect, seg.Get_Nodes_Tag("indenter")).all(axis=1)
        body_contact = np.where(in_contact & on_body)[0]
        indenter_contact = np.where(in_contact & on_indenter)[0]

        # interior reference of each body, to orient the contact normals outward
        coord = self.mesh.coord
        body_center = coord[self.mesh.Nodes_Tags(["body"])].mean(axis=0)
        indenter_center = coord[self.mesh.Nodes_Tags(["indenter"])].mean(axis=0)

        # deformed Gauss coordinates x = X + u of a set of interface edges
        N_pg = seg.Get_N_pg(matrixType)[:, 0, :]
        X_e_pg = seg.Get_GaussCoordinates_e_pg(matrixType)

        def deformed(elems):
            x = X_e_pg[elems].copy()
            x[..., :dim] += np.einsum("pn,enc->epc", N_pg, U[connect[elems]])[..., :dim]
            return x

        Kc = Fc = None
        for query, target, ref in [
            (body_contact, indenter_contact, indenter_center),
            (indenter_contact, body_contact, body_center),
        ]:
            if query.size == 0 or target.size == 0:
                continue
            # project the deformed query surface onto the deformed target surface
            gap_e_pg, normal_e_pg = seg._Get_gap_and_normal(
                deformed(query),
                elements=target,
                coord=ref,
                matrixType=matrixType,
                displacementMatrix=U,
            )
            Kc_e, Fc_e = Operators.NonLinear.PenaltyContact(
                groupElem=seg,
                penalty=0.5 * self.penalty,
                gap_e_pg=gap_e_pg,
                normal_e_pg=normal_e_pg,
                elements=query,
                matrixType=matrixType,
            )
            Kc = Kc_e if Kc is None else Kc + Kc_e
            Fc = Fc_e if Fc is None else Fc + Fc_e

        if Kc is not None:
            out[seg] = (thickness * Kc, None, None, thickness * Fc)

        return out
