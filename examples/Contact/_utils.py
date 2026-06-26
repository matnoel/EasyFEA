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
        indenter: Mesh = self._contactMesh
        assert indenter is not None
        matrixType = MatrixType.mass
        for contactGroup in indenter.Get_list_groupElem(indenter.dim - 1):
            elements = (
                contactGroup.Get_Elements_Tag("contact")
                if "contact" in contactGroup.elementTags
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
                gap_e_pg, normal_e_pg = contactGroup._Get_gap_and_normal(
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
