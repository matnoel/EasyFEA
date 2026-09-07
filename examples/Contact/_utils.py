# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Optional

import numpy as np

from EasyFEA import Simulations
from EasyFEA.FEM import Operators, Mesh, MatrixType, _GroupElem
from EasyFEA.FEM._linalg import FeArray
from EasyFEA.Utilities import _params, _types
from EasyFEA.Simulations._terms import Term


class RigidContact(Simulations.Elastic):
    """Small-strain elasticity + rigid penalty contact, solved with Newton-Raphson.

    Subclasses :class:`Simulations.Elastic` (so all stress/strain results stay available) but switches the solver to Newton: each volume group contributes the constant elastic tangent ``K`` and the internal-force residual ``-K·u``; the body's contact surface adds the penalty contact tangent/residual, with the gap/normal obtained against the rigid obstacle mesh ``_contactMesh``.
    """

    penalty = _params.ScalarParameter()

    def __init__(self, mesh, model, penalty, **kwargs):
        super().__init__(mesh, model, **kwargs)
        self._Solver_Set_Newton_Raphson_Algorithm(absTol=1e-5, maxIter=50)
        self.penalty = penalty
        self._contactMesh: Mesh = None

    def Get_terms(self, problemType=None) -> list[Term]:
        return super().Get_terms(problemType) + [
            Term("KR", self.__Contact, dim=self.dim - 1)
        ]

    def __Contact(
        self, groupElem: _GroupElem, elements: Optional[_types.IntArray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Penalty-contact tangent/force on one surface group of the body.

        Integrating over the body's surface is what makes the contribution assemble onto the body's dofs; the gap and outward normal come from projecting that group's deformed Gauss points onto the rigid obstacle `_contactMesh`.
        """
        indenter: Mesh = self._contactMesh
        assert indenter is not None
        matrixType = MatrixType.mass

        u = self.Get_u_v_a()[0]

        # deformed contact-surface Gauss coordinates x = X + u
        N_pg = groupElem.Get_N_pg(matrixType)[:, 0, :]
        x_e_pg = groupElem.Get_GaussCoordinates_e_pg(matrixType).copy()
        u_e = u.reshape(-1, self.dim)[groupElem.connect]
        x_e_pg[..., : self.dim] += FeArray.asfearray(
            np.einsum("pn,enc->epc", N_pg, u_e)
        )

        K_e, F_e = 0.0, 0.0
        for contactGroup in indenter.Get_list_groupElem(indenter.dim - 1):
            # `obstacle` indexes the *indenter*'s elements — which faces to project onto — so it is not this term's `elements`, which would index the body's surface group.
            obstacle = (
                contactGroup.Get_Elements_Tag("contact")
                if "contact" in contactGroup.elementTags
                else None
            )

            # project onto the obstacle surface -> outward normal + signed gap
            gap_e_pg, normal_e_pg = contactGroup._Get_gap_and_normal(
                x_e_pg,
                elements=obstacle,
                coord=indenter.center,
                matrixType=matrixType,
            )

            Kc_e, Fc_e = Operators.NonLinear.PenaltyContact(
                groupElem=groupElem,
                penalty=self.penalty,
                gap_e_pg=gap_e_pg,
                normal_e_pg=normal_e_pg,
                elements=elements,
                matrixType=matrixType,
            )
            # several obstacle groups all press on the same body surface, so they add up
            K_e, F_e = K_e + Kc_e, F_e + Fc_e

        return K_e, F_e
