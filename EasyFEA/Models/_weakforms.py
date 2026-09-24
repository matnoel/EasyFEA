# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the WeakForms class used to assemble arbitrary finite element matrices."""

from ..FEM import Field, BiLinearForm, LinearForm

from ._utils import _IModel
from ..Utilities import _params


class WeakForms(_IModel):
    r"""Class responsible for computing the finite element matrices used in the system :math:`\Krm \, \mathrm{u} + \Crm \, \vrm + \Mrm \, \arm = \Frm`."""

    thickness: float = _params.PositiveScalarParameter()

    def __init__(
        self,
        field: Field,
        computeK: BiLinearForm,
        computeC: BiLinearForm | None = None,
        computeM: BiLinearForm | None = None,
        computeF: LinearForm | None = None,
        thickness: float = 1.0,
    ):
        r"""Creates a weak form manager responsible for computing the finite element matrices used in the system :math:`\Krm \, \mathrm{u} + \Crm \, \vrm + \Mrm \, \arm = \Frm`.

        Parameters
        ----------
        field : Field
            Finite element field u
        computeK : BiLinearForm
            Function used to build stiffness matrix K
        computeC : BiLinearForm | None, optional
            Function used to build damping matrix C, by default None
        computeM : BiLinearForm | None, optional
            Function used to build mass matrix M, by default None
        computeF : LinearForm | None, optional
            Function used to build force vector F, by default None
        thickness : float, optional
            thickness used in the model, by default 1.0
        """

        self.__field = field

        self.__computeK = computeK
        self.__computeC = computeC
        self.__computeM = computeM
        self.__computeF = computeF
        self.thickness = thickness

    @property
    def field(self) -> Field:
        """Finite element field."""
        return self.__field

    @property
    def computeK(self) -> BiLinearForm | None:
        r"""Function used to build stiffness matrix :math:`\Krm`."""
        return self.__computeK

    @property
    def computeC(self) -> BiLinearForm | None:
        r"""Function used to build stiffness matrix :math:`\Crm`."""
        return self.__computeC

    @property
    def computeM(self) -> BiLinearForm | None:
        r"""Function used to build stiffness matrix :math:`\Mrm`."""
        return self.__computeM

    @property
    def computeF(self) -> LinearForm | None:
        r"""Function used to build force vector :math:`\Frm`."""
        return self.__computeF

    @property
    def dim(self) -> int:
        return self.__field.dof_n
