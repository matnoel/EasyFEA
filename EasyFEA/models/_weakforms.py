# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the WeakForms class used to assemble arbitrary finite element matrices."""

from typing import Optional, Union

from ..FEM import Field, BiLinearForm, LinearForm

from ._utils import _IModel, ModelType
from ..Utilities import _params


class WeakForms(_IModel):
    r"""Class responsible for computing the finite element matrices used in the system :math:`\Krm \, \mathrm{u} + \Crm \, \vrm + \Mrm \, \arm = \Frm`."""

    thickness: float = _params.PositiveScalarParameter()

    def __init__(
        self,
        field: Field,
        computeK: BiLinearForm,
        computeC: Optional[BiLinearForm] = None,
        computeM: Optional[BiLinearForm] = None,
        computeF: Optional[LinearForm] = None,
        thickness: float = 1.0,
    ):
        r"""Creates a weak form manager responsible for computing the finite element matrices used in the system :math:`\Krm \, \mathrm{u} + \Crm \, \vrm + \Mrm \, \arm = \Frm`.

        Parameters
        ----------
        field : Field
            Finite element field u
        computeK : BiLinearForm
            Function used to build stiffness matrix K
        computeC : Optional[BiLinearForm], optional
            Function used to build damping matrix C, by default None
        computeM : Optional[BiLinearForm], optional
            Function used to build mass matrix M, by default None
        computeF : Optional[LinearForm], optional
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
    def computeK(self) -> Union[BiLinearForm, None]:
        r"""Function used to build stiffness matrix :math:`\Krm`."""
        return self.__computeK

    @property
    def computeC(self) -> Union[BiLinearForm, None]:
        r"""Function used to build stiffness matrix :math:`\Crm`."""
        return self.__computeC

    @property
    def computeM(self) -> Union[BiLinearForm, None]:
        r"""Function used to build stiffness matrix :math:`\Mrm`."""
        return self.__computeM

    @property
    def computeF(self) -> Union[LinearForm, None]:
        r"""Function used to build force vector :math:`\Frm`."""
        return self.__computeF

    @property
    def modelType(self) -> ModelType:
        return ModelType.weakForm

    @property
    def dim(self) -> int:
        return self.__field.dof_n
