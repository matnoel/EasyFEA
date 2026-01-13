# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing classes used to create boundary conditions."""

import numpy as np

from ..Utilities import _types


class BoundaryCondition:
    """Boundary condition."""

    def __init__(
        self,
        problemType: str,
        nodes: _types.IntArray,
        dofs: _types.IntArray,
        unknowns: list[str],
        dofsValues: _types.FloatArray,
        description: str,
    ):
        """Creates a boundary condition object.

        Parameters
        ----------
        problemType : str
            Problem type.
        nodes : _types.IntArray
            Nodes on which the condition is applied.
        dofs : _types.IntArray
            Degrees of freedom (associated with nodes and unknowns).
        unknowns : list[str]
            Dofs unknowns (e.g. [“x”, “y”], [“rz”]).
        dofsValues : _types.FloatArray
            Dofs values.
        description : str
            Description of the boundary condition.
        """
        self.__problemType = problemType
        self.__unknowns = unknowns
        self.__nodes = np.asarray(nodes, dtype=int)
        self.__dofs = np.asarray(dofs, dtype=int)
        assert (
            self.dofs.size % self.nodes.size == 0
        ), f"dofs.size must be a multiple of {self.nodes.size}"
        self.__dofsValues = np.asarray(dofsValues, dtype=float)
        # assert dofs.size == dofsValues.size, "must be the same size." don't uncomment !
        # This assertion is commented out to illustrate that using Lagrange conditions might bypass this size check.
        self.description = description

    @property
    def problemType(self) -> str:
        """type of problem"""
        return self.__problemType

    @property
    def nodes(self) -> _types.IntArray:
        """nodes on which the condition is applied"""
        return self.__nodes.copy()

    @property
    def dofs(self) -> _types.IntArray:
        """degrees of freedom associated with the nodes and unknowns"""
        return self.__dofs.copy()

    @property
    def dofsValues(self) -> _types.FloatArray:
        """values applied"""
        return self.__dofsValues.copy()

    @property
    def unknowns(self) -> list[str]:
        """dofs unknowns"""
        return self.__unknowns.copy()

    @staticmethod
    def Get_nBc(problemType: str, list_Bc_Condition: list["BoundaryCondition"]) -> int:
        """Returns the number of conditions for the problem type.

        Parameters
        ----------
        problemType : str
            Problem type.
        list_Bc_Condition : list[BoundaryCondition]
            List of boundary conditions.

        Returns
        -------
        int
            Number of boundary conditions (nBc).
        """
        return len([1 for bc in list_Bc_Condition if bc.problemType == problemType])

    @staticmethod
    def Get_dofs(
        problemType: str, list_Bc_Condition: list["BoundaryCondition"]
    ) -> _types.IntArray:
        """Returns the degrees of freedom for the problem type.

        Parameters
        ----------
        problemType : str
            Problem type.
        list_Bc_Condition : list[BoundaryCondition]
            List of boundary conditions.

        Returns
        -------
        _types.IntArray
            degrees of freedom.
        """
        dofs: list[int] = []
        [
            dofs.extend(bc.dofs)  # type: ignore [func-returns-value]
            for bc in list_Bc_Condition
            if bc.problemType == problemType
        ]
        return np.asarray(dofs, dtype=int)

    @staticmethod
    def Get_values(
        problemType: str, list_Bc_Condition: list["BoundaryCondition"]
    ) -> _types.FloatArray:
        """Returns the dofs values for problem type.

        Parameters
        ----------
        problemType : str
            Problem type.
        list_Bc_Condition : list[BoundaryCondition]
            List of boundary condition.

        Returns
        -------
        _types.FloatArray
            dofs values.
        """
        values: list[float] = []
        [
            values.extend(bc.dofsValues)  # type: ignore [func-returns-value]
            for bc in list_Bc_Condition
            if bc.problemType == problemType
        ]
        return np.asarray(values, dtype=float)

    @staticmethod
    def Get_dofs_nodes(
        availableUnknowns: list[str], nodes: _types.IntArray, unknowns: list[str]
    ) -> _types.IntArray:
        """Retrieves degrees of freedom (dofs) associated with the nodes.

        Parameters
        ----------
        availableUnknowns : list[str]
            Available dofs as a list of strings. Must be a unique string list.
        nodes : _types.IntArray
            Nodes for which dofs are calculated.
        unknowns : list[str]
            unknowns.

        Returns
        -------
        _types.IntArray
            degrees of freedom.
        """

        nodes = np.asarray(nodes, dtype=int).ravel()
        dim = len(availableUnknowns)
        nDir = len(unknowns)

        dofs_d = np.zeros((nodes.size, nDir), dtype=int)

        for d, direction in enumerate(unknowns):
            if direction not in availableUnknowns:
                from EasyFEA import Display

                Display.MyPrintError(
                    f"direction ({direction}) must be in {availableUnknowns}."
                )
                continue

            idx = availableUnknowns.index(direction)

            dofs_d[:, d] = nodes * dim + idx

        return dofs_d.ravel()


class LagrangeCondition(BoundaryCondition):
    """Lagrange boundary condition"""

    def __init__(
        self,
        problemType: str,
        nodes: _types.IntArray,
        dofs: _types.IntArray,
        unknowns: list[str],
        dofsValues: _types.FloatArray,
        lagrangeCoefs: _types.FloatArray,
        description: str = "",
    ):
        """Creates a Lagrange boundary condition.

        Parameters
        ----------
        problemType : str
            Problem type.
        nodes : _types.IntArray
            Nodes on which the condition is applied.
        dofs : _types.IntArray
            Degrees of freedom (associated with nodes and unknowns).
        unknowns : list[str]
            Dofs unknowns.
        dofsValues : _types.FloatArray
            Dofs values.
        lagrangeCoefs : _types.FloatArray
            Lagrange coefficients.
        description : str, optional
            Description of the Lagrange condition, by default "".
        """
        super().__init__(problemType, nodes, dofs, unknowns, dofsValues, description)
        self.__lagrangeCoefs = np.asarray(lagrangeCoefs)

    @property
    def lagrangeCoefs(self) -> _types.FloatArray:
        """Lagrange coefficients."""
        return self.__lagrangeCoefs.copy()
