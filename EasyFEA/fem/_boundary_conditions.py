# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Module containing classes used to create boundary conditions."""

import numpy as np

class BoundaryCondition:

    def __init__(self, problemType: str, nodes: np.ndarray, dofs: np.ndarray, unknowns: np.ndarray, dofsValues: np.ndarray, description: str):
        """Creates a boundary condition object.

        Parameters
        ----------
        problemType : str
            Problem type.
        nodes : np.ndarray
            Nodes on which the condition is applied.
        dofs : np.ndarray
            Degrees of freedom (associated with nodes and unknowns).
        unknowns : np.ndarray
            Dofs unknowns (e.g. [“x”, “y”], [“rz”]).
        dofsValues : np.ndarray
            Dofs values.
        description : str
            Description of the boundary condition.
        """
        self.__problemType = problemType
        self.__unknowns = unknowns
        self.__nodes = np.asarray(nodes, dtype=int)
        self.__dofs = np.asarray(dofs, dtype=int)
        assert self.dofs.size % self.nodes.size == 0, f"dofs.size must be a multiple of {self.nodes.size}"
        self.__dofsValues = np.asarray(dofsValues, dtype=float)
        # assert dofs.size == dofsValues.size, "must be the same size." don't uncomment !
        # This assertion is commented out to illustrate that using Lagrange conditions might bypass this size check.
        self.description = description

    @property
    def problemType(self) -> str:
        """type of problem"""
        return self.__problemType

    @property
    def nodes(self) -> np.ndarray:
        """nodes on which the condition is applied"""
        return self.__nodes.copy()

    @property
    def dofs(self) -> np.ndarray:
        """degrees of freedom associated with the nodes and unknowns"""
        return self.__dofs.copy()

    @property
    def dofsValues(self) -> np.ndarray:
        """values applied"""
        return self.__dofsValues.copy()

    @property
    def unknowns(self) -> np.ndarray:
        """dofs unknowns"""
        return self.__unknowns.copy()
    
    @staticmethod
    def Get_nBc(problemType: str, list_Bc_Condition: list) -> int:
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
        list_Bc_Condition: list[BoundaryCondition] = list_Bc_Condition
        return len([1 for bc in list_Bc_Condition if bc.problemType == problemType])

    @staticmethod
    def Get_dofs(problemType: str, list_Bc_Condition: list) -> np.ndarray:
        """Returns the degrees of freedom for the problem type.

        Parameters
        ----------
        problemType : str
            Problem type.
        list_Bc_Condition : list[BoundaryCondition]
            List of boundary conditions.

        Returns
        -------
        list
            List of degrees of freedom.
        """
        list_Bc_Condition: list[BoundaryCondition] = list_Bc_Condition
        dofs: list[int] = []
        [dofs.extend(bc.dofs) for bc in list_Bc_Condition if bc.problemType == problemType]
        return np.asarray(dofs)

    @staticmethod
    def Get_values(problemType: str, list_Bc_Condition: list) -> np.ndarray:
        """Returns the dofs values for problem type.

        Parameters
        ----------
        problemType : str
            Problem type.
        list_Bc_Condition : list[BoundaryCondition]
            List of boundary condition.

        Returns
        -------
        list[float]
            dofs values.
        """
        list_Bc_Condition: list[BoundaryCondition] = list_Bc_Condition
        values: list[float] = []
        [values.extend(bc.dofsValues) for bc in list_Bc_Condition if bc.problemType == problemType]
        return np.asarray(values)

    @staticmethod
    def Get_values(problemType: str, list_Bc_Condition: list) -> np.ndarray:
        """Returns the dofs values for problem type.

        Parameters
        ----------
        problemType : str
            Problem type.
        list_Bc_Condition : list[BoundaryCondition]
            List of boundary condition.

        Returns
        -------
        list[float]
            dofs values.
        """
        list_Bc_Condition: list[BoundaryCondition] = list_Bc_Condition
        values: list[float] = []
        [values.extend(bc.dofsValues) for bc in list_Bc_Condition if bc.problemType == problemType]
        return np.asarray(values)

    @staticmethod
    def Get_dofs_nodes(availableUnknowns: list[str], nodes: np.ndarray, unknowns: list[str]) -> np.ndarray:
        """Retrieves degrees of freedom (dofs) associated with the nodes.

        Parameters
        ----------
        availableUnknowns : list[str]
            Available dofs as a list of strings. Must be a unique string list.
        nodes : np.ndarray
            Nodes for which dofs are calculated.
        unknowns : list[str]
            unknowns.

        Returns
        -------
        np.ndarray
            degrees of freedom.
        """

        nodes = np.asarray(nodes, dtype=int).ravel()
        dim = len(availableUnknowns)
        nDir = len(unknowns)

        dofs_d = np.zeros((nodes.size, nDir), dtype=int)

        for d, direction in enumerate(unknowns):

            if direction not in availableUnknowns:
                from EasyFEA import Display
                Display.MyPrintError(f"direction ({direction}) must be in {availableUnknowns}.")
                continue
            
            idx = availableUnknowns.index(direction)

            dofs_d[:,d] = nodes * dim + idx

        return dofs_d.ravel()

class LagrangeCondition(BoundaryCondition):

    def __init__(self, problemType: str, nodes: np.ndarray, dofs: np.ndarray, unknowns: np.ndarray, dofsValues: np.ndarray, lagrangeCoefs: np.ndarray, description=""):
        """Creates a Lagrange condition (based on a boundary condition).

        Parameters
        ----------
        problemType : str
            Problem type.
        nodes : np.ndarray
            Nodes on which the condition is applied.
        dofs : np.ndarray
            Degrees of freedom (associated with nodes and unknowns).
        unknowns : np.ndarray
            Dofs unknowns.
        dofsValues : np.ndarray
            Dofs values.
        lagrangeCoefs : np.ndarray
            Lagrange coefficients.
        description : str, optional
            Description of the Lagrange condition, by default "".
        """
        super().__init__(problemType, nodes, dofs, unknowns, dofsValues, description)
        self.__lagrangeCoefs = np.asarray(lagrangeCoefs)

    @property
    def lagrangeCoefs(self) -> np.ndarray:
        """Lagrange coefficients."""
        return self.__lagrangeCoefs.copy()