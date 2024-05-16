# Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Module containing classes used to create boundary conditions."""

import numpy as np

class BoundaryCondition:

    def __init__(self, problemType: str, nodes: np.ndarray, dofs: np.ndarray, directions: np.ndarray, dofsValues: np.ndarray, description: str):
        """Creates a boundary condition object.

        Parameters
        ----------
        problemType : str
            Type of the problem.
        nodes : np.ndarray
            Nodes on which the condition is applied.
        dofs : np.ndarray
            Degrees of freedom associated with nodes and directions.
        directions : np.ndarray
            Directions.
        dofsValues : np.ndarray
            Values applied to degrees of freedom.
        description : str
            Description of the boundary condition.
        """
        self.__problemType = problemType
        self.__directions = directions
        self.__nodes = np.asarray(nodes, dtype=int)
        self.__dofs = np.asarray(dofs, dtype=int)
        assert self.dofs.size % self.nodes.size == 0, f"dofs.size must be a multiple of {self.nodes.size}"
        self.__dofsValues = np.asarray(dofsValues, dtype=float)
        self.description = description

    @property
    def problemType(self) -> str:
        """Type of the problem."""
        return self.__problemType

    @property
    def nodes(self) -> np.ndarray:
        """Nodes on which the condition is applied."""
        return self.__nodes.copy()

    @property
    def dofs(self) -> np.ndarray:
        """Degrees of freedom associated with nodes and directions."""
        return self.__dofs.copy()

    @property
    def dofsValues(self) -> np.ndarray:
        """Values applied to degrees of freedom."""
        return self.__dofsValues.copy()

    @property
    def directions(self) -> np.ndarray:
        """Associated directions."""
        return self.__directions.copy()
    
    @staticmethod
    def Get_nBc(problemType: str, list_Bc_Condition: list) -> int:
        """Recovers the number of conditions applied to the problem.

        Parameters
        ----------
        problemType : str
            Type of the problem.
        list_Bc_Condition : list[BoundaryCondition]
            List of boundary conditions.

        Returns
        -------
        int
            nBc
        """
        list_Bc_Condition: list[BoundaryCondition] = list_Bc_Condition
        return len([1 for bc in list_Bc_Condition if bc.problemType == problemType])


    @staticmethod
    def Get_dofs(problemType: str, list_Bc_Condition: list) -> list[int]:
        """Get the degrees of freedom of the given problem and condition list.

        Parameters
        ----------
        problemType : str
            Type of the problem.
        list_Bc_Condition : list[BoundaryCondition]
            List of boundary conditions.

        Returns
        -------
        list
            Degrees of freedom.
        """
        list_Bc_Condition: list[BoundaryCondition] = list_Bc_Condition
        dofs: list[int] = []
        [dofs.extend(bc.dofs) for bc in list_Bc_Condition if bc.problemType == problemType]
        return dofs

    @staticmethod
    def Get_values(problemType: str, list_Bc_Condition: list) -> list[float]:
        """Get the values of degrees of freedom for the given problem and condition list.

        Parameters
        ----------
        problemType : str
            Type of the problem.
        list_Bc_Condition : list[BoundaryCondition]
            List of boundary conditions.

        Returns
        -------
        list
            Values of degrees of freedom.
        """
        list_Bc_Condition: list[BoundaryCondition] = list_Bc_Condition
        values: list[float] = []
        [values.extend(bc.dofsValues) for bc in list_Bc_Condition if bc.problemType == problemType]
        return values

    @staticmethod
    def Get_dofs_nodes(availableDirections: list[str], nodes: np.ndarray, directions: list[str]) -> np.ndarray:
        """Get degrees of freedom (dofs) associated with the nodes.

        Parameters
        ----------
        availableDirections : list[str]
            Directions available in the form of a string list. Must be a unique string list.
        nodes : np.ndarray
            Nodes.
        directions : list[str]
            Directions.

        Returns
        -------
        np.ndarray
            Degrees of freedom.
        """

        nodes = np.asarray(nodes, dtype=int).ravel()
        dim = len(availableDirections)
        nDir = len(directions)
        
        from EasyFEA import Display

        dofs_d = np.zeros((nodes.size, nDir), dtype=int)

        for d, direction in enumerate(directions):

            if direction not in availableDirections:
                Display.MyPrintError(f"direction ({direction}) must be in {availableDirections}.")
                continue
            
            idx = availableDirections.index(direction)

            dofs_d[:,d] = nodes * dim + idx

        return dofs_d.ravel()

class LagrangeCondition(BoundaryCondition):

    def __init__(self, problemType: str, nodes: np.ndarray, dofs: np.ndarray, directions: np.ndarray, dofsValues: np.ndarray, lagrangeCoefs: np.ndarray, description=""):
        """Construct a Lagrange condition based on a boundary condition.

        Parameters
        ----------
        problemType : str
            Type of the problem.
        nodes : np.ndarray
            Nodes on which the condition is applied.
        dofs : np.ndarray
            Degrees of freedom associated with nodes and directions.
        directions : np.ndarray
            Directions.
        dofsValues : np.ndarray
            Values applied to degrees of freedom.
        lagrangeCoefs : np.ndarray
            Lagrange coefficients.
        description : str, optional
            Description of the Lagrange condition, by default "".
        """
        super().__init__(problemType, nodes, dofs, directions, dofsValues, description)
        self.__lagrangeCoefs = np.asarray(lagrangeCoefs)

    @property
    def lagrangeCoefs(self) -> np.ndarray:
        """Lagrange coefficients."""
        return self.__lagrangeCoefs.copy()
