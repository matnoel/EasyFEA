"""Module for creating boundary conditions."""

from typing import cast
import numpy as np

class BoundaryCondition:
    def __init__(self, problemType: str, nodes: np.ndarray, dofs: np.ndarray, directions: np.ndarray, dofsValues: np.ndarray, description: str):
        """Create a boundary condition object.

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
        self.__nodes = nodes
        self.__dofs = np.asarray(dofs, dtype=int)
        self.__dofsValues = np.asarray(dofsValues)
        self.description = description

    @property
    def problemType(self) -> str:
        """Type of the problem."""
        return self.__problemType

    @property
    def nodes(self) -> np.ndarray:
        """Nodes on which the condition is applied."""
        return self.__nodes

    @property
    def dofs(self) -> np.ndarray:
        """Degrees of freedom associated with nodes and directions."""
        return self.__dofs

    @property
    def dofsValues(self) -> np.ndarray:
        """Values applied to degrees of freedom."""
        return self.__dofsValues

    @property
    def directions(self) -> np.ndarray:
        """Associated directions."""
        return self.__directions
    
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
        dofs = []
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
        values = []
        [values.extend(bc.dofsValues) for bc in list_Bc_Condition if bc.problemType == problemType]
        return values

    @staticmethod
    def Get_dofs_nodes(param: int, problemType: str, nodes: np.ndarray, directions: list[str]) -> np.ndarray:
        """Get degrees of freedom associated with nodes based on the problem and directions.

        Parameters
        ----------
        param : int
            Problem parameter, beam -> nbddl_e, otherwise dim.
        problemType : str
            Problem type.
        nodes : np.ndarray
            Nodes.
        directions : list
            Directions.

        Returns
        -------
        np.ndarray
            Degrees of freedom.
        """
        from Simulations import ModelType

        if problemType in [ModelType.damage, ModelType.thermal]:
            return nodes.reshape(-1)
        elif problemType == ModelType.displacement:
            ddls_dir = np.zeros((nodes.shape[0], len(directions)), dtype=int)
            dim = param
            for d, direction in enumerate(directions):
                if direction == "x":
                    index = 0
                elif direction == "y":
                    index = 1
                elif direction == "z":
                    assert dim == 3, "A 2D study does not allow forces to be applied along z"
                    index = 2
                else:
                    "Unknown direction"
                ddls_dir[:, d] = nodes * dim + index
            return ddls_dir.reshape(-1)

        elif problemType == ModelType.beam:
            ddls_dir = np.zeros((nodes.shape[0], len(directions)), dtype=int)

            nbddl_e = param

            if nbddl_e == 1:
                dimModel = "1D"
            elif nbddl_e == 3:
                dimModel = "2D"
            elif nbddl_e == 6:
                dimModel = "3D"

            for d, direction in enumerate(directions):
                if direction == "x":
                    index = 0
                elif direction == "y":
                    if dimModel in ["2D", "3D"]:
                        index = 1
                    else:
                        raise Exception("A 2D or 3D Beam Study is required to access the following dofs y")
                elif direction == "z":
                    assert dimModel == "3D", "A 3D Beam Study is required to access the following dofs z"
                    index = 2
                elif direction == "rx":
                    if dimModel == "3D":
                        # 3D beam model
                        index = 3
                    else:
                        raise Exception("A 3D beam study is required to access the rx dofs.")
                elif direction == "ry":
                    if dimModel == "3D":
                        # 3D beam model
                        index = 4
                    else:
                        raise Exception("A 3D beam study is required to access the ry dofs.")
                elif direction == "rz":
                    if dimModel == "2D":
                        # 2D beam model
                        index = 2
                    elif dimModel == "3D":
                        # 3D beam model
                        index = 5
                    else:
                        raise Exception("A 2D or 3D beam study is required to access the rz dofs.")
                else:
                    raise Exception("Unknown direction")
                ddls_dir[:, d] = nodes * nbddl_e + index
            return ddls_dir.reshape(-1)
        else:
            print("Unknown problem")

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
