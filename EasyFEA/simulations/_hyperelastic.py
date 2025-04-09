# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information

# utilities
from ..utilities import Tic
# fem
from ..fem import Mesh, MatrixType
# materials
from ..materials import ModelType, Reshape_variable
from ..materials._hyperelastic_laws import _HyperElas
# simu
from ._simu import _Simu
from .Solvers import AlgoType

class HyperElasticSimu(_Simu):

    def __init__(self, mesh: Mesh, model: _HyperElas, verbosity=True, useNumba=True, useIterativeSolvers=True):
        """Creates a simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh used.
        model : _HyperElas
            The hyperelatic model used.
        verbosity : bool, optional
            If True, the simulation can write in the terminal. Defaults to True.
        useNumba : bool, optional
            If True and numba is installed numba can be used. Defaults to True.
        useIterativeSolvers : bool, optional
            If True, iterative solvers can be used. Defaults to True.
        """

        super().__init__(mesh, model, verbosity, useNumba, useIterativeSolvers)

        # init
        self.Solver_Set_Elliptic_Algorithm()

    # --------------------------------------------------------------------------
    # General
    # --------------------------------------------------------------------------

    def Get_problemTypes(self):
        return [ModelType.hyperelastic]
    
    def Get_dofs(self, problemType=None) -> list[str]:
        dict_dim_directions = {
            2 : ["x", "y"],
            3 : ["x", "y", "z"]
        }
        return dict_dim_directions[self.dim]
    
    def Get_dof_n(self, problemType=None) -> int:
        return self.dim
    
    # --------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------- 

    def Get_K_C_M_F(self, problemType=None):
        return super().Get_K_C_M_F(problemType)
    
    def Get_x0(self, problemType=None):
        return super().Get_x0(problemType)
    
    def Assembly(self):
        return super().Assembly()

    # --------------------------------------------------------------------------
    # Iterations
    # --------------------------------------------------------------------------

    def Save_Iter(self):
        return super().Save_Iter()
    
    def Set_Iter(self, iter = -1, resetAll=False):
        return super().Set_Iter(iter, resetAll)

    # --------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------- 

    def Results_Available(self):
        return super().Results_Available()
    
    def Result(self, option, nodeValues=True, iter=None):
        return super().Result(option, nodeValues, iter)
    
    def Results_Iter_Summary(self):
        return super().Results_Iter_Summary()
    
    def Results_dict_Energy(self):
        return super().Results_dict_Energy()
    
    def Results_displacement_matrix(self):
        return super().Results_displacement_matrix()
    
    def Results_nodesField_elementsField(self, details=False):
        return super().Results_nodesField_elementsField(details)