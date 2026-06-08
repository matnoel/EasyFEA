from enum import Enum

import numpy as np

from EasyFEA import Display, Folder, PyVista, MatrixType, Models, Simulations, AlgoType
from EasyFEA.FEM import Operators, MatrixType
from EasyFEA.Utilities import _params

from utils import RESULTS_DIR, DATA_DIR, Get_config, Get_tau_and_pressure


class Config(str, Enum):
    D = "D"
    A = "A"
    B = "B"


class CardiacElastoDynamics(Simulations.HyperElastic):

    pressure = _params.PositiveScalarParameter()

    active_stress = _params.PositiveScalarParameter()

    def __init__(
        self, mesh, model, folder="", tolConv=0.00001, maxIter=20, verbosity=False
    ):
        super().__init__(mesh, model, folder, tolConv, maxIter, verbosity)

        self.pressure = 0.0
        self.active_stress = 0.0

    def Construct_local_matrix_system(self, problemType):
        results = super().Construct_local_matrix_system(problemType)

        # current Newton-Raphson iterate (updated via u += delta_u)
        displacement = self._Solver_Get_Newton_Raphson_current_solution()
        if self.algo in AlgoType.Get_Hyperbolic_Types():
            displacement, _, _ = self._Solver_Evaluate_u_v_a_for_time_scheme(
                problemType, displacement
            )

        for groupElem in self.mesh.Get_list_groupElem(self.dim - 1):

            tangent_e, residual_e = Operators.NonLinear.FollowingPressure(
                displacement,
                groupElem,
                self.pressure,
                groupElem.Get_Elements_Tag("3"),
                MatrixType.mass,
            )

            # Newton: A(u) Δu = -R(u) = -(F(u) - b) = -F(u) + b
            # Follower pressure is a `b` term that depends on u
            results[groupElem] = (tangent_e, None, None, residual_e)

        return results


if __name__ == "__main__":

    Display.Clear()

    # ----------------------------------------------
    # Config
    # ----------------------------------------------

    ellipsoid = "ellipsoid_0.01"

    config = Config.D

    results_dir = Folder.Join(RESULTS_DIR, ellipsoid + f"_{config.name}")

    plotGraph = False

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    mesh, fibers_e_pg, sheets_e_pg = Get_config(
        Folder.Join(DATA_DIR, ellipsoid),
        MatrixType.rigi,
        plotMesh=False,
        plotTags=False,
        plotFibers=False,
    )

    # ----------------------------------------------
    #
    # ----------------------------------------------

    t_values, tau_values, p_values = Get_tau_and_pressure(Tmax=1.0, Nt=50)
    if config is Config.B:
        tau_values *= 0
    if config is Config.A:
        p_values *= 0

    if plotGraph:
        ax = Display.Init_Axes()
        ax.grid()
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$\tau(t)$ [Pa]")
        ax.plot(t_values, tau_values)
        name = "active_pressure"
        Display.Save_fig(results_dir, name)

        ax = Display.Init_Axes()
        ax.grid()
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$p(t)$ [Pa]")
        ax.plot(t_values, p_values)
        name = "pressure"
        Display.Save_fig(results_dir, name)

    # ----------------------------------------------
    # Material
    # ----------------------------------------------

    # solid
    a = 59.0
    a_f = 18472.0
    a_fs = 216.0
    a_s = 2481.0
    b = 8.023
    b_f = 16.026
    b_fs = 11.436
    b_s = 11.12

    material = Models.HyperElastic.HolzapfelOgden(
        dim=3,
        C0=a / 2 / b,
        C1=b,
        C2=a_f / 2 / b_f,
        C3=b_f,
        C4=a_s / 2 / b_s,
        C5=b_s,
        C6=a_fs / 2 / b_fs,
        C7=b_fs,
        K=1e6,
        Mu1=0.0,
        Mu2=0.0,
        T1=fibers_e_pg,
        T2=sheets_e_pg,
        ks=100,
    )
    material.eta = 100.0

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    simu = CardiacElastoDynamics(mesh, material)

    dt = t_values[1] - t_values[0]
    simu.Solver_Set_Hyperbolic_Algorithm(dt, algo=AlgoType.midpoint)
    simu.rho = 1000

    nodes_dirichlet = mesh.Nodes_Tags("4")

    for t in t_values:
        simu.Bc_Init()
        simu.add_dirichlet(nodes_dirichlet, [0] * 3, simu.Get_unknowns())
        simu.pressure = np.interp(t + dt / 2, t_values, p_values)
        simu.Solve()
        simu.Save_Iter()

        # PyVista.Plot(simu, "ux", deformFactor=1, plotMesh=True).show()

    # PyVista.Movie_simu(simu, "uy", results_dir, "uy.mp4", deformFactor=1, plotMesh=True)

    Display.plt.show()
