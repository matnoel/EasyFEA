from enum import Enum

from EasyFEA import Display, Folder, PyVista, MatrixType, Models, Simulations

from utils import RESULTS_DIR, DATA_DIR, Get_config, Get_tau_and_pressure


class Config(str, Enum):
    D = "D"
    A = "A"
    B = "B"


if __name__ == "__main__":

    Display.Clear()

    # ----------------------------------------------
    # Config
    # ----------------------------------------------

    ellipsoid = "ellipsoid_0.01"

    config = Config.D

    results_dir = Folder.Join(RESULTS_DIR, ellipsoid + f"_{config.name}")

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

    t_values, tau_values, p_values = Get_tau_and_pressure(Tmax=1.0, Nt=100)
    if config is Config.B:
        tau_values *= 0
    if config is Config.A:
        p_values *= 0

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

    # updated_lua_dict["Solid"]["VolumicMass"]["value"] = 1000
    # updated_lua_dict["Solid"]["Viscosity"]["value"] = 100

    Display.plt.show()
