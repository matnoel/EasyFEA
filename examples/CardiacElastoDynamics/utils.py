import numpy as np

from EasyFEA import Folder, PyVista, MeshIO, MatrixType, Mesher
from EasyFEA.FEM import ElemType, FeArray, Norm

RESULTS_DIR = Folder.Join(Folder.Dir(__file__), "results")

DATA_DIR = Folder.Join(Folder.Dir(__file__), "data")


def Get_config(
    path: str,
    matrixType: MatrixType = MatrixType.rigi,
    plotMesh=False,
    plotTags=False,
    plotFibers=False,
):

    # Mesh -------------------------

    mesh = Mesher().Mesh_Import_mesh(Folder.Join(path, "mesh.msh"))

    if plotMesh:
        PyVista.Plot_Mesh(mesh).show()

    assert mesh.elemType == ElemType.TETRA4

    # tags 3d
    groupElem3D = mesh.Get_list_groupElem(3)[0]
    groupElem3D.Set_Tag(mesh.Nodes_Tags(["V1", "V2", "V3", "V4"]), "volume")
    # tags 2d
    groupElem2D = mesh.Get_list_groupElem(2)[0]
    groupElem2D.Set_Tag(mesh.Nodes_Tags(["S15", "S32", "S49", "S66"]), "epi")
    groupElem2D.Set_Tag(mesh.Nodes_Tags(["S22", "S39", "S56", "S73"]), "endo")
    groupElem2D.Set_Tag(mesh.Nodes_Tags(["S19", "S36", "S53", "S70"]), "top")

    if plotTags:
        PyVista.Plot_Tags(mesh, useColorCycler=True).show()

    # fiber + sheet -------------------------
    # `.vtu` data is sized to the GLOBAL node count; `mesh.nodes` is rank-local global indices.
    # We slice the vtu by those rank-local indices to populate the rank-local rows of fibers_n / sheets_n.
    fiberData = MeshIO.meshio.vtu.read(Folder.Join(path, "fiber.vtu"))
    fibers_n = np.zeros((groupElem3D.Ncoords, 3), dtype=float)
    nodes = mesh.nodes
    fibers_n[nodes] = fiberData.point_data["f0"][nodes]

    sheetData = MeshIO.meshio.vtu.read(Folder.Join(path, "sheet.vtu"))
    sheets_n = np.zeros_like(fibers_n)
    sheets_n[nodes] = sheetData.point_data["s0"][nodes]

    if plotFibers:
        plotter = PyVista.Plot(mesh, color="gray", alpha=0.1)
        coef = mesh.Get_meshSize().mean() * 0.5

        plotter.add_arrows(mesh.coord, fibers_n, coef, color="r", label="fibers")
        plotter.add_arrows(mesh.coord, sheets_n, coef, color="b", label="sheets")
        plotter.show_grid()
        plotter.add_legend()
        plotter.show()

    # on gauss points -------------------------

    # (Ne, nPe, 3)
    fibers_e = mesh.Locates_sol_e(fibers_n)
    sheets_e = mesh.Locates_sol_e(sheets_n)

    # (nPg, 1, nPe)
    # nPg = 15
    # N_pg = mesh.groupElem.Get_N_pg(nPg)
    N_pg = mesh.groupElem.Get_N_pg(matrixType)

    # interpolate on gauss points
    fibers_e_pg = FeArray(np.einsum("pin,end->epd", N_pg, fibers_e))
    sheets_e_pg = FeArray(np.einsum("pin,end->epd", N_pg, sheets_e))

    # Gram-Schidt
    sheets_e_pg -= (
        (fibers_e_pg @ sheets_e_pg) / Norm(fibers_e_pg, axis=-1) ** 2 * fibers_e_pg
    )

    dot_e_pg = fibers_e_pg @ sheets_e_pg
    assert np.any(np.abs(dot_e_pg) < 1e-12)

    return mesh, fibers_e_pg, sheets_e_pg


def Get_values(Tmax=1.0, Nt=100):
    """Returns t_values, activeStress_values, pressure_values"""

    # Create the activation function ---------------

    sig_0 = 1.5 * 1e5
    gamma = 0.005
    alpha_min = -30
    alpha_max = 5
    t_sys = 0.16
    t_dias = 0.484

    def get_f(t: float):
        Sp = 1 / 2 * (1 + np.tanh((t - t_sys) / gamma))
        Sm = 1 / 2 * (1 - np.tanh((t - t_dias) / gamma))
        return Sp * Sm

    def get_a(t: float):
        return alpha_max * get_f(t) + alpha_min * (1 - get_f(t))

    def dtau_dt(t: float, tau: float):
        return -np.abs(get_a(t)) * tau + sig_0 * np.max([get_a(t), 0])

    def get_tau(tf: float, N: int):
        dt = tf / (N + 1)
        t_values = np.linspace(0, tf, (N + 1))
        tau_values = np.zeros_like(t_values)
        tau_values[0] = 0.0

        for i in range(tau_values.size - 1):
            tau_values[i + 1] = tau_values[i] + dt * dtau_dt(t_values[i], tau_values[i])

        return t_values, tau_values

    t_values, activeStress_values = get_tau(Tmax, Nt)

    # Create the pressure function -----------------

    alpha_min = -30
    alpha_max = 5
    alpha_pre = 5
    alpha_mid = 1
    sig_pre = 7000
    sig_mid = 16000
    t_sys_pre = 0.17
    t_dias_pre = 0.484
    gamma = 0.005

    def get_f_pre(t: float):
        Sp = 1 / 2 * (1 + np.tanh((t - t_sys_pre) / gamma))
        Sm = 1 / 2 * (1 - np.tanh((t - t_dias_pre) / gamma))
        return Sp * Sm

    def get_g_pre(t: float):
        Sm = 1 / 2 * (1 - np.tanh((t - t_dias_pre) / gamma))
        return Sm

    def get_a_pre(t: float):
        return alpha_max * get_f_pre(t) + alpha_min * (1 - get_f_pre(t))

    def get_b(
        t: float,
    ):
        return get_a_pre(t) + alpha_pre * get_g_pre(t) + alpha_mid

    def dp_dt(t: float, p: float):
        return (
            -np.abs(get_b(t)) * p
            + sig_mid * np.max([get_b(t), 0])
            + sig_pre * np.max([get_g_pre(t), 0])
        )

    def get_p(tf: float, N: int):
        dt = tf / (N + 1)
        t_values = np.linspace(0, tf, (N + 1))
        p_values = np.zeros_like(t_values)
        p_values[0] = 0.0

        for i in range(p_values.size - 1):
            p_values[i + 1] = p_values[i] + dt * dp_dt(t_values[i], p_values[i])

        return t_values, p_values

    _, pressure_values = get_p(Tmax, Nt)

    return t_values, activeStress_values, pressure_values
