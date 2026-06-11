from dataclasses import dataclass
from typing import Literal

import numpy as np

from EasyFEA import Folder, PyVista, MeshIO, MatrixType, Mesher, Models, Simulations
from EasyFEA.FEM import Mesh, ElemType, FeArray, Norm
from EasyFEA.Utilities._types import FloatArray, IntArray

RESULTS_DIR = Folder.Join(Folder.Dir(__file__), "results")

DATA_DIR = Folder.Join(Folder.Dir(__file__), "data")


@dataclass(frozen=True)
class EllipsoidGeometry:
    """Truncated-ellipsoid geometry + fibre-angle parameters of the cardiac-mechanics benchmark. Ported from cardiac_benchmark_toolkit.data.DEFAULTS so EasyFEA can build the fibre/sheet fields analytically, without the FEniCS-generated `fiber.vtu` / `sheet.vtu`."""

    R_SHORT_ENDO: float = 0.025  # endocardial short-axis radius [m]
    R_SHORT_EPI: float = 0.035  # epicardial  short-axis radius [m]
    R_LONG_ENDO: float = 0.09  # endocardial long-axis  radius [m]
    R_LONG_EPI: float = 0.097  # epicardial  long-axis  radius [m]
    ALPHA_ENDO: float = -60.0  # fibre helix angle on endocardium [deg]
    ALPHA_EPI: float = +60.0  # fibre helix angle on epicardium  [deg]


ELLIPSOID = EllipsoidGeometry()


def __Compute_transmural_distance(
    mesh: Mesh,
    endoNodes: IntArray,
    epiNodes: IntArray,
) -> FloatArray:
    """Solve the transmural Laplace problem `-div(grad t) = 0` with `t = 0` on the endocardium and `t = 1` on the epicardium. Returns the nodal field `t in [0, 1]` (shape `(Nn,)`), the normalised wall coordinate that drives the fibre rotation. Posed as a steady heat-conduction problem (`k = 1`, `c = 0`, no source), so it reduces to the same Laplace operator as `transmural_distance_problem` in cardiac_benchmark_toolkit."""
    thermalModel = Models.Thermal(k=1.0, c=0.0)
    simu = Simulations.Thermal(mesh, thermalModel)
    simu.add_dirichlet(endoNodes, [0.0], ["t"])
    simu.add_dirichlet(epiNodes, [1.0], ["t"])
    return simu.Solve()  # (Nn,)


def __Compute_fiber_directions(
    coord: FloatArray,
    td: FloatArray,
    geom: EllipsoidGeometry = ELLIPSOID,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Analytic fibre / sheet / sheet-normal triad at each node, vectorised over all nodes. Ported from the per-point `FiberExpression` / `SheetNormalExpression` / `SheetExpression` of cardiac_benchmark_toolkit; the local (e_u, e_v) frame comes from the derivative of the prolate-spheroidal parametrisation, the fibre is that frame rotated by the transmural-interpolated helix angle, the sheet-normal is `e_u x e_v`, and the sheet is `f0 x n0`.

    Works for any leading shape: pass nodal coordinates `(Nn, 3)` with `td (Nn,)`, or Gauss-point coordinates `(Ne, nPg, 3)` with `td (Ne, nPg)`.

    Parameters
    ----------
    coord : (..., 3) array — evaluation-point coordinates (nodes or Gauss points).
    td : (...) array — transmural distance in [0, 1] (0 = endo, 1 = epi), broadcasting with `coord[..., 0]`.

    Returns
    -------
    (f0, s0, n0) — each a `(..., 3)` unit-vector field (fibre, sheet, sheet-normal)."""

    x, y, z = coord[..., 0], coord[..., 1], coord[..., 2]

    # transmural interpolation of the ellipsoid radii
    r_s = geom.R_SHORT_ENDO + (geom.R_SHORT_EPI - geom.R_SHORT_ENDO) * td
    r_l = geom.R_LONG_ENDO + (geom.R_LONG_EPI - geom.R_LONG_ENDO) * td

    # prolate-spheroidal-like angles (u, v) from the Cartesian coordinates
    u = np.arctan2(np.sqrt(y * y + z * z) / r_s, x / r_l)
    v = np.where(u < 1e-7, 0.0, np.pi - np.arctan2(z, -y))

    cos_u, sin_u = np.cos(u), np.sin(u)
    cos_v, sin_v = np.cos(v), np.sin(v)

    # columns e_u, e_v of the derivative matrix D(e_t, e_u, e_v); e_t (first column) is unused
    e_u = np.stack([-r_l * sin_u, r_s * cos_u * cos_v, r_s * cos_u * sin_v], axis=-1)
    e_v = np.stack(
        [np.zeros_like(u), -r_s * sin_u * sin_v, r_s * sin_u * cos_v], axis=-1
    )
    e_u /= np.linalg.norm(e_u, axis=-1, keepdims=True)
    e_v /= np.linalg.norm(e_v, axis=-1, keepdims=True)

    # rotate (e_u, e_v) by the transmural-interpolated helix angle to get the fibre
    alpha = (geom.ALPHA_ENDO + (geom.ALPHA_EPI - geom.ALPHA_ENDO) * td) * np.pi / 180
    f0 = np.sin(alpha)[..., None] * e_u + np.cos(alpha)[..., None] * e_v

    n0 = np.cross(e_u, e_v)
    n0 /= np.linalg.norm(n0, axis=-1, keepdims=True)

    s0 = np.cross(f0, n0)
    s0 /= np.linalg.norm(s0, axis=-1, keepdims=True)

    return f0, s0, n0


def _fibers_from_vtu(
    mesh: Mesh,
    path: str,
    matrixType: MatrixType,
) -> tuple[FeArray, FeArray]:
    """`fiber.vtu` / `sheet.vtu` source: read the FEniCS-generated nodal `f0` / `s0`, interpolate them to the Gauss points, then Gram-Schmidt to restore `f _|_ s` (interpolation through the shape functions bends the alignment). `mesh.nodes` are the rank-local owned indices; the unowned rows stay at zero, as Locates_sol_e expects."""
    groupElem3D = mesh.Get_list_groupElem(3)[0]
    nodes = mesh.nodes

    fibers_n = np.zeros((groupElem3D.Ncoords, 3))
    sheets_n = np.zeros_like(fibers_n)
    fiberFile = Folder.Join(path, "fiber.vtu")
    fibers_n[nodes] = MeshIO.meshio.vtu.read(fiberFile).point_data["f0"][nodes]
    sheetFile = Folder.Join(path, "sheet.vtu")
    sheets_n[nodes] = MeshIO.meshio.vtu.read(sheetFile).point_data["s0"][nodes]

    N_pg = mesh.groupElem.Get_N_pg(matrixType)
    fibers_e_pg = FeArray(np.einsum("pin,end->epd", N_pg, mesh.Locates_sol_e(fibers_n)))
    sheets_e_pg = FeArray(np.einsum("pin,end->epd", N_pg, mesh.Locates_sol_e(sheets_n)))

    # re-normalise the interpolated fibre (needed for higher-order elements where N_pg bends its magnitude), then project it out of the sheet
    fibers_e_pg = fibers_e_pg / Norm(fibers_e_pg, axis=-1)
    sheets_e_pg -= (
        (fibers_e_pg @ sheets_e_pg) / Norm(fibers_e_pg, axis=-1) ** 2 * fibers_e_pg
    )

    return fibers_e_pg, sheets_e_pg


def _fibers_analytic(
    mesh: Mesh,
    endoNodes: IntArray,
    epiNodes: IntArray,
    matrixType: MatrixType,
) -> tuple[FeArray, FeArray]:
    """Analytic source: solve the transmural distance, interpolate it to the Gauss points, then evaluate the analytic triad directly at the Gauss-point coordinates. The frame is exact and orthonormal at every quadrature point, so no node->Gauss interpolation of the vectors and no Gram-Schmidt are needed."""
    td_n = __Compute_transmural_distance(mesh, endoNodes, epiNodes)

    N_pg = mesh.groupElem.Get_N_pg(matrixType)
    coord_e_pg = np.asarray(
        mesh.groupElem.Get_GaussCoordinates_e_pg(matrixType)
    )  # (Ne, nPg, 3)
    td_e_pg = np.einsum("pin,en->ep", N_pg, mesh.Locates_sol_e(td_n))  # (Ne, nPg)

    fibers_e_pg, sheets_e_pg, _ = __Compute_fiber_directions(coord_e_pg, td_e_pg)

    return FeArray(fibers_e_pg), FeArray(sheets_e_pg)


def Get_config(
    path: str,
    matrixType: MatrixType = MatrixType.rigi,
    fiberSource: Literal["vtu", "analytic"] = "vtu",
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

    # fiber + sheet on the Gauss points -------------------------
    if fiberSource == "vtu":
        fibers_e_pg, sheets_e_pg = _fibers_from_vtu(mesh, path, matrixType)
    elif fiberSource == "analytic":
        fibers_e_pg, sheets_e_pg = _fibers_analytic(
            mesh,
            mesh.Nodes_Tags("endo"),
            mesh.Nodes_Tags("epi"),
            matrixType,
        )
    else:
        raise ValueError(
            f"Unknown fiberSource '{fiberSource}' (expected 'vtu' or 'analytic')."
        )

    # f _|_ s at every Gauss point (analytic: by construction; vtu: after Gram-Schmidt)
    assert np.all(np.abs(fibers_e_pg @ sheets_e_pg) < 1e-12)

    if plotFibers:
        coord = np.asarray(
            mesh.groupElem.Get_GaussCoordinates_e_pg(matrixType)
        ).reshape(-1, 3)
        plotter = PyVista.Plot(mesh, color="gray", alpha=0.1)
        coef = mesh.Get_meshSize().mean() * 0.5
        plotter.add_arrows(
            coord,
            np.asarray(fibers_e_pg).reshape(-1, 3),
            coef,
            color="r",
            label="fibers",
        )
        plotter.add_arrows(
            coord,
            np.asarray(sheets_e_pg).reshape(-1, 3),
            coef,
            color="b",
            label="sheets",
        )
        plotter.show_grid()
        plotter.add_legend()
        plotter.show()

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
