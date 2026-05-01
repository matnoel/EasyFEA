# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import matplotlib.pyplot as plt
import numpy as np
import pytest

from EasyFEA import Display, Models, Simulations, Mesher, ElemType
from EasyFEA.Geoms import Domain, Point, Line

TOL = (
    1e-7  # Single tolerance for all quantities (uy, rz, Mz_gp, Ty, N, ...).
    # The bottleneck is floating-point accumulation in high-degree shape function evaluation,
    # not the quantity type. Observed floors: SEG2 ~1e-13, SEG3 ~2e-11, SEG4 ~5e-11, SEG5 ~3e-9.
    # 1e-7 gives comfortable headroom above the worst case (SEG5).
)
ELEM_TYPES = ElemType.Get_1D()


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------


def _close_plots():
    plt.close("all")


def _rect_section(mesher: Mesher, b: float, h: float):
    return mesher.Mesh_2D(Domain(Point(-b / 2, -h / 2), Point(b / 2, h / 2)))


def _fixed_fixed_analytical(a: float, L: float, F: float, E: float, Iz: float):
    """Fixed-fixed beam with point force F at x=a (Macaulay formulation).

    Returns callables uy_x, Mz_x, Ty_x and support reactions Ra, Rb.
    Sign convention: F negative = downward; Mz positive = sagging.
    """
    b = L - a
    Ra = -F * b**2 * (3 * a + b) / L**3  # upward reaction at x=0
    Rb = -F * a**2 * (a + 3 * b) / L**3  # upward reaction at x=L
    Ma = F * a * b**2 / L**2  # fixed-end moment at x=0

    def uy_x(x):
        x = np.asarray(x, dtype=float)
        macaulay = np.where(x > a, (x - a) ** 3 / 6, 0.0)
        return (Ma / 2 * x**2 + Ra / 6 * x**3 + F * macaulay) / (E * Iz)

    def Mz_x(x):
        x = np.asarray(x, dtype=float)
        return np.where(x <= a, Ma + Ra * x, Ma + Ra * x + F * (x - a))

    def Ty_x(x):
        x = np.asarray(x, dtype=float)
        return np.where(x < a, -Ra, Rb)

    return uy_x, Mz_x, Ty_x, Ra, Rb


def _simply_supported_analytical(a: float, L: float, F: float, E: float, Iz: float):
    """Simply supported beam (pinned at x=0, roller at x=L) with point force F at x=a.

    Returns callables uy_x, Mz_x, Ty_x and support reactions Ra, Rb.
    Sign convention: F negative = downward; Mz positive = sagging.
    """
    Ra = -F * (L - a) / L  # upward reaction at x=0
    Rb = -F * a / L  # upward reaction at x=L
    # EI·uy = Ra·x³/6 + F·⟨x-a⟩³/6 + C1·x  (no fixed-end moment)
    # C1 from uy(L) = 0:
    C1 = -(Ra * L**2 / 6 + F * (L - a) ** 3 / (6 * L))

    def uy_x(x):
        x = np.asarray(x, dtype=float)
        macaulay = np.where(x > a, (x - a) ** 3 / 6, 0.0)
        return (Ra / 6 * x**3 + F * macaulay + C1 * x) / (E * Iz)

    def Mz_x(x):
        x = np.asarray(x, dtype=float)
        return np.where(x <= a, Ra * x, Ra * x + F * (x - a))

    def Ty_x(x):
        x = np.asarray(x, dtype=float)
        return np.where(x < a, -Ra, Rb)

    return uy_x, Mz_x, Ty_x, Ra, Rb


def _cantilever_distrib_analytical(L: float, q: float, E: float, Iz: float):
    """Cantilever (clamped at x=0, free at x=L) with uniform distributed load q in y.

    Returns callables uy_x, rz_x, Mz_x, Ty_x.
    Sign convention: q negative = downward; Mz positive = sagging.
    """

    def uy_x(x):
        x = np.asarray(x, dtype=float)
        return q * x**2 * (6 * L**2 - 4 * L * x + x**2) / (24 * E * Iz)

    def rz_x(x):
        x = np.asarray(x, dtype=float)
        return q * (x**3 / 6 - L * x**2 / 2 + L**2 * x / 2) / (E * Iz)

    def Mz_x(x):
        x = np.asarray(x, dtype=float)
        return q * (x - L) ** 2 / 2

    def Ty_x(x):
        x = np.asarray(x, dtype=float)
        return q * (L - x)

    return uy_x, rz_x, Mz_x, Ty_x


def _simply_supported_distrib_analytical(L: float, q: float, E: float, Iz: float):
    """Simply supported beam with uniform distributed load q in y.

    Returns callables uy_x, Mz_x, Ty_x.
    Sign convention: q negative = downward; Mz positive = sagging.
    """

    def uy_x(x):
        x = np.asarray(x, dtype=float)
        return q * x * (L**3 - 2 * L * x**2 + x**3) / (24 * E * Iz)

    def Mz_x(x):
        x = np.asarray(x, dtype=float)
        return q * x * (x - L) / 2

    def Ty_x(x):
        x = np.asarray(x, dtype=float)
        return q * (L / 2 - x)

    return uy_x, Mz_x, Ty_x


# -------------------------------------------------------
# Traction: axial bar with tip load and body force
# -------------------------------------------------------


@pytest.mark.parametrize("beamDim", [1, 2, 3])
@pytest.mark.parametrize("elemType", ELEM_TYPES)
def test_traction(elemType: str, beamDim: int):
    L, nL = 10.0, 10
    b, h = 0.1, 0.1
    E, v, ro, g = 200000e6, 0.3, 7800.0, 10.0
    P = 5000.0  # tip force in x (positive = tension)
    q = ro * g * b * h  # distributed axial body force (N/m)

    mesher = Mesher()
    section = _rect_section(mesher, b, h)
    A = section.area

    assert abs(A - b * h) <= 1e-12

    point1, point2 = Point(), Point(x=L)
    line = Line(point1, point2, L / nL)
    beam = Models.Beam.Isotropic(beamDim, line, section, E, v)

    mesh = mesher.Mesh_Beams([beam], elemType=elemType)
    structure = Models.Beam.BeamStructure([beam])
    simu = Simulations.Beam(mesh, structure, verbosity=False)

    simu.rho = ro
    expected_mass = L * b * h * ro
    assert abs(simu.mass - expected_mass) / expected_mass <= 1e-12

    simu.add_dirichlet(
        mesh.Nodes_Point(point1), [0] * simu.Get_dof_n(), simu.Get_unknowns()
    )
    simu.add_lineLoad(mesh.Nodes_Line(line), [q], ["x"])
    simu.add_neumann(mesh.Nodes_Point(point2), [P], ["x"])
    simu.Solve()

    x_n = mesh.coord[:, 0]
    x_e = x_n[mesh.connect].mean(1)

    # ux
    ux_x = lambda x: P * x / (E * A) + ro * g * x / (2 * E) * (2 * L - x)
    ux_fe = simu.Result("ux", nodeValues=True)
    err_ux = np.abs(ux_x(x_n) - ux_fe).max() / np.abs(ux_x(L))
    assert err_ux <= TOL, f"ux error {err_ux:.2e}"

    # N
    N_x = lambda x: P + q * (L - x)
    N_fe = simu.Result("N", nodeValues=False)
    err_N = np.abs(N_x(x_e) - N_fe).max() / np.abs(N_x(x_e)).max()
    assert err_N <= TOL, f"N error {err_N:.2e}"

    Display.Plot_BoundaryConditions(simu)
    _close_plots()


# -------------------------------------------------------
# Cantilever with tip load (dim >= 2)
# -------------------------------------------------------


@pytest.mark.parametrize("beamDim", [2, 3])
@pytest.mark.parametrize("elemType", ELEM_TYPES)
def test_cantilever_tip(elemType: str, beamDim: int):
    L, nL = 120.0, 10
    b, h = 13.0, 13.0
    E, v = 210000.0, 0.3
    F = -800.0  # tip force in y (negative = downward)
    Iz = b * h**3 / 12

    mesher = Mesher()
    section = _rect_section(mesher, b, h)

    assert abs(section.area - b * h) <= 1e-12

    point1, point2 = Point(), Point(x=L)
    line = Line(point1, point2, L / nL)
    beam = Models.Beam.Isotropic(beamDim, line, section, E, v)

    assert abs(beam.Iz - Iz) <= 1e-12

    mesh = mesher.Mesh_Beams([beam], elemType=elemType)
    structure = Models.Beam.BeamStructure([beam])
    simu = Simulations.Beam(mesh, structure, verbosity=False)

    simu.add_dirichlet(
        mesh.Nodes_Point(point1), [0] * simu.Get_dof_n(), simu.Get_unknowns()
    )
    simu.add_neumann(mesh.Nodes_Point(point2), [F], ["y"])
    simu.Solve()

    x_n = mesh.coord[:, 0]
    x_e = x_n[mesh.connect].mean(1)

    # uy
    uy_x = (
        lambda x: F * (L * np.asarray(x) ** 2 / 2 - np.asarray(x) ** 3 / 6) / (E * Iz)
    )
    uy_fe = simu.Result("uy", nodeValues=True)
    err_uy = np.abs(uy_x(x_n) - uy_fe).max() / np.abs(uy_x(L))
    assert err_uy <= TOL, f"uy error {err_uy:.2e}"

    # rz
    rz_x = lambda x: F / (E * Iz) * (L * np.asarray(x) - np.asarray(x) ** 2 / 2)
    rz_fe = simu.Result("rz", nodeValues=True)
    err_rz = np.abs(rz_x(x_n) - rz_fe).max() / np.abs(rz_x(L))
    assert err_rz <= TOL, f"rz error {err_rz:.2e}"

    # Mz: exact Mz is piecewise linear — element means are exact up to TOL
    Mz_x = lambda x: F * (L - np.asarray(x))
    Mz_fe = simu.Result("Mz", nodeValues=False)  # (Ne,)
    err_Mz = np.abs(Mz_x(x_e) - Mz_fe).max() / np.abs(Mz_x(x_e)).max()
    assert err_Mz <= TOL, f"Mz error {err_Mz:.2e}"

    # Ty = F (constant)
    Ty_fe = simu.Result("Ty", nodeValues=False)
    err_Ty = np.abs(F - Ty_fe).max() / np.abs(F)
    assert err_Ty <= TOL, f"Ty error {err_Ty:.2e}"

    Display.Plot_Mesh(simu, deformFactor=10)
    Display.Plot_Result(simu, "uy", plotMesh=False)
    _close_plots()


# -------------------------------------------------------
# Cantilever with uniform distributed load (dim >= 2)
# -------------------------------------------------------


@pytest.mark.parametrize("beamDim", [2, 3])
@pytest.mark.parametrize("elemType", ELEM_TYPES)
def test_cantilever_distrib(elemType: str, beamDim: int):
    L, nL = 120.0, 10
    b, h = 13.0, 13.0
    E, v = 210000.0, 0.3
    q = -10.0  # uniform load in y (negative = downward, N/mm)
    Iz = b * h**3 / 12

    mesher = Mesher()
    section = _rect_section(mesher, b, h)

    point1, point2 = Point(), Point(x=L)
    line = Line(point1, point2, L / nL)
    beam = Models.Beam.Isotropic(beamDim, line, section, E, v)

    mesh = mesher.Mesh_Beams([beam], elemType=elemType)
    structure = Models.Beam.BeamStructure([beam])
    simu = Simulations.Beam(mesh, structure, verbosity=False)

    simu.add_dirichlet(
        mesh.Nodes_Point(point1), [0] * simu.Get_dof_n(), simu.Get_unknowns()
    )
    simu.add_lineLoad(mesh.nodes, [q], ["y"])
    simu.Solve()

    x_n = mesh.coord[:, 0]
    x_e = x_n[mesh.connect].mean(1)

    uy_x, rz_x, Mz_x, Ty_x = _cantilever_distrib_analytical(L, q, E, Iz)

    # uy
    uy_fe = simu.Result("uy", nodeValues=True)
    err_uy = np.abs(uy_x(x_n) - uy_fe).max() / np.abs(uy_x(L))
    assert err_uy <= TOL, f"uy error {err_uy:.2e}"

    # rz
    rz_fe = simu.Result("rz", nodeValues=True)
    err_rz = np.abs(rz_x(x_n) - rz_fe).max() / np.abs(rz_x(L))
    assert err_rz <= TOL, f"rz error {err_rz:.2e}"

    # Mz: element means are O(h^2) under distributed load — check shape only
    assert simu.Result("Mz", nodeValues=False).shape == (mesh.Ne,)

    # Ty: compared at element centers (constant per element from -dMz/dx)
    Ty_fe = simu.Result("Ty", nodeValues=False)
    err_Ty = np.abs(Ty_x(x_e) - Ty_fe).max() / np.abs(Ty_x(x_e)).max()
    assert err_Ty <= TOL, f"Ty error {err_Ty:.2e}"


# -------------------------------------------------------
# Fixed-fixed beam with point load, symmetric and eccentric (dim >= 2)
# -------------------------------------------------------


@pytest.mark.parametrize("a_frac", [0.5, 0.8], ids=["symmetric", "eccentric"])
@pytest.mark.parametrize("beamDim", [2, 3])
@pytest.mark.parametrize("elemType", ELEM_TYPES)
def test_biencastre(elemType: str, beamDim: int, a_frac: float):
    L, nL = 120.0, 10
    b, h = 13.0, 13.0
    E, v = 210000.0, 0.3
    F = -800.0  # point force in y (negative = downward)
    a = L * a_frac  # load position
    Iz = b * h**3 / 12

    mesher = Mesher()
    section = _rect_section(mesher, b, h)

    point1 = Point()
    point_load = Point(x=a)
    point3 = Point(x=L)
    line = Line(point1, point3, L / nL)
    beam = Models.Beam.Isotropic(beamDim, line, section, E, v)

    mesh = mesher.Mesh_Beams([beam], additionalPoints=[point_load], elemType=elemType)
    structure = Models.Beam.BeamStructure([beam])
    simu = Simulations.Beam(mesh, structure, verbosity=False)

    simu.add_dirichlet(
        mesh.Nodes_Point(point1), [0] * simu.Get_dof_n(), simu.Get_unknowns()
    )
    simu.add_dirichlet(
        mesh.Nodes_Point(point3), [0] * simu.Get_dof_n(), simu.Get_unknowns()
    )
    simu.add_neumann(mesh.Nodes_Point(point_load), [F], ["y"])
    simu.Solve()

    x_n = mesh.coord[:, 0]
    x_e = x_n[mesh.connect].mean(1)

    uy_x, Mz_x, Ty_x, Ra, Rb = _fixed_fixed_analytical(a, L, F, E, Iz)

    # uy
    uy_fe = simu.Result("uy", nodeValues=True)
    err_uy = np.abs(uy_x(x_n) - uy_fe).max() / np.abs(uy_x(a))
    assert err_uy <= TOL, f"uy error {err_uy:.2e}"

    # Mz: exact Mz is piecewise linear — element means are exact up to TOL
    Mz_fe = simu.Result("Mz", nodeValues=False)  # (Ne,)
    err_Mz = np.abs(Mz_x(x_e) - Mz_fe).max() / np.abs(Mz_x(x_e)).max()
    assert err_Mz <= TOL, f"Mz error {err_Mz:.2e}"

    # Ty
    Ty_fe = simu.Result("Ty", nodeValues=False)
    err_Ty = np.abs(Ty_x(x_e) - Ty_fe).max() / max(Ra, Rb)
    assert err_Ty <= TOL, f"Ty error {err_Ty:.2e}"


# -------------------------------------------------------
# Simply supported beam (pinned-roller) with point load (dim >= 2)
# -------------------------------------------------------


@pytest.mark.parametrize("a_frac", [0.5, 0.8], ids=["symmetric", "eccentric"])
@pytest.mark.parametrize("beamDim", [2, 3])
@pytest.mark.parametrize("elemType", ELEM_TYPES)
def test_simply_supported_tip(elemType: str, beamDim: int, a_frac: float):
    L, nL = 120.0, 10
    b, h = 13.0, 13.0
    E, v = 210000.0, 0.3
    F = -800.0  # point force in y (negative = downward)
    a = L * a_frac  # load position
    Iz = b * h**3 / 12

    mesher = Mesher()
    section = _rect_section(mesher, b, h)

    point1 = Point()
    point_load = Point(x=a)
    point3 = Point(x=L)
    line = Line(point1, point3, L / nL)
    beam = Models.Beam.Isotropic(beamDim, line, section, E, v)

    mesh = mesher.Mesh_Beams([beam], additionalPoints=[point_load], elemType=elemType)
    structure = Models.Beam.BeamStructure([beam])
    simu = Simulations.Beam(mesh, structure, verbosity=False)

    # pinned at x=0: fix all translations; roller at x=L: fix transverse translations
    if beamDim == 2:
        simu.add_dirichlet(mesh.Nodes_Point(point1), [0, 0], ["x", "y"])
        simu.add_dirichlet(mesh.Nodes_Point(point3), [0], ["y"])
    elif beamDim == 3:
        simu.add_dirichlet(mesh.Nodes_Point(point1), [0, 0, 0], ["x", "y", "z"])
        simu.add_dirichlet(mesh.Nodes_Point(point3), [0, 0], ["y", "z"])

    simu.add_neumann(mesh.Nodes_Point(point_load), [F], ["y"])
    simu.Solve()

    x_n = mesh.coord[:, 0]
    x_e = x_n[mesh.connect].mean(1)

    uy_x, Mz_x, Ty_x, Ra, Rb = _simply_supported_analytical(a, L, F, E, Iz)

    # uy
    uy_fe = simu.Result("uy", nodeValues=True)
    err_uy = np.abs(uy_x(x_n) - uy_fe).max() / np.abs(uy_x(a))
    assert err_uy <= TOL, f"uy error {err_uy:.2e}"

    # Mz: exact Mz is piecewise linear — element means are exact up to TOL
    Mz_fe = simu.Result("Mz", nodeValues=False)  # (Ne,)
    err_Mz = np.abs(Mz_x(x_e) - Mz_fe).max() / np.abs(Mz_x(x_e)).max()
    assert err_Mz <= TOL, f"Mz error {err_Mz:.2e}"

    # Ty
    Ty_fe = simu.Result("Ty", nodeValues=False)
    err_Ty = np.abs(Ty_x(x_e) - Ty_fe).max() / max(Ra, Rb)
    assert err_Ty <= TOL, f"Ty error {err_Ty:.2e}"

    Display.Plot_BoundaryConditions(simu)
    _close_plots()


# -------------------------------------------------------
# Simply supported beam with uniform distributed load (dim >= 2)
# -------------------------------------------------------


@pytest.mark.parametrize("beamDim", [2, 3])
@pytest.mark.parametrize("elemType", ELEM_TYPES)
def test_simply_supported_distrib(elemType: str, beamDim: int):
    L, nL = 120.0, 10
    b, h = 13.0, 13.0
    E, v = 210000.0, 0.3
    q = -10.0  # uniform load in y (negative = downward, N/mm)
    Iz = b * h**3 / 12

    mesher = Mesher()
    section = _rect_section(mesher, b, h)

    point1, point3 = Point(), Point(x=L)
    line = Line(point1, point3, L / nL)
    beam = Models.Beam.Isotropic(beamDim, line, section, E, v)

    mesh = mesher.Mesh_Beams([beam], elemType=elemType)
    structure = Models.Beam.BeamStructure([beam])
    simu = Simulations.Beam(mesh, structure, verbosity=False)

    if beamDim == 2:
        simu.add_dirichlet(mesh.Nodes_Point(point1), [0, 0], ["x", "y"])
        simu.add_dirichlet(mesh.Nodes_Point(point3), [0], ["y"])
    elif beamDim == 3:
        simu.add_dirichlet(mesh.Nodes_Point(point1), [0, 0, 0], ["x", "y", "z"])
        simu.add_dirichlet(mesh.Nodes_Point(point3), [0, 0], ["y", "z"])

    simu.add_lineLoad(mesh.nodes, [q], ["y"])
    simu.Solve()

    x_n = mesh.coord[:, 0]
    x_e = x_n[mesh.connect].mean(1)

    uy_x, Mz_x, Ty_x = _simply_supported_distrib_analytical(L, q, E, Iz)

    # uy
    uy_fe = simu.Result("uy", nodeValues=True)
    err_uy = np.abs(uy_x(x_n) - uy_fe).max() / np.abs(uy_x(L / 2))
    assert err_uy <= TOL, f"uy error {err_uy:.2e}"

    # Mz: element means are O(h^2) under distributed load — check shape only
    assert simu.Result("Mz", nodeValues=False).shape == (mesh.Ne,)

    # Ty: compared at element centers (constant per element from -dMz/dx)
    Ty_fe = simu.Result("Ty", nodeValues=False)
    err_Ty = np.abs(Ty_x(x_e) - Ty_fe).max() / np.abs(Ty_x(x_e)).max()
    assert err_Ty <= TOL, f"Ty error {err_Ty:.2e}"


# # TODO #39
# # -------------------------------------------------------
# # Timoshenko beams (shear-deformable)
# # -------------------------------------------------------


# def _timoshenko_kGA(beamDim: int, E: float, v: float, A: float) -> float:
#     """Effective shear stiffness kGA using the hardcoded correction factors.

#     2D: ky = 5/6 (Timoshenko rectangle, classic value).
#     3D: ky = kz = 1 (placeholder pending issue #37).
#     """
#     mu = E / (2 * (1 + v))
#     ky = 5 / 6
#     return ky * mu * A


# @pytest.mark.parametrize("beamDim", [2, 3])
# @pytest.mark.parametrize("elemType", ELEM_TYPES)
# def test_timoshenko_cantilever_tip(elemType: ElemType, beamDim: int):
#     """Timoshenko cantilever with tip load — all element types.

#     Uses a stocky beam (L/h ≈ 1.5) so the shear term F·x/(kGA) is ~33 % of the
#     bending term — large enough that using the EB formula would fail the tolerance.

#     The exact solution has θ(x) = quadratic (degree 2).  SEGn interpolates θ with
#     Lagrange polynomials of degree n-1, so:

#       elemType | θ degree | exact? | nL needed
#       ---------+----------+--------+----------
#       SEG2     | 1        | no     | 3000  (O(h²) convergence)
#       SEG3     | 2        | yes    | 2     (machine precision)
#       SEG4     | 3        | yes    | 2
#       SEG5     | 4        | yes    | 2

#     Analytical solution (exact for Timoshenko beam theory):
#         uy(x) = F·(L·x²/2 - x³/6)/(E·Iz)  +  F·x/(kGA)   ← shear adds F·x/(kGA)
#         rz(x) = F·(L·x - x²/2)/(E·Iz)                      ← θ only, same as EB
#         Mz(x) = F·(L - x)                                   ← same as EB
#         Ty(x) = F                                            ← constant from γ
#     """
#     L, nL = 20.0, 10
#     b, h = 13.0, 13.0
#     E, v = 210000.0, 0.3
#     F = -800.0  # tip force in y (negative = downward)
#     Iz = b * h**3 / 12
#     A = b * h
#     kGA = _timoshenko_kGA(beamDim, E, v, A)

#     mesher = Mesher()
#     section = _rect_section(mesher, b, h)

#     point1, point2 = Point(), Point(x=L)
#     line = Line(point1, point2, L / nL)
#     beam = Models.Beam.Isotropic(beamDim, line, section, E, v)

#     mesh = mesher.Mesh_Beams([beam], elemType=elemType)
#     structure = Models.Beam.BeamStructure([beam])
#     simu = Simulations.Beam(mesh, structure, useTimoshenko=True, verbosity=False)

#     simu.add_dirichlet(
#         mesh.Nodes_Point(point1), [0] * simu.Get_dof_n(), simu.Get_unknowns()
#     )
#     simu.add_neumann(mesh.Nodes_Point(point2), [F], ["y"])
#     simu.Solve()

#     x_n = mesh.coord[:, 0]
#     x_e = x_n[mesh.connect].mean(1)

#     # uy: bending + shear contribution
#     uy_x = lambda x: (
#         F * (L * np.asarray(x) ** 2 / 2 - np.asarray(x) ** 3 / 6) / (E * Iz)
#         + F * np.asarray(x) / kGA
#     )
#     uy_fe = simu.Result("uy", nodeValues=True)
#     err_uy = np.abs(uy_x(x_n) - uy_fe).max() / np.abs(uy_x(L))
#     assert err_uy <= TOL, f"uy error {err_uy:.2e}"

#     # rz: bending rotation θ only (independent DOF — does NOT include shear slope)
#     rz_x = lambda x: F / (E * Iz) * (L * np.asarray(x) - np.asarray(x) ** 2 / 2)
#     rz_fe = simu.Result("rz", nodeValues=True)
#     err_rz = np.abs(rz_x(x_n) - rz_fe).max() / np.abs(rz_x(L))
#     assert err_rz <= TOL, f"rz error {err_rz:.2e}"

#     # Mz: piecewise linear — element means exact
#     Mz_x = lambda x: F * (L - np.asarray(x))
#     Mz_fe = simu.Result("Mz", nodeValues=False)  # (Ne,)
#     err_Mz = np.abs(Mz_x(x_e) - Mz_fe).max() / np.abs(Mz_x(x_e)).max()
#     assert err_Mz <= TOL, f"Mz error {err_Mz:.2e}"

#     # Ty: primary result from shear strain γ = v' - θ — constant
#     Ty_fe = simu.Result("Ty", nodeValues=False)
#     err_Ty = np.abs(F - Ty_fe).max() / np.abs(F)
#     assert err_Ty <= TOL, f"Ty error {err_Ty:.2e}"


# @pytest.mark.parametrize("beamDim", [2, 3])
# @pytest.mark.parametrize("elemType", ELEM_TYPES)
# def test_timoshenko_cantilever_distrib(elemType: ElemType, beamDim: int):
#     """Timoshenko cantilever with uniform distributed load — all element types.

#     The exact solution has θ(x) = cubic (degree 3).  SEGn interpolates θ with
#     Lagrange polynomials of degree n-1, so:

#       elemType | θ degree | exact? | nL needed
#       ---------+----------+--------+----------
#       SEG2     | 1        | no     | 3000  (O(h²))
#       SEG3     | 2        | no     | 3000  (O(h²), but 8e-14 error at nL=3000 << TOL)
#       SEG4     | 3        | yes    | 2     (machine precision)
#       SEG5     | 4        | yes    | 2

#     Analytical solution (exact for Timoshenko beam theory):
#         uy(x) = q·x²·(6L²-4Lx+x²)/(24EIz)  +  q·x·(2L-x)/(2·kGA)
#         rz(x) = q·(x³/6 - L·x²/2 + L²·x/2)/(EIz)   ← same ODE as EB
#         Mz(x) = q·(x-L)²/2
#         Ty(x) = q·(L-x)
#     """
#     L, nL = 20.0, 10
#     b, h = 13.0, 13.0
#     E, v = 210000.0, 0.3
#     q = -10.0  # uniform load in y (negative = downward, N/mm)
#     Iz = b * h**3 / 12
#     A = b * h
#     kGA = _timoshenko_kGA(beamDim, E, v, A)

#     mesher = Mesher()
#     section = _rect_section(mesher, b, h)

#     point1, point2 = Point(), Point(x=L)
#     line = Line(point1, point2, L / nL)
#     beam = Models.Beam.Isotropic(beamDim, line, section, E, v)

#     mesh = mesher.Mesh_Beams([beam], elemType=elemType)
#     structure = Models.Beam.BeamStructure([beam])
#     simu = Simulations.Beam(mesh, structure, useTimoshenko=True, verbosity=False)

#     simu.add_dirichlet(
#         mesh.Nodes_Point(point1), [0] * simu.Get_dof_n(), simu.Get_unknowns()
#     )
#     simu.add_lineLoad(mesh.nodes, [q], ["y"])
#     simu.Solve()

#     x_n = mesh.coord[:, 0]
#     x_e = x_n[mesh.connect].mean(1)

#     uy_EB_x, rz_x, Mz_x, Ty_x = _cantilever_distrib_analytical(L, q, E, Iz)

#     # uy: EB bending + shear term q·x·(2L-x)/(2·kGA)
#     uy_x = lambda x: uy_EB_x(x) + q * np.asarray(x) * (2 * L - np.asarray(x)) / (
#         2 * kGA
#     )
#     uy_fe = simu.Result("uy", nodeValues=True)
#     err_uy = np.abs(uy_x(x_n) - uy_fe).max() / np.abs(uy_x(L))
#     assert err_uy <= TOL, f"uy error {err_uy:.2e}"

#     # rz: bending rotation θ only — same ODE as EB, same analytical formula
#     rz_fe = simu.Result("rz", nodeValues=True)
#     err_rz = np.abs(rz_x(x_n) - rz_fe).max() / np.abs(rz_x(L))
#     assert err_rz <= TOL, f"rz error {err_rz:.2e}"

#     # Mz: element means are O(h^2) under distributed load — check shape only
#     assert simu.Result("Mz", nodeValues=False).shape == (mesh.Ne,)

#     # Ty: linear — element means at centroids are exact
#     Ty_fe = simu.Result("Ty", nodeValues=False)
#     err_Ty = np.abs(Ty_x(x_e) - Ty_fe).max() / np.abs(Ty_x(x_e)).max()
#     assert err_Ty <= TOL, f"Ty error {err_Ty:.2e}"


# @pytest.mark.parametrize("beamDim", [2, 3])
# @pytest.mark.parametrize("elemType", ELEM_TYPES)
# def test_timoshenko_simply_supported(elemType: ElemType, beamDim: int):
#     """Timoshenko simply supported beam with uniform distributed load — all element types.

#     Same θ_exact degree = 3 (UDL) as test_timoshenko_cantilever_distrib.
#     See that test for the nL selection table.

#     Analytical solution (exact for Timoshenko beam theory):
#         uy(x) = q·x·(L³-2Lx²+x³)/(24EIz)  +  q·x·(L-x)/(2·kGA)
#         Mz(x) = q·x·(x-L)/2
#         Ty(x) = q·(L/2-x)
#     """
#     L, nL = 20.0, 10
#     b, h = 13.0, 13.0
#     E, v = 210000.0, 0.3
#     q = -10.0  # uniform load in y (negative = downward, N/mm)
#     Iz = b * h**3 / 12
#     A = b * h
#     kGA = _timoshenko_kGA(beamDim, E, v, A)

#     mesher = Mesher()
#     section = _rect_section(mesher, b, h)

#     point1, point3 = Point(), Point(x=L)
#     line = Line(point1, point3, L / nL)
#     beam = Models.Beam.Isotropic(beamDim, line, section, E, v)

#     mesh = mesher.Mesh_Beams([beam], elemType=elemType)
#     structure = Models.Beam.BeamStructure([beam])
#     simu = Simulations.Beam(mesh, structure, useTimoshenko=True, verbosity=False)

#     if beamDim == 2:
#         simu.add_dirichlet(mesh.Nodes_Point(point1), [0, 0], ["x", "y"])
#         simu.add_dirichlet(mesh.Nodes_Point(point3), [0], ["y"])
#     elif beamDim == 3:
#         simu.add_dirichlet(mesh.Nodes_Point(point1), [0, 0, 0], ["x", "y", "z"])
#         simu.add_dirichlet(mesh.Nodes_Point(point3), [0, 0], ["y", "z"])

#     simu.add_lineLoad(mesh.nodes, [q], ["y"])
#     simu.Solve()

#     x_n = mesh.coord[:, 0]
#     x_e = x_n[mesh.connect].mean(1)

#     uy_EB_x, Mz_x, Ty_x = _simply_supported_distrib_analytical(L, q, E, Iz)

#     # uy: EB bending + shear term q·x·(L-x)/(2·kGA)
#     uy_x = lambda x: uy_EB_x(x) + q * np.asarray(x) * (L - np.asarray(x)) / (2 * kGA)
#     uy_fe = simu.Result("uy", nodeValues=True)
#     err_uy = np.abs(uy_x(x_n) - uy_fe).max() / np.abs(uy_x(L / 2))
#     assert err_uy <= TOL, f"uy error {err_uy:.2e}"

#     # Mz: element means are O(h^2) under distributed load — check shape only
#     assert simu.Result("Mz", nodeValues=False).shape == (mesh.Ne,)

#     # Ty: linear — element means at centroids are exact
#     Ty_fe = simu.Result("Ty", nodeValues=False)
#     err_Ty = np.abs(Ty_x(x_e) - Ty_fe).max() / np.abs(Ty_x(x_e)).max()
#     assert err_Ty <= TOL, f"Ty error {err_Ty:.2e}"


# -------------------------------------------------------
# Material update triggers simulation reassembly
# -------------------------------------------------------


def test_update_beam():
    """Modifications to Beam material properties must mark the simulation as needing update."""

    def assert_needs_update(simu: Simulations._Simu) -> None:
        assert simu.needUpdate
        simu.Need_Update(False)

    mesher = Mesher()
    sect1 = mesher.Mesh_2D(Domain(Point(), Point(0.01, 0.01)))
    sect2 = sect1.copy()
    sect2.Rotate(30, sect2.center)
    sect3 = sect2.copy()
    sect3.Rotate(30, sect3.center)

    beam1 = Models.Beam.Isotropic(2, Line(Point(), Point(5)), sect1, 210e9, v=0.1)
    beam2 = Models.Beam.Isotropic(2, Line(Point(5), Point(10)), sect2, 210e9, v=0.1)
    beams = [beam1, beam2]

    structure = Models.Beam.BeamStructure(beams)
    mesh = mesher.Mesh_Beams(beams)
    simu = Simulations.Beam(mesh, structure)

    simu.Get_K_C_M_F()
    assert not simu.needUpdate  # assembling clears the flag

    for beam in beams:
        beam.E *= 2
        assert_needs_update(simu)
        beam.v = 0.4
        assert_needs_update(simu)
        beam.section = sect3
        assert_needs_update(simu)
