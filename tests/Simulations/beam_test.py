# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import matplotlib
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

matplotlib.use("Agg")

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


# -------------------------------------------------------
# Timoshenko beams (shear-deformable)
# -------------------------------------------------------


def _timoshenko_kGA(
    beam: Models.Beam.Isotropic, E: float, v: float, A: float, axis: str = "y"
) -> float:
    """Effective shear stiffness kGA — reads the shear correction factor from
    the beam's section via the Saint-Venant Poisson formulation, so the
    analytical reference matches whatever the simulation actually uses
    (≈ 5/6 for a well-meshed rectangular section, ≈ 6/7 for a circle —
    Cowper-with-ν=0 values).
    """
    mu = E / (2 * (1 + v))
    k = beam._Get_shear_kappa(axis)
    return k * mu * A


# Per-element tolerance / mesh size for the Timoshenko tests.
# With selective reduced integration, the bending/torsion/axial terms are
# integrated exactly for any polynomial Mz the element can represent, so the
# only error source is whether the Lagrange degree (n-1) can represent the
# exact transverse displacement v.
#
#   exact v shape       | SEG2 (P¹)  | SEG3 (P²)  | SEG4 (P³)  | SEG5 (P⁴)
#   --------------------+------------+------------+------------+----------
#   cubic   (tip load)  | O(h²)      | exact      | exact      | exact
#   quartic (distrib q) | O(h²)      | O(h⁴)      | O(h⁵)      | exact
#
# At nL=10, SEG2 gives ~1.2e-3 error, SEG3 ~1.5e-6 (distrib), SEG4 ~1.1e-7,
# SEG5 machine precision.  TOLS_TIP / TOLS_DISTRIB lock these in.

TOLS_TIP = {"SEG2": 2e-3, "SEG3": 1e-12, "SEG4": 1e-12, "SEG5": 1e-12}
TOLS_DISTRIB = {"SEG2": 1e-2, "SEG3": 1e-5, "SEG4": 1e-6, "SEG5": 1e-12}


@pytest.mark.parametrize("beamDim", [2, 3])
@pytest.mark.parametrize("elemType", ELEM_TYPES)
def test_timoshenko_cantilever_tip(elemType: str, beamDim: int):
    """Timoshenko cantilever with tip load — all element types.

    Uses a stocky beam (L/h ≈ 1.5) so the shear term F·x/(kGA) is ~33 % of the
    bending term — large enough that using the EB formula would fail the tolerance.

    Pure-Lagrange + selective reduced integration on the shear term (Hughes 1987;
    Bathe 2006) eliminates shear locking.  The exact tip-load v is cubic, so:

      elemType | v degree | exact? | nL=10 max-node err
      ---------+----------+--------+--------------------
      SEG2     | 1        | no     | 1.88e-3   (O(h²) convergence)
      SEG3     | 2        | yes    | ~1e-14   (machine precision)
      SEG4     | 3        | yes    | ~1e-14
      SEG5     | 4        | yes    | ~1e-14

    Analytical solution (exact for Timoshenko beam theory):
        uy(x) = F·(L·x²/2 - x³/6)/(E·Iz)  +  F·x/(kGA)   ← shear adds F·x/(kGA)
        rz(x) = F·(L·x - x²/2)/(E·Iz)                      ← θ only, same as EB
        Mz(x) = F·(L - x)                                   ← same as EB
        Ty(x) = F                                            ← constant from γ
    """
    L, nL = 20.0, 10
    b, h = 13.0, 13.0
    E, v = 210000.0, 0.3
    F = -800.0  # tip force in y (negative = downward)
    Iz = b * h**3 / 12
    A = b * h
    tol = TOLS_TIP[elemType]

    mesher = Mesher()
    section = _rect_section(mesher, b, h)

    point1, point2 = Point(), Point(x=L)
    line = Line(point1, point2, L / nL)
    beam = Models.Beam.Isotropic(beamDim, line, section, E, v)
    kGA = _timoshenko_kGA(beam, E, v, A)

    mesh = mesher.Mesh_Beams([beam], elemType=elemType)
    structure = Models.Beam.BeamStructure([beam])
    simu = Simulations.Beam(mesh, structure, useTimoshenko=True, verbosity=False)

    simu.add_dirichlet(
        mesh.Nodes_Point(point1), [0] * simu.Get_dof_n(), simu.Get_unknowns()
    )
    simu.add_neumann(mesh.Nodes_Point(point2), [F], ["y"])
    simu.Solve()

    x_n = mesh.coord[:, 0]
    x_e = x_n[mesh.connect].mean(1)

    # uy: bending + shear contribution
    uy_x = lambda x: (
        F * (L * np.asarray(x) ** 2 / 2 - np.asarray(x) ** 3 / 6) / (E * Iz)
        + F * np.asarray(x) / kGA
    )
    uy_fe = simu.Result("uy", nodeValues=True)
    err_uy = np.abs(uy_x(x_n) - uy_fe).max() / np.abs(uy_x(L))
    assert err_uy <= tol, f"uy error {err_uy:.2e} (tol {tol:.0e})"

    # rz: bending rotation θ only (independent DOF — does NOT include shear slope)
    rz_x = lambda x: F / (E * Iz) * (L * np.asarray(x) - np.asarray(x) ** 2 / 2)
    rz_fe = simu.Result("rz", nodeValues=True)
    err_rz = np.abs(rz_x(x_n) - rz_fe).max() / np.abs(rz_x(L))
    assert err_rz <= tol, f"rz error {err_rz:.2e} (tol {tol:.0e})"

    # Mz: piecewise linear — element means exact
    Mz_x = lambda x: F * (L - np.asarray(x))
    Mz_fe = simu.Result("Mz", nodeValues=False)  # (Ne,)
    err_Mz = np.abs(Mz_x(x_e) - Mz_fe).max() / np.abs(Mz_x(x_e)).max()
    assert err_Mz <= TOL, f"Mz error {err_Mz:.2e}"

    # Ty: primary result from shear strain γ = v' - θ — constant.
    # Evaluated at the reduced Gauss points where γ is physical.
    Ty_fe = simu.Result("Ty", nodeValues=False)
    err_Ty = np.abs(F - Ty_fe).max() / np.abs(F)
    assert err_Ty <= TOL, f"Ty error {err_Ty:.2e}"


def test_timoshenko_cantilever_tip_beam2_example():
    """Regression test mirroring ``examples/Beam/Beam2.py`` exactly:

      - 2D Timoshenko cantilever
      - slender geometry (L = 120, b = h = 13 → L/h ≈ 9.2)
      - TRI6 section (quadratic — matches the example's choice and our
        recommendation for accurate _shear_kappa)
      - SEG3 beam (cubic v → exact for tip-load deflection)
      - nL = 10

    Catches regressions that affect the slender / TRI6 path specifically.
    """
    L, nL = 120.0, 10
    b, h = 13.0, 13.0
    E, v = 210000.0, 0.3
    F = -800.0
    Iz = b * h**3 / 12
    A = b * h

    mesher = Mesher()
    # TRI6 section (matches Beam2.py); domain at (0, 0)-(b, h) as in the example
    section = mesher.Mesh_2D(Domain(Point(0, 0), Point(b, h)), elemType=ElemType.TRI6)

    p1, p2 = Point(0, 0), Point(L, 0)
    line = Line(p1, p2, L / nL)
    beam = Models.Beam.Isotropic(2, line, section, E, v)
    kGA = _timoshenko_kGA(beam, E, v, A)  # uses the beam's actual _shear_kappa

    mesh = mesher.Mesh_Beams([beam], elemType=ElemType.SEG3)
    structure = Models.Beam.BeamStructure([beam])
    simu = Simulations.Beam(mesh, structure, useTimoshenko=True, verbosity=False)

    simu.add_dirichlet(
        mesh.Nodes_Point(p1), [0] * simu.Get_dof_n(), simu.Get_unknowns()
    )
    simu.add_neumann(mesh.Nodes_Point(p2), [F], ["y"])
    simu.Solve()

    x_n = mesh.coord[:, 0]
    x_e = x_n[mesh.connect].mean(1)

    uy_x = lambda x: (
        F * (L * np.asarray(x) ** 2 / 2 - np.asarray(x) ** 3 / 6) / (E * Iz)
        + F * np.asarray(x) / kGA
    )
    rz_x = lambda x: F / (E * Iz) * (L * np.asarray(x) - np.asarray(x) ** 2 / 2)
    Mz_x = lambda x: F * (L - np.asarray(x))

    uy_fe = simu.Result("uy", nodeValues=True)
    rz_fe = simu.Result("rz", nodeValues=True)
    Mz_fe = simu.Result("Mz", nodeValues=False)
    Ty_fe = simu.Result("Ty", nodeValues=False)

    # SEG3 + cubic-exact v + consistent kGA → all four checks are essentially
    # exact.  Tolerance 1e-9 accommodates the small FP accumulation noise from
    # the larger L (120) and the TRI6 Poisson solve, while still catching any
    # algorithmic regression.
    tol = 1e-9

    err_uy = np.abs(uy_x(x_n) - uy_fe).max() / np.abs(uy_x(L))
    err_rz = np.abs(rz_x(x_n) - rz_fe).max() / np.abs(rz_x(L))
    err_Mz = np.abs(Mz_x(x_e) - Mz_fe).max() / np.abs(Mz_x(x_e)).max()
    err_Ty = np.abs(F - Ty_fe).max() / np.abs(F)

    assert err_uy <= tol, f"uy error {err_uy:.2e} (tol {tol:.0e})"
    assert err_rz <= tol, f"rz error {err_rz:.2e} (tol {tol:.0e})"
    assert err_Mz <= TOL, f"Mz error {err_Mz:.2e}"
    assert err_Ty <= TOL, f"Ty error {err_Ty:.2e}"


@pytest.mark.parametrize("beamDim", [2, 3])
@pytest.mark.parametrize("elemType", ELEM_TYPES)
def test_timoshenko_cantilever_distrib(elemType: str, beamDim: int):
    """Timoshenko cantilever with uniform distributed load — all element types.

    Pure-Lagrange + selective reduced integration.  The exact v is quartic.

      elemType | v degree | exact? | nL=10 max-node err
      ---------+----------+--------+--------------------
      SEG2     | 1        | no     | 1.16e-3   (O(h²))
      SEG3     | 2        | no     | 1.45e-6   (O(h⁴))
      SEG4     | 3        | no     | 1.14e-7   (O(h⁵))
      SEG5     | 4        | yes    | ~1e-13   (machine precision)

    Analytical solution (exact for Timoshenko beam theory):
        uy(x) = q·x²·(6L²-4Lx+x²)/(24EIz)  +  q·x·(2L-x)/(2·kGA)
        rz(x) = q·(x³/6 - L·x²/2 + L²·x/2)/(EIz)   ← same ODE as EB
        Mz(x) = q·(x-L)²/2
        Ty(x) = q·(L-x)
    """
    L, nL = 20.0, 10
    b, h = 13.0, 13.0
    E, v = 210000.0, 0.3
    q = -10.0  # uniform load in y (negative = downward, N/mm)
    Iz = b * h**3 / 12
    A = b * h
    tol = TOLS_DISTRIB[elemType]

    mesher = Mesher()
    section = _rect_section(mesher, b, h)

    point1, point2 = Point(), Point(x=L)
    line = Line(point1, point2, L / nL)
    beam = Models.Beam.Isotropic(beamDim, line, section, E, v)
    kGA = _timoshenko_kGA(beam, E, v, A)

    mesh = mesher.Mesh_Beams([beam], elemType=elemType)
    structure = Models.Beam.BeamStructure([beam])
    simu = Simulations.Beam(mesh, structure, useTimoshenko=True, verbosity=False)

    simu.add_dirichlet(
        mesh.Nodes_Point(point1), [0] * simu.Get_dof_n(), simu.Get_unknowns()
    )
    simu.add_lineLoad(mesh.nodes, [q], ["y"])
    simu.Solve()

    x_n = mesh.coord[:, 0]
    x_e = x_n[mesh.connect].mean(1)

    uy_EB_x, rz_x, Mz_x, Ty_x = _cantilever_distrib_analytical(L, q, E, Iz)

    # uy: EB bending + shear term q·x·(2L-x)/(2·kGA)
    uy_x = lambda x: uy_EB_x(x) + q * np.asarray(x) * (2 * L - np.asarray(x)) / (
        2 * kGA
    )
    uy_fe = simu.Result("uy", nodeValues=True)
    err_uy = np.abs(uy_x(x_n) - uy_fe).max() / np.abs(uy_x(L))
    assert err_uy <= tol, f"uy error {err_uy:.2e} (tol {tol:.0e})"

    # rz: bending rotation θ only — same ODE as EB, same analytical formula
    rz_fe = simu.Result("rz", nodeValues=True)
    err_rz = np.abs(rz_x(x_n) - rz_fe).max() / np.abs(rz_x(L))
    assert err_rz <= tol, f"rz error {err_rz:.2e} (tol {tol:.0e})"

    # Mz: element means are O(h^2) under distributed load — check shape only
    assert simu.Result("Mz", nodeValues=False).shape == (mesh.Ne,)

    # Ty: linear — element means at reduced Gauss points capture this accurately.
    Ty_fe = simu.Result("Ty", nodeValues=False)
    err_Ty = np.abs(Ty_x(x_e) - Ty_fe).max() / np.abs(Ty_x(x_e)).max()
    assert err_Ty <= tol, f"Ty error {err_Ty:.2e} (tol {tol:.0e})"


@pytest.mark.parametrize("beamDim", [2, 3])
@pytest.mark.parametrize("elemType", ELEM_TYPES)
def test_timoshenko_simply_supported(elemType: str, beamDim: int):
    """Timoshenko simply supported beam with uniform distributed load — all element types.

    Same exact v degree = 4 as test_timoshenko_cantilever_distrib; convergence
    table is the same — see TOLS_DISTRIB.

    Analytical solution (exact for Timoshenko beam theory):
        uy(x) = q·x·(L³-2Lx²+x³)/(24EIz)  +  q·x·(L-x)/(2·kGA)
        Mz(x) = q·x·(x-L)/2
        Ty(x) = q·(L/2-x)
    """
    L, nL = 20.0, 10
    b, h = 13.0, 13.0
    E, v = 210000.0, 0.3
    q = -10.0  # uniform load in y (negative = downward, N/mm)
    Iz = b * h**3 / 12
    A = b * h
    tol = TOLS_DISTRIB[elemType]

    mesher = Mesher()
    section = _rect_section(mesher, b, h)

    point1, point3 = Point(), Point(x=L)
    line = Line(point1, point3, L / nL)
    beam = Models.Beam.Isotropic(beamDim, line, section, E, v)
    kGA = _timoshenko_kGA(beam, E, v, A)

    mesh = mesher.Mesh_Beams([beam], elemType=elemType)
    structure = Models.Beam.BeamStructure([beam])
    simu = Simulations.Beam(mesh, structure, useTimoshenko=True, verbosity=False)

    if beamDim == 2:
        simu.add_dirichlet(mesh.Nodes_Point(point1), [0, 0], ["x", "y"])
        simu.add_dirichlet(mesh.Nodes_Point(point3), [0], ["y"])
    elif beamDim == 3:
        # rx must be pinned somewhere — pure-Lagrange Timoshenko leaves the
        # torsion field with a free rigid-body mode otherwise (no torsion load).
        simu.add_dirichlet(
            mesh.Nodes_Point(point1), [0, 0, 0, 0], ["x", "y", "z", "rx"]
        )
        simu.add_dirichlet(mesh.Nodes_Point(point3), [0, 0], ["y", "z"])

    simu.add_lineLoad(mesh.nodes, [q], ["y"])
    simu.Solve()

    x_n = mesh.coord[:, 0]
    x_e = x_n[mesh.connect].mean(1)

    uy_EB_x, Mz_x, Ty_x = _simply_supported_distrib_analytical(L, q, E, Iz)

    # uy: EB bending + shear term q·x·(L-x)/(2·kGA)
    uy_x = lambda x: uy_EB_x(x) + q * np.asarray(x) * (L - np.asarray(x)) / (2 * kGA)
    uy_fe = simu.Result("uy", nodeValues=True)
    err_uy = np.abs(uy_x(x_n) - uy_fe).max() / np.abs(uy_x(L / 2))
    assert err_uy <= tol, f"uy error {err_uy:.2e} (tol {tol:.0e})"

    # Mz: element means are O(h^2) under distributed load — check shape only
    assert simu.Result("Mz", nodeValues=False).shape == (mesh.Ne,)

    # Ty: linear — element means at reduced Gauss points are exact.
    Ty_fe = simu.Result("Ty", nodeValues=False)
    err_Ty = np.abs(Ty_x(x_e) - Ty_fe).max() / np.abs(Ty_x(x_e)).max()
    assert err_Ty <= tol, f"Ty error {err_Ty:.2e} (tol {tol:.0e})"


# -------------------------------------------------------
# Saint-Venant shear correction factor — section geometry
# -------------------------------------------------------
#
# The dedicated _shear_kappa tests use TRI6 sections (quadratic — what the
# code's own warning recommends).  φ is a cubic polynomial in (x, y) for
# rectangles / circles, and TRI6 captures it with O(h³) error → ~1e-5 at the
# default meshing, so we can assert a tight 1e-4 tolerance.
#
# The TRI3 path is exercised implicitly by every test_timoshenko_* test
# (those use _rect_section, which defaults to TRI3) — no need for a dedicated
# TRI3 case here.


@pytest.mark.parametrize(
    "b,h",
    [(10.0, 10.0), (10.0, 30.0), (30.0, 10.0), (2.0, 20.0)],
    ids=["square", "tall", "wide", "thin-tall"],
)
def test_shear_kappa_rectangle(b: float, h: float):
    """Saint-Venant Poisson k for a rectangular cross-section converges to 5/6
    (same as the canonical Timoshenko shear coefficient — Cowper-ν=0)."""
    from EasyFEA.Geoms import Line

    mesher = Mesher()
    section = mesher.Mesh_2D(
        Domain((-b / 2, -h / 2), (b / 2, h / 2), meshSize=min(b, h) / 10),
        [],
        ElemType.TRI6,
    )
    line = Line((0, 0), (10, 0), 1)
    beam = Models.Beam.Isotropic(2, line, section, 210e9, 0.3)

    k_expected = 5 / 6
    tol = 1e-4
    ky = beam._Get_shear_kappa("y")
    kz = beam._Get_shear_kappa("z")
    err_y = abs(ky - k_expected) / k_expected
    err_z = abs(kz - k_expected) / k_expected
    assert err_y <= tol, f"ky={ky:.6f} err {err_y:.2e} (tol {tol:.0e})"
    assert err_z <= tol, f"kz={kz:.6f} err {err_z:.2e} (tol {tol:.0e})"


@pytest.mark.parametrize("diam", [5.0, 10.0, 20.0])
def test_shear_kappa_circle(diam: float):
    """Saint-Venant Poisson k for a circular section converges to 6/7
    (Cowper-ν=0; pure Jouravski's 9/10 misses the 2-D shear distribution)."""
    from EasyFEA.Geoms import Circle, Line

    mesher = Mesher()
    section = mesher.Mesh_2D(
        Circle(Point(), diam=diam, meshSize=diam / 30), [], ElemType.TRI6
    )
    line = Line(Point(0, 0), Point(x=10), 1)
    beam = Models.Beam.Isotropic(2, line, section, 210e9, 0.3)

    k_expected = 6 / 7
    tol = 1e-4
    ky = beam._Get_shear_kappa("y")
    kz = beam._Get_shear_kappa("z")
    err_y = abs(ky - k_expected) / k_expected
    err_z = abs(kz - k_expected) / k_expected
    assert err_y <= tol, f"ky={ky:.6f} err {err_y:.2e} (tol {tol:.0e})"
    assert err_z <= tol, f"kz={kz:.6f} err {err_z:.2e} (tol {tol:.0e})"


def test_shear_kappa_convergence():
    """Refining the section mesh drives the Saint-Venant k error toward zero."""
    from EasyFEA.Geoms import Line

    mesher = Mesher()
    line = Line(Point(0, 0), Point(x=10), 1)
    k_expected = 5 / 6

    errors = []
    for meshSize in [2.0, 1.0, 0.5, 0.25]:
        section = mesher.Mesh_2D(Domain(Point(-5, -5), Point(5, 5), meshSize=meshSize))
        beam = Models.Beam.Isotropic(2, line, section, 210e9, 0.3)
        errors.append(abs(beam._Get_shear_kappa("y") - k_expected) / k_expected)

    # error should decrease (or already be at machine precision)
    for fine, coarse in zip(errors[1:], errors[:-1]):
        assert fine <= coarse + 1e-12, f"non-monotonic convergence: {errors}"
    assert errors[-1] <= 5e-4, f"finest error {errors[-1]:.2e} too large; got {errors}"


def test_shear_kappa_axis_error():
    """_shear_kappa raises on invalid axis."""
    from EasyFEA.Geoms import Line

    mesher = Mesher()
    section = _rect_section(mesher, 10.0, 10.0)
    beam = Models.Beam.Isotropic(
        2, Line(Point(0, 0), Point(x=10), 1), section, 210e9, 0.3
    )
    with pytest.raises(ValueError):
        beam._Get_shear_kappa("x")


# -------------------------------------------------------
# Frame / projection test: L-shape (axis-aligned beams + rigid joint)
# -------------------------------------------------------


@pytest.mark.parametrize("useTimoshenko", [False, True], ids=["EB", "Timoshenko"])
@pytest.mark.parametrize("elemType", ELEM_TYPES)
def test_L_frame(elemType: str, useTimoshenko: bool):
    """Two-beam L-frame with rigid joint, clamped at origin, horizontal tip
    force at the corner — exercises BeamStructure's projection matrix and the
    add_connection_fixed Lagrange constraint.

    Geometry:                           Loading:
            F → o (L, L)                F in +x at (L, L)
                |
                | beam 2 (vertical)
                |
        o-------o (L, 0)                rigid joint at (L, 0)
        ^ clamp
        (0, 0)

    Analytical solution:
      Beam 1 (horizontal): inherits constant moment Mz = -F·L from beam 2's
      tip force; tip x deflection F·L/EA (axial), tip rotation -F·L²/EI,
      tip y deflection ±F·L³/(2EI) (sign depends on moment-sign convention,
      compared as magnitude).

      Beam 2 (vertical, cantilever from base): tip x deflection from bending
      F·L³/(3EI), plus rigid-body x translation from base rotation -rz·L =
      F·L³/EI, plus Timoshenko shear F·L/(kGA).

      Total at (L, L):
        ux = F·L/EA + 4·F·L³/(3·EI) + (F·L/kGA if Timoshenko else 0)
        |uy| = F·L³/(2·EI)
    """
    L = 10.0
    b, h = 0.5, 0.5
    E, v = 210e9, 0.3
    F = 1000.0
    A = b * h
    Iz = b * h**3 / 12

    mesher = Mesher()
    section = _rect_section(mesher, b, h)
    line1 = Line(Point(0, 0), Point(L, 0), L / 10)
    line2 = Line(Point(L, 0), Point(L, L), L / 10)
    beam1 = Models.Beam.Isotropic(2, line1, section, E, v)
    beam2 = Models.Beam.Isotropic(2, line2, section, E, v)
    kGA = _timoshenko_kGA(beam2, E, v, A)

    mesh = mesher.Mesh_Beams([beam1, beam2], elemType=elemType)
    structure = Models.Beam.BeamStructure([beam1, beam2])
    simu = Simulations.Beam(
        mesh, structure, useTimoshenko=useTimoshenko, verbosity=False
    )
    clamp = simu.mesh.Nodes_Point(Point(0, 0))
    corner = simu.mesh.Nodes_Point(Point(L, 0))
    tip = simu.mesh.Nodes_Point(Point(L, L))

    simu.add_dirichlet(clamp, [0, 0, 0], ["x", "y", "rz"])
    simu.add_connection_fixed(corner)  # weld the two beams at the corner
    simu.add_neumann(tip, [F], ["x"])
    simu.Solve()

    ux_tip = simu.Result("ux", nodeValues=True)[tip][0]
    uy_tip = simu.Result("uy", nodeValues=True)[tip][0]

    shear_term = F * L / kGA if useTimoshenko else 0.0
    ux_ex = F * L / (E * A) + 4 * F * L**3 / (3 * E * Iz) + shear_term
    uy_ex_mag = F * L**3 / (2 * E * Iz)

    # SEG2 has linear v → O(h²) convergence; SEG3+ exact for this load case.
    tol = 1e-3 if elemType == "SEG2" else 1e-8

    err_ux = abs(ux_tip - ux_ex) / abs(ux_ex)
    err_uy = abs(abs(uy_tip) - uy_ex_mag) / uy_ex_mag
    assert err_ux <= tol, f"ux error {err_ux:.2e} (tol {tol:.0e})"
    assert err_uy <= tol, f"|uy| error {err_uy:.2e} (tol {tol:.0e})"


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
