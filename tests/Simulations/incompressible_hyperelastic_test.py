# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import itertools
from typing import Callable

import numpy as np
import pytest
from scipy import linalg, sparse

from EasyFEA import ElemType, Mesh, Models, Simulations
from EasyFEA.FEM import GroupElemFactory, Kinematics, MatrixType, Operators
from EasyFEA.Geoms import Domain

from _numerics import rel_error
from ._simus import PRESSURE, incompressible

U = Simulations.HyperElastic.problemType


def _square(elemType: ElemType, layers: int = 2) -> Mesh:
    contour = Domain((0, 0), (1, 1), 1 / layers)
    return contour.Mesh_2D([], elemType, isOrganised=True)


def _cube(elemType: ElemType, layers: int = 1) -> Mesh:
    contour = Domain((0, 0), (1, 1), 1 / layers)
    return contour.Mesh_Extrude(
        [],
        [0, 0, 1],
        [layers],
        elemType,
        isOrganised=True,
    )


def _kuhn_cube(layers: int = 2) -> Mesh:
    """TETRA4 unit cube, each cell split in 6 tets around its main diagonal, no tet without an interior vertex."""
    n = layers + 1
    coord = np.array(list(itertools.product(range(n), repeat=3))) / layers
    connect = []
    for cell in itertools.product(range(layers), repeat=3):
        for axes in itertools.permutations(np.eye(3, dtype=int)):
            tet = np.cumsum([np.array(cell), *axes], axis=0) @ [n * n, n, 1]
            connect.append(tet if np.linalg.det(axes) > 0 else tet[[0, 1, 3, 2]])
    groupElem = GroupElemFactory.Create(ElemType.TETRA4, np.array(connect), coord)
    return Mesh({ElemType.TETRA4: groupElem})


def _kuhn_cube_P2() -> Mesh:
    return _kuhn_cube()._With_FunctionSpace(U, 2)


def _kuhn_cube_TETRA10() -> Mesh:
    return _kuhn_cube_P2()._Get_FunctionSpace(U).Get_mesh()


# ----------------------------------------------
# Construction
# ----------------------------------------------


def test_IncompressibleHyperElastic_puts_p_on_the_vertices_of_u_elements(subtests):
    cases = (
        ("TRI6", _square(ElemType.TRI6), ElemType.TRI3),
        ("QUAD9", _square(ElemType.QUAD9), ElemType.QUAD4),
        ("TETRA10", _cube(ElemType.TETRA10), ElemType.TETRA4),
        ("HEXA27", _cube(ElemType.HEXA27), ElemType.HEXA8),
        ("PRISM18", _cube(ElemType.PRISM18), ElemType.PRISM6),
        (
            "P2 on TETRA4",
            _cube(ElemType.TETRA4)._With_FunctionSpace(U, 2),
            ElemType.TETRA4,
        ),
    )
    for name, mesh, pressureElem in cases:
        with subtests.test(name):
            simu = incompressible(mesh)
            space = simu._Get_FunctionSpace(PRESSURE)
            groupElem = simu.mesh.Get_list_groupElem(simu.dim)[0]
            vertices = groupElem.connect[:, : groupElem.Nvertex]

            assert space.Get_list_groupElem(simu.dim)[0].elemType == pressureElem
            assert simu.pressure.shape == (np.unique(vertices).size,)
            assert simu._Get_monolithic_problems() == [(U, PRESSURE)]


def test_IncompressibleHyperElastic_leaves_the_given_mesh_on_its_spaces():
    mesh = _square(ElemType.TRI6)

    simu = incompressible(mesh)

    assert simu.mesh is not mesh
    assert mesh._Get_FunctionSpace(PRESSURE).Nn == mesh.Nn
    assert simu._Get_FunctionSpace(PRESSURE).Nn < mesh.Nn


def test_IncompressibleHyperElastic_refuses_a_u_without_a_stable_pressure_space(
    subtests,
):
    cases = (
        ("TRI3", _square(ElemType.TRI3)),
        ("QUAD4", _square(ElemType.QUAD4)),
        ("TETRA4", _cube(ElemType.TETRA4)),
        ("HEXA8", _cube(ElemType.HEXA8)),
        ("PRISM6", _cube(ElemType.PRISM6)),
        ("QUAD8", _square(ElemType.QUAD8)),
        ("HEXA20", _cube(ElemType.HEXA20)),
        ("PRISM15", _cube(ElemType.PRISM15)),
    )
    for name, mesh in cases:
        with subtests.test(name):
            with pytest.raises(ValueError, match=f"pressure space for u on {name};"):
                incompressible(mesh)


def test_IncompressibleHyperElastic_refuses_a_1D_mesh():
    coord = np.array([[0.0, 0, 0], [1, 0, 0], [0.5, 0, 0]])
    groupElem = GroupElemFactory.Create(ElemType.SEG3, np.array([[0, 1, 2]]), coord)
    mesh = Mesh({ElemType.SEG3: groupElem})

    with pytest.raises(ValueError, match="1D"):
        incompressible(mesh)


def test_IncompressibleHyperElastic_refuses_a_mesh_carrying_another_p_space(subtests):
    meshP2 = _square(ElemType.TRI3)._With_FunctionSpace(U, 2)
    cases = (
        ("P2 p on TRI3", meshP2._With_FunctionSpace(PRESSURE, 2)),
        ("P3 p on TRI6", _square(ElemType.TRI6)._With_FunctionSpace(PRESSURE, 3)),
    )
    for name, mesh in cases:
        with subtests.test(name):
            with pytest.raises(ValueError, match="another pressure space"):
                incompressible(mesh)


def test_IncompressibleHyperElastic_accepts_a_mesh_already_carrying_the_pair():
    mesh = _square(ElemType.TRI6)._With_FunctionSpace(PRESSURE, 1)

    simu = incompressible(mesh)

    assert simu.pressure.size == mesh._Get_FunctionSpace(PRESSURE).Nn


def test_IncompressibleHyperElastic_mesh_setter_derives_p_on_the_new_mesh():
    simu = incompressible(_square(ElemType.TRI6))
    mesh = _square(ElemType.TRI6, layers=4)

    simu.mesh = mesh

    assert simu.pressure.size == 25
    with pytest.raises(ValueError, match="pressure space for u on TRI3;"):
        simu.mesh = _square(ElemType.TRI3)


def _holzapfel_ogden(K: float) -> Models.HyperElastic.HolzapfelOgden:
    """Both fibres along x: transversely isotropic, so a stretch along x stays uniaxial."""
    return Models.HyperElastic.HolzapfelOgden(
        3,
        C0=1.0,
        C1=1.0,
        C2=1.0,
        C3=1.0,
        C4=1.0,
        C5=1.0,
        C6=1.0,
        C7=1.0,
        K=K,
        Mu1=1.0,
        Mu2=1.0,
        T1=np.array([1.0, 0, 0]),
        T2=np.array([1.0, 0, 0]),
    )


def test_IncompressibleHyperElastic_refuses_a_law_with_a_bulk_term(subtests):
    mesh = _cube(ElemType.HEXA27)
    cases = (
        ("MooneyRivlin", Models.HyperElastic.MooneyRivlin(3, 1.0, 1.0, K=2.0)),
        ("CiarletGeymonat", Models.HyperElastic.CiarletGeymonat(3, 1.0, 1.0, K=2.0)),
        (
            "SaintVenantKirchhoff",
            Models.HyperElastic.SaintVenantKirchhoff(3, 1.0, 1.0, K=2.0),
        ),
        ("HolzapfelOgden", _holzapfel_ogden(2.0)),
        (
            "one entry of an array",
            Models.HyperElastic.MooneyRivlin(3, 1.0, 1.0, K=np.array([0, 0, 2.0])),
        ),
    )
    for name, material in cases:
        with subtests.test(name):
            with pytest.raises(ValueError, match="bulk term"):
                Simulations.IncompressibleHyperElastic(mesh, material)


def test_IncompressibleHyperElastic_leaves_AutoDiff_unchecked():
    jax = pytest.importorskip("jax")

    def W_with_bulk_term(C: jax.Array) -> jax.Array:
        return 1e3 * (jax.numpy.sqrt(jax.numpy.linalg.det(C)) - 1) ** 2

    material = Models.HyperElastic.AutoDiff(3, W_with_bulk_term)

    simu = Simulations.IncompressibleHyperElastic(_cube(ElemType.HEXA27), material)

    assert simu.model is material


def test_IncompressibleHyperElastic_add_dirichlet_on_p_raises(subtests):
    simu = incompressible(_square(ElemType.TRI6))
    nodes = simu.mesh.Nodes_Conditions(lambda x, y, z: x == 0)

    with subtests.test("by unknown"):
        with pytest.raises(ValueError, match="p is computed"):
            simu.add_dirichlet(nodes, [0.0], ["p"])
    with subtests.test("by problem type"):
        with pytest.raises(ValueError, match="p is computed"):
            simu.add_dirichlet(nodes, [0.0], ["p"], PRESSURE)
    with subtests.test("mixed with u"):
        with pytest.raises(ValueError, match="p is computed"):
            simu.add_dirichlet(nodes, [0.0, 0.0], ["x", "p"])


# ----------------------------------------------
# Solve and results
# ----------------------------------------------


def _beside(quad: ElemType, tri: ElemType) -> Mesh:
    """Quads on [0, 1]², triangles on [1, 2] × [0, 1], one group each."""
    left = Domain((0, 0), (1, 1), 0.5)
    right = Domain((1, 0), (2, 1), 0.5)
    quads = left.Mesh_2D([], quad, isOrganised=True)
    tris = right.Mesh_2D([], tri, isOrganised=True)
    return Mesh.Merge([quads, tris])


def _hexa_beside_prism(hexa: ElemType, prism: ElemType) -> Mesh:
    """Unit cube: hexahedra for x < 0.5, prisms beyond, sharing quad faces."""
    left = Domain((0, 0), (0.5, 1), 0.5)
    right = Domain((0.5, 0), (1, 1), 0.5)
    hexas = left.Mesh_Extrude(
        [],
        [0, 0, 1],
        [2],
        hexa,
        isOrganised=True,
    )
    prisms = right.Mesh_Extrude(
        [],
        [0, 0, 1],
        [2],
        prism,
        isOrganised=True,
    )
    return Mesh.Merge([hexas, prisms])


def _tets_under_prisms(tetra: ElemType, prism: ElemType) -> Mesh:
    """Unit cube: tetrahedra for z < 0.5, prisms above, sharing triangular faces."""
    below = Domain((0, 0, 0), (1, 1, 0), 0.5)
    above = Domain((0, 0, 0.5), (1, 1, 0.5), 0.5)
    tets = below.Mesh_Extrude(
        [],
        (0, 0, 0.5),
        [1],
        tetra,
        isOrganised=True,
    )
    prisms = above.Mesh_Extrude(
        [],
        (0, 0, 0.5),
        [1],
        prism,
        isOrganised=True,
    )
    return Mesh.Merge([tets, prisms])


PAIRS = {
    "TRI6": lambda: _square(ElemType.TRI6),
    "QUAD9": lambda: _square(ElemType.QUAD9),
    "TETRA10": lambda: _cube(ElemType.TETRA10, 2),
    "HEXA27": lambda: _cube(ElemType.HEXA27, 2),
    "PRISM18": lambda: _cube(ElemType.PRISM18),
    "P2 on TETRA4": lambda: _cube(ElemType.TETRA4, 2)._With_FunctionSpace(U, 2),
    "TRI6+QUAD9": lambda: _beside(ElemType.QUAD9, ElemType.TRI6),
    "HEXA27+PRISM18": lambda: _hexa_beside_prism(ElemType.HEXA27, ElemType.PRISM18),
    "P2 on TETRA4+PRISM6": lambda: _tets_under_prisms(
        ElemType.TETRA4, ElemType.PRISM6
    )._With_FunctionSpace(U, 2),
}
"""The 5 Taylor–Hood pairs over their geometry, P2 u on a linear mesh, and the mixed groups."""


def test_IncompressibleHyperElastic_mixed_pairs_hold_two_conforming_groups(subtests):
    for name in ("TRI6+QUAD9", "HEXA27+PRISM18", "P2 on TETRA4+PRISM6"):
        with subtests.test(name):
            mesh = PAIRS[name]()
            groups = mesh.Get_list_groupElem(mesh.dim)
            assert len(groups) == 2
            shared = np.intersect1d(groups[0].nodes, groups[1].nodes)
            assert shared.size > 0


K_LAW = 1.0
STRETCH = 1.1


def _uniaxial_stretch(simu: Simulations.IncompressibleHyperElastic) -> None:
    """Rollers on the x, y (and z) = 0 sides, the x = max side pulled to `STRETCH` times the length, the others free."""
    mesh = simu.mesh
    length = mesh.coord[:, 0].max()
    nodesPulled = mesh.Nodes_Conditions(lambda x, y, z: x == length)
    simu.add_dirichlet(nodesPulled, [(STRETCH - 1) * length], ["x"])
    for index, unknown in enumerate(simu.Get_unknowns()):
        nodes = mesh.Nodes_Conditions(lambda *xyz: xyz[index] == 0)
        simu.add_dirichlet(nodes, [0], [unknown])


def _exact_stretch(dim: int) -> tuple[float, float, float]:
    """Transverse stretch, p and S_xx of the free uniaxial state, S_yy = 0."""
    t = 1 / STRETCH if dim == 2 else STRETCH**-0.5
    I1 = STRETCH**2 + t**2 + (1 if dim == 2 else t**2)
    p = 2 * K_LAW * (t**2 - I1 / 3)
    Sxx = 2 * K_LAW * (1 - I1 / 3 * STRETCH**-2) - p * STRETCH**-2
    return t, p, Sxx


def _exact_stretch_u(simu: Simulations.IncompressibleHyperElastic) -> np.ndarray:
    """u of the free uniaxial state, whatever the law: fixed by J = 1 and the symmetry."""
    coord = simu._Get_FunctionSpace(U).coord[:, : simu.dim]
    expected = (_exact_stretch(simu.dim)[0] - 1) * coord
    expected[:, 0] = (STRETCH - 1) * coord[:, 0]
    return expected.ravel()


def test_IncompressibleHyperElastic_Solve_recovers_a_uniaxial_stretch_exactly(subtests):
    for name, build in PAIRS.items():
        with subtests.test(name):
            simu = incompressible(build(), K_LAW)
            _uniaxial_stretch(simu)

            u = simu.Solve()

            _, p, Sxx = _exact_stretch(simu.dim)
            assert np.array_equal(u, simu.displacement)
            assert np.abs(u - _exact_stretch_u(simu)).max() < 1e-10
            assert np.abs(simu.pressure - p).max() < 1e-10
            assert np.abs(simu.Result("p") - p).max() < 1e-10
            assert simu.Result("p").shape == (simu.mesh.Nn,)
            assert np.abs(simu.Result("Sxx", nodeValues=False) - Sxx).max() < 1e-10
            assert np.abs(simu.Result("Syy", nodeValues=False)).max() < 1e-10


def test_IncompressibleHyperElastic_accepts_a_law_without_a_bulk_term(subtests):
    cases = (
        ("NeoHookean, K = μ/2", Models.HyperElastic.NeoHookean(3, 2.0)),
        ("MooneyRivlin", Models.HyperElastic.MooneyRivlin(3, 1.0, 1.0)),
        ("CiarletGeymonat", Models.HyperElastic.CiarletGeymonat(3, 1.0, 1.0)),
        ("SaintVenantKirchhoff", Models.HyperElastic.SaintVenantKirchhoff(3, 1.0, 1.0)),
        ("HolzapfelOgden", _holzapfel_ogden(0.0)),
        (
            "array of zeros",
            Models.HyperElastic.MooneyRivlin(3, 1.0, 1.0, K=np.zeros(1)),
        ),
        ("NeoHookean per element", Models.HyperElastic.NeoHookean(3, np.array([2.0]))),
    )
    for name, material in cases:
        with subtests.test(name):
            mesh = _cube(ElemType.HEXA27)
            simu = Simulations.IncompressibleHyperElastic(mesh, material)
            _uniaxial_stretch(simu)

            u = simu.Solve()

            assert np.abs(u - _exact_stretch_u(simu)).max() < 1e-10


def test_IncompressibleHyperElastic_Result_p_lifts_p_exactly(subtests):
    # p linear in x, y, z: P1 ⊂ P2, so the lift is the same linear field at every node
    a = np.array([0.5, -2.0, 0.25])
    for name, build in PAIRS.items():
        with subtests.test(name):
            simu = incompressible(build())
            pCoord = simu._Get_FunctionSpace(PRESSURE).coord
            nodes = simu._Get_FunctionSpace(PRESSURE).ownedNodes
            simu._Set_solutions(PRESSURE, 1 + pCoord[nodes] @ a)

            values = simu.Result("p")

            assert values.shape == (simu.mesh.Nn,)
            assert np.abs(values - (1 + simu.mesh.coord @ a)).max() < 1e-14


def test_IncompressibleHyperElastic_Result_p_on_elements_is_the_vertex_mean(subtests):
    a = np.array([0.5, -2.0, 0.25])
    for name, build in PAIRS.items():
        with subtests.test(name):
            simu = incompressible(build())
            space = simu._Get_FunctionSpace(PRESSURE)
            simu._Set_solutions(PRESSURE, 1 + space.coord[space.ownedNodes] @ a)

            values = simu.Result("p", nodeValues=False)

            expected = [
                np.mean(
                    (1 + simu.mesh.coord @ a)[
                        groupElem.connect[:, : groupElem.Nvertex]
                    ],
                    axis=1,
                )
                for groupElem in simu.mesh.Get_list_groupElem(simu.dim)
            ]
            assert np.abs(values - np.concatenate(expected)).max() < 1e-14


def test_IncompressibleHyperElastic_Results_list_p_as_a_node_field():
    simu = incompressible(_square(ElemType.TRI6))

    assert "p" in simu.Results_Available()
    assert "p" in simu.Results_nodeFields_elementFields()[0]


def test_IncompressibleHyperElastic_Save_Iter_stores_p_by_name(subtests, tmp_path):
    for folder in ("", str(tmp_path)):
        with subtests.test(folder=folder):
            simu = incompressible(_square(ElemType.TRI6))
            simu.folder = folder
            _uniaxial_stretch(simu)
            simu.Solve()
            simu.Save_Iter()
            first_u, first_p = simu.displacement, simu.pressure

            simu._Set_solutions(PRESSURE, first_p + 1.0)
            simu.Save_Iter()

            simu.Set_Iter(0)
            assert np.array_equal(simu.displacement, first_u)
            assert np.array_equal(simu.pressure, first_p)
            assert np.abs(simu.Result("p", iter=1) - (first_p[0] + 1.0)).max() < 1e-10
            assert np.array_equal(simu.pressure, first_p + 1.0)


def test_IncompressibleHyperElastic_P2_on_TETRA4_is_the_straight_TETRA10_under_loads(
    subtests,
):
    def solved(mesh: Mesh, load) -> Simulations.IncompressibleHyperElastic:
        simu = incompressible(mesh)
        nodesLeft = simu.mesh.Nodes_Conditions(lambda x, y, z: x == 0)
        nodesRight = simu.mesh.Nodes_Conditions(lambda x, y, z: x == 1)
        simu.add_dirichlet(nodesLeft, [0, 0, 0], ["x", "y", "z"])
        load(simu, nodesRight)
        simu.Solve()
        return simu

    for name, load in (
        ("surface", lambda simu, nodes: simu.add_surfLoad(nodes, [0.05], ["y"])),
        ("pressure", lambda simu, nodes: simu.add_pressureLoad(nodes, 0.05)),
    ):
        with subtests.test(name):
            simu = solved(_cube(ElemType.TETRA4, 2)._With_FunctionSpace(U, 2), load)
            iso = solved(_cube(ElemType.TETRA10, 2), load)
            coord = simu._Get_FunctionSpace(U).Get_mesh().coord
            rows = np.argmin(
                np.linalg.norm(coord[:, None] - iso.mesh.coord, axis=2), axis=1
            )
            dofs = (rows[:, None] * 3 + np.arange(3)).ravel()

            assert rel_error(simu.displacement, iso.displacement[dofs]) < 1e-10
            assert (
                rel_error(simu.Result("p"), iso.Result("p")[rows[: simu.mesh.Nn]])
                < 1e-10
            )


# ----------------------------------------------
# Tangent
# ----------------------------------------------


def _assembled(
    simu: Simulations.IncompressibleHyperElastic, x: np.ndarray
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Tangent and right-hand side of ``[u | p]`` at `x`, standing in for Newton's current solution."""
    size = simu.displacement.size
    fields = {U: x[:size], PRESSURE: x[size:]}
    simu._Solver_Get_Newton_Raphson_current_solution = (  # type: ignore [method-assign]
        lambda problemType=None: fields[problemType or U]
    )
    K, _, _, F = simu.Assembly((U, PRESSURE))
    return K, F.toarray().ravel()


def _fd(
    simu: Simulations.IncompressibleHyperElastic,
    x: np.ndarray,
    dx: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """∂(internal force)/∂x · dx by central differences: F is the negated internal force."""
    _, Fp = _assembled(simu, x + eps * dx)
    _, Fm = _assembled(simu, x - eps * dx)
    return -(Fp - Fm) / (2 * eps)


def test_IncompressibleHyperElastic_tangent_matches_fd_by_block_and_whole(subtests):
    for name, build in PAIRS.items():
        simu = incompressible(build())
        nu, np_ = simu.displacement.size, simu.pressure.size
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.standard_normal(nu) * 0.02, rng.standard_normal(np_)])
        du = np.concatenate([rng.standard_normal(nu), np.zeros(np_)])
        dp = np.concatenate([np.zeros(nu), rng.standard_normal(np_)])
        K, _ = _assembled(simu, x)
        fd_u, fd_p = _fd(simu, x, du), _fd(simu, x, dp)
        Kdu, Kdp = K @ du, K @ dp

        cases = (
            ("K_uu", Kdu[:nu], fd_u[:nu]),
            ("K_pu", Kdu[nu:], fd_u[nu:]),
            ("K_up", Kdp[:nu], fd_p[:nu]),
            ("whole", Kdu + Kdp, _fd(simu, x, du + dp)),
        )
        for block, tangent, expected in cases:
            with subtests.test(f"{name} {block}"):
                assert rel_error(tangent, expected) < 1e-7
        with subtests.test(f"{name} K_pp"):
            assert np.abs(fd_p[nu:]).max() < 1e-7
            assert np.abs(K[nu:, nu:]).max() == 0.0
        with subtests.test(f"{name} K_pu = K_upᵀ"):
            K_up, K_pu = K[:nu, nu:], K[nu:, :nu]
            assert abs(K_up.T - K_pu).max() <= 1e-14 * abs(K_pu).max()


# ----------------------------------------------
# Inf-sup
# ----------------------------------------------


def _scatter(
    A_e: np.ndarray, rows_e: np.ndarray, cols_e: np.ndarray, shape: tuple[int, int]
) -> sparse.csr_matrix:
    """Element matrices (Ne, nRows, nCols) summed at their global rows and columns."""
    rows = np.repeat(rows_e, cols_e.shape[1], axis=1).ravel()
    cols = np.tile(cols_e, (1, rows_e.shape[1])).ravel()
    return sparse.csr_matrix((np.ravel(A_e), (rows, cols)), shape=shape)


def _beta(
    B: sparse.csr_matrix, K: sparse.csr_matrix, M: sparse.csr_matrix, free: np.ndarray
) -> tuple[float, int]:
    """β_h = √(first non-zero eigenvalue of B K⁻¹ Bᵀ against M_p) on the free u DOFs, and the number of zero eigenvalues."""
    B_free = B[:, free].toarray()
    S = B_free @ linalg.solve(K[free][:, free].toarray(), B_free.T, assume_a="sym")
    eigenvalues = linalg.eigh(S, M.toarray(), eigvals_only=True)
    zeros = int((eigenvalues < 1e-8 * eigenvalues.max()).sum())
    return float(np.sqrt(eigenvalues[zeros])), zeros


def _inf_sup(
    mesh: Mesh,
    isClamped: Callable[[np.ndarray], np.ndarray],
    isDistorted: bool = False,
) -> tuple[float, int]:
    """`_beta` of the assembled ``K_pu`` at u = 0, against the vector H¹ seminorm and p's mass, with u = 0 where `isClamped(coord)`."""
    simu = incompressible(mesh)
    mesh = simu.mesh
    if isDistorted:
        mesh.coord = _smooth_map(mesh.coord)
    space_u = simu._Get_FunctionSpace(U)
    space_p = simu._Get_FunctionSpace(PRESSURE)
    dim = simu.dim
    nu, np_ = space_u.Nn * dim, space_p.Nn

    K_pu = _assembled(simu, np.zeros(nu + np_))[0][nu:, :nu]
    H1 = sparse.csr_matrix((nu, nu))
    M = sparse.csr_matrix((np_, np_))
    for groupElem, uGroup, pGroup in zip(
        mesh.Get_list_groupElem(dim),
        space_u.Get_list_groupElem(dim),
        space_p.Get_list_groupElem(dim),
    ):
        assembly_u = space_u.Get_assembly_e(groupElem, dim)
        assembly_p = space_p.Get_assembly_e(groupElem, 1)
        H1 = H1 + _scatter(
            Operators.Bilinear.GradUGradV(uGroup, dof_n=dim),
            assembly_u,
            assembly_u,
            (nu, nu),
        )
        M = M + _scatter(
            Operators.Bilinear.UV(pGroup),
            assembly_p,
            assembly_p,
            (np_, np_),
        )

    free = np.flatnonzero(~np.repeat(isClamped(space_u.coord), dim))
    return _beta(K_pu, H1, M, free)


def _smooth_map(coord: np.ndarray) -> np.ndarray:
    """Non-affine map of the unit square or cube onto itself, its boundary fixed."""
    dim = 2 if np.allclose(coord[:, 2], 0) else 3
    x, y = coord[:, 0], coord[:, 1]
    f = np.prod(np.sin(np.pi * coord[:, :dim]), axis=1)
    mapped = coord.copy()
    mapped[:, 0] += 0.15 * f * np.cos(np.pi * y)
    mapped[:, 1] += 0.15 * f * np.cos(np.pi * x)
    return mapped


def _whole_boundary(coord: np.ndarray) -> np.ndarray:
    isPlanar = np.allclose(coord[:, 2], 0)
    xyz = coord[:, :2] if isPlanar else coord
    return np.any((np.abs(xyz) < 1e-9) | (np.abs(xyz - 1) < 1e-9), axis=1)


def _left_face(coord: np.ndarray) -> np.ndarray:
    return np.abs(coord[:, 0]) < 1e-9


def test_IncompressibleHyperElastic_pairs_are_inf_sup_stable(subtests):
    # u = 0 on the whole boundary: only the constant p is a zero mode
    for name, build, expected in (
        ("TRI6", lambda: _square(ElemType.TRI6, 4), 0.368),
        ("QUAD9", lambda: _square(ElemType.QUAD9, 4), 0.475),
        ("HEXA27", lambda: _cube(ElemType.HEXA27, 2), 0.309),
        ("PRISM18", lambda: _cube(ElemType.PRISM18, 2), 0.208),
        ("TETRA10", _kuhn_cube_TETRA10, 0.173),
        ("P2 on TETRA4", _kuhn_cube_P2, 0.173),
        (
            "HEXA27|PRISM18",
            lambda: _hexa_beside_prism(ElemType.HEXA27, ElemType.PRISM18),
            0.208,
        ),
    ):
        with subtests.test(name):
            beta, zeros = _inf_sup(build(), _whole_boundary)
            assert zeros == 1
            assert abs(beta / expected - 1) < 0.03


def test_IncompressibleHyperElastic_pairs_stay_stable_on_distorted_cells(subtests):
    # β depends on which way the map shears a triangle's diagonal: looser where the mesh has them
    for name, build, expected, tol in (
        ("QUAD9", lambda: _square(ElemType.QUAD9, 4), 0.461, 0.03),
        ("HEXA27", lambda: _cube(ElemType.HEXA27, 2), 0.304, 0.03),
        ("TETRA10", _kuhn_cube_TETRA10, 0.179, 0.03),
        ("TRI6", lambda: _square(ElemType.TRI6, 4), 0.354, 0.10),
        ("PRISM18", lambda: _cube(ElemType.PRISM18, 2), 0.214, 0.10),
        (
            "HEXA27|PRISM18",
            lambda: _hexa_beside_prism(ElemType.HEXA27, ElemType.PRISM18),
            0.214,
            0.10,
        ),
    ):
        with subtests.test(name):
            beta, zeros = _inf_sup(build(), _whole_boundary, isDistorted=True)
            assert zeros == 1
            assert abs(beta / expected - 1) < tol


def test_IncompressibleHyperElastic_tets_beside_prisms_are_stable_under_one_clamped_face(
    subtests,
):
    # an organised cube has a tet with no interior vertex, a second zero mode when the whole boundary is clamped; one clamped face has none, on other diagonals than the expected β's
    for name, build, expected in (
        (
            "TETRA10|PRISM18",
            lambda: _tets_under_prisms(ElemType.TETRA10, ElemType.PRISM18),
            0.561,
        ),
        (
            "P2 on TETRA4|PRISM6",
            lambda: _tets_under_prisms(
                ElemType.TETRA4, ElemType.PRISM6
            )._With_FunctionSpace(U, 2),
            0.561,
        ),
    ):
        with subtests.test(name):
            beta, zeros = _inf_sup(build(), _left_face)
            assert zeros == 0
            assert abs(beta / expected - 1) < 0.06


def test_equal_order_QUAD4_is_the_inf_sup_negative_control():
    mesh = _square(ElemType.QUAD4, 4)
    groupElem = mesh.groupElem
    assembly_u = groupElem.Get_assembly_e(2)
    connect = groupElem.connect
    u = np.zeros(mesh.Nn * 2)
    kinematics = Kinematics(groupElem, u, MatrixType.Order(4))
    K_pu_e = Operators.NonLinear.MixedIncompressibility(
        kinematics, groupElem, np.zeros(mesh.Nn)
    )[3]
    H1_e = Operators.Bilinear.GradUGradV(groupElem, dof_n=2)
    M_e = Operators.Bilinear.UV(groupElem)
    shape_u, shape_p = (mesh.Nn * 2, mesh.Nn * 2), (mesh.Nn, mesh.Nn)

    beta, zeros = _beta(
        _scatter(K_pu_e, connect, assembly_u, (mesh.Nn, mesh.Nn * 2)),
        _scatter(H1_e, assembly_u, assembly_u, shape_u),
        _scatter(M_e, connect, connect, shape_p),
        np.flatnonzero(~np.repeat(_whole_boundary(mesh.coord), 2)),
    )

    assert zeros == 8
    assert abs(beta / 0.192 - 1) < 0.03


def test_IncompressibleHyperElastic_keeps_the_constant_p_mode_on_curved_tets():
    # a clamped body keeps its volume whatever p: the coupling must integrate exactly, ∫ div v dΩ = 0
    simu = incompressible(_kuhn_cube_TETRA10())
    simu.mesh.coord = _smooth_map(simu.mesh.coord)
    nu, np_ = simu.displacement.size, simu.pressure.size

    K = _assembled(simu, np.zeros(nu + np_))[0]

    isClamped = _whole_boundary(simu._Get_FunctionSpace(U).coord)
    free = np.flatnonzero(~np.repeat(isClamped, 3))
    K_up = K[free][:, nu:]
    assert np.abs(K_up @ np.ones(np_)).max() <= 1e-13 * abs(K_up).max()


def test_IncompressibleHyperElastic_stress_is_total(subtests):
    # u = 0: S = −p I
    p = 7.0
    for name, mesh in (
        ("TRI6", _square(ElemType.TRI6)),
        ("HEXA27", _cube(ElemType.HEXA27, 2)),
    ):
        with subtests.test(name):
            simu = incompressible(mesh)
            simu._Set_solutions(PRESSURE, np.full(simu.pressure.size, p))
            Sxx = simu.Result("Sxx", nodeValues=False)
            assert np.abs(Sxx + p).max() < 1e-12


def test_IncompressibleHyperElastic_Svm_of_a_pure_pressure_is_zero():
    simu = incompressible(_cube(ElemType.HEXA27, 2))
    simu._Set_solutions(PRESSURE, np.full(simu.pressure.size, 7.0))

    assert np.abs(simu.Result("Svm", nodeValues=False)).max() < 1e-12


# ----------------------------------------------
# MPI
# ----------------------------------------------


@pytest.mark.mpi
def test_IncompressibleHyperElastic_Solve_under_mpi_is_the_exact_stretch():
    simu = incompressible(_square(ElemType.TRI6, 4), K_LAW)
    mesh = simu.mesh
    nodesLeft = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesBottom = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
    nodesRight = mesh.Nodes_Conditions(lambda x, y, z: x == 1)
    simu.add_dirichlet(nodesLeft, [0], ["x"])
    simu.add_dirichlet(nodesBottom, [0], ["y"])
    simu.add_dirichlet(nodesRight, [STRETCH - 1], ["x"])

    simu.Solve()

    t, p, _ = _exact_stretch(simu.dim)
    points = simu._Get_FunctionSpace(U).ownedNodes
    coord = simu._Get_FunctionSpace(U).coord[points, :2]
    u = simu.displacement.reshape(-1, 2)[points]
    assert np.abs(u[:, 0] - (STRETCH - 1) * coord[:, 0]).max() < 1e-10
    assert np.abs(u[:, 1] - (t - 1) * coord[:, 1]).max() < 1e-10
    assert np.abs(simu.pressure - p).max() < 1e-10
