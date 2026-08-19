# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Checks the hand-written constitutive derivatives against automatic differentiation.

Each shipped potential is restated once in jax; ``grad`` and ``hessian`` are compared against
``Compute_dWde`` and ``Compute_d2Wde``. Skipped whole when jax is absent.
"""

import pickle
import subprocess
import sys

import numpy as np
import pytest

from EasyFEA import ElemType, MatrixType, Models, Simulations
from EasyFEA.FEM._linalg import FeArray
from EasyFEA.Geoms import Domain, Line
from EasyFEA.Models import _autodiff
from EasyFEA.Models.HyperElastic import HyperElasticPotential
from EasyFEA.Models.HyperElastic._state import HyperElasticState

jax = pytest.importorskip("jax")
jnp = jax.numpy

_autodiff.Enable_x64()

# every comparison below sits between 1e-16 and 1e-14; 1e-12 leaves two orders of margin
TOL = 1e-12


def _rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _state(dim: int) -> HyperElasticState:
    """A small mesh under a random displacement, large enough to strain every invariant."""
    if dim == 1:
        line = Line((0, 0), (1, 0), meshSize=0.25)
        mesh = line.Mesh_1D(ElemType.SEG2)
    elif dim == 2:
        domain = Domain((0, 0), (1, 1), meshSize=0.5)
        mesh = domain.Mesh_2D([], ElemType.QUAD4)
    else:
        domain = Domain((0, 0), (1, 1), meshSize=1)
        mesh = domain.Mesh_Extrude([], [0, 0, 1], [1], ElemType.HEXA8)
    u = np.random.default_rng(0).normal(0.0, 0.02, mesh.Nn * dim)
    return HyperElasticState(mesh.groupElem, u, MatrixType.rigi)


@pytest.fixture(scope="module")
def state() -> HyperElasticState:
    """A HEXA8 block, for the tests that do not vary the dimension."""
    return _state(3)


# ----------------------------------------------
# The five potentials, restated in jax
# ----------------------------------------------


# same names as ComputeHyperelasticLaws.py, the sympy source of the hand-written derivatives
def _invariants(C):
    """``I1, I2, I3`` of ``C``."""
    I1 = jnp.trace(C)
    return I1, (I1**2 - jnp.trace(C @ C)) / 2, jnp.linalg.det(C)


def _reduced(I1, I2, I3):
    """``J1 = I1 I3^(-1/3)``, ``J2 = I2 I3^(-2/3)``, ``J = sqrt(I3)``."""
    return I1 * I3 ** (-1 / 3), I2 * I3 ** (-2 / 3), jnp.sqrt(I3)


K, K1, K2 = 200.0, 30.0, 8.0
LMBDA, MU = 121153.0, 80769.0

# passive myocardium, Holzapfel & Ogden (2009), in Pa
HO = dict(
    C0=19.0 / 2 / 8.023,
    C1=8.023,
    C2=6157.0 / 2 / 16.026,
    C3=16.026,
    C4=827.0 / 2 / 11.12,
    C5=11.12,
    C6=72.0 / 2 / 11.436,
    C7=11.436,
    K=1e6,
    Mu1=0.0,
    Mu2=0.0,
    ks=100.0,
)
T1 = np.array([1.0, 0.0, 0.0])
T2 = np.array([0.0, 1.0, 0.0])


def _neo_hookean(C):
    J1, _, _ = _reduced(*_invariants(C))
    return K * (J1 - 3)


def _mooney_rivlin(C):
    J1, J2, J = _reduced(*_invariants(C))
    return K1 * (J1 - 3) + K2 * (J2 - 3) + K * (J - 1) ** 2


def _ciarlet_geymonat(C):
    J1, J2, J = _reduced(*_invariants(C))
    return K1 * (J1 - 3) + K2 * (J2 - 3) + K * (J - 1 - jnp.log(J))


def _saint_venant_kirchhoff(C):
    I1, I2, I3 = _invariants(C)
    return (
        I1**2 * (LMBDA / 8 + MU / 4)
        - I1 * (3 * LMBDA / 4 + MU / 2)
        - I2 * MU / 2
        + 0.5 * K * (I3 - 1) ** 2
        + 9 * LMBDA / 8
        + 3 * MU / 4
    )


def _holzapfel_ogden(C, T1, T2):
    """Two fibre directions, three anisotropic invariants, and a sigmoid tension/compression switch."""
    J1, J2, J = _reduced(*_invariants(C))
    I4 = T1 @ C @ T1
    I6 = T2 @ C @ T2
    I8 = T1 @ C @ T2

    def chi(Ii):
        return 1 / (1 + jnp.exp(-HO["ks"] * (Ii - 1)))

    return (
        HO["C0"] * (jnp.exp(HO["C1"] * (J1 - 3)) - 1)
        + HO["C2"] * chi(I4) * (jnp.exp(HO["C3"] * (I4 - 1) ** 2) - 1)
        + HO["C4"] * chi(I6) * (jnp.exp(HO["C5"] * (I6 - 1) ** 2) - 1)
        + HO["C6"] * (jnp.exp(HO["C7"] * I8**2) - 1)
        + HO["K"] / 4 * (J**2 - 1 - 2 * jnp.log(J))
        + HO["Mu1"] * (J1 - 3)
        + HO["Mu2"] * (J2 - 3)
    )


LAWS = {
    "HolzapfelOgden": (
        _holzapfel_ogden,
        lambda dim: Models.HyperElastic.HolzapfelOgden(dim, T1=T1, T2=T2, **HO),
        (T1, T2),
        (0, None, None),
    ),
    "NeoHookean": (
        _neo_hookean,
        lambda dim: Models.HyperElastic.NeoHookean(dim, K=K),
        (),
        0,
    ),
    "MooneyRivlin": (
        _mooney_rivlin,
        lambda dim: Models.HyperElastic.MooneyRivlin(dim, K1=K1, K2=K2, K=K),
        (),
        0,
    ),
    "CiarletGeymonat": (
        _ciarlet_geymonat,
        lambda dim: Models.HyperElastic.CiarletGeymonat(dim, K1=K1, K2=K2, K=K),
        (),
        0,
    ),
    "SaintVenantKirchhoff": (
        _saint_venant_kirchhoff,
        lambda dim: Models.HyperElastic.SaintVenantKirchhoff(
            dim, lmbda=LMBDA, mu=MU, K=K
        ),
        (),
        0,
    ),
}


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("law", list(LAWS))
def test_potential_derivatives_match_autodiff(law: str, dim: int):
    """``Compute_dWde`` and ``Compute_d2Wde`` are the derivatives of ``Compute_W`` they claim to be.

    Over every dimension, since nothing else asserts the ``_Slice_Vector`` / ``_Slice_Matrix`` step.
    """
    W_point, Material, aux, in_axes = LAWS[law]
    state = _state(dim)
    material = Material(dim)
    W, dWde, d2Wde = HyperElasticPotential(W_point, in_axes)

    assert _rel(W(state, *aux), material.Compute_W(state)) < TOL
    assert _rel(dWde(state, *aux), material.Compute_dWde(state)) < TOL
    assert _rel(d2Wde(state, *aux), material.Compute_d2Wde(state)) < TOL


def test_holzapfel_ogden_accepts_fibre_fields(state: HyperElasticState):
    """The fibre directions may vary per Gauss point, as a heart mesh supplies them."""
    Ne, nPg, _ = state._GetDims()
    angle = np.linspace(-np.pi / 3, np.pi / 3, Ne)[:, None] * np.ones((1, nPg))
    zero = np.zeros_like(angle)
    T1_e_pg = FeArray.asfearray(np.stack([np.cos(angle), np.sin(angle), zero], -1))
    T2_e_pg = FeArray.asfearray(np.stack([-np.sin(angle), np.cos(angle), zero], -1))

    W, dWde, d2Wde = HyperElasticPotential(_holzapfel_ogden, 0)
    material = Models.HyperElastic.HolzapfelOgden(3, T1=T1_e_pg, T2=T2_e_pg, **HO)

    assert _rel(W(state, T1_e_pg, T2_e_pg), material.Compute_W(state)) < TOL
    assert _rel(dWde(state, T1_e_pg, T2_e_pg), material.Compute_dWde(state)) < TOL
    assert _rel(d2Wde(state, T1_e_pg, T2_e_pg), material.Compute_d2Wde(state)) < TOL


# ----------------------------------------------
# The seam itself
# ----------------------------------------------


def test_kelvin_to_tensor_inverts_the_shipped_projection(state: HyperElasticState):
    """``Kelvin_to_tensor`` undoes ``Project_matrix_to_vector``."""
    C_e_pg = state.Compute_C()
    rebuilt = _autodiff.Vmap_e_pg(_autodiff.Kelvin_to_tensor)(
        Models.Project_matrix_to_vector(C_e_pg)
    )

    assert _rel(rebuilt, C_e_pg) < TOL


def test_kelvin_basis_is_orthonormal():
    """``B_I : B_J = delta_IJ``."""
    basis = _autodiff._KELVIN_BASIS
    gram = np.einsum("Iij,Jij->IJ", basis, basis)

    assert _rel(gram, np.eye(6)) < TOL


def test_vmap_returns_fearray_fields(state: HyperElasticState):
    Ne, nPg, _ = state._GetDims()
    out = _autodiff.Vmap_e_pg(lambda C: C @ C)(state.Compute_C())

    assert isinstance(out, FeArray)
    assert out.shape == (Ne, nPg, 3, 3)


def test_vmap_maps_auxiliary_fields():
    """Extra arguments are mapped like the first — this is what the fibre directions ride on."""
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(4, 3, 6)), rng.normal(size=(4, 3, 6))
    out = _autodiff.Vmap_e_pg(lambda a, b: jnp.dot(a, b))(x, y)

    assert out.shape == (4, 3)
    assert _rel(out, np.sum(x * y, axis=-1)) < TOL


def test_vmap_shares_unmapped_arguments():
    """``in_axes=None`` holds an argument constant across every point."""
    rng = np.random.default_rng(3)
    x, t = rng.normal(size=(4, 3, 6)), rng.normal(size=6)
    out = _autodiff.Vmap_e_pg(lambda a, b: jnp.dot(a, b), (0, None))(x, t)

    assert out.shape == (4, 3)
    assert _rel(out, np.sum(x * t, axis=-1)) < TOL


# ----------------------------------------------
# The AutoDiff law
# ----------------------------------------------


def _Solve(material) -> np.ndarray:
    """Stretches a HEXA8 block along z and returns the converged displacement."""
    domain = Domain((0, 0), (1, 1), meshSize=0.5)
    mesh = domain.Mesh_Extrude([], [0, 0, 1], [2], ElemType.HEXA8)
    simu = Simulations.HyperElastic(mesh, material)
    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x, y, z: z == 0), [0] * 3, ["x", "y", "z"])  # fmt: skip
    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x, y, z: z == 1), [0.05], ["z"])
    return simu.Solve()


def test_autodiff_law_solves_like_the_shipped_one():
    """An ``AutoDiff`` law drives a simulation to the same answer as the hand-written equivalent."""
    autodiff = Models.HyperElastic.AutoDiff(3, _ciarlet_geymonat)
    shipped = Models.HyperElastic.CiarletGeymonat(3, K1=K1, K2=K2, K=K)

    assert _rel(_Solve(autodiff), _Solve(shipped)) < 1e-10


def test_autodiff_law_carries_per_gauss_point_fields(state: HyperElasticState):
    """``aux`` reaches the kernel: HolzapfelOgden with a fibre direction per Gauss point."""
    Ne, nPg, _ = state._GetDims()
    angle = np.linspace(-np.pi / 3, np.pi / 3, Ne)[:, None] * np.ones((1, nPg))
    zero = np.zeros_like(angle)
    T1_e_pg = FeArray.asfearray(np.stack([np.cos(angle), np.sin(angle), zero], -1))
    T2_e_pg = FeArray.asfearray(np.stack([-np.sin(angle), np.cos(angle), zero], -1))

    autodiff = Models.HyperElastic.AutoDiff(3, _holzapfel_ogden, (T1_e_pg, T2_e_pg))
    shipped = Models.HyperElastic.HolzapfelOgden(3, T1=T1_e_pg, T2=T2_e_pg, **HO)

    assert _rel(autodiff.Compute_W(state), shipped.Compute_W(state)) < TOL
    assert _rel(autodiff.Compute_dWde(state), shipped.Compute_dWde(state)) < TOL
    assert _rel(autodiff.Compute_d2Wde(state), shipped.Compute_d2Wde(state)) < TOL


def test_autodiff_law_survives_a_pickle_round_trip(state: HyperElasticState):
    """``Simu.Save`` pickles the whole simulation, material included, so the law must survive it."""
    law = Models.HyperElastic.AutoDiff(3, _ciarlet_geymonat)
    reloaded = pickle.loads(pickle.dumps(law))

    assert _rel(reloaded.Compute_W(state), law.Compute_W(state)) < TOL
    assert _rel(reloaded.Compute_dWde(state), law.Compute_dWde(state)) < TOL
    assert _rel(reloaded.Compute_d2Wde(state), law.Compute_d2Wde(state)) < TOL


def test_importing_easyfea_does_not_pull_jax():
    """``AutoDiff`` imports jax lazily. In a subprocess, because this module imports jax itself."""
    code = "import EasyFEA, sys; sys.exit('jax' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
