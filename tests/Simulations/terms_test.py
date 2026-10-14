# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Each guard catches a failure that is otherwise silent: an unknown slot letter would surface deep inside the fold, a ``tag`` on an operator that takes no ``elements`` would assemble the whole group, and a slot count that does not match what the operator returns would drop an array or fabricate one."""

import numpy as np
import pytest

from EasyFEA import ElemType, Mesh, Models, Simulations
from EasyFEA.FEM import Operators
from EasyFEA.Geoms import Circle, Domain
from EasyFEA.Simulations import Fold_terms, ProblemType, Term

from ._simus import laplacian


def Operator(groupElem, elements=None):
    """Stand-in operator with the shape the fold calls."""
    return None


def Operator_without_elements(groupElem):
    """Stand-in operator that cannot be restricted to a subset of its group."""
    return None


def One_array(groupElem):
    """A matrix, as a single-slot operator returns it (dof_n = 1, as in Thermal)."""
    return np.zeros((groupElem.Ne, groupElem.nPe, groupElem.nPe))


def One_vector(groupElem):
    """A load vector, as an F/R slot expects it."""
    return np.zeros((groupElem.Ne, groupElem.nPe))


def Two_arrays(groupElem):
    """A matrix and a vector, as a two-slot operator returns them."""
    return One_array(groupElem), One_vector(groupElem)


def Simu() -> Simulations.Thermal:
    domain = Domain((0, 0), (1, 1))
    mesh = domain.Mesh_2D()
    return Simulations.Thermal(mesh, Models.Thermal(1, 1))


def Fold(slots: str, fn) -> dict:
    return Fold_terms(Simu(), [Term(slots, fn)])


def _inclusion_mesh() -> Mesh:
    """Rectangle tagged S0 around a filled circle tagged S1."""
    domain = Domain((0, 0), (2, 1), 0.25)
    circle = Circle((1, 0.5), 0.4, 0.1, isFilled=True)
    return domain.Mesh_2D([circle], ElemType.TRI3)


# ----------------------------------------------
# Term
# ----------------------------------------------


def test_Term_rejects_unknown_slot_letters(subtests):
    for slots in ("KG", "k", "", "KRX"):
        with subtests.test(slots=slots):
            with pytest.raises(ValueError, match="KCMFR"):
                Term(slots, Operator)


def test_Term_rejects_a_tag_the_operator_cannot_honour():
    with pytest.raises(ValueError, match="elements"):
        Term("K", Operator_without_elements, tag="epi")


def test_Term_rejects_a_tag_on_an_operator_integrating_its_whole_group():
    with pytest.raises(ValueError, match="elements"):
        Term("K", Operators.Bilinear.GradU_A_GradV, tag="epi")


def test_Term_rejects_a_dim_contradicting_its_group():
    groupElem = _inclusion_mesh().groupElem
    with pytest.raises(ValueError, match="contradicts"):
        Term("K", Operator, dim=1, groupElem=groupElem)


def test_Term_repr_names_the_operator_its_tag_and_its_group():
    groupElem = _inclusion_mesh().groupElem

    term = Term("KF", Operator, tag="S1", groupElem=groupElem)

    assert repr(term) == "Term('KF', Operator, tag='S1', groupElem=TRI3)"


def test_Term_rejects_a_constant_bound_method():
    simu = Simu()
    with pytest.raises(TypeError, match="bound method"):
        Term("K", simu.Get_x0, constant=True)


def test_Term_rejects_a_constant_argument_it_cannot_compare(subtests):
    for value in ([1.0], {"k": 1.0}, object()):
        with subtests.test(value=type(value).__name__):
            with pytest.raises(TypeError, match="constant=True"):
                Term("K", Operator, constant=True, value=value)


def test_Term_slots_are_the_declared_letters():
    for slots in ("K", "C", "M", "F", "KR", "KRC", "KF"):
        assert Term(slots, Operator).slots == tuple(slots)


def test_Term_constant_is_the_declared_flag():
    for constant in (False, True):
        assert Term("K", Operator, constant=constant).constant is constant


def test_Term_scale_is_one_until_scaled():
    term = Term("K", Operator)

    assert term.scale == 1.0
    assert term.Scaled(2.0).Scaled(3.0).scale == 6.0


def test_Term_Scaled_routes_a_copy_to_other_slots():
    term = Term("K", Operator)

    damping = term.Scaled(0.5, "C")

    assert damping.slots == ("C",)
    assert term.slots == ("K",) and term.scale == 1.0


def test_Term__Get_groups_restricts_to_the_term_group_and_tag():
    mesh = _inclusion_mesh()
    groupElem = mesh.groupElem

    assert Term("K", Operator)._Get_groups(mesh) == [groupElem]
    assert Term("K", Operator, dim=1)._Get_groups(mesh) == mesh.Get_list_groupElem(1)
    assert Term("K", Operator, tag="S1")._Get_groups(mesh) == [groupElem]
    assert Term("K", Operator, tag="missing")._Get_groups(mesh) == []


def test_Term__Get_groups_rejects_a_group_of_another_mesh():
    groupElem = _inclusion_mesh().groupElem
    term = Term("K", Operator, groupElem=groupElem)

    with pytest.raises(ValueError, match="not a group"):
        term._Get_groups(_inclusion_mesh())


def test_Term__Get_elements_is_the_tagged_subset():
    groupElem = _inclusion_mesh().groupElem

    assert Term("K", Operator)._Get_elements(groupElem) is None
    elements = Term("K", Operator, tag="S1")._Get_elements(groupElem)
    assert np.array_equal(elements, groupElem.Get_Elements_Tag("S1"))


def test_Term__Evaluate_injects_u_and_elements_the_caller_left_out():
    groupElem = _inclusion_mesh().groupElem
    u = np.arange(groupElem.Nn, dtype=float)

    def Echo(groupElem, u=None, elements=None, coef=1.0):
        return u, elements, coef

    untagged = Term("K", Echo, coef=2.0)._Evaluate(groupElem, u)
    tagged = Term("K", Echo, tag="S1", u=-u)._Evaluate(groupElem, u)

    assert untagged[0] is u and untagged[1] is None and untagged[2] == 2.0
    assert np.array_equal(tagged[0], -u)
    assert np.array_equal(tagged[1], groupElem.Get_Elements_Tag("S1"))


class Counted:
    """`Operators.Bilinear.UV`, counting its calls: an object, not a bound method, so a constant term accepts it."""

    def __init__(self):
        self.calls = 0

    def __call__(self, groupElem, coef=1.0, elements=None):
        self.calls += 1
        return Operators.Bilinear.UV(groupElem, coef=coef, elements=elements)


def test_Term__Cached_integrates_once_while_the_arguments_are_unchanged():
    simu = Simu()
    groupElem = simu.mesh.groupElem
    operator = Counted()

    first = Term("K", operator, coef=1.0, constant=True)._Cached(simu, groupElem)
    again = Term("K", operator, coef=1.0, constant=True)._Cached(simu, groupElem)
    changed = Term("K", operator, coef=2.0, constant=True)._Cached(simu, groupElem)

    assert again is first
    assert operator.calls == 2
    assert np.allclose(changed, 2 * first)


# ----------------------------------------------
# Fold_terms
# ----------------------------------------------


def test_Fold_terms_rejects_more_slots_than_arrays():
    with pytest.raises(ValueError, match="2 slot"):
        Fold("KC", One_array)


def test_Fold_terms_rejects_fewer_slots_than_arrays():
    with pytest.raises(ValueError, match="1 slot"):
        Fold("K", Two_arrays)


def test_Fold_terms_accepts_one_slot_per_array():
    assert Fold("K", One_array)
    assert Fold("KF", Two_arrays)


def test_Fold_terms_rejects_a_matrix_in_a_vector_slot():
    with pytest.raises(ValueError, match="takes a vector"):
        Fold("R", One_array)


def test_Fold_terms_rejects_a_vector_in_a_matrix_slot():
    with pytest.raises(ValueError, match="takes a matrix"):
        Fold("K", One_vector)


def test_Fold_terms_rejects_the_reversed_order():
    # "RK" instead of "KR": both letters are valid, only the order is wrong
    with pytest.raises(ValueError, match="positional"):
        Fold("RK", Two_arrays)


def test_Fold_terms_sums_tagged_masses_to_the_whole_one():
    mesh = _inclusion_mesh()
    simu = Simulations.Thermal(mesh, Models.Thermal(1, 1))
    groupElem = mesh.groupElem

    def Mass(*tags: str) -> float:
        terms = [Term("M", Operators.Bilinear.UV, tag=tag) for tag in tags]
        return Fold_terms(simu, terms)[groupElem][2].sum()

    assert np.isclose(Mass("S0", "S1"), Mass(None))
    assert np.isclose(Mass(None), mesh.area)


def test_Fold_terms_sends_an_internal_force_to_F_negated():
    simu = Simu()

    def Force(groupElem):
        return One_array(groupElem), np.ones((groupElem.Ne, groupElem.nPe))

    K_e, _, _, F_e = Fold_terms(simu, [Term("KR", Force)])[simu.mesh.groupElem]

    assert np.all(F_e == -1.0)
    assert np.all(K_e == 0.0)


def test_Fold_terms_gives_a_single_slot_term_its_residual_when_nonlinear(
    monkeypatch,
):
    mesh = Simu().mesh
    simu = Simulations.WeakForms(mesh, laplacian(mesh), isNonLinear=True)
    u = np.arange(mesh.Nn, dtype=float)
    monkeypatch.setattr(simu, "_Solver_Get_Newton_Raphson_current_solution", lambda: u)
    groupElem = mesh.groupElem

    terms = [Term("K", Operators.Bilinear.GradUGradV)]
    K_e, _, _, F_e = Fold_terms(simu, terms)[groupElem]

    u_e = groupElem.Locates_sol_e(u, 1)
    assert np.abs(F_e).max() > 0
    assert np.allclose(F_e, -np.einsum("eij,ej->ei", K_e, u_e))


# ----------------------------------------------
# A simulation extended by its terms
# ----------------------------------------------


class Heated(Simulations.Thermal):
    """Thermal extended by subclassing: a uniform heat source that changes between solves."""

    def __init__(self):
        domain = Domain((0, 0), (1, 1))
        mesh = domain.Mesh_2D()
        super().__init__(mesh, Models.Thermal(1, 1))
        self.source = 1.0

    @property
    def source(self) -> float:
        return self._source

    @source.setter
    def source(self, value: float) -> None:
        self._source = value
        self.Need_Update()

    def Get_terms(self, problemType: ProblemType | None = None) -> list[Term]:
        return super().Get_terms(problemType) + [
            Term("F", Operators.Linear.V, f=self.source)
        ]

    def Solve_held_at_zero(self) -> np.ndarray:
        """Solves with the temperature fixed to 0 on the edge x = 0."""
        self.Bc_Init()
        nodesX0 = self.mesh.Nodes_Conditions(lambda x, y, z: x == 0)
        self.add_dirichlet(nodesX0, [0], ["t"])
        return self.Solve().copy()


class Stiffened(Heated):
    """`Heated` plus a constant `K` term integrated by a `Counted` operator."""

    def __init__(self):
        self.operator = Counted()
        super().__init__()
        self.stiffness = 1.0

    @property
    def stiffness(self) -> float:
        return self._stiffness

    @stiffness.setter
    def stiffness(self, value: float) -> None:
        self._stiffness = value
        self.Need_Update()

    def Get_terms(self, problemType: ProblemType | None = None) -> list[Term]:
        return super().Get_terms(problemType) + [
            Term("K", self.operator, coef=self.stiffness, constant=True)
        ]


class Sprung(Heated):
    """`Heated` plus one constant term scaled into `K` by `alpha` and into `C` by `beta`, as a Robin spring."""

    def __init__(self):
        self.operator = Counted()
        self.alpha = 2.0
        self.beta = 3.0
        super().__init__()

    def Get_terms(self, problemType: ProblemType | None = None) -> list[Term]:
        spring = Term("K", self.operator, constant=True)
        return super().Get_terms(problemType) + [
            spring.Scaled(self.alpha),
            spring.Scaled(self.beta, "C"),
        ]


def _UV() -> np.ndarray:
    """One unit of the `Counted` term, assembled: `Stiffened`'s stiffness over conduction alone."""
    return (Stiffened().Get_K_C_M_F()[0] - Heated().Get_K_C_M_F()[0]).toarray()


def test_Term_reads_its_arguments_at_every_assembly():
    simu = Heated()
    once = simu.Solve_held_at_zero()

    simu.source = 2.0
    twice = simu.Solve_held_at_zero()

    # linear in the source: doubling it doubles the temperature
    assert np.abs(once).max() > 0
    np.testing.assert_allclose(twice, 2 * once, rtol=1e-12)


def test_Term_constant_integrates_once_while_its_arguments_are_unchanged():
    simu = Stiffened()
    for source in (1.0, 2.0, 3.0):
        simu.source = source  # re-assembles every solve; only the cache spares K
        simu.Solve_held_at_zero()

    assert simu.operator.calls == 1


def test_Term_constant_integrates_again_when_an_argument_changes():
    simu = Stiffened()
    K_once = simu.Get_K_C_M_F()[0]

    simu.stiffness = 2.0
    K_twice = simu.Get_K_C_M_F()[0]

    assert simu.operator.calls == 2
    np.testing.assert_allclose((K_twice - K_once).toarray(), _UV(), atol=1e-14)


def test_Term_Scaled_copies_of_a_constant_term_share_one_integration():
    simu = Sprung()
    K, C = simu.Get_K_C_M_F()[:2]

    assert simu.operator.calls == 1
    K_conduction, C_capacity = Heated().Get_K_C_M_F()[:2]
    np.testing.assert_allclose((K - K_conduction).toarray(), 2.0 * _UV(), atol=1e-14)
    np.testing.assert_allclose((C - C_capacity).toarray(), 3.0 * _UV(), atol=1e-14)


def test_Term_scale_reaches_the_matrix_without_integrating_again():
    simu = Sprung()
    simu.Get_K_C_M_F()

    simu.alpha = 5.0
    simu.Need_Update()
    K = simu.Get_K_C_M_F()[0]

    assert simu.operator.calls == 1
    K_conduction = Heated().Get_K_C_M_F()[0]
    np.testing.assert_allclose((K - K_conduction).toarray(), 5.0 * _UV(), atol=1e-14)


class MixedElastic(Simulations.Elastic):
    """Elastic with one model per element group, one stiffness term each."""

    def __init__(self, mesh, materials):
        super().__init__(mesh, next(iter(materials.values())))
        self.materials = materials

    def Get_terms(self, problemType: ProblemType | None = None) -> list[Term]:
        return [
            Term(
                "K",
                Operators.Bilinear.LinearizedElasticity,
                groupElem=g,
                C=m.C,
                constant=True,
            )
            for g, m in self.materials.items()
        ]


def _quad4_tri3() -> Mesh:
    domain = Domain((0, 0), (120, 13), 6.5)
    mesh = domain.Mesh_2D([], ElemType.QUAD4)
    assert len(mesh.Get_list_groupElem(2)) == 2
    return mesh


def _material(E: float = 210000.0) -> Models.Elastic.Isotropic:
    return Models.Elastic.Isotropic(2, E=E, v=0.3, planeStress=True)


def _K(simu) -> np.ndarray:
    return simu.Get_K_C_M_F()[0].toarray()


def test_Term_groupElem_terms_with_one_model_equal_one_term():
    mesh = _quad4_tri3()
    materials = {g: _material() for g in mesh.Get_list_groupElem(2)}

    K_groups = _K(MixedElastic(mesh, materials))
    K_whole = _K(Simulations.Elastic(mesh, _material()))

    np.testing.assert_allclose(K_groups, K_whole, rtol=1e-12, atol=1e-9)


def test_Term_groupElem_gives_each_group_its_own_model():
    mesh = _quad4_tri3()
    quad, tri = mesh.Get_list_groupElem(2)
    K = _K(MixedElastic(mesh, {quad: _material(), tri: _material(2e5)}))

    # linear in E: the TRI3 part alone is what changes with its modulus
    K_tri = _K(MixedElastic(mesh, {tri: _material(1.0)}))
    K_quad = _K(MixedElastic(mesh, {quad: _material()}))
    np.testing.assert_allclose(K, K_quad + 2e5 * K_tri, rtol=1e-10, atol=1e-6)
