# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Each guard catches a failure that is otherwise silent: an unknown slot letter would surface deep inside the integration, a ``tag`` on an operator that takes no ``elements`` would assemble the whole group, and a slot count that does not match what the operator returns would drop an array or fabricate one."""

import numpy as np
import pytest

from EasyFEA import ElemType, Mesh, Models, Simulations
from EasyFEA.FEM import Operators
from EasyFEA.Geoms import Circle, Domain
from EasyFEA.Simulations import Integrate_terms, ProblemType, Term, TrialSolution

from ._simus import PRESSURE, coupled_phasefield, coupling, incompressible

U = Simulations.PhaseField.U
D = Simulations.PhaseField.D


def Operator(groupElem, elements=None):
    """Stand-in operator with the shape the integration calls."""
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


def Integrate(slots: str, fn) -> list:
    return Integrate_terms(Simu(), [Term(slots, fn)])


def Array(list_contribution: list, slot: str, block=None) -> np.ndarray:
    """The one array of `list_contribution` in `slot` (and `block`)."""
    (array,) = [
        contribution.array
        for contribution in list_contribution
        if contribution.slot == slot and block in (None, contribution.block)
    ]
    return array


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
        Term("K", simu.Get_unknowns, constant=True)


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


def test_Term__Get_field_groups_follows_each_group_with_those_of_the_fields():
    domain = Domain((0, 0), (1, 1), 0.5)
    simu = incompressible(domain.Mesh_2D([], ElemType.QUAD9, isOrganised=True))
    groupElem = simu.mesh.groupElem

    [(group, pGroup)] = Term("K", Operator)._Get_problemTypes_groups(simu, (PRESSURE,))
    assert group is groupElem and pGroup.elemType == ElemType.QUAD4

    [(group, pGroup)] = Term("K", Operator, dim=1)._Get_problemTypes_groups(
        simu, (PRESSURE,)
    )
    assert group.elemType == ElemType.SEG3 and pGroup.elemType == ElemType.SEG2

    assert Term("K", Operator)._Get_problemTypes_groups(simu, ()) == [(groupElem,)]
    assert (
        Term("K", Operator, tag="missing")._Get_problemTypes_groups(simu, (PRESSURE,))
        == []
    )


def test_Term__Get_blocks_resolves_a_free_slot_to_the_integrated_problem():
    assert Term("KR", Two_arrays)._Get_blocks(U) == ((U, U), (U, U))

    term = Term("KR", Two_arrays, problemTypes=((U, D), None))
    assert term._Get_blocks(D) == ((U, D), (D, D))

    term = Term("R", One_vector, problemTypes=(D,))
    assert term._Get_blocks(U) == ((D, D),)


def test_Term__Get_elements_is_the_tagged_subset():
    groupElem = _inclusion_mesh().groupElem

    assert Term("K", Operator)._Get_elements(groupElem) is None
    elements = Term("K", Operator, tag="S1")._Get_elements(groupElem)
    assert np.array_equal(elements, groupElem.Get_Elements_Tag("S1"))


def test_Term__Evaluate_passes_its_kwargs_and_elements_only_when_tagged():
    groupElem = _inclusion_mesh().groupElem
    u = np.arange(groupElem.Nn, dtype=float)

    def Echo(groupElem, u=None, elements=None, coef=1.0):
        return u, elements, coef

    bare = Term("K", Echo)._Evaluate(groupElem)
    untagged = Term("K", Echo, u=u, coef=2.0)._Evaluate(groupElem)
    tagged = Term("K", Echo, tag="S1", u=u)._Evaluate(groupElem)

    assert bare == (None, None, 1.0)
    assert untagged[0] is u and untagged[1] is None and untagged[2] == 2.0
    assert tagged[0] is u
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
# Integrate_terms
# ----------------------------------------------


def test_Integrate_terms_rejects_more_slots_than_arrays():
    with pytest.raises(ValueError, match="2 slot"):
        Integrate("KC", One_array)


def test_Integrate_terms_rejects_fewer_slots_than_arrays():
    with pytest.raises(ValueError, match="1 slot"):
        Integrate("K", Two_arrays)


def test_Integrate_terms_accepts_one_slot_per_array():
    assert Integrate("K", One_array)
    assert Integrate("KF", Two_arrays)


def test_Integrate_terms_rejects_a_matrix_in_a_vector_slot():
    with pytest.raises(ValueError, match="takes a vector"):
        Integrate("R", One_array)


def test_Integrate_terms_rejects_a_vector_in_a_matrix_slot():
    with pytest.raises(ValueError, match="takes a matrix"):
        Integrate("K", One_vector)


def test_Integrate_terms_rejects_the_reversed_order():
    # "RK" instead of "KR": both letters are valid, only the order is wrong
    with pytest.raises(ValueError, match="positional"):
        Integrate("RK", Two_arrays)


def test_Integrate_terms_sums_tagged_masses_to_the_whole_one():
    mesh = _inclusion_mesh()
    simu = Simulations.Thermal(mesh, Models.Thermal(1, 1))

    def Mass(*tags: str) -> float:
        terms = [Term("M", Operators.Bilinear.UV, tag=tag) for tag in tags]
        list_contribution = Integrate_terms(simu, terms)
        return sum(
            contribution.array.sum()
            for contribution in list_contribution
            if contribution.slot == "M"
        )

    assert np.isclose(Mass("S0", "S1"), Mass(None))
    assert np.isclose(Mass(None), mesh.area)


def test_Integrate_terms_keeps_each_array_unsummed_by_reference():
    simu = Simu()
    arrays = {}

    def Force(groupElem):
        arrays["K"], arrays["R"] = One_array(groupElem), One_vector(groupElem) + 1
        return arrays["K"], arrays["R"]

    pt = simu.problemType
    list_contribution = Integrate_terms(simu, [Term("KR", Force)])

    assert [
        (contribution.slot, contribution.block, contribution.group)
        for contribution in list_contribution
    ] == [
        ("K", (pt, pt), simu.mesh.groupElem),
        ("R", (pt, pt), simu.mesh.groupElem),
    ]
    assert all(
        contribution.array is arrays[contribution.slot]
        for contribution in list_contribution
    )
    assert all(
        contribution.coef == simu.thickness for contribution in list_contribution
    )


def test_Integrate_terms_keeps_the_cached_array_of_a_constant_term():
    simu = Simu()
    term = Term("M", Operators.Bilinear.UV, constant=True)

    first, again = (Array(Integrate_terms(simu, [term]), "M") for _ in range(2))

    assert again is first


def test_Integrate_terms_hands_operators_the_group_of_the_field_space():
    domain = Domain((0, 0), (1, 1))
    mesh = domain.Mesh_2D([], ElemType.TRI3)
    meshP2 = mesh._With_FunctionSpace(Simulations.Thermal.problemType, 2)
    simu = Simulations.Thermal(meshP2, Models.Thermal(1, 1))
    groupElem = meshP2.groupElem

    K_e = Array(Integrate_terms(simu, [Term("K", One_array)]), "K")

    assert K_e.shape == (groupElem.Ne, 6, 6)


# ----------------------------------------------
# Blocks
# ----------------------------------------------


def test_Term_rejects_problemTypes_not_one_per_slot():
    with pytest.raises(ValueError, match="one per slot"):
        Term("KR", Operator, problemTypes=((D, U),))


def test_Term_rejects_a_pair_on_a_vector_slot():
    with pytest.raises(ValueError, match="takes a vector"):
        Term("R", Operator, problemTypes=((D, U),))


def test_Integrate_terms_keys_a_diagonal_contribution_by_its_problem_type():
    simu = Simu()
    pt = simu.problemType

    list_contribution = Integrate_terms(simu, [Term("K", One_array)])

    assert [contribution.block for contribution in list_contribution] == [(pt, pt)]


def test_Integrate_terms_routes_each_array_to_its_block():
    simu = coupled_phasefield()
    groupElem = simu.mesh.groupElem

    def Spanning(groupElem, uGroup):
        return One_array(groupElem), One_vector(groupElem) + 1, coupling(groupElem)

    term = Term("KRK", Spanning, problemTypes=((D, D), D, (D, U)))
    list_contribution = Integrate_terms(simu, [term], D)

    assert [
        (contribution.slot, contribution.block) for contribution in list_contribution
    ] == [
        ("K", (D, D)),
        ("R", (D, D)),
        ("K", (D, U)),
    ]
    assert all(contribution.group is groupElem for contribution in list_contribution)
    assert np.all(Array(list_contribution, "K", (D, D)) == 0.0)
    assert np.all(Array(list_contribution, "R") == 1.0)
    assert np.array_equal(Array(list_contribution, "K", (D, U)), coupling(groupElem))


def test_Integrate_terms_places_a_term_in_the_rows_it_declares():
    simu = coupled_phasefield()
    term = Term("K", coupling, problemTypes=((D, U),))
    groupElem = simu.mesh.groupElem

    list_contribution = Integrate_terms(simu, [term], U)

    assert np.array_equal(Array(list_contribution, "K", (D, U)), coupling(groupElem))
    assert {contribution.block[0] for contribution in list_contribution} == {D}


def test_Integrate_terms_rejects_a_swapped_block():
    simu = coupled_phasefield()

    def Swapped(groupElem, uGroup):
        return np.swapaxes(coupling(groupElem), 1, 2)

    with pytest.raises(ValueError, match="rows"):
        Integrate_terms(simu, [Term("K", Swapped, problemTypes=((D, U),))], D)


def test_Integrate_terms_hands_a_spanning_term_one_group_per_field():
    domain = Domain((0, 0), (1, 1), 0.5)
    mesh = domain.Mesh_2D([], ElemType.TRI3)
    meshP2 = mesh._With_FunctionSpace(D, 2)
    simu = coupled_phasefield(mesh=meshP2)
    seen = {}

    def Spy(dGroup, uGroup):
        seen["groups"] = dGroup, uGroup
        # P2 damage rows, P1 displacement columns
        return np.zeros((dGroup.Ne, 6, 6))

    Integrate_terms(simu, [Term("K", Spy, problemTypes=((D, U),))], D)

    assert seen["groups"][0] is meshP2._Get_FunctionSpace(D).Get_list_groupElem()[0]
    assert seen["groups"][1] is meshP2.groupElem


def test_Term_Scaled_keeps_its_blocks():
    simu = coupled_phasefield()
    groupElem = simu.mesh.groupElem
    term = Term("K", coupling, problemTypes=((D, U),)).Scaled(2.0)

    (c,) = Integrate_terms(simu, [term], D)

    assert c.coef == 2.0 * simu.thickness
    assert np.array_equal(c.array, coupling(groupElem))


def test_Term__Cached_keys_the_blocks():
    simu = Simu()
    groupElem = simu.mesh.groupElem
    operator = Counted()
    term = Term("K", operator, constant=True)

    term._Cached(simu, groupElem, None, ((U, U),))
    term._Cached(simu, groupElem, None, ((D, D),))

    assert operator.calls == 2


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

    def Get_terms(
        self, x: TrialSolution, problemType: ProblemType | None = None
    ) -> list[Term]:
        return super().Get_terms(x, problemType) + [
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

    def Get_terms(
        self, x: TrialSolution, problemType: ProblemType | None = None
    ) -> list[Term]:
        return super().Get_terms(x, problemType) + [
            Term("K", self.operator, coef=self.stiffness, constant=True)
        ]


class Sprung(Heated):
    """`Heated` plus one constant term scaled into `K` by `alpha` and into `C` by `beta`, as a Robin spring."""

    def __init__(self):
        self.operator = Counted()
        self.alpha = 2.0
        self.beta = 3.0
        super().__init__()

    def Get_terms(
        self, x: TrialSolution, problemType: ProblemType | None = None
    ) -> list[Term]:
        spring = Term("K", self.operator, constant=True)
        return super().Get_terms(x, problemType) + [
            spring.Scaled(self.alpha),
            spring.Scaled(self.beta, "C"),
        ]


def _UV() -> np.ndarray:
    """One unit of the `Counted` term, assembled: `Stiffened`'s stiffness over conduction alone."""
    return (Stiffened().Assembly()[0] - Heated().Assembly()[0]).toarray()


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
    K_once = simu.Assembly()[0]

    simu.stiffness = 2.0
    K_twice = simu.Assembly()[0]

    assert simu.operator.calls == 2
    np.testing.assert_allclose((K_twice - K_once).toarray(), _UV(), atol=1e-14)


def test_Term_Scaled_copies_of_a_constant_term_share_one_integration():
    simu = Sprung()
    K, C = simu.Assembly()[:2]

    assert simu.operator.calls == 1
    K_conduction, C_capacity = Heated().Assembly()[:2]
    np.testing.assert_allclose((K - K_conduction).toarray(), 2.0 * _UV(), atol=1e-14)
    np.testing.assert_allclose((C - C_capacity).toarray(), 3.0 * _UV(), atol=1e-14)


def test_Term_scale_reaches_the_matrix_without_integrating_again():
    simu = Sprung()
    simu.Assembly()

    simu.alpha = 5.0
    simu.Need_Update()
    K = simu.Assembly()[0]

    assert simu.operator.calls == 1
    K_conduction = Heated().Assembly()[0]
    np.testing.assert_allclose((K - K_conduction).toarray(), 5.0 * _UV(), atol=1e-14)


class MixedElastic(Simulations.Elastic):
    """Elastic with one model per element group, one stiffness term each."""

    def __init__(self, mesh, materials):
        super().__init__(mesh, next(iter(materials.values())))
        self.materials = materials

    def Get_terms(
        self, x: TrialSolution, problemType: ProblemType | None = None
    ) -> list[Term]:
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
    return simu.Assembly()[0].toarray()


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
