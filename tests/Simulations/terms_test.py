# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Guards on :class:`~EasyFEA.Simulations.Term`.

Each catches a failure that is otherwise silent: an unknown slot letter would surface deep inside the fold, a ``tag`` on an operator that takes no ``elements`` would assemble the whole group instead of the tagged subset, and a slot count that does not match what the operator returns would drop an array or fabricate one.
"""

import numpy as np
import pytest

from EasyFEA import Models, Simulations
from EasyFEA.FEM import Operators
from EasyFEA.Geoms import Domain, Point
from EasyFEA.Simulations import Fold_terms, ProblemType, Term
from EasyFEA.Utilities import _params


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
    mesh = Domain(Point(), Point(1, 1)).Mesh_2D()
    return Simulations.Thermal(mesh, Models.Thermal(1, 1))  # k, c


class TestSlots:
    @pytest.mark.parametrize("slots", ["K", "C", "M", "F", "KR", "KRC", "KF"])
    def test_accepts_every_combination_in_use(self, slots):
        assert Term(slots, Operator).slots == tuple(slots)

    @pytest.mark.parametrize("slots", ["KG", "k", "", "KRX"])
    def test_rejects_unknown_letters(self, slots):
        with pytest.raises(ValueError, match="KCMFR"):
            Term(slots, Operator)


class TestTagNeedsElements:
    def test_accepts_a_tag_when_the_operator_takes_elements(self):
        assert Term("K", Operator, tag="epi").slots == ("K",)

    def test_rejects_a_tag_the_operator_cannot_honour(self):
        with pytest.raises(ValueError, match="elements"):
            Term("K", Operator_without_elements, tag="epi")

    def test_rejects_a_tag_on_a_real_operator_without_elements(self):
        # GradU_A_GradV integrates its whole group, so a tag would be silently ignored
        with pytest.raises(ValueError, match="elements"):
            Term("K", Operators.Bilinear.GradU_A_GradV, tag="epi")

    def test_allows_that_operator_untagged(self):
        assert Term("K", Operators.Bilinear.GradU_A_GradV).slots == ("K",)


class TestSlotsMatchReturnedArrays:
    """One slot letter per returned array; the fold rejects any other count.

    Too many letters is the ``Term("KC", op)`` mistake — reading slots as "put this matrix in K *and* C", which they never do. Too few is the dangerous direction: the extra array would be dropped in silence, and a term left holding a single slot then has its residual synthesised as ``-K·u_t``, which is only the residual of a *linear* contribution.
    """

    @staticmethod
    def Fold(slots, fn):
        return Fold_terms(Simu(), [Term(slots, fn)])

    def test_rejects_more_slots_than_arrays(self):
        with pytest.raises(ValueError, match="2 slot"):
            self.Fold("KC", One_array)

    def test_rejects_fewer_slots_than_arrays(self):
        with pytest.raises(ValueError, match="1 slot"):
            self.Fold("K", Two_arrays)

    @pytest.mark.parametrize("slots,fn", [("K", One_array), ("KF", Two_arrays)])
    def test_accepts_a_matching_count(self, slots, fn):
        assert self.Fold(slots, fn)


class TestSlotRank:
    """A slot's contribution must be a matrix for ``K``/``C``/``M`` and a vector for ``F``/``R``.

    Slot letters are positional, so ``"RK"`` where ``"KR"`` was meant sends the tangent to the right-hand side and the force to the stiffness. The rank is what makes that visible; two same-shaped matrices (``"MK"`` for ``"KM"``) still swap silently.
    """

    @staticmethod
    def Fold(slots, fn):
        return Fold_terms(Simu(), [Term(slots, fn)])

    def test_rejects_a_matrix_in_a_vector_slot(self):
        with pytest.raises(ValueError, match="takes a vector"):
            self.Fold("R", One_array)

    def test_rejects_a_vector_in_a_matrix_slot(self):
        with pytest.raises(ValueError, match="takes a matrix"):
            self.Fold("K", One_vector)

    def test_rejects_the_reversed_order(self):
        # "RK" instead of "KR": both letters are valid, only the order is wrong
        with pytest.raises(ValueError, match="positional"):
            self.Fold("RK", Two_arrays)

    def test_accepts_the_right_order(self):
        assert self.Fold("KR", Two_arrays)


class Heated(Simulations.Thermal):
    """Thermal extended by subclassing: a uniform heat source that changes between solves."""

    source = _params.ScalarParameter()

    def __init__(self):
        domain = Domain((0, 0), (1, 1))
        mesh = domain.Mesh_2D()
        super().__init__(mesh, Models.Thermal(1, 1))
        self.source = 1.0

    def Get_terms(self, problemType: ProblemType | None = None) -> list[Term]:
        return super().Get_terms(problemType) + [
            Term("F", Operators.Linear.V, f=self.source)
        ]

    def Solve_held_at_zero(self) -> np.ndarray:
        """Solves with the temperature fixed to 0 on the edge x = 0."""
        self.Bc_Init()
        self.add_dirichlet(
            self.mesh.Nodes_Conditions(lambda x, y, z: x == 0), [0], ["t"]
        )
        return self.Solve().copy()


class Counted:
    """`Operators.Bilinear.UV`, counting its calls: an object, not a bound method, so a constant term accepts it."""

    def __init__(self):
        self.calls = 0

    def __call__(self, groupElem, coef=1.0, elements=None):
        self.calls += 1
        return Operators.Bilinear.UV(groupElem, coef=coef, elements=elements)


class Stiffened(Heated):
    """`Heated` plus a constant `K` term integrated by a `Counted` operator."""

    stiffness = _params.ScalarParameter()

    def __init__(self):
        super().__init__()
        self.operator = Counted()
        self.stiffness = 1.0

    def Get_terms(self, problemType: ProblemType | None = None) -> list[Term]:
        return super().Get_terms(problemType) + [
            Term("K", self.operator, coef=self.stiffness, constant=True)
        ]


class Sprung(Heated):
    """`Heated` plus one constant term scaled into `K` by `alpha` and into `C` by `beta`, as a Robin spring."""

    alpha = _params.ScalarParameter()
    beta = _params.ScalarParameter()

    def __init__(self):
        super().__init__()
        self.operator = Counted()
        self.alpha = 2.0
        self.beta = 3.0

    def Get_terms(self, problemType: ProblemType | None = None) -> list[Term]:
        spring = Term("K", self.operator, constant=True)
        return super().Get_terms(problemType) + [
            spring.Scaled(self.alpha),
            spring.Scaled(self.beta, "C"),
        ]


class TestExtendBySubclass:
    """A subclass adds terms to `Get_terms`; a value that changes between solves is a parameter of it."""

    def test_a_parameter_written_between_solves_reaches_the_next_solve(self):
        simu = Heated()
        once = simu.Solve_held_at_zero()

        simu.source = 2.0
        twice = simu.Solve_held_at_zero()

        # linear in the source: doubling it doubles the temperature
        assert np.abs(once).max() > 0
        np.testing.assert_allclose(twice, 2 * once, rtol=1e-12)

    def test_a_constant_term_is_integrated_once_while_its_arguments_are_unchanged(self):
        simu = Stiffened()
        for source in (1.0, 2.0, 3.0):
            simu.source = source  # re-assembles every solve; only the cache spares K
            simu.Solve_held_at_zero()

        assert len(simu.mesh.Get_list_groupElem(simu.dim)) == 1
        assert simu.operator.calls == 1

    def test_a_constant_term_whose_argument_changes_is_integrated_again(self):
        simu = Stiffened()
        K_once = simu.Get_K_C_M_F()[0]

        simu.stiffness = 2.0
        K_twice = simu.Get_K_C_M_F()[0]

        assert simu.operator.calls == 2
        # each unit of stiffness adds one UV on top of the conduction matrix
        K_conduction = Heated().Get_K_C_M_F()[0]
        UV = (K_once - K_conduction).toarray()
        assert np.abs(UV).max() > 0
        np.testing.assert_allclose((K_twice - K_once).toarray(), UV, atol=1e-14)

    @staticmethod
    def UV() -> np.ndarray:
        """One unit of the `Counted` term, assembled: `Stiffened`'s stiffness over conduction alone."""
        return (Stiffened().Get_K_C_M_F()[0] - Heated().Get_K_C_M_F()[0]).toarray()

    def test_scaled_copies_of_a_constant_term_share_one_integration(self):
        simu = Sprung()
        K, C = simu.Get_K_C_M_F()[:2]

        assert simu.operator.calls == 1
        K_conduction, C_capacity = Heated().Get_K_C_M_F()[:2]
        UV = self.UV()
        np.testing.assert_allclose((K - K_conduction).toarray(), 2.0 * UV, atol=1e-14)
        np.testing.assert_allclose((C - C_capacity).toarray(), 3.0 * UV, atol=1e-14)

    def test_a_new_scale_reaches_the_matrix_without_integrating_again(self):
        simu = Sprung()
        simu.Get_K_C_M_F()

        simu.alpha = 5.0
        K = simu.Get_K_C_M_F()[0]

        assert simu.operator.calls == 1
        K_conduction = Heated().Get_K_C_M_F()[0]
        np.testing.assert_allclose(
            (K - K_conduction).toarray(), 5.0 * self.UV(), atol=1e-14
        )
