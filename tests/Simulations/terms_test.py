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
from EasyFEA.Simulations import Fold_terms, Term


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


class TestAddTerms:
    """``Add_terms`` is variadic: one term and many take the same path.

    Assertions compare by ``is``, never ``==``: :py:meth:`Term.__eq__` is *value* identity so the cache can find a rebuilt term, which makes two separately built ``Term("K", op)`` compare equal.
    """

    Simu = staticmethod(Simu)

    @staticmethod
    def Added(simu) -> list[Term]:
        """The terms added on the instance. `Get_terms` returns the declared ones, a separate list."""
        return simu._Simu__terms

    def test_one_term_is_added_and_bound(self):
        simu = self.Simu()
        term = Term("K", Operator)

        returned = simu.Add_terms(term)

        assert returned == [term] and returned[0] is term
        assert self.Added(simu) == [term]
        assert term._simu is simu

    def test_many_terms_are_all_added_in_order_and_bound(self):
        simu = self.Simu()
        terms = [Term("K", Operator), Term("C", Operator), Term("M", Operator)]

        returned = simu.Add_terms(*terms)

        assert [t is u for t, u in zip(returned, terms)] == [True] * 3
        assert [t is u for t, u in zip(self.Added(simu), terms)] == [True] * 3
        assert all(term._simu is simu for term in terms)

    def test_one_call_of_many_matches_many_calls_of_one(self):
        batched, oneByOne = self.Simu(), self.Simu()
        slots = ["K", "C", "M"]

        batched.Add_terms(*(Term(slot, Operator) for slot in slots))
        for slot in slots:
            oneByOne.Add_terms(Term(slot, Operator))

        assert [t.slots for t in self.Added(batched)] == [
            t.slots for t in self.Added(oneByOne)
        ]

    def test_no_terms_is_a_no_op(self):
        # `Add_terms(*collection)` over an empty collection must not raise
        simu = self.Simu()
        assert simu.Add_terms() == []
        assert self.Added(simu) == []

    def test_rejects_a_non_term_without_adding_any(self):
        simu = self.Simu()
        with pytest.raises(AssertionError, match="must be a Term"):
            simu.Add_terms(Term("K", Operator), "not a term")
        assert self.Added(simu) == []

    def test_Terms_Init_clears_them(self):
        simu = self.Simu()
        simu.Add_terms(Term("K", Operator), Term("C", Operator))
        simu.Terms_Init()
        assert self.Added(simu) == []
