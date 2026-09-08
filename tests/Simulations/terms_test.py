# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Declaration-time guards on :class:`~EasyFEA.Simulations.Term`.

Both failures they catch are silent at run time: an unknown slot letter would only surface deep inside the fold, and a ``tag`` on an operator that takes no ``elements`` would assemble the whole group instead of the tagged subset while reporting nothing.
"""

import pytest

from EasyFEA import Models, Simulations
from EasyFEA.FEM import Operators
from EasyFEA.Geoms import Domain, Point
from EasyFEA.Simulations import Term


def Operator(groupElem, elements=None):
    """Stand-in operator with the shape the fold calls."""
    return None


def Operator_without_elements(groupElem):
    """Stand-in operator that cannot be restricted to a subset of its group."""
    return None


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


class TestAddTerms:
    """``Add_terms`` is variadic: one term and many take the same path.

    Assertions compare by ``is``, never ``==``: :py:meth:`Term.__eq__` is *value* identity so the cache can find a rebuilt term, which makes two separately built ``Term("K", op)`` compare equal.
    """

    @staticmethod
    def Simu() -> Simulations.Thermal:
        mesh = Domain(Point(), Point(1, 1)).Mesh_2D()
        return Simulations.Thermal(mesh, Models.Thermal(1, 1))  # k, c

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
