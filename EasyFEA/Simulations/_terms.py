# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""Declarative description of a simulation's local matrix system: a list of :class:`Term`, folded into ``{groupElem: (K_e, C_e, M_e, F_e)}`` by :func:`Fold_terms`."""

import inspect
from typing import Any, Callable, Optional, TYPE_CHECKING

import numpy as np

from ..FEM import _GroupElem

if TYPE_CHECKING:
    from ..FEM import Mesh
from ..Utilities import _types

if TYPE_CHECKING:
    from ._simu import _Simu
    from ._problem_type import ProblemType


_SLOTS = "KCMFR"
"""Slot letters, one per array the operator returns. ``K``/``C``/``M`` are the stiffness, damping and mass matrices; ``F`` is an external load, added to the right-hand side as-is; ``R`` is an **internal** force, so it reaches ``F`` negated. Every nonlinear operator returns an ``R`` — it is the residual of ``A·Δu = -R(u)`` — and ``F`` is reserved for genuine loads such as a phase-field source."""

_SLOT_FIELD = {"K": "u", "C": "v", "M": "a"}
"""Field a single-slot term's residual contracts against: ``K·u_t``, ``C·v_t``, ``M·a_t``."""

_INJECTABLE = frozenset({"u", "elements"})
"""Parameters the fold fills in by name when the caller does not supply them. Kept short and generic on purpose: injection makes an operator's *parameter names* part of the contract, so it covers only quantities every simulation has. Anything domain-specific — a hyperelastic state, a material — is passed by a named method instead, as in :py:meth:`HyperElastic.Get_terms`."""


class Term:
    r"""One operator's contribution to the local matrix system :math:`\Krm \, \mathrm{u} + \Crm \, \vrm + \Mrm \, \arm = \Frm`.

    ``slots`` names where the returned arrays go, one letter of ``KCMFR`` each::

        Term("K", Bilinear.LinearizedElasticity, C=material.C)
        Term("KR", self.__Stress)                # R is an *internal* force: F -= R

    ``fn`` is called as ``fn(groupElem, **kwargs)``, so any operator whose first positional argument is a ``_GroupElem`` can be named directly. Anything needing per-group work — a hyperelastic state, a contact projection — is a named method of the simulation with the same shape.
    """

    def __init__(
        self,
        slots: str,
        fn: Callable[..., Any],
        /,
        *,
        dim: Optional[int] = None,
        tag: Optional[str] = None,
        constant: bool = False,
        **kwargs,
    ):
        """Declares one term.

        Parameters
        ----------
        slots : str
            Where ``fn``'s returned arrays go, one letter of ``KCMFR`` each — see :py:data:`_SLOTS`.
        fn : Callable
            The operator, called as ``fn(groupElem, **kwargs)``.
        dim : int, optional
            Dimension of the element groups this term integrates over, ``mesh.dim`` by default.
        tag : str, optional
            Restricts the term to a tagged element subset; groups without the tag are skipped.
        constant : bool, optional
            Declares the contribution independent of the solution, so it is built once and reused across Newton iterations and time steps. Defaults to False.
        **kwargs
            Passed on to ``fn``. ``u`` and ``elements`` are supplied by the fold when ``fn`` declares them and they are left out here.
        """

        if not slots or not set(slots) <= set(_SLOTS):
            raise ValueError(
                f"slots must be one or more letters of '{_SLOTS}', got {slots!r}."
            )

        parameters = list(inspect.signature(fn).parameters)

        if tag is not None and "elements" not in parameters:
            raise ValueError(
                f"{getattr(fn, '__name__', fn)} takes no `elements` argument, so tag={tag!r} "
                "would be silently ignored on the elements it selects."
            )

        self.slots = tuple(slots)
        self.fn = fn
        self.kwargs: dict[str, Any] = kwargs
        self.dim = dim
        self.tag = tag
        self.constant = constant
        self._simu: Optional["_Simu"] = None
        """Set by :py:meth:`_Simu.Add_terms`, so :py:meth:`Set` can invalidate the assembled matrices."""

        # resolved once, not per group per Newton iteration
        self.__injectable = tuple(n for n in parameters[1:] if n in _INJECTABLE)

        if constant:
            # the cache is keyed on the term's value, so every argument must be hashable
            try:
                hash(self)
            except TypeError as error:
                raise TypeError(
                    f"constant=True needs hashable arguments and a stable function, but {fn} "
                    f"cannot key the cache ({error}). Drop constant=True, or pass the varying "
                    "argument through a term that is rebuilt each assembly."
                ) from error

    def __repr__(self) -> str:
        name = getattr(self.fn, "__name__", repr(self.fn))
        tag = "" if self.tag is None else f", tag={self.tag!r}"
        return f"Term({''.join(self.slots)!r}, {name}{tag})"

    def Set(self, **kwargs) -> "Term":
        """Updates arguments in place, for a value that changes between steps (a pressure, a penalty). Returns the term, so it can be chained."""
        self.kwargs.update(kwargs)
        if self._simu is not None:
            self._simu.Need_Update()
        return self

    # ----------------------------------------------
    # Value identity, so `constant=True` survives the per-assembly rebuild of the list
    # ----------------------------------------------

    def _Cache_key(self) -> tuple:
        """Value identity of the term, so a ``constant=True`` contribution is still found in the cache after :py:meth:`_Simu.Get_terms` rebuilds the list on the next assembly."""
        return (
            self.fn,
            self.slots,
            self.dim,
            self.tag,
            tuple(sorted(self.kwargs.items(), key=lambda item: item[0])),
        )

    def __hash__(self):
        return hash(self._Cache_key())

    def __eq__(self, other):
        return isinstance(other, Term) and self._Cache_key() == other._Cache_key()

    # ----------------------------------------------
    # Evaluation
    # ----------------------------------------------

    def _Get_groups(self, mesh: "Mesh") -> list[_GroupElem]:
        """Element groups this term integrates over, tag-filtered."""
        groups = mesh.Get_list_groupElem(self.dim)
        if self.tag is None:
            return groups
        return [g for g in groups if self.tag in g.elementTags]

    def _Get_elements(self, groupElem: _GroupElem) -> Optional[_types.IntArray]:
        """Element indices this term is restricted to within `groupElem`, or None."""
        return None if self.tag is None else groupElem.Get_Elements_Tag(self.tag)

    def _Evaluate(
        self, groupElem: _GroupElem, u: Optional[_types.FloatArray] = None
    ) -> Any:
        """Calls the operator on one group, injecting the arguments it declares and the caller left out."""
        values = {"u": u, "elements": self._Get_elements(groupElem)}
        kwargs = dict(self.kwargs)
        for name in self.__injectable:
            if name not in kwargs and values[name] is not None:
                kwargs[name] = values[name]
        return self.fn(groupElem, **kwargs)

    def _Evaluate_constant(self, groupElem: _GroupElem) -> Any:
        """Evaluates a ``constant=True`` term. ``u`` is deliberately unavailable here: a term that needs it is not constant, and would fail loudly on the missing argument rather than silently freeze the first iterate."""
        return self._Evaluate(groupElem)


def Fold_terms(
    simu: "_Simu",
    terms: list[Term],
    problemType: Optional["ProblemType"] = None,
) -> dict[_GroupElem, tuple]:
    r"""Folds `terms` into the local matrix system ``{groupElem: (K_e, C_e, M_e, F_e)}``.

    Contributions **accumulate**: several terms may target one group, and each fills only the slots it declares. Every contribution is scaled by :py:attr:`_Simu.thickness`, so operators stay free of simulation-level geometry.

    In a nonlinear simulation a **single-slot** term also contributes its own residual — :math:`-\Krm_e \, \mathrm{u}_t`, :math:`-\Crm_e \, \vrm_t`, :math:`-\Mrm_e \, \arm_t`. A single slot means the contribution is linear in its field, so that product *is* its residual. Multi-slot terms return their own force and are never given one.
    """

    if problemType is None:
        problemType = simu.problemType

    mesh = simu.mesh
    thickness = simu.thickness
    dof_n = simu.Get_dof_n(problemType)
    isNonLinear = simu.isNonLinear

    u_t, v_t, a_t = simu.Get_u_v_a(problemType)
    fields = {"u": u_t, "v": v_t, "a": a_t}

    index = {"K": 0, "C": 1, "M": 2, "F": 3}  # slot name -> index in (K, C, M, F)
    out: dict[_GroupElem, list] = {}

    for term in terms:
        for groupElem in term._Get_groups(mesh):

            if term.constant:
                contributions = simu._Term_cached(term, groupElem)
            else:
                contributions = term._Evaluate(groupElem, u_t)
            if not isinstance(contributions, tuple):
                contributions = (contributions,)
            assert len(contributions) >= len(term.slots), (
                f"{term.fn.__name__} returned {len(contributions)} array(s) but the term declares "
                f"{len(term.slots)} slot(s) {term.slots}."
            )

            slots = out.setdefault(groupElem, [None, None, None, None])
            own = None  # this term's own matrix contribution, for its residual

            for slot, contribution in zip(term.slots, contributions):
                if contribution is None:
                    continue
                # always allocates, so a cached `constant=True` array is never written into
                contribution = thickness * contribution
                if slot == "R":  # an internal force opposes the right-hand side
                    contribution, slot = -contribution, "F"
                else:
                    own = contribution
                i = index[slot]
                slots[i] = contribution if slots[i] is None else slots[i] + contribution

            # from `own`, never the accumulated slot: two terms may fill one slot on one group
            if isNonLinear and len(term.slots) == 1:
                residual = _Residual(term, groupElem, own, fields, dof_n)
                if residual is not None:
                    slots[3] = residual if slots[3] is None else slots[3] + residual

    return {groupElem: tuple(slots) for groupElem, slots in out.items()}


def _Residual(
    term: Term,
    groupElem: _GroupElem,
    matrix_e: Optional[np.ndarray],
    fields: dict[str, Optional[_types.FloatArray]],
    dof_n: int,
) -> Optional[np.ndarray]:
    """``-A_e · x_t`` for a single-slot term, where `x_t` is the field that slot multiplies.

    Sound because a single slot means the contribution is linear in that field, so the product *is* the residual. Returns None when the slot carries no matrix (an ``F`` term) or when the field does not exist under the active time scheme (no velocity in a static problem).
    """

    slot = term.slots[0]
    if slot not in _SLOT_FIELD or matrix_e is None:
        return None

    x_t = fields[_SLOT_FIELD[slot]]
    if x_t is None:
        return None

    x_e = groupElem.Locates_sol_e(x_t, dof_n)

    return -np.einsum("eij,ej->ei", matrix_e, x_e, optimize=True)
