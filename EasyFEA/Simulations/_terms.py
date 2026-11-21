# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""Declarative description of a simulation's local matrix system: a list of :class:`Term`, integrated into a flat list of :class:`_Contribution` by :func:`Integrate_terms`."""

import copy
import inspect
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

from ..Utilities import _types
from ..Utilities._cache import cached_computed_values

if TYPE_CHECKING:
    from typing import Concatenate

    from ..FEM import Mesh, _GroupElem
    from ._simu import _Simu
    from ._problem_type import ProblemType

    # what a checker enforces: the first positional argument is the element group, the rest are the term's kwargs
    _Operator = Callable[Concatenate[_GroupElem, ...], Any]
else:
    # Concatenate reached typing in 3.10
    _Operator = Callable[..., Any]

_Block = tuple["ProblemType", "ProblemType"]
"""``(rows, cols)``: the field a contribution tests, and the field it is differentiated by."""


@dataclass(frozen=True, slots=True, eq=False)
class _Contribution:
    """One unsummed array a term evaluated to, for one slot on one group."""

    slot: str
    """one of ``KCMFR``"""
    block: _Block
    """``(rows, cols)``; a vector's cols are its rows"""
    group: "_GroupElem"
    array: np.ndarray
    """the operator's own, a cached one by reference: never written into"""
    coef: float
    """thickness·scale, applied by the reader"""
    term: "Term"

    def Scaled(self) -> np.ndarray:
        """``coef·array``, negated for an ``R``: an internal force opposes the load. Always a new array."""
        return (-self.coef if self.slot == "R" else self.coef) * self.array


_SLOTS = "KCMFR"
"""Slot letters, one per array the operator returns. ``K``/``C``/``M`` are the stiffness, damping and mass matrices; ``F`` is an external load, added to the right-hand side as-is; ``R`` is an **internal** force, so it reaches ``F`` negated. Every nonlinear operator returns an ``R`` — it is the residual of ``A·Δu = -R(u)`` — and ``F`` is reserved for genuine loads such as a phase-field source."""


def _Check_comparable(fn: Callable, name: str, value: Any) -> None:
    """Refuses a ``constant=True`` argument that no value check can see change."""
    if isinstance(value, (list, dict, set)) or (
        value is not None
        and not isinstance(value, np.ndarray)
        and type(value).__eq__ is object.__eq__
    ):
        raise TypeError(
            f"{getattr(fn, '__name__', fn)}: argument {name!r} ({type(value).__name__}) can "
            "change without the term seeing it, so it cannot be constant=True. Pass the "
            "values the operator reads instead."
        )


def _Same(a: Any, b: Any) -> bool:
    """Whether a ``constant=True`` argument is unchanged since the contribution was built."""
    return a is b or np.array_equal(a, b)


class Term:
    r"""One operator, declared: where its arrays go, what it integrates over, what it is called with.

    Evaluating a term on one element group yields one **contribution** per slot — the arrays :func:`Integrate_terms` lists, the local form of :math:`\Krm \, \mathrm{u} + \Crm \, \vrm + \Mrm \, \arm = \Frm`.

    ``slots`` names where the returned arrays go — one letter of ``KCMFR`` per array, in order, so an operator returning two arrays takes exactly two letters. Slots *route* arrays; they never copy one into several, so ``"KC"`` on an operator returning a single matrix is an error, not a request to damp with the stiffness::

        Term("K", Bilinear.LinearizedElasticity, C=material.C)
        Term("KR", self.__Stress)                # R is an *internal* force: F -= R

    ``fn`` is called as ``fn(groupElem, **kwargs)``, so any operator whose first positional argument is a ``_GroupElem`` can be named directly. Anything needing per-group work — a hyperelastic state, a contact projection — is a named method of the simulation with the same shape.

    ``problemTypes`` places each array in a block of a system over several fields, one entry per slot: a ``(rows, cols)`` pair for a matrix, the rows alone for a vector, None for the integrated problem's own diagonal (the default)::

        Term("KRKKR", self.__MixedIncompressibility, problemTypes=((u, u), u, (u, p), (p, u), p))

    A term spanning several fields is handed one group per field, in the order its blocks name them.

    The operator gets exactly the kwargs given here, plus ``elements`` when ``tag`` is set: the iterate it reads is passed explicitly, e.g. ``Term("KR", self.__Stress, u=u)`` with ``u`` the node values from :py:meth:`_Simu.Get_u_v_a`.
    """

    def __init__(
        self,
        slots: str,
        fn: _Operator,
        /,
        *,
        dim: int | None = None,
        tag: str | None = None,
        groupElem: "_GroupElem | None" = None,
        constant: bool = False,
        problemTypes: tuple["_Block | ProblemType | None", ...] | None = None,
        **kwargs,
    ):
        """Declares one term.

        Parameters
        ----------
        slots : str
            Where ``fn``'s returned arrays go, one letter of ``KCMFR`` per returned array — see :py:data:`_SLOTS`.
        fn : Callable
            The operator, called as ``fn(groupElem, **kwargs)``, so its first positional parameter must be a :py:class:`~EasyFEA.FEM._GroupElem`.
        dim : int, optional
            Dimension of the element groups this term integrates over, `groupElem`'s or ``mesh.dim`` by default.
        tag : str, optional
            Restricts the term to a tagged element subset; groups without the tag are skipped.
        groupElem : _GroupElem, optional
            Restricts the term to this element group of the simulation's mesh, so a mixed mesh can carry one model per group.
        constant : bool, optional
            Declares the contribution independent of the solution, so it is built once and reused across Newton iterations and time steps while its arguments are unchanged. `fn` must then read nothing but its arguments, so it cannot be a bound method. Defaults to False.
        problemTypes : tuple, optional
            Block of each returned array, one entry per slot: ``(rows, cols)`` for a matrix, ``rows`` for a vector, None for the integrated problem's diagonal. Defaults to the diagonal for every slot.
        **kwargs
            Passed on to ``fn`` as given; ``elements`` is added when `tag` is set.
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

        if problemTypes is None:
            problemTypes = (None,) * len(slots)
        if len(problemTypes) != len(slots):
            raise ValueError(
                f"problemTypes must give one per slot: {len(slots)} for {slots!r}, got "
                f"{len(problemTypes)}."
            )
        # None stays the integrated problem's diagonal, resolved per integration
        blocks: list[_Block | None] = []
        for slot, block in zip(slots, problemTypes):
            if block is not None and not isinstance(block, tuple):
                block = (block, block)
            if block is not None and _SLOT_RANK[slot] != 3 and block[0] != block[1]:
                raise ValueError(
                    f"slot {slot!r} takes a vector: give it its rows' problem type, not the "
                    f"pair {block}."
                )
            blocks.append(block)

        if groupElem is not None and dim not in (None, groupElem.dim):
            raise ValueError(
                f"dim={dim} contradicts groupElem={groupElem.elemType}, of dimension {groupElem.dim}."
            )

        # written only here: a `constant=True` term's cache is keyed and checked on these
        self.__slots = tuple(slots)
        self.__fn = fn
        self.__kwargs: dict[str, Any] = kwargs
        self.__scale: float = 1.0
        self.__declared = (
            self.__slots
        )  # a scaled copy keeps its source's, to share its cache entry
        self.__dim = groupElem.dim if dim is None and groupElem is not None else dim
        self.__tag = tag
        self.__groupElem = groupElem
        self.__constant = constant
        self.__problemTypes = tuple(blocks)

        if constant:
            if inspect.ismethod(fn):
                raise TypeError(
                    f"{fn.__name__} is a bound method: it can read state no argument carries, "
                    "so constant=True could not see it change. Pass what it reads as arguments "
                    "of a plain function, or drop constant=True."
                )
            for n, v in kwargs.items():
                _Check_comparable(fn, n, v)

    def __repr__(self) -> str:
        name = getattr(self.__fn, "__name__", repr(self.__fn))
        tag = "" if self.__tag is None else f", tag={self.__tag!r}"
        group = (
            ""
            if self.__groupElem is None
            else f", groupElem={self.__groupElem.elemType}"
        )
        blocks = (
            ""
            if all(b is None for b in self.__problemTypes)
            else f", problemTypes={self.__problemTypes}"
        )
        return f"Term({''.join(self.__slots)!r}, {name}{tag}{group}{blocks})"

    @property
    def slots(self) -> tuple[str, ...]:
        """Where each array the operator returns belongs, one letter of ``KCMFR`` each. Read-only: the integration routes on it."""
        return self.__slots

    @property
    def constant(self) -> bool:
        """Whether the contribution is built once and reused across Newton iterations and time steps."""
        return self.__constant

    @property
    def scale(self) -> float:
        """Multiplies every array the operator returns, set by :py:meth:`Scaled`."""
        return self.__scale

    def Scaled(self, coef: float, slots: str | None = None) -> "Term":
        """A copy of this term with its arrays multiplied by `coef`, routed to `slots` if given. Applied outside the cache, so a constant term and its scaled copies integrate once."""
        term = Term(
            slots or "".join(self.__slots),
            self.__fn,
            dim=self.__dim,
            tag=self.__tag,
            groupElem=self.__groupElem,
            constant=self.__constant,
            problemTypes=self.__problemTypes,
            **self.__kwargs,
        )
        term.__scale = self.__scale * coef
        term.__declared = self.__declared
        return term

    # ----------------------------------------------
    # Reuse across the per-assembly rebuild of the list
    # ----------------------------------------------

    def _Cached(
        self,
        simu: "_Simu",
        groupElem: "_GroupElem",
        spaceGroups: list["_GroupElem"] | None = None,
        blocks: tuple[_Block, ...] | None = None,
    ) -> Any:
        """Unscaled contribution of a ``constant=True`` term on one group and `blocks`, kept with `simu`'s cached computed values (so a mesh change drops it) and reused by the term declared the same way, and by its :py:meth:`Scaled` copies, while the argument values are unchanged."""
        key = (
            self.__fn,
            self.__declared,
            self.__dim,
            self.__tag,
            self.__groupElem,
            groupElem,
            blocks,
        )
        kwargs = self.__kwargs
        cache = cached_computed_values(simu)
        hit = cache.get(key)
        if (
            hit is None
            or hit[0].keys() != kwargs.keys()
            or not all(_Same(v, kwargs[n]) for n, v in hit[0].items())
        ):
            hit = cache[key] = (
                copy.deepcopy(kwargs),
                self._Evaluate(groupElem, spaceGroups=spaceGroups),
            )
        return hit[1]

    # ----------------------------------------------
    # Evaluation
    # ----------------------------------------------

    def _Get_groups(self, mesh: "Mesh") -> list["_GroupElem"]:
        """Element groups this term integrates over, restricted to its group and tag-filtered."""
        groups = mesh.Get_list_groupElem(self.__dim)
        if self.__groupElem is not None:
            if not any(g is self.__groupElem for g in groups):
                raise ValueError(
                    f"{self!r}: its {self.__groupElem.elemType} group is not a group of the "
                    f"simulation's mesh in dimension {self.__dim}, e.g. a group of another mesh "
                    "or of a mesh replaced since."
                )
            groups = [self.__groupElem]
        if self.__tag is None:
            return groups
        return [g for g in groups if self.__tag in g.elementTags]

    def _Get_field_groups(
        self, simu: "_Simu", fields: tuple["ProblemType", ...]
    ) -> list[tuple["_GroupElem", ...]]:
        """Each of :py:meth:`_Get_groups`, followed by the group of each of `fields` on its elements."""
        groups = self._Get_groups(simu.mesh)
        lists = [
            simu._Get_FunctionSpace(field).Get_list_groupElem(self.__dim)
            for field in fields
        ]
        return [
            (groupElem, *spaceGroups)
            for groupElem, *spaceGroups in zip(
                simu.mesh.Get_list_groupElem(self.__dim), *lists
            )
            if any(groupElem is group for group in groups)
        ]

    def _Get_blocks(self, problemType: "ProblemType") -> tuple[_Block, ...]:
        """``(rows, cols)`` of each slot in `problemType`'s contributions, a vector's cols being its rows."""
        return tuple(
            (problemType, problemType) if block is None else block
            for block in self.__problemTypes
        )

    def _Get_elements(self, groupElem: "_GroupElem") -> _types.IntArray | None:
        """Element indices this term is restricted to within `groupElem`, or None."""
        return None if self.__tag is None else groupElem.Get_Elements_Tag(self.__tag)

    def _Evaluate(
        self,
        groupElem: "_GroupElem",
        spaceGroups: list["_GroupElem"] | None = None,
    ) -> Any:
        """Calls the operator on `spaceGroups`, the group of each field on `groupElem`'s elements (`groupElem` itself by default), with its kwargs and, when tagged, ``elements``."""
        kwargs = self.__kwargs
        if self.__tag is not None:
            kwargs = {**kwargs, "elements": self._Get_elements(groupElem)}
        return self.__fn(*(spaceGroups or (groupElem,)), **kwargs)


def Integrate_terms(
    simu: "_Simu",
    terms: list[Term],
    problemType: "ProblemType | None" = None,
) -> list[_Contribution]:
    r"""Every contribution of `terms`, unsummed, in term then group then slot order; filter it by slot, block or group.

    Each carries :py:attr:`_Simu.thickness` times the term's scale as its ``coef``, so operators stay free of simulation-level geometry; a vector's block is ``(rows, rows)``.
    """

    if problemType is None:
        problemType = simu.problemType

    thickness = simu.thickness
    out: list[_Contribution] = []

    for term in terms:
        blocks = term._Get_blocks(problemType)
        fields = tuple(dict.fromkeys(field for block in blocks for field in block))
        coef = thickness * term.scale

        # operators see each field's own group: P2 u on TETRA4 integrates on TETRA10
        for groupElem, *spaceGroups in term._Get_field_groups(simu, fields):
            if term.constant:
                arrays = term._Cached(simu, groupElem, spaceGroups, blocks)
            else:
                arrays = term._Evaluate(groupElem, spaceGroups)
            if not isinstance(arrays, tuple):
                arrays = (arrays,)
            if len(arrays) != len(term.slots):
                raise ValueError(
                    f"{term!r} declares {len(term.slots)} slot(s) but the operator returned "
                    f"{len(arrays)} array(s). One slot per returned array — slots route "
                    "arrays, they do not copy one into several."
                )

            for slot, block, array in zip(term.slots, blocks, arrays):
                if array is None:
                    continue
                _Check_rank(term, slot, array)
                _Check_shape(term, slot, block, array, groupElem, simu)
                out.append(_Contribution(slot, block, groupElem, array, coef, term))

            # not held through the next term's evaluation
            del arrays

    return out


_SLOT_RANK = {"K": 3, "C": 3, "M": 3, "F": 2, "R": 2}
"""Rank a slot's contribution must have: ``K``/``C``/``M`` take an ``(Ne, n, n)`` matrix, ``F``/``R`` an ``(Ne, n)`` vector."""


def _Check_rank(term: Term, slot: str, contribution: Any) -> None:
    """Rejects a contribution whose rank does not match its slot — almost always slot letters written in the wrong order, since they are positional."""
    rank = _SLOT_RANK[slot]
    if np.ndim(contribution) == rank:
        return
    kind = "matrix" if rank == 3 else "vector"
    raise ValueError(
        f"{term!r} sent a {np.ndim(contribution)}-D {np.shape(contribution)} array to slot "
        f"{slot!r}, which takes a {kind}. Slot letters are positional — one per returned array, "
        "in the order the operator returns them."
    )


def _Check_shape(
    term: Term,
    slot: str,
    block: _Block,
    contribution: Any,
    groupElem: "_GroupElem",
    simu: "_Simu",
) -> None:
    """Rejects a contribution not shaped by its block's spaces on `groupElem`: ``(Ne, n_rows, n_cols)`` for a matrix, ``(Ne, n_rows)`` for a vector — a swapped ``(rows, cols)`` pair fails here."""
    fields = block[: _SLOT_RANK[slot] - 1]
    shape = (
        groupElem.Ne,
        *(
            simu._Get_FunctionSpace(field).Get_connect(groupElem).shape[1]
            * simu.Get_dof_n(field)
            for field in fields
        ),
    )
    if np.shape(contribution) == shape:
        return
    raise ValueError(
        f"{term!r} sent a {np.shape(contribution)} array to slot {slot!r} of block "
        f"{block}, which takes {shape}: rows from {block[0]}'s space, columns from "
        f"{block[1]}'s."
    )
