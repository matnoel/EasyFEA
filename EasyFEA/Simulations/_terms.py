# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""Declarative description of a simulation's local matrix system: a list of :class:`Term`, folded into ``{"K": {(rows, cols): {groupElem: K_e}}, "C": ..., "M": ..., "F": {rows: {groupElem: F_e}}}`` by :func:`Fold_terms`."""

import copy
import inspect
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

from ..Utilities import _types
from ..Utilities._cache import cached_computed_values

if TYPE_CHECKING:
    from typing import Concatenate

    from ..FEM import Mesh, _GroupElem
    from ..FEM._function_space import _FunctionSpace
    from ._simu import _Simu
    from ._problem_type import ProblemType

    # what a checker enforces: the first positional argument is the element group, the rest are the term's kwargs
    _Operator = Callable[Concatenate[_GroupElem, ...], Any]
else:
    # Concatenate reached typing in 3.10
    _Operator = Callable[..., Any]

_Block = tuple["ProblemType", "ProblemType"]
"""``(rows, cols)``: the field a contribution tests, and the field it is differentiated by."""

_LocalSystem = dict[str, dict["_Block | ProblemType", dict["_GroupElem", np.ndarray]]]
"""What :func:`Fold_terms` returns, by slot then block then group: a matrix's block is ``(rows, cols)``, a vector's its ``rows``."""


_SLOTS = "KCMFR"
"""Slot letters, one per array the operator returns. ``K``/``C``/``M`` are the stiffness, damping and mass matrices; ``F`` is an external load, added to the right-hand side as-is; ``R`` is an **internal** force, so it reaches ``F`` negated. Every nonlinear operator returns an ``R`` — it is the residual of ``A·Δu = -R(u)`` — and ``F`` is reserved for genuine loads such as a phase-field source."""

_SLOT_FIELD = {"K": "u", "C": "v", "M": "a"}
"""Field a single-slot term's residual contracts against: ``K·u_t``, ``C·v_t``, ``M·a_t``."""

_INJECTABLE = frozenset({"u", "elements"})
"""Parameters the fold fills in by name when the caller does not supply them. Kept short and generic on purpose: injection makes an operator's *parameter names* part of the contract, so it covers only quantities every simulation has. Anything domain-specific — a hyperelastic state, a material — is passed by a named method instead, as in :py:meth:`HyperElastic.Get_terms`."""


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

    Evaluating a term on one element group yields one **contribution** per slot — the arrays :func:`Fold_terms` accumulates into ``{"K": {(rows, cols): {groupElem: K_e}}, "C": ..., "M": ..., "F": {rows: {groupElem: F_e}}}``, the local form of :math:`\Krm \, \mathrm{u} + \Crm \, \vrm + \Mrm \, \arm = \Frm`.

    ``slots`` names where the returned arrays go — one letter of ``KCMFR`` per array, in order, so an operator returning two arrays takes exactly two letters. Slots *route* arrays; they never copy one into several, so ``"KC"`` on an operator returning a single matrix is an error, not a request to damp with the stiffness::

        Term("K", Bilinear.LinearizedElasticity, C=material.C)
        Term("KR", self.__Stress)                # R is an *internal* force: F -= R

    ``fn`` is called as ``fn(groupElem, **kwargs)``, so any operator whose first positional argument is a ``_GroupElem`` can be named directly. Anything needing per-group work — a hyperelastic state, a contact projection — is a named method of the simulation with the same shape.

    ``problemTypes`` places each array in a block of a system over several fields, one entry per slot: a ``(rows, cols)`` pair for a matrix, the rows alone for a vector, None for the folded problem's own diagonal (the default)::

        Term("KRKKR", self.__MixedIncompressibility, problemTypes=((u, u), u, (u, p), (p, u), p))

    A term spanning several fields is handed one group per field, in the order its blocks name them, and no ``u``: it reaches each field's values itself.
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
            Block of each returned array, one entry per slot: ``(rows, cols)`` for a matrix, ``rows`` for a vector, None for the folded problem's diagonal. Defaults to the diagonal for every slot.
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

        if problemTypes is None:
            problemTypes = (None,) * len(slots)
        if len(problemTypes) != len(slots):
            raise ValueError(
                f"problemTypes must give one per slot: {len(slots)} for {slots!r}, got "
                f"{len(problemTypes)}."
            )
        # None stays the folded problem's diagonal, resolved per fold
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

        # resolved once, not per group per Newton iteration
        self.__injectable = tuple(n for n in parameters[1:] if n in _INJECTABLE)

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
        """Where each array the operator returns belongs, one letter of ``KCMFR`` each. Read-only: the fold routes on it."""
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
        """Unscaled contribution of a ``constant=True`` term on one group and `blocks`, kept with `simu`'s cached computed values (so a mesh change drops it) and reused by the term declared the same way, and by its :py:meth:`Scaled` copies, while the argument values are unchanged. ``u`` is deliberately unavailable: a term that needs it is not constant, and fails loudly on the missing argument rather than silently freeze the first iterate."""
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
        """``(rows, cols)`` of each slot in `problemType`'s fold, a vector's cols being its rows."""
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
        u: _types.FloatArray | None = None,
        spaceGroups: list["_GroupElem"] | None = None,
    ) -> Any:
        """Calls the operator on `spaceGroups`, the group of each field on `groupElem`'s elements (`groupElem` itself by default), injecting the arguments it declares and the caller left out."""
        values = {"u": u, "elements": self._Get_elements(groupElem)}
        kwargs = dict(self.__kwargs)
        for name in self.__injectable:
            if name not in kwargs and values[name] is not None:
                kwargs[name] = values[name]
        return self.__fn(*(spaceGroups or (groupElem,)), **kwargs)


def Fold_terms(
    simu: "_Simu",
    terms: list[Term],
    problemType: "ProblemType | None" = None,
) -> _LocalSystem:
    r"""Folds `terms` into the local matrix system ``{"K": {(rows, cols): {groupElem: K_e}}, "C": ..., "M": ..., "F": {rows: {groupElem: F_e}}}``; a slot no term fills stays empty.

    Contributions **accumulate**: several terms may target one group, and each fills only the slots it declares. Every contribution is scaled by :py:attr:`_Simu.thickness`, so operators stay free of simulation-level geometry.

    In a nonlinear simulation a **single-slot** term also contributes its own residual — :math:`-\Krm_e \, \mathrm{u}_t`, :math:`-\Crm_e \, \vrm_t`, :math:`-\Mrm_e \, \arm_t`, with the field of its columns, into its rows' ``F``. A single slot means the contribution is linear in its field, so that product *is* its residual. Multi-slot terms return their own force and are never given one.
    """

    if problemType is None:
        problemType = simu.problemType

    thickness = simu.thickness
    isNonLinear = simu.isNonLinear
    diagonal = (problemType, problemType)

    iterates: dict["ProblemType", dict[str, _types.FloatArray | None]] = {}

    def Iterates(problemType: "ProblemType") -> dict[str, _types.FloatArray | None]:
        if problemType not in iterates:
            u, v, a = simu.Get_u_v_a(problemType)
            iterates[problemType] = {"u": u, "v": v, "a": a}
        return iterates[problemType]

    u_t = Iterates(problemType)["u"]

    out: _LocalSystem = {"K": {}, "C": {}, "M": {}, "F": {}}

    def Add(
        slot: str,
        block: "_Block | ProblemType",
        groupElem: "_GroupElem",
        contribution: np.ndarray,
    ) -> None:
        groups = out[slot].setdefault(block, {})
        if groupElem in groups:
            contribution = groups[groupElem] + contribution
        groups[groupElem] = contribution

    for term in terms:
        blocks = term._Get_blocks(problemType)
        isDiagonal = all(block == diagonal for block in blocks)
        fields = tuple(dict.fromkeys(field for block in blocks for field in block))

        # operators see each field's own group: P2 u on TETRA4 integrates on TETRA10
        for groupElem, *spaceGroups in term._Get_field_groups(simu, fields):
            if term.constant:
                contributions = term._Cached(simu, groupElem, spaceGroups, blocks)
            else:
                contributions = term._Evaluate(
                    groupElem, u_t if isDiagonal else None, spaceGroups
                )
            if not isinstance(contributions, tuple):
                contributions = (contributions,)
            if len(contributions) != len(term.slots):
                raise ValueError(
                    f"{term!r} declares {len(term.slots)} slot(s) but the operator returned "
                    f"{len(contributions)} array(s). One slot per returned array — slots route "
                    "arrays, they do not copy one into several."
                )

            own = None  # this term's own matrix contribution, for its residual

            for slot, block, contribution in zip(term.slots, blocks, contributions):
                if contribution is None:
                    continue
                _Check_rank(term, slot, contribution)
                _Check_shape(
                    term,
                    slot,
                    block,
                    contribution,
                    groupElem,
                    simu,
                )
                # always allocates, so a cached `constant=True` array is never written into
                contribution = thickness * term.scale * contribution
                if slot == "R":  # an internal force opposes the right-hand side
                    Add("F", block[0], groupElem, -contribution)
                elif slot == "F":
                    Add("F", block[0], groupElem, contribution)
                else:
                    Add(slot, block, groupElem, contribution)
                    own = contribution

            # from `own`, never the accumulated slot: two terms may fill one slot on one group
            if isNonLinear and len(term.slots) == 1 and own is not None:
                cols = blocks[0][1]
                residual = _Residual(
                    term,
                    groupElem,
                    own,
                    Iterates(cols),
                    simu._Get_FunctionSpace(cols),
                    simu.Get_dof_n(cols),
                )
                if residual is not None:
                    Add("F", blocks[0][0], groupElem, residual)

            # not held through the next term's evaluation
            del contributions, contribution, own

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


def _Residual(
    term: Term,
    groupElem: "_GroupElem",
    matrix_e: np.ndarray,
    fields: dict[str, _types.FloatArray | None],
    space: "_FunctionSpace",
    dof_n: int,
) -> np.ndarray | None:
    """``-A_e · x_t`` for a single-slot term, where `x_t` is the field that slot multiplies, gathered on `groupElem`'s DOFs in `space`.

    Sound because a single slot means the contribution is linear in that field, so the product *is* the residual. Returns None when the field does not exist under the active time scheme (no velocity in a static problem).
    """

    x_t = fields[_SLOT_FIELD[term.slots[0]]]
    if x_t is None:
        return None

    x_e = x_t[space.Get_assembly_e(groupElem, dof_n)]

    return -np.einsum("eij,ej->ei", matrix_e, x_e, optimize=True)
