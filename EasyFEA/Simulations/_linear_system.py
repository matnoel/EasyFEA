# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Linear system `A x = b` of one problem."""

from typing import Callable, NamedTuple

import numpy as np
from scipy import sparse

from ..Utilities import _types
from ._terms import Same_provenance


class _Slot(NamedTuple):
    """One assembled K, C, M or F: its csr map key, what it was folded from, the matrix."""

    key: tuple
    provenance: list
    matrix: sparse.csr_matrix


class _LinearSystem:
    """`A x = b` of one problem key (a problem type or a monolithic problem's fields), dropped with the mesh: the last fold's K, C, M, F and A summed from them; it alone decides when A or its pattern changed (`stamp`, `patternStamp`)."""

    def __init__(self) -> None:
        self.__slots: dict[str, _Slot] = {}
        self.__valuesDirty = True
        self.__patternDirty = True
        self.__summed: list[tuple[float, str]] = []
        """`(coef, slot)` A was summed from"""
        self.__pattern: tuple = ()
        self.__A: sparse.csr_matrix | None = None
        self.stamp = 0
        """bumped on every new A"""
        self.patternStamp = 0
        """bumped on every new pattern of A"""

    def Assembled(
        self,
        slot: str,
        key: tuple,
        provenance: list,
        assemble: Callable[[], sparse.csr_matrix],
    ) -> sparse.csr_matrix:
        """The slot's matrix, assembled again unless folded from the same cached arrays on the same map; a new K, C or M dirties A, its pattern too on a new map."""
        old = self.__slots.get(slot)
        if (
            old is not None
            and old.key == key
            and Same_provenance(old.provenance, provenance)
        ):
            return old.matrix
        matrix = assemble()
        if slot in "KCM":
            self.__valuesDirty = True
            if old is None or old.key != key:
                self.__patternDirty = True
        self.__slots[slot] = _Slot(key, provenance, matrix)
        return matrix

    def Get_A(
        self, coefs: tuple[float, float, float], orphanDofs: _types.IntArray
    ) -> sparse.csr_matrix:
        """`coefK·K + coefC·C + coefM·M` (K alone when the others weigh nothing), a unit diagonal at `orphanDofs`; the same read-only object until a slot or a coefficient changes."""
        slots = self.__slots
        summed = [
            (coef, slot)
            for coef, slot in zip(coefs, "KCM")
            if slot == "K" or (coef != 0 and slots[slot].matrix.nnz)
        ]
        if self.__A is not None and not self.__valuesDirty and summed == self.__summed:
            return self.__A

        names = [slot for _, slot in summed]
        if self.__patternDirty or names != [slot for _, slot in self.__summed]:
            self.__pattern = self.__Pattern(names, orphanDofs)
            self.__patternDirty = False
            self.patternStamp += 1

        K = slots["K"].matrix
        indices, indptr, positions, diagonal = self.__pattern
        if [coef for coef, _ in summed] == [1.0] and diagonal.size == 0:
            A = K
        else:
            data = np.zeros(indices.size)
            for (coef, slot), position in zip(summed, positions):
                data[position] += coef * slots[slot].matrix.data
            data[diagonal] += 1.0
            A = sparse.csr_matrix((data, indices, indptr), shape=K.shape)
        A.has_canonical_format = True
        A.data.flags.writeable = False

        self.__A = A
        self.__summed = summed
        self.__valuesDirty = False
        self.stamp += 1
        return A

    def __Pattern(
        self, names: list[str], orphanDofs: _types.IntArray
    ) -> tuple[
        _types.IntArray, _types.IntArray, list[_types.IntArray | slice], _types.IntArray
    ]:
        """Union of the slots' patterns and the orphan diagonal: `(indices, indptr)`, each slot's positions in it, the diagonal's; exact zeros stay, so A's pattern never follows its values."""
        slots = [self.__slots[name] for name in names]
        K = slots[0].matrix
        if orphanDofs.size == 0 and all(slot.key == slots[0].key for slot in slots):
            return K.indices, K.indptr, [slice(None)] * len(slots), orphanDofs

        # entries as row·n + col, sorted: the canonical CSR order
        n = K.shape[0]
        keys = [
            np.repeat(np.arange(n, dtype=np.int64), np.diff(X.indptr)) * n + X.indices
            for X in (slot.matrix for slot in slots)
        ]
        keys.append(orphanDofs.astype(np.int64) * (n + 1))
        canon = np.unique(np.concatenate(keys))
        union = sparse.csr_matrix((np.ones(canon.size), divmod(canon, n)), shape=(n, n))
        positions = [np.searchsorted(canon, key) for key in keys]
        return union.indices, union.indptr, positions[:-1], positions[-1]
