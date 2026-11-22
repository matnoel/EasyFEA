# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Where the dofs of problem types sit end to end in one system, and which rank owns each."""

from typing import TYPE_CHECKING, Callable

import numpy as np

from ..FEM._function_space import _FunctionSpace
from ..Utilities import _types
from ..Utilities._mpi import Owned
from ._problem_type import ProblemType

if TYPE_CHECKING:
    from ..FEM import _GroupElem
    from ._simu import _Simu


def _Field_size(simu: "_Simu", problemType: ProblemType) -> int:
    """Number of dofs of one problem type."""
    return simu._Get_FunctionSpace(problemType).Nn * simu.Get_dof_n(problemType)


class _DofLayout:
    """The dofs of `problemTypes` end to end: sizes, slices, element dofs, owners and global ids."""

    def __init__(self, simu: "_Simu", problemTypes: tuple[ProblemType, ...]) -> None:
        self.problemTypes = problemTypes
        self.blocks: dict[ProblemType, tuple[_FunctionSpace, int, int]] = {}
        """`{problemType: (space, dof_n, offset)}`"""
        self.slices: dict[ProblemType, slice] = {}
        offset = 0
        for problemType in problemTypes:
            space = simu._Get_FunctionSpace(problemType)
            dof_n = simu.Get_dof_n(problemType)
            self.blocks[problemType] = (space, dof_n, offset)
            self.slices[problemType] = slice(offset, offset + space.Nn * dof_n)
            offset += space.Nn * dof_n
        self.size = offset
        """number of dofs"""

    def Dofs_e(self, problemType: ProblemType, group: "_GroupElem") -> _types.IntArray:
        """`problemType`'s dofs on each element of `group`, in the system's vector."""
        space, dof_n, offset = self.blocks[problemType]
        return space.Get_assembly_e(group, dof_n) + offset

    def Compose(
        self, get: Callable[[ProblemType], np.ndarray], isDofs: bool = False
    ) -> np.ndarray:
        """`get(problemType)` of each problemType end to end, dof ids shifted to the problemType's offset when `isDofs`; `get(problemType)` itself for one alone."""
        if len(self.problemTypes) == 1:
            return get(self.problemTypes[0])
        parts = [np.asarray(get(problemType)) for problemType in self.problemTypes]
        if isDofs:
            parts = [
                part.astype(int) + self.blocks[problemType][2]
                for part, problemType in zip(parts, self.problemTypes)
            ]
        return np.concatenate(parts)

    def Owners(self) -> tuple[_types.IntArray | None, int]:
        """Owner rank of each dof, None when never partitioned, and this rank."""
        spaces = [space for space, *_ in self.blocks.values()]
        rank = spaces[0].rank
        if any(space._dofOwners is None for space in spaces):
            return None, rank
        owners = [
            np.repeat(space._dofOwners, dof_n)
            for space, dof_n, _ in self.blocks.values()
        ]
        return np.concatenate(owners), rank

    def Owned_mask(self) -> _types.BoolArray:
        """Whether this rank owns each dof."""
        owners, rank = self.Owners()
        isOwned = np.zeros(self.size, dtype=bool)
        isOwned[Owned(owners, rank)] = True
        return isOwned

    def GlobalIds(self) -> _types.IntArray:
        """Global id of each dof, unique over the ranks."""
        globalIds = []
        for f, (space, dof_n, _) in enumerate(self.blocks.values()):
            nodes = space.globalNodes[space._dofNodes].astype(np.int64)
            dofs = nodes[:, None] * dof_n + np.arange(dof_n)
            globalIds.append((dofs * len(self.problemTypes) + f).ravel())
        return np.concatenate(globalIds)
