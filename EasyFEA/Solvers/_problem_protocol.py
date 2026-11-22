# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""The `_Problem` protocol: a system, its comm, A and b and constraints at an iterate."""

from typing import Any, Protocol

from scipy import sparse

from ..Utilities import _types
from ._constraints import Constraints
from ._system import System


class _Problem(Protocol):
    """What a nonlinear strategy reads: A and b at an iterate, its constraints, the system and `comm` (None = no reduction)."""

    @property
    def system(self) -> System: ...

    @property
    def comm(self) -> Any: ...

    def Get_A_b(
        self, x: _types.FloatArray
    ) -> tuple[sparse.csr_matrix, _types.FloatArray]: ...

    def Constraints(self, x: _types.FloatArray) -> Constraints: ...
