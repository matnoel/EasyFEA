# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Backends solving `A x = b`: scipy, pypardiso and PETSc."""

import sys
import warnings
from enum import Enum
from functools import cache
from types import ModuleType
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as sla
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

from ..Utilities import _types

from ..Utilities._mpi import MPI_COMM, MPI_SIZE, Reduce_sum

if TYPE_CHECKING:
    from ._system import System

try:
    import pypardiso

    CAN_USE_PYPARDISO = True
except ModuleNotFoundError:
    CAN_USE_PYPARDISO = False


@cache
def _Get_PETSc() -> ModuleType | None:
    """Returns the initialized `petsc4py.PETSc` module, or None if petsc4py is unusable."""
    try:
        import petsc4py

        # must precede `from petsc4py import PETSc`, which would initialize PETSc with an empty command line
        petsc4py.init(
            [sys.argv[0], "-options_left", "no", *sys.argv[1:]], comm=MPI_COMM
        )

        from petsc4py import PETSc

        return PETSc

    except Exception:
        return None


class SolverType(str, Enum):
    r"""Solver type used to solve the linear system :math:`\Arm \, \mathrm{x} = \brm`"""

    pypardiso = "pypardiso"
    """pypardiso.spsolve"""

    petsc = "petsc"

    scipy = "scipy"
    """scipy.sparse.linalg.spsolve"""
    lsq_linear = "lsq_linear"
    """scipy.optimize.lsq_linear"""
    cg = "cg"
    """scipy.sparse.linalg.cg"""
    bicg = "bicg"
    """scipy.sparse.linalg.bicg"""
    gmres = "gmres"
    """scipy.sparse.linalg.gmres"""
    lgmres = "lgmres"
    """scipy.sparse.linalg.lgmres"""

    def __str__(self):
        return self.name


class Backend(ABC):
    """Solves `A y = b` for a `System`; `x0` the guess of y (an increment under Newton), padded with 0 past it."""

    def __init__(self, type: SolverType, x0: _types.FloatArray | None = None) -> None:
        self.type = type
        self.x0 = np.zeros(0) if x0 is None else x0

    def __str__(self) -> str:
        return str(self.type)

    @abstractmethod
    def Solve(
        self, A: sparse.csr_matrix, b: _types.FloatArray, system: "System", comm
    ) -> _types.FloatArray:
        """y of `A y = b` on the rows it solves, 0 elsewhere. Collective on `comm`."""

    def _Factors_saddle(self) -> bool:
        """Whether it factors a saddle point."""
        return False

    def _Describe(self) -> str:
        """Its name in an error."""
        return str(self.type)

    def Check_saddle_point(self, isSaddle: Callable[[], bool]) -> None:
        """Rejects a backend that cannot factor a saddle point when `isSaddle()`: a direct one only, PETSc's LU through an external package (Mumps or SuperLU_dist under MPI)."""
        if self._Factors_saddle() or not isSaddle():
            return
        accepted = (
            "petsc with preonly, lu and mumps or superlu_dist"
            if MPI_SIZE > 1
            else "pypardiso, scipy or petsc with preonly, lu and an external package (mumps)"
        )
        raise ValueError(
            f"A saddle point (a zero diagonal block or Lagrange multipliers) only a direct solver factors: use {accepted}, not {self._Describe()}."
        )

    def _X0(self, size: int) -> _types.FloatArray:
        x0 = np.zeros(size)
        x0[: self.x0.size] = self.x0
        return x0


class _Serial(Backend):
    """A serial backend, on every unknown."""

    def Solve(self, A, b, system, comm):
        rows = np.flatnonzero(~system.known)
        y = np.zeros(b.size)
        y[rows] = self._Solve_Axb(A[rows][:, rows], b[rows], self._X0(b.size)[rows])
        return y

    @abstractmethod
    def _Solve_Axb(
        self, A: sparse.csr_matrix, b: _types.FloatArray, x0: _types.FloatArray
    ) -> _types.FloatArray: ...


class _Direct(_Serial):
    """scipy's or pypardiso's spsolve."""

    def _Solve_Axb(self, A, b, x0):
        if self.type == SolverType.pypardiso and CAN_USE_PYPARDISO:
            return pypardiso.spsolve(A, b)
        if self.type == SolverType.scipy:
            return sla.spsolve(A, b)
        raise NotImplementedError(f"{self.type} is not implemented.")

    def _Factors_saddle(self):
        return MPI_SIZE == 1


class _Iterative(_Serial):
    """scipy's cg, bicg, gmres or lgmres."""

    def _Solve_Axb(self, A, b, x0):
        if self.type == SolverType.cg:
            x, _ = sla.cg(A, b, x0, maxiter=None)
        elif self.type == SolverType.bicg:
            x, _ = sla.bicg(A, b, x0, maxiter=None)
        elif self.type == SolverType.gmres:
            x, _ = sla.gmres(A, b, x0, maxiter=None)
        else:
            x, _ = sla.lgmres(A, b, x0)
        return x


class _LsqLinear(_Serial):
    """scipy's lsq_linear within the bounds `lb`, `ub`."""

    def __init__(self, x0=None, lb=None, ub=None) -> None:
        super().__init__(SolverType.lsq_linear, x0)
        self.lb = [] if lb is None else lb
        self.ub = [] if ub is None else ub

    def _Solve_Axb(self, A, b, x0):
        lb, ub = self.lb, self.ub
        assert len(lb) == len(ub) != 0
        from scipy.optimize import lsq_linear

        return lsq_linear(
            A,
            b,
            bounds=(lb, ub),
            tol=1e-10,
            method="trf",
            verbose=0,
        )["x"]


class _PETSc(Backend):
    """PETSc's KSP with `(ksp, pc, factor)` options, kept by the system, on its owned unknowns."""

    def __init__(self, options: tuple[str, str, str], x0=None) -> None:
        super().__init__(SolverType.petsc, x0)
        self.options = options

    def __str__(self) -> str:
        return ", ".join(self.options)

    def Solve(self, A, b, system, comm):
        y = np.zeros(b.size)
        y[system.rows] = system._petsc.Solve(
            A,
            b,
            self._X0(b.size),
            system.ids,
            system.rowIds,
            self.options,
            comm,
        )
        return y

    def _Factors_saddle(self):
        kspType, pcType, factor = self.options
        external = (
            factor in ("mumps", "superlu_dist") if MPI_SIZE > 1 else factor != "petsc"
        )
        return kspType == "preonly" and pcType == "lu" and external

    def _Describe(self):
        kspType, pcType, factor = self.options
        return f"petsc ({kspType}, {pcType}, {factor})"


def Get_backend(
    solverType: SolverType | str,
    options: tuple[str, str, str] | None = None,
    lb: _types.AnyArray | _types.Numbers | None = None,
    ub: _types.AnyArray | _types.Numbers | None = None,
    x0: _types.FloatArray | None = None,
) -> Backend:
    """The backend of a solver type: PETSc's `(ksp, pc, factor)` `options`, lsq_linear's bounds `lb`, `ub`, an initial guess `x0`."""
    solverType = SolverType(solverType)
    if solverType == SolverType.petsc:
        assert options is not None, "petsc needs its (ksp, pc, factor) options"
        return _PETSc(options, x0)
    if solverType == SolverType.lsq_linear:
        return _LsqLinear(x0, lb, ub)
    if solverType in (SolverType.pypardiso, SolverType.scipy):
        return _Direct(solverType, x0)
    return _Iterative(solverType, x0)


class _PETScSolver:
    """PETSc's KSP and Mat, kept: laid out again on new options, numbering or pattern, refilled on new values. Collective."""

    def __init__(self) -> None:
        self.__ksp: Any = None
        self.__matrix: Any = None
        self.__layout: tuple = ()
        """`(options, ids, rowIds, indptr, indices)` the Mat was laid out on, never written into"""
        self.__data: _types.FloatArray | None = None
        """the values it was filled with, never written into"""

    def Solve(
        self,
        A: sparse.csr_matrix,
        b: _types.FloatArray,
        x0: _types.FloatArray,
        ids: _types.IntArray,
        rowIds: _types.IntArray,
        options: tuple[str, str, str],
        comm,
    ) -> _types.FloatArray:
        """x at the rows `rowIds` numbers of `A x = b`; `ids` numbers the columns, -1 dropped. Collective on `comm`."""
        PETSc = _Get_PETSc()
        assert PETSc is not None, "petsc4py is not available"
        layout = (options, ids, rowIds, A.indptr, A.indices)
        relayout = self.__ksp is None or not _Same(layout, self.__layout)
        refill = relayout or not _Same((A.data,), (self.__data,))
        relayout, refill = Reduce_sum(np.array([relayout, refill], dtype=int), comm)
        if relayout:
            # two factorisations never held at once
            self.Free()
            self.__Create(PETSc, A, ids, rowIds, options, comm)
            self.__layout = layout
        if refill:
            self.__matrix.setValuesCOO(A.data)
            self.__data = A.data

        rows = np.flatnonzero(rowIds >= 0)
        x, rhs = self.__matrix.createVecs()
        rhs.array[:] = b[rows]
        x.array[:] = x0[rows]
        self.__ksp.solve(rhs, x)
        values = x.array.copy()
        x.destroy()
        rhs.destroy()
        if not self.__ksp.is_converged:
            kspType, pcType, factor = options
            raise Exception(
                f"petsc did not converge with ksp:{kspType}, pc:{pcType} and solver:{factor}."
            )
        return values

    def __Create(
        self,
        PETSc: ModuleType,
        A: sparse.csr_matrix,
        ids: _types.IntArray,
        rowIds: _types.IntArray,
        options: tuple[str, str, str],
        comm,
    ) -> None:
        """KSP and Mat laid out on A's pattern and the numbering, unfilled."""
        kspType, pcType, factor = options
        comm = comm or PETSc.COMM_SELF
        size = int(np.count_nonzero(rowIds >= 0))
        # -1 rows and columns are dropped: ghost rows, known dofs
        rowMap = PETSc.LGMap().create(rowIds.astype(PETSc.IntType), comm=comm)
        columnMap = PETSc.LGMap().create(ids.astype(PETSc.IntType), comm=comm)
        matrix = PETSc.Mat().create(comm=comm)
        matrix.setType("aij")
        matrix.setSizes(((size, None), (size, None)))
        matrix.setLGMap(rowMap, columnMap)
        rowMap.destroy()
        columnMap.destroy()
        # copies: PETSc overwrites them
        rowsCoo = np.repeat(
            np.arange(A.shape[0], dtype=PETSc.IntType), np.diff(A.indptr)
        )
        matrix.setPreallocationCOOLocal(rowsCoo, A.indices.astype(PETSc.IntType))

        ksp = PETSc.KSP().create(comm=comm)
        ksp.setOperators(matrix)
        ksp.setType(kspType)
        pc = ksp.getPC()
        pc.setType(pcType)
        pc.setFactorSolverType(factor)
        petscOptions = PETSc.Options()
        if factor == "mumps":
            if not petscOptions.hasName("mat_mumps_icntl_20"):
                # centralised right-hand side, see the warning; a no-op on one process
                petscOptions.setValue("mat_mumps_icntl_20", 0)
            elif (
                comm.Get_size() > 1 and petscOptions.getInt("mat_mumps_icntl_20") == 10
            ):
                warnings.warn(
                    "mat_mumps_icntl_20 10 (distributed right-hand side) gave wrong solutions with Mumps 5.8.2 and 5.9.1 over several processes: set it to 0.",
                    RuntimeWarning,
                )
        ksp.setFromOptions()
        self.__ksp, self.__matrix = ksp, matrix

    def Free(self) -> None:
        """Destroys the KSP and Mat. Collective on their communicator."""
        if self.__ksp is None:
            return
        ksp, matrix = self.__ksp, self.__matrix
        self.__ksp = self.__matrix = None
        self.__layout, self.__data = (), None
        # flush any remaining petsc4py reference cycles as suggested here:
        # https://gitlab.com/petsc/petsc/-/work_items/1309
        PETSc = _Get_PETSc()
        assert PETSc is not None
        PETSc.garbage_cleanup(ksp.getComm())
        ksp.destroy()
        matrix.destroy()


def _Same(a: tuple, b: tuple) -> bool:
    """Whether two tuples of arrays and values are equal item by item; the same array object is equal."""
    return len(a) == len(b) and all(
        u is v or (np.array_equal(u, v) if isinstance(u, np.ndarray) else u == v)
        for u, v in zip(a, b)
    )
