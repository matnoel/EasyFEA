# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import TYPE_CHECKING

import numpy as np

from ..Utilities import Terminal, Tic, _types

if TYPE_CHECKING:
    from ..FEM import Mesh
from ..FEM._utils import ElemType
from ..FEM import MatrixType, FeArray, Kinematics, Operators, _GroupElem

from ..Models import Result_strain_or_stress_field_e
from ..Models._utils import _Field_per_groupElem
from ..Models.InElastic._behavior import _Behavior

from ._simu import _Simu
from ._terms import Term
from ._problem_type import ProblemType


class InElastic(_Simu):
    r"""Quasi-static mechanics :math:`\diver{\Sig} + \fb = 0` by Newton-Raphson, the stress coming from a :class:`~EasyFEA.Models.InElastic._Behavior`.

    A converged :meth:`Solve` is a step: it commits the internal variables, which the next step integrates from and :meth:`Result` reads; :meth:`Save_Iter` only records them.
    """

    __INTERNAL_KEY = "internal"
    """Prefix of the saved iteration entries holding the internal variables, one per group and variable."""

    __EXTERNAL_KEY = "external"
    """Prefix of the saved iteration entries holding the external variables, one per name."""

    def __init__(
        self,
        mesh: "Mesh",
        model: _Behavior,
        folder: str = "",
        verbosity: bool = False,
        absTol: float = 1e-6,
        relTol: float = 1e-10,
        incTol: float = 1e-11,
        maxIter: int = 20,
    ):
        assert isinstance(model, _Behavior), "model must be a _Behavior"
        super().__init__(mesh, model, folder, verbosity)

        self._Solver_Set_Newton_Raphson_Algorithm(absTol, relTol, incTol, maxIter)

        self.__dt = 0.0
        # per group, at the last converged Solve; empty before any
        self.__internal: dict[_GroupElem, dict[str, FeArray]] = {}
        # nodal, by name, for the next Solve
        self.__external: dict[str, _types.FloatArray] = {}
        # those of the last converged Solve
        self.__solvedExternal: dict[str, _types.FloatArray] = {}

    @property
    def dt(self) -> float:
        """Time increment passed to the behavior."""
        return self.__dt

    @dt.setter
    def dt(self, value: float) -> None:
        assert value >= 0.0, "dt must be >= 0"
        self.__dt = value

    @property
    def external(self) -> dict[str, _types.FloatArray]:
        """The nodal external variables the next steps see, by name; set them with :meth:`Set_external`."""
        return {name: v.copy() for name, v in self.__external.items()}

    def Set_external(self, **values: float | _types.FloatArray) -> None:
        """Sets external variables for the next solves, each a scalar or a ``(Nn,)`` nodal field on this mesh, e.g. ``T=thermal.thermal``; results show them once solved."""
        Nn = self.mesh.Nn
        for name, v in values.items():
            assert (
                name in self.material.externalNames
            ), f"{type(self.material).__name__} reads no '{name}'"
            v = np.asarray(v, dtype=float)
            v = np.full(Nn, float(v)) if v.ndim == 0 else v.copy()
            assert v.shape == (Nn,), f"'{name}' must be a scalar or a (Nn,) array"
            self.__external[name] = v

    @property
    def material(self) -> _Behavior:
        return self.model  # type: ignore[return-value]

    @property
    def displacement(self) -> _types.FloatArray:
        """Displacement vector field, [uxi, uyi, (uzi), ...]."""
        return self._Get_u_n(self.problemType)

    def Get_unknowns(self, problemType=None) -> list[str]:
        return {2: ["x", "y"], 3: ["x", "y", "z"]}[self.dim]

    def Get_problemTypes(self) -> list[ProblemType]:
        return [ProblemType("inelastic")]

    def Get_dof_n(self, problemType=None) -> int:
        return self.dim

    def Get_x0(self, problemType=None):
        if self.displacement.size != self.mesh.Nn * self.dim:
            return np.zeros(self.mesh.Nn * self.dim)
        return self.displacement

    def Results_nodeFields_elementFields(
        self, details=False
    ) -> tuple[list[str], list[str]]:
        elementsField = ["Svm", "Stress", "Strain"] if details else ["Svm", "Stress"]
        return ["displacement", *self.material.externalNames], elementsField

    # --------------------------------------------------------------------------
    # Integration
    # --------------------------------------------------------------------------

    def __Groups(self) -> list[_GroupElem]:
        return self.mesh.Get_list_groupElem(self.dim)

    def __Internal_e_pg(self, groupElem: _GroupElem) -> dict[str, FeArray]:
        """The internal variables committed at the last converged solve, virgin before any."""
        if not self.__internal:
            nPg = groupElem.Get_gauss(MatrixType.rigi).nPg
            return self.material.Virgin_internals_e_pg(groupElem.Ne, nPg)
        internal = self.__internal.get(groupElem)
        # they could be projected onto the new mesh instead
        assert (
            internal is not None
        ), "the internal variables cannot follow a mesh change"
        return internal

    def __Solved_external(self) -> dict[str, _types.FloatArray]:
        """The external variables of the last converged solve, the current ones before any."""
        return self.__solvedExternal if self.__internal else self.__external

    def __External_e_pg(
        self, groupElem: _GroupElem, external: dict[str, _types.FloatArray]
    ) -> dict[str, FeArray.FeArrayALike]:
        """The nodal ``external``, interpolated at the Gauss points."""
        N_pg = FeArray.asfearray(groupElem.Get_N_pg(MatrixType.rigi)[np.newaxis, :, 0])
        # they could be interpolated onto the new mesh instead
        assert all(
            v.size == groupElem.Ncoords for v in external.values()
        ), "the external variables cannot follow a mesh change - set them again"
        return {
            name: N_pg @ groupElem.Locates_sol_e(v, asFeArray=True)
            for name, v in external.items()
        }

    def __Integrate(
        self, u: _types.FloatArray, groupElem: _GroupElem
    ) -> tuple[FeArray, FeArray, dict[str, FeArray]]:
        return self.material.Integrate(
            Kinematics(groupElem, u),
            self.__Internal_e_pg(groupElem),
            self.__dt,
            **self.__External_e_pg(groupElem, self.__external),
        )

    def _Calc_Epsilon(
        self, matrixType: MatrixType = MatrixType.rigi
    ) -> FeArray.FeArrayALike | dict:
        """Strain (Ne, pg, 3 or 6): an ``FeArray`` on one group, ``{groupElem: FeArray}`` on several."""
        return _Field_per_groupElem(
            lambda groupElem: Kinematics(
                groupElem, self.displacement, matrixType
            ).Compute_Epsilon(),
            self.__Groups(),
        )

    def _Calc_Sigma(
        self, matrixType: MatrixType = MatrixType.rigi
    ) -> FeArray.FeArrayALike | dict:
        """Stress (Ne, pg, 3 or 6) under the committed internal state: an ``FeArray`` on one group, ``{groupElem: FeArray}`` on several."""
        assert (
            matrixType == MatrixType.rigi
        ), "the internal state lives at the rigi points"
        return _Field_per_groupElem(
            lambda groupElem: self.material.Compute_Sigma(
                Kinematics(groupElem, self.displacement, matrixType),
                self.__Internal_e_pg(groupElem),
                **self.__External_e_pg(groupElem, self.__Solved_external()),
            ),
            self.__Groups(),
        )

    def Solve(self) -> _types.FloatArray:
        """Solves one step and commits its internal variables; Newton asserts before committing if it does not converge."""
        u = super().Solve()
        self.__internal = {g: self.__Integrate(u, g)[2] for g in self.__Groups()}
        self.__solvedExternal = dict(self.__external)
        return u

    def Get_terms(self, problemType=None) -> list[Term]:
        """One term: the tangent ``∫BᵀC_alg B`` and the internal force ``∫Bᵀσ``."""
        return [Term("KR", self.__Assemble)]

    def __Assemble(
        self, groupElem: _GroupElem, matrixType: MatrixType = MatrixType.rigi
    ) -> tuple[np.ndarray, np.ndarray]:
        u = self._Solver_Get_Newton_Raphson_current_solution()
        sigma_e_pg, C_e_pg, _ = self.__Integrate(u, groupElem)

        tic = Tic()
        K_e = Operators.Bilinear.LinearizedElasticity(
            groupElem, C_e_pg, matrixType=matrixType
        )
        R_e = Operators.Linear.InternalForce(
            groupElem, sigma_e_pg, matrixType=matrixType
        )
        tic.Tac("Matrix", f"Construct K_e and R_e ({groupElem.elemType})", False)
        return K_e, R_e

    # --------------------------------------------------------------------------
    # Iterations
    # --------------------------------------------------------------------------

    def Save_Iter(self, iter=None):
        external = self.__Solved_external()
        # so that every saved iteration can give its stress back
        missing = set(self.material.externalNames) - set(external)
        assert not missing, f"set {sorted(missing)} with Set_external before saving"
        if iter is None:
            iter = {}
        iter["displacement"] = self.displacement
        # flat, so that each is seen as an element field
        for groupElem, internal in self.__internal.items():
            for name, v in internal.items():
                key = f"{InElastic.__INTERNAL_KEY}/{groupElem.elemType.value}/{name}"
                iter[key] = v
        for name, v in external.items():
            iter[f"{InElastic.__EXTERNAL_KEY}/{name}"] = v

        return super().Save_Iter(iter)

    def Set_Iter(self, iter: int = -1, resetAll=False) -> dict:
        results = super().Set_Iter(iter)
        if results is None:
            return results

        u = results["displacement"]
        self._Set_solutions(self.problemType, u, np.zeros_like(u), np.zeros_like(u))
        # the groups of the iteration's mesh, which super().Set_Iter restored
        groups = {g.elemType: g for g in self.__Groups()}
        internal: dict[_GroupElem, dict[str, FeArray]] = {}
        external: dict[str, _types.FloatArray] = {}
        for key, v in results.items():
            prefix, _, rest = key.partition("/")
            if prefix == InElastic.__INTERNAL_KEY:
                elemType, name = rest.split("/")
                internal.setdefault(groups[ElemType(elemType)], {})[name] = v
            elif prefix == InElastic.__EXTERNAL_KEY:
                external[rest] = v
        self.__internal = internal
        self.__external = external
        self.__solvedExternal = dict(external)
        return results

    # --------------------------------------------------------------------------
    # Results
    # --------------------------------------------------------------------------

    def __Scalar_states(self) -> list[str]:
        virgin = self.material.Virgin_internals_e_pg(1, 1)
        return [name for name, v in virgin.items() if v.shape == (1, 1)]

    def Results_Available(self) -> list[str]:
        results = ["displacement", "displacement_norm", "displacement_matrix"]
        if self.dim == 2:
            results.extend(["ux", "uy", "Sxx", "Syy", "Sxy", "Exx", "Eyy", "Exy"])
        else:
            results.extend(["ux", "uy", "uz"])
            results.extend(["Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy"])
            results.extend(["Exx", "Eyy", "Ezz", "Eyz", "Exz", "Exy"])
        results.extend(["Svm", "Stress", "Evm", "Strain"])
        # scalar internal variables are plottable
        results.extend(self.__Scalar_states())
        results.extend(self.material.externalNames)
        return results

    def Result(
        self,
        result: str,
        nodeValues: bool = True,
        iter: int | None = None,
    ) -> _types.FloatArray | float:
        if iter is not None:
            self.Set_Iter(iter)

        if not self._Results_Check_Available(result):
            return None  # type: ignore[return-value]

        Nn = self.mesh.Nn
        u = self.displacement

        if result in ["ux", "uy", "uz"]:
            values = u.reshape(Nn, -1)[:, {"x": 0, "y": 1, "z": 2}[result[-1]]]

        elif result == "displacement":
            values = u

        elif result == "displacement_norm":
            values = np.linalg.norm(u.reshape(Nn, -1), axis=1)

        elif result == "displacement_matrix":
            values = self.Results_displacement_matrix()

        elif result in self.material.externalNames:
            external = self.__Solved_external()
            if result not in external:
                Terminal.MyPrintError(f"'{result}' is not set, see Set_external.")
                return None  # type: ignore[return-value]
            values = external[result]

        elif result in self.__Scalar_states():
            values = np.concatenate(
                [
                    np.mean(self.__Internal_e_pg(g)[result], axis=1)
                    for g in self.__Groups()
                ]
            )

        elif ("S" in result or "E" in result) and "_norm" not in result:
            isStress = "S" in result and result != "Strain"
            res = result if result in ["Strain", "Stress"] else result[-2:]

            field = self._Calc_Sigma() if isStress else self._Calc_Epsilon()
            values = Result_strain_or_stress_field_e(field, res, self.material.coef)

        else:
            Terminal.MyPrintError(f"The result '{result}' is not implemented yet.")
            return None  # type: ignore[return-value]

        return self.Results_Reshape_values(values, nodeValues)

    def Results_Iter_Summary(
        self,
    ) -> tuple[list[int], list[tuple[str, _types.FloatArray]]]:
        return super().Results_Iter_Summary()

    def Results_dict_Energy(self) -> dict[str, float]:
        # the contract declares no free energy
        return {}

    def Results_displacement_matrix(self) -> _types.FloatArray:
        Nn = self.mesh.Nn
        matrix = np.zeros((Nn, 3))
        matrix[:, : self.dim] = self.displacement.reshape((Nn, -1))
        return matrix
