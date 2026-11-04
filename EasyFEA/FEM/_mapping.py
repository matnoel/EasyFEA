# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Mapping module.\n
The geometry map x(ξ) = Σ Nᵢ(ξ) xᵢ of an element group."""

from typing import TYPE_CHECKING

import numpy as np

from ._fearray import FeArray, Det, Inv, Normalize
from ._utils import MatrixType

from ..Utilities import _types
from ..Utilities._cache import cache_computed_values

if TYPE_CHECKING:
    from ._group_elem import _GroupElem


class _Mapping:
    """Geometry map of a group's elements: x(ξ), Jacobians, normals, element frame and ξ(x); reference configuration cached, deformed per call."""

    def __init__(self, groupElem: "_GroupElem"):
        self.__groupElem = groupElem

    @cache_computed_values
    def __Get_reference_coord_e(self) -> _types.FloatArray:
        """Node coordinates of each element (Ne, nPe, 3)."""
        groupElem = self.__groupElem
        return groupElem.coord[groupElem._global_to_local_nodes[groupElem.connect]]

    def __Get_coord_e(
        self, displacementMatrix: _types.AnyArray | None = None
    ) -> _types.FloatArray:
        """Node coordinates of each element (Ne, nPe, 3), displaced if ``displacementMatrix`` is given."""
        if displacementMatrix is None:
            return self.__Get_reference_coord_e()
        groupElem = self.__groupElem
        displacementMatrix = np.asarray(displacementMatrix, dtype=float)
        shape = (groupElem.Ncoords, 3)
        assert (
            displacementMatrix.shape == shape
        ), f"displacmentMatrix must be of shape {shape}"
        coord = groupElem.coord + displacementMatrix[groupElem.nodes]
        return coord[groupElem._global_to_local_nodes[groupElem.connect]]

    # --------------------------------------------------------------------------------------------
    # Element frame and normals
    # --------------------------------------------------------------------------------------------

    def Get_frame_e(
        self, displacementMatrix: _types.AnyArray | None = None
    ) -> _types.FloatArray:
        """Element frame (Ne, 3, 3), columns (i, j, k): i along the first edge, k normal to a face."""

        groupElem = self.__groupElem
        dim, Ne = groupElem.dim, groupElem.Ne
        coord_e = self.__Get_coord_e(displacementMatrix)

        if dim in [0, 3]:
            return np.eye(3)[np.newaxis, :].repeat(Ne, axis=0)

        if dim == 2:
            # Ensure outward-facing normals for all mesh faces.
            coord_e = coord_e[:, groupElem.faces]

        points1 = coord_e[:, 0]
        points2 = coord_e[:, 1]

        i = Normalize(points2 - points1)

        if dim == 1:
            # Segments

            e1 = np.array([1, 0, 0])[np.newaxis, :].repeat(i.shape[0], axis=0)
            e2 = np.array([0, 1, 0])[np.newaxis, :].repeat(i.shape[0], axis=0)
            e3 = np.array([0, 0, 1])[np.newaxis, :].repeat(i.shape[0], axis=0)

            inDim = groupElem.inDim
            if inDim == 1:
                j = e2
                k = e3

            elif inDim == 2:
                j = np.cross((0, 0, 1), i)
                k = np.cross(i, j, axis=1)

            elif inDim == 3:
                j = np.cross(i, e1, axis=1)

                rep2 = np.where(np.linalg.norm(j, axis=1) < 1e-12)
                rep1 = np.setdiff1d(range(i.shape[0]), rep2)

                k1 = j.copy()
                j1 = np.cross(k1, i, axis=1)

                j2 = np.cross(e2, i, axis=1)
                k2 = np.cross(i, j2, axis=1)

                j = np.zeros_like(i)
                j[rep1] = j1[rep1]
                j[rep2] = j2[rep2]
                j = Normalize(j)

                k = np.zeros_like(i, dtype=float)
                k[rep1] = k1[rep1]
                k[rep2] = k2[rep2]
                k = Normalize(k)

        else:
            if groupElem.topology == "TRI":
                points3 = coord_e[:, 2]
            elif groupElem.topology == "QUAD":
                points3 = coord_e[:, 3]
            else:
                raise TypeError("unknown type")

            j = Normalize(points3 - points1)
            k = Normalize(np.cross(i, j, axis=1))
            j = Normalize(np.cross(k, i, axis=1))

        frame_e = np.zeros((Ne, 3, 3), dtype=float)

        frame_e[:, :, 0] = i
        frame_e[:, :, 1] = j
        frame_e[:, :, 2] = k

        return frame_e

    def Get_normals_e_pg(
        self,
        matrixType: MatrixType,
        displacementMatrix: _types.FloatArray | None = None,
        normalize: bool = True,
    ) -> FeArray:
        """Normals (Ne, nPg, 3): cross product of the tangents, unit if ``normalize``."""

        groupElem = self.__groupElem
        dim = groupElem.dim
        assert dim in [1, 2], "You can't compute normals for 0D or 3D elements."

        coord_e = self.__Get_coord_e(displacementMatrix)

        # get the first derivatives of the shape functions as a (nPg, dim, nPe) array
        dN_pg = groupElem.Get_dN_pg(matrixType)
        dNdr_pg = dN_pg[:, 0]

        dxdr_e_pg = np.einsum("pn,end->epd", dNdr_pg, coord_e, optimize="optimal")

        if dim == 1:
            normals_e_pg = np.cross((0, 0, 1), dxdr_e_pg)
        else:
            dNds_pg = dN_pg[:, 1]
            dxds_e_pg = np.einsum("pn,end->epd", dNds_pg, coord_e, optimize="optimal")
            normals_e_pg = np.cross(dxdr_e_pg, dxds_e_pg)

        if normalize:
            normals_e_pg = Normalize(normals_e_pg)

        return FeArray.asfearray(normals_e_pg)

    # --------------------------------------------------------------------------------------------
    # Forward map
    # --------------------------------------------------------------------------------------------

    def Get_x_e_pg(
        self,
        matrixType: MatrixType,
        elements: _types.IntArray | None = None,
        displacementMatrix: _types.FloatArray | None = None,
    ) -> FeArray.FeArrayALike:
        """x(ξ) at the integration points (Ne, nPg, 3)."""

        N_pg = self.__groupElem.Get_N_pg(matrixType)

        coord_e = self.__Get_coord_e(displacementMatrix)
        if elements is not None and elements.size != 0:
            coord_e = coord_e[elements]

        x_e_pg = np.einsum("pin,end->epd", N_pg, coord_e, optimize="optimal")

        return FeArray.asfearray(x_e_pg)

    def Get_x_n(
        self, xi_n: _types.FloatArray, elements_n: _types.IntArray
    ) -> _types.FloatArray:
        """x(ξ) of each (ξ, element) pair (Nn, 3)."""
        groupElem = self.__groupElem
        coord_n = self.__Get_coord_e()[elements_n]  # (Nn, nPe, 3)
        if groupElem.dim == 0:
            return coord_n[:, 0]
        N_n = groupElem._Eval_Functions(groupElem._N(), xi_n)[:, 0]  # (Nn, nPe)
        return np.einsum("np,npd->nd", N_n, coord_n)

    def __Get_dx_n(
        self, xi_n: _types.FloatArray, elements_n: _types.IntArray
    ) -> _types.FloatArray:
        """∂x/∂ξ of each (ξ, element) pair (Nn, dim, 3), one row per reference axis."""
        groupElem = self.__groupElem
        coord_n = self.__Get_coord_e()[elements_n]  # (Nn, nPe, 3)
        dN_n = groupElem._Eval_Functions(groupElem._dN(), xi_n)  # (Nn, dim, nPe)
        return np.einsum("nkp,npd->nkd", dN_n, coord_n)

    # --------------------------------------------------------------------------------------------
    # Jacobians
    # --------------------------------------------------------------------------------------------

    def __Get_L(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        """L = [ξᵢ, 1] (nPe, dim+1) and L⁺: x = ξ J + b at every node is L [J; b] = xᵢ."""
        groupElem = self.__groupElem
        L = np.column_stack([groupElem.Get_Local_Coords(), np.ones(groupElem.nPe)])
        return L, np.linalg.pinv(L)

    @cache_computed_values
    def Get_affine_e(self) -> _types.BoolArray:
        """Whether each element's map is affine (Ne,): always on a linear simplex, else (I - L L⁺) xᵢ = 0."""
        groupElem = self.__groupElem
        dim, nPe = groupElem.dim, groupElem.nPe
        if nPe == dim + 1:
            return np.ones(groupElem.Ne, dtype=bool)

        coord_e = self.__Get_coord_e()[..., : groupElem.inDim]  # (Ne, nPe, inDim)
        L, invL = self.__Get_L()
        # part of the node coordinates that no affine map gives: (I - L L⁺) x_i
        nonAffine_e = (np.eye(nPe) - L @ invL) @ coord_e  # (Ne, nPe, inDim)
        size_e = np.ptp(coord_e, axis=1).max(1)  # (Ne,)
        return np.abs(nonAffine_e).max((1, 2)) <= 1e-12 * size_e

    @cache_computed_values
    def __Get_affine_inverse(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        """(invJ_e, b_e) of x = ξ J + b, [J; b] = L⁺ xᵢ in the first inDim coordinates: J⁻¹, or J⁺ for a line or face in space; nan off affine elements."""
        groupElem = self.__groupElem
        dim = groupElem.dim
        coord_e = self.__Get_coord_e()[..., : groupElem.inDim]  # (Ne, nPe, inDim)
        affine_e = self.Get_affine_e()

        Jb_e = self.__Get_L()[1] @ coord_e  # (Ne, dim+1, inDim)
        J_e, b_e = Jb_e[:, :dim], Jb_e[:, dim]  # (Ne, dim, inDim), (Ne, inDim)
        invJ_e = np.full(J_e.transpose(0, 2, 1).shape, np.nan)
        if J_e.shape[1] == J_e.shape[2]:
            invJ_e[affine_e] = np.linalg.inv(J_e[affine_e])
        else:
            invJ_e[affine_e] = np.linalg.pinv(J_e[affine_e])

        return invJ_e, b_e

    def __Get_dx_e_pg(
        self, matrixType: MatrixType, coord_e: _types.FloatArray
    ) -> _types.FloatArray:
        """∂(coord_e)/∂ξ at the integration points (Ne, nPg, dim, ncols), once per affine element."""
        dN_pg = self.__groupElem.Get_dN_pg(matrixType)  # (nPg, dim, nPe)
        affine_e = self.Get_affine_e()
        if affine_e.all():
            dx_e = dN_pg[0] @ coord_e  # (Ne, dim, ncols)
            return np.broadcast_to(
                dx_e[:, None], (dx_e.shape[0], *dN_pg.shape[:2], dx_e.shape[2])
            )
        if not affine_e.any():
            return dN_pg @ coord_e[:, None]
        dx_e_pg = np.empty((coord_e.shape[0], *dN_pg.shape[:2], coord_e.shape[2]))
        dx_e_pg[affine_e] = (dN_pg[0] @ coord_e[affine_e])[:, None]
        dx_e_pg[~affine_e] = dN_pg @ coord_e[~affine_e, None]
        return dx_e_pg

    def __Get_frame_coord_e(self) -> _types.FloatArray:
        """Node coordinates (Ne, nPe, dim): global, or in the element frame for a line or face in space."""
        groupElem = self.__groupElem
        coord_e = self.__Get_coord_e()
        if groupElem.dim != groupElem.inDim:
            coord_e = coord_e @ self.Get_frame_e()
        return coord_e[:, :, : groupElem.dim]

    @cache_computed_values
    def Get_J_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Jacobian matrix, one row per reference axis: ``J[k, d] = ∂x_d/∂ξ_k``, in the element frame for a line or face in space."""
        if self.__groupElem.dim == 0:
            return None  # type: ignore [return-value]
        return FeArray.asfearray(
            self.__Get_dx_e_pg(matrixType, self.__Get_frame_coord_e())
        )

    @cache_computed_values
    def Get_jacobian_e_pg(
        self, matrixType: MatrixType, absoluteValues=True
    ) -> FeArray.FeArrayALike:
        """Jacobian: actual over reference length, area or volume; √det(T Tᵀ) of the tangents T = ∂x/∂ξ for a line or face in space, so a curved one keeps its curvature."""
        groupElem = self.__groupElem
        if groupElem.dim == 0:
            return None  # type: ignore [return-value]

        if groupElem.dim == groupElem.inDim:
            jacobian_e_pg = Det(self.Get_J_e_pg(matrixType))
            if absoluteValues:
                jacobian_e_pg = np.abs(jacobian_e_pg)
        else:
            T_e_pg = self.__Get_dx_e_pg(matrixType, self.__Get_coord_e())
            jacobian_e_pg = np.sqrt(Det(T_e_pg @ T_e_pg.swapaxes(-1, -2)))

        return FeArray.asfearray(jacobian_e_pg)

    @cache_computed_values
    def Get_invJ_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Inverse Jacobian matrix: ``dN_e_pg = invJ_e_pg @ dN_pg``."""
        if self.__groupElem.dim == 0:
            return None  # type: ignore [return-value]
        return Inv(self.Get_J_e_pg(matrixType))

    def __Get_hessian_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """Reference second derivatives ``N,kl`` (nPg, dim, dim, nPe): pure ones from ``_ddN``, mixed ones by a 5-point stencil on ``_dN``, exact on these ≤ quartic bases."""
        groupElem = self.__groupElem
        dim = groupElem.dim
        xi_pg = groupElem.Get_gauss(matrixType).coord
        ddN_pg = groupElem.Get_ddN_pg(matrixType)  # (nPg, dim, nPe)

        hessian_pg = np.zeros((ddN_pg.shape[0], dim, dim, ddN_pg.shape[2]))
        dN = groupElem._dN()
        h = 0.5
        for l in range(dim):
            dN_s = [
                groupElem._Eval_Functions(dN, xi_pg + s * h * np.eye(dim)[l])
                for s in (-2, -1, 1, 2)
            ]
            ddN_l = (dN_s[0] - 8 * dN_s[1] + 8 * dN_s[2] - dN_s[3]) / (12 * h)
            hessian_pg[:, :, l] = ddN_l
            hessian_pg[:, l, l] = ddN_pg[:, l]

        return hessian_pg

    @cache_computed_values
    def Get_ddN_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Physical second derivatives ``N,xx N,yy N,zz`` (Ne, nPg, dim, nPe), with the mixed reference derivatives and the curvature term ∂²x/∂ξ²."""
        groupElem = self.__groupElem

        invJ_e_pg = self.Get_invJ_e_pg(matrixType)  # (Ne, nPg, dim, dim)
        dN_e_pg = groupElem.Get_dN_e_pg(matrixType)  # (Ne, nPg, dim, nPe)
        hessian_pg = self.__Get_hessian_pg(matrixType)  # (nPg, dim, dim, nPe)

        # ∂²x_m/∂ξ_k∂ξ_l (Ne, nPg, dim, dim, dim)
        coord_e = self.__Get_frame_coord_e()
        ddx_e_pg = np.einsum("pkln,enm->epklm", hessian_pg, coord_e)
        # N,kl - N,m ∂²x_m/∂ξ_k∂ξ_l
        hessian_e_pg = FeArray.from_pg(hessian_pg, groupElem.Ne) - np.einsum(
            "epmn,epklm->epkln", dN_e_pg, ddx_e_pg
        )
        # N,aa = invJ_ak invJ_al (...)_kl
        ddN_e_pg = np.einsum(
            "epak,epal,epkln->epan", invJ_e_pg, invJ_e_pg, hessian_e_pg
        )

        return FeArray.asfearray(ddN_e_pg)

    # --------------------------------------------------------------------------------------------
    # Inverse map
    # --------------------------------------------------------------------------------------------

    def Get_xi(
        self, coordinates_n: _types.NumberArray, elements_n: _types.IntArray
    ) -> _types.FloatArray:
        """(ξ, η, ζ) of each node in its element by inverse mapping, outside the reference element if the node is outside, inf if Newton fails; the closest point for a line or face in space."""

        groupElem = self.__groupElem
        dim, inDim = groupElem.dim, groupElem.inDim
        coordinates_n = np.asarray(coordinates_n, dtype=float)  # (Nn, 3)
        Nn = coordinates_n.shape[0]
        if dim == 0 or Nn == 0:
            return np.zeros((Nn, dim))

        x_n = coordinates_n[:, :inDim]  # (Nn, inDim)
        invJ_e, b_e = self.__Get_affine_inverse()
        affine_n = self.Get_affine_e()[elements_n]
        iterative_n = ~affine_n

        # affine: ξ = (x - b) J⁻¹
        xi_n = np.empty((Nn, dim))
        xi_n[affine_n] = np.einsum(
            "nd,ndk->nk",
            x_n[affine_n] - b_e[elements_n[affine_n]],
            invJ_e[elements_n[affine_n]],
        )
        if not iterative_n.any():
            return xi_n

        # iterative: Newton (Gauss-Newton in space) on x(ξ) = x, from the reference element's center
        # _i = per iterative node, inf where it does not converge
        elements_i = elements_n[iterative_n]
        x_i = x_n[iterative_n]  # (Ni, inDim)
        Ni = x_i.shape[0]
        xi_i = np.tile(groupElem.Get_Local_Coords().mean(0), (Ni, 1))  # (Ni, dim)
        active_i = np.ones(Ni, dtype=bool)
        with np.errstate(all="ignore"):
            for _ in range(20):
                # _a = per active node
                nodes_a = np.flatnonzero(active_i)
                if nodes_a.size == 0:
                    break
                xi_a = xi_i[nodes_a]
                elements_a = elements_i[nodes_a]
                residual_a = self.Get_x_n(xi_a, elements_a)[:, :inDim] - x_i[nodes_a]
                # Jᵀ[d, k] = ∂x_d/∂ξ_k (Na, inDim, dim)
                Jt_a = self.__Get_dx_n(xi_a, elements_a)[..., :inDim].swapaxes(1, 2)
                delta_a = np.full_like(xi_a, np.inf)
                if dim == inDim:
                    regular_a = np.isfinite(Jt_a).all((1, 2)) & (
                        np.abs(np.linalg.det(Jt_a)) > 0
                    )
                    # Jᵀ δ = -r
                    delta_a[regular_a] = np.linalg.solve(
                        Jt_a[regular_a], -residual_a[regular_a, :, None]
                    )[..., 0]
                else:
                    JJt_a = Jt_a.swapaxes(1, 2) @ Jt_a  # (Na, dim, dim)
                    regular_a = np.isfinite(JJt_a).all((1, 2)) & (
                        np.abs(np.linalg.det(JJt_a)) > 0
                    )
                    # δ = -(Jᵀ)⁺ r
                    delta_a[regular_a] = -(
                        np.linalg.pinv(Jt_a[regular_a]) @ residual_a[regular_a, :, None]
                    )[..., 0]
                xi_i[nodes_a] = xi_a + delta_a
                # stops once converged or diverged
                active_i[nodes_a] = ~(np.abs(delta_a).max(1) <= 1e-12)
                active_i[nodes_a[~np.isfinite(delta_a).all(1)]] = False
        xi_i[active_i | ~np.isfinite(xi_i).all(1)] = np.inf
        xi_n[iterative_n] = xi_i

        return xi_n
