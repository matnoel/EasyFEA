# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Function space module.\n
The points a scalar field lives on, and their DOFs ``point * dof_n + d``."""

from typing import TYPE_CHECKING

import numpy as np

from ..Utilities import _types
from ..Utilities._mpi import MPI_RANK, MPI_SIZE, Concatenate_array, Reduce_sum
from ._utils import ElemType, MatrixType, _Order

if TYPE_CHECKING:
    from ._group_elem import _GroupElem
    from ._mesh import Mesh


class _FunctionSpace:
    """Scalar function space over a mesh's groups: numbers its points and expands them to DOFs point-major, the only place ``point * dof_n + d`` is computed."""

    def __init__(
        self,
        points: _types.IntArray,
        ownedPoints: _types.IntArray,
        localPoints: _types.IntArray,
        dict_groupElem: dict["_GroupElem", "_GroupElem"],
        supports: dict["_GroupElem", list[tuple]] | None = None,
    ):
        """Over ``points``, the sorted ids carrying a DOF, on the group mapped to each mesh group (whose connect holds the point ids), the group itself when isoparametric; ``supports`` gives, per main-dimension group, its partner columns off the vertices and the vertex columns of the entity each sits on."""
        self.__dict_groupElem = dict_groupElem
        self.__supports = {} if supports is None else supports
        self.__Npoints = points.size
        self.__ownedPoints = ownedPoints
        self.__localPoints = localPoints
        self.__rows: _types.IntArray | None = None
        # super-parametric: the nodes no group reaches carry no DOF
        if points.size > 0 and points[-1] >= points.size:
            rows = np.full(points[-1] + 1, -1, dtype=int)
            rows[points] = np.arange(points.size)
            rows.flags.writeable = False
            self.__rows = rows

    @property
    def Npoints(self) -> int:
        """number of points carrying a DOF (the mesh nodes when isoparametric)"""
        return self.__Npoints

    @property
    def ownedPoints(self) -> _types.IntArray:
        """points owned by this MPI rank (every point in serial): a vertex by its node's owner, a cell point by its element's rank, any other (added, or a geometry mid node) by the lowest owner of its support vertices"""
        return self.__ownedPoints

    @property
    def localPoints(self) -> _types.IntArray:
        """points of this MPI rank's main-dimension elements, ghosts included (every point in serial)"""
        return self.__localPoints

    @property
    def order(self) -> int:
        """highest order of the groups the field lives on, from which `Mesh._With_FunctionSpace` rebuilds it"""
        return max(group.order for group in self.__dict_groupElem.values())

    @property
    def coord(self) -> _types.FloatArray:
        """point coordinates (Nn + added points, 3), indexed by point id: the mesh nodes, then the added points; under MPI zero rows off this rank, as `Mesh.coord`"""
        groups = self.__dict_groupElem
        coord = np.zeros((max(group.Ncoords for group in groups.values()), 3))
        for groupElem in [*groups, *groups.values()]:
            coord[groupElem.nodes] = groupElem.coord
        return coord

    def _Update_coord(self) -> None:
        """Moves the added points with the geometry, from each group's own mapping (curved for TRI6 → 3); the numbering stays."""
        groups = self.__dict_groupElem
        # this rank's mesh nodes (Ncoords, 3), and which rows they fill (Ncoords,)
        coord = np.zeros((max(group.Ncoords for group in groups.values()), 3))
        known = np.zeros(coord.shape[0], dtype=bool)
        for groupElem in groups:
            coord[groupElem.nodes] = groupElem.coord
            known[groupElem.nodes] = True
        # every other point of a partner, at its local coordinates through the group's shape functions (Ne, nOff, 3); main-dimension groups last, so that they win on shared points
        for groupElem, partner in sorted(groups.items(), key=lambda gp: gp[0].dim):
            if partner is groupElem:
                continue
            xi = partner.Get_Local_Coords()[partner.Nvertex :]
            N = groupElem._Eval_Functions(groupElem._N(), xi)[:, 0]
            ids = partner.connect[:, partner.Nvertex :]
            # a rank holding a point but not its node (TRI3 beside a QUAD9 of another rank) takes it from its own element
            isOff = ~known[ids]
            coord[ids[isOff]] = (N @ coord[groupElem.connect])[isOff]
        for groupElem, partner in groups.items():
            if partner is not groupElem:
                partner.coord = coord

    def Get_groupElem(self, groupElem: "_GroupElem") -> "_GroupElem":
        """Group this space puts on the elements of ``groupElem`` (the group itself when isoparametric), whose nodes are point ids."""
        return self.__dict_groupElem[groupElem]

    def Get_rule(self, groupElem: "_GroupElem", matrixType: MatrixType) -> _Order:
        """Integration rule of ``matrixType`` on this space's elements of ``groupElem`` (rigi, P2 on TETRA4: TETRA10's 4 points, not TETRA4's 1), as an explicit order, so that data sampled on the geometry at it lands where the field's terms integrate."""
        # local: `_group_elem` imports this module at its top
        from ._group_elem import GroupElemFactory

        elemType = self.Get_groupElem(groupElem).elemType
        referenceElement = GroupElemFactory._Get_reference_element(elemType)
        return MatrixType.Order(referenceElement._Get_order(matrixType))

    def Get_connect(self, groupElem: "_GroupElem") -> _types.IntArray:
        """Point ids of each element (Ne, nPe), in the layout of the group's element in this space."""
        return self.Get_groupElem(groupElem).connect

    def Get_points(self, ids: _types.IntArray) -> _types.IntArray:
        """The ``ids`` carrying a DOF, in their order (TETRA10 mid nodes under P1 skipped), then every point on an edge, face or cell whose vertices ``ids`` all hold (element closure: the chord midpoint a circle predicate misses)."""
        ids = np.asarray(ids).ravel()
        # the ids carrying a DOF (nIds,)
        if self.__rows is not None:
            hasDof = ids < self.__rows.size
            hasDof[hasDof] = self.__rows[ids[hasDof]] >= 0
            ids = ids[hasDof]
        groups = self.__dict_groupElem
        selected = np.zeros(max(group.Ncoords for group in groups.values()), bool)
        selected[ids] = True
        # the points whose support vertices are all selected, read on the main-dimension elements, which every rank holding a point has (nAdded,)
        closure = [np.empty(0, dtype=int)]
        for groupElem, items in self.__supports.items():
            connect = groups[groupElem].connect
            for columns, support in items:
                closed = selected[groupElem.connect[:, support]].all(axis=2)
                closure.append(connect[:, columns][closed])
        added = np.unique(np.concatenate(closure))
        return np.concatenate([ids, added[~selected[added]]])

    def Get_nodeValues(
        self, dofsValues: _types.FloatArray, dof_n: int
    ) -> _types.FloatArray:
        """Values (Nn, dof_n) at the mesh nodes of the DOF vector ``dofsValues`` (Npoints*dof_n,): the reshape when isoparametric, else per group the values at its nodes or, super-parametric, lifted by the space's shape functions; zero rows at the nodes no main-dimension element covers (off this rank under MPI, as `Mesh.coord`)."""
        groups = self.__dict_groupElem
        # the values of each point (Npoints, dof_n)
        values = np.asarray(dofsValues).reshape(-1, dof_n)
        if all(partner is groupElem for groupElem, partner in groups.items()):
            return values
        # the node values (Nn, dof_n), filled by the main-dimension groups
        dim = max(groupElem.dim for groupElem in groups)
        nodeValues = np.zeros((max(g.Ncoords for g in groups), dof_n))
        for groupElem, partner in groups.items():
            if groupElem.dim != dim:
                continue
            if partner.nPe >= groupElem.nPe:
                # iso or sub-parametric: every geometry node is a point (Ne, nPe, dof_n)
                rows = self.__Get_rows(groupElem.connect)
                nodeValues[groupElem.connect] = values[rows]
            else:
                # super-parametric: the space's functions at the geometry's local coordinates (nPe, nPe partner) times the element values (Ne, nPe partner, dof_n); P1 on TETRA10 gives the edge means
                xi = groupElem.Get_Local_Coords()
                N = partner._Eval_Functions(partner._N(), xi)[:, 0]
                rows = self.__Get_rows(partner.connect)
                nodeValues[groupElem.connect] = N @ values[rows]
        return nodeValues

    def Get_pointValues(
        self, dofsValues: _types.FloatArray, dof_n: int
    ) -> _types.FloatArray:
        """DOF vector ``dofsValues`` indexed by point id, the layout `Kinematics` reads through a partner's connect: itself unless super-parametric, where the nodes without a DOF hold 0."""
        if self.__rows is None:
            return dofsValues
        Ncoords = max(group.Ncoords for group in self.__dict_groupElem.values())
        values = np.zeros(Ncoords * dof_n)
        points = np.flatnonzero(self.__rows >= 0)
        values[_FunctionSpace._Get_dofs(points, dof_n)] = dofsValues
        return values

    def Get_mesh(self) -> "Mesh":
        """Space mesh: the partner groups over the points, numbered as the DOF rows, so that a DOF vector reshaped (Npoints, dof_n) is its node field (TETRA4, 2 → TETRA10 cells; TETRA10, 1 → TETRA4); built on each call, so it follows the geometry; under MPI this rank's elements, ghosts included."""
        # local: `_mesh` and `_group_elem` import this module at their top
        from ._group_elem import GroupElemFactory
        from ._mesh import Mesh

        # the point coordinates in row order (Npoints, 3), the nodes no group reaches left out (super-parametric)
        coord = self.coord
        if self.__rows is not None:
            coord = coord[np.flatnonzero(self.__rows >= 0)]
        # the partner connects in rows, by element type (Ne, nPe)
        connects: dict[ElemType, list[_types.IntArray]] = {}
        for partner in self.__dict_groupElem.values():
            rows = self.__Get_rows(partner.connect)
            connects.setdefault(partner.elemType, []).append(rows)
        # one group per type: those of one type joined (SEG3 of a QUAD9 and SEG2 -> SEG3 of a TRI3), a cell both hold kept once, in order
        groups = {}
        for elemType, list_connect in connects.items():
            connect = np.concatenate(list_connect)
            _, first = np.unique(np.sort(connect, axis=1), axis=0, return_index=True)
            groups[elemType] = GroupElemFactory.Create(
                elemType, connect[np.sort(first)], coord
            )
        return Mesh(groups)

    def __Get_rows(self, points: _types.IntArray) -> _types.IntArray:
        """Position of each point id in the point list."""
        if self.__rows is None:
            return points
        rows = self.__rows[points]
        assert (rows >= 0).all(), "Some points carry no DOF in this space."
        return rows

    def Get_assembly_e(self, groupElem: "_GroupElem", dof_n: int) -> _types.IntArray:
        """DOFs of each element (Ne, nPe*dof_n), point-major."""
        return _FunctionSpace._Get_dofs(
            self.__Get_rows(self.Get_connect(groupElem)), dof_n
        )

    def Get_dofs(
        self,
        points: _types.IntArray,
        dof_n: int,
        components: list[int] | None = None,
    ) -> _types.IntArray:
        """DOFs of ``points`` for ``components`` (all by default), point-major and flat."""
        rows = self.__Get_rows(np.asarray(points).ravel())
        return _FunctionSpace._Get_dofs(rows, dof_n, components)

    @staticmethod
    def _Create_partner_groups(mesh: "Mesh", order: int) -> "_FunctionSpace | None":
        """Space on the partner group of each mesh group, the complete Lagrange element of ``order`` (TETRA4, 2 → TETRA10; QUAD8, 2 → QUAD9), None when every group is its own partner (iso); collective on a partitioned mesh."""
        # mesh nodes keep their ids, the added points take `Nn + k`, shared by every group and by every rank
        # local: `_group_elem` imports this module at its top
        from ._group_elem import GroupElemFactory

        # idea: a point off the geometry nodes is identified by its linear vertex function values keyed by global vertex id ({1: 2/3, 2: 1/3} on edge 1-2 at P3), the same in every element holding it; unlike xyz, it is exact and keeps the two lips of an open crack apart
        if order < 1:
            raise ValueError(f"No Lagrange element of order {order}.")
        Nn = mesh.Nn
        # these values are multiples of 1/order per direction, so integers once multiplied by `scale`; `base` and `limit` bound the int64 keys of step 5
        scale = order**3
        base = Nn * (scale + 1)
        limit = (np.iinfo(np.int64).max - base) // base
        # a partitioned mesh is numbered by all its ranks together, so that each holds the serial ids; a gathered one (rank partition set by `_Gather`, on rank 0 alone) by itself
        partitioned = MPI_SIZE > 1 and all(
            groupElem._rankPartition is None
            for groupElem in mesh.dict_groupElem.values()
        )

        def gather(array: np.ndarray) -> np.ndarray:
            """Every rank's ``array`` end to end on a partitioned mesh, ``array`` itself otherwise."""
            return Concatenate_array(array) if partitioned else array

        # owner rank of each node (Nn,), and how many ranks own it, summed over the ranks: one at most when the owned nodes partition the nodes
        owner = np.zeros((2, Nn), dtype=np.int64)
        owner[:, mesh._Get_mpi_owned_nodes()] = [[MPI_RANK], [1]]
        if partitioned:
            owner = Reduce_sum(owner)
        assert owner[1].max(initial=0) <= 1, "Nodes owned by several ranks."
        owner = owner[0]
        # mesh group -> its partner group, returned
        partners: dict["_GroupElem", "_GroupElem"] = {}
        # mesh group -> (partner elemType, partner connect (Ne, nPe), filled in step 5)
        dict_connect: dict["_GroupElem", tuple[ElemType, _types.IntArray]] = {}
        # support size -> one tuple per group: (its partner connect, the partner columns of that size (nCol,), their rows (Ne*nCol, size), whether this rank owns them (Ne, nCol))
        bySize: dict[int, list[tuple]] = {}
        # main-dimension group -> one tuple per support size: (its partner columns (nCol,), their support vertex columns (nCol, size)), for the element closure of `Get_points`
        supports: dict["_GroupElem", list[tuple]] = {}

        for groupElem in mesh.dict_groupElem.values():
            if groupElem.dim == 0:
                partners[groupElem] = groupElem
                continue
            # 1. the partner reference element (complete Lagrange of `order`, most nodes: QUAD9 not QUAD8) and the linear one, whose vertex functions give the keys
            topology = groupElem.elemType.topology
            # reference elements of the group's topology, every order
            refs = [
                GroupElemFactory._Get_reference_element(elemType)
                for elemType in ElemType
                if elemType.topology == topology
            ]
            complete = [ref for ref in refs if ref.order == order]
            if not complete:
                raise ValueError(f"No {topology} element of order {order} in EasyFEA.")
            partner = max(complete, key=lambda ref: ref.nPe)
            linear = min(refs, key=lambda ref: ref.nPe)
            # geometry node ids (Ne, nPe geometry), partner local coordinates (nPe, dim) and the elements this rank owns (Ne,)
            nodes = groupElem.connect.astype(np.int64)
            xi = partner.Get_Local_Coords()
            isOwned = np.isin(
                groupElem._globalElements, groupElem._Get_partitioned_data()[1]
            )

            # 2. the partner connect (Ne, nPe): the node id where a partner point sits on a geometry node (vertices, QUAD8 mid nodes), -1 elsewhere until step 5
            connect = np.full((groupElem.Ne, partner.nPe), -1, dtype=np.int64)
            # (nPe, nPe geometry): True where partner point j sits on geometry node i
            atNode = np.abs(xi[:, None] - groupElem.Get_Local_Coords()).max(2) < 1e-12
            j, i = np.nonzero(atNode)
            connect[:, j] = nodes[:, i]
            dict_connect[groupElem] = (partner.elemType, connect)

            # 3. for the partner points off the vertices: integer weights, the same for every element
            # partner columns off the vertices (nOff,) and the linear vertex functions there (nOff, Nvertex)
            offVertex = np.arange(partner.Nvertex, partner.nPe)
            N = linear._Eval_Functions(linear._N(), xi[offVertex])[:, 0]
            weights = np.rint(N * scale).astype(np.int64)
            assert np.allclose(N * scale, weights), "Partner points not equispaced."
            # 4. one row per (element, point), its (vertex, weight) pairs packed one integer each and sorted, grouped by support size (2 edge, 3 TRI face, 4 QUAD face or TETRA cell, ...) since rows must have equal length; weights counted down, so that ids grow away from the lower vertex
            # support size of each point off the vertices (nOff,)
            sizes = np.count_nonzero(weights, axis=1)
            for size in np.unique(sizes):
                # its points, as positions in offVertex (nCol,), their local vertices (nCol, size) and global vertices (Ne, nCol, size)
                columns = np.flatnonzero(sizes == size)
                support = np.nonzero(weights[columns])[1].reshape(columns.size, size)
                vertices = nodes[:, support]
                # vertex * (scale + 1) + scale - weight (Ne, nCol, size), then one row per (element, point)
                rows = vertices * (scale + 1) + scale
                rows -= np.take_along_axis(weights[columns], support, 1)
                rows = np.sort(rows, axis=2).reshape(-1, size)
                # this rank owns a cell point if it owns the element, any other (added or a QUAD8 mid node) if it is the lowest owner of the support vertices: then it holds every element containing the point, as a ghost at least (Ne, nCol)
                if groupElem.dim == mesh.dim and size == partner.Nvertex:
                    mine = np.repeat(isOwned[:, None], columns.size, axis=1)
                else:
                    mine = owner[vertices].min(axis=2) == MPI_RANK
                bySize.setdefault(size, []).append(
                    (connect, offVertex[columns], rows, mine)
                )
                if groupElem.dim == mesh.dim:
                    supports.setdefault(groupElem, []).append(
                        (offVertex[columns], support)
                    )

        # 5. the ids, one support size at a time over all groups and ranks: each row folded into one int64 key; equal keys are one point, the geometry node holding it or else Nn + k in the serial key order
        Npoints = Nn
        # the support sizes of every rank, so that all run the same collectives
        for size in np.unique(gather(np.array(list(bySize), dtype=np.int64))):
            items = bySize.get(size, [])
            # the rows of every group (nRows, size), folded into one key each (nRows,), below `bound`; past `limit`, keys are first ranked among every rank's keys, on the bound so that all ranks rank together
            table = np.vstack(
                [np.empty((0, size), np.int64)] + [rows for _, _, rows, _ in items]
            )
            key, bound = table[:, 0], base
            for column in table.T[1:]:
                if bound > limit:
                    ranked = np.unique(gather(np.unique(key)))
                    key, bound = np.searchsorted(ranked, key), ranked.size
                    assert bound <= limit, "Too many points for int64 keys."
                key = key * base + column
                bound *= base
            # every rank's distinct keys (Nkeys,), in the serial order, and the key of each row there (nRows,)
            keys, ids = np.unique(key, return_inverse=True)
            allKeys = np.unique(gather(keys))
            ids = np.searchsorted(allKeys, keys)[ids]
            # the geometry node each row's connect entry already holds, -1 if none (nRows,)
            held = np.concatenate(
                [np.empty(0, np.int64)]
                + [c[:, columns].ravel() for c, columns, _, _ in items]
            )
            # the point id of each key (Nkeys,): the geometry node holding it on any rank, else a new id Npoints + k
            keyPoint = np.full(allKeys.size, -1, dtype=np.int64)
            keyPoint[ids[held >= 0]] = held[held >= 0]
            # then every rank's held keys: a rank holding the point but not the node knows neither (TRI3 beside a QUAD9 of another rank)
            isHeld = np.flatnonzero(keyPoint >= 0)
            keyPoint[np.searchsorted(allKeys, gather(allKeys[isHeld]))] = gather(
                keyPoint[isHeld]
            )
            added = keyPoint < 0
            keyPoint[added] = Npoints + np.arange(np.count_nonzero(added))
            Npoints += np.count_nonzero(added)
            # each group's rows are the slice [start, end) of the table
            start = 0
            for connect, columns, rows, _ in items:
                end = start + rows.shape[0]
                connect[:, columns] = keyPoint[ids[start:end]].reshape(-1, columns.size)
                start = end

        # 6. the points this rank owns (Npoints,): the mesh nodes by their owner, then the points of step 4
        isMine = np.zeros(Npoints, dtype=bool)
        isMine[:Nn] = owner == MPI_RANK
        for items in bySize.values():
            for connect, columns, _, mine in items:
                isMine[connect[:, columns]] = mine
        # 7. the partner groups, or the group itself when it already is its partner; their coordinates are set by `_Update_coord`
        for groupElem, (elemType, connect) in dict_connect.items():
            if elemType == groupElem.elemType:
                partners[groupElem] = groupElem
            else:
                partners[groupElem] = GroupElemFactory.Create(
                    elemType, connect, np.zeros((Npoints, 3))
                )
        # 8. the points carrying a DOF (Npoints minus the nodes no partner reaches on any rank: TETRA10, 1 → the mid nodes), those this rank owns and those of its main-dimension elements
        reached = np.unique(np.concatenate([g.nodes for g in partners.values()]))
        points = np.concatenate(
            [np.unique(gather(reached[reached < Nn])), np.arange(Nn, Npoints)]
        )
        if all(partner is groupElem for groupElem, partner in partners.items()):
            return None
        main = [partners[g].nodes for g in mesh.Get_list_groupElem()]
        localPoints = np.unique(np.concatenate(main))
        space = _FunctionSpace(
            points, points[isMine[points]], localPoints, partners, supports
        )
        space._Update_coord()
        return space

    @staticmethod
    def _Get_dofs(
        points: _types.IntArray,
        dof_n: int,
        components: list[int] | None = None,
    ) -> _types.IntArray:
        """DOFs ``point * dof_n + d`` of each point, side by side (d = index in `Get_unknowns()`, x = 0, y = 1): points [7, 9] at dof_n = 2 give [14, 15, 18, 19]; a (Ne, nPe) connect gives (Ne, nPe*len(components))."""
        points = np.asarray(points, dtype=np.int64)
        d = np.arange(dof_n) if components is None else np.asarray(components)
        dofs = points[..., None] * dof_n + d.astype(np.int64)
        # explicit width: an empty (0, nPe) connect must keep its columns
        return dofs.reshape(*points.shape[:-1], points.shape[-1] * d.size)

    @staticmethod
    def _Get_rows_e(
        assembly_e: _types.IntArray, ncols: int | None = None
    ) -> _types.IntArray:
        """Row of each local matrix entry from the row DOFs (Ne, nrows), `ncols` columns wide (square by default)."""
        Ne, nrows = assembly_e.shape
        ncols = nrows if ncols is None else ncols
        return np.repeat(assembly_e, ncols).reshape((Ne, nrows * ncols))

    @staticmethod
    def _Get_columns_e(
        assembly_e: _types.IntArray, nrows: int | None = None
    ) -> _types.IntArray:
        """Column of each local matrix entry from the column DOFs (Ne, ncols), `nrows` rows tall (square by default)."""
        Ne, ncols = assembly_e.shape
        nrows = ncols if nrows is None else nrows
        return np.repeat(assembly_e, nrows, axis=0).reshape((Ne, nrows * ncols))
