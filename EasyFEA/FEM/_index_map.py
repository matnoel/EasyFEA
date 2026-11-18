# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Index map module.\n
The ghost nodes of a rank and their owners, refreshed by one neighbour exchange."""

import numpy as np

from ..Utilities import _types


class _IndexMap:
    """Rows a rank holds without owning them (ghosts), and the rows its neighbours ask of it; `Scatter_forward` copies the owners' values onto the ghosts with one neighbour all-to-all. Its communicator never pickles: rebuild the map instead."""

    def __init__(
        self,
        globalNodes: _types.IntArray,
        rows: _types.IntArray,
        owners: _types.IntArray,
        rank: int,
        comm,
    ):
        """Over the held nodes' global ids (sorted), their rows in a node-major vector and their owners (-1 for none); collective on `comm`."""
        isGhost = (owners != rank) & (owners >= 0)
        order = np.argsort(owners[isGhost], kind="stable")
        ghostOwners = owners[isGhost][order]
        ghostNodes = globalNodes[isGhost][order]
        self.__recvRows = rows[isGhost][order]
        sources, recvCounts = np.unique(ghostOwners, return_counts=True)

        # the ghosts each rank owns, asked of it; what the others ask of this rank
        splits = np.searchsorted(ghostOwners, np.arange(1, comm.Get_size()))
        asked = comm.alltoall(np.split(ghostNodes, splits))
        sendCounts = np.array([nodes.size for nodes in asked])
        destinations = np.flatnonzero(sendCounts)
        asked = np.concatenate([np.empty(0, dtype=globalNodes.dtype), *asked])
        assert np.isin(asked, globalNodes).all(), "A rank owns a node it does not hold."
        self.__sendRows = rows[np.searchsorted(globalNodes, asked)]

        self.__sendCounts = sendCounts[destinations]
        self.__recvCounts = recvCounts
        self.__graph = comm.Create_dist_graph_adjacent(
            sources.tolist(), destinations.tolist(), reorder=False
        )

    def Scatter_forward(self, array: np.ndarray, dof_n: int) -> None:
        """Copies onto the ghost rows of ``array`` (node-major, ``dof_n`` per node, any dtype) their owners' values, in place; collective."""
        values = array.reshape(-1, dof_n)
        assert np.shares_memory(values, array), "The array must be contiguous."
        send = np.ascontiguousarray(values[self.__sendRows])
        recv = np.empty((self.__recvRows.size, dof_n), dtype=values.dtype)
        self.__graph.Neighbor_alltoallv(
            [send, _IndexMap.__Counts(self.__sendCounts * dof_n)],
            [recv, _IndexMap.__Counts(self.__recvCounts * dof_n)],
        )
        values[self.__recvRows] = recv

    @staticmethod
    def __Counts(counts: _types.IntArray) -> tuple[list[int], list[int]]:
        """``(counts, displacements)`` of blocks laid end to end."""
        return counts.tolist(), (np.cumsum(counts) - counts).tolist()
