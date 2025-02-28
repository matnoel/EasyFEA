# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA.fem._utils import MatrixType
from EasyFEA import Mesher, np

class TestMesh:

    def test_construc_matrix(self):

        meshes = Mesher._Construct_2D_meshes()
        meshes.extend(Mesher._Construct_3D_meshes()[:2])

        matrixTypes = [MatrixType.rigi, MatrixType.mass]

        for mesh in meshes:

            dim = mesh.dim
            connect = mesh.connect
            elements = range(mesh.Ne)

            # Verify assembly
            assembly_e_test = np.array([[int(n * dim + d)for n in connect[e] for d in range(dim)] for e in elements])
            testAssembly = np.testing.assert_array_almost_equal(mesh.assembly_e, assembly_e_test, verbose=False)
            assert testAssembly is None

            # Verify lines_e 
            lines_e_test = np.array([[i for i in mesh.assembly_e[e] for j in mesh.assembly_e[e]] for e in elements])
            testLines = np.testing.assert_array_almost_equal(lines_e_test, mesh.linesVector_e, verbose=False)
            assert testLines is None

            # Verify lines_e 
            colonnes_e_test = np.array([[j for i in mesh.assembly_e[e] for j in mesh.assembly_e[e]] for e in elements])
            testColumns = np.testing.assert_array_almost_equal(colonnes_e_test, mesh.columnsVector_e, verbose=False)
            assert testColumns is None

            for matrixType in matrixTypes:
                mesh.Get_jacobian_e_pg(matrixType)
                mesh.Get_dN_e_pg(matrixType)
                mesh.Get_ddN_e_pg(matrixType)                
                mesh.Get_B_e_pg(matrixType)

    # TODO: def test_shape_functions