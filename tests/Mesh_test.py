# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA.fem._utils import MatrixType, FeArray
from EasyFEA import Mesher, ElemType, Mesh, np, Materials, Simulations
from EasyFEA.Geoms import Points

L = 2
H = 1

def __move_meshes(list_mesh: list[Mesh]):

    for mesh in list_mesh.copy():
        
        meshRot90x = mesh.copy(); meshRot90x.Rotate(90, direction=(1,0,0))
        meshRot90y = mesh.copy(); meshRot90y.Rotate(90, direction=(0,1,0))
        meshRot90z = mesh.copy(); meshRot90z.Rotate(90, direction=(0,0,1))

        mesh.Rotate(45, direction=(1,0,0))
        mesh.Rotate(45, direction=(0,1,0))
        mesh.Translate(-1)

        list_mesh.extend([meshRot90x, meshRot90y, meshRot90z, mesh])

    return list_mesh

@pytest.fixture
def meshes_2D() -> list[Mesh]:

    meshSize =H/3

    contour = Points([(0,0), (L,0), (L,H), (0,H)], meshSize)

    meshes_2D: list[Mesh] = []

    for elemType in ElemType.Get_2D():

        mesh = Mesher().Mesh_2D(contour, [], elemType, isOrganised=True)

        meshes_2D.append(mesh)

    meshes_2D = __move_meshes(meshes_2D)

    return meshes_2D

@pytest.fixture
def meshes_3D() -> list[Mesh]:

    meshSize =H/3

    contour = Points([(0,0), (L,0), (L,H), (0,H)], meshSize)

    meshes_3D: list[Mesh] = []

    for elemType in ElemType.Get_3D():

        mesh = Mesher().Mesh_Extrude(contour, [], [0,0,L], [3], elemType, isOrganised=True)

        meshes_3D.append(mesh)

    meshes_3D = __move_meshes(meshes_3D)

    return meshes_3D

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

    def test_area(self, meshes_2D: list[Mesh]):

        area = L * H

        for mesh in meshes_2D:

            assert (area - mesh.area)/area < 1e-10

    def test_volume(self, meshes_3D: list[Mesh]):

        volume = L * H * L

        for mesh in meshes_3D:

            assert (volume - mesh.volume)/volume < 1e-12

    def test_load(self, meshes_3D: list[Mesh]):

        mat = Materials.Elas_Isot(3, 210000*1e6, 0.33)
        rho = 7850 # kg/m3

        volume = L * H * L
        mass = volume * rho # kg
        F = mass * 9.81 # N
        P = 50

        for mesh in meshes_3D:

            simu = Simulations.ElasticSimu(mesh, mat)
            simu.rho = rho

            assert (mass - simu.mass)/mass < 1e-12

            simu.add_volumeLoad(mesh.nodes, [-rho*9.81], ["z"])

            rhs = simu._Solver_Apply_Neumann(simu.problemType)

            assert (F + rhs.sum())/F < 1e-12

            groupSurf = mesh.Get_list_groupElem(2)[-1]
            elems = groupSurf.Get_Elements_Tag("S0")
            nodes = groupSurf.Get_Nodes_Tag("S0")
            area = groupSurf.area_e[elems].sum()
            
            simu.Bc_Init()
            simu.add_surfLoad(nodes, [P/area], ["z"])
            
            rhs = simu._Solver_Apply_Neumann(simu.problemType)

            assert (P/area - rhs.sum())/(P/area) < 1e-12

            simu.Bc_Init()
            simu.add_pressureLoad(nodes, P/area)
            
            rhs = simu._Solver_Apply_Neumann(simu.problemType).toarray()
            load = np.linalg.norm(rhs.reshape(-1,3), axis=1)
            
            assert (P/area - load.sum())/(P/area) < 1e-12

@pytest.fixture
def FeArrays(meshes_2D):
    mesh: Mesh = meshes_2D[0]
    matrixType = MatrixType.mass

    scalar_e_pg = FeArray(mesh.Get_jacobian_e_pg(matrixType))
    vector_e_pg = FeArray(mesh.Get_dN_e_pg(matrixType)[:,:,0])
    matrix_e_pg = FeArray(mesh.Get_B_e_pg(matrixType))
    tensor_e_pg = FeArray(np.random.random((*scalar_e_pg.shape[:2],2,2,2,2)))

    return [scalar_e_pg, vector_e_pg, matrix_e_pg, tensor_e_pg]

def _check_arrays(array1, array2):
    if isinstance(array1, FeArray):
        array1 = np.asarray(array1)
    if isinstance(array2, FeArray):
        array2 = np.asarray(array2)
    norm_diff = np.linalg.norm(array1 - array2)
    assert norm_diff < 1e-12

class TestFeArray:

    def test_new_array(self, FeArrays: list[FeArray]):

        try:
            FeArray([0,1])
        except ValueError:
            pass
        
        scalar_e_pg, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays

        assert scalar_e_pg._idx == "ep"
        assert vector_e_pg._idx == "epi"
        assert matrix_e_pg._idx == "epij"
        assert tensor_e_pg._idx == "epijkl"

        assert scalar_e_pg._type == "scalar"
        assert vector_e_pg._type == "vector"
        assert matrix_e_pg._type == "matrix"
        assert tensor_e_pg._type == "tensor"


    def test_add_array(self, FeArrays: list[FeArray]):
        
        scalar_e_pg, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays

        Ne, nPg = scalar_e_pg.shape[:2]

        scalar = 1
        vector = np.arange(10)
        matrix = np.eye(10)*.1
        tensor = np.random.random((2,2,2,2))

        # (Ne, nPg) + (...)
        res = scalar_e_pg + scalar # + ()
        _check_arrays(res, scalar_e_pg + scalar)
        res = scalar_e_pg + vector # + (i)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1)) + vector)
        res = scalar_e_pg + matrix # + (i,j)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1,1)) + matrix)
        res = scalar_e_pg + tensor # + (i,j,k,l)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1,1,1,1)) + tensor)

        # (Ne, nPg) + (Ne, nPg, ...)
        res = scalar_e_pg + scalar_e_pg # + (Ne,nPg)
        _check_arrays(res, scalar_e_pg + scalar_e_pg)
        res = scalar_e_pg + vector_e_pg # + (Ne,nPg,i)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1)) + vector_e_pg)
        res = scalar_e_pg + matrix_e_pg # + (Ne,nPg,i,j)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1,1)) + matrix_e_pg)
        res = scalar_e_pg + tensor_e_pg # + (Ne,nPg,i,j,k,l)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1,1,1,1)) + tensor_e_pg)

        # (Ne, nPg, i) + (Ne, nPg, ...)
        try:
            res = vector_e_pg + scalar_e_pg # + (Ne,nPg)
        except ValueError: 
            pass
        res = vector_e_pg + vector_e_pg # + (Ne,nPg,i)
        try:
            res = vector_e_pg + matrix_e_pg # + (Ne,nPg,i,j)
        except ValueError:
            pass
        try:
            res = vector_e_pg + tensor_e_pg # + (Ne,nPg,i,j,k,l)
        except ValueError:
            pass

        # (Ne, nPg, i, j) + (Ne, nPg, ...)
        try:
            res = matrix_e_pg + scalar_e_pg # + (Ne,nPg)
        except ValueError: 
            pass
        try:
            res = matrix_e_pg + vector_e_pg # + (Ne,nPg,i)    
        except ValueError:
            pass
        res = matrix_e_pg + matrix_e_pg # + (Ne,nPg,i,j)
        try:
            res = matrix_e_pg + tensor_e_pg # + (Ne,nPg,i,j,k,l)
        except ValueError:
            pass

        # (Ne, nPg, i, j, k, l) + (Ne, nPg, ...)
        try:
            res = tensor_e_pg + scalar_e_pg # + (Ne,nPg)
        except ValueError: 
            pass
        try:
            res = tensor_e_pg + vector_e_pg # + (Ne,nPg,i)    
        except ValueError:
            pass
        try:
            res = tensor_e_pg + matrix_e_pg # + (Ne,nPg,i,j)
        except ValueError:
            pass
        res = tensor_e_pg + tensor_e_pg # + (Ne,nPg,i,j,k,l)

    def test_sub_array(self, FeArrays: list[FeArray]):
        
        scalar_e_pg, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays

        Ne, nPg = scalar_e_pg.shape[:2]

        scalar = 1
        vector = np.arange(10)
        matrix = np.eye(10)*.1
        tensor = np.random.random((2,2,2,2))

        # (Ne, nPg) - (...)
        res = scalar_e_pg - scalar # - ()
        _check_arrays(res, scalar_e_pg - scalar)
        res = scalar_e_pg - vector # - (i)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1)) - vector)
        res = scalar_e_pg - matrix # - (i,j)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1,1)) - matrix)
        res = scalar_e_pg - tensor # - (i,j,k,l)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1,1,1,1)) - tensor)

        # (Ne, nPg) - (Ne, nPg, ...)
        res = scalar_e_pg - scalar_e_pg # - (Ne,nPg)
        _check_arrays(res, scalar_e_pg - scalar_e_pg)
        res = scalar_e_pg - vector_e_pg # - (Ne,nPg,i)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1)) - vector_e_pg)
        res = scalar_e_pg - matrix_e_pg # - (Ne,nPg,i,j)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1,1)) - matrix_e_pg)
        res = scalar_e_pg - tensor_e_pg # - (Ne,nPg,i,j,k,l)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1,1,1,1)) - tensor_e_pg)

        # (Ne, nPg, i) - (Ne, nPg, ...)
        try:
            res = vector_e_pg - scalar_e_pg # - (Ne,nPg)
        except ValueError: 
            pass
        res = vector_e_pg - vector_e_pg # - (Ne,nPg,i)
        try:
            res = vector_e_pg - matrix_e_pg # - (Ne,nPg,i,j)
        except ValueError:
            pass
        try:
            res = vector_e_pg - tensor_e_pg # - (Ne,nPg,i,j,k,l)
        except ValueError:
            pass

        # (Ne, nPg, i, j) - (Ne, nPg, ...)
        try:
            res = matrix_e_pg - scalar_e_pg # - (Ne,nPg)
        except ValueError: 
            pass
        try:
            res = matrix_e_pg - vector_e_pg # - (Ne,nPg,i)    
        except ValueError:
            pass
        res = matrix_e_pg - matrix_e_pg # - (Ne,nPg,i,j)
        try:
            res = matrix_e_pg - tensor_e_pg # - (Ne,nPg,i,j,k,l)
        except ValueError:
            pass

        # (Ne, nPg, i, j, k, l) - (Ne, nPg, ...)
        try:
            res = tensor_e_pg - scalar_e_pg # - (Ne,nPg)
        except ValueError: 
            pass
        try:
            res = tensor_e_pg - vector_e_pg # - (Ne,nPg,i)    
        except ValueError:
            pass
        try:
            res = tensor_e_pg - matrix_e_pg # - (Ne,nPg,i,j)
        except ValueError:
            pass
        res = tensor_e_pg - tensor_e_pg # - (Ne,nPg,i,j,k,l)

    def test_mul_array(self, FeArrays: list[FeArray]):
        
        scalar_e_pg, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays

        Ne, nPg = scalar_e_pg.shape[:2]

        scalar = 1
        vector = np.arange(10)
        matrix = np.eye(10)*.1
        tensor = np.random.random((2,2,2,2))

        # (Ne, nPg) * (...)
        res = scalar_e_pg * scalar # * ()
        _check_arrays(res, scalar_e_pg * scalar)
        res = scalar_e_pg * vector # * (i)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1)) * vector)
        res = scalar_e_pg * matrix # * (i,j)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1,1)) * matrix)
        res = scalar_e_pg * tensor # * (i,j,k,l)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1,1,1,1)) * tensor)

        # (Ne, nPg) * (Ne, nPg, ...)
        res = scalar_e_pg * scalar_e_pg # * (Ne,nPg)
        _check_arrays(res, scalar_e_pg * scalar_e_pg)
        res = scalar_e_pg * vector_e_pg # * (Ne,nPg,i)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1)) * vector_e_pg)
        res = scalar_e_pg * matrix_e_pg # * (Ne,nPg,i,j)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1,1)) * matrix_e_pg)
        res = scalar_e_pg * tensor_e_pg # * (Ne,nPg,i,j,k,l)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1,1,1,1)) * tensor_e_pg)

        # (Ne, nPg, i) * (Ne, nPg, ...)
        try:
            res = vector_e_pg * scalar_e_pg # * (Ne,nPg)
        except ValueError: 
            pass
        res = vector_e_pg * vector_e_pg # * (Ne,nPg,i)
        try:
            res = vector_e_pg * matrix_e_pg # * (Ne,nPg,i,j)
        except ValueError:
            pass
        try:
            res = vector_e_pg * tensor_e_pg # * (Ne,nPg,i,j,k,l)
        except ValueError:
            pass

        # (Ne, nPg, i, j) * (Ne, nPg, ...)
        try:
            res = matrix_e_pg * scalar_e_pg # * (Ne,nPg)
        except ValueError: 
            pass
        try:
            res = matrix_e_pg * vector_e_pg # * (Ne,nPg,i)    
        except ValueError:
            pass
        res = matrix_e_pg * matrix_e_pg # * (Ne,nPg,i,j)
        try:
            res = matrix_e_pg * tensor_e_pg # * (Ne,nPg,i,j,k,l)
        except ValueError:
            pass

        # (Ne, nPg, i, j, k, l) * (Ne, nPg, ...)
        try:
            res = tensor_e_pg * scalar_e_pg # * (Ne,nPg)
        except ValueError: 
            pass
        try:
            res = tensor_e_pg * vector_e_pg # * (Ne,nPg,i)    
        except ValueError:
            pass
        try:
            res = tensor_e_pg * matrix_e_pg # * (Ne,nPg,i,j)
        except ValueError:
            pass
        res = tensor_e_pg * tensor_e_pg # * (Ne,nPg,i,j,k,l)

    def test_truediv_array(self, FeArrays: list[FeArray]):
        
        scalar_e_pg, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays

        # make sure there is not 0 values
        vector_e_pg += 1
        matrix_e_pg += 1
        tensor_e_pg += 1

        Ne, nPg = scalar_e_pg.shape[:2]

        scalar = 1
        vector = np.arange(1,11)
        matrix = np.arange(1,101).reshape(10,10)
        tensor = np.random.random((2,2,2,2))

        # (Ne, nPg) / (...)
        res = scalar_e_pg / scalar # / ()
        _check_arrays(res, scalar_e_pg / scalar)
        res = scalar_e_pg / vector # / (i)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1)) / vector)
        res = scalar_e_pg / matrix # / (i,j)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1,1)) / matrix)
        res = scalar_e_pg / tensor # / (i,j,k,l)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1,1,1,1)) / tensor)

        # (Ne, nPg) / (Ne, nPg, ...)
        res = scalar_e_pg / scalar_e_pg # / (Ne,nPg)
        _check_arrays(res, scalar_e_pg / scalar_e_pg)
        res = scalar_e_pg / vector_e_pg # / (Ne,nPg,i)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1)) / vector_e_pg)
        res = scalar_e_pg / matrix_e_pg # / (Ne,nPg,i,j)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1,1)) / matrix_e_pg)
        res = scalar_e_pg / tensor_e_pg # / (Ne,nPg,i,j,k,l)
        _check_arrays(res, np.asarray(scalar_e_pg).reshape((Ne,nPg,1,1,1,1)) / tensor_e_pg)

        # (Ne, nPg, i) / (Ne, nPg, ...)
        try:
            res = vector_e_pg / scalar_e_pg # / (Ne,nPg)
        except ValueError: 
            pass
        res = vector_e_pg / vector_e_pg # / (Ne,nPg,i)
        try:
            res = vector_e_pg / matrix_e_pg # / (Ne,nPg,i,j)
        except ValueError:
            pass
        try:
            res = vector_e_pg / tensor_e_pg # / (Ne,nPg,i,j,k,l)
        except ValueError:
            pass

        # (Ne, nPg, i, j) / (Ne, nPg, ...)
        try:
            res = matrix_e_pg / scalar_e_pg # / (Ne,nPg)
        except ValueError: 
            pass
        try:
            res = matrix_e_pg / vector_e_pg # / (Ne,nPg,i)    
        except ValueError:
            pass
        res = matrix_e_pg / matrix_e_pg # / (Ne,nPg,i,j)
        try:
            res = matrix_e_pg / tensor_e_pg # / (Ne,nPg,i,j,k,l)
        except ValueError:
            pass

        # (Ne, nPg, i, j, k, l) / (Ne, nPg, ...)
        try:
            res = tensor_e_pg / scalar_e_pg # / (Ne,nPg)
        except ValueError: 
            pass
        try:
            res = tensor_e_pg / vector_e_pg # / (Ne,nPg,i)    
        except ValueError:
            pass
        try:
            res = tensor_e_pg / matrix_e_pg # / (Ne,nPg,i,j)
        except ValueError:
            pass
        res = tensor_e_pg / tensor_e_pg # / (Ne,nPg,i,j,k,l)

    def test_T(self, FeArrays: list[FeArray]):
        
        scalar_e_pg, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays

        _check_arrays(scalar_e_pg.T, scalar_e_pg)
        _check_arrays(vector_e_pg.T, vector_e_pg)
        _check_arrays(matrix_e_pg.T, matrix_e_pg.transpose((0,1,3,2)))
        _check_arrays(tensor_e_pg.T, tensor_e_pg.transpose((0,1,5,4,3,2)))
