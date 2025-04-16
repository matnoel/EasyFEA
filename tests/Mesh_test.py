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

def _check_arrays(array1, array2):
    if isinstance(array1, FeArray):
        array1 = np.asarray(array1)
    if isinstance(array2, FeArray):
        array2 = np.asarray(array2)
    norm_diff = np.linalg.norm(array1 - array2)
    assert norm_diff < 1e-12

def _check_ValueError(func):
    valueErrorDetected = False
    try:
        func()
    except ValueError:
        valueErrorDetected = True
    assert valueErrorDetected

def _check_operation(array1, op, array2, reshape1=(), reshape2=(), willTriggerError=False):

    if reshape1 == ():
        reshape1 = np.shape(array1)
    reshaped1 = np.asarray(array1).reshape(reshape1)

    if reshape2 == ():
        reshape2 = np.shape(array2)
    reshaped2 = np.asarray(array2).reshape(reshape2)

    try:
        if op == "+":
            computed = array1 + array2
            excepted = reshaped1 + reshaped2
        elif op == "-":
            computed = array1 - array2
            excepted = reshaped1 - reshaped2
        elif op == "*":
            computed = array1 * array2
            excepted = reshaped1 * reshaped2 
        elif op == "/":
            computed = array1 / array2
            excepted = reshaped1 / reshaped2 
        else:
            raise Exception("unknown operator")
        _check_arrays(computed, excepted)
    except ValueError:
        if willTriggerError:
            pass
        else:
            # should not be here
            raise ValueError("ValueError detected")

def FeArrays():

    Ne, nPg = 1000, 4
    scalar_e_pg = FeArray(np.random.random_integers(1, 10, (Ne, nPg))*.1)
    vector_e_pg = FeArray(np.random.random_integers(1, 10, (Ne, nPg, 3))*.1)
    matrix_e_pg = FeArray(np.random.random_integers(1, 10, (Ne, nPg, 3, 3))*.1)
    tensor_e_pg = FeArray(np.random.random_integers(1, 10, (Ne, nPg, 3, 3, 3, 3))*.1)

    return [scalar_e_pg, vector_e_pg, matrix_e_pg, tensor_e_pg]

def do_operation(op: str):

    scalar_e_pg, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays()

    Ne, nPg = scalar_e_pg.shape

    scalar = np.random.random_integers(1, 10)
    vector = np.random.random_integers(1, 10, (3))*.1
    matrix = np.random.random_integers(1, 10, (3, 3))*.1
    tensor = np.random.random_integers(1, 10, (3, 3, 3, 3))*.1

    # scalar + ...
    _check_operation(scalar_e_pg, op, scalar)
    _check_operation(scalar_e_pg, op, vector, (Ne,nPg,1))
    _check_operation(scalar_e_pg, op, matrix, (Ne,nPg,1,1))
    _check_operation(scalar_e_pg, op, tensor, (Ne,nPg,1,1,1,1))
    
    # ... + scalar 
    _check_operation(scalar, op, scalar_e_pg)
    _check_operation(scalar, op, vector_e_pg)
    _check_operation(scalar, op, matrix_e_pg)
    _check_operation(scalar, op, tensor_e_pg)

    # vector + ...
    _check_operation(vector_e_pg, op, scalar) 
    _check_operation(vector_e_pg, op, vector) 
    _check_operation(vector_e_pg, op, matrix, willTriggerError=True)
    _check_operation(vector_e_pg, op, tensor, willTriggerError=True)

    # ... + vector
    _check_operation(scalar, op, vector_e_pg) 
    _check_operation(vector, op, vector_e_pg) 
    _check_operation(matrix, op, vector_e_pg, willTriggerError=True)
    _check_operation(tensor, op, vector_e_pg, willTriggerError=True)
                
    # matrix + ...
    _check_operation(matrix_e_pg, op, scalar) 
    _check_operation(matrix_e_pg, op, vector, willTriggerError=True) 
    _check_operation(matrix_e_pg, op, matrix)
    _check_operation(matrix_e_pg, op, tensor, willTriggerError=True)

    # ... + matrix
    _check_operation(scalar, op, matrix_e_pg)
    _check_operation(vector, op, matrix_e_pg, willTriggerError=True) 
    _check_operation(matrix, op, matrix_e_pg)
    _check_operation(tensor, op, matrix_e_pg, willTriggerError=True)

    # tensor + ...
    _check_operation(tensor_e_pg, op, scalar) 
    _check_operation(tensor_e_pg, op, vector, willTriggerError=True) 
    _check_operation(tensor_e_pg, op, matrix, willTriggerError=True)
    _check_operation(tensor_e_pg, op, tensor)

    # ... + matrix
    _check_operation(scalar, op, tensor_e_pg)
    _check_operation(vector, op, tensor_e_pg, willTriggerError=True) 
    _check_operation(matrix, op, tensor_e_pg, willTriggerError=True)
    _check_operation(tensor, op, tensor_e_pg)
    

class TestFeArray:

    def test_new_array(self):

        try:
            FeArray([0,1])
        except ValueError:
            pass
        
        scalar_e_pg, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays()

        assert scalar_e_pg._idx == ""
        assert vector_e_pg._idx == "i"
        assert matrix_e_pg._idx == "ij"
        assert tensor_e_pg._idx == "ijkl"

        assert scalar_e_pg._type == "scalar"
        assert vector_e_pg._type == "vector"
        assert matrix_e_pg._type == "matrix"
        assert tensor_e_pg._type == "tensor"

        _check_ValueError(lambda: FeArray(0, addFeAxis=False))
        _check_ValueError(lambda: FeArray([0], addFeAxis=False))

    def test_add_array(self):

        do_operation("+")

    def test_sub_array(self):

        do_operation("-")

    def test_mul_array(self):

        do_operation("*")
    
    def test_trudiv_array(self):

        do_operation("/")

    def test_T(self):
        
        scalar_e_pg, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays()

        _check_arrays(scalar_e_pg.T, scalar_e_pg)
        _check_arrays(vector_e_pg.T, vector_e_pg)
        _check_arrays(matrix_e_pg.T, matrix_e_pg.transpose((0,1,3,2)))
        _check_arrays(tensor_e_pg.T, tensor_e_pg.transpose((0,1,5,4,3,2)))
    
    def test_dot(self):
        
        _, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays()

        # Avoid testing scalars, as this holds no significance.
        
        # i i
        _check_arrays(vector_e_pg.dot(vector_e_pg), np.einsum("...i,...i->...", vector_e_pg, vector_e_pg))
        # i ij 
        _check_arrays(vector_e_pg.dot(matrix_e_pg), np.einsum("...i,...ij->...j", vector_e_pg, matrix_e_pg))
        # i ijkl
        _check_ValueError(lambda: vector_e_pg.dot(tensor_e_pg)) # because jkl not in [0,1,2,4]

        # ij j
        _check_arrays(matrix_e_pg.T.dot(vector_e_pg), np.einsum("...ji,...j->...i", matrix_e_pg, vector_e_pg))
        # ij jk
        _check_arrays(matrix_e_pg.T.dot(matrix_e_pg), np.einsum("...ji,...jk->...ik", matrix_e_pg, matrix_e_pg))
        # ij jklm
        _check_arrays(matrix_e_pg.T.dot(tensor_e_pg), np.einsum("...ji,...jklm->...iklm", matrix_e_pg, tensor_e_pg))

        # ijkl l
        _check_ValueError(lambda: tensor_e_pg.dot(vector_e_pg)) # because ijk not in [0,1,2,4]
        # ijkl lm
        _check_arrays(tensor_e_pg.dot(matrix_e_pg), np.einsum("...ijkl,...lm->...ijkm", tensor_e_pg, matrix_e_pg))
        # ijkl lmno
        _check_ValueError(lambda: tensor_e_pg.dot(tensor_e_pg)) # because ijkmno not in [0,1,2,4]

    def test_matmul(self):
        
        _, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays()

        # Avoid testing scalars, as this holds no significance.
        
        # i i
        _check_arrays(vector_e_pg @ vector_e_pg, np.einsum("...i,...i->...", vector_e_pg, vector_e_pg))
        # i ij 
        _check_arrays(vector_e_pg @ matrix_e_pg, np.einsum("...i,...ij->...j", vector_e_pg, matrix_e_pg))
        # i ijkl
        _check_ValueError(lambda: vector_e_pg @ tensor_e_pg) # because jkl not in [0,1,2,4]

        # ij j
        _check_arrays(matrix_e_pg.T @ vector_e_pg, np.einsum("...ji,...j->...i", matrix_e_pg, vector_e_pg))
        # ij jk
        _check_arrays(matrix_e_pg.T @ matrix_e_pg, np.einsum("...ji,...jk->...ik", matrix_e_pg, matrix_e_pg))
        # ij jklm
        _check_arrays(matrix_e_pg.T @ tensor_e_pg, np.einsum("...ji,...jklm->...iklm", matrix_e_pg, tensor_e_pg))

        # ijkl l
        _check_ValueError(lambda: tensor_e_pg @ vector_e_pg) # because ijk not in [0,1,2,4]
        # ijkl lm
        _check_arrays(tensor_e_pg @ matrix_e_pg, np.einsum("...ijkl,...lm->...ijkm", tensor_e_pg, matrix_e_pg))
        # ijkl lmno
        _check_ValueError(lambda: tensor_e_pg @ tensor_e_pg) # because ijkmno not in [0,1,2,4]

    
    def test_ddot(self):
        
        _, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays()

        # Avoid testing scalars, as this holds no significance.
        
        # i i
        _check_ValueError(lambda: vector_e_pg.ddot(vector_e_pg)) # wrong dimensions
        # i ij 
        _check_ValueError(lambda: vector_e_pg.ddot(matrix_e_pg)) # wrong dimensions
        # i ijkl
        _check_ValueError(lambda: vector_e_pg.ddot(tensor_e_pg)) # wrong dimensions

        # ij i
        _check_ValueError(lambda: matrix_e_pg.ddot(vector_e_pg))
        # ij ij
        _check_arrays(matrix_e_pg.ddot(matrix_e_pg), np.einsum("...ij,...ij->...", matrix_e_pg, matrix_e_pg))
        # ij ijkl
        _check_arrays(matrix_e_pg.ddot(tensor_e_pg), np.einsum("...ij,...ijkl->...kl", matrix_e_pg, tensor_e_pg))        

        # ijkl l
        _check_ValueError(lambda: tensor_e_pg.ddot(vector_e_pg)) # wrong dimensions
        # ijkl kl
        _check_arrays(tensor_e_pg.ddot(matrix_e_pg), np.einsum("...ijkl,...kl->...ij", tensor_e_pg, matrix_e_pg))        
        # ijkl lmno
        _check_arrays(tensor_e_pg.ddot(tensor_e_pg), np.einsum("...ijkl,...klmn->...ijmn", tensor_e_pg, tensor_e_pg))