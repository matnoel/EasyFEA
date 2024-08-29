# Copyright (C) 2021-2024 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

import unittest

from EasyFEA import np
from EasyFEA import Geoms, Mesher, Simulations
# materials
from EasyFEA.Materials import _Elas, Elas_Isot, Elas_IsotTrans, Elas_Anisot,  PhaseField
from EasyFEA.materials import Get_Pmat, Apply_Pmat, KelvinMandel_Matrix

class Test_Materials(unittest.TestCase):
    
    def setUp(self):
        
        elasticMaterials: list[_Elas] = []
        
        for comp in _Elas.Available_Laws():
            if comp == Elas_Isot:
                elasticMaterials.append(
                    Elas_Isot(2, E=210e9, v=0.3, planeStress=True)
                    )
                elasticMaterials.append(
                    Elas_Isot(2, E=210e9, v=0.3, planeStress=False)
                    )
                elasticMaterials.append(
                    Elas_Isot(3, E=210e9, v=0.3)
                    )
            elif comp == Elas_IsotTrans:
                elasticMaterials.append(
                    Elas_IsotTrans(3, El=11580, Et=500, Gl=450, vl=0.02, vt=0.44, axis_l=[1,0,0], axis_t=[0,1,0])
                    )
                elasticMaterials.append(
                    Elas_IsotTrans(3, El=11580, Et=500, Gl=450, vl=0.02, vt=0.44,axis_l=[0,1,0], axis_t=[1,0,0])
                    )
                elasticMaterials.append(
                    Elas_IsotTrans(2, El=11580, Et=500, Gl=450, vl=0.02, vt=0.44, planeStress=True)
                    )
                elasticMaterials.append(
                    Elas_IsotTrans(2, El=11580, Et=500, Gl=450, vl=0.02, vt=0.44, planeStress=False))

            elif comp == Elas_Anisot:
                C_voigt2D = np.array([  [60, 20, 0],
                                        [20, 120, 0],
                                        [0, 0, 30]])

                axis1_1 = np.array([1,0,0])
                axis2_1 = np.array([0,1,0])

                tetha = 30*np.pi/130
                axis1_2 = np.array([np.cos(tetha),np.sin(tetha),0])
                axis2_2 = np.array([-np.sin(tetha),np.cos(tetha),0])

                elasticMaterials.append(
                    Elas_Anisot(2, C_voigt2D, True, axis1_1, axis2_1)
                    )

                elasticMaterials.append(
                    Elas_Anisot(2, C_voigt2D, True, axis1_1, axis2_1)
                )
                elasticMaterials.append(
                    Elas_Anisot(2, C_voigt2D, True, axis1_2, axis2_2)
                    )
                elasticMaterials.append(
                    Elas_Anisot(2, C_voigt2D, True, axis1_2, axis2_2)
                    )
        
        self.splits = PhaseField.Get_splits()
        self.regularizations = PhaseField.Get_regularisations()
        self.phaseFieldModels: list[PhaseField] = []

        splits_Isot = [PhaseField.SplitType.Amor, PhaseField.SplitType.Miehe, PhaseField.SplitType.Stress]

        self.elasticMaterials = elasticMaterials

        for c in elasticMaterials:
            for s in self.splits:
                for r in self.regularizations:
                        
                    if (isinstance(c, Elas_IsotTrans) or isinstance(c, Elas_Anisot)) and s in splits_Isot:
                        continue

                    pfm = PhaseField(c,s,r,1,1)
                    self.phaseFieldModels.append(pfm)

    def test_Elas_Isot(self):

        for mat in self.elasticMaterials:
            self.assertIsInstance(mat, _Elas)
            if isinstance(mat, Elas_Isot):
                E = mat.E
                v = mat.v
                if mat.dim == 2:
                    if mat.planeStress:
                        C_voigt = E/(1-v**2) * np.array([   [1, v, 0],
                                                            [v, 1, 0],
                                                            [0, 0, (1-v)/2]])
                    else:
                        C_voigt = E/((1+v)*(1-2*v)) * np.array([ [1-v, v, 0],
                                                                    [v, 1-v, 0],
                                                                    [0, 0, (1-2*v)/2]])
                else:
                    C_voigt = E/((1+v)*(1-2*v))*np.array([   [1-v, v, v, 0, 0, 0],
                                                                [v, 1-v, v, 0, 0, 0],
                                                                [v, v, 1-v, 0, 0, 0],
                                                                [0, 0, 0, (1-2*v)/2, 0, 0],
                                                                [0, 0, 0, 0, (1-2*v)/2, 0],
                                                                [0, 0, 0, 0, 0, (1-2*v)/2]  ])
                
                c = KelvinMandel_Matrix(mat.dim, C_voigt)
                    
                test_C = np.linalg.norm(c-mat.C)/np.linalg.norm(c)
                self.assertTrue(test_C < 1e-12, f"test_C = {test_C:.3e}")

    def test_Elas_Anisot(self):

        C_voigt2D = np.array([  [60, 20, 0],
                                [20, 120, 0],
                                [0, 0, 30]])
        
        C_voigt3D = np.array([  [60, 20, 10, 0, 0, 0],
                                [20, 120, 80, 0, 0, 0],
                                [10, 80, 300, 0, 0, 0],
                                [0, 0, 0, 400, 0, 0],
                                [0, 0, 0, 0, 500, 0],
                                [0, 0, 0, 0, 0, 600]])

        axis1_1 = np.array([1,0,0])
        axis2_1 = np.array([0,1,0])

        a = 30*np.pi/130
        axis1_2 = np.array([np.cos(a),np.sin(a),0])
        axis2_2 = np.array([-np.sin(a),np.cos(a),0])

        mat_2D_1 = Elas_Anisot(2, C_voigt2D, True, axis1_1, axis2_1)
        
        mat_2D_2 = Elas_Anisot(2, C_voigt2D, True, axis1_2, axis2_2)        

        mat_2D_3 = Elas_Anisot(2, C_voigt2D, True)        
        
        mat_3D_1 = Elas_Anisot(3, C_voigt3D, True, axis1_1, axis2_1)
        mat_3D_2 = Elas_Anisot(3, C_voigt3D, True, axis1_2, axis2_2)

        listComp = [mat_2D_1, mat_2D_2, mat_2D_3, mat_3D_1, mat_3D_2]

        for comp in listComp: 
            matC = comp.C
            test_Symetry = np.linalg.norm(matC.T - matC)
            assert test_Symetry <= 1e-12
    
    def test_Elas_IsotTrans(self):

        El=11580
        Et=500
        Gl=450
        vl=0.02
        vt=0.44

        # material_cM = np.array([[El+4*vl**2*kt, 2*kt*vl, 2*kt*vl, 0, 0, 0],
        #               [2*kt*vl, kt+Gt, kt-Gt, 0, 0, 0],
        #               [2*kt*vl, kt-Gt, kt+Gt, 0, 0, 0],
        #               [0, 0, 0, 2*Gt, 0, 0],
        #               [0, 0, 0, 0, 2*Gl, 0],
        #               [0, 0, 0, 0, 0, 2*Gl]])

        # axis_l = [1, 0, 0] et axis_t = [0, 1, 0]
        mat1 = Elas_IsotTrans(2,
                    El=11580, Et=500, Gl=450, vl=0.02, vt=0.44,
                    planeStress=False,
                    axis_l=np.array([1,0,0]), axis_t=np.array([0,1,0]))

        Gt = mat1.Gt
        kt = mat1.kt

        c1 = np.array([[El+4*vl**2*kt, 2*kt*vl, 0],
                      [2*kt*vl, kt+Gt, 0],
                      [0, 0, 2*Gl]])

        test_c1 = np.linalg.norm(c1 - mat1.C)/np.linalg.norm(c1)
        self.assertTrue(test_c1 < 1e-12, f"test_c1 = {test_c1:.3e}")

        # axis_l = [0, 1, 0] et axis_t = [1, 0, 0]
        mat2 = Elas_IsotTrans(2,
                    El=11580, Et=500, Gl=450, vl=0.02, vt=0.44,
                    planeStress=False,
                    axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]))

        c2 = np.array([[kt+Gt, 2*kt*vl, 0],
                      [2*kt*vl, El+4*vl**2*kt, 0],
                      [0, 0, 2*Gl]])

        test_c2 = np.linalg.norm(c2 - mat2.C)/np.linalg.norm(c2)
        self.assertTrue(test_c2 < 1e-12, f"test_c2 = {test_c2:.3e}")

        # axis_l = [0, 0, 1] et axis_t = [1, 0, 0]
        mat = Elas_IsotTrans(2,
                    El=11580, Et=500, Gl=450, vl=0.02, vt=0.44,
                    planeStress=False,
                    axis_l=[0,0,1], axis_t=[1,0,0])

        c3 = np.array([[kt+Gt, kt-Gt, 0],
                      [kt-Gt, kt+Gt, 0],
                      [0, 0, 2*Gt]])

        test_c3 = np.linalg.norm(c3 - mat.C)/np.linalg.norm(c3)
        self.assertTrue(test_c3 < 1e-12, f"test_c3 = {test_c3:.3e}")

    def test_getPmat(self):

        Ne = 10
        p = 3

        _ = 1
        _e = np.ones((Ne))
        _e_pg = np.ones((Ne,p))
        _e2 = np.linspace(1, 1.001, Ne)        

        
        El = 15716.16722094732 
        Et = 232.6981580878141
        Gl = 557.3231495541391
        vl = 0.02
        vt = 0.44        

        for dim in [2, 3]:

            axis1 = np.array([1,0,0])[:dim]
            axis2 = np.array([0,1,0])[:dim]

            angles = np.linspace(0, np.pi, Ne)
            x1, y1 = np.cos(angles) , np.sin(angles)
            x2, y2 = -np.sin(angles) , np.cos(angles)
            axis1_e = np.zeros((Ne,dim)); axis1_e[:,0] = x1; axis1_e[:,1] = y1        
            axis2_e = np.zeros((Ne,dim)); axis2_e[:,0] = x2; axis2_e[:,1] = y2

            axis1_e_p = axis1_e[:,np.newaxis].repeat(p, 1)
            axis2_e_p = axis2_e[:,np.newaxis].repeat(p, 1)

            for c in [_, _e, _e_pg, _e2]:

                mat = Elas_IsotTrans(dim, El*c, Et*c, Gl*c, vl*c, vt*c)
                C = mat.C
                S = mat.S

                for ax1, ax2 in [(axis1, axis2),(axis1_e, axis2_e),(axis1_e_p, axis2_e_p)]:                
                    Pmat = Get_Pmat(ax1, ax2)
                    
                    # checks mat to global coord
                    Cglob = Apply_Pmat(Pmat, C)
                    Sglob = Apply_Pmat(Pmat, S)    
                    self.__check_invariants(Cglob, C)
                    self.__check_invariants(Sglob, S)

                    # checks global to mat coord
                    Cmat = Apply_Pmat(Pmat, Cglob, toGlobal=False)
                    Smat = Apply_Pmat(Pmat, Sglob, toGlobal=False)
                    self.__check_invariants(Cmat, C, True)
                    self.__check_invariants(Smat, S, True)

                    # checks Ps, Pe
                    Ps, Pe = Get_Pmat(ax1, ax2, False)
                    transp = np.arange(Ps.ndim)
                    transp[-1], transp[-2] = transp[-2], transp[-1]
                    # checks inv(Ps) = Pe'
                    testPs = np.linalg.norm(np.linalg.inv(Ps) - Pe.transpose(transp))/np.linalg.norm(Pe.transpose(transp))
                    assert testPs <= 1e-12, f"inv(Ps) != Pe' -> {testPs:.3e}"
                    # checks inv(Pe) = Ps'
                    testPe = np.linalg.norm(np.linalg.inv(Pe) - Ps.transpose(transp))/np.linalg.norm(Ps.transpose(transp))
                    assert testPe <= 1e-12, f"inv(Pe) = Ps' -> {testPe:.3e}"

    def __check_invariants(self, mat1: np.ndarray, mat2: np.ndarray, checkSame=False):
        
        tol = 1e-12

        shape1, dim1 = mat1.shape, mat1.ndim
        shape2, dim2 = mat2.shape, mat2.ndim
        
        if dim1 > dim2:
            pass
            if dim2==3:
                mat2 = mat2[:,np.newaxis].repeat(shape1[1], 1)
            elif dim2==4:                
                mat2 = mat2[np.newaxis,np.newaxis].repeat(shape1[0],0)
                mat2 = mat2.repeat(shape1[1],1)
        elif dim2 > dim1:
            pass
            if dim1==3:
                mat1 = mat1[:,np.newaxis].repeat(shape2[1], 1)
            elif dim1==4:                
                mat1 = mat1[np.newaxis,np.newaxis].repeat(shape2[0],0)
                mat1 = mat1.repeat(shape2[1],1)
        
        tr1 = np.trace(mat1, axis1=-2, axis2=-1)
        tr2 = np.trace(mat2, axis1=-2, axis2=-1)
        trErr = (tr1 - tr2)/tr2
        test_trace = np.linalg.norm(trErr)
        assert test_trace <= tol, f"The trace is not preserved during the process (test_trace = {test_trace:.3e})"
        
        det1 = np.linalg.det(mat1)
        det2 = np.linalg.det(mat2)
        detErr = (det1 - det2)/det2
        test_det = np.linalg.norm(detErr)
        assert test_det <= tol, f"The determinant is not preserved during the process (test_det = {test_det:.3e})"

        if checkSame:
            matErr = (mat1 - mat2)
            test_mat = np.linalg.norm(matErr)/np.linalg.norm(mat2)
            assert test_mat <= tol, "mat1 != mat2"
    
    def __cal_eps(self, dim) -> np.ndarray:

        mat = Elas_Isot(dim)

        L = 2
        H = 1
        domain = Geoms.Domain(Geoms.Point(), Geoms.Point(L, H), L/20)
        circle = Geoms.Circle(Geoms.Point(L/2, H/2), H/4, L/20)
        
        if dim == 2:
            mesh = Mesher().Mesh_2D(domain, [circle])
        else:            
            mesh = Mesher().Mesh_Extrude(domain, [circle], [0,0,H/3])

        simu = Simulations.ElasticSimu(mesh, mat)
        simu.add_dirichlet(mesh.Nodes_Conditions(lambda x,y,z: x==0), [0]*simu.Get_dof_n(), simu.Get_dofs())
        simu.add_dirichlet(mesh.Nodes_Conditions(lambda x,y,z: x==L), [L*1e-5], ["x"])
        u = simu.Solve()

        Epsilon_e_pg = simu._Calc_Epsilon_e_pg(u)

        return Epsilon_e_pg

    def test_split_phaseField(self):
        
        print()

        Ne = 1000
        nPg = 3

        # Creates 2D random strain field
        Epsilon2D_e_pg = self.__cal_eps(2)
        # Epsilon2D_e_pg = np.random.rand(Ne,nPg,3) * 1e-3

        # Creates 3D random strain field
        Epsilon3D_e_pg = self.__cal_eps(3)        
        # Epsilon3D_e_pg = np.random.rand(Ne,nPg,6) * 1e-3

        for p, pfm in enumerate(self.phaseFieldModels):            

            mat: _Elas = pfm.material
            c = mat.C

            tol = 1e-12 if mat.dim == 2 else 1e-12
            
            print(f"{type(mat).__name__} {mat.simplification} {pfm.split} {pfm.regularization}")

            if mat.dim == 2:
                Epsilon_e_pg = Epsilon2D_e_pg
            elif mat.dim == 3:
                Epsilon_e_pg = Epsilon3D_e_pg

            cP_e_pg, cM_e_pg = pfm.Calc_C(Epsilon_e_pg.copy(), verif=True)

            # Checks that cP + cM = c
            cpm = cP_e_pg + cM_e_pg
            decomp_C = c - cpm
            test_C = np.max(np.linalg.norm(decomp_C, axis=(-2,-1))/np.linalg.norm(c, axis=(-2,-1)))
            if test_C >= tol:
                pass
            self.assertTrue(test_C <= tol, f"test_C = {test_C:.3e}")

            # Checks that SigP + SigM = Sig
            Sig_e_pg = np.einsum('ij,epj->epi', c, Epsilon_e_pg, optimize='optimal')
            SigP = np.einsum('epij,epj->epi', cP_e_pg, Epsilon_e_pg, optimize='optimal')
            SigM = np.einsum('epij,epj->epi', cM_e_pg, Epsilon_e_pg, optimize='optimal') 
            decomp_Sig = Sig_e_pg - (SigP+SigM)           
            test_Sig = np.max(np.linalg.norm(decomp_Sig, axis=-1)/np.linalg.norm(Sig_e_pg, axis=-1))
            if np.linalg.norm(Sig_e_pg)>0:                
                self.assertTrue(test_Sig <= tol, f"test_Sig = {test_Sig:.3e}")
                
            # Checks that Eps:C:Eps = Eps:(cP+cM):Eps
            psi = 1/2 * np.einsum('epi,ij,epj->', Epsilon_e_pg, c, Epsilon_e_pg, optimize='optimal')
            psi_P = 1/2 * np.einsum('epi,epij,epj->', Epsilon_e_pg, cP_e_pg, Epsilon_e_pg, optimize='optimal')
            psi_M = 1/2 * np.einsum('epi,epij,epj->', Epsilon_e_pg, cM_e_pg, Epsilon_e_pg, optimize='optimal')
            test_psi = np.abs(psi-(psi_P+psi_M))/psi
            if np.linalg.norm(psi)>0:
                self.assertTrue(np.max(test_psi) <= tol, f"test_psi = {test_psi:.3e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)