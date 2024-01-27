# %%
import unittest
import os
from Materials import Elas_Anisot, Elas_IsotTrans, PhaseField_Model, _Displacement_Model, Elas_Isot
import Materials
import numpy as np

class Test_Materials(unittest.TestCase):
    def setUp(self):

        # Isotropic Elastic Behavior
        E = 210e9
        v = 0.3
        self.comportements2D = []
        self.comportements3D = []
        for comp in _Displacement_Model.get_behaviorLaws():
            if comp == Elas_Isot:
                self.comportements2D.append(
                    Elas_Isot(2, E=E, v=v, planeStress=True)
                    )
                self.comportements2D.append(
                    Elas_Isot(2, E=E, v=v, planeStress=False)
                    )
                self.comportements3D.append(
                    Elas_Isot(3, E=E, v=v)
                    )
            elif comp == Elas_IsotTrans:
                self.comportements3D.append(
                    Elas_IsotTrans(3, El=11580, Et=500, Gl=450, vl=0.02, vt=0.44, axis_l=[1,0,0], axis_t=[0,1,0])
                    )
                self.comportements3D.append(
                    Elas_IsotTrans(3, El=11580, Et=500, Gl=450, vl=0.02, vt=0.44,axis_l=[0,1,0], axis_t=[1,0,0])
                    )
                self.comportements2D.append(
                    Elas_IsotTrans(2, El=11580, Et=500, Gl=450, vl=0.02, vt=0.44, planeStress=True)
                    )
                self.comportements2D.append(
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

                self.comportements2D.append(
                    Elas_Anisot(2, C_voigt2D, True, axis1_1, axis2_1, planeStress=True)
                    )

                self.comportements2D.append(
                    Elas_Anisot(2, C_voigt2D, True, axis1_1, axis2_1, planeStress=False)
                )
                self.comportements2D.append(
                    Elas_Anisot(2, C_voigt2D, True, axis1_2, axis2_2, planeStress=True)
                    )
                self.comportements2D.append(
                    Elas_Anisot(2, C_voigt2D, True, axis1_2, axis2_2, planeStress=False)
                    )
        
        # phasefieldModel
        self.splits = PhaseField_Model.get_splits()
        self.regularizations = PhaseField_Model.get_regularisations()
        self.phaseFieldModels = []

        splits_Isot = [PhaseField_Model.SplitType.Amor, PhaseField_Model.SplitType.Miehe, PhaseField_Model.SplitType.Stress]

        comportements = self.comportements2D
        comportements.extend(self.comportements3D)

        for c in comportements:
            for s in self.splits:
                for r in self.regularizations:
                        
                    if (isinstance(c, Elas_IsotTrans) or isinstance(c, Elas_Anisot)) and s in splits_Isot:
                        continue

                    pfm = PhaseField_Model(c,s,r,1,1)
                    self.phaseFieldModels.append(pfm)
            

    def test_Elas_Isot(self):

        for comp in self.comportements3D:
            self.assertIsInstance(comp, _Displacement_Model)
            if isinstance(comp, Elas_Isot):
                E = comp.E
                v = comp.v
                if comp.dim == 2:
                    if comp.planeStress:
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
                
                c = Materials.KelvinMandel_Matrix(comp.dim, C_voigt)
                    
                verifC = np.linalg.norm(c-comp.C)/np.linalg.norm(c)
                self.assertTrue(verifC < 1e-12)

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

        tetha = 30*np.pi/130
        axis1_2 = np.array([np.cos(tetha),np.sin(tetha),0])
        axis2_2 = np.array([-np.sin(tetha),np.cos(tetha),0])

        comportement2D_CP_1 = Elas_Anisot(2, C_voigt2D, True, axis1_1, axis2_1, planeStress=True)
        comportement2D_DP_1 = Elas_Anisot(2, C_voigt2D, True, axis1_1, axis2_1, planeStress=False)
        
        comportement2D_CP_2 = Elas_Anisot(2, C_voigt2D, True, axis1_2, axis2_2, planeStress=True)
        comportement2D_DP_2 = Elas_Anisot(2, C_voigt2D, True, axis1_2, axis2_2, planeStress=False)

        comportement2D_CP_3 = Elas_Anisot(2, C_voigt2D, True, planeStress=True)
        comportement2D_DP_3 = Elas_Anisot(2, C_voigt2D, True, planeStress=False)
        
        comportement3D_1 = Elas_Anisot(3, C_voigt3D, True, axis1_1, axis2_1)
        comportement3D_2 = Elas_Anisot(3, C_voigt3D, True, axis1_2, axis2_2)

        listComp = [comportement2D_CP_1, comportement2D_DP_1, comportement2D_CP_2, comportement2D_DP_2, comportement3D_1, comportement3D_2]

        for comp in listComp: 
            matC = comp.C
            testSymetry = np.linalg.norm(matC.T - matC)
            assert testSymetry <= 1e-12
    
    def test_ElasIsotTrans(self):
        # Here we check that when we change the axes it works well

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

        # Verif1 axis_l = [1, 0, 0] et axis_t = [0, 1, 0]
        compElasIsotTrans1 = Elas_IsotTrans(2,
                    El=11580, Et=500, Gl=450, vl=0.02, vt=0.44,
                    planeStress=False,
                    axis_l=np.array([1,0,0]), axis_t=np.array([0,1,0]))

        Gt = compElasIsotTrans1.Gt
        kt = compElasIsotTrans1.kt

        c1 = np.array([[El+4*vl**2*kt, 2*kt*vl, 0],
                      [2*kt*vl, kt+Gt, 0],
                      [0, 0, 2*Gl]])

        verifc1 = np.linalg.norm(c1 - compElasIsotTrans1.C)/np.linalg.norm(c1)
        self.assertTrue(verifc1 < 1e-12)

        # Verif2 axis_l = [0, 1, 0] et axis_t = [1, 0, 0]
        compElasIsotTrans2 = Elas_IsotTrans(2,
                    El=11580, Et=500, Gl=450, vl=0.02, vt=0.44,
                    planeStress=False,
                    axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]))

        c2 = np.array([[kt+Gt, 2*kt*vl, 0],
                      [2*kt*vl, El+4*vl**2*kt, 0],
                      [0, 0, 2*Gl]])

        verifc2 = np.linalg.norm(c2 - compElasIsotTrans2.C)/np.linalg.norm(c2)
        self.assertTrue(verifc2 < 1e-12)

        # # Verif3 axis_l = [0, 0, 1] et axis_t = [1, 0, 0]
        # compElasIsotTrans3 = Elas_IsotTrans(2,
        #             El=11580, Et=500, Gl=450, vl=0.02, vt=0.44,
        #             planeStress=False,
        #             axis_l=[0,0,1], axis_t=[1,0,0])
        
        compElasIsotTrans3 = Elas_IsotTrans(2,
                    El=11580, Et=500, Gl=450, vl=0.02, vt=0.44,
                    planeStress=False,
                    axis_l=[0,0,1], axis_t=[1,0,0])

        c3 = np.array([[kt+Gt, kt-Gt, 0],
                      [kt-Gt, kt+Gt, 0],
                      [0, 0, 2*Gt]])

        verifc3 = np.linalg.norm(c3 - compElasIsotTrans3.C)/np.linalg.norm(c3)
        self.assertTrue(verifc3 < 1e-12)

    def test_getPmat(self):
        """Test the way we construct Pmat and apply it"""

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
                    Pmat = Materials.Get_Pmat(ax1, ax2)
                    
                    # test mat to global coordo
                    Cglob = Materials.Apply_Pmat(Pmat, C)
                    Sglob = Materials.Apply_Pmat(Pmat, S)    
                    self.__checkInvariants(Cglob, C)
                    self.__checkInvariants(Sglob, S)

                    # test global to mat coordo
                    Cmat = Materials.Apply_Pmat(Pmat, Cglob, toGlobal=False)
                    Smat = Materials.Apply_Pmat(Pmat, Sglob, toGlobal=False)
                    self.__checkInvariants(Cmat, C, True)
                    self.__checkInvariants(Smat, S, True)

                    # test Ps, Pe
                    Ps, Pe = Materials.Get_Pmat(ax1, ax2, False)
                    transp = np.arange(Ps.ndim)
                    transp[-1], transp[-2] = transp[-2], transp[-1]
                    # inv(Ps) = Pe'
                    testPs = np.linalg.norm(np.linalg.inv(Ps) - Pe.transpose(transp))/np.linalg.norm(Pe.transpose(transp))
                    assert testPs <= 1e-12, "inv(Ps) != Pe'"
                    # inv(Pe) = Ps'
                    testPe = np.linalg.norm(np.linalg.inv(Pe) - Ps.transpose(transp))/np.linalg.norm(Ps.transpose(transp))
                    assert testPe <= 1e-12, "inv(Pe) = Ps'"


    def __checkInvariants(self, mat1: np.ndarray, mat2: np.ndarray, checkSame=False):
        # We check that the tensor invariants do not change!
        # first we need to reshape Matrix

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
        assert test_trace <= 1e-12, "The trace is not preserved during the process"            
        
        det1 = np.linalg.det(mat1)
        det2 = np.linalg.det(mat2)
        detErr = (det1 - det2)/det2
        test_det = np.linalg.norm(detErr)
        assert test_det <= 1e-12, "The determinant is not preserved during the process"

        if checkSame:
            matErr = (mat1 - mat2)
            # test_mat = np.linalg.norm(matErr)
            test_mat = np.linalg.norm((mat1 - mat2))/np.linalg.norm(mat2)
            assert test_mat <= 1e-12, "mat1 != mat2"
    
    def test_split_phaseField(self):
        """Function that allows you to test all energy decomposition models"""
        print()
        Ne = 50
        nPg = 2

        np.random.seed(3)

        # Creation of any 2 2D spilons
        Epsilon2D_e_pg = np.random.randn(Ne,nPg,3)

        # Creation of any 2 3D spilons
        Epsilon3D_e_pg = np.random.randn(Ne,nPg,6)
                
        tol = 1e-11

        for pfm in self.phaseFieldModels:
            
            assert isinstance(pfm, PhaseField_Model)

            comportement = pfm.material
            
            if isinstance(comportement, _Displacement_Model):
                c = comportement.C
            
            print(f"{type(comportement).__name__} {comportement.simplification} {pfm.split} {pfm.regularization}")

            if comportement.dim == 2:
                Epsilon_e_pg = Epsilon2D_e_pg
            elif comportement.dim == 3:
                Epsilon_e_pg = Epsilon3D_e_pg

            cP_e_pg, cM_e_pg = pfm.Calc_C(Epsilon_e_pg.copy(), verif=True)

            # Test that cP + cM = c
            cpm = cP_e_pg+cM_e_pg
            decompC = c-cpm
            verifC = np.linalg.norm(decompC)/np.linalg.norm(c)
            if pfm.split != "He":
                self.assertTrue(np.abs(verifC) <= tol)

            # Test that SigP + SigM = Sig
            Sig_e_pg = np.einsum('ij,epj->epi', c, Epsilon_e_pg, optimize='optimal')
            
            SigP = np.einsum('epij,epj->epi', cP_e_pg, Epsilon_e_pg, optimize='optimal')
            SigM = np.einsum('epij,epj->epi', cM_e_pg, Epsilon_e_pg, optimize='optimal') 
            decompSig = Sig_e_pg-(SigP+SigM)           
            verifSig = np.linalg.norm(decompSig)/np.linalg.norm(Sig_e_pg)
            if np.linalg.norm(Sig_e_pg)>0:                
                self.assertTrue(np.abs(verifSig) <= tol)
            
            
            # Test that Eps:C:Eps = Eps:(cP+cM):Eps
            energiec = np.einsum('epj,ij,epi->ep', Epsilon_e_pg, c, Epsilon_e_pg, optimize='optimal')
            energiecP = np.einsum('epj,epij,epi->ep', Epsilon_e_pg, cP_e_pg, Epsilon_e_pg, optimize='optimal')
            energiecM = np.einsum('epj,epij,epi->ep', Epsilon_e_pg, cM_e_pg, Epsilon_e_pg, optimize='optimal')
            verifEnergie = np.linalg.norm(energiec-(energiecP+energiecM))/np.linalg.norm(energiec)
            if np.linalg.norm(energiec)>0:
                self.assertTrue(np.abs(verifEnergie) <= tol)



if __name__ == '__main__':
    try:
        import Display
        Display.Clear()
        unittest.main(verbosity=2)
    except:
        print("")