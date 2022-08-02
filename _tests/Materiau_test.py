# %%

from posixpath import split
import unittest
import os
from Materiau import Elas_IsotTrans, PhaseFieldModel, LoiDeComportement, Elas_Isot
import numpy as np

class Test_Materiau(unittest.TestCase):
    def setUp(self):

        # Comportement Elatique Isotrope
        E = 210e9
        v = 0.3
        self.comportements2D = []
        self.comportements3D = []
        for comp in LoiDeComportement.get_LoisDeComportement():
            if comp == Elas_Isot:
                self.comportements2D.append(
                    Elas_Isot(2, E=E, v=v, contraintesPlanes=True)
                    )
                self.comportements2D.append(
                    Elas_Isot(2, E=E, v=v, contraintesPlanes=False)
                    )
                self.comportements3D.append(
                    Elas_Isot(3, E=E, v=v)
                    )
            elif comp == Elas_IsotTrans:
                self.comportements3D.append(
                    Elas_IsotTrans(3,
                    El=11580, Et=500, Gl=450, vl=0.02, vt=0.44,
                    axis_l=[1,0,0], axis_t=[0,1,0])
                    )
                self.comportements3D.append(
                    Elas_IsotTrans(3,
                    El=11580, Et=500, Gl=450, vl=0.02, vt=0.44,
                    axis_l=[0,1,0], axis_t=[1,0,0])
                    )
                self.comportements2D.append(
                    Elas_IsotTrans(2,
                    El=11580, Et=500, Gl=450, vl=0.02, vt=0.44,
                    contraintesPlanes=True)
                    )
                self.comportements2D.append(
                    Elas_IsotTrans(2,
                    El=11580, Et=500, Gl=450, vl=0.02, vt=0.44,
                    contraintesPlanes=False))
        
        # phasefieldModel
        self.splits = PhaseFieldModel.get_splits()
        self.regularizations = PhaseFieldModel.get_regularisations()
        self.phaseFieldModels = []

        for c in self.comportements2D:
            for s in self.splits:
                for r in self.regularizations:
                        
                    if isinstance(c, Elas_IsotTrans) and s in ["Amor", "Miehe", "Stress"]:
                        continue

                    pfm = PhaseFieldModel(c,s,r,1,1)
                    self.phaseFieldModels.append(pfm)
            

    def test_LoiComportementBienCree(self):

        for comp in self.comportements3D:
            self.assertIsInstance(comp, LoiDeComportement)
            if isinstance(comp, Elas_Isot):
                E = comp.E
                v = comp.v
                if comp.dim == 2:
                    if comp.contraintesPlanes:
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
                
                c = LoiDeComportement.ApplyKelvinMandelCoefTo_Matrice(comp.dim, C_voigt)
                    
                verifC = np.linalg.norm(c-comp.get_C())/np.linalg.norm(c)
                self.assertTrue(verifC < 1e-12)
    
    def test_ElasIsoTrans2D(self):
        # Ici on verife que lorsque l'on change les axes ça fonctionne bien

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
                    contraintesPlanes=False,
                    axis_l=np.array([1,0,0]), axis_t=np.array([0,1,0]))

        Gt = compElasIsotTrans1.Gt
        kt = compElasIsotTrans1.kt

        c1 = np.array([[El+4*vl**2*kt, 2*kt*vl, 0],
                      [2*kt*vl, kt+Gt, 0],
                      [0, 0, 2*Gl]])

        verifc1 = np.linalg.norm(c1 - compElasIsotTrans1.get_C())/np.linalg.norm(c1)
        self.assertTrue(verifc1 < 1e-12)

        # Verif2 axis_l = [0, 1, 0] et axis_t = [1, 0, 0]
        compElasIsotTrans2 = Elas_IsotTrans(2,
                    El=11580, Et=500, Gl=450, vl=0.02, vt=0.44,
                    contraintesPlanes=False,
                    axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]))

        c2 = np.array([[kt+Gt, 2*kt*vl, 0],
                      [2*kt*vl, El+4*vl**2*kt, 0],
                      [0, 0, 2*Gl]])

        verifc2 = np.linalg.norm(c2 - compElasIsotTrans2.get_C())/np.linalg.norm(c2)
        self.assertTrue(verifc2 < 1e-12)

        # Verif3 axis_l = [0, 0, 1] et axis_t = [1, 0, 0]
        compElasIsotTrans3 = Elas_IsotTrans(2,
                    El=11580, Et=500, Gl=450, vl=0.02, vt=0.44,
                    contraintesPlanes=False,
                    axis_l=np.array([0,0,1]), axis_t=np.array([1,0,0]))

        c3 = np.array([[kt+Gt, kt-Gt, 0],
                      [kt-Gt, kt+Gt, 0],
                      [0, 0, 2*Gt]])

        verifc3 = np.linalg.norm(c3 - compElasIsotTrans3.get_C())/np.linalg.norm(c3)
        self.assertTrue(verifc3 < 1e-12)

    
    def test_Decomposition_psi(self):
        
        Ne = 50
        nPg = 1

        # Création de 2 espilons quelconques 2D
        Epsilon_e_pg = np.random.randn(Ne,nPg,3)
        
        # Epsilon_e_pg = np.random.rand(1,1,3)
        # Epsilon_e_pg[0,:] = np.array([1,-1,0])
        # # Epsilon_e_pg[1,:] = np.array([-100,500,0])


        # Epsilon_e_pg[0,0,:]=0
        # Epsilon_e_pg = np.zeros((Ne,1,nPg))
                
        tol = 1e-12

        for pfm in self.phaseFieldModels:
            
            assert isinstance(pfm, PhaseFieldModel)

            comportement = pfm.loiDeComportement
            coef = comportement.coef
            
            if isinstance(comportement, Elas_Isot) or isinstance(comportement, Elas_IsotTrans):                
                c = comportement.get_C()
            
            print(f"{comportement.nom} {comportement.contraintesPlanes} {pfm.split} {pfm.regularization}")

            
            if pfm.split == "Stress":
                # Ici il y a un beug quand v=0.499999 et en deformation plane
                pass

            if pfm.split == "AnisotStress":
                # Ici il y a un beug quand v=0.499999 et en deformation plane
                if isinstance(comportement, Elas_IsotTrans):
                    pass

            cP_e_pg, cM_e_pg = pfm.Calc_C(Epsilon_e_pg.copy(), True)

            # Test que cP + cM = c
            decompC = c-(cP_e_pg+cM_e_pg)
            verifC = np.linalg.norm(decompC)/np.linalg.norm(c)
            if pfm.split != "He":
                self.assertTrue(np.abs(verifC) < tol)
            else:
                # TODO regarder le probleme pour le split de He
                pass

            # Test que SigP + SigM = Sig
            Sig_e_pg = np.einsum('ij,epj->epi', c, Epsilon_e_pg, optimize='optimal')
            
            SigP = np.einsum('epij,epj->epi', cP_e_pg, Epsilon_e_pg, optimize='optimal')
            SigM = np.einsum('epij,epj->epi', cM_e_pg, Epsilon_e_pg, optimize='optimal') 
            decompSig = Sig_e_pg-(SigP+SigM)           
            verifSig = np.linalg.norm(decompSig)/np.linalg.norm(Sig_e_pg)
            if np.linalg.norm(Sig_e_pg)>0:                
                self.assertTrue(np.abs(verifSig) < tol)
            
            # Test que Eps:C:Eps = Eps:(cP+cM):Eps
            energiec = np.einsum('epj,ij,epi->ep', Epsilon_e_pg, c, Epsilon_e_pg, optimize='optimal')
            energiecP = np.einsum('epj,epij,epi->ep', Epsilon_e_pg, cP_e_pg, Epsilon_e_pg, optimize='optimal')
            energiecM = np.einsum('epj,epij,epi->ep', Epsilon_e_pg, cM_e_pg, Epsilon_e_pg, optimize='optimal')
            verifEnergie = np.linalg.norm(energiec-(energiecP+energiecM))/np.linalg.norm(energiec)
            if np.linalg.norm(energiec)>0:
                self.assertTrue(np.abs(verifEnergie) < tol)



if __name__ == '__main__':
    try:
        os.system("cls")
        unittest.main(verbosity=2)
    except:
        print("")
# %%
