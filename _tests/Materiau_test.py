import unittest
import os
from Materiau import PhaseFieldModel, LoiDeComportement, Elas_Isot
import numpy as np

class Test_Materiau(unittest.TestCase):
    def setUp(self):

        self.voigtNotations = [True, False]

        # Comportement Elatique Isotrope
        E = 210e9
        v = 0.3
        self.comportements = []
        for vn in self.voigtNotations:
            for comp in LoiDeComportement.get_LoisDeComportement():
                if comp == Elas_Isot:
                    self.comportements.append(Elas_Isot(2, E=E, v=v, contraintesPlanes=False, useVoigtNotation=vn))
                    self.comportements.append(Elas_Isot(2, E=E, v=v, contraintesPlanes=True, useVoigtNotation=vn))
                    self.comportements.append(Elas_Isot(3, E=E, v=v, useVoigtNotation=vn))

        # phasefieldModel
        self.splits = PhaseFieldModel.get_splits()
        self.regularizations = PhaseFieldModel.get_regularisations()
        self.phaseFieldModels = []
        for vn in self.voigtNotations:
            comportement = Elas_Isot(2,E=E,v=v, useVoigtNotation=vn)
            for s in self.splits:
                for r in self.regularizations:
                    if vn:
                        pfm = PhaseFieldModel(comportement,s,r,1,1)
                    else:
                        pfm = PhaseFieldModel(comportement,s,r,1,1)
                    self.phaseFieldModels.append(pfm)

    def test_BienCree_Isotrope(self):

        for comp in self.comportements:
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
                if comp.useVoigtNotation:
                    c = C_voigt
                else:
                    c = LoiDeComportement.ToKelvinMandelNotation(comp.dim, C_voigt)
                verifC = np.linalg.norm(c-comp.get_C())/np.linalg.norm(c)
                self.assertTrue(verifC < 1e-12)

    
    def test_Decomposition_Bourdin_Amor_Miehe(self):
        
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
            
            if isinstance(comportement, Elas_Isot):
                c = comportement.get_C()
            else:
                raise "Pas implémenté"
            
            cP_e_pg, cM_e_pg = pfm.Calc_C(Epsilon_e_pg)

            # Test que cP + cM = c
            decompC = c-(cP_e_pg+cM_e_pg)
            verifC = np.linalg.norm(decompC)/np.linalg.norm(c)
            self.assertTrue(np.abs(verifC) < tol)

            # Test que SigP + SigM = Sig
            Sig = np.einsum('ij,epj->epj', c, Epsilon_e_pg, optimize=True)
            SigP = np.einsum('epij,epj->epj', cP_e_pg, Epsilon_e_pg, optimize=True)
            SigM = np.einsum('epij,epj->epj', cM_e_pg, Epsilon_e_pg, optimize=True)
            verifSig = np.linalg.norm(Sig-(SigP+SigM))/np.linalg.norm(Sig)
            if np.linalg.norm(Sig)>0:
                self.assertTrue(np.abs(verifSig) < tol)
            
            # Test que Eps:C:Eps = Eps:(cP+cM):Eps
            energiec = np.einsum('epj,ij,epj->ep', Epsilon_e_pg, c, Epsilon_e_pg, optimize=True)
            energiecP = np.einsum('epj,epij,epj->ep', Epsilon_e_pg, cP_e_pg, Epsilon_e_pg, optimize=True)
            energiecM = np.einsum('epj,epij,epj->ep', Epsilon_e_pg, cM_e_pg, Epsilon_e_pg, optimize=True)
            verifEnergie = np.linalg.norm(energiec-(energiecP+energiecM))/np.linalg.norm(energiec)
            if np.linalg.norm(energiec)>0:
                self.assertTrue(np.abs(verifEnergie) < tol)

if __name__ == '__main__':
    try:
        os.system("cls")
        unittest.main(verbosity=2)
    except:
        print("")