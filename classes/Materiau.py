from typing import cast
from Mesh import Mesh
import numpy as np
import Affichage

class LoiDeComportement(object):

    @staticmethod
    def get_LoisDeComportement():
        liste = [Elas_Isot]
        return liste

    """Classe des lois de comportements C de (Sigma = C * Epsilon)
    (Elas_isot, ...)
    """
    def __init__(self, dim: int, C: np.ndarray, S: np.ndarray, epaisseur: float):
        
        self.__dim = dim
        """dimension lié a la loi de comportement"""

        if dim == 2:
            assert epaisseur > 0 , "Doit être supérieur à 0"
            self.__epaisseur = epaisseur
        
        self.__C = C
        """Loi de comportement pour la loi de Lamé en kelvin mandel"""

        self.__S = S
        """Loi de comportement pour la loi de Hooke en kelvin mandel"""

    def __get_coef_notation(self):
        return np.sqrt(2)            
    coef = property(__get_coef_notation)
    """Coef lié à la notation de kelvin mandel=racine(2)"""

    def get_C(self):
        """Renvoie une copie de la loi de comportement pour la loi de Lamé en Kelvin Mandel\n
        En 2D:
        -----
        C -> C : Epsilon = Sigma [Sxx Syy racine(2)*Sxy]\n
        En 3D:
        -----
        C -> C : Epsilon = Sigma [Sxx Syy Szz racine(2)*Syz racine(2)*Sxz racine(2)*Sxy]
        """        
        return self.__C.copy()

    def get_S(self):
        """Renvoie une copie de la loi de comportement pour la loi de Hooke en Kelvin Mandel\n
        En 2D:
        -----        
        S -> S : Sigma = Epsilon [Exx Eyy racine(2)*Exy]\n
        En 3D:
        -----        
        S -> S : Sigma = Epsilon [Exx Eyy Ezz racine(2)*Eyz racine(2)*Exz racine(2)*Exy]
        """
        return self.__S.copy()

    def __getdim(self):
        return self.__dim
    dim = property(__getdim)

    def __getepaisseur(self):
        if self.__dim == 2:
            return self.__epaisseur
        else:
            return 1.0
    epaisseur = property(__getepaisseur)

    def __get_nom(self):
        return type(self).__name__
    nom = property(__get_nom)

    def __get_resume(self):
        return ""
    resume = cast(str, property(__get_resume))

    def AppliqueCoefSurBrigi(self, B_rigi_e_pg: np.ndarray):

        if self.__dim == 2:
            coord=2
        elif self.__dim == 3:
            coord=[3,4,5]
        else:
            raise "Pas implémenté"

        B_rigi_e_pg[:,:,coord,:] = B_rigi_e_pg[:,:,coord,:]/self.coef

        return B_rigi_e_pg
    
    @staticmethod
    def ApplyKelvinMandelCoef(dim: int, Matrice: np.ndarray):        
        """Applique ces coefs à la matrice\n
        si 2D:
        \n
        [1,1,r2]\n
        [1,1,r2]\n
        [r2, r2, 2]]\n

        \nsi 3D:
        \n
        [1,1,1,r2,r2,r2]\n
        [1,1,1,r2,r2,r2]\n
        [1,1,1,r2,r2,r2]\n
        [r2,r2,r2,2,2,2]\n
        [r2,r2,r2,2,2,2]\n
        [r2,r2,r2,2,2,2]]\n
        
        """

        r2 = np.sqrt(2)

        if dim == 2:            
            transform = np.array([  [1,1,r2],
                                    [1,1,r2],
                                    [r2, r2, 2]])
        elif dim == 3:
            transform = np.array([  [1,1,1,r2,r2,r2],
                                    [1,1,1,r2,r2,r2],
                                    [1,1,1,r2,r2,r2],
                                    [r2,r2,r2,2,2,2],
                                    [r2,r2,r2,2,2,2],
                                    [r2,r2,r2,2,2,2]])
        else:
            raise "Pas implémenté"

        matriceMandelCoef = Matrice*transform

        return matriceMandelCoef

class Elas_Isot(LoiDeComportement):   

    def __init__(self, dim: int, E=210000.0, v=0.3, contraintesPlanes=True, epaisseur=1.0):
        """Creer la matrice de comportement d'un matériau : Elastique isotrope

        Parameters
        ----------
        dim : int
            Dimension de la simulation 2D ou 3D
        E : float, optional
            Module d'elasticité du matériau en MPa (> 0)
        v : float, optional
            Coef de poisson ]-1;0.5]
        contraintesPlanes : bool
            Contraintes planes si dim = 2 et True, by default True        
        """       

        # Vérification des valeurs
        assert dim in [2,3], "doit être en 2 et 3"
        self.__dim = dim

        assert E > 0.0, "Le module élastique doit être > 0 !"
        self.E=E
        """Module de Young"""

        poisson = "Le coef de poisson doit être compris entre ]-1;0.5["
        assert v > -1.0 and v < 0.5, poisson
        self.v=v
        """Coef de poisson"""        

        if dim == 2:
            self.contraintesPlanes = contraintesPlanes
            """type de simplification 2D"""

        C, S = self.__Comportement_Elas_Isot()

        LoiDeComportement.__init__(self, dim, C, S, epaisseur)

    def __get_resume(self):
        if self.__dim == 2:
            resume = f"\nElas_Isot :\nE = {self.E:.2e}, v = {self.v}\nCP = {self.contraintesPlanes}, ep = {self.epaisseur:.2e}"
        else:
            resume = f"\nElas_Isot :\nE = {self.E:.2e}, v = {self.v}"
        return resume
    resume = property(__get_resume)

    def get_lambda(self):

        E=self.E
        v=self.v
        
        l = E*v/((1+v)*(1-2*v))

        if self.__dim == 2 and self.contraintesPlanes:
            l = E*v/(1-v**2)
        
        return l
    
    def get_mu(self):
        
        E=self.E
        v=self.v

        mu = E/(2*(1+v))

        return mu
    
    def get_bulk(self):

        E=self.E
        v=self.v

        mu = self.get_mu()
        l = self.get_lambda()
        
        bulk = l + 2*mu/self.dim        

        return bulk

    def __Comportement_Elas_Isot(self):
        """"Construit les matrices de comportement en kelvin mandel\n
        
        En 2D:
        -----

        C -> C : Epsilon = Sigma [Sxx Syy racine(2)*Sxy]\n
        S -> S : Sigma = Epsilon [Exx Eyy racine(2)*Exy]

        En 3D:
        -----

        C -> C : Epsilon = Sigma [Sxx Syy Szz racine(2)*Syz racine(2)*Sxz racine(2)*Sxy]\n
        S -> S : Sigma = Epsilon [Exx Eyy Ezz racine(2)*Eyz racine(2)*Exz racine(2)*Exy]

        """

        E=self.E
        v=self.v

        dim = self.__dim

        mu = self.get_mu()
        l = self.get_lambda()

        if dim == 2:
            if self.contraintesPlanes:
                # C = np.array([  [4*(mu+l), 2*l, 0],
                #                 [2*l, 4*(mu+l), 0],
                #                 [0, 0, 2*mu+l]]) * mu/(2*mu+l)

                cVoigt = np.array([ [1, v, 0],
                                    [v, 1, 0],
                                    [0, 0, (1-v)/2]]) * E/(1-v**2)
                
            else:
                cVoigt = np.array([ [l + 2*mu, l, 0],
                                    [l, l + 2*mu, 0],
                                    [0, 0, mu]])

                # C = np.array([  [1, v/(1-v), 0],
                #                 [v/(1-v), 1, 0],
                #                 [0, 0, (1-2*v)/(2*(1-v))]]) * E*(1-v)/((1+v)*(1-2*v))

        elif dim == 3:
            
            cVoigt = np.array([ [l+2*mu, l, l, 0, 0, 0],
                                [l, l+2*mu, l, 0, 0, 0],
                                [l, l, l+2*mu, 0, 0, 0],
                                [0, 0, 0, mu, 0, 0],
                                [0, 0, 0, 0, mu, 0],
                                [0, 0, 0, 0, 0, mu]])
        
        c = LoiDeComportement.ApplyKelvinMandelCoef(dim, cVoigt)

        s = np.linalg.inv(c)

        return c, s


class PhaseFieldModel:

    @staticmethod
    def get_splits():
        __splits = ["Bourdin","Amor","Miehe","Stress","AnisotStress"]
        return __splits
    
    @staticmethod
    def get_regularisations():
        __regularizations = ["AT1","AT2"]
        return __regularizations

    def __get_k(self):
        Gc = self.__Gc
        l0 = self.__l0

        k = Gc * l0

        if self.__regularization == "AT1":
            k = 3/4 * k

        return k        
    k = property(__get_k)

    def get_r_e_pg(self, PsiP_e_pg: np.ndarray):
        Gc = self.__Gc
        l0 = self.__l0

        r = 2 * PsiP_e_pg

        if self.__regularization == "AT2":
            r = r + (Gc/l0)
        
        return r

    def get_f_e_pg(self, PsiP_e_pg: np.ndarray):
        Gc = self.__Gc
        l0 = self.__l0

        f = 2 * PsiP_e_pg

        if self.__regularization == "AT1":            
            f = f - ( (3*Gc) / (8*l0) )            
            absF = np.abs(f)
            f = (f+absF)/2
        
        return f

    def get_g_e_pg(self, d_n: np.ndarray, mesh: Mesh, matriceType: str, k_residu=1e-10):
        """Fonction de dégradation en energies / contraintes
        k_residu=1e-10
        Args:
            d_n (np.ndarray): Endomagement localisé aux noeuds (Nn,1)
            mesh (Mesh): maillage
        """
        d_e_n = mesh.Localises_sol_e(d_n)
        Nd_pg = mesh.get_N_scalaire_pg(matriceType)

        d_e_pg = np.einsum('pij,ej->ep', Nd_pg, d_e_n, optimize=True)        

        if self.__regularization in ["AT1","AT2"]:
            g_e_pg = (1-d_e_pg)**2 + k_residu
        else:
            raise "Pas implémenté"

        assert mesh.Ne == g_e_pg.shape[0]
        assert mesh.get_nPg(matriceType) == g_e_pg.shape[1]
        
        return g_e_pg

    def __resume(self):
        resum = '\nPhaseField :'        
        resum += f'\nsplit : {self.__split}'
        resum += f'\nregularisation : {self.__regularization}'
        resum += f'\nGc : {self.__Gc:.2e}'
        resum += f'\nl0 : {self.__l0:.2e}'
        return resum
    resume = property(__resume)

    def __get_split(self):
        return self.__split
    split = property(__get_split)

    def __get_regularization(self):
        return self.__regularization
    regularization = property(__get_regularization)
    
    def __get_loiDeComportement(self):
        return self.__loiDeComportement
    loiDeComportement = cast(LoiDeComportement, property(__get_loiDeComportement))

    def __get_useHistory(self):
        return self.__useHistory
    useHistory = property(__get_useHistory)

    def __init__(self, loiDeComportement: LoiDeComportement, split: str, regularization: str, Gc: float, l_0: float,
    useHistory=True):
        """Création d'un comportement Phase Field

            Parameters
            ----------
            loiDeComportement : LoiDeComportement
                Loi de comportement du matériau ["Elas_Isot"]
            split : str
                Split de la densité d'energie elastique ["Bourdin","Amor","Miehe","Stress","AnisotStress"]
            regularization : str
                Modèle de régularisation de la fissure ["AT1","AT2"]
            Gc : float
                Taux de libération d'énergie critque [J/m^2]
            l_0 : float
                Largeur de régularisation de la fissure
        """

        assert isinstance(loiDeComportement, LoiDeComportement), "Doit être une loi de comportement"
        self.__loiDeComportement = loiDeComportement

        assert split in PhaseFieldModel.get_splits(), f"Doit être compris dans {PhaseFieldModel.get_splits()}"
        if not isinstance(loiDeComportement, Elas_Isot):
            assert not split in ["Amor, Miehe"], "Ces splits ne sont implémentés que pour Elas_Isot"
        self.__split =  split
        """Split de la densité d'energie elastique ["Bourdin","Amor","Miehe","Stress","AnisotStress"]"""
        
        assert regularization in PhaseFieldModel.get_regularisations(), f"Doit être compris dans {PhaseFieldModel.get_regularisations()}"
        self.__regularization = regularization
        """Modèle de régularisation de la fissure ["AT1","AT2"]"""

        assert Gc > 0, "Doit être supérieur à 0" 
        self.__Gc = Gc
        """Taux de libération d'énergie critque [J/m^2]"""

        assert l_0 > 0, "Doit être supérieur à 0"
        self.__l0 = l_0
        """Largeur de régularisation de la fissure"""

        self.__useHistory = useHistory
        """Utilise ou non le champ histoire"""        
            
    def Calc_psi_e_pg(self, Epsilon_e_pg: np.ndarray):
        """Calcul de la densité d'energie elastique\n
        psiP_e_pg = 1/2 SigmaP_e_pg * Epsilon_e_pg\n
        psiM_e_pg = 1/2 SigmaM_e_pg * Epsilon_e_pg\n
        Tel que :\n
        SigmaP_e_pg = cP_e_pg * Epsilon_e_pg\n
        SigmaM_e_pg = cM_e_pg * Epsilon_e_pg        
        """
        
        # Data
        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]

        SigmaP_e_pg, SigmaM_e_pg = self.Calc_Sigma_e_pg(Epsilon_e_pg)

        psiP_e_pg = 1/2 * np.einsum('epi,epi->ep', SigmaP_e_pg, Epsilon_e_pg, optimize=True).reshape((Ne, nPg))
        psiM_e_pg = 1/2 * np.einsum('epi,epi->ep', SigmaM_e_pg, Epsilon_e_pg, optimize=True).reshape((Ne, nPg))
        
        return psiP_e_pg, psiM_e_pg

    def Calc_Sigma_e_pg(self, Epsilon_e_pg: np.ndarray):
        """Calcul la contrainte en fonction de la deformation et du split\n
        Ici on calcul :\n
        SigmaP_e_pg = cP_e_pg * Epsilon_e_pg \n
        SigmaM_e_pg = cM_e_pg * Epsilon_e_pg

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            deformations stockées aux elements et Points de Gauss

        Returns
        -------
        np.ndarray
            SigmaP_e_pg, SigmaM_e_pg : les contraintes stockées aux elements et Points de Gauss
        """       

        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]
        comp = Epsilon_e_pg.shape[2]

        cP_e_pg, cM_e_pg = self.Calc_C(Epsilon_e_pg)

        SigmaP_e_pg = np.einsum('epij,epj->epi', cP_e_pg, Epsilon_e_pg, optimize=True).reshape((Ne, nPg, comp))
        SigmaM_e_pg = np.einsum('epij,epj->epi', cM_e_pg, Epsilon_e_pg, optimize=True).reshape((Ne, nPg, comp))

        return SigmaP_e_pg, SigmaM_e_pg
    
    def Calc_C(self, Epsilon_e_pg: np.ndarray, verif=False):
        """Calcul la loi de comportement en fonction du split

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            deformations stockées aux élements et points de gauss (Pas utilisé si bourdin)

        Returns
        -------
        np.ndarray
            Renvoie cP_e_pg, cM_e_pg
        """

        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]

        
            
        if self.__split == "Bourdin":
            c = self.__loiDeComportement.get_C()
            c = c[np.newaxis, np.newaxis,:,:]
            c = np.repeat(c, Ne, axis=0)
            c = np.repeat(c, nPg, axis=1)

            cP_e_pg = c
            cM_e_pg = 0*cP_e_pg

        elif self.__split == "Amor":
            cP_e_pg, cM_e_pg = self.__Split_Amor(Epsilon_e_pg)

        elif self.__split == "Miehe":
            cP_e_pg, cM_e_pg = self.__Split_Miehe(Epsilon_e_pg, verif)
        
        elif self.__split in ["Stress","AnisotStress"]:
            cP_e_pg, cM_e_pg = self.__Split_Stress(Epsilon_e_pg, verif)
        
        return cP_e_pg, cM_e_pg            

    def __Split_Amor(self, Epsilon_e_pg: np.ndarray):

        assert isinstance(self.__loiDeComportement, Elas_Isot), f"Implémenté que pour un matériau Elas_Isot"

        loiDeComportement = self.__loiDeComportement                

        bulk = loiDeComportement.get_bulk()
        mu = loiDeComportement.get_mu()

        Rp_e_pg, Rm_e_pg = self.__Rp_Rm(Epsilon_e_pg)

        dim = self.__loiDeComportement.dim

        if dim == 2:
            Ivoigt = np.array([1,1,0]).reshape((3,1))
            taille = 3
        else:
            Ivoigt = np.array([1,1,1,0,0,0]).reshape((6,1))
            taille = 6

        IxI = Ivoigt.dot(Ivoigt.T)

        # Projecteur deviatorique
        Pdev_e_pg = np.eye(taille) - 1/dim * IxI
        partieDeviateur = 2*mu*Pdev_e_pg
            
        
        # projetcteur spherique
        spherP_e_pg = np.einsum('ep,ij->epij', Rp_e_pg, IxI, optimize=True)
        spherM_e_pg = np.einsum('ep,ij->epij', Rm_e_pg, IxI, optimize=True)
       
        cP_e_pg = bulk*spherP_e_pg + partieDeviateur
        cM_e_pg = bulk*spherM_e_pg

        projecteurs = {
            "spherP_e_pg" : spherP_e_pg,
            "spherM_e_pg" : spherM_e_pg,
            "Pdev_e_pg" : Pdev_e_pg
        }

        return cP_e_pg, cM_e_pg

    def __Rp_Rm(self, vecteur_e_pg: np.ndarray):
        """Renvoie Rp_e_pg, Rm_e_pg"""

        Ne = vecteur_e_pg.shape[0]
        nPg = vecteur_e_pg.shape[1]

        dim = self.__loiDeComportement.dim

        tr_Eps = np.zeros((Ne, nPg))

        tr_Eps = vecteur_e_pg[:,:,0] + vecteur_e_pg[:,:,1]

        if dim == 3:
            tr_Eps += vecteur_e_pg[:,:,2]

        Rp_e_pg = (1+np.sign(tr_Eps))/2
        Rm_e_pg = (1+np.sign(-tr_Eps))/2

        return Rp_e_pg, Rm_e_pg
    
    def __Split_Miehe(self, Epsilon_e_pg: np.ndarray, verif=False):

        assert isinstance(self.__loiDeComportement, Elas_Isot), f"Implémenté que pour un matériau Elas_Isot"

        dim = self.__loiDeComportement.dim
        assert dim == 2, "Implémenté que en 2D"

        projP_e_pg, projM_e_pg = self.__Decomposition_Spectrale(Epsilon_e_pg, verif)

        # Calcul Rp et Rm
        Rp_e_pg, Rm_e_pg = self.__Rp_Rm(Epsilon_e_pg)
        
        # Calcul IxI
        I = np.array([1,1,0]).reshape((3,1))
        IxI = I.dot(I.T)

        # Calcul partie sphérique
        spherP_e_pg = np.einsum('ep,ij->epij', Rp_e_pg, IxI, optimize=True)
        spherM_e_pg = np.einsum('ep,ij->epij', Rm_e_pg, IxI, optimize=True)

        # Calcul de la loi de comportement
        lamb = self.__loiDeComportement.get_lambda()
        mu = self.__loiDeComportement.get_mu()

        cP_e_pg = lamb*spherP_e_pg + 2*mu*projP_e_pg
        cM_e_pg = lamb*spherM_e_pg + 2*mu*projM_e_pg

        projecteurs = {
            "projP_e_pg" : projP_e_pg,
            "projM_e_pg" : projM_e_pg,
            "spherP_e_pg" : spherP_e_pg,
            "spherM_e_pg" : spherM_e_pg
        }

        return cP_e_pg, cM_e_pg
    def __Split_Stress(self, Epsilon_e_pg: np.ndarray, verif=False):
        """Construit Cp et Cm pour le split en contraintse"""

        # Récupère les contraintes
        # Ici le matériau est supposé homogène
        loiDeComportement = self.__loiDeComportement
        C = loiDeComportement.get_C()    
        Sigma_e_pg = np.einsum('ij,epj->epi',C, Epsilon_e_pg, optimize=True)

        # Construit les projecteurs tel que SigmaP = Pp : Sigma et SigmaM = Pm : Sigma                    
        projP_e_pg, projM_e_pg = self.__Decomposition_Spectrale(Sigma_e_pg, verif)

        if self.__split == "Stress":
        
            assert isinstance(loiDeComportement, Elas_Isot)

            E = loiDeComportement.E
            v = loiDeComportement.v

            c = loiDeComportement.get_C()

            # Calcul Rp et Rm
            Rp_e_pg, Rm_e_pg = self.__Rp_Rm(Sigma_e_pg)
            
            # Calcul IxI
            I = np.array([1,1,0]).reshape((3,1))
            IxI = I.dot(I.T)

            RpIxI_e_pg = np.einsum('ep,ij->epij',Rp_e_pg, IxI, optimize=True)
            RmIxI_e_pg = np.einsum('ep,ij->epij',Rm_e_pg, IxI, optimize=True)

            if loiDeComportement.contraintesPlanes:
                sP_e_pg = (1+v)/E*projP_e_pg - v/E * RpIxI_e_pg
                sM_e_pg = (1+v)/E*projM_e_pg - v/E * RmIxI_e_pg
            else:
                sP_e_pg = (1+v)/E*projP_e_pg - v*(1+v)/E * RpIxI_e_pg
                sM_e_pg = (1+v)/E*projM_e_pg - v*(1+v)/E * RmIxI_e_pg


            cT = c.T
            cP_e_pg = np.einsum('ij,epjk,kl->epil', cT, sP_e_pg, c, optimize=True)
            cM_e_pg = np.einsum('ij,epjk,kl->epil', cT, sM_e_pg, c, optimize=True)

            # # Ici c'est un test pour verifier que cT : S : c = inv(S)

            # detP_e_pg = np.linalg.det(sP_e_pg); e_pg_detPnot0 = np.where(detP_e_pg!=0)
            # detM_e_pg = np.linalg.det(sM_e_pg); e_pg_detMnot0 = np.where(detM_e_pg!=0)

            # invSP_e_pg = np.zeros(sP_e_pg.shape)
            # invSM_e_pg = np.zeros(sM_e_pg.shape)
            
            # invSP_e_pg[e_pg_detPnot0] = np.linalg.inv(sP_e_pg[e_pg_detPnot0])
            # invSM_e_pg[e_pg_detMnot0] = np.linalg.inv(sM_e_pg[e_pg_detMnot0])

            # testP = np.linalg.norm(invSP_e_pg-cP_e_pg)/np.linalg.norm(cP_e_pg)
            # testM = np.linalg.norm(invSM_e_pg-cM_e_pg)/np.linalg.norm(cM_e_pg)
            # pass
        
        elif self.__split == "AnisotStress":

            # Construit les ppc_e_pg = Pp : C et ppcT_e_pg = transpose(Pp : C)
            Ppc_e_pg = np.einsum('epij,jk->epik', projP_e_pg, C, optimize=True)
            Pmc_e_pg = np.einsum('epij,jk->epik', projM_e_pg, C, optimize=True)

            # Construit Cp et Cm
            S = loiDeComportement.get_S()
            Cpp = np.einsum('epij,jk,epkl->epil', Ppc_e_pg, S, Ppc_e_pg, optimize=True)
            Cpm = np.einsum('epij,jk,epkl->epil', Ppc_e_pg, S, Pmc_e_pg, optimize=True)
            Cmm = np.einsum('epij,jk,epkl->epil', Pmc_e_pg, S, Pmc_e_pg, optimize=True)
            Cmp = np.einsum('epij,jk,epkl->epil', Pmc_e_pg, S, Ppc_e_pg, optimize=True)

            # cP_e_pg = Cpp #Diffuse
            # cM_e_pg = Cmm + Cpm + Cmp

            cP_e_pg = Cpp + Cpm + Cmp #Diffuse
            cM_e_pg = Cmm 

            # cP_e_pg = Cpp + Cpm #Diffuse
            # cM_e_pg = Cmm + Cmp

            # cP_e_pg = Cpp #Diffuse
            # cM_e_pg = Cmm + Cpm + Cmp
        
        else:
            raise "Erreur"

        return cP_e_pg, cM_e_pg

    def __Decomposition_Spectrale(self, vecteur_e_pg: np.ndarray, verif=False):
        """Calcul projP et projM tel que :\n

        vecteur_e_pg = [1 1 racine(2)] \n
        
        vecteurP = projP : vecteur -> [1, 1, racine(2)] si mandel\n
        vecteurM = projM : vecteur -> [1, 1, racine(2)] si mandel\n

        renvoie projP, projM
        """

        # remet en voigt
        # vecteur_e_pg[:,:,2] *= 1/np.sqrt(2)

        # A partir d'ici on est en voigt
        coef = self.__loiDeComportement.coef

        dim = self.__loiDeComportement.dim
        assert dim == 2, "Implémenté que en 2D"

        Ne = vecteur_e_pg.shape[0]
        nPg = vecteur_e_pg.shape[1]

        # Reconsruit le tenseur des deformations [e,pg,dim,dim]
        matrice_e_pg = np.zeros((Ne,nPg,2,2))
        matrice_e_pg[:,:,0,0] = vecteur_e_pg[:,:,0]
        matrice_e_pg[:,:,1,1] = vecteur_e_pg[:,:,1]
        matrice_e_pg[:,:,0,1] = vecteur_e_pg[:,:,2]/coef
        matrice_e_pg[:,:,1,0] = vecteur_e_pg[:,:,2]/coef        
        
        # invariants du tenseur des deformations [e,pg]
        trace_e_pg = np.trace(matrice_e_pg, axis1=2, axis2=3)
        determinant_e_pg = np.linalg.det(matrice_e_pg)

        # Calculs des valeurs propres [e,pg]
        delta = trace_e_pg**2 - (4*determinant_e_pg)
        val_e_pg = np.zeros((Ne,nPg,2))
        val_e_pg[:,:,0] = (trace_e_pg - np.sqrt(delta))/2
        val_e_pg[:,:,1] = (trace_e_pg + np.sqrt(delta))/2

        # Constantes pour calcul de m1 = (matrice_e_pg - v2*I)/(v1-v2)
        v2I = np.einsum('ep,ij->epij', val_e_pg[:,:,1], np.eye(2), optimize=True)
        v1_m_v2 = val_e_pg[:,:,0] - val_e_pg[:,:,1]
        
        # identifications des elements et points de gauss ou vp1 != vp2
        # elements, pdgs = np.where(v1_m_v2 != 0)
        elements, pdgs = np.where(val_e_pg[:,:,0] != val_e_pg[:,:,1])

        # construction des bases propres m1 et m2 [e,pg,dim,dim]
        M1 = np.zeros((Ne,nPg,2,2))
        M1[:,:,0,0] = 1
        if elements.size > 0:
            m1_tot = np.einsum('epij,ep->epij', matrice_e_pg-v2I, 1/v1_m_v2, optimize=True)
            M1[elements, pdgs] = m1_tot[elements, pdgs]            
        M2 = np.eye(2) - M1

        if verif:
            # test ortho entre M1 et M2 
            verifOrtho_M1M2 = np.einsum('epij,epij->ep', M1, M2, optimize=True)
            assert np.abs(verifOrtho_M1M2).max() < 1e-10, "Orthogonalité entre M1 et M2 non vérifié"
        
        # Passage des bases propres sous la forme dun vecteur [e,pg,3]  ou [e,pg,6]
        m1 = np.zeros((Ne,nPg,3)); m2 = np.zeros((Ne,nPg,3))
        m1[:,:,0] = M1[:,:,0,0];   m2[:,:,0] = M2[:,:,0,0]
        m1[:,:,1] = M1[:,:,1,1];   m2[:,:,1] = M2[:,:,1,1]
        # m1[:,:,2] = M1[:,:,0,1];   m2[:,:,2] = M2[:,:,0,1]
        m1[:,:,2] = M1[:,:,0,1]*coef;   m2[:,:,2] = M2[:,:,0,1]*coef # Ici on met pas le coef pour que ce soit en [1 1 1]

        # Calcul de mixmi [e,pg,3,3] ou [e,pg,6,6]        
        m1xm1 = np.einsum('epi,epj->epij', m1, m1, optimize=True)
        m2xm2 = np.einsum('epi,epj->epij', m2, m2, optimize=True)

        # Récupération des parties positives et négatives des valeurs propres [e,pg,2]
        valp = (val_e_pg+np.abs(val_e_pg))/2
        valm = (val_e_pg-np.abs(val_e_pg))/2

        # Calcul des di [e,pg,2]
        dvalp = np.heaviside(val_e_pg,0.5)
        dvalm = np.heaviside(-val_e_pg,0.5)        

        # Calcul des Beta Plus [e,pg,1]
        BetaP = dvalp[:,:,0].copy()
        BetaP[elements,pdgs] = (valp[elements,pdgs,0]-valp[elements,pdgs,1])/v1_m_v2[elements,pdgs]
        
        # Calcul de Beta Moin [e,pg,1]
        BetaM = dvalm[:,:,0].copy()
        BetaM[elements,pdgs] = (valm[elements,pdgs,0]-valm[elements,pdgs,1])/v1_m_v2[elements,pdgs]

        # Calcul de gamma [e,pg,2]
        gammap = dvalp - np.repeat(BetaP.reshape((Ne,nPg,1)),2, axis=2)
        gammam = dvalm - np.repeat(BetaM.reshape((Ne,nPg,1)), 2, axis=2)
        
        matriceI = np.eye(3)

        # Projecteur P tel que vecteur_e_pg = projP_e_pg : vecteur_e_pg
        BetaP_x_matriceI = np.einsum('ep,ij->epij', BetaP, matriceI, optimize=True)
        gamma1P_x_m1xm1 = np.einsum('ep,epij->epij', gammap[:,:,0], m1xm1, optimize=True)
        gamma2P_x_m2xm2 = np.einsum('ep,epij->epij', gammap[:,:,1], m2xm2, optimize=True)
        projP = BetaP_x_matriceI + gamma1P_x_m1xm1 + gamma2P_x_m2xm2

        # Projecteur M tel que EpsM = projM : Eps
        BetaM_x_matriceI = np.einsum('ep,ij->epij', BetaM, matriceI, optimize=True)
        gamma1M_x_m1xm1 = np.einsum('ep,epij->epij', gammam[:,:,0], m1xm1, optimize=True)
        gamma2M_x_m2xm2 = np.einsum('ep,epij->epij', gammam[:,:,1], m2xm2, optimize=True)
        projM = BetaM_x_matriceI + gamma1M_x_m1xm1 + gamma2M_x_m2xm2

        if verif:
            # Verification de la décomposition et de l'orthogonalité
            # projecteur en [1; 1; 1]
            vecteurP = np.einsum('epij,epj->epi', projP, vecteur_e_pg, optimize=True)
            vecteurM = np.einsum('epij,epj->epi', projM, vecteur_e_pg, optimize=True)           
            
            # Décomposition vecteur_e_pg = vecteurP_e_pg + vecteurM_e_pg
            decomp = vecteur_e_pg-(vecteurP + vecteurM)
            if np.linalg.norm(vecteur_e_pg) > 0:
                verifDecomp = np.linalg.norm(decomp)/np.linalg.norm(vecteur_e_pg)
                assert verifDecomp < 1e-12

            # Orthogonalité
            ortho_vP_vM = np.abs(np.einsum('epi,epi->ep',vecteurP, vecteurM, optimize=True))
            ortho_vM_vP = np.abs(np.einsum('epi,epi->ep',vecteurM, vecteurP, optimize=True))
            ortho_v_v = np.abs(np.einsum('epi,epi->ep', vecteur_e_pg, vecteur_e_pg, optimize=True))
            if ortho_v_v.min() > 0:
                vertifOrthoEpsPM = np.max(ortho_vP_vM/ortho_v_v)
                tvertifOrthoEpsPM = ortho_vP_vM/ortho_v_v
                assert vertifOrthoEpsPM < 1e-12
                vertifOrthoEpsMP = np.max(ortho_vM_vP/ortho_v_v)
                assert vertifOrthoEpsMP < 1e-12
            
        return projP, projM

class Materiau:
    
    def __get_dim(self):
        return self.comportement.dim
    dim = property(__get_dim)

    def __get_comportement(self):
        if self.isDamaged:
            return self.__phaseFieldModel.loiDeComportement
        else:
            return self.__comportement
    comportement = cast(LoiDeComportement, property(__get_comportement))

    def __get_isDamaged(self):
        if self.__phaseFieldModel == None:
            return False
        else:
            return True
    isDamaged = property(__get_isDamaged)

    def __get_phaseFieldModel(self):
        if self.isDamaged:
            return self.__phaseFieldModel
        else:
            print("Le matériau n'est pas endommageable (pas de modèle PhaseField)")
            return None
            
    phaseFieldModel = cast(PhaseFieldModel, property(__get_phaseFieldModel))
    """Modèle d'endommagement"""

    def __init__(self, comportement=None, phaseFieldModel=None, ro=8100.0, verbosity=True):
        """Creer un materiau avec la loi de comportement ou le phase field model communiqué

        Parameters
        ----------                        
        ro : float, optional
            Masse volumique en kg.m^-3
        epaisseur : float, optional
            epaisseur du matériau si en 2D > 0 !
        """
        if verbosity:
            Affichage.NouvelleSection("Matériau")        

        if comportement != None:
            assert isinstance(comportement, LoiDeComportement)

        assert ro > 0 , "Doit être supérieur à 0"
        self.ro = ro

        # Initialisation des variables de la classe

        if isinstance(phaseFieldModel, PhaseFieldModel):
            self.__phaseFieldModel = phaseFieldModel
            """Phase field model"""                
        else:
            self.__comportement = comportement
            self.__phaseFieldModel = None
        
        self.__verbosity = verbosity

        if self.__verbosity:
            self.Resume()

    def Resume(self):        
        if self.isDamaged:
            print(self.__phaseFieldModel.loiDeComportement.resume)
            print(self.__phaseFieldModel.resume)
        else:
            print(self.__comportement.resume)

