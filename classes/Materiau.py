from Mesh import Mesh
import numpy as np

class LoiDeComportement(object):

    @staticmethod
    def get_LoisDeComportement():
        liste = [Elas_Isot]
        return liste

    """Classe des lois de comportements C de (Sigma = C * Epsilon)
    (Elas_isot, ...)
    """
    def __init__(self, dim: int, C: np.ndarray, S: np.ndarray, epaisseur: float, useVoigtNotation: bool):
        
        self.__dim = dim
        """dimension lié a la loi de comportement"""

        if dim == 2:
            assert epaisseur > 0 , "Doit être supérieur à 0"
            self.__epaisseur = epaisseur
        
        self.__useVoigtNotation = useVoigtNotation
        """Notation de voigt True or False"""
        
        self.__C = C
        """Loi de comportement pour la loi de Lamé en voigt"""

        self.__S = S
        """Loi de comportement pour la loi de Hooke en voigt"""
    
    def __get_useVoigtNotation(self):
        return self.__useVoigtNotation
    useVoigtNotation = property(__get_useVoigtNotation)

    def __get_coef_notation(self):
        if self.__useVoigtNotation:
            return 2
        else:
            return np.sqrt(2)
    coef = property(__get_coef_notation)
    """Coef lié à la notation utilisé voigt=2 kelvinMandel=racine(2)"""

    def get_C(self):
        """Renvoie une copie de la loi de comportement pour la loi de Lamé"""        
        return self.__C.copy()

    def get_S(self):
        """Renvoie une copie de la loi de comportement pour la loi de Hooke"""
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

    def AppliqueCoefSurBrigi(self, B_rigi_e_pg: np.ndarray):

        # Si on a pas une notation de voigt on doit divisier les lignes 3(2D) et [4,5,6](3D)
        if self.__useVoigtNotation:
            return

        if self.__dim == 2:
            coord=2
        elif self.__dim == 3:
            coord=[3,4,5]
        else:
            raise "Pas implémenté"

        B_rigi_e_pg[:,:,coord,:] = B_rigi_e_pg[:,:,coord,:]/self.coef

        return B_rigi_e_pg

    @staticmethod
    def ToKelvinMandelNotation(dim: int, voigtMatrice: np.ndarray):

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

        mandelMatrice = voigtMatrice*transform

        return mandelMatrice

class Elas_Isot(LoiDeComportement):   

    def __init__(self, dim: int, E=210000.0, v=0.3, contraintesPlanes=True, epaisseur=1.0,
    useVoigtNotation=False):
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

        C, S = self.__Comportement_Elas_Isot(useVoigtNotation)

        LoiDeComportement.__init__(self, dim, C, S, epaisseur, useVoigtNotation)

    def get_lambda(self):

        E=self.E
        v=self.v
        
        l = v*E/((1+v)*(1-2*v))        

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

    def __Comportement_Elas_Isot(self, useVoigtNotation: bool):
        """"Construit les matrices de comportement"""

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
        
        if useVoigtNotation:
            c = cVoigt
        else:
            # To kelvin mandel's notation
            c = LoiDeComportement.ToKelvinMandelNotation(dim, cVoigt)

        return c, np.linalg.inv(c)


class PhaseFieldModel:

    @staticmethod
    def get_splits():
        __splits = ["Bourdin","Amor","Miehe"]
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

    def get_g_e_pg(self, d_n: np.ndarray, mesh: Mesh, k_residu=1e-10):
        """Fonction de dégradation en energies / contraintes

        Args:
            d_n (np.ndarray): Endomagement localisé aux noeuds (Nn,1)
            mesh (Mesh): maillage
        """
        d_e_n = mesh.Localise_e(d_n)
        Nd_pg = np.array(mesh.N_mass_pg)

        d_e_pg = np.einsum('pij,ej->ep', Nd_pg, d_e_n, optimize=True)        

        if self.__regularization in ["AT1","AT2"]:
            g_e_pg = (1-d_e_pg)**2 + k_residu
        else:
            raise "Pas implémenté"

        assert mesh.Ne == g_e_pg.shape[0]
        assert mesh.nPg == g_e_pg.shape[1]
        
        return g_e_pg

    def __resume(self):
        resum = f'\ncomportement : {self.__loiDeComportement.nom}'
        resum += f'\nsplit : {self.__split}'
        resum += f'\nregularisation : {self.__regularization}\n'
        return resum
    resume = property(__resume)

    def __get_useVoigtNotation(self):
        return self.__loiDeComportement.useVoigtNotation
    useVoigtNotation = property(__get_useVoigtNotation)
    
    def __get_loiDeComportement(self):
        return self.__loiDeComportement
    loiDeComportement = property(__get_loiDeComportement)

    def __init__(self, loiDeComportement: LoiDeComportement,split: str, regularization: str, Gc: float, l_0: float, verbosity=False):
        """Création d'un objet comportement Phase Field

            Parameters
            ----------
            loiDeComportement : LoiDeComportement
                Loi de comportement du matériau ["Elas_Isot"]
            split : str
                Split de la densité d'energie elastique ["Bourdin","Amor","Miehe"]
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
        """Split de la densité d'energie elastique ["Bourdin","Amor","Miehe"]"""
        
        assert regularization in PhaseFieldModel.get_regularisations(), f"Doit être compris dans {PhaseFieldModel.get_regularisations()}"
        self.__regularization = regularization
        """Modèle de régularisation de la fissure ["AT1","AT2"]"""

        assert Gc > 0, "Doit être supérieur à 0" 
        self.__Gc = Gc
        """Taux de libération d'énergie critque [J/m^2]"""

        assert l_0 > 0, "Doit être supérieur à 0"
        self.__l0 = l_0
        """Largeur de régularisation de la fissure"""
            
    def Calc_Psi_e_pg(self, Epsilon_e_pg: np.ndarray):
        """Calcul de la densité d'energie elastique\n
        Ici on va caluler PsiP_e_pg = 1/2 SigmaP_e_pg : Epsilon_e_pg et PsiM_e_pg = 1/2 SigmaM_e_pg : Epsilon_e_pg\n
        Avec SigmaP_e_pg = CP * Epsilon_e_pg et SigmaM_e_pg = CM * Epsilon_e_pg

        Args:
            Epsilon_e_pg (np.ndarray): Deformation (Ne,pg,(3 ou 6))
            g_e_pg (list, optional): Fonction d'endomagement enégétique (Ne, pg) . Defaults to [].
        """
        
        # Data
        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]

        SigmaP_e_pg, SigmaM_e_pg = self.Calc_Sigma_e_pg(Epsilon_e_pg)

        PsiP_e_pg = 1/2 * np.einsum('epi,epi->ep', SigmaP_e_pg, Epsilon_e_pg, optimize=True).reshape((Ne, nPg))
        PsiM_e_pg = 1/2 * np.einsum('epi,epi->ep', SigmaM_e_pg, Epsilon_e_pg, optimize=True).reshape((Ne, nPg))
        
        return PsiP_e_pg, PsiM_e_pg

    def Calc_Sigma_e_pg(self, Epsilon_e_pg: np.ndarray):
        """Calcul la contrainte en fonction de la deformation et du split
        Ici on calcul SigmaP_e_pg = CP * Epsilon_e_pg et SigmaM_e_pg = CM * Epsilon_e_pg

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
    
    def Calc_C(self, Epsilon_e_pg: np.ndarray):
        """Calcul la loi de comportement en fonction du split

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            deformations stockées aux élements et points de gauss (Pas utilisé si bourdin)

        Returns
        -------
        np.ndarray
            Revoie cP_e_pg, cM_e_pg
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

            cP_e_pg, cM_e_pg = self.__AmorSplit(Epsilon_e_pg)
            
        elif self.__split == "Miehe":

            cP_e_pg, cM_e_pg = self.__MieheSplit(Epsilon_e_pg)

        return cP_e_pg, cM_e_pg

    def __AmorSplit(self, Epsilon_e_pg: np.ndarray):

        assert isinstance(self.__loiDeComportement, Elas_Isot), f"Implémenté que pour un matériau Elas_Isot"

        loiDeComportement = self.__loiDeComportement
        
        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]

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
        if loiDeComportement.useVoigtNotation:
            Pdev = np.eye(taille) + np.diagflat(Ivoigt) - IxI
            partieDeviateur = mu*Pdev            
        else:
            Pdev = np.eye(taille) - 1/dim * IxI
            partieDeviateur = 2*mu*Pdev
        
        # projetcteur spherique
        spherP = np.einsum('ep,ij->epij', Rp_e_pg, IxI, optimize=True)
        spherM = np.einsum('ep,ij->epij', Rm_e_pg, IxI, optimize=True)
       
        cP_e_pg = bulk*spherP + partieDeviateur
        cM_e_pg = bulk*spherM        

        return cP_e_pg, cM_e_pg    

    def __Rp_Rm(self, Epsilon_e_pg: np.ndarray):

        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]

        dim = self.__loiDeComportement.dim

        tr_Eps = np.zeros((Ne, nPg))

        tr_Eps = Epsilon_e_pg[:,:,0] + Epsilon_e_pg[:,:,1]

        if dim == 3:
            tr_Eps += Epsilon_e_pg[:,:,2]

        Rp_e_pg = (1+np.sign(tr_Eps))/2
        Rm_e_pg = (1+np.sign(-tr_Eps))/2

        return Rp_e_pg, Rm_e_pg
    
    def __MieheSplit(self, Epsilon_e_pg: np.ndarray):

        assert isinstance(self.__loiDeComportement, Elas_Isot), f"Implémenté que pour un matériau Elas_Isot"

        dim = self.__loiDeComportement.dim
        assert dim == 2, "Implémenté que en 2D"

        Ne = Epsilon_e_pg.shape[0]
        nPG = Epsilon_e_pg.shape[1]
        coef = self.__loiDeComportement.coef

        # Reconsruit le tenseur des deformations
        Eps_e_pg = np.zeros((Ne,nPG,2,2))
        Eps_e_pg[:,:,0,0] = Epsilon_e_pg[:,:,0]
        Eps_e_pg[:,:,1,1] = Epsilon_e_pg[:,:,1]
        Eps_e_pg[:,:,0,1] = Epsilon_e_pg[:,:,2]/coef; Eps_e_pg[:,:,1,0] = Epsilon_e_pg[:,:,2]/coef
        
        # invariants du tenseur des deformations
        tr_Eps = np.trace(Eps_e_pg, axis1=2, axis2=3)
        det_Eps = np.linalg.det(Eps_e_pg)
        if np.linalg.det(Eps_e_pg[0,0]) > 0:
            verif1 = (det_Eps[0,0] - np.linalg.det(Eps_e_pg[0,0]))/np.linalg.det(Eps_e_pg[0,0]) 
            assert verif1 < 1e-12

        # Calculs des valeurs propres
        Eps1 = (tr_Eps-np.sqrt(tr_Eps**2-(4*det_Eps)))/2
        Eps2 = (tr_Eps+np.sqrt(tr_Eps**2-(4*det_Eps)))/2

        Eps2I= np.einsum('ep,ij->epij', Eps2, np.eye(2), optimize=True)

        Eps1_m_Eps2 = Eps1 - Eps2
        Eps_m_Eps2I = Eps_e_pg - Eps2I
        
        # identifications des lignes et colonnes ou Eps1 != Eps2
        lignes, colonnes = np.where(Eps1_m_Eps2 != 0)

        # construction des bases propres m1 et m2
        m1 = np.zeros((Ne,nPG,2,2))
        m1[:,:,0,0] = 1
        if lignes.size > 0:
            m1_tot = np.einsum('epij,ep->epij', Eps_m_Eps2I, 1/Eps1_m_Eps2, optimize=True)
            m1[lignes, colonnes] = m1_tot[lignes, colonnes]            
        m2 = np.eye(2) - m1

        # # test ortho entre M1 et M2
        # verifOrthoM1M2 = np.einsum('epij,epij->', m1, m2, optimize=True)
        # assert np.linalg.norm(verifOrthoM1M2) < 1e-12, "Orthogonalité entre M1 et M2 non vérifié"
        
        # Passage des bases propres sous la forme de voigt 
        m1Voigt = np.zeros((Ne,nPG,3)); m2Voigt = np.zeros((Ne,nPG,3))
        m1Voigt[:,:,0] = m1[:,:,0,0];   m2Voigt[:,:,0] = m2[:,:,0,0]
        m1Voigt[:,:,1] = m1[:,:,1,1];   m2Voigt[:,:,1] = m2[:,:,1,1]
        # m1Voigt[:,:,2] = m1[:,:,0,1]*2/coef;   m2Voigt[:,:,2] = m2[:,:,0,1]*2/coef
        m1Voigt[:,:,2] = m1[:,:,0,1];   m2Voigt[:,:,2] = m2[:,:,0,1]

        # Récupération des parties positives et négatives des valeurs propres
        Eps1P = (Eps1+np.abs(Eps1))/2;  Eps2P = (Eps2+np.abs(Eps2))/2
        Eps1M = (Eps1-np.abs(Eps1))/2;  Eps2M = (Eps2-np.abs(Eps2))/2        

        # Calcul de Beta Plus
        BetaP = np.ones((Ne,nPG))*0.5        
        BetaP[lignes, colonnes] = (Eps1P[lignes, colonnes]-Eps2P[lignes, colonnes])/Eps1_m_Eps2[lignes, colonnes]

        # Calcul de Beta Moin
        BetaM = np.ones((Ne,nPG))*0.5        
        BetaM[lignes, colonnes] = (Eps1M[lignes, colonnes]-Eps2M[lignes, colonnes])/Eps1_m_Eps2[lignes, colonnes]

        # Calcul de di
        d1P = np.heaviside(Eps1,0.5);  d2P = np.heaviside(Eps2,0.5)
        d1M = np.heaviside(-Eps1,0.5);  d2M = np.heaviside(-Eps2,0.5)

        # Calcul de gamma
        gamma1P = d1P - BetaP;  gamma2P = d2P - BetaP
        gamma1M = d1M - BetaM;  gamma2M = d2M - BetaM

        # Calcul de mixmi        
        m1xm1 = np.einsum('epi,epj->epij', m1Voigt, m1Voigt, optimize=True)
        m2xm2 = np.einsum('epi,epj->epij', m2Voigt, m2Voigt, optimize=True)
        
        matriceI = np.eye(3)
        if self.__loiDeComportement.useVoigtNotation:
            matriceI[2,2] = matriceI[2,2]/2

        # Projecteur P tel que EpsP = projP : Eps
        BetaP_x_matriceI = np.einsum('ep,ij->epij', BetaP, matriceI, optimize=True)
        gamma1P_x_m1xm1 = np.einsum('ep,epij->epij', gamma1P, m1xm1, optimize=True)
        gamma2P_x_m2xm2 = np.einsum('ep,epij->epij', gamma2P, m2xm2, optimize=True)
        projP = BetaP_x_matriceI + gamma1P_x_m1xm1 + gamma2P_x_m2xm2

        # Projecteur M tel que EpsM = projM : Eps
        BetaM_x_matriceI = np.einsum('ep,ij->epij', BetaM, matriceI, optimize=True)
        gamma1M_x_m1xm1 = np.einsum('ep,epij->epij', gamma1M, m1xm1, optimize=True)
        gamma2M_x_m2xm2 = np.einsum('ep,epij->epij', gamma2M, m2xm2, optimize=True)
        projM = BetaM_x_matriceI + gamma1M_x_m1xm1 + gamma2M_x_m2xm2

        # Calcul Rp et Rm
        Rp_e_pg, Rm_e_pg = self.__Rp_Rm(Epsilon_e_pg)
        
        # Calcul IxI
        Ivoigt = np.array([1,1,0]).reshape((3,1))
        IxI = Ivoigt.dot(Ivoigt.T)

        # Calcul partie sphérique
        spherP = np.einsum('ep,ij->epij', Rp_e_pg, IxI, optimize=True)
        spherM = np.einsum('ep,ij->epij', Rm_e_pg, IxI, optimize=True)

        # # test partie sphérique
        # spher = np.einsum('ep,ij->epij',tr_Eps, np.eye(2), optimize=True)
        # spherv = np.zeros((Ne,nPG,3))
        # spherv[:,:,0] = spher[:,:,0,0]
        # spherv[:,:,1] = spher[:,:,1,1]
        # spherTest = np.einsum('epij,epj->epi', spherP+spherM, Epsilon_e_pg, optimize=True)
        # verivSpher = np.linalg.norm(spherv-spherTest)
        # assert verivSpher < 1e-12

        # Calcul de la loi de comportement
        lamb = self.__loiDeComportement.get_lambda()
        mu = self.__loiDeComportement.get_mu()

        cP_e_pg = lamb*spherP + 2*mu*projP
        cM_e_pg = lamb*spherM + 2*mu*projM

        return cP_e_pg, cM_e_pg


class Materiau:
    
    def __get_dim(self):
        return self.comportement.dim
    dim = property(__get_dim)    

    def __init__(self, comportement: LoiDeComportement, ro=8100.0, phaseFieldModel=None):
        """Creer un materiau

        Parameters
        ----------                        
        ro : float, optional
            Masse volumique en kg.m^-3
        epaisseur : float, optional
            epaisseur du matériau si en 2D > 0 !
        """
        
        assert isinstance(comportement, LoiDeComportement)

        assert ro > 0 , "Doit être supérieur à 0"
        self.ro = ro               

        # Initialisation des variables de la classe

        self.comportement = comportement
        """Comportement du matériau"""

        if isinstance(phaseFieldModel, PhaseFieldModel):
            self.phaseFieldModel = phaseFieldModel
            """Phase field model"""
        else:
            self.phaseFieldModel = None


# TEST ==============================

import unittest
import os

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
            for s in self.splits:
                for r in self.regularizations:
                    if vn:
                        pfm = PhaseFieldModel(self.comportements[0],s,r,1,1)
                    else:
                        pfm = PhaseFieldModel(self.comportements[0],s,r,1,1)
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

        # Création de 2 espilons quelconques 2D
        Epsilon_e_pg = np.random.rand(1000,1,3)
        # Epsilon_e_pg[0,0,:]=0
        # Epsilon_e_pg = np.zeros((1000,1,3))
        
        tol = 1e-12

        for pfm in self.phaseFieldModels:
            
            assert isinstance(pfm, PhaseFieldModel)

            useVoigt = pfm.useVoigtNotation

            comportement = pfm.loiDeComportement
            
            if isinstance(comportement, Elas_Isot):
                c = comportement.get_C()
            else:
                raise "Pas implémenté"            
            
            cP_e_pg, cM_e_pg = pfm.Calc_C(Epsilon_e_pg)

            # Test que cP + cM = c
            verifC = np.linalg.norm(c-(cP_e_pg+cM_e_pg))/np.linalg.norm(c)
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
            energiecPcM = np.einsum('epj,epij,epj->ep', Epsilon_e_pg, (cP_e_pg+cM_e_pg), Epsilon_e_pg, optimize=True)
            verifEnergie = np.linalg.norm(energiec-energiecPcM)/np.linalg.norm(energiec)
            if np.linalg.norm(energiec)>0:
                self.assertTrue(np.abs(verifEnergie) < tol)

if __name__ == '__main__':
    try:
        os.system("cls")
        unittest.main(verbosity=2)
    except:
        print("")