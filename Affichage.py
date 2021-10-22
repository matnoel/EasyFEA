from typing import cast
from matplotlib.colors import Colormap
import numpy as np

import matplotlib as plt
import matplotlib.collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from class_Noeud import Noeud
from class_Element import Element
from class_Mesh import Mesh
from class_Materiau import Materiau

class Affichage:

    def PlotResult(mesh: Mesh, resultats: dict, val: str , deformation=False, facteurDef=4, affichageMaillage=False):     
        # Va chercher les valeurs 0 a affciher

        valeurs = np.array(resultats[val])
        
        dim = mesh.get_dim()

        if deformation:
                coordo = mesh.coordo + (resultats["deplacementCoordo"]*facteurDef)
        else:
            coordo = mesh.coordo

        connectPolygon = mesh.get_connectPolygon()

        if dim == 2:
            # Construit les vertices
            coord_xy = coordo[:,[0,1]]            
            vertices = [[coord_xy[connectPolygon[ix][iy]] for iy in range(len(connectPolygon[0]))] for ix in range(len(connectPolygon))]

            fig, ax = plt.subplots()

            # Trace le maillage
            if affichageMaillage:
                pc = matplotlib.collections.LineCollection(vertices, edgecolor='black', lw=0.5)
                ax.add_collection(pc)

            # Valeurs aux element
            if "_e" in val and mesh.Ne == len(valeurs):                
                pc = matplotlib.collections.PolyCollection(vertices, lw=0.5, cmap='jet')                
                pc.set_clim(valeurs.min(), valeurs.max())
                pc.set_array(valeurs)
                ax.add_collection(pc)

            # Valeur aux noeuds
            elif "_n" in val and mesh.Nn == len(valeurs):
                pc = ax.tricontourf(coordo[:,0], coordo[:,1], mesh.get_connectTriangle(), valeurs, cmap='jet', antialiased=True)
                # pc = ax.tricontour(coordo[:,0], coordo[:,1], mesh.get_connectTriangle(), valeurs, cmap='jet', antialiased=True)                
  
            fig.colorbar(pc, ax=ax)
            ax.axis('equal')
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')

        
        elif mesh.get_dim() == 3:

            assert "_e" in val, "Pour une étude 3D on ne trace qua partir des valeurs de l'élément"


            # Construit les vertices
            coord_xyz = coordo            
            vertices = [[coord_xyz[connectPolygon[ix][iy]] for iy in range(len(connectPolygon[0]))] for ix in range(len(connectPolygon))]
            
            fig = plt.figure()            
            ax = fig.add_subplot(projection="3d")
            
            # Trace le maillage
            if affichageMaillage:                
                pc = Poly3DCollection(vertices, edgecolor='black', linewidths=0.5, cmap='jet')
            else:
                pc = Poly3DCollection(vertices, cmap='jet')
            ax.add_collection3d(pc)

            # Construit le vecteur qui contient
            valeursAuFaces = []
            for e in mesh.elements:
                e = cast(Element, e)
                for i in range(e.get_nbFaces()):
                    valeursAuFaces.append(valeurs[e.id])

            valeursAuFaces = np.array(valeursAuFaces)
            # ax.scatter(coordo[:,0],coordo[:,1],coordo[:,2], linewidth=0, alpha=0)
            pc.set_clim(valeursAuFaces.min(), valeursAuFaces.max())
            pc.set_array(valeursAuFaces)

            fig.colorbar(pc, ax=ax)       
            ax.add_collection(pc)            
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_zlabel("z [mm]")            
            
            Affichage.__ChangeEchelle(ax, coordo)
                
        unite = ""
        if "S" in val:
            unite = " en Mpa"
        if "d" in val:
            unite = " en mm"
        ax.set_title(val+unite)
        
        
    def PlotMesh(mesh: Mesh ,resultats: dict, facteurDef=4, deformation=False):
        """Dessine le maillage de la simulation
        """
        
        assert facteurDef >= 1, "Le facteur de deformation doit être >= 1"

        dim = mesh.get_dim()

        if deformation:
            coordo = mesh.coordo + (resultats["deplacementCoordo"]*facteurDef)
        else:
            coordo = mesh.coordo

        connectPolygon = mesh.get_connectPolygon()

        # ETUDE 2D
        if dim == 2:
            
            fig, ax = plt.subplots()
            
            coord_xyNonDeforme = mesh.coordo[:,[0,1]]
            verticesNonDeforme = [[coord_xyNonDeforme[connectPolygon[ix][iy]] for iy in range(len(connectPolygon[0]))] for ix in range(len(connectPolygon))]
            
            if deformation:
                # Superpose maillage non deformé et deformé
                # Maillage non deformés
                pc = matplotlib.collections.LineCollection(verticesNonDeforme, edgecolor='black', lw=0.5)
                ax.add_collection(pc)

                # Maillage deformé
                coordoDeforme = mesh.coordo + (resultats["deplacementCoordo"]*facteurDef)
                coordo_xyDeforme = coordoDeforme[:,[0,1]]                    
                new_faces = [[coordo_xyDeforme[connectPolygon[ix][iy]] for iy in range(len(connectPolygon[0]))] for ix in range(len(connectPolygon))]
                pc = matplotlib.collections.LineCollection(new_faces, edgecolor='red', lw=0.5)
                ax.add_collection(pc)
            else:
                # Maillage non deformé
                pc = matplotlib.collections.PolyCollection(verticesNonDeforme, facecolors='c', edgecolor='black', lw=0.5)
                ax.add_collection(pc)
            
            ax.autoscale()
            ax.axis('equal')
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_title("Ne = {} et Nn = {}".format(mesh.Ne, mesh.Nn))
        
        # ETUDE 3D    
        if mesh.get_dim() == 3:
            
            fig = plt.figure()            
            ax = fig.add_subplot(projection="3d")
            
            x = mesh.coordo[:,0]
            y = mesh.coordo[:,1]
            z = mesh.coordo[:,2]

            verticesNonDeforme = [[mesh.coordo[connectPolygon[ix][iy]] for iy in range(len(connectPolygon[0]))] for ix in range(len(connectPolygon))]

            if deformation:
                # Supperpose les deux maillages
                # Maillage non deformé
                # ax.scatter(x,y,z, linewidth=0, alpha=0)
                ax.add_collection3d(Poly3DCollection(verticesNonDeforme, edgecolor='black', linewidths=0.5, alpha=0))

                # Maillage deformé
                coordoDeforme = mesh.coordo + (resultats["deplacementCoordo"]*facteurDef)
                verticesDeforme = [[coordoDeforme[connectPolygon[ix][iy]] for iy in range(len(connectPolygon[0]))] for ix in range(len(connectPolygon))]
                ax.add_collection3d(Poly3DCollection(verticesDeforme, edgecolor='red', linewidths=0.5, alpha=0))
            else:
                # ax.scatter(x,y,z, linewidth=0, alpha=0)
                ax.add_collection3d(Poly3DCollection(verticesNonDeforme, facecolors='c', edgecolor='black', linewidths=0.5, alpha=1))


            # ax.autoscale()
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_zlabel("z [mm]")
            ax.set_title("Ne = {} et Nn = {}".format(mesh.Ne, mesh.Nn))

            Affichage.__ChangeEchelle(ax, coordo)
        
        return fig, ax

    def __ChangeEchelle(ax, coordo: np.ndarray):
        """Change la taille des axes pour l'affichage 3D

        Parameters
        ----------
        ax : plt.Axes
            Axes dans lequel on va creer la figure
        """
        # Change la taille des axes
        xmin = np.min(coordo[:,0]); xmax = np.max(coordo[:,0])
        ymin = np.min(coordo[:,1]); ymax = np.max(coordo[:,1])
        zmin = np.min(coordo[:,2]); zmax = np.max(coordo[:,2])
        
        max = np.max(np.abs([xmin, xmax, ymin, ymax, zmin, zmax]))
        
        factX = np.max(np.abs([xmin, xmax]))/max
        factY = np.max(np.abs([ymin, ymax]))/max
        factZ = np.max(np.abs([zmin, zmax]))/max
        
        ecartAuBord = 5

        ax.set_xlim([xmin-ecartAuBord, xmax+ecartAuBord])
        ax.set_ylim([ymin-ecartAuBord, ymax+ecartAuBord])
        ax.set_zlim([zmin-ecartAuBord, zmax+ecartAuBord])

        ax.set_box_aspect((factX, factY, factZ))
        