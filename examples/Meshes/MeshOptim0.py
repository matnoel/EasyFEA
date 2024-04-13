"""Optimization of a happy mesh with quality criteria."""

from EasyFEA import (Display, Folder, Tic, plt, np,
                     Mesher, ElemType, Mesh,
                     Paraview_Interface,
                     PyVista_Interface as pvi)
from EasyFEA.Geoms import Point, Circle, CircleArc, Contour
from EasyFEA.fem import Mesh_Optim

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    dim = 2
    criteria = 'angular'
    quality = 0.8 if dim == 2 else .6 # lower bound of the target quality
    ratio = .6 # the ratio of mesh elements that must satisfy the target    
    iterMax = 20 # Maximum number of iterations
    coef = 1/2 # Scaling coefficient for the optimization process

    # Selecting the element type for the mesh
    if dim == 2:
        elemType = ElemType.TRI3 # TRI3, TRI6, TRI10, QUAD4, QUAD8
    else:
        elemType = ElemType.TETRA4 # TETRA4, TETRA10, HEXA8, HEXA20, PRISM6, PRISM15

    # Creating a folder to store the results
    folder = Folder.New_File(Folder.Join('Meshes', f'Optim{dim}D'), results=True)
    if not Folder.Exists(folder):
        Folder.os.makedirs(folder)

    # ----------------------------------------------
    # Meshing
    # ----------------------------------------------
        
    D = 1
    r = D * 1/4 
    e = 0.1
    b = D*.1

    mS = e/3 * 10

    # face
    circle = Circle(Point(), D, mS)

    # eyes
    theta = 45*np.pi/180
    sin = np.sin(theta)
    cos = np.cos(theta)
    eye1 = Circle(Point(-r*sin, r*cos), e, mS)
    eye2 = Circle(Point(r*sin, r*cos), e, mS)

    # happy smile
    s = (D * 0.6)/2
    s1 = CircleArc(Point(-s), Point(s), Point(), coef=-1, meshSize=mS)
    s2 = CircleArc(Point(s), Point(s-e), Point(s-e/2), coef=-1, meshSize=mS)
    s3 = CircleArc(Point(s-e), Point(-s+e), Point(), meshSize=mS)
    s4 = CircleArc(Point(-s+e), Point(-s), Point(-s+e/2), coef=-1, meshSize=mS)
    happy = Contour([s1, s2, s3, s4])

    inclusions = [eye1, eye2, happy]

    def DoMesh(refineGeom=None) -> Mesh:
        """Function used for mesh generation"""
        if dim == 2:
            return Mesher().Mesh_2D(circle, inclusions, elemType, [], [refineGeom])
        else:
            return Mesher().Mesh_Extrude(circle, inclusions, [0,0,b], [], elemType, [], [refineGeom])

    # Construct the initial mesh
    mesh = DoMesh()  

    mesh, ratio = Mesh_Optim(DoMesh, folder, criteria, quality, ratio)
    
    # ----------------------------------------------
    # Plot
    # ----------------------------------------------

    qual_e = mesh.Get_Quality(criteria, False)

    Display.Plot_Result(mesh, qual_e, nodeValues=False, plotMesh=True, clim=(0,quality), cmap='viridis', title=criteria)

    axHist = Display.init_Axes()

    axHist.hist(qual_e, 11, (0,1))
    axHist.set_xlabel("quality")
    axHist.set_ylabel("elements")
    axHist.set_title(f'ratio = {ratio*100:.3f} %')
    axHist.vlines([quality], [0], [ratio*mesh.Ne], color='red')

    plt.show()