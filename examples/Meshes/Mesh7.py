"""Meshing of a grooved 3D part with calculation of element quality."""

from EasyFEA import (Display, Folder, np,
                     Mesher, ElemType, gmsh,
                     PyVista_Interface as pvi,)
from EasyFEA.Geoms import Point, Circle, Points

folder = Folder.Get_Path(__file__)

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Geom
    # ----------------------------------------------
    R = 10
    e = 2
    r = R-e
    h = R * 2/3

    meshSize = e/3

    center = Point()

    circle_ext = Circle(center, R*2, meshSize)
    circle_int = Circle(center, r*2, meshSize, True)

    useFillet = circle_int.isHollow
    addCylinder = True
    addBox = True
    addRevolve = True

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    mesher = Mesher(False, False)
    # dim, elemType = 2, ElemType.TRI3
    dim, elemType = 3, ElemType.TETRA4
    
    factory = mesher._factory

    surfaces = mesher._Surfaces(circle_ext, [circle_int])[0]
    mesher._Extrude(surfaces, [0,0,R], elemType)
    vol1 = factory.getEntities(3)    

    mesher._Synchronize()    

    if useFillet:
        surfs = gmsh.model.getBoundary(vol1)
        lines = gmsh.model.getBoundary(surfs, False)        
        gmsh.model.occ.fillet([vol1[0][1]], [abs(line[1]) for line in lines], [e/3])
    
    if addCylinder:
        cylinder = factory.addCylinder(R+e,0,h,-2*R-e,0,0,e)
        factory.cut(vol1, [(3, cylinder)])

    if addBox:
        box = factory.addBox(-R/4,-R-e,R/2, R/2, 2*R+e, R/2)
        factory.cut(factory.getEntities(3), [(3, box)])

    if addRevolve:
        p1 = Point(R-e/2, 0, e, r=e/4)
        p2 = Point(R, 0, e)
        p3 = Point(R,0, e*4)
        p4 = Point(R-e/2, 0, e*4, r=e/4)
        contour = Points([p1, p2, p3, p4])    
        surf = mesher._Surfaces(contour, [])[0][0]

        rev1 = factory.revolve([(2, surf)], 0,0,0,0,0,R,np.pi)
        rev2 = factory.revolve([(2, surf)], 0,0,0,0,0,R,-np.pi)
        factory.cut(factory.getEntities(3), [rev1[1], rev2[1]])
    
    mesher._Synchronize()

    if meshSize > 0:
        mesher.Set_meshSize(meshSize)

    mesher._Set_PhysicalGroups(setPoints=False, setLines=True, setSurfaces=True, setVolumes=False)
    
    mesher._Meshing(dim, elemType)

    mesh = mesher._Construct_Mesh()

    # ----------------------------------------------
    # Plot
    # ----------------------------------------------

    print(mesh)

    if dim == 3:
        print(f'volume = {mesh.volume:.3f}')

    plotter = pvi._Plotter(shape=(1,2))

    pvi.Plot_Mesh(mesh, plotter=plotter)

    plotter.subplot(0,1)
    plotter.add_title('aspect ratio')
    qual = mesh.Get_Quality()
    pvi.Plot_Elements(mesh, dimElem=1, plotter=plotter, color='k')
    pvi.Plot(mesh, qual, nodeValues=False, cmap='viridis', clim=(0,1), show_edges=True, plotter=plotter).show()