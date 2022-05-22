import matplotlib.pyplot as plt
import matplotlib.collections
import numpy as np
import gmsh
import sys

isOpen = False

gmsh.initialize(sys.argv)

surf = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
pt1 = gmsh.model.occ.addPoint(0, 0.5, 0)
pt2 = gmsh.model.occ.addPoint(0.5, 0.5, 0)
line = gmsh.model.occ.addLine(pt1, pt2)

o, m = gmsh.model.occ.fragment([(0, pt1), (1, line)], [(2, surf)])
gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(2, [surf], 100)
gmsh.model.addPhysicalGroup(1, [line], 101)
gmsh.model.addPhysicalGroup(0, [pt1], 102)

gmsh.option.setNumber('Mesh.MeshSizeMin', 2)
gmsh.model.mesh.generate(2)

if isOpen:
    gmsh.plugin.setNumber("Crack", "PhysicalGroup", 101)
    gmsh.plugin.setNumber("Crack", "OpenBoundaryPhysicalGroup", 102)
    # gmsh.plugin.setNumber("Crack", "DebugView", 1)
    gmsh.plugin.run("Crack")

# print(gmsh.model.mesh.getNodes()) # does indeed contain 1 new node

# gmsh.fltk.run()

# Here I want to recover coordo and connect

elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements()
nodes, coord, parametricCoord = gmsh.model.mesh.getNodes()

coord = coord.reshape(-1,3)

fig, ax = plt.subplots()

for t, gmshId in enumerate(elementTypes):

    Ne = elementTags[t].shape[0]
    connect = nodeTags[t].reshape(Ne, -1)-1    
    
    nodes = np.unique(nodeTags[t]-1)

    Nmax = nodes.max()
    assert Nmax <= (coord.shape[0]-1), f"Nodes {Nmax} doesn't exist in coordo"
    
    coordo = coord[nodes]

    match gmshId:
        case 1: #SEG2
            coordFaces = coordo[connect,:2]
            color='black'
            lw=2
            label='SEG2'
        case 2: #TRI3
            connectFaces = connect[:, [0,1,2,0]]
            coordFaces = coordo[connectFaces,:2]
            color='red'
            lw=0.5
            label='TRI3'        
        case _:
            continue


    ax.scatter(coordo[:,0], coordo[:,1], label=label)

    pc = matplotlib.collections.LineCollection(coordFaces, edgecolor=color, lw=lw, label=label)
    ax.add_collection(pc)

    plt.legend()

    plt.pause(2)

gmsh.finalize()

plt.show()

