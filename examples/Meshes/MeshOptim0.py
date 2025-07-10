# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
MeshOptim0
==========

Optimization of a happy mesh with quality criteria.
"""
# sphinx_gallery_thumbnail_number = 2

from EasyFEA import Display, Folder, plt, np, Mesher, ElemType, Mesh
from EasyFEA.Geoms import Point, Circle, CircleArc, Contour
from EasyFEA.fem import Mesh_Optim

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # outputs
    folder = Folder.Join(Folder.RESULTS_DIR, "Meshes", "Optim2D", mkdir=True)

    # geom
    D = 1
    r = D * 1 / 4
    e = 0.1
    b = D * 0.1

    # criteria
    criteria = "aspect"
    quality = 0.8  # lower bound of the target quality
    ratio = 0.6  # the ratio of mesh elements that must satisfy the target
    iterMax = 20  # Maximum number of iterations
    coef = 1 / 2  # Scaling coefficient for the optimization process

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    elemType = ElemType.TRI3
    mS = e / 3 * 10

    # face
    circle = Circle(Point(), D, mS)

    # eyes
    theta = 45 * np.pi / 180
    sin = np.sin(theta)
    cos = np.cos(theta)
    eye1 = Circle(Point(-r * sin, r * cos), e, mS)
    eye2 = Circle(Point(r * sin, r * cos), e, mS)

    # happy smile
    s = (D * 0.6) / 2
    s1 = CircleArc(Point(-s), Point(s), Point(), coef=-1, meshSize=mS)
    s2 = CircleArc(Point(s), Point(s - e), Point(s - e / 2), coef=-1, meshSize=mS)
    s3 = CircleArc(Point(s - e), Point(-s + e), Point(), meshSize=mS)
    s4 = CircleArc(Point(-s + e), Point(-s), Point(-s + e / 2), coef=-1, meshSize=mS)
    happy = Contour([s1, s2, s3, s4])

    inclusions = [eye1, eye2, happy]

    def DoMesh(refineGeom=None) -> Mesh:
        """Function used for mesh generation"""
        return Mesher().Mesh_2D(circle, inclusions, elemType, [], [refineGeom])

    # Construct the initial mesh
    mesh = DoMesh()
    Display.Plot_Mesh(mesh, title="first mesh")

    mesh, ratio = Mesh_Optim(DoMesh, folder, criteria, quality, ratio)

    # ----------------------------------------------
    # Plot
    # ----------------------------------------------

    qual_e = mesh.Get_Quality(criteria, False)

    Display.Plot_Mesh(mesh, title="last mesh")
    Display.Plot_Result(
        mesh,
        qual_e,
        nodeValues=False,
        plotMesh=True,
        clim=(0, quality),
        cmap="viridis",
        title=criteria,
    )

    axHist = Display.Init_Axes()

    axHist.hist(qual_e, 11, (0, 1))
    axHist.set_xlabel("quality")
    axHist.set_ylabel("elements")
    axHist.set_title(f"ratio = {ratio * 100:.3f} %")
    axHist.vlines([quality], [0], [ratio * mesh.Ne], color="red")

    plt.show()
