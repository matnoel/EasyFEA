"""Meshing of a part designed by cad software."""

from EasyFEA import Display, Folder, Mesher, ElemType

folder = Folder.Get_Path(__file__) # Meshes folder
examples_folder = Folder.Get_Path(folder)

stp = Folder.Join(examples_folder, "_parts", "beam.stp")
mesh_stp = Mesher().Mesh_Import_part(stp, 3, 13/5, ElemType.HEXA8)

igs = Folder.Join(examples_folder, "_parts", "beam.igs")
mesh_igs = Mesher().Mesh_Import_part(igs, 3, 13/5, ElemType.TETRA10)
# An igs file will provide a contour mesh only.

Display.Plot_Mesh(mesh_stp)
Display.Plot_Mesh(mesh_igs)

Display.plt.show()