"""Interactive glTF gallery, exported by a sphinx-gallery scraper during the example run."""

import os
import shutil
from typing import Iterable, Callable
from pathlib import Path

from sphinx_gallery.py_source_parser import split_code_and_text_blocks

from EasyFEA import Folder, Mesh, IO
from EasyFEA.IO._utils import _Init_obj
from EasyFEA.Simulations import Load_Simu

docsDir = Folder.Dir(__file__, 2)
examplesDir = Folder.Join(Folder.Dir(docsDir), "examples")
sgExamplesDir = Folder.Join(docsDir, "examples")  # sphinx-gallery's gallery_dirs
modelViewerDir = Folder.Join(docsDir, "_static", "model-viewer")

# served at the html root by `html_extra_path`; survives `make clean`, like docs/examples
extraDir = Folder.Join(docsDir, "_cache", "html")
galleryDir = Folder.Join(extraDir, "gallery")
cacheModelViewerDir = Folder.Join(extraDir, "_static", "model-viewer")


class Item:
    def __init__(
        self,
        script: str,
        title: str,
        variables: list[str],
        function: Callable,
        kwargs: dict = {},
    ):
        self._script = script
        self._title = title
        self._variables = variables
        self._function = function
        self._kwargs = kwargs
        self._outputFolder = Folder.Join(galleryDir, script.replace(".py", ""))

    @property
    def htmlFile(self) -> str:
        return Folder.Join(self._outputFolder, f"{Path(self._outputFolder).stem}.html")

    def export(self, dict_globals: dict) -> None:
        if Folder.Exists(self._outputFolder):
            shutil.rmtree(self._outputFolder)

        self._function(self, dict_globals, self._variables, self._kwargs)

        if "results" in self._kwargs:
            defaultResult = self._kwargs["results"][0]
        else:
            defaultResult = None

        IO.GLTF.Create_html(
            path=self._outputFolder,
            modelViewerDir=cacheModelViewerDir,
            defaultResult=defaultResult,
            allowModelSelectorButton=True,
            allowAninationButton=True,
            allowColorbar=True,
        )


def PlotMesh(
    item: Item, dict_globals: dict[str], variables: list[str], kwargs: dict
) -> str:
    mesh = dict_globals[variables[0]]
    IO.GLTF.Save_mesh(mesh, folder=item._outputFolder, plotMesh=True, **kwargs)


def PlotMeshQuality(
    item: Item, dict_globals: dict[str], variables: list[str], kwargs: dict
) -> str:
    mesh: Mesh = dict_globals[variables[0]]
    IO.GLTF.Save_mesh(
        mesh,
        folder=item._outputFolder,
        list_nodesValues_n=[mesh.Get_Quality(nodeValues=True)],
        plotMesh=True,
        cmap="viridis",
    )


def PlotSimu(
    item: Item, dict_globals: dict[str], variables: list[str], kwargs: dict
) -> str:
    variable = dict_globals[variables[0]]

    if isinstance(variable, Iterable):
        simu = Load_Simu(variable[0])
    else:
        simu, mesh, _, _ = _Init_obj(variable)

    if simu.Niter == 0:
        simu.Save_Iter()

    if "fps" not in kwargs.keys():
        kwargs["fps"] = 10

    if "N" not in kwargs.keys():
        kwargs["N"] = 20

    IO.GLTF.Save_simu(simu, folder=item._outputFolder, **kwargs)


def PlotOptimTopo(
    item: Item, dict_globals: dict[str], variables: list[str], kwargs
) -> str:

    mesh: Mesh = dict_globals[variables[0]]
    list_p_e = dict_globals[variables[1]]

    list_nodesValues_n = [mesh.Get_Node_Values(p_e) for p_e in list_p_e]

    IO.GLTF.Save_mesh(
        mesh,
        folder=item._outputFolder,
        list_nodesValues_n=list_nodesValues_n,
        cmap="binary",
        **kwargs,
    )


def PlotHomog(item: Item, dict_globals: dict[str], variables: list[str], kwargs) -> str:

    simu = dict_globals[variables[0]]

    N = simu.Niter + 1

    IO.GLTF.Save_simu(
        simu,
        item._outputFolder,
        results=["displacement"],
        deformFactor=0.3,
        plotMesh=True,
        N=N,
        fps=1,
    )


useMesh = {"plotMesh": True}

list_item = [
    Item(
        "PhaseField/Shear.py",
        "Damage simulation for a plate subjected to shear.",
        ["simu"],
        PlotSimu,
        {
            "results": ["damage", "Svm", "displacement"],
            "deformFactor": 2,
            **useMesh,
        },
    ),
    Item(
        "PhaseField/CT.py",
        "Performs damage simulation on a CT specimen.",
        ["simu"],
        PlotSimu,
        {
            "results": ["damage", "displacement"],
            "deformFactor": 2,
            **useMesh,
        },
    ),
    Item(
        "WeakForms/TopologyOptimisation1.py",
        "An educational implementation of topology optimization.",
        ["mesh", "list_p_e"],
        PlotOptimTopo,
    ),
    Item(
        "CardiacElastoDynamics/MonoVentricular.py",
        "Passive + active hyperelastic simulation of an ellipsoidal left-ventricle model.",
        ["simu"],
        PlotSimu,
        {
            "results": ["displacement"],
            "deformFactor": 1,
            **useMesh,
            "fps": 5,
        },
    ),
    Item(
        "Contact/Contact3.py",
        "Frictionless contact between a thin elastic arch strip and a rigid block.",
        ["simu"],
        PlotSimu,
        {
            "results": ["displacement"],
            "deformFactor": 1,
            **useMesh,
            "fps": 5,
        },
    ),
    Item(
        "LinearizedElasticity/Homog1.py",
        "Conduct homogenization using an example outlined in Computational Homogenization of Heterogeneous Materials with Finite Elements.",
        ["simu"],
        PlotHomog,
    ),
    Item(
        "LinearizedElasticity/Homog4.py",
        "Conduct 3d homogenization on a simple RVE.",
        ["simu"],
        PlotHomog,
    ),
    Item(
        "LinearizedElasticity/Homog5.py",
        "Conduct 3d homogenization on a periodic mesh generated with microgen",
        ["simu"],
        PlotHomog,
    ),
    Item(
        "LinearizedElasticity/MeshOptim1.py",
        "Mesh optimization using the ZZ1 criterion for a bending bracket.",
        ["simu"],
        PlotSimu,
        {
            "results": ["displacement", "Svm"],
            "deformFactor": 1500,
            **useMesh,
            "fps": 5,
        },
    ),
    Item(
        "LinearizedElasticity/Elas4.py",
        "Plate with a hole subjected to uniform tensile loading.",
        ["simu"],
        PlotSimu,
        {
            "results": ["displacement", "Svm"],
            **useMesh,
            "deformFactor": 200,
        },
    ),
    Item(
        "LinearizedElasticity/Elas7.py",
        "Control lever for a molding machine used to blow plastic bottles.",
        ["simu"],
        PlotSimu,
        {
            "results": ["Svm", "displacement"],
            **useMesh,
            "deformFactor": 200,
            "fps": 6,
            "N": 6,
        },
    ),
    Item(
        "LinearizedElasticity/Elas10.py",
        "Wave propagation.",
        ["simu"],
        PlotSimu,
        {"results": ["displacement", "speed", "accel"]},
    ),
    Item(
        "Hyperelasticity/Hyperelas3.py",
        "A L shape part undergoing bending deformation.",
        ["simu"],
        PlotSimu,
        {"results": ["displacement"], **useMesh},
    ),
    Item(
        "Hyperelasticity/Hyperelas4.py",
        "A cantilever beam undergoing bending deformation in dynamic.",
        ["simu"],
        PlotSimu,
        {"results": ["displacement"], **useMesh},
    ),
    Item(
        "Meshes/Mesh5_2D.py",
        "Mesh of a 2D cracked part.",
        ["simu"],
        PlotSimu,
        {"results": ["displacement"], **useMesh},
    ),
    Item(
        "Meshes/Mesh5_3D.py",
        "Mesh of a 3D cracked part.",
        ["simu"],
        PlotSimu,
        {"results": ["displacement"], **useMesh},
    ),
    Item(
        "Thermal/Thermal2.py",
        "Transient thermal simulation.",
        ["simu"],
        PlotSimu,
        {"results": ["thermal"], **useMesh},
    ),
    Item("Meshes/Mesh6_3D.py", "Refined 3D mesh.", ["mesh"], PlotMesh),
    Item(
        "Meshes/Mesh8.py",
        "Meshing of a grooved 3D part with calculation of element quality.",
        ["mesh"],
        PlotMeshQuality,
    ),
    Item(
        "Meshes/Mesh10.py",
        "Simplified turbine mesh with data extraction in matlab.",
        ["mesh"],
        PlotMesh,
    ),
]

dict_item = {item._script: item for item in list_item}


class GLTFScraper:
    """Exports a gallery item's glTF from its example globals, after the last code block."""

    def __call__(self, block, block_vars, gallery_conf) -> str:
        srcFile = block_vars["src_file"]
        script = Path(os.path.relpath(srcFile, examplesDir)).as_posix()
        item = dict_item.get(script)
        if item is None:
            return ""
        _, blocks = split_code_and_text_blocks(srcFile)
        lastBlock = [b for b in blocks if b.type == "code"][-1]
        if block.lineno == lastBlock.lineno:
            item.export(block_vars["example_globals"])
        return ""

    def __repr__(self):
        return "GLTFScraper"


def Prepare() -> None:
    """Writes index.md, and forces sphinx-gallery to rerun any item whose export is missing."""

    shutil.copytree(modelViewerDir, cacheModelViewerDir, dirs_exist_ok=True)

    for item in list_item:
        md5File = Folder.Join(sgExamplesDir, f"{item._script}.md5")
        if not Folder.Exists(item.htmlFile) and Folder.Exists(md5File):
            os.remove(md5File)

    Write_index()


def Write_index() -> None:
    indexFile = Folder.Join(Folder.Dir(__file__), "index.md", mkdir=True)

    content = """
    ```{eval-rst}
    :html_theme.sidebar_secondary.remove:
    ```

    # Gallery

    This page shows interactive scenes from simulations made with **EasyFEA**.

    For additional details, please refer to the comprehensive collection of {doc}`examples <../examples/index>`.

    ```{raw} html

    <script>document.body.classList.add('gallery');</script>

    <div class="gallery-grid">
"""

    for item in list_item:
        htmlPath = os.path.relpath(item.htmlFile, galleryDir)
        scriptPath = os.path.relpath(
            Folder.Join(extraDir, "examples", item._script.replace(".py", ".html")),
            galleryDir,
        )
        content += f"""
        <div class="gallery-item">
            <iframe src="{htmlPath}"></iframe>
            <p><em><a href="{scriptPath}">{Path(scriptPath).stem}</a>: {item._title}</em></p>
        </div>
    """

    content += """

    </div>
    ```
"""

    IO.GLTF._write_file(indexFile, content)
