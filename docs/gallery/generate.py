import runpy
import os
from typing import Iterable, Callable
from pathlib import Path
import shutil

os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["MPLBACKEND"] = "Agg"

from EasyFEA import Folder, GLTF
from EasyFEA.Simulations._simu import _Init_obj, Load_Simu

docsDir = Folder.Dir(__file__, 2)
examplesDir = Folder.Join(Folder.Dir(docsDir), "examples")
modelViewerDir = Folder.Join(docsDir, "_static", "model-viewer")

buildDir = os.environ.get("READTHEDOCS_OUTPUT", Folder.Join(docsDir, "_build"))
htmlDir = Folder.Join(buildDir, "html")
# print(f"\n\n{htmlDir}\n\n")
galleryDir = Folder.Join(htmlDir, "gallery")


class Config:

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

    def run(self) -> str:

        scriptPath = Folder.Join(examplesDir, self._script)
        dict_globals = runpy.run_path(scriptPath, run_name="__main__")

        self._function(self, dict_globals, self._variables, self._kwargs)

        modelViewer = Folder.Join(htmlDir, "_static", "model-viewer")
        if not Folder.Exists(modelViewer):
            shutil.copytree(modelViewerDir, modelViewer)

        htmlFile, _ = GLTF.Create_html(
            self._outputFolder,
            modelViewer,
            allowModelSelectorButton=True,
            allowAninationButton=True,
            allowColorbar=True,
        )

        return htmlFile


def PlotMesh(
    config: Config, dict_globals: dict[str], variables: list[str], kwargs
) -> str:
    mesh = dict_globals[variables[0]]
    GLTF.Save_mesh(mesh, folder=config._outputFolder, plotMesh=True, **kwargs)


def PlotSimu(
    config: Config, dict_globals: dict[str], variables: list[str], kwargs
) -> str:
    variable = dict_globals[variables[0]]

    if isinstance(variable, Iterable):
        simu = Load_Simu(variable[0])
    else:
        simu, mesh, _, _ = _Init_obj(variable)

    if simu.Niter == 0:
        simu.Save_Iter()

    GLTF.Save_simu(simu, folder=config._outputFolder, N=20, fps=10, **kwargs)


def main(list_config: list[Config], replace=False):

    list_htmlFile: list[str] = []

    for config in list_config:
        if replace and Folder.Exists(config._outputFolder):
            continue
        list_htmlFile.append(config.run())

    # generate gallery index.md
    indexFile = Folder.Join(Folder.Dir(__file__), "index.md", mkdir=True)

    content = """
    ```{eval-rst}
    :html_theme.sidebar_secondary.remove:
    ```

    # Gallery

    This page collects screenshots from various simulations based on EasyFEA.

    ```{raw} html 

    <script>document.body.classList.add('gallery');</script>

    <div class="gallery-grid">
"""

    for htmlFile, config in zip(list_htmlFile, list_config):

        htmlPath = Folder.os.path.relpath(htmlFile, galleryDir)
        scriptPath = Folder.Join(htmlDir, "examples", config._script)
        scriptPath = Folder.os.path.relpath(
            scriptPath.replace(".py", ".html"), galleryDir
        )
        content += f"""
        <div class="gallery-item">
            <iframe src="{htmlPath}"></iframe>
            <p><em><a href="{scriptPath}">{Path(scriptPath).stem}</a>: {config._title}.</em></p>
        </div>
    """

    content += """

    </div>
    ```
"""

    GLTF._write_file(indexFile, content)


if __name__ == "__main__":

    useMesh = {"plotMesh": True}

    list_config = [
        Config(
            "PhaseField/Shear.py",
            "Damage simulation for a plate subjected to shear.",
            ["list_folder"],
            PlotSimu,
            {"results": ["damage", "displacement"], "deformFactor": 2, **useMesh},
        ),
        Config(
            "Hyperelasticity/Hyperelas3.py",
            "A L shape part undergoing bending deformation.",
            ["simu"],
            PlotSimu,
            {"results": ["displacement"], **useMesh},
        ),
        Config(
            "Meshes/Mesh10.py",
            "Simplified turbine mesh with data extraction in matlab.",
            ["mesh"],
            PlotMesh,
        ),
    ]

    main(list_config, replace=False)
