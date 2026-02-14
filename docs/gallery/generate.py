import runpy
import os
from typing import Iterable

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


def main(list_config: list[dict], replace=False):

    list_htmlFile: list[str] = []

    for script, variable, kwargs, _ in list_config:
        scriptPath = Folder.Join(examplesDir, script)

        outputFolder = Folder.Join(galleryDir, script.replace(".py", ""))

        if replace and Folder.Exists(outputFolder):
            continue

        # run script and get variable
        dict_globals = runpy.run_path(scriptPath, run_name="__main__")
        variable = dict_globals[variable]

        if isinstance(variable, Iterable):
            simu = Load_Simu(variable[0])
            GLTF.Save_simu(simu, folder=outputFolder, openWebBrowser=False, **kwargs)
        else:

            simu, mesh, _, _ = _Init_obj(variable)

            if simu is None:
                GLTF.Save_mesh(mesh, folder=outputFolder, **kwargs)
            else:
                GLTF.Save_simu(simu, folder=outputFolder, **kwargs)

        htmlFile, _ = GLTF.Create_html(
            outputFolder,
            modelViewerDir,
            allowModelSelector=False,
            allowAnination=False,
            allowColorbar=False,
        )
        list_htmlFile.append(htmlFile)

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

    for htmlFile, (script, _, _, title) in zip(list_htmlFile, list_config):

        htmlPath = Folder.os.path.relpath(htmlFile, galleryDir)
        scriptPath = Folder.Join(htmlDir, "examples", script)
        scriptPath = Folder.os.path.relpath(
            scriptPath.replace(".py", ".html"), galleryDir
        )
        content += f"""
        <div class="gallery-item">
            <iframe src="{htmlPath}"></iframe>
            <p><em>{title} (<a href="{scriptPath}">source</a>).</em></p>
        </div>
    """

    content += """

    </div>
    ```
"""

    GLTF._write_file(indexFile, content)


if __name__ == "__main__":

    smallAnimation = {
        "N": 20,
        "fps": 10,
    }

    useMesh = {"plotMesh": True}

    # script, variable, kwargs, title
    list_config = [
        (
            "PhaseField/Shear.py",
            "list_folder",
            {"results": ["damage"], "deformFactor": 2, **useMesh, **smallAnimation},
            "Damage simulation for a plate subjected to shear.",
        ),
        (
            "Meshes/Mesh10.py",
            "mesh",
            useMesh,
            "Simplified turbine mesh with data extraction in matlab.",
        ),
    ]

    main(list_config, replace=False)
