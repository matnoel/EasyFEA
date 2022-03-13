import os

def GetPath(file: str):
    """Renvoie la path du fichier renseigné

    Args:
        file (str): fichier source

    Returns:
        path (str): Emplacement du fichier
    """
    path = os.path.dirname(file)
    return path

def GetFile(file: str, filenameWithExtension: str):
    path = GetPath(file)
    filename = path + "\\" + filenameWithExtension
    return filename


