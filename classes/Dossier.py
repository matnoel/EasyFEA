import os

def GetPath(filename=None):
    """Renvoie le path du fichier ou renvoie le path vers PythonEF"""
    
    if filename == None:
        # Renvoie le path vers PythonEF
        path = os.path.dirname(__file__)
        path = os.path.dirname(path)
    else:
        # Renvoie le path vers le fichier
        path = os.path.dirname(filename)    

    return path

def NewFile(filenameWithExtension: str):
    """Renvoie le path vers le fichier avec l'extension"""
    path = GetPath()
    filename = path + "\\" + filenameWithExtension
    return filename


