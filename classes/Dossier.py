import os

def GetPath(filename=None): 
    
    if filename == None:
        # Renvoie le path vers PythonEF
        path = os.path.dirname(__file__)
        path = os.path.dirname(path)
    else:
        # Renvoie le path vers le fichier
        path = os.path.dirname(filename)    

    return path

def GetFile(file: str, filenameWithExtension: str):
    path = GetPath(file)
    filename = path + "\\" + filenameWithExtension
    return filename


