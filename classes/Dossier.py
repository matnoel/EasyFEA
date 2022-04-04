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

def NewFile(filename: str, path=GetPath(), results=False):
    """Renvoie le path vers le fichier avec l'extension ou non
    
    exemple.toto

    if results:
        filename = path\\results\\exemple.toto
    else:
        filename = path\\exemple.toto
        

    """
    path = path

    if results:
        filename = path + "\\results\\" + filename
    else:
        filename = path + "\\" + filename
        
    destination = GetPath(filename)    

    if not os.path.isdir(destination):
        # os.mkdir(destination)
        os.makedirs(destination)

    return filename

def Append(list):
    file = ""
    for s, string in enumerate(list):
        file += string
        if s+1 < len(list):
            file += "\\"
    
    if not os.path.isdir(file):
        # os.mkdir(file)
        os.makedirs(file)
    
    return file





