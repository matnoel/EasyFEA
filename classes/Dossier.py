import os
# import 
# export PYTHONPATH=$PYTHONPATH:/home/matthieu/Documents/PythonEF/classes

def GetPath(filename=None):
    """Renvoie le path du fichier ou renvoie le path vers le fichier Dossier donc renvoie Python Ef"""
    
    if filename == None:
        # Renvoie le path vers PythonEF
        path = os.path.dirname(__file__)
        path = os.path.dirname(path)
    else:
        # Renvoie le path vers le fichier
        path = os.path.dirname(filename)    

    return path

def NewFile(filename: str, pathname=GetPath(), results=False):
    """Renvoie le path vers le fichier avec l'extension ou non
    filename peut etre : un fichier ou un dossier
    De base le path renvoie vers le path ou est PythonEF
    
    if results:
        filename = resultsPath/filename
    else:
        filename = path/filename
    
    la liaison dépend de l'os ubuntu mac linux
    """

    if results:
        pathname = os.path.join(pathname, "results")        
        # path = ResultsHere.Get_Results_Path()
        # ResultsHere.py
        # def Get_Results_Path():
        #     import Dossier
        #     return Dossier.GetPath(__file__)
    filename = os.path.join(pathname, filename)    
        
    destination = GetPath(filename)    

    if not os.path.isdir(destination):
        # os.mkdir(destination)
        os.makedirs(destination)

    return filename

def Join(list):
    file = ""
    for f in list:
        file = os.path.join(file, f)

    if not os.path.isdir(file):
        if '.' in file:
            path = GetPath(file)
            if not os.path.exists(path):
                # os.mkdir(file)
                os.makedirs(path)
        else:
            if not os.path.exists(file):
                # os.mkdir(file)
                os.makedirs(file)
        
    return file





