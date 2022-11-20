from distutils.core import setup
from Cython.Build import cythonize
import Dossier
import numpy
import os

# Faire ça pour compiler les fonctions ecrit en cython

folder = Dossier.Get_Path(__file__)

dossiersEtFichiers = os.listdir(folder)

for path in dossiersEtFichiers:

    nomEtExtension = path.split('.')

    if len(nomEtExtension) < 2: continue

    extension = nomEtExtension[1]

    if extension != 'pyx': continue

    setup(
        ext_modules=cythonize(path),
        include_dirs=[numpy.get_include()]
    )

# python setup.py build_ext --inplace
# cython -a CalcCython.pyx