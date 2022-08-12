from setuptools import setup, find_packages
# from setuptools import find_packages
# from distutils.core import setup

listPackages = find_packages()

setup(
    name='PythonEF',    
    author='Matthieu Noel',
    author_email='matthieu.noel@univ-eiffel.fr',
    packages=listPackages
)