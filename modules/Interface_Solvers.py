"""
I considered renaming 'Interface_Solvers.py' to 'Solvers.py', but faced issues with loading old damage simulations.
As a temporary solution, I've created this module to serve as a redirect, ensuring smooth access to the new module while maintaining compatibility with the old one. TODO: Delete this module at the end of the thesis.
"""
from Display import myPrintError
myPrintError("Link Interface_Solvers.py to Solvers.py (DO NOT USE IT)")
from Solvers import *