"""
I considered renaming 'Interface_Solvers' to 'Solvers,' but encountered issues with loading old damage simulations.
As a workaround, I've created this module to act as a redirect, allowing seamless access to the new module while preserving compatibility with the old one.
"""
from Display import myPrintError
myPrintError("Link Interface_Solvers.py to Solvers.py\n(YOU SHOULD NOT USE IT)")
from Solvers import _Solve, _Solve_Axb, _Available_Solvers, ResolType, AlgoType 