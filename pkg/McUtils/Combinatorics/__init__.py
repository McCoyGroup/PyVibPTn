"""
A place to store all utilities related to combinatorics.
Currently only contains a subpackage for working with permutations.
That package comes from the BasisReps work in Psience.
Might be worth extending to handle more lattice-path stuff.
"""

from .Permutations import *

__all__ = []
from .Permutations import __all__ as exposed
__all__ += exposed