"""
This provides a unified interface for handling "surfaces" i.e. functions that map from coordinate space to the real line
The interface provides multiple different modes of surface specification, allowing for Taylor series data,
expansions basis expansions, or analytic representations.
The idea is to provide a unified interface that represents the _concept_ while allowing for flexibility in implementation.
Specific types of surfaces, like PES and dipole surfaces can be layered on top as superclasses and provided their own bespoke importers.
"""

from .BaseSurface import *
from .Surface import *

__all__ = []
from .BaseSurface import __all__ as exposed
__all__ += exposed
from .Surface import __all__ as exposed
__all__ += exposed