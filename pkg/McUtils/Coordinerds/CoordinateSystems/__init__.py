"""
Provides a framework for using coordinates with explicit reference to an underlying coordinate system
"""


__all__ = []
from .CoordinateSystemConverter import *; from .CoordinateSystemConverter import __all__ as exposed
__all__ += exposed
from .CommonCoordinateSystems import *; from .CommonCoordinateSystems import __all__ as exposed
__all__ += exposed
from .CoordinateSystem import *; from .CoordinateSystem import __all__ as exposed
__all__ += exposed
from .CompositeCoordinateSystems import *; from .CompositeCoordinateSystems import __all__ as exposed
__all__ += exposed
from .CoordinateSet import *; from .CoordinateSet import __all__ as exposed
__all__ += exposed