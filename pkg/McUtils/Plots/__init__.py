"""
A plotting framework that builds off of matplotlib, but potentially could use a different backend

"""

from .Graphics import *
from .Plots import *
from .Primitives import *
from .Interactive import *
from .Styling import *
from .Image import *
from .Properties import *

__all__ = []
from .Graphics import __all__ as exposed
__all__ += exposed
from .Plots import __all__ as exposed
__all__ += exposed
from .Primitives import __all__ as exposed
__all__ += exposed
from .Interactive import __all__ as exposed
__all__ += exposed
from .Styling import __all__ as exposed
__all__ += exposed
from .Image import __all__ as exposed
__all__ += exposed
from .Properties import __all__ as exposed
__all__ += exposed
