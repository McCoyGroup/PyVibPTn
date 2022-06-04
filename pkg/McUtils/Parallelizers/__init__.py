"""
Provides utilities for setting up platform-independent parallelism
in a hopefully unobtrusive way.

This is used more extensively in `Psience`, but the design is to unify the MPI and `multiprocessing` APIs
so that one can simply pass in a `Parallelizer` object to a function and obtain parallelism over as
many processes as that object supports.
As a fallthrough, a `SerialNonParallelizer` is provided as a subclass that handles serial evaluation with
the same API so fewer special cases need to be checked.
Any function that supports parallelism should take the `parallelizer` keyword, which will be fed
the `Parallelizer` object itself.
"""

__all__ = []
from .Parallelizers import *; from .Parallelizers import __all__ as exposed
__all__ += exposed
from .Runner import *; from .Runner import __all__ as exposed
__all__ += exposed
from .SharedMemory import *; from .SharedMemory import __all__ as exposed
__all__ += exposed