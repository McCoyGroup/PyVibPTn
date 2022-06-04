"""
Provides a small data framework for wrapping up datasets into classes for access and loading.

The basic structure for a new dataset is defined in `CommonData.DataHandler`.
A simple, concrete example is in `AtomData.AtomData`.
A slightly more involved example is in `ConstantsData.UnitsData`.
"""

__all__ = []
from .CommonData import *; from .CommonData import __all__ as _all
__all__ += _all
from .AtomData import *; from .AtomData import __all__ as _all
__all__ += _all
from .ConstantsData import *; from .ConstantsData import __all__ as _all
__all__ += _all
from .BondData import *; from .BondData import __all__ as _all
__all__ += _all
from .WavefunctionData import *; from .WavefunctionData import __all__ as _all
__all__ += _all
from .PotentialData import *; from .PotentialData import __all__ as _all
__all__ += _all