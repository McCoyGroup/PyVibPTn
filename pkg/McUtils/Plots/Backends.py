"""
For now, just a super simple Enum of supported backends
Maybe in the future we'll add better support so that the backends themselves can all support a common subset
of features, but I think we'll 90% of the time just want to use MPL or VTK so who knows...
If that happens, lots of the 'if backend == MPL' stuff will change to use a Backend object
"""

__all__ = [
    "Backends"
]

import enum

class Backends(enum.Enum):
    MPL = 'matplotlib'
    VTK = 'VTK'
