"""
A module for making use of the results of calculations run by the Gaussian electronic structure package.
We'd like to be able to also support the NWChem and Psi4 packages, but haven't had the time, yet, to write it out.

Two main avenues of support are provided:
    1. importing Gaussian results
    2. setting up Gaussian jobs

The first is likely to be more useful to you, but we're hoping to be able to hook (2.) into the `Psience.Molecools` package.
The goal there is to provide automated support for setting up scans of molecular vibrations & the like.

There are already direct hooks into (1.) in `Psience.Data` through the `DipoleSurface` and `PotentialSurface` objects.
These are still in the prototype stage, but hopefully will allow us to unify strands of our Gaussian support,
 and also make it easy to unify support for Psi4 and NWChem data, once we have the basic interface down.
"""
from .GaussianImporter import *
from .GaussianJob import *
from .FChkDerivatives import *

__all__ = []
from .GaussianImporter import __all__ as exposed
__all__ += exposed
from .GaussianJob import __all__ as exposed
__all__ += exposed
from .FChkDerivatives import __all__ as exposed
__all__ += exposed