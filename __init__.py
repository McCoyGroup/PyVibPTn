"""
A loader for Psience.VPT2 to expose it specifically essentially just so we can rename this...
"""

import os, sys

sys.path.insert(0,
                os.path.dirname(os.path.abspath(__file__))
                )

from Psience.VPT2 import *
from Psience.VPT2 import __all__