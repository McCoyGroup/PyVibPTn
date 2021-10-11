"""
Defines options for different numerical things
"""

__all__ = [
    "Options"
]

import numpy as np

class OptionsContainer:
    """
    A singleton Options object that can be used to configure options for numerical stuff
    """

    NORM_ZERO_THRESH = None
    ZERO_THRESHOLD = 1.0e-17
    ZERO_PLACEHOLDER = None

    @property
    def zero_threshold(self):
        # just here so I can modify later
        return self.ZERO_THRESHOLD
    @zero_threshold.setter
    def zero_threshold(self, v):
        self.ZERO_THRESHOLD = v

    @property
    def norm_zero_threshold(self):
        if self.NORM_ZERO_THRESH is None:
            return self.zero_threshold
        else:
            return self.NORM_ZERO_THRESH
    @norm_zero_threshold.setter
    def norm_zero_threshold(self, v):
        self.NORM_ZERO_THRESH = v

    @property
    def zero_placeholder(self):
        if self.ZERO_PLACEHOLDER is None:
            return self.zero_threshold
        else:
            return self.ZERO_PLACEHOLDER
    @zero_placeholder.setter
    def zero_placeholder(self, v):
        self.ZERO_PLACEHOLDER = v

Options = OptionsContainer()
Options.__name__ = "Options"
Options.__doc__ = OptionsContainer.__doc__
