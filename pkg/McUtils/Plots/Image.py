"""
Provides hooks for viewing images, mostly through the Matplotlib Image interface
"""

__all__ = [
    "Image"
]

import numpy as np
from .Plots import ArrayPlot

class Image(ArrayPlot):
    """
    Simple subclass of ArrayPlot that just turns off most of the unnecessary features
    """

    default_opts = {
        'frame' : False,
        'ticks_style' : (False, False),#, dict(left=False, right=False))
        'ticks' : ([], []),
        'padding': ([0, 0], [0, 0]),
        'aspect_ratio':'auto'
    }
    def __init__(self, data, plot_range=None, image_size=None, **kwargs):
        if image_size is None:
            image_size = (data.shape[1], data.shape[0])
        if plot_range is None:
            plot_range = ((0, data.shape[1]), (0, data.shape[0]))
        kwargs = dict(self.default_opts, **kwargs)
        super().__init__(data, plot_range=plot_range, image_size=image_size, **kwargs)

    @classmethod
    def from_file(cls, file_name, format=None, **opts):
        import matplotlib.image as mpimg

        data = mpimg.imread(file_name, format=format)
        data = np.flip(data, axis=0)
        return cls(data, **opts)