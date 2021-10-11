"""
Provides hooks for viewing images, mostly through the Matplotlib Image interface
"""

__all__ = [
    "Image"
]

from .Plots import ArrayPlot

class Image(ArrayPlot):
    """
    Simple subclass of ArrayPlot that just turns off most of the unnecessary features
    """

    def __init__(self, data, **kwargs):
        kwargs['frame'] = False
        kwargs['ticks_style'] = (False, False)#, dict(left=False, right=False))
        kwargs['ticks'] = ([], [])
        kwargs['padding'] = (0, 0)
        if 'image_size' not in kwargs:
            image_size = tuple(x*72/100 for x in data.shape[:2])
            kwargs['image_size'] = image_size
        super().__init__(data, **kwargs)
        self.ticks = ([], [])

    @classmethod
    def from_file(cls, file_name, format=None, **opts):
        import matplotlib.image as mpimg
        data = mpimg.imread(file_name, format=format)
        return cls(data, **opts)