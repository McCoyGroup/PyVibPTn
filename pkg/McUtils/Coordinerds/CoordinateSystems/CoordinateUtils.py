"""
Little utils that both CoordinateSet and CoordinateSystem needed
"""

import numpy as np

__all__ = [
    "is_multiconfig",
    "mc_safe_apply"
]

def is_multiconfig(coords, coord_shape=None):
    if coord_shape is None:
        coord_shape = (None, None)
    return len(coords.shape) > len(coord_shape)

def mc_safe_apply(fun, coords):
    """Applies fun to the coords in such a way that it will apply to an array of valid
    coordinates (as determined by dimension of the basis). This will either be a single configuration
    or multiple configurations

    :param fun:
    :type fun:
    :return:
    :rtype:
    """

    if is_multiconfig(coords):
        base_shape = coords.shape
        new_shape = (-1,) + base_shape[-2:]
        coords = np.reshape(coords, new_shape)
        new_coords = fun(coords)
        rest = None
        if not isinstance(new_coords, np.ndarray):
            rest = new_coords[1:]
            new_coords = new_coords[0]
        revert_shape = tuple(base_shape[:-2]) + new_coords.shape[1:]
        new_coords = np.reshape(new_coords, revert_shape)
        if rest is not None:
            new_coords = (new_coords,) + rest
    else:
        new_coords = fun(coords)
    return new_coords