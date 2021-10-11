"""
Convenience functions that are inefficient, but are maybe a bit easier to work with?
"""

from .CoordinateSystems import *
from collections import namedtuple
import numpy as np

__all__ = [
    "cartesian_to_zmatrix",
    "zmatrix_to_cartesian"
]

def cartesian_to_zmatrix(coords, ordering=None, use_rad = True):
    """
    Converts Cartesians to Z-Matrix coords and returns the underlying arrays

    :param coords: input Cartesians
    :type coords: np.ndarray
    :param ordering: the Z-matrix ordering as a list of lists
    :type ordering:
    :return: Z-matrix coords
    :rtype: Iterable[np.ndarray | list]
    """

    zms = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=ordering, use_rad=use_rad)
    crds = namedtuple("zms", ["coords", "ordering"])
    return crds(np.asarray(zms), zms.converter_options['ordering'])

def zmatrix_to_cartesian(coords, ordering=None, origins=None, axes=None, use_rad=True):
    """
    Converts Z-maztrix coords to Cartesians

    :param coords:
    :type coords: np.ndarray
    :param ordering:
    :type ordering:
    :param origins:
    :type origins:
    :param axes:
    :type axes:
    :param use_rad:
    :type use_rad:
    :return:
    :rtype:
    """
    carts = CoordinateSet(coords, ZMatrixCoordinates).convert(CartesianCoordinates3D,
                                                            ordering=ordering, origins=origins, axes=axes, use_rad=use_rad)
    return np.asarray(carts)