from .CoordinateSystemConverter import CoordinateSystemConverter
from .CommonCoordinateSystems import CartesianCoordinates3D, SphericalCoordinates
from ...Numputils import *
import numpy as np

class SphericalToCartesianConverter(CoordinateSystemConverter):
    """
    A converter class for going from ZMatrix coordinates to Cartesian coordinates
    """

    @property
    def types(self):
        return (SphericalCoordinates, CartesianCoordinates3D)

    def convert_many(self,
                     coords,
                     origin=None,
                     axes=None,
                     use_rad=True,
                     **kw
                     ):
        """
        Expects to get a list of configurations
        These will look like:
            [
                [dist, angle, dihedral]
                ...
            ]

        **For efficiency it is assumed that all configurations have the same length**

        :param coords:
        :type coords:
        :param origins:
        :type origins:
        :param axes:
        :type axes:
        :param use_rad:
        :type use_rad:
        :param kw:
        :type kw:
        :param ordering:
        :type ordering:
        :param return_derivs:
        :type return_derivs:
        :return:
        :rtype:
        """

        coords = np.asanyarray(coords)
        if origin is None:
            origin = [0., 0., 0.]
        origin = np.asanyarray(origin)
        for d in range(coords.ndim - origin.ndim):
            origin = np.expand_dims(origin, 0)
        if axes is None:
            axes = [[1., 0., 0.], [0., 0., 1.]]
        axes = np.asanyarray(axes)
        for d in range(coords.ndim - axes[0].ndim):
            axes = [np.expand_dims(a, 0) for a in axes]

        r = coords[..., 0]
        a = coords[..., 1]
        if not use_rad:
            a = np.deg2rad(a)
        d = coords[..., 2]
        if not use_rad:
            d = np.deg2rad(d)
        pts = polar_to_cartesian(origin, axes[0], axes[1], r, a, d)

        return pts, dict(**kw, origin=origin, axes=axes)

    def convert(self, coords, **kw):
        """dipatches to convert_many but only pulls the first"""
        total_points, opts = self.convert_many(coords[np.newaxis], **kw)
        return total_points[0], opts

__converters__ = [ SphericalToCartesianConverter() ]