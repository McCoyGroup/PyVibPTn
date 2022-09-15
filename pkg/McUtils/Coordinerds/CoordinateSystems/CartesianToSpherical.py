from .CoordinateSystemConverter import CoordinateSystemConverter
from .CommonCoordinateSystems import CartesianCoordinates3D, SphericalCoordinates
from ...Numputils import vec_dots, vec_angles
import numpy as np
# this import gets bound at load time, so unfortunately PyCharm can't know just yet
# what properties its class will have and will try to claim that the files don't exist

class CartesianToSphericalConverter(CoordinateSystemConverter):
    """
    A converter class for going from Cartesian coordinates to ZMatrix coordinates
    """

    @property
    def types(self):
        return (CartesianCoordinates3D, SphericalCoordinates)

    def convert_many(self, coords, use_rad=True, origin=None, axes=None, **kw):
        """
        We'll implement this by having the ordering arg wrap around in coords?
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

        coords = coords - origin
        dists = np.linalg.norm(coords, axis=-1)
        proj_axes = np.broadcast_to(axes[1], coords.shape)
        angles, _ = vec_angles(coords - vec_dots(proj_axes, coords)[..., np.newaxis]*proj_axes, axes[0]) # projected into x-y plane
        if not use_rad:
            angles = np.rad2deg(angles)
        polars, _ = vec_angles(coords, axes[1])
        if not use_rad:
            polars = np.rad2deg(polars)

        return (
            np.concatenate([np.expand_dims(x, -1) for x in [dists, angles, polars]], axis=-1),
            dict(**kw, origin=origin, axes=axes)
        )

    def convert(self, coords, ordering=None, use_rad=True, return_derivs=False, **kw):
        coords, opts = self.convert_many(np.expand_dims(coords[0]))
        coords = coords[0]
        opts['origin'] = opts['origin'][0]
        opts['axes'] = [a[0] for a in opts['axes']]
        return coords, opts

__converters__ = [ CartesianToSphericalConverter() ]