"""
Provides a CoordinateSet class that acts as a symbolic extension of np.ndarray to provide an explicit basis
"""
import numpy as np
from .CoordinateSystem import CoordinateSystem, CoordinateSystemError
from .CommonCoordinateSystems import CartesianCoordinates3D
from .CoordinateUtils import is_multiconfig

__all__ = [
    "CoordinateSet"
]

__reload_hook__ = ['.CoordinateSystem', '.CommonCoordinateSystems', '.CoordinateUtils']

######################################################################################################
##
##                                   CoordinateSet Class
##
######################################################################################################
class CoordinateSet(np.ndarray):
    """
    A subclass of np.ndarray that lives in an explicit coordinate system and can convert between them
    """
    # note that the subclassing procedure here is very non-standard because of how ndarray works

    def __new__(cls, coords, system = CartesianCoordinates3D, converter_options = None):
        self = np.asarray(coords).view(cls)
        self.system = system
        self.converter_options = converter_options
        return self

    def __init__(self, coords, system = CartesianCoordinates3D, converter_options = None):
        # all the heavy lifting is done in _validate
        # and we only want to ever call this once
        self._validate()

    def __array_finalize__(self, coords):
        # basically just a validator...
        if coords is None:
            return None

        self.system = getattr(coords, "system", CartesianCoordinates3D)
        self.converter_options = getattr(coords, "converter_options", None)

    def _validate(self):
        base_dim = self.system.dimension
        if isinstance(base_dim, int):
            core_dim = self.shape[-1]
        else:
            cdim = self.shape[-len(base_dim):]
            base_dim = tuple(base_dim)
            core_dim = tuple( a if a is None else b for a, b in zip(base_dim, cdim) )
        if base_dim != core_dim:
            raise CoordinateSystemError(
                "Dimension of basis {} '{}' and dimension of coordinate set '{}' misaligned".format(
                    self.system.name,
                    self.system.dimension,
                    core_dim
                )
            )

    def __str__(self):
        return "{}({}, {})".format(type(self).__name__, self.system.name, super().__str__())

    def __eq__(self, other):
        if isinstance(other, CoordinateSet) and self.system is not other.system:
            return False
        else:
            return super().__eq__(other)

    @property
    def multiconfig(self):
        """Determines whether self.coords represents multiple configurations of the coordinates

        :return:
        :rtype:
        """
        return is_multiconfig(self)

    def transform(self, tf):
        """Applies a transformation to the stored coordinates

        :param tf: the transformation function to apply to the system
        :type tf:
        :return:
        :rtype:
        """
        coords = self
        new_coords = tf(coords)
        return type(self)(new_coords, system=self.system)

    def convert(self, system, **kw):
        """Converts across coordinate systems

        :param system: the target coordinate system
        :type system: CoordinateSystem
        :return: new_coords
        :rtype: CoordinateSet
        """
        cops = self.converter_options
        if cops is None:
            cops = {}
        kw = dict(cops, **kw)
        res = self.system.convert_coords(self, system, **kw)
        if isinstance(res, tuple):
            new_coords, ops = res
        else:
            new_coords = res
            ops = {}
        return type(self)(new_coords, system=system, converter_options=ops)

    def derivatives(self,
                    function,
                    order=1,
                    coordinates=None,
                    result_shape=None,
                    **fd_options
                    ):
        """
        Takes derivatives of `function` with respect to the current geometry

        :param function:
        :type function:
        :param order:
        :type order:
        :param coordinates:
        :type coordinates:
        :param fd_options:
        :type fd_options:
        :return:
        :rtype:
        """
        return self.system.derivatives(self, function, order=order,
                                       coordinates=coordinates, result_shape=result_shape, **fd_options)

    def jacobian(self,
                 system,
                 order=1,
                 coordinates=None,
                 converter_options=None,
                 all_numerical=False,
                 analytic_deriv_order=None,
                 **fd_options
                 ):
        """Delegates to the jacobian function of the current coordinate system

        :param system:
        :type system:
        :param order:
        :type order:
        :param mesh_spacing:
        :type mesh_spacing:
        :param prep:
        :type prep:
        :param coordinates:
        :type coordinates:
        :param fd_options:
        :type fd_options:
        :return:
        :rtype:
        """

        cops = self.converter_options
        if cops is None:
            cops = {}
        if converter_options is None:
            converter_options = {}
        kw = dict(cops, **converter_options)
        return self.system.jacobian(self,
                                    system,
                                    order=order,
                                    coordinates=coordinates,
                                    converter_options=kw,
                                    all_numerical=all_numerical,
                                    analytic_deriv_order=analytic_deriv_order,
                                    **fd_options
                                    )
