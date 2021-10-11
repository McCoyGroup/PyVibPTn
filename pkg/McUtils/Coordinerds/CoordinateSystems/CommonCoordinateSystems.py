from .CoordinateSystem import BaseCoordinateSystem

__all__ = [
    "CartesianCoordinateSystem",
    "InternalCoordinateSystem",
    "CartesianCoordinateSystem3D",
    "CartesianCoordinates3D",
    "SphericalCoordinateSystem",
    "SphericalCoordinates",
    "ZMatrixCoordinateSystem",
    "ZMatrixCoordinates"
    ]

######################################################################################################
##
##                                   CartesianCoordinateSystem Class
##
######################################################################################################
class CartesianCoordinateSystem(BaseCoordinateSystem):
    """
    Represents Cartesian coordinates generally
    """
    name = "Cartesian"
    def __init__(self, dimension=None, converter_options=None, **opts):
        """
        :param converter_options: options to be passed through to a `CoordinateSystemConverter`
        :type converter_options: None | dict
        :param dimension: the dimension of the coordinate system
        :type dimension: Iterable[None | int]
        :param opts: other options, if `converter_options` is None, these are used as the `converter_options`
        :type opts:
        """
        if converter_options is None:
            converter_options = opts
        super().__init__(self.name, dimension=dimension, converter_options=converter_options)

######################################################################################################
##
##                                   InternalCoordinateSystem Class
##
######################################################################################################
class InternalCoordinateSystem(BaseCoordinateSystem):
    """
    Represents Internal coordinates generally
    """

    name = "Internal"
    def __init__(self, dimension = None, coordinate_shape=None, converter_options=None, **opts):
        """
        :param converter_options: options to be passed through to a `CoordinateSystemConverter`
        :type converter_options: None | dict
        :param coordinate_shape: shape of a single coordinate in this coordiante system
        :type coordinate_shape: Iterable[None | int]
        :param dimension: the dimension of the coordinate system
        :type dimension: Iterable[None | int]
        :param opts: other options, if `converter_options` is None, these are used as the `converter_options`
        :type opts:
        """
        if converter_options is None:
            converter_options = opts
        super().__init__(self.name, dimension=dimension, coordinate_shape=coordinate_shape, converter_options=converter_options)

######################################################################################################
##
##                                   CartesianCoordinates3D Class
##
######################################################################################################
class CartesianCoordinateSystem3D(CartesianCoordinateSystem):
    """
    Represents Cartesian coordinates in 3D
    """
    name = "Cartesian3D"
    def __init__(self, converter_options=None, dimension=(None, 3), **opts):
        """
        :param converter_options: options to be passed through to a `CoordinateSystemConverter`
        :type converter_options: None | dict
        :param dimension: the dimension of the coordinate system
        :type dimension: Iterable[None | int]
        :param opts: other options, if `converter_options` is None, these are used as the `converter_options`
        :type opts:
        """
        if converter_options is None:
            converter_options = opts
        super().__init__(dimension=dimension, converter_options=converter_options)
CartesianCoordinates3D = CartesianCoordinateSystem3D()
CartesianCoordinates3D.__name__ = "CartesianCoordinates3D"
CartesianCoordinates3D.__doc__ = """
    A concrete instance of `CartesianCoordinateSystem3D`
    """

######################################################################################################
##
##                                   ZMatrixCoordinateSystem Class
##
######################################################################################################
class ZMatrixCoordinateSystem(InternalCoordinateSystem):
    """
    Represents ZMatrix coordinates generally
    """
    name = "ZMatrix"
    def __init__(self,
                 converter_options=None,
                 dimension=(None, None),
                 coordinate_shape=(None, 3),
                 **opts):
        """
        :param converter_options: options to be passed through to a `CoordinateSystemConverter`
        :type converter_options: None | dict
        :param coordinate_shape: shape of a single coordinate in this coordiante system
        :type coordinate_shape: Iterable[None | int]
        :param dimension: the dimension of the coordinate system
        :type dimension: Iterable[None | int]
        :param opts: other options, if `converter_options` is None, these are used as the `converter_options`
        :type opts:
        """
        if converter_options is None:
            converter_options = opts
        super().__init__(dimension=dimension, coordinate_shape=coordinate_shape, converter_options=converter_options)
        # self.jacobian_prep = self.jacobian_prep_coordinates
    # def jacobian_prep_coordinates(self, coord, displacements, values):
    #     values = values[..., :, (1, 3, 5)]
    #     # we will want to make sure all angles and dihedrals stay within a range of eachother...
    #     return displacements, values
ZMatrixCoordinates = ZMatrixCoordinateSystem()
ZMatrixCoordinates.__name__ = "ZMatrixCoordinates"
ZMatrixCoordinates.__doc__ = """
    A concrete instance of `ZMatrixCoordinateSystem`
    """

######################################################################################################
##
##                                   SphericalCoordinateSystem Class
##
######################################################################################################
class SphericalCoordinateSystem(BaseCoordinateSystem):
    """
    Represents Spherical coordinates generally
    """
    name = "SphericalCoordinates"
    def __init__(self, converter_options=None, **opts):
        """
        :param converter_options: options to be passed through to a `CoordinateSystemConverter`
        :type converter_options: None | dict
        :param opts: other options, if `converter_options` is None, these are used as the `converter_options`
        :type opts:
        """
        if converter_options is None:
            converter_options = opts
        super().__init__(self.name, dimension=3, converter_options=converter_options)
SphericalCoordinates = SphericalCoordinateSystem()
SphericalCoordinates.__name__ = "SphericalCoordinates"
SphericalCoordinates.__doc__ = """
    A concrete instance of `SphericalCoordinateSystem`
    """