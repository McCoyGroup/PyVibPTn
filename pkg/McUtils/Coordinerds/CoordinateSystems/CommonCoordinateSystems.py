
import numpy as np
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
        self.jacobian_prep = self.jacobian_prep_coordinates
    def jacobian_prep_coordinates(self,
                                  coord, displacements, values,
                                  dihedral_cutoff=6
                                  ):
        # target_ndim = len(self.dimension) + len(coord) + 1
        extra_dim = displacements.ndim - values.ndim
        raw_displacement_shape = displacements.shape[:-2]
        analytic_order = extra_dim // 2
        dihedrals = values[..., 2]
        central_point = tuple((x - 1) // 2 for x in raw_displacement_shape) # just need a rough check
        ref = dihedrals[central_point]
        for x in central_point:
            ref = ref[np.newaxis]
        # ref = np.broadcast_to(ref, dihedrals.shape)
        true_diffs = dihedrals - ref
        bad_spots = np.where(abs(true_diffs) > dihedral_cutoff)
        if len(bad_spots) > 0 and len(bad_spots[0]) > 0:
            if analytic_order == 0:
                # ref_vals = ref[bad_spots]
                patch_vals = dihedrals[bad_spots]
                patch_signs = np.sign(patch_vals)
                # if we have a negative 2pi, we have a negative displaced val and positive start val
                # in this case we want to do np.pi + (displaced_val - np.pi) = 2pi + displaced_val
                # if we have a positive 2pi, we have a positive displaced_val and negative start val
                # in this case we want to do -np.pi + (displaced_val - np.pi) = -2pi + displaced_val
                fix_spots = bad_spots + (np.full(len(bad_spots[0]), 2),)
                values[fix_spots] = patch_vals + (-patch_signs)*2*np.pi
            elif analytic_order == 1:
                raise NotImplementedError('correcting periodicity wraparound not handled for analytic derivative order {}'.format(analytic_order))
            else:
                raise NotImplementedError('correcting periodicity wraparound not handled for analytic derivative order {}'.format(analytic_order))

        # we will want to make sure all angles and dihedrals stay within a range of eachother...
        return displacements, values


    @classmethod
    def canonicalize_order_list(self, ncoords, order_list):
        """
        Normalizes the way the ZMatrix coordinates are built out

        :param ncoords:
        :type ncoords:
        :param order_list: the basic ordering to apply for the
        :type order_list: iterable or None
        :return:
        :rtype: iterator of int triples
        """
        if order_list is None:
            normalized_list = np.array((
                np.arange(ncoords),
                np.arange(-1, ncoords - 1),
                np.arange(-2, ncoords - 2),
                np.arange(-3, ncoords - 3),
                0
            )).T
        else:
            normalized_list = [[]] * len(order_list)
            any3 = None
            def bad_ol():
                raise ValueError((
                    "order list {ol} mixes internal spec forms "
                    "Expected ({n}, {d}, {a}, {t}), ({n}, {d}, {a}, {t}, {f}), ({d}, {a}, {t}), or {n}"
                ).format(
                    ol=order_list,
                    el=el,
                    n="atomNum",
                    d="distAtomNum",
                    a="angleAtomNum",
                    t="dihedAtomNum",
                    f="dihedForm"
                ))
            for i, el in enumerate(order_list):
                if isinstance(el, int):
                    if any3 is None:
                        any3 = True
                    elif not any3:
                        bad_ol()
                    spec = (
                        el,
                        normalized_list[i - 1][0] if i > 0 else -1,
                        normalized_list[i - 2][0] if i > 1 else -1,
                        normalized_list[i - 3][0] if i > 2 else -1
                    )
                else:
                    spec = tuple(el)
                    if len(spec) == 3:
                        if any3 is None:
                            any3 = True
                        elif not any3:
                            bad_ol()
                        spec = (i,) + spec + (
                            normalized_list[i - 1][0] if i > 0 else -1,
                            normalized_list[i - 2][0] if i > 1 else -1,
                            normalized_list[i - 3][0] if i > 2 else -1
                        )
                    elif len(spec) == 4:
                        if any3:
                            bad_ol()
                        any3 = False
                    elif len(spec) == 5:
                        if any3:
                            bad_ol()
                        any3 = False
                    else:
                        raise ValueError(
                            "Z-matrix conversion spec {el} not understood. "
                            "Expected ({n}, {d}, {a}, {t}), ({n}, {d}, {a}, {t}, {f}), ({d}, {a}, {t}), or {n}".format(
                                el=el,
                                n="atomNum",
                                d="distAtomNum",
                                a="angleAtomNum",
                                t="dihedAtomNum",
                                f="dihedForm"
                            ))
                normalized_list[i] = spec
            nlist_len = max(len(x) for x in normalized_list)
            if nlist_len == 5:
                for i,s in enumerate(normalized_list):
                    if len(s) == 4:
                        normalized_list[i] = s + (0,)
        return np.asanyarray(normalized_list, dtype=np.int8)

    @classmethod
    def tile_order_list(self, ol, ncoords):
        nol = len(ol)
        ncol = len(ol[0])
        fsteps = ncoords / nol
        steps = int(fsteps)
        if steps != fsteps:
            raise ValueError(
                "{}: Number of coordinates {} and number of specifed elements {} misaligned".format(
                    type(self),
                    ncoords,
                    nol
                )
            )
        # broadcasts a single order spec to be a multiple order spec
        base_tile = np.broadcast_to(ol[:, :4], (steps, nol, 4))
        shift = np.reshape(np.arange(0, ncoords, nol), (steps, 1, 1))
        ol_tiled = base_tile + shift
        # now we add on extra flags
        if ncol > 4:
            flags_tiled = np.broadcast_to(ol[:, 4:], (steps, nol, ncol-4))
            ol_tiled = np.concatenate([ol_tiled, flags_tiled], axis=-1)
        return np.reshape(ol_tiled, (ncoords, ncol))
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