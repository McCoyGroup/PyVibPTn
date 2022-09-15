"""
Sets up a general Interpolator class that looks like Mathematica's InterpolatingFunction class
"""
import typing

import numpy as np, abc, enum
import scipy.interpolate as interpolate
import scipy.spatial as spat
from .Mesh import Mesh, MeshType
from ..Numputils import vec_outer

__all__ = [
    "Interpolator",
    "Extrapolator",
    "ProductGridInterpolator",
    "UnstructuredGridInterpolator"
]


class InterpolatorException(Exception):
    pass


######################################################################################################
##
##                                   Interpolator Class
##
######################################################################################################

class BasicInterpolator(metaclass=abc.ABCMeta):
    """
    Defines the abstract interface we'll need interpolator instances to satisfy so that we can use
    `Interpolator` as a calling layer
    """

    @abc.abstractmethod
    def __init__(self, grid, values, **opts):
        raise NotImplementedError("abstract interface")

    @abc.abstractmethod
    def __call__(self, points, **kwargs):
        raise NotImplementedError("abstract interface")

    @abc.abstractmethod
    def derivative(self, order):
        """
        Constructs the derivatives of the interpolator at the given order
        :param order:
        :type order:
        :return:
        :rtype: BasicInterpolator
        """
        raise NotImplementedError("abstract interface")

class ProductGridInterpolator(BasicInterpolator):
    """
    A set of interpolators that support interpolation
    on a regular (tensor product) grid
    """

    def __init__(self, grids, vals, caller=None, order=None, extrapolate=True):
        """

        :param grids:
        :type grids:
        :param points:
        :type points:
        :param caller:
        :type caller:
        :param order:
        :type order: int | Iterable[int]
        """

        if order is None:
            order = 3

        self.grids = grids
        self.vals = vals
        if caller is None:
            if isinstance(grids[0], (int, float, np.integer, np.floating)):
                grids = [grids]
            ndim = len(grids)
            if ndim == 1:
                opts = {}
                if order is not None:
                    opts["k"] = order

                caller = interpolate.PPoly.from_spline(interpolate.splrep(grids[0], vals, k=order),
                                                      extrapolate=extrapolate
                                                      )
            else:
                caller = self.construct_ndspline(grids, vals, order, extrapolate=extrapolate)
        self.caller = caller

    @classmethod
    def construct_ndspline(cls, grids, vals, order, extrapolate=True):
        """
        Builds a tensor product ndspline by constructing a product of 1D splines

        :param grids: grids for each dimension independently
        :type grids: Iterable[np.ndarray]
        :param vals:
        :type vals: np.ndarray
        :param order:
        :type order: int | Iterable[int]
        :return:
        :rtype: interpolate.NdPPoly
        """

        # inspired by `csaps` python package
        # idea is to build a spline approximation in
        # every direction (where the return values are multidimensional)
        ndim = len(grids)
        if isinstance(order, (int, np.integer)):
            order = [order] * ndim

        coeffs = vals
        x = [None]*ndim
        for i, (g, o) in enumerate(zip(grids, order)):
            og_shape = coeffs.shape
            coeffs = coeffs.reshape((len(g), -1)).T
            sub_coeffs = [np.empty(0)]*len(coeffs)
            for e,v in enumerate(coeffs):
                ppoly = interpolate.PPoly.from_spline(interpolate.splrep(g, v, k=o))
                x[i] = ppoly.x
                sub_coeffs[e] = ppoly.c
            coeffs = np.array(sub_coeffs)
            coeffs = coeffs.reshape(
                og_shape[1:]+
                    sub_coeffs[0].shape
            )
            tot_dim = ndim+i+1
            coeffs = np.moveaxis(coeffs, tot_dim-2, tot_dim-2-i)

        return interpolate.NdPPoly(coeffs, x, extrapolate=extrapolate)

    def __call__(self, *args, **kwargs):
        """
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype: np.ndarray
        """
        return self.caller(*args, **kwargs)

    def derivative(self, order):
        """
        :param order:
        :type order:
        :return:
        :rtype: ProductGridInterpolator
        """
        # ndim = len(self.grids)
        # if ndim == 1:
        return type(self)(
            self.grids,
            self.vals,
            caller=self.caller.derivative(order)
        )
        # elif ndim == 2:
        #     def caller(coords, _og=self.caller, **kwargs):
        #         return _og(coords, dx=order[0], dy=order[1], **kwargs)
        #     return type(self)(
        #         self.grids,
        #         self.vals,
        #         caller=caller
        #     )
        # else:
        #     derivs = self.caller.derivative(order)
        #     raise NotImplementedError("woof")

class UnstructuredGridInterpolator(BasicInterpolator):
    """
    Defines an interpolator appropriate for totally unstructured grids by
    delegating to the scipy `RBF` interpolators
    """

    default_neighbors=25
    def __init__(self, grid, values, order=None, neighbors=None, extrapolate=True, **opts):
        """
        :param grid:
        :type grid: np.ndarray
        :param values:
        :type values:  np.ndarray
        :param order:
        :type order: int
        :param neighbors:
        :type neighbors: int
        :param extrapolate:
        :type extrapolate: bool
        :param opts:
        :type opts:
        """
        self.extrapolate=extrapolate
        self._hull = None
        self._grid = grid

        if neighbors is None:
            neighbors = np.min([self.default_neighbors, len(grid)])

        if order is not None:
            if isinstance(order, int):
                if order == 1:
                    order = "linear"
                elif order == 3:
                    order = "cubic"
                elif order == 5:
                    order = "quintic"
                else:
                    raise InterpolatorException("{} doesn't support interpolation order '{}'".format(
                        interpolate.RBFInterpolator,
                        order
                    ))
            self.caller = interpolate.RBFInterpolator(grid, values, kernel=order, neighbors=neighbors, **opts)
        else:
            self.caller = interpolate.RBFInterpolator(grid, values, neighbors=neighbors, **opts)

    def _member_q(self, points):
        """
        Checks if the points are in the interpolated convex hull
        in the case that we aren't extrpolating so we can return
        NaN for those points
        :param points:
        :type points:
        :return:
        :rtype:
        """
        if self._hull is None:
            self._hull = spat.ConvexHull(self._grid)
            self._hull = spat.Delaunay(self._hull)
        return self._hull.find_simplex(points) >= 0
    def __call__(self, points):
        if self.extrapolate:
            return self.caller(points)
        else:
            hull_points = self._member_q(points)
            res = np.full(len(points), np.nan)
            res[hull_points] = self.caller(points[hull_points])
            return res

    def derivative(self, order):
        """
        Constructs the derivatives of the interpolator at the given order
        :param order:
        :type order:
        :return:
        :rtype: UnstructuredGridInterpolator
        """
        raise NotImplementedError("derivatives not implemented for unstructured grids")

class RBFInterpolator:
    """
    Provides a flexible RBF interpolator that also allows
    for matching function derivatives
    """



    def __init__(self,
                 pts, values, *derivatives,
                 kernel:typing.Union[callable,dict],
                 auxiliary_basis=None,
                 extra_degree=-1
                 ):

        pts = np.asanyarray(pts)
        if pts.ndim == 1:
            pts = pts[:, np.newaxis]
        self.grid, self.grid_shifts, self.grid_scaling = self.renormalize_grid(pts)
        self.tree = spat.KDTree(self.grid)
        self.vals, self.val_shift, self.val_scaling = values
        self.derivs = derivatives
        if not isinstance(kernel, dict):
            kernel = {'function':kernel}
        if 'derivatives' not in kernel:
            kernel['derivatives'] = None
        self.kernel = kernel
        if not isinstance(auxiliary_basis, dict):
            auxiliary_basis = {'function':auxiliary_basis}
        if 'derivatives' not in auxiliary_basis:
            auxiliary_basis['derivatives'] = None
        self.aux_poly = auxiliary_basis
        self.extra_degree = extra_degree

    def renormalize_grid(self, pts):
        dim_slices = np.transpose(pts)
        mins = [np.min(x) for x in dim_slices]
        maxs = [np.max(x) for x in dim_slices]
        scalings = [M-m for M,m in zip(mins, maxs)]
        rescaled_slices = [(x-m)/s for x,m,s in zip(dim_slices, mins, scalings)]
        return np.transpose(rescaled_slices), mins, scalings
    def renormalize_values(self, values):
        min = np.min(values)
        scaling = np.max(values) - min
        return (values-min)/scaling, min, scaling
    def renomalize_derivs(self, derivs):
        #TODO: rescale, handle coordinate transforms, and take upper triangles
        ...

    # @staticmethod
    # def _poly_eval
    @staticmethod
    def _eval_poly(pts, order, var=None, var_deriv=None, deriv_order=0):
        #  use direct product expansion of terms & corresponding derivs!
        #       decreasese requested nominal order and allows for better fits
        #       and potentially more stable extrapolation?
        ndim = pts.shape[-1]

        # if var is not None:
        #     pts = var(pts)

        # TODO: add in custom vars with derivs
        res = [[] for _ in range(deriv_order+1)]
        for d in range(deriv_order+1):
            x = pts
            I = np.broadcast_to(np.eye(ndim)[np.newaxis], (pts.shape[0], ndim, ndim))
            if d == 0: # add in initial terms
                res[d].append(x)
            elif d == 1:
                res[d].append(I)
            else:
                res[d].append(0)
            for i in range(d-1):  # add in zero tensors
                res[d].append(0)
            for i in range(d, order-1): # everything below is handled or zeros
                # evaluate basic monomial terms
                x = np.expand_dims(x, 1)  # npts x 1 ... [i times]  x ndim
                A = np.expand_dims(res[d][-1], -1) # npts x ndim x ... [(d+1)*i times] x 1
                X_i = A * pts
                if d > 0:
                    # we have the expr d_k X^N = d_k X^(N-1) * x + k d_(k-1) X^(N-1) * I
                    I = np.expand_dims(I, 1) # npts x 1 x ... x ndim x ndim
                    B = res[d-1][i-1] # npts x ndim x ...[d * i-1 times] x 1 x 1
                    X_i += d * np.expand_dims(B, -1)
                res[d].append(X_i)
        return res
    @staticmethod
    def _eval_r_derivs(pts, order):
        ...
    @classmethod
    def _poly(cls, order):
        def p(pts):
            return cls._eval_poly(pts, order, ...)
        return p
    def evaluate_poly_matrix(self, pts, degree, poly_origin=0.5, include_constant_term=True):
        #TODO: include deriv order and merge all at once
        fn = self.aux_poly['function']
        pts = pts - poly_origin
        blocks = ([np.ones((len(pts), 1))] if include_constant_term else []) + [
            fn(o)(pts)
            for o in range(1, degree+1)
        ]
        return np.concatenate(blocks, axis=1)
    def evaluate_rbf_matrix(self, pts, centers, deriv_order=0):
        displacements_matrix = pts[:, np.newaxis, :] - centers[np.newaxis, :, :]
        distance_mat = np.linalg.norm(displacements_matrix, axis=1)
        rbf_vals = self.kernel(distance_mat)
        res = [rbf_vals]
        # TODO: now ~~include derivs~~
        der_fun = self.kernel['derivatives']
        # There _must_ be a removable singularity at zero and no singularities anywhere else
        rbf_dvals = [d(distance_mat) for d in self.kernel['derivatives']]






class ExtrapolatorType(enum.Enum):
    Default='Automatic'
    Error='Raise'

class Interpolator:
    """
    A general purpose that takes your data and just interpolates it without whining or making you do a pile of extra work
    """
    DefaultExtrapolator = ExtrapolatorType.Default
    def __init__(self,
                 grid,
                 vals,
                 interpolation_function=None,
                 interpolation_order=None,
                 extrapolator=None,
                 extrapolation_order=None,
                 **interpolation_opts
                 ):
        """
        :param grid: an unstructured grid of points **or** a structured grid of points **or** a 1D array
        :type grid: np.ndarray
        :param vals: the values at the grid points
        :type vals: np.ndarray
        :param interpolation_function: the basic function to be used to handle the raw interpolation
        :type interpolation_function: None | BasicInterpolator
        :param interpolation_order: the order of extrapolation to use (when applicable)
        :type interpolation_order: int | str | None
        :param extrapolator: the extrapolator to use for data points not on the grid
        :type extrapolator: Extrapolator | None | str | function
        :param extrapolation_order: the order of extrapolation to use by default
        :type extrapolation_order: int | str | None
        :param interpolation_opts: the options to be fed into the interpolating_function
        :type interpolation_opts:
        """
        self.grid = grid = Mesh(grid) if not isinstance(grid, Mesh) else grid
        self.vals = vals
        if interpolation_function is None:
            interpolation_function = self.get_interpolator(grid, vals,
                                                           interpolation_order=interpolation_order,
                                                           allow_extrapolation=extrapolator is None,
                                                           **interpolation_opts
                                                           )
        self.interpolator = interpolation_function

        if extrapolator is not None:
            if isinstance(extrapolator, ExtrapolatorType):
                if extrapolator == ExtrapolatorType.Default:
                    extrapolator = self.get_extrapolator(grid, vals, extrapolation_order=extrapolation_order)
                else:
                    raise ValueError("don't know what do with extrapolator type {}".format(extrapolator))
            elif not isinstance(extrapolator, Extrapolator):
                extrapolator = Extrapolator(extrapolator)
        self.extrapolator = extrapolator

    @classmethod
    def get_interpolator(cls, grid, vals, interpolation_order=None, allow_extrapolation=True, **opts):
        """Returns a function that can be called on grid points to interpolate them

        :param grid:
        :type grid: Mesh
        :param vals:
        :type vals: np.ndarray
        :param interpolation_order:
        :type interpolation_order: int | str | None
        :param opts:
        :type opts:
        :return: interpolator
        :rtype: function
        """
        if grid.ndim == 1:
            interpolator = ProductGridInterpolator(
                grid,
                vals,
                order=interpolation_order,
                extrapolate=allow_extrapolation
            )
        elif (
                grid.mesh_type == MeshType.Structured
                or grid.mesh_type == MeshType.Regular
        ):
            interpolator = ProductGridInterpolator(
                grid.subgrids,
                vals,
                order=interpolation_order,
                extrapolate=allow_extrapolation
            )
        elif grid.mesh_type == MeshType.Unstructured:
            # for now we'll only use the RadialBasisFunction interpolator, but this may be extended in the future
            interpolator = UnstructuredGridInterpolator(
                grid,
                vals,
                order=interpolation_order,
                extrapolate=allow_extrapolation
            )
        elif grid.mesh_type == MeshType.SemiStructured:
            raise NotImplementedError("don't know what I want to do with semistructured meshes anymore")
        else:
            raise InterpolatorException("{}.{}: can't handle mesh_type '{}'".format(
                cls.__name__,
               'get_interpolator',
                grid.mesh_type
            ))

        return interpolator

    @classmethod
    def get_extrapolator(cls, grid, vals, extrapolation_order=1, **opts):
        """
        Returns an Extrapolator that can be called on grid points to extrapolate them

        :param grid:
        :type grid: Mesh
        :param extrapolation_order:
        :type extrapolation_order: int
        :return: extrapolator
        :rtype: Extrapolator
        """

        # Extrapolator(
        #     cls(
        #         grid,
        #         vals,
        #         interpolation_order=extrapolation_order,
        #         extrapolator=Extrapolator(lambda g: np.full(g.shape, np.nan))
        #     )
        # )

        if grid.ndim == 1:
            extrapolator = ProductGridInterpolator(
                grid,
                vals,
                order=extrapolation_order,
                extrapolate=True
            )
        elif (
                grid.mesh_type == MeshType.Structured
                or grid.mesh_type == MeshType.Regular
        ):
            extrapolator = ProductGridInterpolator(
                grid.subgrids,
                vals,
                order=extrapolation_order,
                extrapolate=True
            )
        elif grid.mesh_type == MeshType.Unstructured:
            # for now we'll only use the RadialBasisFunction interpolator, but this may be extended in the future
            extrapolator = UnstructuredGridInterpolator(
                grid,
                vals,
                neighbors=extrapolation_order+1,
                order=extrapolation_order,
                extrapolate=True
            )
        elif grid.mesh_type == MeshType.SemiStructured:
            raise NotImplementedError("don't know what I want to do with semistructured meshes anymore")
        else:
            raise InterpolatorException("{}.{}: can't handle mesh_type '{}'".format(
                cls.__name__,
               'get_interpolator',
                grid.mesh_type
            ))

        return Extrapolator(
            extrapolator,
            **opts
        )

    def apply(self, grid_points, **opts):
        """Interpolates then extrapolates the function at the grid_points

        :param grid_points:
        :type grid_points:
        :return:
        :rtype:
        """
        # determining what points are "inside" an interpolator region is quite tough
        # instead it is probably better to allow the basic thing to interpolate and do its thing
        # and then allow the extrapolator to post-process that result
        vals = self.interpolator(grid_points, **opts)
        if self.extrapolator is not None:
            vals = self.extrapolator(grid_points, vals)
        return vals

    def derivative(self, order):
        """
        Returns a new function representing the requested derivative
        of the current interpolator

        :param order:
        :type order:
        :return:
        :rtype:
        """
        return type(self)(
            self.grid,
            self.vals,
            interpolation_function=self.interpolator.derivative(order),
            extrapolator=self.extrapolator.derivative(order) if self.extrapolator is not None else self.extrapolator
        )

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

######################################################################################################
##
##                                   Extrapolator Class
##
######################################################################################################
class Extrapolator:
    """
    A general purpose that takes your data and just extrapolates it.
    This currently only exists in template format.
    """
    def __init__(self,
                 extrapolation_function,
                 warning=False,
                 **opts
                 ):
        """
        :param extrapolation_function: the function to handle extrapolation off the interpolation grid
        :type extrapolation_function: None | function | Callable | Interpolator
        :param warning: whether to emit a message warning about extrapolation occurring
        :type warning: bool
        :param opts: the options to feed into the extrapolator call
        :type opts:
        """
        self.extrapolator = extrapolation_function
        self.extrap_warning = warning
        self.opts = opts

    def derivative(self, n):
        return type(self)(
            self.extrapolator.derivative(n),
            warning=self.extrapolator
        )

    def find_extrapolated_points(self, gps, vals, extrap_value=np.nan):
        """
        Currently super rough heuristics to determine at which points we need to extrapolate
        :param gps:
        :type gps:
        :param vals:
        :type vals:
        :return:
        :rtype:
        """
        if extrap_value is np.nan:
            where = np.isnan(vals)
        elif extrap_value is np.inf:
            where = np.isinf(vals)
        elif not isinstance(extrap_value, (int, float, np.floating, np.integer)):
            where = np.logical_and(vals <= extrap_value[0], vals >= extrap_value[1])
        else:
            where = np.where(vals == extrap_value)

        return gps[where], where

    def apply(self, gps, vals, extrap_value=np.nan):
        ext_gps, inds = self.find_extrapolated_points(gps, vals, extrap_value=extrap_value)
        if len(ext_gps) > 0:
            new_vals = self.extrapolator(ext_gps)
            # TODO: emit a warning about extrapolating if we're doing so?
            vals[inds] = new_vals
        return vals

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)