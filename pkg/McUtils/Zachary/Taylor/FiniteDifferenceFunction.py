"""
Provides a general, convenient FiniteDifferenceFunction class to handle all of our difference FD imps
"""
import numpy as np, scipy.sparse as sparse
from ..Mesh import Mesh, MeshType
__reload_hook__ = [ '..Mesh' ]

__all__ = [
    'FiniteDifferenceFunction',
    'FiniteDifferenceError',
    'finite_difference',
    "FiniteDifference1D",
    "RegularGridFiniteDifference",
    "IrregularGridFiniteDifference",
    "FiniteDifferenceData",
    "FiniteDifferenceMatrix"
]

##########################################################################################
#
#                                    FiniteDifferenceFunction
#
##########################################################################################
class FiniteDifferenceError(Exception):
    pass
class FiniteDifferenceFunction:
    """
    The FiniteDifferenceFunction encapsulates a bunch of functionality extracted from [Fornberger's
    Calculation of Wieghts in Finite Difference Formulas](https://epubs.siam.org/doi/pdf/10.1137/S0036144596322507)

    Only applies to direct product grids, but each subgrid can be regular or irregular.
    Used in a large number of other places, but relatively rarely on its own.
    A convenient application is the `FiniteDifferenceDerivative` class in the `Derivatives` module.
    """
    def __init__(self, *diffs, axes=0, contract = False):
        """Constructs an object to take finite differences derivatives of grids of data

        :param diffs: A set of differences to take along successive axes in the data
        :type diffs: FiniteDifference1D
        :param axes: The axes to take the specified differences along
        :type axes: int | Iterable[int]
        :param contract: Whether to reduce the shape of the returned tensor if applicable after application
        :type contract: bool
        """
        self.differences = tuple(diffs)
        self.axes = axes
        self.contract = contract

    def apply(self, vals, axes=None, mesh_spacing = None, contract = None):
        """
        Iteratively applies the stored finite difference objects to the vals

        :param vals: The tensor of values to take the difference on
        :type vals: np.ndarray
        :param axes: The axis or axes to take the differences along (defaults to `self.axes`)
        :type axes: int | Iterable[int]
        :return: The tensor of derivatives
        :rtype: np.ndarray
        """
        dim = len(self.differences)
        val_dim = len(vals.shape)

        if contract is None:
            contract = self.contract

        axis = axes
        if axis is None:
            axis = self.axes
        if mesh_spacing is None or isinstance(mesh_spacing, (int, np.integer, float, np.floating)):
            mesh_spacing = [mesh_spacing] * dim

        # by default we'll assume dimension one is the slowest changing index, dimension 2 is the next slowest, etc.
        if isinstance(axis, (int, np.integer)):
            axis = axis + np.arange(dim)
        for i, m, h in zip(axis, self.differences, mesh_spacing):
            if m is not None:
                vals = m.apply(vals, val_dim = val_dim, axis = i, mesh_spacing = h)

        if contract:
            vals = vals.squeeze()
        return vals

    def __call__(self, vals, axes=None, mesh_spacing = None):
        return self.apply(vals, axes=axes, mesh_spacing = mesh_spacing)

    @property
    def order(self):
        """
        :return: the order of the derivative requested
        :rtype: tuple[int]
        """
        return tuple(d.order for d in self.differences)

    @property
    def weights(self):
        """
        :return: the weights for the specified stencil
        :rtype: tuple[np.array[float]]
        """
        return tuple(d.weights for d in self.differences)

    @property
    def widths(self):
        """
        :return: the number of points in each dimension, left and right, for the specified stencil
        :rtype: tuple[(int, int)]
        """
        return tuple(d.widths for d in self.differences)

    @classmethod
    def regular_difference(cls,
                           order,
                           mesh_spacing = None,
                           accuracy=2,
                           stencil=None,
                           end_point_accuracy=2,
                           axes=0,
                           contract=True,
                           **kwargs
                           ):
        """
        Constructs a `FiniteDifferenceFunction` appropriate for a _regular grid_ with the given stencil

        :param order: the order of the derivative
        :type order: tuple[int]
        :param mesh_spacing: the spacing between grid points in the regular grid `h`
        :type mesh_spacing: None | float | tuple[float]
        :param accuracy: the accuracy of the derivative that we'll try to achieve as a power on `h`
        :type accuracy: None | int | tuple[int]
        :param stencil: the stencil to use for the derivative (overrides `accuracy`)
        :type stencil: None | int | tuple[int]
        :param end_point_accuracy: the amount of extra accuracy to use at the edges of the grid
        :type end_point_accuracy: None | int | tuple[int]
        :param axes: the axes of the passed array for the derivative to be applied along
        :type axes: None | int | tuple[int]
        :param contract: whether to eliminate any axes of size `1` from the results
        :type contract: bool
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        dim = len(order)
        if isinstance(stencil, (int, np.integer)) or stencil is None:
            stencil = [stencil]*dim
        if isinstance(accuracy, (int, np.integer)) or accuracy is None:
            accuracy = [accuracy]*dim
        if isinstance(end_point_accuracy, (int, np.integer)) or end_point_accuracy is None:
            end_point_accuracy = [end_point_accuracy] * dim
        if mesh_spacing is None or isinstance(mesh_spacing, (int, np.integer, float, np.floating)):
            mesh_spacing = [mesh_spacing] * dim

        return cls(
            *(
                RegularGridFiniteDifference(
                    d,
                    stencil=s,
                    accuracy=a,
                    end_point_accuracy=e,
                    mesh_spacing = h,
                    **kwargs
                    ) for d, s, a, e, h in zip(order, stencil, accuracy, end_point_accuracy, mesh_spacing)
            ),
            axes=axes,
            contract=contract
        )

    @classmethod
    def from_grid(cls,
                  grid, order,
                  accuracy = 2,
                  stencil = None,
                  end_point_accuracy = 2,
                  axes = 0,
                  contract = True,
                  **kwargs
                  ):
        """
        Constructs a `FiniteDifferenceFunction` from a grid and order.
         Deconstructs the grid into its subgrids and builds a different differencer for each dimension

        :param grid: The grid to use as input data when defining the derivative
        :type grid: np.ndarray
        :param order: order of the derivative to compute
        :type order: int or list of ints
        :param stencil: number of points to use in the stencil
        :type stencil: int or list of ints
        :return: deriv func
        :rtype: FiniteDifferenceFunction
        """

        if isinstance(order, (int, np.integer)):
            order = (order,)
        dim = len(order)
        if isinstance(stencil, (int, np.integer)) or stencil is None:
            stencil = [stencil]*dim
        if isinstance(accuracy, (int, np.integer)) or accuracy is None:
            accuracy = [accuracy]*dim
        if isinstance(end_point_accuracy, (int, np.integer)) or end_point_accuracy is None:
            end_point_accuracy = [end_point_accuracy] * dim

        if axes is None:
            axes = grid.ndim - dim - 1
        if isinstance(axes, (int, np.integer)):
            axes = axes + np.arange(dim)
        axes = tuple(axes)

        if dim > 1:
            if grid.ndim < dim+1:
                raise FiniteDifferenceError("{}.{}: grid has {} dimensions but {} dimensional difference requested".format(
                    cls.__name__,
                    "from_grid",
                    grid.ndim-1,
                    dim
                ))

            # get the grid we actually will be doing the FD on
            sel_ax = axes + (grid.ndim-1,)
            missing = tuple(ax for ax in range(grid.ndim) if ax not in sel_ax)
            transp_ax = missing+sel_ax
            new_grid = grid.transpose(transp_ax)
            if len(missing)>0:
                new_grid = new_grid[(0,)*len(missing)]

            gp = Mesh(new_grid)
            subgrids = gp.subgrids#[sg[a] if a is not None else a for a in sel_ax]
            if subgrids is None:
                raise ValueError("can't apply finite difference to mesh of type {}".format(gp.mesh_type))
        else:
            subgrids = [grid]

        diffs = [None] * dim
        for i, d in enumerate(zip(order, stencil, accuracy, end_point_accuracy, subgrids)):
            o, s, a, e, g = d
            if o is None or o == 0:
                diffs[i] = None
            else:
                if len(g) == 0:
                    raise ValueError("finite difference can't be applied to subgrid {} of {}".format(g, grid))

                m = Mesh(g)
                # print(g[1]-g[0])
                if m.mesh_type == MeshType.Regular:
                    diffs[i] = RegularGridFiniteDifference(
                        o,
                        stencil=s,
                        accuracy=a,
                        end_point_accuracy=e,
                        mesh_spacing=g[1]-g[0],
                        **kwargs
                    )
                elif m.mesh_type is MeshType.Structured:
                    diffs[i] = IrregularGridFiniteDifference(
                        g,
                        o,
                        stencil=s,
                        accuracy=a,
                        end_point_accuracy=e,
                        **kwargs
                    )
                else:
                    raise ValueError("don't know how to do FD on a Mesh with type {}".format(m.mesh_type))
        return cls(*diffs, contract=contract, axes=axes)

##########################################################################################
#
#                                    FiniteDifference1D
#
##########################################################################################
class FiniteDifference1D:
    """
    A one-dimensional finite difference derivative object.
    Higher-dimensional derivatives are built by chaining these.
    """
    def __init__(self, finite_difference_data, matrix):
        self.data = finite_difference_data
        self.mat = matrix

    @property
    def order(self):
        return self.data.order
    @property
    def weights(self):
        return tuple(tuple(w) for w in self.data.weights)
    @property
    def widths(self):
        return tuple(tuple(w) for w in self.data.widths)

    @classmethod
    def get_stencil(cls, order, stencil, accuracy):
        if stencil is None:
            sten = order + accuracy - 1
        else:
            sten = stencil - 1
        return sten

    def apply(self, vals, val_dim=None, axis=0, mesh_spacing=None, check_shape=True):
        """
        Applies the held `FiniteDifferenceMatrix` to the array of values

        :param vals: values to do the difference over
        :type vals: np.ndarray | sparse.csr_matrix
        :param val_dim: dimensions of the vals
        :type val_dim: int
        :param axis: the axis to apply along
        :type axis: int | tuple[int]
        :param mesh_spacing: the mesh spacing for the weights
        :type mesh_spacing: float
        :return:
        :rtype: np.ndarray
        """

        if val_dim is None:
            val_dim = len(vals.shape)
        vs = vals.shape
        if val_dim == 1:
            # print(type(self.mat.matrix))
            if check_shape:
                if self.mat.only_core or self.mat.only_center:
                    shp = len(self.weights[len(self.weights) // 2])
                else:
                    shp = max(len(w) for w in self.weights)
                if shp != vs[0]:
                    if self.mat.only_core or self.mat.only_center:
                        raise FiniteDifferenceError(
                            "FD weights {} with can't be applied to data with shape {} along axis {}".format(
                                self.weights[len(self.weights) // 2],
                                self.weights,
                                vs,
                                axis
                            ))
                    else:
                        raise FiniteDifferenceError(
                            "FD weights {} with len {} can't be applied to data with shape {} along axis {}".format(
                                self.weights,
                                shp,
                                vs,
                                axis
                            ))
            if mesh_spacing is not None:
                self.mat.mesh_spacing = mesh_spacing
            self.mat.npts = vs[0]
            vals = self.mat.matrix.dot(vals)  # axis doesn't even make sense here...
            # print(vals.shape)
        else:
            if check_shape:
                if self.mat.only_core or self.mat.only_center:
                    shp = len(self.weights[ len(self.weights)//2 ])
                else:
                    shp = max(len(w) for w in self.weights)
                if shp != vs[axis]:
                    if self.mat.only_core or self.mat.only_center:
                        raise FiniteDifferenceError(
                            "FD weights {} with can't be applied to data with shape {} along axis {}".format(
                                self.weights[len(self.weights) // 2],
                                self.weights,
                                vs,
                                axis
                            ))
                    else:
                        raise FiniteDifferenceError(
                            "FD weights {} with len {} can't be applied to data with shape {} along axis {}".format(
                                self.weights,
                                shp,
                                vs,
                                axis
                            ))
            if mesh_spacing is not None:
                self.mat.mesh_spacing = mesh_spacing
            self.mat.npts = vs[axis]
            if self.mat.mode == "sparse":
                vals = self.sparse_tensordot(self.mat.matrix, vals, axis)
            else:
                vals = np.tensordot(self.mat.matrix, vals, axes=((1), (axis)))
            roll = np.concatenate((
                np.roll(np.arange(axis + 1), -1),
                np.arange(axis + 1, val_dim)
            ))
            vals = vals.transpose(roll)

        return vals

    @staticmethod
    def sparse_tensordot(sparse, mat, axis):
        """Not sure how fast this will be, but does a very simple contraction of `mat` along `axis` by the final axis of `sparse`

        Heavily de-generalized from here: https://github.com/pydata/sparse/blob/9dc40e15a04eda8d8efff35dfc08950b4c07a810/sparse/_coo/common.py
        :param sparse:
        :type sparse: sparse.sparsemat
        :param mat:
        :type mat: np.ndarray
        :param axis:
        :type axis:
        :return:
        :rtype:
        """

        axis_a = 1
        axis_b = axis

        as_ = sparse.shape
        nda = sparse.ndim
        bs = mat.shape
        ndb = mat.ndim

        if as_[axis_a] != bs[axis_b]:
            raise ValueError("shape-mismatch for sum")
        if axis_b < 0:
            axis_b += ndb

        # Move the axes to sum over to the end of "a"
        # and to the front of "b"
        notin = [k for k in range(nda) if k != axis_a]
        newaxes_a = notin + [axis_a]
        N2 = 1
        N2 *= as_[axis_a]
        newshape_a = (-1, N2)
        olda = [as_[axis] for axis in notin]

        notin = [k for k in range(ndb) if k != axis_b]
        newaxes_b = [axis_b] + notin
        N2 = 1
        N2 *= bs[axis_b]
        newshape_b = (N2, -1)
        oldb = [bs[axis] for axis in notin]

        # don't think I need this...
        # at = sparse.transpose(newaxes_a).reshape(newshape_a)
        bt = mat.transpose(newaxes_b).reshape(newshape_b)
        res = np.asarray(sparse.dot(bt))
        return res.reshape(olda + oldb)

class RegularGridFiniteDifference(FiniteDifference1D):
    """
    Defines a 1D finite difference over a regular grid
    """
    def __init__(self,
                 order,
                 stencil = None,
                 accuracy = 4,
                 end_point_accuracy = 2,
                 **kw
                 ):
        """

        :param order: the order of the derivative to take
        :type order: int
        :param stencil: the number of stencil points to add
        :type stencil: int | None
        :param accuracy: the approximate accuracy to target with the method
        :type accuracy: int | None
        :param end_point_accuracy: the extra number of stencil points to add to the end points
        :type end_point_accuracy: int | None
        :param kw: options passed through to the `FiniteDifferenceMatrix`
        :type kw:
        """

        data = self.finite_difference_data(order, self.get_stencil(order, stencil, accuracy), end_point_accuracy)
        mat = FiniteDifferenceMatrix(data, **kw)
        super().__init__(data, mat)

    @classmethod
    def finite_difference_data(cls, order, stencil, end_point_precision):
        """Builds a FiniteDifferenceData object from an order, stencil, and end_point_precision

        :param order:
        :type order:
        :param stencil:
        :type stencil:
        :param end_point_precision:
        :type end_point_precision:
        :return:
        :rtype:
        """

        m = order
        outer_stencil = stencil + end_point_precision
        lefthand_coeffs = cls.get_weights(m, 0, outer_stencil)
        centered_coeffs = cls.get_weights(m, np.math.ceil(stencil / 2), stencil)
        righthand_coeffs = cls.get_weights(m, outer_stencil, outer_stencil)
        sten = len(centered_coeffs)
        widths = [
            [0, len(lefthand_coeffs)],
            [np.math.ceil(sten / 2), np.math.floor(sten / 2)],
            [len(righthand_coeffs), 0]
        ]
        coeffs = [lefthand_coeffs, centered_coeffs, righthand_coeffs]

        return FiniteDifferenceData(coeffs, widths, order)

    @staticmethod
    def get_weights(m, s, n):
        """Extracts the weights for an evenly spaced grid

        :param m:
        :type m:
        :param s:
        :type s:
        :param n:
        :type n:
        :return:
        :rtype:
        """
        from .ZachLib import EvenFiniteDifferenceWeights

        return EvenFiniteDifferenceWeights(m, s, n)

class IrregularGridFiniteDifference(FiniteDifference1D):
    """
    Defines a finite difference over an irregular grid
    """
    def __init__(self,
                 grid,
                 order,
                 stencil=None,
                 accuracy=2,
                 end_point_accuracy=2,
                 **kw
                 ):
        """

        :param grid: the grid to get the weights from
        :type grid: np.ndarray
        :param order: the order of the derivative to take
        :type order: int
        :param stencil: the number of stencil points to add
        :type stencil: int | None
        :param accuracy: the approximate accuracy to target with the method
        :type accuracy: int | None
        :param end_point_accuracy: the extra number of stencil points to add to the end points
        :type end_point_accuracy: int | None
        :param kw: options passed through to the `FiniteDifferenceMatrix`
        :type kw:
        """
        data = self.finite_difference_data(grid, order, self.get_stencil(order, stencil, accuracy), end_point_accuracy)
        mat = FiniteDifferenceMatrix(data, npts=len(grid), mesh_spacing=1, **kw)
        super().__init__(data, mat)

    @staticmethod
    def get_grid_slices(grid, stencil):
        """

        :param grid:
        :type grid:
        :param stencil:
        :type stencil:
        :return:
        :rtype:
        """
        return [grid[a:b] for a,b in zip(range(0, len(grid)-stencil), range(stencil, len(grid)))]

    @staticmethod
    def get_weights(m, z, x):
        """Extracts the grid weights for an unevenly spaced grid based off of the algorithm outlined by
        Fronberger in https://pdfs.semanticscholar.org/8bf5/912bde884f6bd4cfb4991ba3d077cace94c0.pdf

        :param m: highest derivative order
        :type m:
        :param z: center of the derivatives
        :type z:
        :param X: grid of points
        :type X:
        """
        from .ZachLib import UnevenFiniteDifferenceWeights  # loads from C extension

        x = np.asarray(x)
        return UnevenFiniteDifferenceWeights(m, z, x).T

    @classmethod
    def finite_difference_data(cls, grid, order, stencil, end_point_precision):
        """Constructs a finite-difference function that computes the nth derivative with a given width

                :param deriv:
                :type deriv:
                :param accuracy:
                :type accuracy:
                :return:
                :rtype:
                """

        outer_stencil = stencil + end_point_precision

        # we're gonna do just the 1D case for now
        left_pad = np.math.floor(stencil / 2)
        right_pad = np.math.ceil(stencil / 2)
        slices_outer = cls.get_grid_slices(grid, outer_stencil)
        slices_core = cls.get_grid_slices(grid, stencil)
        slices_left = slices_outer[:left_pad]
        slices_right = slices_outer[-right_pad:]
        coeffs = [
            [cls.get_weights(order, x[0], x) for x in slices_left],
            [cls.get_weights(order, x[left_pad], x) for x in slices_core],
            [cls.get_weights(order, x[-1], x) for x in slices_right]
            ]
        widths = [(outer_stencil, 0), (left_pad, right_pad), (0, outer_stencil)]

        return FiniteDifferenceData(coeffs, widths, order)

##########################################################################################
#
#                                    FiniteDifferenceData
#
##########################################################################################
class FiniteDifferenceData:
    """
    Holds the data used by to construct a finite difference matrix
    """
    def __init__(self, weights, widths, order):
        self._weights = self._clean_coeffs(weights)  # used to apply the weights to function values
        # NOTE: the coefficients should be supplied without 'h' if an evenly spaced grid
        # These will be forced back in later
        self._widths = self._clean_widths(widths)  # used to determine how to apply the coefficients
        self._order = self._clean_order(order)  # the order of the derivative

    # region Properties
    @property
    def weights(self):
        return tuple(tuple(x) for x in self._weights)
    @property
    def widths(self):
        return tuple(tuple(x) for x in self._widths)
    @property
    def order(self):
        return self._order
    #endregion

    @classmethod
    def _clean_coeffs(cls, cfs):
        try:
            cl, cc, cr = cfs
            islcr = True
        except ValueError:
            islcr = False
        if not islcr:
            raise TypeError("{}: coefficients {} is expected to be list of list coefficients on the left, center, and right",
                            cls.__name__, cfs)
        return cfs
    @classmethod
    def _clean_order(cls, order):
        if not isinstance(order, (int, np.integer)):
            raise TypeError("{}: order {} is expected to be an int",
                            cls.__name__, order)
        return order
    @classmethod
    def _clean_widths(cls, ws):

        if isinstance(ws, (int, float, np.integer, np.floating)):
            ws = ( (int(ws), int(ws)) )
        else:
            w2s = [ None ] * len(ws)
            for i, w in enumerate(ws):
                if isinstance(w, (int, float, np.integer, np.floating)):
                    w2s[i] = (int(w), int(w))
                elif w is None:
                    pass
                elif isinstance(w[0], (int, float, np.integer, np.floating)):
                    w2s[i] = (int(w[0]), int(w[1]))
                elif w is not None:
                    w2s[i] = cls._clean_widths(w)
            ws = tuple(w2s)
        return ws

##########################################################################################
#
#                                    FiniteDifferenceMatrix
#
##########################################################################################
class FiniteDifferenceMatrix:
    """
    Defines a matrix that can be applied to a regular grid of values to take a finite difference
    """

    def __init__(self,
                 finite_difference_data,
                 npts = None, mesh_spacing = None,
                 only_core = False, only_center = False,
                 mode = "sparse", dtype = "float64"
                 ):
        """
        :param finite_difference_data:
        :type finite_difference_data: FiniteDifferenceData
        :param npts:
        :type npts:
        :param mesh_spacing:
        :type mesh_spacing:
        :param only_core:
        :type only_core:
        :param only_center:
        :type only_center:
        :param mode:
        :type mode:
        """
        self.data = finite_difference_data
        self._npts = npts
        self._mesh_spacing = mesh_spacing
        self._only_core = only_core
        self._only_center = only_center
        self._mode = mode
        self._mat = None
        self._dtype = dtype

    #region Properties
    @property
    def weights(self):
        return self.data.weights
    @property
    def order(self):
        return self.data.order
    @property
    def npts(self):
        return self._npts
    @npts.setter
    def npts(self, val):
        if val != self._npts:
            self._mat = None
            self._npts = val
    @property
    def mesh_spacing(self):
        return self._mesh_spacing
    @mesh_spacing.setter
    def mesh_spacing(self, val):
        if val != self._mesh_spacing:
            self._mat = None
            self._mesh_spacing = val
    @property
    def only_core(self):
        return self._only_core
    @only_core.setter
    def only_core(self, val):
        if val != self._only_core:
            self._mat = None
            self._only_core = val
    @property
    def only_center(self):
        return self._only_center
    @only_center.setter
    def only_center(self, val):
        if val != self._only_center:
            self._mat = None
            self._only_center = val
    @property
    def mode(self):
        return self._mode
    @mode.setter
    def mode(self, val):
        if val != self._mode:
            self._mat = None
            self._mode = val
    @property
    def dtype(self):
        return self._dtype
    @dtype.setter
    def dtype(self, val):
        if val != self._dtype:
            self._mat = None
            self._dtype = val
    @property
    def matrix(self):
        if self._mat is None:
            self._mat = self.fd_matrix()
        return self._mat
    #endregion

    def fd_matrix(self):
        """Builds a 1D finite difference matrix for a set of boundary weights, central weights, and num of points
        Will look like:
            b1 b2 b3 ...
            w1 w2 w3 ...
            0  w1 w2 w3 ...
            0  0  w1 w2 w3 ...
                 ...
                 ...
                 ...
                    .... b3 b2 b1
        :return: fd_mat
        :rtype: np.ndarray | sp.csr_matrix
        """
        npts = int(self.npts)
        only_core = self.only_core
        only_center = self.only_center
        o = self.order
        h = self.mesh_spacing
        dtype = self.dtype
        if h is not None:
            c_left, c_center, c_right = [np.array(x, dtype=dtype) / (h**o) for x in self.weights]
        else:
            c_left, c_center, c_right = [np.array(x, dtype=dtype) for x in self.weights]
        mode = self.mode

        if isinstance(c_center[0], (int, float, np.integer, np.floating)):
            fdm = self._fdm_regular(c_left, c_center, c_right, npts, only_core, only_center, mode, dtype)
        else:
            fdm = self._fdm_irregular(c_left, c_center, c_right, npts, only_core, only_center, mode, dtype)

        return fdm

    @classmethod
    def _fdm_irregular(cls, c_left, c_center, c_right, npts, only_core, only_center, mode, dtype):
        lcc = len(c_center)
        if only_center:
            shape = (1, npts)
            if mode == "sparse":
                fdm = sparse.lil_matrix(shape)
            else:
                fdm = np.zeros(shape)
            lcf = len(c_left)
            mid = np.math.floor(lcc)
            x = c_center[mid]
            p = lcf+mid
            fdm[p:p+len(x)] = x
        elif only_core:
            lcf = len(c_left)
            shape = (npts - lcc + (lcc % 2), npts)
            if mode == "sparse":
                fdm = sparse.lil_matrix(shape)
            else:
                fdm = np.zeros(shape)
            for i, x in enumerate(c_center):
                p = lcf+i
                fdm[i, p:p+lcc] = x
        else:
            shape = (npts, npts)
            if mode == "sparse":
                fdm = sparse.lil_matrix(shape)
            else:
                fdm = np.zeros(shape)
            lcf = len(c_left)
            for i, x in enumerate(c_left):
                fdm[i, i:i+lcf] = x
            for j, x in enumerate(c_center):
                fdm[j, j:j+lcc] = x
            lcr = len(c_right)
            for k, x in enumerate(c_right):
                fdm[-k, -(k+lcr):-k] = x

        return fdm

    @classmethod
    def _fdm_regular(cls, c_left, c_center, c_right, npts, only_core, only_center, mode, dtype):
        lcc = len(c_center)
        if only_center:

            if mode == "sparse":
                fdm = sparse.lil_matrix((1, npts), dtype=dtype)
            else:
                fdm = np.zeros((1, npts))
            grid_mid = np.math.floor((npts - lcc + (lcc % 2))/2)
            x = c_center
            p = grid_mid
            fdm[0, p:p + len(x)] = x
        elif only_core:
            fdm = cls._fdm_core(c_center, npts - lcc + (lcc % 2), npts, mode=mode)
        else:
            bound_l = min(int(np.floor(lcc / 2)), int(np.floor(npts / 2)))
            # TODO: sometimes the grid-point stencils overrun the grid I actually have (for small grids)
            # which means I need to handle these boundary cases better

            bdm_l = cls._fdm_core(c_left, bound_l, npts, mode=mode)
            bdm_r = cls._fdm_core(c_right, bound_l, npts, side="r", mode=mode)
            core = cls._fdm_core(c_center, npts - lcc + (lcc % 2), npts, mode=mode)
            if mode == "sparse":
                fdm = sparse.vstack((bdm_l, core, bdm_r))
                # print("\n", bdm_r.todense())
            else:
                fdm = np.array(np.vstack((bdm_l, core, bdm_r)))
                # print("\n", fdm)

        return fdm

    @classmethod
    def _fdm_core(cls, a, n1, n2, side="l", mode = "dense"):
        # pulled from https://stackoverflow.com/a/52464135/5720002
        # returns the inner blocks of the FD mat
        if mode == "sparse":
            offs = (n2-len(a)-n1+1) + np.arange(len(a)) if side == "r" else np.arange(len(a))
            mmm = sparse.diags(a, offsets=offs, shape=(n1, n2), dtype=a.dtype)
        else:
            a = np.asarray(a)
            p = np.zeros(n2, dtype=a.dtype)
            b = np.concatenate((p, a, p))
            s = b.strides[0]
            strided = np.lib.stride_tricks.as_strided
            if side == "r":
                mmm = strided(b[len(a) + n1 - 1:], shape=(n1, n2), strides=(-s, s))
            else:
                mmm = strided(b[n2:], shape=(n1, n2), strides=(-s, s))
        return mmm

##########################################################################################
#
#                                    finite_difference
#
##########################################################################################

def finite_difference(grid, values, order,
                      accuracy=2,
                      stencil=None,
                      end_point_accuracy=1,
                      axes=None,
                      only_core=False,
                      only_center=False,
                      dtype="float64",
                      **kw
                      ):
    """Computes a finite difference derivative for the values on the grid

    :param grid: the grid of points for which the vlaues lie on
    :type grid: np.ndarray
    :param values: the values on the grid
    :type values: np.ndarray
    :param order: order of the derivative to compute
    :type order: int | Iterable[int]
    :param stencil: number of points to use in the stencil
    :type stencil: int | Iterable[int]
    :param accuracy: approximate accuracy of the derivative to request (overridden by `stencil`)
    :type accuracy: int | Iterable[int]
    :param end_point_accuracy: extra stencil points to use on the edges
    :type end_point_accuracy: int | Iterable[int]
    :param end_point_accuracy: extra stencil points to use on the edges
    :param axes: which axes to perform the successive derivatives over (defaults to the first _n_ axes)
    :type axes: int | Iterable[int]
    :param only_center: whether or not to only take the central value
    :type only_center: bool
    :param only_core: whether or not to avoid edge values where a different stencil would be used
    :type only_core: bool
    :return:
    :rtype:
    """
    gv = np.asarray(values)
    if 'end_point_precision' in kw:
        end_point_accuracy = kw['end_point_precision']
    if 'axis' in kw:
        axes = kw['axis']
    func = FiniteDifferenceFunction.from_grid(
        grid, order,
        accuracy = accuracy,
        stencil = stencil,
        end_point_accuracy = end_point_accuracy,
        axes=axes,
        dtype=dtype,
        only_core=only_core,
        only_center=only_center,
        **kw
    )
    return func(gv)
    # if func.gridtype != FiniteDifferenceFunction.IRREGULAR_GRID:
    #     return fdf(gv)
    # else:
    #     return fdf(gv)