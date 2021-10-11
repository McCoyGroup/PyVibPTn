from ..LazyTensors import Tensor
from .Derivatives import FiniteDifferenceDerivative
from ...Coordinerds import CoordinateSet, CoordinateSystem
from ..TensorDerivativeConverter import TensorDerivativeConverter
from ...Numputils import vec_tensordot
import numpy as np, itertools

__all__ = [
    'FunctionExpansion'
]

########################################################################################################################
#
#                                           FunctionExpansion
#
class FunctionExpansionException(Exception):
    pass
class FunctionExpansion:
    """
    A class for handling expansions of an internal coordinate potential up to 4th order
    Uses Cartesian derivative matrices and the Cartesian <-> Internal normal mode Jacobian
    """

    def __init__(self,
                 derivatives,
                 transforms=None,
                 center=None,
                 ref=0,
                 weight_coefficients=True
                 ):
        """

        :param derivatives: Derivatives of the function being expanded
        :type derivatives: Iterable[np.ndarray | Tensor]
        :param transforms: Jacobian and higher order derivatives in the coordinates
        :type transforms: Iterable[np.ndarray | Tensor] | None
        :param center: the reference point for expanding aobut
        :type center: np.ndarray | None
        :param ref: the reference point value for shifting the expansion
        :type ref: float | np.ndarray
        :param weight_coefficients: whether the derivative terms need to be weighted or not
        :type weight_coefficients: bool
        """

        # raise NotImplementedError("doesn't deal with higher-order expansions properly yet")
        self._derivs = self.FunctionDerivatives(derivatives, weight_coefficients)
        self._center = center
        self.ref = ref
        if transforms is None:
            self._transf = None
        else:
            # transformation matrices from cartesians to internals
            self._transf = self.CoordinateTransforms(transforms)
        self._tensors = None

    @classmethod
    def expand_function(cls,
                        f, point,
                        order=4,
                        basis=None,
                        function_shape=None,
                        transforms=None,
                        weight_coefficients=True,
                        **fd_options
                        ):
        """
        Expands a function about a point up to the given order

        :param f:
        :type f: function
        :param point:
        :type point: np.ndarray | CoordinateSet
        :param order:
        :type order: int
        :param basis:
        :type basis: None | CoordinateSystem
        :param fd_options:
        :type fd_options:
        :return:
        :rtype:
        """

        derivs = FiniteDifferenceDerivative(f, function_shape=function_shape)(point, **fd_options)
        ref = f(point)
        dts = [derivs.derivative_tensor(i) for i in range(1, order+1)]
        if transforms is None:
            if basis is not None and isinstance(point, CoordinateSet):
                transforms = [point.jacobian(basis, order=i) for i in range(order)]
            else:
                transforms = None

        return cls(dts, center=point, ref=ref, transforms=transforms, weight_coefficients=weight_coefficients)

    @property
    def expansion_tensors(self):
        """
        Provides the tensors that will contracted

        :return:
        :rtype:
        """
        if self._tensors is None:
            if self._transf is None:
                self._tensors = self._derivs
            else:
                self._tensors = TensorDerivativeConverter(self._transf, self._derivs).convert()
        return self._tensors

    def get_expansions(self, coords, squeeze=True):
        """

        :param coords: Coordinates to evaluate the expansion at
        :type coords: np.ndarray | CoordinateSet
        :return:
        :rtype:
        """

        if self._center is None:
            # we assume we have a vector of points
            disp = coords
            coord_axis = 1
        else:
            if coords.shape == self._center.shape:
                coords = coords[np.newaxis]
            disp = coords - self._center
            coord_axis = coords.ndim - self._center.ndim
        expansions = []
        for i, t in enumerate(self.expansion_tensors):
            # contract the tensor by the displacements until it's completely reduced
            tensr = t #type: np.ndarray
            for j in range(i+1):
                try:
                    if j == 0:
                        tensr = np.tensordot(disp, tensr, axes=[[coord_axis], [tensr.ndim-1]])
                    else:
                        tensr = vec_tensordot(disp, tensr, axes=[[coord_axis], [tensr.ndim-1]])
                except:
                    raise Exception(disp.shape, tensr.shape, coord_axis, tensr.ndim)
            contraction = tensr
            if squeeze:
                contraction = contraction.squeeze()
            expansions.append(contraction)

        return expansions
    def expand(self, coords, squeeze = True):
        """Returns a numerical value for the expanded coordinates

        :param coords:
        :type coords: np.ndarray
        :return:
        :rtype: float | np.ndarray
        """
        ref = self.ref
        exps = self.get_expansions(coords, squeeze = squeeze)
        return ref + sum(exps)

    def __call__(self, coords, **kw):
        return self.expand(coords, **kw)

    class CoordinateTransforms:
        def __init__(self, transforms):
            self._transf = [np.asanyarray(t) for t in transforms]
        def __getitem__(self, i):
            if len(self._transf) < i:
                raise FunctionExpansionException("{}: transformations requested up to order {} but only provided to order {}".format(
                    type(self).__name__,
                    i,
                    len(self._transf)
                ))
            return self._transf[i]
        def __len__(self):
            return len(self._transf)
    class FunctionDerivatives:
        def __init__(self, derivs, weight_coefficients=True):
            self.derivs = [ np.asanyarray(t) for t in derivs ]
            if weight_coefficients:
                self.derivs = [self.weight_derivs(t, o+1) if weight_coefficients else t for o, t in enumerate(self.derivs)]
        def weight_derivs(self, t, order = None):
            """

            :param order:
            :type order: int
            :param t:
            :type t: Tensor
            :return:
            :rtype:
            """
            weighted = t
            if order is None:
                order = len(t.shape)
            if order > 1:
                weighted = weighted * (1/np.math.factorial(order))
                # s = t.shape
                # weights = np.ones(s)
                # all_inds = list(range(len(s)))
                # for i in range(2, order+1):
                #     for inds in itertools.combinations(all_inds, i):
                #         # define a diagonal slice through
                #         sel = tuple(slice(None, None, None) if a not in inds else np.arange(s[a]) for a in all_inds)
                #         weights[sel] = 1/np.math.factorial(i)
                # weighted = weighted.mul(weights)
                # print(weights, weighted.array)

            return weighted

        def __getitem__(self, i):
            if len(self.derivs) < i:
                raise FunctionExpansionException("{}: derivatives requested up to order {} but only provided to order {}".format(
                    type(self).__name__,
                    i,
                    len(self.derivs)
                ))
            return self.derivs[i]
        def __len__(self):
            return len(self.derivs)