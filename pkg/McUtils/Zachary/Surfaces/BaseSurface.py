"""
Provides an abstract base class off of which concrete surface implementations can be built
"""

import abc, numpy as np

__all__ = [
    "BaseSurface",
    "TaylorSeriesSurface",
    "LinearExpansionSurface",
    "LinearFitSurface",
    "InterpolatedSurface"
]

class BaseSurface(metaclass=abc.ABCMeta):
    """
    Surface base class which can be subclassed for relevant cases
    """
    def __init__(self, data, dimension):
        # we'll just give the core data a consistent attribute name
        self.data = data
        self.dimension = dimension
        # there's no 1000% general way to assess the dimension, so we just let it be passed
    @abc.abstractmethod
    def evaluate(self, points, **kwargs):
        """
        Evaluates the function at the points based off of "data"

        :param points:
        :type points:
        :return:
        :rtype:
        """
        raise NotImplemented
    def __call__(self, gridpoints, **kwargs):
        """

        :param gridpoints:
        :type gridpoints: np.ndarray
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        if isinstance(gridpoints, (int, float, np.floating, np.integer)):
            gridpoints = np.array([gridpoints])
        gridpoints = np.asarray(gridpoints)
        if self.dimension is not None:
            if self.dimension == 1:
                if gridpoints.ndim > 1:
                    raise ValueError("{}: dimension mismatch in call, surface expects grid points to be 1D".format(
                        type(self).__name__
                    ))
            else:
                gp_dim = gridpoints.shape[-1]
                if gp_dim != self.dimension:
                    raise ValueError("{}: dimension mismatch in call, grid points had dim {} but surface expects dim {}".format(
                        type(self).__name__,
                        gp_dim,
                        self.dimension
                    ))
        return self.evaluate(gridpoints, **kwargs)

    def minimize(self, initial_guess, function_options=None, **opts):
        """
        Just calls into `scipy.optimize.minimize` as the default implementation

        :param initial_guess: starting position for the minimzation
        :type initial_guess: np.ndarray
        :param function_options:
        :type function_options: dict | None
        :param opts:
        :type opts:
        :return:
        :rtype:
        """

        import scipy.optimize as opt

        if function_options is None:
            function_options = {}
        test_call = self(initial_guess, **function_options) # if this fails the initial guess is bad

        func = lambda x, f=self.evaluate, o=function_options: f(x, **o)

        min = opt.minimize(func, initial_guess, **opts)




class TaylorSeriesSurface(BaseSurface):
    """
    A surface with an evaluator built off of a Taylor series expansion
    """
    def __init__(self, *derivs, dimension=None, **opts):
        """
        :param data: derivs or a tuple of derivs + options
        :type data:
        :param dimension:
        :type dimension:
        """
        from ..Taylor import FunctionExpansion

        if dimension is None:
            dimension = len(derivs[0])
        self.expansion = FunctionExpansion(derivs, **opts)
        opts["derivs"] = derivs
        super().__init__(opts, dimension)

    def evaluate(self, points, **kwargs):
        """
        Since the Taylor expansion stuff is already built out this is super easy

        :param points:
        :type points:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return self.expansion(points, **kwargs)

class LinearExpansionSurface(BaseSurface):
    """
    A surface with an evaluator built off of an expansion in some user specified basis
    """
    def __init__(self, coefficients, basis=None, dimension=None):
        """
        :param coefficients: the expansion coefficients in the basis
        :type coefficients: np.ndarray
        :param basis: a basis of functions to use (defaults to power series)
        :type basis: Iterable[function] | None
        """
        if basis is None:
            basis = [(lambda x,n=n: x**n) for n in range(len(coefficients))]
        self.basis = basis
        self.coeffs = coefficients
        super().__init__({"basis":basis, "coeffs":coefficients}, dimension)

    def evaluate(self, points, **kwargs):
        """
        First we just apply the basis to the gridpoints, then we dot this into the coeffs

        :param points:
        :type points:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        basis_values = [f(points, **kwargs) for f in self.basis]
        return np.dot(self.coeffs, basis_values)

#TODO: add in a NonLinearFitSurface...
class LinearFitSurface(LinearExpansionSurface):
    """
    A surface built off of a LinearExpansionSurface, but done by fitting.
    The basis selection
    """
    def __init__(self,
                 points,
                 basis=None,
                 order=4,
                 dimension=None
                 ):
        """
        :param points: a set of points to fit to
        :type points: np.ndarray
        :param basis: a basis of functions to use (defaults to power series)
        :type basis: Iterable[function] | None
        """
        from ..FittableModels import LinearFittableModel, LinearFitBasis

        dim = points.shape[-1]
        if basis is None:
            basis = LinearFitBasis([LinearFitBasis.power_series]*dim, order=order)
        self.model = LinearFittableModel(basis)
        self.model.fit(points)
        self.fit_data = points
        super().__init__(self.model.parameters, basis, dimension)

    def evaluate(self, points, **kwargs):
        """

        :param points:
        :type points:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        return self.model.evaluate(points)

    # Is there any way to cleverly minimize a general function like this? My guess is I'd need to know how to take derivs
    def minimize(self, initial_guess=None, function_options=None, **opts):
        if initial_guess is None:
            initial_guess = self.fit_data[np.argmin(self.fit_data[:, -1])][:-1]
        return super().minimize(initial_guess, function_options=function_options, **opts)

class InterpolatedSurface(BaseSurface):
    """
    A surface that operates by doing an interpolation of passed mesh data
    """
    def __init__(self, xdata, ydata=None, dimension=None, **opts):
        from ..Interpolator import Interpolator
        if ydata is None:
            ydata = xdata[...,  -1]
            xdata = xdata[..., :-1]
        dimension = 1 if xdata.ndim == 1 else xdata.shape[-1]
        self.interp_data = (xdata, ydata)
        self.interp = Interpolator(xdata, ydata, **opts)
        super().__init__({"interpolator": self.interp}, dimension)

    def evaluate(self, points, **kwargs):
        """
        We delegate all the dirty work to the Interpolator so hopefully that's working...
        :param points:
        :type points:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return self.interp(points)

    def minimize(self, initial_guess=None, function_options=None, **opts):
        if initial_guess is None:
            initial_guess = self.interp_data[0][np.argmin(self.interp_data[1], axis=-1)]
        return super().minimize(initial_guess, function_options=function_options, **opts)
