"""
Defines classes for providing different approaches to fitting.
For the most part, the idea is to use `scipy.optimize` to do the actual fitting process,
but we layer on conveniences w.r.t. specification of bases and automation of the actual fitting process
"""

import numpy as np, scipy.optimize as opt, enum
from collections import OrderedDict as odict

__all__ = [
    "FittableModel",
    "LinearFittableModel",
    "LinearFitBasis"
]

class FittableModel:
    """
    Defines a model that can be fit
    """
    def __init__(self, parameters, function, pre_fit = False, covariance = None):
        if not isinstance(parameters, odict):
            ...
        elif isinstance(parameters, np.ndarray) or isinstance(parameters[0], (int, float, np.floating, np.integer)):
            param_names = ["c"+str(i) for i in range(len(parameters))]
            parameters = odict(zip(param_names, parameters))
        else:
            parameters = odict(parameters)
        self.initial_parameters = parameters
        self._param_names = list(parameters.keys())
        if pre_fit:
            self._parameter_values = np.array(parameters.values())
        else:
            self._parameter_values = None
        self.covariance = covariance
        self.func = function

    @property
    def parameters(self):
        if not self.fitted:
            raise ValueError("{}: model hasn't been fitted".format(type(self).__name__))
        return odict(zip(self._param_names, self._parameter_values))
    @property
    def parameter_values(self):
        return self._parameter_values
    @property
    def parameter_names(self):
        return self._param_names
    @property
    def fitted(self):
        return self._parameter_values is not None

    def fit(self, xdata, ydata=None, fitter=None, **methopts):
        """
        Fits the model to the data using scipy.optimize.curve_fit or a function that provides the same interface

        :param points:
        :type points:
        :param methopts:
        :type methopts:
        :return:
        :rtype:
        """
        if fitter is None:
            fitter = opt.curve_fit

        if self.fitted:
            p0 = self._parameter_values
        else:
            p0 = np.array(self.initial_parameters.values())

        if 'p0' not in methopts:
            methopts['p0'] = p0

        if ydata is None:
            ydata = xdata[:, -1]
            xdata = xdata[:, :-1]
        params, cov = fitter(xdata, ydata, **methopts)
        self._parameter_values = params
        self.covariance = cov

    def get_parameter(self, name):
        """
        Returns the fitted value of the parameter given by 'name'
        :param name:
        :type name:
        :return:
        :rtype:
        """
        try:
            ind = self._param_names.index(name)
        except IndexError:
            ind = None
        if ind is None:
            raise KeyError("{}: model has no parameter '{}'".format(type(self).__name__, name))
        return self._parameter_values[ind]
    def __getitem__(self, item):
        return self.get_parameter(item)

    def evaluate(self, xdata):
        if not self.fitted:
            raise ValueError("{}: model hasn't been fitted".format(type(self).__name__))
        return self.func(xdata, *self._parameter_values)
    def __call__(self, xdata):
        return self.evaluate(xdata)

class LinearFittableModel(FittableModel):
    """
    Defines a class of models that can be expressed as linear expansions of basis functions.
    We _could_ define an alternate fit function by explicitly building & fitting a design matrix, but I think we're good on that for now
    """

    def __init__(self, basis, initial_params=None, pre_fit=False, covariance=None):
        if isinstance(basis, LinearFitBasis):
            names = basis.names
            funcs = basis.functions
            if initial_params is None:
                initial_params = odict((n, 1) for n in names)
        if initial_params is None:
            initial_params = [1]*len(basis)
        self.basis = basis
        func = lambda xdata, *coefs, basis=basis: np.dot(np.array(coefs), np.array([b(xdata) for b in basis]))
        super().__init__(initial_params, func, pre_fit=pre_fit, covariance=covariance)

    def evaluate(self, xdata):
        if not self.fitted:
            raise ValueError("{}: model hasn't been fitted".format(type(self).__name__))
        return np.dot(self.parameter_values, np.array([b(xdata) for b in self.basis]))

class LinearFitBasis:
    """
    Provides a container to build bases of functions for fitting.
    Asks for a generator for each dimension, which is just a function that takes an integer and returns a basis function at that order.
    Product functions are taken up to some max order
    """
    def __init__(self, *generators, order=3):
        """

        :param generators: the generating functions for the bases in each dimenion
        :type generators: Iterable[function]
        :param order: the maximum order for the basis functions (currently turning off coupling isn't possible, but that could come)
        :type order: int
        """
        self.ndim = len(generators)
        self.generators = generators
        self._order = order
        self._functions = None
        self._basis_names = None

    @property
    def functions(self):
        if self._functions is None:
            self._functions, self._basis_names = self.construct_basis()
        return self._functions
    @property
    def names(self):
        if self._basis_names is None:
            self._functions, self._basis_names = self.construct_basis()
        return self._basis_names
    @property
    def order(self):
        return self._order
    @order.setter
    def order(self, order):
        if not isinstance(order, int):
            raise TypeError("{}: basis order must be an int, got '{}'".format(
                type(self).__name__,
                order
            ))
        self._functions = None
        self._order = order
    def construct_basis(self):
        import itertools as ip

        if self.ndim > 1:
            inds = ip.product(range(self._order), repeat=self.ndim)
            funcs = []
            names = []
            for i in inds:
                # filter to only pick up basis functions where the
                if sum(i) <= self._order:
                    prods = tuple(f(n) for f,n in zip(self.generators, i))
                    # build basis function by taking products
                    funcs.append(lambda xdata, fs=prods: np.prod([f(xdata[..., n]) for n, f in enumerate(fs)]))
                    names.append("c"+"".join(str(n) for n in i))
        else:
            funcs = [self.generators[0](n) for n in range(self._order)]
            names = ["c"+str(n) for n in range(self._order)]

        return funcs, names

    # just here for convenience
    fourier_series = lambda x,k: np.cos(k*x)
    power_series = lambda x,n: x**n



