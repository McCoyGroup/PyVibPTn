
from .BaseSurface import *
import numpy as np
from collections import namedtuple

__all__ = [
    "Surface",
    "MultiSurface"
]

class Surface:
    """
    This actually isn't a concrete implementation of BaseSurface.
    Instead it's a class that _dispatches_ to an implementation of BaseSurface to do its core evaluations (plus it does shape checking)
    """
    def __init__(self, data, dimension=None, base=None, **metadata):
        """

        :param data:
        :type data:
        :param dimension:
        :type dimension:
        :param base:
        :type base: None | Type[BaseSurface]
        :param metadata:
        :type metadata:
        """
        self.metadata = metadata
        if len(data) == 2 and isinstance(data[1], dict):
            data, opts = data
        elif isinstance(data, dict):
            opts = data
            data = ()
        else:
            opts = {}
        if base is None:
            base = self.detect_base(data, opts)
        self.base = base(*data, dimension=dimension, **opts)

    @property
    def data(self):
        return self.base.data

    def minimize(self, initial_guess=None, function_options=None, **opts):
        """
        Provides a uniform interface for minimization, basically just dispatching to the BaseSurface implementation if provided

        :param initial_guess: initial starting point for the minimization
        :type initial_guess: np.ndarray | None
        :param function_options:
        :type function_options: None | dict
        :param opts:
        :type opts:
        :return:
        :rtype:
        """

        # Some day I'll provide better minimization strategies for things like
        # Taylor series surfaces where there's a really well-defined analytic way
        # we can tackle the minimzation process
        # For now, though, we'll mostly just use scipy.optimize.minimize
        return self.base.minimize(initial_guess=initial_guess, function_options=function_options, **opts)

    @classmethod
    def detect_base(cls, data, opts):
        """
        Infers what type of base surface works for the data that's passed in.
        It's _super_ roughly done so...yeah generally better to pass the base class you want explicitly.
        But in the absence of that we can do this ?_?

        Basic strategy:
            1. look for options that go with specific methods
            2. look at data structures to guess
                i.   gradient as the first data arg + all data args are ndarrays -> Taylor Series
                ii.  callables as second arg -> Linear expansion or Linear fit
                iii. just like...one big array -> Interpolatin

        :param data:
        :type data: tuple
        :param opts:
        :type opts: dict
        :return:
        :rtype:
        """
        from ..FittableModels import LinearFitBasis

        # try a Taylor expansion
        if (
                any(key in opts for key in ['center', 'ref', 'weight_coefficients']) or (
                all(isinstance(d, np.ndarray) for d in data) and data[0].ndim==1
            )
        ):
            return TaylorSeriesSurface
        # next look into fitting
        elif (
                any(key in opts for key in ['basis']) or (
                len(data)==2 and (isinstance(data[1], LinearFitBasis) or all(callable(d) for d in data[1]))
            )
        ):
            # See if we were given fit coefficients or actual data to fit
            if data[0].ndim == 1:
                return LinearExpansionSurface
            else:
                return LinearFitSurface
        else:
            return InterpolatedSurface

    def __call__(self, gridpoints, **kwargs):
        return self.base(gridpoints, **kwargs)

class MultiSurface:
    """
    A _reallly_ simple extension to the Surface infrastructure to handle vector valued functions,
    assuming each vector value corresponds to a different Surfaces
    """
    def __init__(self, *surfs):
        """

        :param surfs: a set of Surface objects to use when evaluating
        :type surfs: Iterable[Surface]
        """
        self.surfs = surfs
    def __call__(self, gridpoints, **kwargs):
        return np.column_stack([s(gridpoints, **kwargs) for s in self.surfs])