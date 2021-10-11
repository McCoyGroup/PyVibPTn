"""
Provides a set of decorators that allow code to be agnostic to whether Numba exists or not
"""
import warnings, typing

__all__ = [
    'njit',
    'jit',
    'type_spec',
    'without_numba',
    'numba_decorator',
    'import_from_numba',
    'objmode',
    'prange'
]

class NumbaState:
    numba_disabled = False

class without_numba:
    def __init__(self):
        self._numba_state = None
    def __enter__(self):
        self._numba_state = NumbaState.numba_disabled
        _numba_disabled = True
    def __exit__(self, exc_type, exc_val, exc_tb):
        NumbaState.numba_disabled = self._numba_state
        self._numba_state = None

def load_numba(warn=False):
    if NumbaState.numba_disabled:
        return None
    try:
        import numba
    except ImportError:
        if isinstance(warn, str) and warn == 'raise':
            raise
        if warn:
            warnings.warn("Numba not installed/code will be slower")
        numba = None

    return numba

def njit(*args, warn=False, **kwargs):
    return numba_decorator(*args, method='njit', warn=warn, **kwargs)

def jit(*args, warn=False, nopython=False, **kwargs):
    return numba_decorator(*args, method='jit', warn=warn, nopython=nopython, **kwargs)

def numba_decorator(*args, method=None, warn=False, **kwargs):
    numba = load_numba(warn=warn)
    if numba is not None:
        return getattr(numba, method)(*args, **kwargs)
    else:
        if len(args) > 0:
            return args[0]
        else:
            return lambda f:f

def type_spec(t, warn=False):
    numba = load_numba(warn=warn)
    if numba is not None:
        return getattr(numba, t)
    else:
        return typing.Any

def import_from_numba(name, default):
    numba = load_numba()
    if numba is not None:
        return getattr(numba, name)
    else:
        return default

class _noop_context:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

objmode = import_from_numba('objmode', _noop_context)
prange = import_from_numba('prange', range)