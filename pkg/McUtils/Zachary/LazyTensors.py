"""
LazyTensors provides a small framework for symbolically working with Tensors
"""

import numpy as np
from ..Numputils import SparseArray

__all__ = [
    'Tensor',
    'TensorOp',
    'LazyOperatorTensor',
    'SparseTensor'
]

########################################################################################################################
#
#                                           Tensor
#
class Tensor:
    """A semi-symbolic representation of a tensor. Allows for lazy processing of tensor operations."""
    def __init__(self, a, shape = None):
        self._a = a
        self._shape = shape
    @classmethod
    def from_array(cls, a, shape = None):
        if isinstance(a, Tensor):
            return a
        else:
            return cls(a, shape = shape)
    @property
    def array(self):
        if isinstance(self._a, np.ndarray):
            return self._a
        else:
            return self._a.array

    def get_shape(self, a):
        return a.shape
    @property
    def shape(self):
        return self.get_shape(self._a)
    def get_dim(self):
        return len(self.shape)
    @property
    def dim(self):
        return self.get_dim()

    def add(self, other, **kw):
        return TensorPlus(self, other, **kw)
    def mul(self, other, **kw):
        return TensorMul(self, other, **kw)
    def dot(self, other, **kw):
        return TensorDot(self, other, **kw)
    def transpose(self, axes, **kw):
        return TensorTranspose(self, axes, **kw)
    def pow(self, other, **kw):
        return TensorPow(self, other, **kw)

    def __mul__(self, other):
        if isinstance(other, float):
            return TensorMul(self, other)
        else:
            return TensorDot(self, other)
    def __rmul__(self, other):
        if isinstance(other, float):
            return TensorMul(self, other)
        else:
            return TensorDot(self, other)
    def __add__(self, other):
        return TensorPlus(self, other)
    def __pow__(self, power, modulo=None):
        return TensorPow(self, power)

    def handle_missing_indices(self, missing, extant):
        # the default assumption is basically to use the diagonal element -- i.e. to assume that only the extant indices matter
        return self._a[extant]
    def pull_index(self, *idx):
        """Defines custom logic for handling how we pull indices

        :param idx:
        :type idx:
        :return:
        :rtype:
        """
        dim = len(idx)
        sdim = self.dim
        if dim > sdim:
            raise ValueError("{} has only {} dimensions but {} were requested".format(self, sdim, dim))
        else:
            real_shape = self._a.shape
            # here we add our hacks
            real_dim = len(real_shape)
            if real_dim < dim:
                a = self.handle_missing_indices(idx[:dim-real_dim], idx[dim-real_dim:])
            else:
                a = self._a[idx]
            try:
                new = type(self)(a)
            except ValueError:
                new = Tensor(a)
            return new

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.pull_index(item)
        else:
            return self.pull_index(*item)

    def __repr__(self):
        return "{}(<{}>)".format(type(self).__name__, ", ".join((str(x) for x in self.shape)))

class SparseTensor(Tensor):
    """
    Tensor class that uses SparseArray
    """

    def __init__(self, a, shape = None):
        super().__init__(SparseArray(a), shape = shape)

    @property
    def array(self):
        return self._a.todense()

########################################################################################################################
#
#                                           TensorOps
#
class TensorOp(Tensor):
    """A lazy representation of tensor operations to save memory"""
    def __init__(self, a, b, axis = None):
        # if isinstance(a, Tensor):
        #     a = a._a
        # if isinstance(b, Tensor):
        #     a = b._a
        self._a = a
        self._b = b
        if axis is None:
            self._axis = None
            self._axis_2 = None
        elif isinstance(axis, int):
            self._axis = axis
            self._axis_2 = None
        else:
            self._axis = axis[1]
            self._axis_2 = axis[0]
        self._kw = dict(axis = axis)

    def op(self, a, b):
        raise NotImplementedError(
            "{}: {} not implemented until a subclass does it".format(type(self.__name__), 'op')
        )
    def get_shape(self, a, b):
        return a.shape
    @property
    def shape(self):
        return self.get_shape(self._a, self._b)
    @property
    def array(self):
        """Ought to always compile down to a proper ndarray

        :return:
        :rtype: np.ndarray
        """
        return self.op(self._a, self._b)
    def __getitem__(self, i):
        return self.op(self._a[i], self._b)
class TensorPlus(TensorOp):
    """Represents an addition of two tensors"""
    def op(self, a, b):
        if isinstance(a, Tensor):
            a = a.array # with addition we can't really avoid it...
        if isinstance(b, Tensor):
            b = b.array
        return a + b
    def get_shape(self, a, b):
        return super().get_shape(a, b)
    def __getitem__(self, i):
        return type(self)(self._a[i], self._b[i])
class TensorPow(TensorOp):
    """Represents a power of a tensors"""
    def op(self, a, b):
        if isinstance(a, Tensor):
            a = a.array
        return np.linalg.matrix_power(a, b)
    def get_shape(self, a, b):
        return super().get_shape(a, b)
    def __getitem__(self, i):
        return type(self)(self._a[i], self._b)
class TensorMul(TensorOp):
    """Represents a multiplication of a tensor and a scalar"""
    def op(self, a, b):
        if isinstance(a, Tensor):
            a = a.array # with scalar mult we can't really avoid it...
        return a * b
    def get_shape(self, a, b):
        return super().get_shape(a, b)
    def __getitem__(self, i):
        return type(self)(self._a[i], self._b)
class TensorTranspose(TensorOp):
    """Represents a tensor transposition"""
    def get_shape(self, a, b):
        a_shp = a.shape
        return tuple(a_shp[a] for a in b)
    def op(self, a, b):
        if isinstance(a, Tensor):
            a = a.array
        return a.transpose(b)
class TensorDot(TensorOp):
    """Represents a tensor contraction"""
    def get_shape(self, a, b):
        a_shp = a.shape
        b_shp = b.shape
        ax = 0 if self._axis is None else self._axis
        ax2 = -1 if self._axis_2 is None else self._axis_2
        ax = set(len(b_shp) + a if a < 0 else a for a in ((ax,) if isinstance(ax, int) else ax))
        ax2 = set(len(a_shp) + a if a < 0 else a for a in ((ax2,) if isinstance(ax2, int) else ax2))
        shared = 0
        for shared, s in enumerate(zip(a_shp, b_shp)):
            if s[0] != s[1]:
                break
        shared = min(shared, min(ax), min(ax2))

        return (
            a_shp[:shared] +
            tuple(a for i,a in enumerate(a_shp[shared:]) if i+shared not in ax2) +
            tuple(a for i,a in enumerate(b_shp[shared:]) if i+shared not in ax)
        )
    def op(self, a, b):
        from ..Numputils import vec_tensordot

        # we'll allow this to cast to dense as we'll assume people are only ever calling this on low dimensional tensors...
        if isinstance(a, Tensor):
            a = a.array
        if isinstance(b, Tensor):
            b = b.array
        ax = 0 if self._axis is None else self._axis
        ax2 = -1 if self._axis_2 is None else self._axis_2
        if isinstance(ax, int):
            ax = (ax,)
        if isinstance(ax2, int):
            ax2 = (ax2,)
        contract = vec_tensordot(a, b, axes = (ax2, ax))
        return contract
    def __getitem__(self, i):
        return type(self)(self._a[i], self._b, **self._kw)

########################################################################################################################
#
#                                           LazyOperatorTensor
#
class LazyOperatorTensor(Tensor):
    """A super-lazy tensor that represents the elements of an operator """

    def __init__(self, operator, shape, memoization = True, dtype = object, fill = None):
        if memoization is True:
            memoization = {}
        self.memoization = isinstance(memoization, dict)
        if self.memoization:
            a = memoization
        else:
            a = None

        self.operator = operator
        self.dtype = dtype
        self.fill = fill
        super().__init__(a, shape=shape)

    @property
    def array(self):
        import itertools as it

        base = np.full(self.shape, self.fill, dtype=self.dtype)
        mem = self._a
        try:
            self._a = None
            _cast = False
            for idx in it.product(range(x) for x in self.shape):
                res = self._get_element(idx)
                if not _cast:
                    if isinstance(res, (int, float, np.integer, np.float)):
                        base = np.zeros(base, dtype=type(res))
                    _cast = True

                base[idx] = res
        finally:
            self._a = mem

        return base

    def _get_element(self, indices):
        if self._a is not None:
            try:
                res = self._a[indices]
            except (KeyError, IndexError):
                res = self.operator(indices)
                self._a[indices] = res
            return res
        else:
            return self.operator(indices)

    def __getitem__(self, item):
        return self._get_element(item)
