import numpy as np, scipy.sparse as sp, itertools as ip, functools as fp, os, abc
from ..Scaffolding import MaxSizeCache, Logger
from .SetOps import contained
from .Misc import infer_inds_dtype

__all__ = [
    "SparseArray",
    "ScipySparseArray",
    "TensorFlowSparseArray",
    "sparse_tensordot"
]

#   I should provide a general "ArrayWrapper" class that allows all these arrays to interact nicely
#   with eachother with a very general `dot` syntax and shared `reshape` interface and all of that
#   that way I would never need to check if things are numpy.ndarray or sp.spmatrix or those kinds of things
#
#   And actually by providing a constructor like np.array that dispatches to the appropriate class
#   for a given data type we'd be able to avoid the need to explicitly declare our data-type
#
#   This would also allow us to more conveniently swap out the backend for our different arrays, in
#   case we want to use a different sparse matrix format.

class SparseArray(metaclass=abc.ABCMeta):
    """
    Represents a generic sparse array format
    which can be subclassed to provide a concrete implementation
    """
    backends = None
    @classmethod
    def get_backends(cls):
        """
        Provides the set of backends to try by default
        :return:
        :rtype:
        """
        if cls.backends is None:
            return (('scipy', ScipySparseArray),)
        else:
            return cls.backends

    @classmethod
    def from_data(cls, data, shape=None, dtype=None, target_backend=None, constructor=None, **kwargs):
        """
        A wrapper so that we can dispatch to the best
        sparse backend we've got defined.
        Can be monkey patched.

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype: SparseArray
        """

        if shape is not None:
            kwargs['shape'] = shape
        if dtype is not None:
            kwargs['dtype'] = dtype
        backend_errors = []
        backends = cls.get_backends()
        for name,backend in backends:
            if target_backend is not None and name == target_backend:
                if constructor is not None:
                    backend = getattr(backend, constructor)
                return backend(data, **kwargs)
            else:
                try:
                    if constructor is not None:
                        backend = getattr(backend, constructor)
                    return backend(data, **kwargs)
                except Exception as e:
                    backend_errors.append(e)
        else:
            if target_backend is None:
                raise backend_errors[0]
            else:
                raise ValueError("`target_backend` {} not in available sparse backends {}".format(
                    target_backend,
                    backends
                ))
    @classmethod
    def from_diag(cls, data, shape=None, dtype=None, **kwargs):
        """
        A wrapper so that we can dispatch to the best
        sparse backend we've got defined.
        Can be monkey patched.
        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return cls.from_data(data,
                             constructor='from_diagonal_data',
                             shape=shape, dtype=dtype,
                             **kwargs
                             )
    @classmethod
    @abc.abstractmethod
    def from_diagonal_data(cls, diags, **kw):
        """
        Constructs a sparse tensor from diagonal elements

        :param diags:
        :type diags:
        :param kw:
        :type kw:
        :return:
        :rtype:
        """
        raise NotImplementedError("{}.{} is an abstract method".format(cls.__name__, 'from_diagonal_data'))
        ...
    @property
    @abc.abstractmethod
    def shape(self):
        """
        Provides the shape of the sparse array
        :return:
        :rtype: tuple[int]
        """
        ...
    @property
    def ndim(self):
        """
        Provides the number of dimensions in the array
        :return:
        :rtype:
        """
        return len(self.shape)
    @abc.abstractmethod
    def to_state(self, serializer=None):
        """
        Provides just the state that is needed to
        serialize the object
        :param serializer:
        :type serializer:
        :return:
        :rtype:
        """
        ...
    @classmethod
    @abc.abstractmethod
    def from_state(cls, state, serializer=None):
        """
        Loads from the stored state
        :param serializer:
        :type serializer:
        :return:
        :rtype:
        """
        raise NotImplementedError("{}.{} is an abstract method".format(cls.__name__, 'from_state'))
        ...
    @classmethod
    def empty(cls, shape, dtype=None, **kw):
        return cls.from_data(shape,
                             constructor='initialize_empty',
                             dtype=dtype,
                             **kw
                             )
    @classmethod
    @abc.abstractmethod
    def initialize_empty(cls, shp, shape=None, **kw):
        """
        Returns an empty SparseArray with the appropriate shape and dtype
        :param shape:
        :type shape:
        :param dtype:
        :type dtype:
        :param kw:
        :type kw:
        :return:
        :rtype:
        """
        raise NotImplementedError("{}.{} is an abstract method".format(cls.__name__, 'empty'))
        ...
    @property
    @abc.abstractmethod
    def block_data(self):
        """
        Returns the vector of values and corresponding indices
        :return:
        :rtype:
        """
        raise NotImplementedError("{}.{} is an abstract method".format(type(self).__name__, 'block_data'))
        ...
    @property
    @abc.abstractmethod
    def block_inds(self):
        """
        Returns indices for the stored values
        :return:
        :rtype:
        """
        raise NotImplementedError("{}.{} is an abstract method".format(type(self).__name__, 'block_inds'))
        ...
    @abc.abstractmethod
    def transpose(self, axes):
        """
        Returns a transposed version of the tensor
        :param axes:
        :type axes:
        :return:
        :rtype:
        """
        raise NotImplementedError("{}.{} is an abstract method".format(type(self).__name__, 'transpose'))
        ...
    @abc.abstractmethod
    def ascoo(self):
        """
        Converts the tensor into a scipy COO matrix...
        :return:
        :rtype: sp.coo_matrix
        """
        raise NotImplementedError("{}.{} is an abstract method".format(type(self).__name__, 'ascoo'))

    @abc.abstractmethod
    def ascsr(self):
        """
        Converts the tensor into a scipy CSR matrix...
        :return:
        :rtype: sp.csr_matrix
        """
        raise NotImplementedError("{}.{} is an abstract method".format(type(self).__name__, 'ascsr'))
    @abc.abstractmethod
    def asarray(self):
        """
        Converts the tensor into a dense np.ndarray
        :return:
        :rtype: np.ndarray
        """
        raise NotImplementedError("{}.{} is an abstract method".format(type(self).__name__, 'asarray'))
    @abc.abstractmethod
    def reshape(self, newshape):
        """
        Returns a reshaped version of the tensor
        :param axes:
        :type axes:
        :return:
        :rtype:
        """
        raise NotImplementedError("{}.{} is an abstract method".format(type(self).__name__, 'reshape'))
    @abc.abstractmethod
    def resize(self, newsize):
        """
        Returns a resized version of the tensor
        :param axes:
        :type axes:
        :return:
        :rtype:
        """
        raise NotImplementedError("{}.{} is an abstract method".format(type(self).__name__, 'resize'))

    def expand_dims(self, axis):
        """
        adapted from np.expand_dims

        :param axis:
        :type axis:
        :return:
        :rtype:
        """

        if isinstance(axis, (int, np.integer)):
            axis = [axis]

        out_ndim = len(axis) + self.ndim
        axis = [out_ndim + a if a < 0 else a for a in axis]

        shape_it = iter(self.shape)
        shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]

        return self.reshape(shape)

    def moveaxis(self, start, end):
        """
        Adapted from np.moveaxis

        :param start:
        :type start:
        :param end:
        :type end:
        :return:
        :rtype:
        """

        if isinstance(start, (int, np.integer)):
            start = [start]
        if isinstance(end, (int, np.integer)):
            end = [end]

        start = [self.ndim + a if a < 0 else a for a in start]
        end = [self.ndim + a if a < 0 else a for a in end]

        order = [n for n in range(self.ndim) if n not in start]

        for dest, src in sorted(zip(end, start)):
            order.insert(dest, src)

        # raise Exception(order)

        return self.transpose(order)

    @abc.abstractmethod
    def concatenate(self, *others, axis=0):
        """
        Concatenates multiple SparseArrays along the specified axis
        :return:
        :rtype: SparseArray
        """
        ...

    def broadcast_to(self, shape):
        """
        Broadcasts self to the given shape.
        Incredibly inefficient implementation but useful in smaller cases.
        Might need to optimize later.

        :param shape:
        :type shape:
        :return:
        :rtype:
        """

        # precheck b.c. concatenate is expensive
        compat = all(a%b == 0 for a,b in zip(shape, self.shape))
        if not compat:
            raise ValueError("{} with shape {} can't be broadcast to shape {}".format(
                type(self).__name__,
                self.shape,
                shape
            ))

        new = self
        for i, (shs, sho) in enumerate(zip(self.shape, shape)):
            if shs != sho:
                reps = sho // shs
                for _ in range(reps):
                    new = new.concatenate(self, axis=i)

        return new

    def __truediv__(self, other):
        return self.true_multiply(1 / other)
    def __rtruediv__(self, other):
        return self.true_multiply(1 / other)
    def __rmul__(self, other):
        return self.true_multiply(other)
    def __mul__(self, other):
        return self.true_multiply(other)

    def _bcast_shapes(self, other):
        if other.ndim != self.ndim:
            raise NotImplementedError("{} object only supports broadcasting when `ndims` align".format(
                type(self).__name__
            ))

        total_shape = []
        for shs, sho in zip(self.shape, other.shape):
            if shs == 1:
                total_shape.append(sho)
            elif sho == 1:
                total_shape.append(shs)
            elif shs == sho:
                total_shape.append(shs)
            else:
                raise ValueError("shape mismatch for multiply of objects with shapes {} and {}".format(
                    self.shape,
                    other.shape
                ))

        if total_shape != self.shape:
            self = self.broadcast_to(total_shape)

        if total_shape != other.shape:
            if isinstance(other, np.ndarray):
                other = np.broadcast_to(other, total_shape)
            else:
                other = other.broadcast_to(total_shape)

        return self,other

    @abc.abstractmethod
    def true_multiply(self, other):
        """
        Multiplies self and other
        :param other:
        :type other:
        :return:
        :rtype: SparseArray
        """
        ...
    def multiply(self, other):
        """
        Multiplies self and other but allows for broadcasting
        :param other:
        :type other: SparseArray | np.ndarray | int | float
        :return:
        :rtype:
        """
        if isinstance(other, (int, float)):
            return self.true_multiply(other)

        self, other = self._bcast_shapes(other)

        return self.true_multiply(other)

    @abc.abstractmethod
    def dot(self, other):
        """
        Takes a regular dot product of self and other
        :param other:
        :type other:
        :param axes:
        :type axes:
        :return:
        :rtype:
        """
        ...

    def tensordot(self, other, axes=2):
        """
        Takes the dot product of self and other along the specified axes
        :param other:
        :type other:
        :param axes: the axes to contract along
        :type axes: Iterable[int] | Iterable[Iterable[int]]
        :return:
        :rtype:
        """

        caching_status = self.get_caching_status()
        try:
            self.disable_caches()
            try:
                iter(axes)
            except TypeError:
                axes_a = list(range(-axes, 0))
                axes_b = list(range(0, axes))
            else:
                axes_a, axes_b = axes
            try:
                na = len(axes_a)
                axes_a = list(axes_a)
            except TypeError:
                axes_a = [axes_a]
                na = 1
            try:
                nb = len(axes_b)
                axes_b = list(axes_b)
            except TypeError:
                axes_b = [axes_b]
                nb = 1

            a = self
            b = other

            as_ = a.shape
            nda = a.ndim
            bs = b.shape
            ndb = b.ndim
            equal = True
            if na != nb:
                equal = False
            elif max(axes_a) >= nda:
                raise ValueError("tensor with shape {} doesn't have {} axes".format(
                    as_, axes_a
                ))
            elif max(axes_b) >= ndb:
                raise ValueError("tensor with shape {} doesn't have {} axes".format(
                    bs, axes_b
                ))
            else:
                for k in range(na):
                    if as_[axes_a[k]] != bs[axes_b[k]]:
                        equal = False
                        break
                    if axes_a[k] < 0:
                        axes_a[k] += nda
                    if axes_b[k] < 0:
                        axes_b[k] += ndb
            if not equal:
                raise ValueError("shape-mismatch for sum ({}{}@{}{})".format(as_, axes_a, bs, axes_b))

            # Move the axes to sum over to the end of "a"
            # and to the front of "b"
            notin = [k for k in range(nda) if k not in axes_a]
            newaxes_a = [notin, axes_a]
            N2 = 1
            for axis in axes_a:
                N2 *= as_[axis]
            totels_a = np.prod(as_)
            newshape_a = (totels_a // N2, N2)
            olda = [as_[axis] for axis in notin]

            notin = [k for k in range(ndb) if k not in axes_b]
            newaxes_b = [axes_b, notin]
            N2 = 1
            for axis in axes_b:
                N2 *= bs[axis]
            totels_b = np.prod(bs)
            newshape_b = (N2, totels_b // N2)
            oldb = [bs[axis] for axis in notin]

            # if any(dim == 0 for dim in ip.chain(newshape_a, newshape_b)):
            #     # shortcut for when we aren't even doing a contraction...?
            #     res = sp.csr_matrix((olda + oldb), dtype=b.dtype)
            #     # dense output...I guess that's clean?
            #     if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            #         res = res.todense()
            #     return res


            if a.ndim > 1:
                at = a.transpose(newaxes_a[0] + newaxes_a[1])
            else:
                at = a
            at = at.reshape(newshape_a)

            if b.ndim > 1:
                bt = b.transpose(newaxes_b[0] + newaxes_b[1])
            else:
                bt = b
            bt = bt.reshape(newshape_b)

            res = at.dot(bt)

            return res.reshape(olda + oldb)
        finally:
            if caching_status:
                self.enable_caches()

    # TODO: Caching should be managed by keeping track of 'parent' `SparseArray`
    #       objects using a weakref so that it's possible to just ask the parents
    #       if they have the data we need in their caches and then all caches
    #       will properly be cleaned up when objects go out of scope
    @classmethod
    def get_caching_status(cls):
        """
        A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to specify if caching is on or not
        :return:
        :rtype:
        """
    @classmethod
    def enable_caches(self):
        """
        A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to turn this on
        :return:
        :rtype:
        """
    @classmethod
    def disable_caches(self):
        """
        A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to turn this off
        :return:
        :rtype:
        """
    @classmethod
    def clear_cache(self):
        """
        A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to clear this out.
        :return:
        :rtype:
        """

    # TODO: need to figure out if I want to support item assignment or not

class ScipySparseArray(SparseArray):
    """
    Array class that generalize the regular `scipy.sparse.spmatrix`.
    Basically acts like a high-dimensional wrapper that manages the _shape_ of a standard `scipy.sparse_matrix`, since that is rigidly 2D.
    We always use a combo of an underlying CSR or CSC matrix & COO-like shape operations.
    """

    def __init__(self, a,
                 shape=None,
                 layout=None,
                 dtype=None,
                 initialize=True,
                 cache_block_data=True,
                 logger=None
                 ):
        """

        :param a:
        :type a:
        :param shape:
        :type shape:
        :param layout:
        :type layout:
        :param dtype:
        :type dtype:
        :param initialize:
        :type initialize:
        :param cache_block_data: whether or not
        :type cache_block_data:
        :param logger: the logger to use for debug purposes
        :type logger: Logger
        """
        self._shape = tuple(shape) if shape is not None else shape
        self._a = a
        self._fmt = layout
        self._dtype = dtype
        self._validated = False
        self._block_data_sorted = False
        self._block_inds = None # cached to speed things up
        self._block_vals = None # cached to speed things up
        self.logger = logger
        if initialize:
            self._init_matrix(cache_block_data=cache_block_data)
            self._validate()

    def to_state(self, serializer=None):
        """
        Provides just the state that is needed to
        serialize the object
        :param serializer:
        :type serializer:
        :return:
        :rtype:
        """
        return {
            'layout': serializer.serialize(self._fmt),
            'shape': self.shape,
            'dtype': self.dtype.name,
            'block_data': self.block_data
        }
    @classmethod
    def from_state(cls, state, serializer=None):
        block_vals, block_inds = state['block_data']
        block_vals = np.asarray(block_vals)
        block_inds = tuple(np.asarray(x) for x in block_inds)
        return cls(
            (block_vals, block_inds),
            dtype=np.dtype(state['dtype']),
            layout=serializer.deserialize(state['layout']),
            shape=state['shape']
        )
    @classmethod
    def initialize_empty(cls, shape, dtype=None, layout=None, **kw):
        matshape = cls._get_balanced_shape(shape)
        if dtype is None:
            dtype=np.float64
        if layout is None:
            if matshape[0] > matshape[1]:
                layout = sp.csc_matrix
            else:
                layout = sp.csr_matrix
        new = cls(layout(matshape, dtype=dtype), shape=shape, **kw)
        return new
    @staticmethod
    def _get_balanced_shape(shp):
        """
        We get a strong memory benefit from having a well-balanced
        matrix, so we do a scan to find the best balancing point
        This works by finding the first spot where the product of the remaining
        inds is less than the first inds and then checking the balance against
        cutting at the previous spot
        :param shp:
        :type shp:
        :return:
        :rtype:
        """

        cumprods_left = np.cumprod(shp)
        cumprods_right = np.flip(np.cumprod(np.flip(shp)))
        vl, vr = o_vl, o_vr = 1, cumprods_right[0]
        for vl, vr in zip(cumprods_left, cumprods_right[1:]):
            if vr == vl: # perfect balance
                break
            elif vr < vl:
                if (vr / vl) < (o_vl / o_vr):
                    vl, vr = o_vl, o_vr
                break
            o_vl, o_vr = vl, vr

        return (vl, vr)

    # from memory_profiler import profile
    # @profile
    def _init_matrix(self, cache_block_data=True):
        a = self._a
        if isinstance(a, ScipySparseArray):
            if self.logger is not None:
                self.logger.log_print("initializing from existing `SparseArray`", log_level=self.logger.LogLevel.Debug)
            if self.fmt is not a.fmt:
                self._a = self.fmt(a._a, shape=a.shape)
            else:
                self._a = a._a
            shp = self._shape
            if shp is not None:
                self.reshape(a.shape)
            else:
                self._shape = a.shape
        elif isinstance(a, sp.spmatrix):
            if self.logger is not None:
                self.logger.log_print("initializing from existing `spmatrix`", log_level=self.logger.LogLevel.Debug)
            if self._shape is None:
                self._shape = a.shape
        elif isinstance(a, np.ndarray):
            self._shape = a.shape
            if len(a.shape) > 2:
                # we want this to be more balanced b.c. it helps for transposes
                total_shape = self._get_balanced_shape(a.shape)
                a = a.reshape(total_shape)
            elif len(a.shape) == 1:
                a = a.reshape(a.shape + (1,))
            if self._fmt is None:
                if a.shape[0] > a.shape[1]:
                    fmt = sp.csc_matrix
                else:
                    fmt = sp.csr_matrix
            else:
                fmt = self._fmt
            if self.logger is not None:
                self.logger.log_print("initializing from `ndarray`", log_level=self.logger.LogLevel.Debug)
            self._a = fmt(a, shape=a.shape)

        # we're gonna support the (vals, (i_1, i_2, i_3, ...)) syntax for constructing
        # an array based on its non-zero positions
        elif len(a) == 2 and len(a[1]) > 0 and len(a[0]) == len(a[1][0]):
            if self.logger is not None:
                self.logger.log_print("initializing from vals and indices", log_level=self.logger.LogLevel.Debug)
            data, block_vals, block_inds, self._shape = self.construct_sparse_from_val_inds(
                a[0], a[1], self._shape, self._fmt
            )
            self._a = data
            if cache_block_data and block_inds is not None: # corner case to avoid a sort in low-mem situation
                self._block_vals = block_vals
                self._block_inds = block_inds
                self._block_data_sorted = True
        else:
            non_sparse, sparse = self._get_shape()
            if non_sparse is None:
                self._a = np.array(self._a)
                self._init_matrix()
            else:
                self._shape = non_sparse + sparse
                data, other = self._get_data(non_sparse, sparse)
                block_data, inds, total_shape = other
                self._a = data
                if cache_block_data:
                    flat = np.ravel_multi_index(inds, data.shape)
                    self._block_inds = flat, inds
                    self._block_data_sorted = False

    @classmethod
    def construct_sparse_from_val_inds(cls, block_vals, block_inds, shape, fmt, cache_block_data=True, logger=None):
        block_vals = np.asanyarray(block_vals)
        if shape is None:
            shape = tuple(np.max(x) for x in block_inds)

        # special case esp. for square matrices
        shape_rat = max(shape) / min(shape)
        if len(shape) == 2 and shape_rat < 5:
            init_inds = block_inds
            total_shape = shape
            block_inds = None
        else:
            block_inds = tuple(np.asanyarray(i) for i in block_inds)
            if len(block_inds) != len(shape):
                raise ValueError("{}: can't initialize array of shape {} from non-zero indices of dimension {}".format(
                    cls.__name__,
                    shape,
                    len(block_inds)
                ))


            if logger is not None:
                logger.log_print("calculating flat indices", log_level=logger.LogLevel.Debug)

            flat = np.ravel_multi_index(block_inds, shape)  # no reason to cache this since we're going to sort it...

            if cache_block_data:

                if logger is not None:
                    logger.log_print("sorting cached data", log_level=logger.LogLevel.Debug)
                # gotta make sure our inds are sorted so we don't run into sorting issues later...
                sort = np.argsort(flat)
                flat = flat[sort]
                block_vals = block_vals[sort]
                block_inds = (flat, tuple(i[sort] for i in block_inds))
                del sort  # clean up for memory reasons

            if logger is not None:
                logger.log_print("constructing 2D indices", log_level=logger.LogLevel.Debug)
            # this can help significantly with memory usage...
            total_shape = cls._get_balanced_shape(shape)
            init_inds = np.unravel_index(flat, total_shape)  # no reason to cache this since we're not going to use it

            if not cache_block_data:
                block_inds = None

        if fmt is None:
            if total_shape[0] > total_shape[1]:
                fmt = sp.csc_matrix
            else:
                fmt = sp.csr_matrix

        if logger is not None:
            logger.log_print("initializing {fmt} from indices and values", fmt=fmt, log_level=logger.LogLevel.Debug)

        try:
            data = fmt((block_vals, init_inds), shape=total_shape)
        except TypeError:
            data = fmt(sp.coo_matrix((block_vals, init_inds)), shape=total_shape)
        except MemoryError:
            if total_shape[0] > total_shape[1]:
                fmt = sp.csc_matrix
            else:
                fmt = sp.csr_matrix
            data = fmt((block_vals, init_inds), shape=total_shape)

        if not cache_block_data:
            block_vals = None

        return data, block_vals, block_inds, shape

    def _validate(self):
        shp1 = self._a.shape
        shp2 = self._shape
        if np.prod(shp1) != np.prod(shp2):
            raise ValueError("sparse data with shape {} cannot correspond to shape {}".format(
                shp1,
                shp2
            ))
        self._validated = True
    def _get_shape(self):
        """
        Walks through the array data we're holding onto and determines
        where the sparse blocks start
        """
        non_sp = []
        elm = self._a
        while not isinstance(elm, sp.spmatrix):
            try:
                elm_2 = elm[0]
            except (TypeError, IndexError):
                return None, None
            else:
                non_sp.append(len(elm))
                elm = elm_2
        else: # only catch if no break
            shape = elm.shape

        non_sparse_shape = tuple(non_sp)
        return non_sparse_shape, tuple(shape)
    def _get_data(self, non_sparse, sparse):
        """
        We'll take our blocks and arrange them as a vector of sparse arrays
        """
        inds = ip.product(*(list(range(x)) for x in non_sparse))
        a = self._a
        bits = np.cumprod(non_sparse) # so that we can build our vector of blocks
        if len(sparse) == 1:
            sparse = sparse+(1,)
        def pull(a, idx, bits=bits, shp = sparse):
            block = fp.reduce(lambda b,i:b[i], idx, a)
            if not isinstance(block, (sp.spmatrix, np.ndarray)):
                block = np.array(block)
            block_data, row_ind, col_ind = sp.find(block)
            offset = np.dot(idx, bits)*shp[1]
            return block_data, row_ind, col_ind + offset

        blocks = [pull(a, idx) for idx in inds]
        block_data = np.concatenate([b[2] for b in blocks])
        row_inds = np.concatenate([b[0] for b in blocks])
        col_inds = np.concatenate([b[1] for b in blocks])
        total_shape = (sparse[0], bits[-1]*sparse[1])
        inds = (row_inds, col_inds)
        data = self._build_data(block_data, inds, total_shape)
        return data, (block_data, inds, total_shape)
    def _build_data(self, block_data, inds, total_shape):
        if len(block_data) == 0 or np.prod(block_data.shape) == 0.: # empty array
            if total_shape[0] > total_shape[1]:
                base_sparse = sp.coo_matrix(total_shape, dtype=self.dtype)
            else:
                base_sparse = sp.coo_matrix(total_shape, dtype=self.dtype)
            try:
                return self.fmt(base_sparse)
            except MemoryError:
                self._fmt = base_sparse.format
                return base_sparse

        # shp = self.shape
        # if len(shp) == 5 and shp[:4] == (12, 12, 12, 12):
        #     raise RuntimeError(self.shape, self.fmt, total_shape, "why")

        try:
            data = self.fmt((block_data, inds), shape=total_shape, dtype=self.dtype)
        except (ValueError, TypeError):
            if total_shape[0] > total_shape[1]:
                base_sparse = sp.csc_matrix(total_shape, dtype=self.dtype)
            else:
                base_sparse = sp.csr_matrix(total_shape, dtype=self.dtype)
            try:
                data = self.fmt(base_sparse)
            except MemoryError:
                data = base_sparse
                self._fmt = base_sparse.format
        return data

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def diag(self):
        if len(self.shape) != 2:
            raise NotImplementedError("haven't decided what I want to do for tensors")
        N = np.min(self.shape)
        diag = self[np.arange(N), np.arange(N)]
        return diag.flatten()

    @classmethod
    def from_diagonal_data(cls, diags, shape=None, **kw):
        if isinstance(diags[0], (int, np.integer, float, np.floating)):
            # just a plain old diagonal matrix
            N = len(diags)
            # print(len(diags))
            if shape is None:
                shape = (N, N)
            return cls(sp.csr_matrix(sp.diags([diags], [0])), shape=shape, **kw)
        else:
            data = sp.block_diag(diags, format='csr')
            block_size = diags[0].shape
            if shape is None:
                shape = (len(diags), block_size[0], len(diags), block_size[1])
            wat = cls(data, shape=shape, **kw).transpose((0, 2, 1, 3))
            return wat
    def asarray(self):
        return np.reshape(self.data.toarray(), self.shape)
    def todense(self):
        return np.reshape(np.asarray(self.data.todense()), self.shape)
    def ascoo(self):
        return sp.coo_matrix(self.data)
    def ascsr(self):
        return sp.csr_matrix(self.data)
    def ascsc(self):
        return sp.csc_matrix(self.data)
    @property
    def data(self):
        if not isinstance(self._a, sp.spmatrix):
            self._init_matrix()
        if not self._validated:
            self._validate()
        return self._a
    @data.setter
    def data(self, new):
        if not isinstance(new, sp.spmatrix):
            raise ValueError("new underlying sparse buffer must be of type {}".format(
                sp.spmatrix
            ))
        elif np.prod(new.shape) != np.prod(self.shape):
            raise ValueError("can't use sparse buffer of shape {} for tensor of shape {}".format(
                new.shape,
                self.shape
            ))
        else:
            if new.format != self.data.format:
                new = self.fmt(new)
            self._a = new
    formats_map = {
        'csr': sp.csr_matrix,
        'csc': sp.csc_matrix,
        'coo': sp.coo_matrix
    }
    @classmethod
    def format_from_string(cls, fmt):
        if isinstance(fmt, str):
            return cls.formats_map[fmt]
        elif isinstance(fmt, type):
            return fmt
        else:
            raise TypeError("not sure how to get a valid sparse format class from {}".format(fmt))
    @property
    def fmt(self):
        if self._fmt is None:
            return self.format_from_string(self.data.format)
        else:
            return self._fmt
    @property
    def shape(self):
        if self._shape is None:
            self._init_matrix()
        if not self._validated:
            self._validate()
        return self._shape
    @property
    def ndim(self):
        return len(self.shape)
    @property
    def non_zero_count(self):
        return self.data.nnz
    # def __len__(self):
    #     return self.shape[0]
    default_cache_size = 2
    caching_enabled=True
    @classmethod
    def get_caching_status(cls):
        return cls.caching_enabled
    @classmethod
    def enable_caches(self):
        """
        A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to turn this on
        :return:
        :rtype:
        """
        self.caching_enabled = True
    @classmethod
    def disable_caches(self):
        """
        A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to turn this off
        :return:
        :rtype:
        """
        self.caching_enabled = False
    @classmethod
    def clear_cache(cls):
        cls.clear_ravel_caches()
    @classmethod
    def clear_ravel_caches(cls):
        cls._unravel_cache = MaxSizeCache(cls.default_cache_size)
        cls._ravel_cache = MaxSizeCache(cls.default_cache_size)

    # this saves time when we have to do a bunch of reshaping into similarly sized arrays,
    # but won't help as much when the shape changes
    _unravel_cache = MaxSizeCache(default_cache_size)  # hopefully faster than bunches of unravel_index calls...

    @classmethod
    def _unravel_indices(cls, n, dims):

        # we're hoping that we call with `n` often enough that we get a performance benefit
        if not cls.caching_enabled:
            minimal_dtype = infer_inds_dtype(np.max(dims))
            res = tuple(x.astype(minimal_dtype) for x in np.unravel_index(n, dims))
            return res
        if dims not in cls._unravel_cache:
            cls._unravel_cache[dims] = MaxSizeCache(cls.default_cache_size)
        cache = cls._unravel_cache[dims]

        if isinstance(n, np.ndarray):
            n_hash = hash(n.data.tobytes())
        else:
            n_hash = hash(n)

        if n_hash in cache:
            res = cache[n_hash]
        else:
            minimal_dtype = infer_inds_dtype(np.max(dims))
            res = tuple(x.astype(minimal_dtype) for x in np.unravel_index(n, dims))
            cache[n_hash] = res
        return res

    # we cache the common ops...
    @classmethod
    def set_ravel_cache_size(cls, size):
        cls.default_cache_size = size
        if cls._ravel_cache is None:
            cls._ravel_cache = MaxSizeCache(size)
        elif cls._ravel_cache.max_items != size:
            cls._ravel_cache = MaxSizeCache(size)

        if cls._unravel_cache is None:
            cls._unravel_cache = MaxSizeCache(size)
        elif cls._unravel_cache.max_items != size:
            cls._unravel_cache = MaxSizeCache(size)


    _ravel_cache = MaxSizeCache(default_cache_size)  # hopefully faster than bunches of ravel_index calls...
    @classmethod
    def _ravel_indices(cls, mult, dims):
        # we're hoping that we call with `n` often enough that we get a performance benefit
        if not cls.caching_enabled:
            return np.ravel_multi_index(mult, dims)
        if isinstance(dims, list):
            dims = tuple(dims)
        if dims not in cls._ravel_cache:
            cls._ravel_cache[dims] = MaxSizeCache(cls.default_cache_size)
        cache = cls._ravel_cache[dims]
        if isinstance(mult[0], np.ndarray):
            n_hash = hash(tuple(m.data.tobytes() for m in mult))
        else:
            try:
                n_hash = hash(mult)
            except TypeError:
                # means hashing failed so we just shortcut out rather than try to be clever
                # raise Exception([mult, dims])
                return np.ravel_multi_index(mult, dims)
        if n_hash in cache:
            res = cache[n_hash]
        else:
            res = np.ravel_multi_index(mult, dims)
            cache[n_hash] = res
        return res

    def _getinds(self):
        # pulled from tocoo except without the final conversion to COO...
        from scipy.sparse import _sparsetools

        data = self.data
        major_dim, minor_dim = data._swap(data.shape)
        minor_indices = data.indices
        major_indices = np.empty(len(minor_indices), dtype=data.indices.dtype)
        _sparsetools.expandptr(major_dim, data.indptr, major_indices)
        row, col = data._swap((major_indices, minor_indices))

        return row, col
    def find(self):
        fmt = self.data.format
        if fmt in ['csr', 'csc']:
            d = self.data # type: sp.csr_matrix
            vals = d.data
            row_inds, col_inds = self._getinds()
        else:
            row_inds, col_inds, vals = sp.find(self.data)
        return row_inds, col_inds, vals


    # import memory_profiler
    # @memory_profiler.profile
    def _load_block_data(self):
        d = self.data

        row_inds, col_inds, data = self.find()

        flat = self._ravel_indices((row_inds, col_inds), d.shape)
        unflat = self._unravel_indices(flat, self.shape)
        self._block_inds = (flat, unflat)
        self._block_vals = data
        self._block_data_sorted = True

    def _load_full_block_inds(self):
        if self._block_inds.ndim == 1:
            flat = self._block_inds
            unflat = self._unravel_indices(flat, self.shape)
        else:
            unflat = self._block_inds
            flat = self._ravel_indices(unflat, self.shape)
        self._block_inds = (flat, unflat)

    def _sort_block_data(self):
        flat, unflat = self._block_inds
        sort = np.argsort(flat)
        flat = flat[sort]
        unflat = tuple(x[sort] for x in unflat)
        self._block_inds = (flat, unflat)
        self._block_vals = self._block_vals[sort]
        self._block_data_sorted = True

    @property
    def block_vals(self):
        if self._block_vals is None:
            self._load_block_data()
        if not self._block_data_sorted:
            self._sort_block_data()
        return self._block_vals

    @property
    def block_inds(self):
        if self._block_inds is None:
            self._load_block_data()
        elif not (
                isinstance(self._block_inds[0], np.ndarray)
                and self._block_inds[0].ndim == 1
        ):
            self._load_full_block_inds()
        if not self._block_data_sorted:
            self._sort_block_data()
        return self._block_inds
    @block_inds.setter
    def block_inds(self, bi):
        if isinstance(bi, tuple) and len(bi) == 2:
            flat, unflat = bi
            if isinstance(unflat[0], int):
                row_inds, col_inds = bi
                flat = self._ravel_indices((row_inds, col_inds), self.data.shape)
                unflat = self._unravel_indices(flat, self.shape)
        elif isinstance(bi[0], int):
            flat = bi
            unflat = self._unravel_indices(flat, self.shape)
        else:
            unflat = bi
            flat = self._ravel_indices(bi, self.shape)
        if len(unflat) != len(self.shape):
            raise ValueError("{}: block indices must have same dimension as array ({}); was given {}".format(
                type(self).__name__,
                len(self.shape),
                len(unflat)
            ))
        if self._block_vals is not None and len(flat) != len(self._block_vals):
                raise ValueError("{}: number of block indices must match number of non-zero elements ({}); was given {}".format(
                    type(self).__name__,
                    len(self._block_vals),
                    len(flat)
                ))
        self._block_inds = (flat, unflat)
    @property
    def block_data(self):
        res = self.block_vals, self.block_inds[1]
        return res

    # import memory_profiler
    # @memory_profiler.profile
    def transpose(self, transp):
        """
        Transposes the array and returns a new one.
        Not necessarily a cheap operation.

        :param transp: the transposition to do
        :type transp: Iterable[int]
        :return:
        :rtype:
        """

        shp = self.shape

        if len(transp) != self.ndim or np.any(np.sort(transp) != np.arange(self.ndim)):
            raise ValueError("transposition {} can't apply to shape {}".format(
                transp, self.shape
            ))

        track_data = self.get_caching_status()
        if self._block_vals is None:
            row_inds, col_inds, data = self.find()
            flat = self._ravel_indices((row_inds, col_inds), self.data.shape)
            inds = self._unravel_indices(flat, self.shape)
        else:
            data = self.block_vals
            (flat, inds) = self.block_inds

        new_inds = [inds[i] for i in transp]
        new_shape = tuple(shp[i] for i in transp)
        if len(data) == 0:
            return type(self).empty(new_shape, layout=self.fmt, dtype=data.dtype)

        if len(new_shape) > 2:
            total_shape = self._get_balanced_shape(new_shape)
            flat = self._ravel_indices(new_inds, new_shape)
            unflat = self._unravel_indices(flat, total_shape)
        else:
            flat = None
            unflat = new_inds
            total_shape = new_shape

        data = self._build_data(data, unflat, total_shape)

        # except MemoryError:
        #     raise Exception(data, unflat, new_shape, total_shape)
        new = type(self)(data, shape=new_shape, layout=self.fmt)

        if track_data:

            if flat is not None:
                arr = np.argsort(flat)
            else:
                arr = np.lexsort(unflat)

            new_inds = [inds[arr] for inds in new_inds]
            if self._block_vals is not None:
                new_v = self._block_vals[arr]
                new._block_vals = new_v
            if flat is None:
                new.block_inds = new_inds
            else:
                # try:
                new.block_inds = (flat[arr], new_inds)
            new._block_data_sorted = True
                # except:
                #     raise Exception(new_shape, len(total_shape))

        return new

    # def transpose(self, transp):
    #     """
    #     Transposes the array and returns a new one.
    #     Not necessarily a cheap operation.
    #
    #     :param transp: the transposition to do
    #     :type transp: Iterable[int]
    #     :return:
    #     :rtype:
    #     """
    #
    #     if len(self.shape) > 4:
    #         return self.profiled_transpose(transp)
    #
    #     shp = self.shape
    #
    #     if len(transp) != self.ndim or np.any(np.sort(transp) != np.arange(self.ndim)):
    #         raise ValueError("transposition {} can't apply to shape {}".format(
    #             transp, self.shape
    #         ))
    #
    #     data, inds = self.block_data
    #
    #     new_inds = [inds[i] for i in transp]
    #     new_shape = tuple(shp[i] for i in transp)
    #     if len(data) == 0:
    #         return type(self).empty(new_shape, layout=self.fmt, dtype=data.dtype)
    #
    #     if len(new_shape) > 2:
    #         total_shape = self._get_balanced_shape(new_shape)
    #         flat = self._ravel_indices(new_inds, new_shape)
    #         unflat = self._unravel_indices(flat, total_shape)
    #     else:
    #         flat = None
    #         unflat = new_inds
    #         total_shape = new_shape
    #
    #     data = self._build_data(data, unflat, total_shape)
    #     new = type(self)(data, shape = new_shape, layout = self.fmt)
    #
    #     arr = np.lexsort(unflat)
    #
    #     new_inds = [inds[arr] for inds in new_inds]
    #     if self._block_vals is not None:
    #         new_v = self._block_vals[arr]
    #         new._block_vals = new_v
    #     if flat is None:
    #         new.block_inds = new_inds
    #     else:
    #         new.block_inds = (flat, new_inds)
    #
    #     return new

    def reshape(self, shp):
        """
        Had to make this op not in-place because otherwise got scary errors...
        :param shp:
        :type shp:
        :return:
        :rtype:
        """

        if np.prod(shp) != np.prod(self.shape):
            raise ValueError("Can't reshape {} into {}".format(self.shape, shp))
        new = self.copy() # I'm not sure I want a whole copy here...?
        new._shape = tuple(shp)

        bi = new._block_inds
        if bi is not None:
            if isinstance(bi, np.ndarray) and bi.ndim == 1:
                new._block_inds = bi
            else:
                flat, unflat = bi
                new._block_inds = flat
        if len(shp) == 2:
            new.data = new.data.reshape(shp)
        return new
    def squeeze(self):
        self.reshape([x for x in self.shape if x != 1])
        return self

    def resize(self, newsize):
        """
        Returns a resized version of the tensor
        :param newsize:
        :type newsize: tuple[int]
        :return:
        :rtype:
        """
        # we'll simply construct a totally new tensor by dropping any indices which are out of bounds
        # in any coordinate
        # otherwise the multidimensional indices are clean so we'll just use those
        if len(newsize) != self.ndim:
            raise ValueError("unclear how to resize a tensor with rank {} into a tensor of rank {}".format(
                self.ndim,
                len(newsize)
            ))

        if len(newsize) == 2:
            return type(self)(self.data.copy().resize(newsize), shape=newsize)

        vals, inds = self.block_data
        # now we've got to filter out the indices which are out of bounds in any dimension
        for i,sizes in enumerate(zip(newsize, self.shape)):
            n, o = sizes
            if n < o:
                mask = inds[:i]
                vals = vals[mask]
                inds = inds[mask]

        return type(self)((vals, inds), shape=newsize)

    @staticmethod
    def _concat_coo(all_inds, all_vals, all_shapes, axis):

        full_vals = np.concatenate(all_vals)

        tot_shape = list(all_shapes[0])
        tot_shape[axis] = sum(a[axis] for a in all_shapes)

        # pull all the shapes along the concatenation axis

        # add the offset to each block along the concatenation axis
        # to create new vector of inds
        full_inds = None
        prev = 0
        for offset, ind_block in zip(all_shapes, all_inds):
            if full_inds is None:
                full_inds = ind_block
                prev += offset[axis]
            else:
                ind_block = ind_block[:axis] + (ind_block[axis] + prev,) + ind_block[axis + 1:]
                prev += offset[axis]
                full_inds = tuple(
                    np.concatenate([a, b])
                    for a, b in zip(full_inds, ind_block)
                )

        return full_vals, full_inds, tot_shape

    def concatenate(self, *others, axis=0):
        """
        Concatenates multiple arrays along the specified axis
        This is relatively inefficient in terms of not tracking indices
        throughout

        :param other:
        :type other:
        :param axis:
        :type axis:
        :return:
        :rtype:
        """

        others = [ScipySparseArray(o) if not isinstance(o, ScipySparseArray) else o for o in others]

        all_inds = [self.block_inds[1]] + [other.block_inds[1] for other in others]
        all_vals = [self.block_vals] + [other.block_vals for other in others]
        all_shapes = [self.shape] + [other.shape for other in others]

        full_vals, full_inds, tot_shape = self._concat_coo(all_inds, all_vals, all_shapes, axis)

        return type(self)(
            (
                full_vals,
                full_inds
            ),
            shape=tot_shape
        )

    def broadcast_to(self, shape):
        """
        Implements broadcast_to using COO-style operations
        to be a little bit more efficient

        :param shape:
        :type shape:
        :return:
        :rtype:
        """

        if shape == self.shape:
            return self

        # precheck b.c. concatenate is expensive
        compat = all(a % b == 0 for a, b in zip(shape, self.shape))
        if not compat:
            raise ValueError("{} with shape {} can't be broadcast to shape {}".format(
                type(self).__name__,
                self.shape,
                shape
            ))

        full_vals = None
        full_inds = None
        tot_shape = self.shape

        for i, (shs, sho) in enumerate(zip(self.shape, shape)):
            if shs != sho:
                if full_vals is None:
                    full_vals = self.block_vals
                if full_inds is None:
                    full_inds = self.block_inds[1]
                reps = sho // shs

                all_vals = [full_vals] * reps
                all_inds = [full_inds] * reps
                all_shapes = [tot_shape] * reps

                full_vals, full_inds, tot_shape = self._concat_coo(all_inds, all_vals, all_shapes, i)

        return type(self)(
            (
                full_vals,
                full_inds
            ),
            shape=tot_shape
        )

    @property
    def T(self):
        if len(self._shape) == 2:
            return self.transpose((1, 0))
        else:
            raise ValueError("dunno what T means for high dimensional arrays")
    def __matmul__(self, other):
        return self.dot(other)
    def dot(self, b, reverse=False):
        bshp = b.shape
        if isinstance(b, SparseArray) and len(bshp) > 2:
            bcols = int(np.prod(bshp[1:]))
            b = b.reshape((bshp[0], bcols))
        if self.shape != self.data.shape:
            arows = int(np.prod(self.shape[:-1]))
            a = self.reshape((arows, self.shape[-1])).data
        else:
            a = self.data

        if isinstance(b, ScipySparseArray):
            if b.shape != b.data.shape:
                bcols = int(np.prod(b.shape[1:]))
                b = b.reshape((b.shape[0], bcols)).data
            b = b.data
        elif isinstance(b, SparseArray):
            b = b.ascsr()

        woof = a.dot(b)

        if isinstance(woof, sp.spmatrix):
            return type(self)(a.dot(b))
        else:
            return woof

    def __neg__(self):
        return -1 * self
    def __pos__(self):
        return self
    def __add__(self, other):
        return self.plus(other)
    def __radd__(self, other):
        return self.plus(other)
    def plus(self, other):
        d = self.data
        if isinstance(other, (int, float, np.integer, np.floating)):
            if other == 0.:
                return self.copy()
            else:
                d = other + d
                new = type(self)(d, shape=self.shape, layout=self.fmt)
                if self._block_vals is not None:
                    bvs = self._block_vals
                    bi = self._block_inds
                    new._block_vals = other + bvs
                    new._block_inds = bi
                    new._block_data_sorted = self._block_data_sorted
                return new

        if isinstance(other, ScipySparseArray):
            other = other.data
        if isinstance(other, sp.spmatrix):
            other = other.reshape(d.shape)
        elif isinstance(other, np.ndarray):
            other = np.broadcast_to(other, self.shape).reshape(d.shape)
        new = d.__add__(other)
        if isinstance(new, sp.spmatrix):
            return type(self)(new, shape=self.shape, layout=self.fmt)
        else:
            return np.array(new).reshape(self.shape)

    def floopy_flop(self):
        return type(self)(1/self.data, shape = self.shape, layout=self.fmt)

    def __truediv__(self, other):
        return self.multiply(1 / other)
    def __rtruediv__(self, other):
        if other == 1:
            return self.floopy_flop()
        return self.multiply(1 / other)
    def __rmul__(self, other):
        return self.multiply(other)
    def __mul__(self, other):
        return self.multiply(other)
    def true_multiply(self, other):
        d = self.data
        if isinstance(other, (int, float, np.integer, np.floating)):
            if other == 0.:
                new = self.fmt(d.shape)
                return type(self)(new, shape=self.shape, layout=self.fmt)
            elif other == 1.:
                return self.copy()
            else:
                d = other * self.data
                new = type(self)(d, shape=self.shape, layout=self.fmt)
                if self._block_vals is not None:
                    bvs = self._block_vals
                    bi = self._block_inds
                    new._block_vals = other * bvs
                    new._block_inds = bi
                    new._block_data_sorted = self._block_data_sorted
                return new

        if isinstance(other, ScipySparseArray):
            other = other.data
        if isinstance(other, sp.spmatrix):
            other = other.reshape(d.shape)
        elif isinstance(other, np.ndarray):
            other = np.broadcast_to(other, self.shape).reshape(d.shape)
        new = d.multiply(other)
        if isinstance(new, sp.spmatrix):
            return type(self)(new, shape=self.shape, layout=self.fmt)
        else:
            return np.array(new).reshape(self.shape)

    def copy(self):
        import copy

        base = copy.copy(self)
        if base.data is self.data:
            base.data = base.data.copy()
        # I'm not sure I mutate anything else?

        # if base._block_inds is not None:
        #
        # if self.data is n
        # base.
        return base

    @classmethod
    def _find_block_alignment(cls, inds, block, shape):
        """
        finds the positions where the block & index align
        """

        # duh...
        filter, _, _ = contained(inds, block,
                                 assume_unique=(False, True),
                                 sortings=(np.arange(len(inds)), np.arange(len(block)))
                                 )
        # if method == 'sorted':
        #
        #     find_spots = np.empty(2*len(block), dtype=inds.dtype)
        #     find_spots[0::2] = block
        #     find_spots[1::2] = block+1
        #     # find where the blocks start and end (doable like this b.c. we ensure sorting)
        #     # whenever things are modded
        #     filter = np.full(len(inds), False)
        #     # print(filter, find_spots)
        #     block_ends = np.searchsorted(inds, find_spots)
        #     print(find_spots[:10], block_ends[:10], inds[:10])
        #     for i in range(0, len(find_spots), 2):
        #         start = block_ends[i]
        #         end = block_ends[i+1]
        #         j = block[i//2]
        #         if start < len(inds) and inds[start] == j and inds[end-1] == j:
        #             filter[start:end] = True
        # else:
        #     # OLD METHOD HERE SO I CAN SHIT ON IT AS AN EXAMPLE
        #     # we do an iterated "and" over the not equals
        #     # and apply a "not" at the end
        #     filter = np.full(len(inds), True)
        #     for j in block:
        #         # hastag meta
        #         filter[filter] = inds[filter] != j
        #     filter = np.logical_not(filter)
        # print(len(inds), np.count_nonzero(filter))

        # I was hoping I could make use of that filter stuff to
        # build the mapping but it honestly seems like the two are completely
        # different?

        # we find the first position where the positions occur (we assume they _do_ occur)
        # using `searchsorted`
        mapping = np.searchsorted(block, np.arange(shape))#, sorter=np.arange(len(block)))

        return filter, mapping

    def _get_filtered_elements(self, blocks, data, inds):

        inds = list(inds)
        for i, b, s in zip(range(len(blocks)), blocks, self.shape):
            if b is not None:
                ixs = inds[i]
                filter, mapping = self._find_block_alignment(ixs, b, s)
                # print(ixs, b, s, filter)
                inds = [ix[filter] for ix in inds]
                data = data[filter]
                inds[i] = mapping[inds[i]]

        return data, inds

    def _get_element(self, idx, pull_elements=None):
        """
        Convert idx into a 1D index or slice or whatever and then convert it back to the appropriate 2D shape

        :param i:
        :type i:
        :return:
        :rtype:
        """

        # TODO: take a look at the numpy "fancy indexing" code to speed this up...
        if pull_elements is None:
            # we check first to see if we were asked for just a single vector of elements
            # the detection heuristic is basically: is everything just a slice of ints or nah
            if isinstance(idx, (int, np.integer)):
                idx = (idx,)
            pull_elements = len(idx) == len(self.shape) and all(isinstance(x, (int, np.integer)) for x in idx)
            if not pull_elements:
                pull_elements = all(not isinstance(x, (int, np.integer, slice)) for x in idx)
                if pull_elements:
                    e1 = len(idx[0])
                    pull_elements = all(len(x) == e1 for x in idx)

        if pull_elements:
            # try:
            flat = self._ravel_indices(idx, self.shape)
            # except:
            #     raise Exception(idx)

            unflat = self._unravel_indices(flat, self.data.shape)
            res = self.data[unflat]
            if not isinstance(flat, int):
                res = np.array(res)
            return res
        else:
            # need to compute the shape of the resultant block
            # we treat slice(None, None, None) as a special case because
            # it's so common
            blocks = [
                (
                    np.array([i]) if isinstance(i, (int, np.integer)) else (
                        np.arange(s)[i,].flatten()
                            if not (isinstance(i, slice) and i == slice(None, None, None)) else
                        None
                    )
                )
                for i, s in zip(idx, self.shape)
            ]
            # we filter out places where new_shape[i] == 1 at a later stage
            # for now we just build out the total shape it _would_ have with axes of len 1
            new_shape = [
                            len(x) if x is not None else s for x,s in zip(blocks, self.shape[:len(blocks)])
                         ] + list(self.shape[len(blocks):])

            # now we iterate over each block and use it as a successive filter on our non-zero positions
            data, inds = self.block_data
            data, inds = self._get_filtered_elements(blocks, data, inds)

            # now that we've filtered our data, we filter out axes of size 1
            # print(inds, new_shape)
            inds = [ix for ix, j in zip(inds, new_shape) if j > 1]
            new_shape = tuple(j for j in new_shape if j > 1)
            # we also apply this to the indices that we're filtering on

            # if idx[0] == 0 and isinstance(idx[2], slice):
            #     raise Exception(blocks)

            # finally, we track our indices so that we don't need to recompute anything later
            if len(new_shape) > 2:
                total_shape = (np.prod(new_shape[:-2]) * new_shape[-2], new_shape[-1])
                flat = self._ravel_indices(inds, new_shape)
                unflat = self._unravel_indices(flat, total_shape)
            elif len(new_shape) == 1:
                flat = None
                unflat = inds+[np.zeros((len(inds[0]),))]
                total_shape = new_shape+(1,)
            else:
                flat = None
                unflat = inds
                total_shape = new_shape

            # raise Exception(blocks, new_shape, len(inds), len(unflat))
            # od = data
            try:
                data = self._build_data(data, unflat, total_shape)
            except Exception as e:
                # print(data, data.shape, unflat, total_shape)
                data = e
            # raise Exception(unflat, od, data, inds, total_shape)
            if isinstance(data, Exception):
                raise IndexError("{}: couldn't take element {} of array {} (Got Error: '{}')".format(
                    type(self).__name__,
                    idx,
                    self,
                    data
                ))

            new = type(self)(data, shape=new_shape, layout=self.fmt)
            if flat is None:
                new.block_inds = inds
            else:
                new.block_inds = flat, inds
            new._block_data_sorted = True
            return new

    def _set_data(self, unflat, val):
        """
        Tries to explicitly assign but if that fails drops back to CSR and then reconverts

        :param unflat:
        :type unflat:
        :param val:
        :type val:
        :return:
        :rtype:
        """
        # Just here so I can be smart
        try:
            self.data[unflat] = val
        except TypeError:
            if self.data.shape[0] > self.data.shape[1]:
                sub_sparse = sp.csc_matrix(self.data)
            else:
                sub_sparse = sp.csr_matrix(self.data)
            sub_sparse[unflat] = val
            self.data = self.fmt(sub_sparse)

        # TODO: figure out how to resolve the collisions
        #       more efficiently
        self._block_vals = None
        self._block_inds = None

    def _set_element(self, idx, val):
        """
        Convert idx into a 1D index or slice or whatever and then convert it back to the appropriate 2D shape.
        Then hope that the val can be set on the sp.spmatrix backend...

        :param i:
        :type i:
        :return:
        :rtype:
        """

        # we check first to see if we were asked for just a single vector of elements
        # the detection heuristic is basically: is everything just a slice of ints or nah
        if isinstance(idx, (int, np.integer)):
            idx = (idx,)
        set_elements = len(idx) == len(self.shape) and all(isinstance(x, (int, np.integer)) for x in idx)
        if not set_elements:
            set_elements = all(not isinstance(x, (int, np.integer, slice)) for x in idx)
            if set_elements:
                e1 = len(idx[0])
                set_elements = all(len(x) == e1 for x in idx)

        if set_elements:
            flat = self._ravel_indices(idx, self.shape)
            unflat = self._unravel_indices(flat, self.data.shape)
            # try:

            self._set_data(unflat, val)
            # except TypeError:
            #     # need to construct a new data object in its entirety :weep:
            #     # or convert to an assignable format?
            #
        else:
            # need to compute the proper block indices, unfortunately
            # so that they can be down-converted to their 2D equivalents
            blocks = [
                (
                    np.array([i]) if isinstance(i, (int, np.integer)) else (
                        np.arange(s)[i,].flatten()
                    )
                )
                for i, s in zip(idx, self.shape)
            ]

            block_inds = np.array([p for p in ip.product(*blocks)]).T

            flat = self._ravel_indices(block_inds, self.shape)
            unflat = self._unravel_indices(flat, self.data.shape)

            self._set_data(unflat, val)

    def _del_element(self, idx):
        """
        Convert idx into a 1D index or slice or whatever and then convert it back to the appropriate 2D shape.
        Then hope that the val can be deleted on the sp.spmatrix backend...

        :param i:
        :type i:
        :return:
        :rtype:
        """

        # we check first to see if we were asked for just a single vector of elements
        # the detection heuristic is basically: is everything just a slice of ints or nah
        if isinstance(idx, (int, np.integer)):
            idx = (idx,)
        set_elements = len(idx) == len(self.shape) and all(isinstance(x, (int, np.integer)) for x in idx)
        if not set_elements:
            set_elements = all(not isinstance(x, (int, np.integer, slice)) for x in idx)
            if set_elements:
                e1 = len(idx[0])
                set_elements = all(len(x) == e1 for x in idx)

        if set_elements:
            flat = self._ravel_indices(idx, self.shape)
            unflat = self._unravel_indices(flat, self.data.shape)
            del self.data[unflat]
            # TODO: figure out how to resolve the collisions
            #       more efficiently
            self._block_vals = None
            self._block_inds = None
        else:
            # need to compute the proper block indices, unfortunately
            # so that they can be down-converted to their 2D equivalents
            blocks = [
                (
                    np.array([i]) if isinstance(i, (int, np.integer)) else (
                        np.arange(s)[i,].flatten()
                    )
                )
                for i, s in zip(idx, self.shape)
            ]

            block_inds = np.array([p for p in ip.product(*blocks)]).T

            flat = self._ravel_indices(block_inds, self.shape)
            unflat = self._unravel_indices(flat, self.data.shape)
            del self.data[unflat]
            # TODO: figure out how to resolve the collisions
            #       more efficiently
            self._block_vals = None
            self._block_inds = None

    def savez(self, file, compressed=True):
        """
        Saves a SparseArray to a file (must have the npz extension)
        :param file:
        :type file:
        :param compressed:
        :type compressed:
        :return: the saved file
        :rtype: str
        """

        # sp.save_npz already sticks the extension on so it doesn't hurt to explicitly add it...
        if isinstance(file, str) and os.path.splitext(file)[1] != ".npz":
            file += ".npz"
        sp.save_npz(file, self.data, compressed=compressed)
        bleh = np.load(file)
        flat, unflat = self.block_inds
        bv = self.block_vals
        if compressed:
            np.savez_compressed(
                file,
                _block_shape=np.array(self.shape),
                _block_inds_flat=np.array(flat),
                _block_inds_unflat=np.array(unflat),
                _block_vals=bv,
                compressed=compressed,
                **bleh
            )
        else:
            np.savez(
                file,
                _block_shape=np.array(self.shape),
                _block_inds_flat=np.array(flat),
                _block_inds_unflat=np.array(unflat),
                _block_vals=bv,
                compressed=compressed,
                **bleh
                )
        return file
    @classmethod
    def loadz(cls, file):
        """
        Loads a SparseArray from an npz file
        :param file:
        :type file:
        :return:
        :rtype: SparseArray
        """
        data = sp.load_npz(file) #type: sp.spmatrix
        other = np.load(file)
        new = cls(data, shape=other['_block_shape'], layout=type(data), initialize=False)
        new._block_inds = (other['_block_inds_flat'], other['_block_inds_unflat'])
        new._block_inds = other['_block_vals']
        return new

    def __getitem__(self, item):
        return self._get_element(item)
    def __setitem__(self, item, val):
        return self._set_element(item, val)
    def __delitem__(self, item):
        return self._del_element(item)

    def __repr__(self):
        return "{}(<{}> nonzero={})".format(type(self).__name__,
                                           ", ".join([str(x) for x in self.shape]),
                                          self.non_zero_count
                                           )

# import tensorflow
class TensorFlowSparseArray(SparseArray):
    """
    Provides a SparseArray implementation that uses TensorFlow as the backend
    """
    def __init__(self, data, dtype=None):
        import tensorflow
        self.tensor = self._init_tensor(data, dtype=dtype) #type: tensorflow.sparse.SparseTensor

    def _init_tensor(self, data, dtype=None):
        """
        :param data:
        :type data:
        :return:
        :rtype: tensorflow.sparse.SparseTensor
        """
        import tensorflow
        if isinstance(data, TensorFlowSparseArray):
            import copy
            new = copy.copy(data.tensor)
            if dtype is not None and new.dtype != dtype:
                raise NotImplementedError("not sure how to convert dtypes...")
            return new
        elif isinstance(data, tensorflow.sparse.SparseTensor):
            if dtype is not None and data.dtype != dtype:
                raise NotImplementedError("not sure how to convert dtypes...")
            return data
        elif isinstance(data, ScipySparseArray):
            vals, inds = data.block_data
            if dtype is not None and vals.dtype != dtype:
                vals = vals.astype(dtype)
            return tensorflow.sparse.SparseTensor(inds, vals, data.shape)
        elif isinstance(data, sp.spmatrix):
            row_inds, col_inds, vals = sp.find(data)
            if dtype is not None and vals.dtype != dtype:
                vals = vals.astype(dtype)
            return tensorflow.sparse.SparseTensor((row_inds, col_inds), vals, data.shape)
        elif isinstance(data, np.ndarray) and data.dtype != np.dtype(object):
            return self._init_tensor(ScipySparseArray(data))
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            if all(isinstance(x, (int, np.integer)) for x in data):
                # empty tensor
                return tensorflow.sparse.SparseTensor([], [], data)
            elif len(data[1]) == 2 and all(isinstance(x, (int, np.integer)) for x in data[1]):
                # second arg is shape
                vals = np.array(data[0][0])
                inds = data[0][1]
                if dtype is not None and vals.dtype != dtype:
                    vals = vals.astype(dtype)
                return tensorflow.sparse.SparseTensor(inds, vals, data[1])
            else:
                raise TypeError("don't know how to turn {} into a tensorflow SparseTensor".format(
                    data
                ))
        else:
            raise TypeError("don't know how to turn {} into a tensorflow SparseTensor".format(
                data
            ))

    @property
    def shape(self):
        """
        Provides the shape of the sparse array
        :return:
        :rtype: tuple[int]
        """
        return tuple(self.tensor.shape)

    def to_state(self, serializer=None):
        """
        Provides just the state that is needed to
        serialize the object
        :param serializer:
        :type serializer:
        :return:
        :rtype:
        """
        raise NotImplementedError("coming")

    @classmethod
    def from_state(cls, state, serializer=None):
        """
        Loads from the stored state
        :param serializer:
        :type serializer:
        :return:
        :rtype:
        """
        raise NotImplementedError("coming")

    @classmethod
    def empty(cls, shape, dtype=None, **kw):
        """
        Returns an empty SparseArray with the appropriate shape and dtype
        :param shape:
        :type shape:
        :param dtype:
        :type dtype:
        :param kw:
        :type kw:
        :return:
        :rtype:
        """
        return cls(shape, dtype=dtype)

    @property
    def block_data(self):
        """
        Returns the row and column indices and vector of
        values that the sparse array is storing
        :return:
        :rtype: Tuple[np.ndarray, Iterable[np.ndarray]]
        """
        return (self.tensor.values, self.tensor.indices)

    def transpose(self, axes):
        """
        Returns a transposed version of the tensor
        :param axes:
        :type axes:
        :return:
        :rtype:
        """
        import tensorflow as tf
        new = tf.sparse.transpose(self.tensor, perm=axes)
        return type(self)(new)

    def ascoo(self):
        """
        Converts the tensor into a scipy COO matrix...
        :return:
        :rtype: sp.coo_matrix
        """
        return ScipySparseArray(self).ascoo()
    def ascsr(self):
        """
        Converts the tensor into a scipy COO matrix...
        :return:
        :rtype: sp.coo_matrix
        """
        return ScipySparseArray(self).ascsr()
    def reshape(self, newshape):
        """
        Returns a reshaped version of the tensor
        :param axes:
        :type axes:
        :return:
        :rtype:
        """
        import tensorflow as tf
        new = tf.sparse.reshape(self.tensor, newshape)
        return type(self)(new)

    def __truediv__(self, other):
        return self.true_multiply(1 / other)

    def __rtruediv__(self, other):
        return self.true_multiply(1 / other)

    def __rmul__(self, other):
        return self.true_multiply(other)

    def __mul__(self, other):
        return self.true_multiply(other)

    def true_multiply(self, other):
        """
        Multiplies self and other
        :param other:
        :type other:
        :return:
        :rtype:
        """
        import tensorflow as tf
        if isinstance(other, TensorFlowSparseArray):
            new = self.tensor * other.tensor
            return type(self)(new)
        elif isinstance(other, (tf.Tensor, tf.SparseTensor)):
            new = self.tensor * other
            return type(self)(new)
        elif isinstance(other, (int, np.integer, float, np.floating)):
            if other == 0:
                return type(self).empty(self.shape)
            else:
                vals, inds = self.block_data
                vals = vals * other
                return type(self)(((vals, inds), self.shape))

    @staticmethod
    def _tf_sparse_dot(a, b):
        from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
        ashp = tuple(a.shape)
        if len(ashp) > 2:
            arows = int(np.prod(ashp[:-1]))
            acols = ashp[-1]
            a = a.reshape((arows, acols))
        bshp = tuple(b.shape)
        if len(bshp) > 2:
            brows = bshp[0]
            bcols = int(np.prod(bshp[1:]))
            b = b.reshape((brows, bcols))
        # woof..
        csr_1 = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            a.indices,
            a.values,
            a.shape
        )
        csr_2 = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            b.indices,
            b.values,
            b.shape
        )
        new = sparse_csr_matrix_ops.sparse_matrix_mat_mul(csr_1, csr_2)
        if len(ashp) > 2 or len(b) > 2:
            remainder_a = ashp[:-1]
            remainder_b = bshp[1:]
            new = new.reshape(remainder_a + remainder_b)
        return new

    def dot(self, other):
        """
        Takes a regular dot product of self and other
        :param other:
        :type other:
        :param axes:
        :type axes:
        :return:
        :rtype:
        """
        import tensorflow as tf

        if isinstance(other, tf.SparseTensor):
            # one main case to which all others will defer

            return type(self)(self._tf_sparse_dot(self.tensor, other))
        elif isinstance(other, TensorFlowSparseArray):
            return self.dot(other.tensor)
        elif isinstance(other, (sp.spmatrix, ScipySparseArray)):
            # heavy duty...
            return self.dot(type(self)(other))
        elif isinstance(other, tf.Tensor):
            return tf.sparse.sparse_dense_matmul(self, other)
        else:
            raise TypeError("dot not defined for {} and {}".format(type(self).__name__, type(other).__name__))


def _dot(a, b):
    if isinstance(a, SparseArray):
        a = a.ascsr()
    if isinstance(b, SparseArray):
        b = b.ascsr()

    if isinstance(a, sp.spmatrix):
        dense_output = False
        if not isinstance(b, sp.spmatrix):
            dense_output = True
            # we convert it to a sparse thing to avoid memory blow-ups when we get dense unpacking...?
            b = sp.csr_matrix(b)

        wat = a.dot(b)
        # raise Exception(wat.shape, a.shape, b.shape)
    else:
        dense_output = True
        # we convert it to a sparse thing to avoid memory blow-ups when we get dense unpacking...?
        a = sp.csr_matrix(a)
        if not isinstance(b, sp.spmatrix):
            dense_output = True
            # we convert it to a sparse thing to avoid memory blow-ups when we get dense unpacking...?
            b = sp.csr_matrix(b)
        wat = a.dot(b)

    # print(">>>>>", a.shape, b.shape, wat.shape)

    if dense_output:
        wat = wat.toarray()
    return wat

# def asCOO(a):
#     if not isinstance(a, sp.coo_matrix):
#         a = sp.coo_matrix(a)
#     return a
def sparse_tensordot(a, b, axes=2):
    """Defines a version of tensordot that uses sparse arrays, adapted from the sparse package on PyPI

    :param a: the array to contract from
    :type a: SparseArray | sp.spmatrix | np.ndarray
    :param b: the array to contract with
    :type b: SparseArray | sp.spmatrix | np.ndarray
    :param axes: the axes to contract along
    :type axes: int | Iterable[int]
    :return:
    :rtype:
    """

    try:
        iter(axes)
    except TypeError:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    as_ = a.shape
    nda = a.ndim
    bs = b.shape
    ndb = b.ndim
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum ({}{}@{}{})".format(as_, axes_a, bs, axes_b))

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = [notin, axes_a]
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    totels_a = np.prod(as_)
    newshape_a = (totels_a // N2, N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = [axes_b, notin]
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    totels_b = np.prod(bs)
    newshape_b = (N2, totels_b // N2)
    oldb = [bs[axis] for axis in notin]

    if any(dim == 0 for dim in ip.chain(newshape_a, newshape_b)):
        res = sp.csr_matrix((olda + oldb), dtype=b.dtype)
        # dense output...I guess that's clean?
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            res = res.todense()
        return res

    if newshape_a[0] > 2 * newshape_b[1]:
        # indicates it will be faster to transpose both arrays, dot, and transpose back
        if a.ndim > 1:
            # print(">>>>", newaxes_a)
            at = a.transpose(newaxes_a[1] + newaxes_a[0])
            # if newshape_a[1] > 5000:
            #     raise Exception("why")
        else:
            at = a

        at = at.reshape(np.flip(newshape_a))

        if b.ndim > 1:
            bt = b.transpose(newaxes_b[1] + newaxes_b[0])
        else:
            bt = b
        bt = bt.reshape(np.flip(newshape_b))

        # if isinstance(bt, np.ndarray):
        #     bt = sp.csr_matrix(bt)

        res = _dot(bt, at).transpose()

    else:
        if a.ndim > 1:
            at = a.transpose(newaxes_a[0] + newaxes_a[1])
        else:
            at = a
        at = at.reshape(newshape_a)

        if b.ndim > 1:
            bt = b.transpose(newaxes_b[0] + newaxes_b[1])
        else:
            bt = b
        bt = bt.reshape(newshape_b)

        res = _dot(at, bt)

    if isinstance(res, sp.spmatrix):
        if isinstance(a, ScipySparseArray):
            res = ScipySparseArray(res, shape=olda + oldb, layout=a.fmt)
        elif isinstance(b, sp.spmatrix):
            res = ScipySparseArray(res, shape=olda + oldb, layout=b.fmt)
        else:
            res = res.reshape(olda + oldb)
    else:
        res = res.reshape(olda + oldb)

    return res