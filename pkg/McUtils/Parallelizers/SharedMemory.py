"""
Provides classes for working with `multiprocessing.SharedMemory`
in a slightly more convenient way
"""

import abc, os, numpy as np, typing

from ..Scaffolding import BaseObjectManager, NDarrayMarshaller

class SharedMemoryInterface(typing.Protocol):

    @abc.abstractmethod
    def __init__(self, name=None, create=False, size=None):
        raise NotImplementedError("interface class")

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError("interface class")

    @abc.abstractmethod
    def unlink(self):
        raise NotImplementedError("interface class")

class SharedNDarray:
    """
    Provides a very simple tracker for shared NumPy arrays
    """

    def __init__(self, shape, dtype, buf, autoclose=True, parallelizer=None):
        """
        :param shape:
        :type shape: tuple[int]
        :param dtype:
        :type dtype: np.dtype
        :param buf:
        :type buf: SharedMemoryInterface
        :param parallelizer:
        :type parallelizer: Parallelizer
        """
        self.dtype = dtype
        self.shape = shape
        self.buf = buf
        self.autoclose = autoclose
        self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.buf)
        self.parallelizer = parallelizer

    def __getstate__(self):
        return {
            'dtype': self.dtype,
            'shape': self.shape,
            'buf': self.buf,
            'autoclose': self.autoclose,
            'parallelizer': self.parallelizer
        }

    def __setstate__(self, state):
        self.dtype = state['dtype']
        self.shape = state['shape']
        self.buf = state['buf']
        self.autoclose = state['autoclose']
        self.parallelizer = state['parallelizer']
        self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.buf)

    @classmethod
    def from_array(cls, arr, buf,
                   autoclose=None,
                   parallelizer=None
                   ):
        """
        Initializes by pulling metainfo from an array

        :param arr:
        :type arr: np.ndarray
        :param buf:
        :type buf: SharedMemoryInterface
        :return:
        :rtype:
        """
        buf[:] = arr
        opts = {}
        if autoclose is not None:
            opts['autoclose'] = autoclose
        if parallelizer is not None:
            opts['parallelizer'] = parallelizer
        return cls(arr.shape, arr.dtype, buf, **opts)

    def __setitem__(self, key, value):
        self.array[key] = value

    def __getitem__(self, item):
        return self.array[item]

    def close(self):
        self.buf.close()

    def unlink(self):
        self.buf.unlink()

    def __del__(self):
        if self.autoclose:
            self.close()
            if self.parallelizer is not None and self.parallelizer.on_main:
                self.unlink()


class SharedObjectManager(BaseObjectManager):
    """
    Provides a high-level interface to create a manager
    that supports shared memory objects through the multiprocessing
    interface
    Only supports data that can be marshalled into a NumPy array.
    """

    def __init__(self, obj, mem_manager=None, marshaller=None, parallelizer=None):
        """
        :param mem_manager: a memory manager like `multiprocessing.SharedMemoryManager`
        :type mem_manager:
        :param obj: the object whose attributes should be given by shared memory objects
        :type obj:
        """

        super().__init__(obj)
        try:
            from multiprocessing import shared_memory
            self.api = shared_memory
        except ImportError:
            self.api = None
            if mem_manager is None:
                raise NotImplementedError(
                    "{}: either `multiprocessing` needs the `shared_memory` submodule or `mem_manager` must be provided".format(
                        type(self).__name__
                    )
                )
        self.mem_manager = mem_manager
        if marshaller:
            marshaller = NDarrayMarshaller()
        self.marshaller = marshaller
        self.buffers = {}
        self.parallelizer = parallelizer

    def create_shared_array(self, data, name=None):
        """
        Makes a SharedNDarray object for an existing data chunk

        :param data:
        :type data: np.ndarray
        :return:
        :rtype: SharedNDarray
        """

        if self.mem_manager is not None:
            shm = self.mem_manager.SharedMemory
        else:
            shm = self.api.SharedMemory
        buf = shm.SharedMemory(name, create=True, size=data.size)
        return SharedNDarray.from_array(data, buf, parallelizer=self.parallelizer)

    def delete_shared_array(self, shared_array):
        """
        Closes a buffer for a numpy array

        :param shared_array:
        :type shared_array: SharedNDarray
        :return:
        :rtype:
        """
        shared_array.close()

    def update_shared_array(self, shared_array, data):
        """
        Updates a buffer for a numpy array

        :param shared_array:
        :type shared_array: SharedNDarray
        :return:
        :rtype:
        """
        try:
            shared_array.buf[:] = data
        except ValueError:
            shared_array = self.create_shared_array(data)
        return shared_array

    def _save_to_buffer(self, tree, name, data):
        arr = self.marshaller.convert(data)
        del data # just to help minimize bugs
        if isinstance(arr, np.ndarray):
            if isinstance(tree, dict):
                if name in tree:
                    tree[name] = self.update_shared_array(tree[name], arr)
                else:
                    tree[name] = self.create_shared_array(arr)
            elif isinstance(name, (int, np.integer)):
                if len(tree) < name:
                    if tree[name] is None:
                        tree[name] = self.create_shared_array(arr)
                    else:
                        tree[name] = self.update_shared_array(tree[name], arr)
                else:
                    padding = len(tree) - name - 1
                    tree = tree + [None] * padding
                    tree[name] = self.create_shared_array(arr)
        else:
            # walk an expression tree, updating buffers as needed
            if name in tree:
                subtree = tree[name]
            elif isinstance(arr, dict):
                subtree = {}
            else:
                subtree = []

            if isinstance(arr, dict):
                for k,v in arr.items():
                    self._save_to_buffer(subtree, k, v)
            else:
                for i,v in enumerate(arr):
                    self._save_to_buffer(subtree, i, v)

    def save_attr(self, attr):
        ...