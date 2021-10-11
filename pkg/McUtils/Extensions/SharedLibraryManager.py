"""
Defines a SharedLibrary object that makes it relatively straightforward to use
extensions direclty from .so files
"""

import os, ctypes
from .ArgumentSignature import FunctionSignature

__all__ = [
    "SharedLibraryFunction"
]



class SharedLibraryFunction:
    """
    An object that provides a way to call into a shared library function
    """

    def __init__(self,
                 shared_library,
                 signature,
                 docstring=None,
                 call_directory=None
                 ):
        """
        :param shared_library: the path to the shared library file you want to use
        :type shared_library: str |
        :param function_signature: the signature of the function to load
        :type function_signature: FunctionSignature
        :param call_directory: the directory for calling
        :type call_directory: str
        :param docstring: the docstring for the function
        :type docstring: str
        """
        self._lib = None
        self._lib_file = shared_library
        self._fun = None
        self._sig = signature
        self._doc = docstring
        if call_directory is None:
            call_directory = os.path.dirname(shared_library)
        self._dir = call_directory  # we could be kinder here and do stuff like add support for ".." and friends

    class InDir:
        """
        A super simple context manager that manages going into a directory and then leaving when finished
        """

        def __init__(self, dir_name):
            self._to = dir_name
            self._from = None

        def __enter__(self):
            self._from = os.getcwd()
            os.chdir(self._to)

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._from is not None:
                os.chdir(self._from)
            self._from = None

    @property
    def lib(self):
        if self._lib is None:
            with self.InDir(self._dir):
                self._lib = ctypes.cdll.LoadLibrary(self._lib_file)
        return self._lib

    def initialize(self):
        if self._fun is None:
            # means we need to load it from the shared lib
            if isinstance(self._fname, str):
                self._fun = getattr(self.lib, self._fname)
            # now initialize the arg signature
            self._fun.restype = self._sig.return_type
            self._fun.argtypes = self._sig.arg_list
    def doc(self):
        return self._sig.cpp_signature+"\n"+self._doc
    def __repr__(self):
        return "{}({}, {})".format(
            type(self).__name__,
            self._sig,
            self.lib
        )
    def call(self, **kwargs):
        """
        Calls the function we loaded.
        This will be parallelized out to handle more complicated usages.

        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        ...
