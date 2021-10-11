"""
Provides suppport for the openchemistry library
"""

__all__ = [
    "OpenChemInterface"
]


class OpenChemInterface:
    """
    A simple class to support operations that make use of the OpenChemistry toolkit
    """
    OC_SUPPORTED = None
    def __init__(self):
        self._lib = None
    @classmethod
    def _lib_installed(cls):
        if cls.OC_SUPPORTED is None:
            try:
                import openchemistry.io
            except ImportError:
                cls.OC_SUPPORTED = False
            else:
                cls.OC_SUPPORTED = True

        return cls.OC_SUPPORTED
    @property
    def lib(self):
        if self._lib is None:
            if self._lib_installed():
                import openchemistry.io as lib
                self._lib = lib
            else:
                raise ImportError("OpenChemistry isn't installed")
        return self._lib
    def method(self, name):
        return getattr(self._lib, name)