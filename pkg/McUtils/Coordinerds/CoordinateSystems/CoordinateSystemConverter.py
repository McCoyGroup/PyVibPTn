"""
Provides the conversion framework between coordinate systems
"""

from collections import OrderedDict as odict
import os, abc, numpy as np, weakref
from ...Extensions import ModuleLoader

__all__ = [
    "CoordinateSystemConverters",
    "CoordinateSystemConverter"
]

__reload_hook__ = ["...Extensions", '.CartesianToZMatrix', '.ZMatrixToCartesian']

######################################################################################################
##
##                                   CoordinateSystemConverter Class
##
######################################################################################################
class CoordinateSystemConverter(metaclass=abc.ABCMeta):
    """
    A base class for type converters
    """

    converters = None

    @property
    @abc.abstractmethod
    def types(self):
        """The types property of a converter returns the types the converter converts

        """
        pass

    def convert_many(self, coords_list, **kwargs):
        """Converts many coordinates. Used in cases where a CoordinateSet has higher dimension
        than its basis dimension. Should be overridden by a converted to provide efficient conversions
        where necessary.

        :param coords_list: many sets of coords
        :type coords_list: np.ndarray
        :param kwargs:
        :type kwargs:
        """
        return np.array((self.convert(coords, **kwargs) for coords in coords_list))

    @abc.abstractmethod
    def convert(self, coords, **kwargs):
        """The main necessary implementation method for a converter class.
        Provides the actual function that converts the coords set

        :param coords:
        :type coords: np.ndarray
        :param kwargs:
        :type kwargs:
        """
        pass

    def register(self, where=None, check=True):
        """
        Registers the CoordinateSystemConverter

        :return:
        :rtype:
        """
        if where is None:
            where = self.converters if not isinstance(self.converters, weakref.ref) else self.converters()
        where.register_converter(*self.types, self, check=check)

######################################################################################################
##
##                                   CoordinateSystemConverters Class
##
######################################################################################################
class CoordinateSystemConverters:
    """
    A coordinate converter class. It's a singleton so can't be instantiated.
    """

    converters = odict([])
    converters_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "Resources",
        "Converters"
    )
    converters_package = ".".join(__name__.split(".")[:-1])
    converter_type = CoordinateSystemConverter

    def __init__(self):
        raise NotImplementedError("{} is a singleton".format(type(self)))

    @classmethod
    def get_coordinates(self, coordinate_set):
        """Extracts coordinates from a coordinate_set
        """
        pass

    @classmethod
    def _get_converter_file(self, file):
        if os.path.exists(file):
            abspath = file
        else:
            abspath = os.path.join(self.converters_dir, file)
        return abspath

    @classmethod
    def load_converter(self, converter):

        file = self._get_converter_file(converter)
        loader = ModuleLoader(self.converters_dir, self.converters_package)
        env = loader.load(file)

        try:
            converters = env.__converters__
        except KeyError:
            raise KeyError("converter at {} missing field '{}'".format(file, "converters"))

        for conv in converters:
            type_pair = tuple(conv.types)
            self.converters[type_pair] = conv

    _converters_loaded = False
    @classmethod
    def _preload_converters(self):
        """
        Preloads Cartesian/ZMatrix converters.
        Maybe will load others in the future.
        :return:
        :rtype:
        """

        if not self._converters_loaded:
            from .CartesianToZMatrix import __converters__ as converters
            for conv in converters:
                type_pair = tuple(conv.types)
                self.converters[type_pair] = conv
                self.converters.move_to_end(type_pair)
            from .ZMatrixToCartesian import __converters__ as converters
            for conv in converters:
                type_pair = tuple(conv.types)
                self.converters[type_pair] = conv
                self.converters.move_to_end(type_pair)
            if os.path.exists(self.converters_dir):
                for file in os.listdir(self.converters_dir):
                    if os.path.splitext(file)[1] == ".py":
                        self.load_converter(file)
            self._converters_loaded = True

    @classmethod
    def get_converter(cls, system1, system2):
        """
        Gets the appropriate converter for two CoordinateSystem objects

        :param system1:
        :type system1: CoordinateSystem
        :param system2:
        :type system2: CoordinateSystem
        :return:
        :rtype:
        """

        cls._preload_converters()

        try:
            converter = cls.converters[(system1, system2)]
        except KeyError:
            for key_pair, conv in reversed(cls.converters.items()):
                if (
                        (isinstance(system1, key_pair[0]) and isinstance(system2, key_pair[1]))
                        or system1.name == key_pair[0].name and system2.name == key_pair[1].name
                ):
                    converter = conv
                    break
            else:
                raise KeyError(
                    "{}: no rules for converting coordinate system {} to {} in {}".format(cls.__name__, system1, system2, [
                        "{}=>{}".format(k.__name__, b.__name__ ) for k,b in cls.converters
                    ])
                )
        return converter

    @classmethod
    def register_converter(cls, system1, system2, converter, check=True):
        """Registers a converter between two coordinate systems

        :param system1:
        :type system1: CoordinateSystem
        :param system2:
        :type system2: CoordinateSystem
        :return:
        :rtype:
        """

        cls._preload_converters()
        if check and not isinstance(converter, cls.converter_type):
            raise TypeError('{}: registered converters should be subclasses of {} <{}> (got {} which inherits from {})'.format(
                cls.__name__,
                cls.converter_type, id(cls.converter_type),
                type(converter),
                ["{} <{}>".format(x, id(x)) for x in type(converter).__bases__]
            ))

        cls.converters[(system1, system2)] = converter

CoordinateSystemConverter.converters = weakref.ref(CoordinateSystemConverters)