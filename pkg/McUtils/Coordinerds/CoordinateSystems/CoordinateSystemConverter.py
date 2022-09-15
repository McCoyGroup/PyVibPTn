"""
Provides the conversion framework between coordinate systems
"""

from collections import OrderedDict as odict, deque
import os, abc, numpy as np, weakref
from ...Extensions import ModuleLoader
from ...Numputils import apply_pointwise

__all__ = [
    "CoordinateSystemConverters",
    "CoordinateSystemConverter",
    "SimpleCoordinateSystemConverter"
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
        return np.array([self.convert(coords, **kwargs) for coords in coords_list])

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

    def __call__(self, coords, **kwargs):
        if coords.ndim > 2: #TODO: make this a more robust check for the future
            return self.convert_many(coords, **kwargs)
        else:
            return self.convert(coords, **kwargs)

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
    converter_graph = None
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
            proxy_function = lambda x, y: x.is_compatible(x, y)
            self.converter_graph = ConversionGraph(proxy_function=proxy_function)

            from .DefaultConverters import __converters__ as converters
            for conv in converters:
                self._register(*conv.types, conv, move_to_end=True)

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
        if (system1, system2) in cls.converters:
            path = [(system1, system2)]
        else:
            path = cls.converter_graph.find_path_bfs(system1, system2)
        if path is None:
            raise KeyError(
                "{}: no rules for converting coordinate system {} to {} in {}".format(cls.__name__, system1, system2, [
                    "{}=>{}".format(
                        k.__name__ if isinstance(k, type) else type(k).__name__,
                        b.__name__ if isinstance(b, type) else type(b).__name__
                    ) for k, b in cls.converters
                ])
            )
        elif len(path) == 1:
            converter = cls.converters[path[0]]
        else:
            conversions = [(cls.converters[p], p) for p in path]
            return ChainedCoordinateSystemConverter([system1, system2], conversions)

        #
        # def _get_pathy_conversion(self, src, targ):
        #     conv_path = self._unit_graph.find_path_bfs(src, targ)
        #     if conv_path is None:
        #         return conv_path
        #     cval = 1
        #     for me, you in zip(conv_path, conv_path[1:]):
        #         cval *= self.data[(me, you)]["Value"]
        #     return cval
        #
        # try:
        #     converter = cls.converters[(system1, system2)]
        # except KeyError:
        #     for key_pair, conv in reversed(cls.converters.items()):
        #         if (
        #                 system1.has_conversion(system2)
        #                 or (system1.is_compatible(key_pair[0]) and system2.is_compatible(key_pair[1]))
        #         ):
        #             converter = conv
        #             break
        #     else:
        #         raise KeyError(
        #             "{}: no rules for converting coordinate system {} to {} in {}".format(cls.__name__, system1, system2, [
        #                 "{}=>{}".format(k.__name__, b.__name__ ) for k,b in cls.converters
        #             ])
        #         )

        return converter

    @classmethod
    def register_converter(cls, system1, system2, converter, check=True):
        """
        Registers a converter between two coordinate systems

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
        cls._register(system1, system2, converter)
    @classmethod
    def _register(cls, system1, system2, converter, move_to_end=False):
        cls.converters[(system1, system2)] = converter
        if move_to_end:
            cls.converters.move_to_end((system1, system2))
        graph = cls.converter_graph
        cls.converter_graph.add(system1, system2)
        for k in cls.converter_graph.keys():
            if graph.proxy_function(system1, k):
                cls.converters[(k, system2)] = converter
                if move_to_end:
                    cls.converters.move_to_end((k, system2))
            elif graph.proxy_function(system2, k):
                cls.converters[(system1, k)] = converter
                if move_to_end:
                    cls.converters.move_to_end((system1, k))

class ConversionGraph:
    """
    Pulled from the UnitGraph stuff
    """

    def __init__(self, stuff_to_update=(), proxy_function=None):
        self._graph = {}
        self.update(stuff_to_update)
        self.proxy_function = proxy_function

    def __contains__(self, item):
        return item in self._graph
    def add(self, node, connection):
        if node in self._graph:
            self._graph[node].add(connection)
        else:
            self._graph[node]={connection}
        if self.proxy_function is not None:
            for k in self._graph:
                if self.proxy_function(node, k):
                    self._graph[k].add(connection)
                elif self.proxy_function(connection, k):
                    self._graph[node].add(k)
        if not connection in self._graph:
            self._graph[connection] = set()
    def keys(self):
        return self._graph.keys()
    def update(self, iterable):
        for connection in iterable:
            self.add(*connection)
    def find_path_bfs(self, start, end):
        # we use a little poor-man's Dijkstra to find the shortest unit conversion path
        identity_function = self.proxy_function
        if identity_function is None:
            if not start in self._graph or not end in self._graph:
                return None
        else:
            if start not in self._graph:
                for k in self._graph:
                    if identity_function(start, k):
                        start = k
                        break
                else:
                    return None
            if end not in self._graph:
                for k in self._graph:
                    if identity_function(end, k):
                        end = k
                        break
                else:
                    return None

        q = deque() # deque as a FIFO queue
        q.append(start)
        parents = { start:None }
        steps = 0
        while len(q)>0:
            cur = q.pop()
            steps += 1
            for k in self._graph[cur]:
                if k is end:
                    parents[k] = cur
                    q = []
                    break
                elif k not in parents:
                    q.append(k)
                    parents[k] = cur
        if end not in parents:
            return None
        else:
            path = []*steps
            cur = end
            while cur is not start:
                nxt = parents[cur]
                path.append((nxt, cur))
                cur = nxt
            # path.append(start)
            return list(reversed(path))

class SimpleCoordinateSystemConverter(CoordinateSystemConverter):
    def __init__(self, types, conversion, **opts):
        super().__init__(**opts)
        self._types = types
        self.conversion = conversion
    @property
    def types(self):
        return self._types
    def convert(self, coords, **kw):
        return self.conversion(coords, **kw)
    def convert_many(self, coords, **kw):
        return self.convert(coords, **kw)
class ChainedCoordinateSystemConverter(CoordinateSystemConverter):

    def __init__(self, types, conversions, **opts):
        super().__init__(**opts)
        self._types = types
        self.conversions = conversions
    @property
    def types(self):
        return self._types
    def convert(self, crds, **kwargs):
        cur = crds
        for f, p in self.conversions:
            if hasattr(p[0], 'convert_coords') and not isinstance(p[0], type):
                cur = p[0].convert_coords(cur, p[1], converter=f, **kwargs)
            else:
                cur = f(cur, **kwargs)
            if isinstance(cur, tuple):
                cur, kwargs = cur
            else:
                kwargs = {}
        return cur, kwargs
    def convert_many(self, coords, **kw):
        return self.convert(coords, **kw)

CoordinateSystemConverter.converters = weakref.ref(CoordinateSystemConverters)