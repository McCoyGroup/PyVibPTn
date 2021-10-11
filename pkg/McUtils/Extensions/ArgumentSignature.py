"""
Provides classes that are necessary for managing argument signatures
"""

__all__ = [
    "ArgumentType",
    "ArrayType",
    "PointerType",
    "RealType",
    "IntType",
    "BoolType",
    "Argument",
    "FunctionSignature"
]

import abc, ctypes, numpy as np, numpy.ctypeslib as npctypes

# TODO: need to finish up with the actual type inference & make it
#   so that we can use a `Argument.from_value` constructor
#   Also need to have call_single do the appropriate restructing of its Arguments
#   list so that stuff can cleanly go into the C-level + add some docstring-support

class ArgumentType(metaclass=abc.ABCMeta):
    """
    Defines a general purpose `ArgumentType` so that we can easily manage complicated type specs
    The basic idea is to define a hierarchy of types that can then convert themselves down to
    a `ctypes`-style spec as well as a C++ argument spec so that we can enable `SharedLibraryFunction`
    to use either the basic `ctypes` FFI or a more efficient, but fragile system based off of extension modules.
    This will be explicitly overridden by the `PrimitiveType`, `ArrayType` and `PointerType` subclasses that provide
    the actual useable classes.
    I'd really live to be integrate with what's in the `typing` module to be able to reuse that type-inference machinery
    """

    @property
    @abc.abstractmethod
    def ctypes_type(self):
        raise NotImplementedError()
    @property
    @abc.abstractmethod
    def cpp_type(self):
        raise NotImplementedError()
    @property
    @abc.abstractmethod
    def types(self):
        raise NotImplementedError()
    @property
    @abc.abstractmethod
    def dtypes(self):
        raise NotImplementedError()
    @abc.abstractmethod
    def isinstance(self, arg):
        raise NotImplementedError()
    @abc.abstractmethod
    def cast(self, arg):
        raise NotImplementedError()
class PrimitiveType(ArgumentType):
    """
    Defines a general purpose ArgumentType so that we can easily manage complicated type specs
    The basic idea is to define a hierarchy of types that can then convert themselves down to
    a `ctypes`-style spec as well as a C++ argument spec so that we can enable `SharedLibraryFunction`
    to use either the basic `ctypes` FFI or a more efficient, but fragile system based off of extension modules
    """

    def __init__(self,
                 name,
                 ctypes_spec,
                 cpp_spec,
                 capi_spec,
                 python_types,
                 numpy_dtypes,
                 serializer,
                 deserializer
                 ):
        """
        :param name: argument name (e.g. 'double')
        :type name: str
        :param ctypes_spec: the ctypes data-type that arguments of this type would be converted to
        :type ctypes_spec:
        :param cpp_spec: the C++ spec for this type (as a string)
        :type cpp_spec: str
        :param capi_spec: the python C-API string for use in `Py_BuildValue`
        :type capi_spec: str
        :param python_types: the python types that this argument maps onto
        :type python_types: Iterable[type]
        :param numpy_dtypes: the numpy dtypes that this argument maps onto
        :type numpy_dtypes: Iterable[np.dtype]
        :param serializer: a serializer for converting this object into a byte-stream
        :type serializer: Callable
        :param deserializer: a deserializer for converting the byte-stream into a C-level object
        :type deserializer: Callable
        """
        self._name = name
        self._ctypes_spec = ctypes_spec
        self._cpp_spec = cpp_spec
        self._capi_spec = capi_spec
        self._types = python_types
        self._dtypes = numpy_dtypes
        self._serializer = serializer
        self._deserializer = deserializer

    @property
    def name(self):
        return self._name
    @property
    def ctypes_type(self):
        return self._ctypes_spec
    @property
    def cpp_type(self):
        return self._cpp_spec
    @property
    def types(self):
        return self._types
    @property
    def dtypes(self):
        return self._dtypes
    def isinstance(self, arg):
        return isinstance(arg, self._types)
    def cast(self, arg):
        return self._types[0](arg)
    def __repr__(self):
        return "{}({})".format(
            type(self).__name__,
            self.name
        )

class ArrayType(ArgumentType):
    """
    Extends the basic `ArgumentType` spec to handle array types of possibly fixed size.
    To start, we're only adding in proper support for numpy arrays.
    Other flavors might come, but given the use case, it's unlikely.
    """
    def __init__(self, base_type, shape=None):
        self.base = base_type
        self.shape = shape

    @property
    def ctypes_type(self):
        if self._ctypes_spec is None:
            self._ctypes_spec = npctypes.ndpointer(self.base.ctypes_type, flags="C_CONTIGUOUS")
        return self._ctypes_spec
    @property
    def cpp_type(self):
        return "*"+self._ctypes_spec
    @property
    def types(self):
        return (np.ndarray,)
    @property
    def dtypes(self):
        return self.base.dtypes
    def isinstance(self, arg):
        return isinstance(arg, self.types) and arg.dtype in self.base.dtypes
    def cast(self, arg):
        return np.asarray(arg).astype(self.base.dtypes[0])
    def __repr__(self):
        return "{}({})".format(
            type(self).__name__,
            self.base
        )

class PointerType(ArgumentType):
    """
    Extends the basic `ArgumentType` spec to handle pointer types
    """
    def __init__(self, base_type):
        """
        :param base_type: The base type we're building off of
        :type base_type: ArgumentType
        """
        self.base = base_type
        self._ctypes_spec = None

    @property
    def ctypes_type(self):
        if self._ctypes_spec is None:
            self._ctypes_spec = ctypes.pointer(self.base.ctypes_type)
        return self._ctypes_spec

    @property
    def cpp_type(self):
        return "*"+self._ctypes_spec
    @property
    def types(self):
        return self.base.types
    @property
    def dtypes(self):
        return self.base.dtypes
    def isinstance(self, arg):
        return self.base.isinstance(arg)
    def cast(self, arg):
        return self.base.cast(arg)
    def __repr__(self):
        return "{}({})".format(
            type(self).__name__,
            self.base
        )

# this is the block where we just declare a shit ton of types...
# Python types that handle the most common case
RealType = PrimitiveType(
    "Real",
    ctypes.c_double,
    "double",
    "d",
    (float,),
    (np.dtype('float64'),),
    None,#serializer
    None#deserializer
)
IntegerType = PrimitiveType(
    "int",
    ctypes.c_int,
    "int",
    "i",
    (int,),
    (np.dtype('int64'),),
    None,#serializer
    None#deserializer
)
BoolType = PrimitiveType(
    "bool",
    ctypes.c_bool,
    "bool",
    "p",
    (bool,),
    (np.dtype('bool'),),
    None,#serializer
    None#deserializer
)
# C-types with the same names
FloatType = PrimitiveType(
    "float",
    ctypes.c_float,
    "float",
    "f",
    (float,),
    (np.dtype('float32'),),
    None,#serializer
    None#deserializer
)
DoubleType = PrimitiveType(
    "double",
    ctypes.c_double,
    "float",
    "d",
    (float,),
    (np.dtype('float64'),),
    None,#serializer
    None#deserializer
)
IntType = PrimitiveType(
    "int",
    ctypes.c_int,
    "int",
    "i",
    (int,),
    (np.dtype('int32'),),
    None,#serializer
    None#deserializer
)
LongType = PrimitiveType(
    "long",
    ctypes.c_int,
    "long",
    "i",
    (int,),
    (np.dtype('int32'),),
    None,#serializer
    None#deserializer
)


class Argument:
    """
    Defines a single Argument for a C-level caller to support default values, etc.
    We use a two-pronged approach where we have a set of ArgumentType serializers/deserializers
    """

    arg_types = [
        RealType,
        IntType,
        BoolType
    ]

    def __init__(self, name, dtype, default=None):
        """
        :param name: the name of the argument
        :type name: str
        :param dtype: the type of the argument; at some point we'll support type inference...
        :type dtype: ArgumentType
        :param default: the default value for the argument
        :type default:
        """
        self.name = name
        self.dtype = dtype #self.infer_type(dtype)
        self.default = default #self.prep_argument(default)

    @classmethod
    def infer_type(cls, arg):
        """
        Infers the type of an argument

        :param arg:
        :type arg: ArgumentType | str | type | ctypes type
        :return:
        :rtype:
        """
        if isinstance(arg, ArgumentType):
            return arg
        for at in cls.arg_types:
            if at.isinstance(arg):
                return at
        # TODO: throw an error for getting down here...

    @classmethod
    def infer_array_type(cls, argstr):
        ...

    @classmethod
    def inferred_type_string(cls, arg):
        """
        returns a type string for the inferred type
        """
        ...

    @property
    def cpp_signature(self):
        return "{} {}".format(
            self.dtype.cpp_type,
            self.name
        )
    def __repr__(self):
        return "{}('{}', {})".format(
            type(self).__name__,
            self.name,
            self.dtype
        )

class FunctionSignature:
    """
    Defines a function signature for a C-level caller.
    To be used inside `SharedLibraryFunction` and things to manage the core interface.
    """

    def __init__(self, name, *args, return_type=None):
        """
        :param name: the name of the function
        :type name: str
        :param args: the arguments passed to the function
        :type args: Iterable[ArgumentType]
        :param return_type: the return type of the function
        :type return_type: ArgumentType | None
        """
        self.name = name
        if return_type is not None:
            return_type = Argument.infer_type(return_type)
        self._ret_type = return_type
        self._arguments = tuple(self.build_argument(x, i) for i, x in enumerate(args))

    def build_argument(self, argtup, which=None):
        """
        Converts an argument tuple into an Argument object
        :param argtup:
        :type argtup:
        :return:
        :rtype:
        """
        if isinstance(argtup, Argument):
            return argtup
        elif isinstance(argtup, str):
            argtup = ('_argument_{}'.format(which), argtup)
        elif isinstance(argtup, dict):
            if 'default' not in argtup:
                argtup['default'] = None
            argtup = (argtup['name'], argtup['dtype'], argtup['default'])
        if len(argtup) == 2:
            name, dtype = argtup
            default = None
        else:
            name, dtype, default = argtup
        return Argument(name=name, dtype=dtype, default=default)

    @property
    def args(self):
        return self._arguments

    @property
    def return_type(self):
        return self._ret_type

    @property
    def cpp_signature(self):
        return "{} {}({})".format(
            "void" if self.return_type is None else self.return_type,
            self.name,
            ", ".join(a.cpp_signature for a in self.args)
        )
    def __repr__(self):
        return "{}({}({})->{})".format(
            type(self).__name__,
            self.name,
            ", ".join(repr(a) for a in self.args),
            "Any" if self.return_type is None else self.return_type
        )


