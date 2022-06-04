"""
Defines a common data handler
"""
import os, sys

__all__ = [ "DataHandler", "DataError", "DataRecord" ]

default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
default_data_package = "TheRealMcCoy"
default_data_key = "data"
default_data_source_key = "source"

class DataError(KeyError):
    """
    Exception subclass for data error
    """

class DataHandler:
    """
    Defines a general data loader class that we can use for `AtomData` and any other data classes we might find useful.
    """
    def __init__(self,
                 data_name,
                 data_key=None,
                 source_key=None,
                 data_dir=None,
                 data_pkg=None,
                 alternate_keys=None,
                 getter=None,
                 record_type=None
                 ):
        """
        :param data_name: the name of the dataset
        :type data_name: str
        :param data_key: the key in the loaded dictionary to use for the actual data (`"data"` by default)
        :type data_key: str | None
        :param source_key: the key in the loaded dictionary for the original data source (`"source"` by default)
        :type source_key: str | None
        :param data_dir: the main directory data will be loaded from (`.` by default)
        :type data_dir: str | None
        :param data_pkg: the python package to load (`TheRealMcCoy` by default)
        :type data_pkg: str | None
        :param alternate_keys: alternate keys that can be used to index into the dataset which can will be populated at runtime
        :type alternate_keys: Iterable[str] | None
        :param getter: a function to use to resolve a key
        :type getter: callable | None
        :param record_type: the class to use for holding data (`DataRecord` by default)
        :type record_type: type | None
        """
        if data_dir is None:
            data_dir = default_data_dir
        if data_key is None:
            data_key = default_data_key
        if source_key is None:
            source_key = default_data_source_key
        if data_pkg is None:
            data_pkg = default_data_package
        self._data = None # this'll be a dict where we store all our data
        self._src = None
        self._dir = data_dir
        self._name = data_name
        self._key = data_key
        self._src_key = source_key
        self._pkg = data_pkg
        self._alts = alternate_keys
        self.record_type = DataRecord if record_type is None else record_type
        self._loaded = False
        self.getter = getter
    @property
    def data_file(self): # in case other people want to load it...
        return os.path.join(self._dir, self._name+".py")
    def _load_alts(self):
        # assumes we have dict data, but this entire structure does that anyway
        if not self._alts is None:
            extras = {}
            if isinstance(self._alts, str):
                self._alts = (self._alts,)
            for k in self._alts:
                extras.update({a[k]:a for a in self._data.values()}) #shouldn't increase memory bc mutable
            self._data.update(extras)
    def load(self, env=None):
        """
        Actually loads the data from `data_file`.
        Currently set up to just use an `import` statement but should
        be reimplemented to use a `Deserializer` from `Scaffolding.Serializers`

        :return:
        :rtype:
        """
        # currently we only load python data
        # TODO: I should rewrite this to use a Deserializer object...
        env = {} if env is None else env
        import sys
        sys.path.insert(0, self._dir) #name needs to be unique enough...
        if self._pkg is None:
            exec_stmt = "from {0} import {1} as {1}".format(self._name, self._key)
        else:
            exec_stmt = "from {0}.{1} import {2} as {2}".format(self._pkg, self._name, self._key)
        exec(exec_stmt, env, env)
        self._data = env[self._key]
        try:
            self._src = env[self._src_key]
        except:
            pass
        self._load_alts()
        self._loaded = True
    @property
    def data(self):
        if self._data is None:
            self.load()
        return self._data
    @property
    def source(self):
        if self._src is None and not self._loaded:
            self.load()
        return self._src
    def _get_data(self, key):
        def _get(a, k):
            if k not in a:
                raise DataError("{}: data source {} doesn't have subkey {} of {}".format(
                    type(self).__name__,
                    self._name,
                    k,
                    key
                ))
            return a[k]

        data = self.data
        if self.getter is None:
            if isinstance(key, tuple):
                from functools import reduce
                return reduce(_get, key, data)
            else:
                if key not in data:
                    raise DataError("{}: data source {} doesn't have key {}".format(
                        type(self).__name__,
                        self._name,
                        key
                    ))
                return data[key]
        else:
            return self.getter(data, key)
    def __getitem__(self, key):
        data = self._get_data(key)
        return self.record_type(self, key, data)
    def __len__(self):
        return len(self.data)
    def __iter__(self):
        return iter(self.data.items())

    # implementing to make pickling of data objects possible...
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_data'] = None
        state['_src'] = None
        state['_loaded'] = False

        # raise Exception(state)
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._loaded = False

    def __repr__(self):
        return "{}('{}', file='{}')".format(
            type(self).__name__,
            self._name,
            self._src
        )

class DataRecord:
    """
    Represents an individual record that might be accessed from a `DataHandler`.
    Implements _most_ of the `dict` interface, but, to make things a bit easier when
    pickling, is not implemented as a proper subclass of `dict`.
    """
    def __init__(self, data_handler, key, records):
        self.data = records
        self.handler = data_handler
        self.key = key

    def keys(self):
        return self.data.keys()
    def values(self):
        return self.data.keys()
    def items(self):
        return self.data.keys()

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        return "{}('{}', {})".format(
            type(self).__name__,
            self.key,
            self.handler
        )

    # implementing to make pickling of data objects possible...
    def __getstate__(self):
        # it turns out we really need the dict.copy()...?
        state = self.__dict__.copy()
        del state['data']
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.data = self.handler._get_data(self.key)
