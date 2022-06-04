"""
Provides scaffolding for creating serializers that dump data to a reloadable format.
Light-weight and unsophisticated, but that's what makes this useful..
"""

import abc, numpy as np, json, io, pickle, importlib, base64, types
from collections import OrderedDict

__all__= [
    "PseudoPickler",
    "BaseSerializer",
    "JSONSerializer",
    "NumPySerializer",
    "NDarrayMarshaller",
    "HDF5Serializer",
    "YAMLSerializer",
    "ModuleSerializer"
]

class PseudoPickler:
    """
    A simple plugin to work _like_ pickle, in that it should
    hopefully support serializing arbitrary python objects, but which
    doesn't attempt to put stuff down to a single `bytearray`, instead
    supporting objects with `to_state` and `from_state` methods by converting
    them to more primitive serializble types like arrays, strings, numbers,
    etc.
    Falls back to naive pickling when necessary.
    """

    _cache = None
    def __init__(self,
                 allow_pickle=False,
                 protocol=1,
                 b64encode=False
                 ):
        self.allow_pickle=allow_pickle
        self.protocol=protocol
        self.b64encode=b64encode

    _primitive_types = (int, float, bool, str,
                        np.integer, np.floating, np.bool
                        )
    _list_types = (tuple, list)
    _dict_types = (dict, OrderedDict)
    _importable_types = (type, #types.MethodType, types.ModuleType,
                         bytes, bytearray, memoryview,
                         np.dtype
                         )
    _safe_modules = ["numpy", "multiprocessing"]
    def _to_state(self, obj, cache):
        """
        Tries to extract state for `obj` by walking through the
        object dict

        :param obj:
        :type obj:
        :param cache: cache of written objects to prevent cycles
        :type cache: set
        :return:
        :rtype:
        """

        if obj is None:
            return None
        elif isinstance(obj, self._primitive_types):
            return obj
        elif isinstance(obj, np.ndarray):
            if obj.dtype == np.dtype(object):
                objid = id(obj)
                if objid in cache:
                    raise ValueError("multiple references to single object not allowed ({} already written)".format(obj))
                cache.add(objid)
                convobj = np.array([self.serialize(v, cache) for v in obj.flatten()])
                obj = convobj.reshape(obj.shape)
                # raise ValueError("don't support arbitrary object NumPy data yet")
            return obj
        elif isinstance(obj, self._list_types):
            return type(obj)(self.serialize(v, cache) for v in obj)
        elif isinstance(obj, self._dict_types):
            return type(obj)((k, self.serialize(v, cache)) for k,v in obj.items())
        elif isinstance(obj, self._importable_types):
            return self._to_importable_state(obj)
        elif self._can_import(obj):
            return self._to_importable_state(obj)
        else:
            objid = id(obj)
            if objid in cache:
                raise ValueError("multiple references to single object not allowed ({} already written)".format(obj))
            cache.add(id(obj))
            try:
                odict = dict(obj.__dict__)
            except AttributeError:
                raise_err = True
            else:
                raise_err = False

            if raise_err:
                raise ValueError("can't psuedopickle object of type {} (not a primitive nor supporting `obj.__dict__`): {}".format(type(obj), obj))

            return self.serialize(odict, cache=cache)

    def _to_importable_state(self, obj):
        try:
            dump = pickle.dumps(obj)
        except:
            raise ValueError("unable to pickle {}; needed to be able to restore state".format(obj))
        if self.b64encode:
            dump = base64.b64encode(dump).decode('ascii')
        return {
            "pseudopickle_protocol": self.protocol,
            "pickle_data": dump
        }
    def _can_import(self, obj):
        # print(type(obj).__module__)
        return (
                type(obj).__module__.split(".")[0] in self._safe_modules
                or isinstance(obj, types.BuiltinFunctionType)
        )

    def to_state(self, obj, cache=None):
        """
        Tries to extract state from `obj`, first through its `to_state`
        interface, but that failing by recursively walking the object
        tree

        :param obj:
        :type obj:
        :return:
        :rtype:
        """

        if hasattr(obj, 'to_state'):
            try:
                return obj.to_state(serializer=self)
            except TypeError: # usually mean we're pickling the class
                pass
        try:
            return self._to_state(obj, cache)
        except ValueError:
            if self.allow_pickle:
                dump = pickle.dumps(obj)
                if self.b64encode:
                    dump = base64.b64encode(dump).decode('ascii')
                return {
                    "pseudopickle_protocol": self.protocol,
                    "pickle_data": dump
                }
            else:
                raise ValueError("couldn't serialize {}".format(obj))

    def serialize(self, obj, cache=None):
        """
        Serializes an object first by checking for a `to_state`
        method, and that missing, by converting to primitive-ish types
        in a recursive strategy if the object passes `is_simple`, otherwise
        falling back to `pickle`

        :param obj: object to be serialized
        :type obj:
        :return: spec for the pseudo-pickled data
        :rtype: dict
        """
        if obj is None:
            return obj

        reset_cache = False
        if cache is None:
            if self._cache is None:
                reset_cache = True
                cache = set()
                type(self)._cache = cache
            else:
                cache = self._cache
        state = self.to_state(obj, cache=cache)
        if isinstance(obj, self._primitive_types + self._list_types + self._dict_types + self._importable_types):
            return state

        state = {
            "pseudopickle_protocol": self.protocol,
            "class": self.to_state(type(obj)),
            "state": state
        }

        if reset_cache:
            type(self)._cache = None

        return state

    # def _from_qualname(self, spec):
    #     woof = importlib.import_module(spec['module_name'])
    #     subnames = spec['qual_name'].split('.')
    #     for a in subnames:
    #         woof = getattr(woof, a)
    #     return woof
    def deserialize(self, spec):
        """
        Deserializes from an object spec, dispatching
        to regular pickle where necessary

        :param object:
        :type object:
        :return:
        :rtype:
        """

        if isinstance(spec, dict) and 'pseudopickle_protocol' in spec:
            if 'class' in spec:
                cls = self.deserialize(spec['class'])
                if hasattr(cls, 'from_state'):
                    return cls.from_state(spec['state'], serializer=self)
                else:
                    return {"class": cls, "state": spec['state']}
            elif 'pickle_data' in spec:
                byte_stream = spec['pickle_data']
                if not isinstance(byte_stream, bytes):
                    if isinstance(byte_stream, np.ndarray):
                        byte_stream = byte_stream.tolist()
                        # corner case that I don't think should ever happen in the
                        # use cases I've worked with...
                        if isinstance(byte_stream, list):
                            byte_stream = bytes(byte_stream)
                    if isinstance(byte_stream, str):
                        byte_stream = byte_stream.encode('ascii')
                    byte_stream = base64.b64decode(byte_stream)
                return pickle.loads(byte_stream)
            else:
                raise NotImplementedError("don't know how to pseudo-unpickle from {}".format(spec))
        elif spec is None or isinstance(spec, self._primitive_types):
            return spec
        elif isinstance(spec, self._list_types):
            return type(spec)(self.deserialize(v) for v in spec)
        elif isinstance(spec, self._dict_types):
            return type(spec)((k, self.deserialize(v)) for k,v in spec.items())
        else:
            return spec

class ConvertedData:
    """
    Wrapper class for holding serialized data so we can be sure it's clean
    """
    def __init__(self, data, serializer):
        self.data = data
        self.serializer = serializer

class BaseSerializer(metaclass=abc.ABCMeta):
    """
    Serializer base class to define the interface
    """

    default_extension="" # mostly useful later

    @abc.abstractmethod
    def convert(self, data):
        """
        Converts data into a serializable format
        :param data:
        :type data:
        :return:
        :rtype:
        """
        raise NotImplementedError("BaseSerializer is an abstract class")
    @abc.abstractmethod
    def deconvert(self, data):
        """
        Converts data from a serialized format into a python format
        :param data:
        :type data:
        :return:
        :rtype:
        """
        raise NotImplementedError("BaseSerializer is an abstract class")
    @abc.abstractmethod
    def serialize(self, file, data, **kwargs):
        """
        Writes the data
        :param file:
        :type file:
        :param data:
        :type data:
        :return:
        :rtype:
        """
        raise NotImplementedError("BaseSerializer is an abstract class")
    @abc.abstractmethod
    def deserialize(self, file, **kwargs):
        """
        Loads data from a file
        :param file:
        :type file:
        :return:
        :rtype:
        """
        raise NotImplementedError("BaseSerializer is an abstract class")

class JSONSerializer(BaseSerializer):
    """
    A serializer that makes dumping data to JSON simpler
    """
    default_extension = ".json"
    class BaseEncoder(json.JSONEncoder):
        def __init__(self, *args, pseudopickler=None, allow_pickle=True, **kwargs):
            super().__init__(*args, **kwargs)
            if pseudopickler is None:
                pseudopickler = PseudoPickler(b64encode=True)
            self.allow_pickle=allow_pickle
            self.pickler = pseudopickler
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            else:
                if self.allow_pickle:
                    try:
                        stream = json.JSONEncoder.default(self, obj)
                    except TypeError:
                        stream = self.pickler.serialize(obj)

                    return stream
                else:
                    return json.JSONEncoder.default(self, obj)

    def __init__(self, encoder=None, allow_pickle=True, pseudopickler=None):
        if pseudopickler is None:
            pseudopickler = PseudoPickler(b64encode=True)
        self.pseudopickler = pseudopickler
        self.allow_pickle = allow_pickle
        if encoder is None:
            self.encoder = self.BaseEncoder(pseudopickler=pseudopickler, allow_pickle=allow_pickle)
    def convert(self, data):
        return ConvertedData(self.encoder.encode(data), self)
    def deconvert(self, data):
        return data
    def serialize(self, file, data, **kwargs):
        if not isinstance(data, ConvertedData):
            data = self.convert(data)
        file.write(data.data)
    def deserialize(self, file, key=None, **kwargs):
        dat = json.load(file)
        dat = self.deconvert(dat)
        if key is not None:
            if '/' in key:
                key = key.split("/")
                for k in key:
                    dat = dat[k]
            else:
                dat = dat[key]

        if self.allow_pickle:
            dat = self.pseudopickler.deserialize(dat)


        return dat

class YAMLSerializer(BaseSerializer):
    """
    A serializer that makes dumping data to YAML simpler.
    Doesn't support arbitrary python objects since that hasn't seemed like
    a huge need yet...
    """
    default_extension = ".yml"
    def __init__(self):
        # just checks that we do really have YAML support...
        import yaml as api
        self.api = api

    def convert(self, data):
        return ConvertedData(data, self)
    def deconvert(self, data):
        return data
    def serialize(self, file, data, **kwargs):
        if not isinstance(data, ConvertedData):
            data = self.convert(data)
        data = data.data
        self.api.dump(data, file, **kwargs)
    def deserialize(self, file, key=None, **kwargs):
        dat = self.api.unshare(file)
        dat = self.deconvert(dat)
        if key is not None:
            if '/' in key:
                key = key.split("/")
                for k in key:
                    dat = dat[k]
            else:
                return dat[key]
        else:
            return dat

class NDarrayMarshaller:
    """
    Support class for `HDF5Serializer` and other
    NumPy-friendly interfaces that marshalls data
    to/from NumPy arrays
    """

    def __init__(self,
                 base_serializer=None,
                 allow_pickle=True,
                 psuedopickler=None,
                 allow_records=False,
                 all_dicts=False,
                 converters=None
                 ):

        self.parent=base_serializer
        self.allow_pickle = allow_pickle
        if allow_pickle and psuedopickler is None:
            psuedopickler = PseudoPickler(b64encode=True)
        self._seen_cache = None
        self.psuedopickler = psuedopickler
        self.all_dicts = all_dicts
        self.allow_records = allow_records
        self._converter_dispatch = converters

    # we define a converter layer that will coerce everything to NumPy arrays
    atomic_types = PseudoPickler._primitive_types

    @classmethod
    def get_default_converters(self):
        return OrderedDict((
            ((np.ndarray,), lambda data, cls: cls._iterable_to_numpy(data)),
            ('to_state', lambda x, s: s._psuedo_pickle_to_numpy(x)),
            ('asarray', lambda data, cls: cls._iterable_to_numpy(data.asarray())),
            ((type(None),), lambda x, cls: cls._none_to_none(x)),
            (self.atomic_types, lambda x, cls: cls._literal_to_numpy(x)),
            ((dict,), lambda data, cls: cls._dict_to_numpy(data)),
            ((list, tuple), lambda data, cls: cls._iterable_to_numpy(data))
        ))

    @property
    def converter_dispatch(self):
        if self._converter_dispatch is None:
            return self.get_default_converters()
        else:
            return self._converter_dispatch

    @classmethod
    def _literal_to_numpy(cls, data):
        return np.array([data]).reshape(())

    def _dict_to_numpy(self, data):
        return {k: self.convert(v) for k, v in data.items()}

    def _none_to_none(self, data):
        return None

    def _prep_iterable(self, data):
        if isinstance(data, np.ndarray):
            arr = data
        else:
            try:
                x0 = data[0]
            except IndexError:
                arr = np.asarray(data)
            else:
                try_numpy = True
                if isinstance(x0, np.ndarray):
                    l = x0.shape
                else:
                    try:
                        l = len(x0)
                    except TypeError:
                        try_numpy = False
                if try_numpy:
                    for x in data:
                        try:
                            iter(x)
                            if isinstance(x0, np.ndarray):
                                try_numpy = x.shape == l
                            else:
                                try_numpy = len(x) == l
                        except (TypeError, AttributeError):
                            try_numpy = False
                        if not try_numpy: break

                if try_numpy:
                    np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
                    try:
                        arr = np.array(data)
                    except np.VisibleDeprecationWarning:
                        arr = np.empty(shape=(len(data),), dtype=object)
                        for i, v in enumerate(data):
                            arr[i] = v
                else:
                    arr = np.empty(shape=(len(data),), dtype=object)
                    for i, v in enumerate(data):
                        arr[i] = v

        return arr

    def _iterable_to_numpy(self, data):
        # we do an initial pass to check if data is a numpy array
        # or if it needs to be/can conceivably be converted

        arr = self._prep_iterable(data)

        if arr.dtype == np.dtype(object):
            # map iterable into a stack of datasets :|
            records = [self.convert(v) for i, v in enumerate(data)]
            if self.allow_records and all(isinstance(v, np.ndarray) for v in records):
                # create record dtype...
                ds_dtype = [
                    ('item_{}'.format(i), v.dtype, v.shape if v.shape != (0,) else ())
                    for i, v in enumerate(records)
                ]
                ds_arr = np.recarray((1,), dtype=ds_dtype)
                for i, v in enumerate(records):
                    key = 'item_{}'.format(i)
                    if v.shape != (0,):
                        ds_arr[key] = v
                return ds_arr
            elif self.all_dicts:
                return dict(
                    {
                        '_list_item_' + str(i): v
                        for i, v in enumerate(records)
                    },
                    _list_numitems=self.convert(len(records))
                )
            else:
                return records
        else:
            return arr

    class _pickle_cache:
        def __init__(self, parent):
            self.parent = parent
            self.tree = []
            self.parents = set()
        def __call__(self, key):
            self.add(key)
            return self
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.pop()
            if len(self.parents) == 0:
                self.parent._seen_cache = None
        def add(self, key):
            if id(key) in self.parents:
                raise RecursionError("conversion on object of type {} hit an infinite recusion: {}".format(type(key), key))
            self.parents.add(id(key))
            self.tree.append(id(key))
        def pop(self):
            self.parents.remove(self.tree.pop())
    def _psuedo_pickle_to_numpy(self, data):
        if self._seen_cache is None:
            self._seen_cache = self._pickle_cache(self)
        with self._seen_cache(data):
            data = self.psuedopickler.serialize(data)
        # return self._convert(data, allow_pickle=False)
        return self.convert(data)

    def convert(self, data, allow_pickle=None):
        """
        Recursively loop through, test data, make sure HDF5 compatible
        :param data:
        :type data:
        :return:
        :rtype:
        """

        if allow_pickle is None:
            allow_pickle = self.allow_pickle
        cur_pickle = self.allow_pickle
        try:
            self.allow_pickle = allow_pickle
            converter = None
            for k, f in self.converter_dispatch.items():
                if isinstance(k, tuple):  # check if we're dispatching based on type
                    if isinstance(data, k):
                        converter = f
                        break
                elif isinstance(k, str):  # check if we're duck typing based on attributes
                    if hasattr(data, k):
                        converter = f
                        break
                elif k(data):  # assume dispatch key is a callable that tells us if data is compatible
                    converter = f
                    break

            if converter is None and allow_pickle:
                converter = self._default_convert

            if converter is None:
                raise TypeError("no registered converter to coerce {} into HDF5 compatible format".format(data))

            woof = converter(data, self)
            return woof

        finally:
            self.allow_pickle = cur_pickle

    @staticmethod
    def _default_convert(x, converter):
        return converter._psuedo_pickle_to_numpy(x)

    def deconvert(self, data):
        """
        Reverses the conversion process
        used to marshall the data

        :param data:
        :type data:
        :return:
        :rtype:
        """

        # we loop through the keys and recursively build up a dict
        if hasattr(data, 'items'):
            res = {}
            for k, v in data.items():
                if self.parent is not None:
                    v = self.parent.deconvert(v)
                res[k] = v
            if '_list_numitems' in res:
                # actually an iterable but with inconsistent shape
                n_items = res['_list_numitems']
                res = [res['_list_item_' + str(i)] for i in range(n_items)]
            elif list(res.keys()) == ['_data']:  # special case for if we just saved a single array to file
                res = res['_data']
        elif not isinstance(data, np.ndarray):
            try:
                iter(data)
            except TypeError:
                res = data
            else:
                res = [self.parent.deconvert(v) if self.parent is not None else v for v in data]
        else:
            res = data

        # for primitive data...
        if isinstance(res, np.ndarray) and res.shape == ():
            dtype_name = str(res.dtype)
            if '|S' in dtype_name:
                # convert back from bytes to unicode...
                try:
                    res = res.astype(dtype=dtype_name.replace('|S', '<U'))
                except:
                    pass
            res = res.tolist()

        if self.allow_pickle:
            res = self.psuedopickler.deserialize(res)

        return res

    def __call__(self, data, allow_pickle=None):
        if allow_pickle is None:
            allow_pickle = self.allow_pickle
        return self.convert(data, allow_pickle=allow_pickle)

class HDF5Serializer(BaseSerializer):
    """
    Defines a serializer that can prep/dump python data to HDF5.
    To minimize complexity, we always use NumPy & Pseudopickle as an interface layer.
    This restricts what we can serialize, but generally in insignificant ways.
    """
    default_extension = ".hdf5"
    def __init__(self, allow_pickle=True, psuedopickler=None, converters=None):
        import h5py as api
        self.api = api
        self.allow_pickle = allow_pickle
        if allow_pickle and psuedopickler is None:
            psuedopickler = PseudoPickler(b64encode=True)
        self.psuedopickler = psuedopickler
        self.marshaller = NDarrayMarshaller(
            self,
            allow_pickle=allow_pickle,
            psuedopickler=psuedopickler,
            all_dicts=True,
            converters=converters
        )

    def convert(self, data):
        """
        Converts data into format that can be serialized easily

        :param data:
        :type data:
        :return:
        :rtype:
        """
        return ConvertedData(self.marshaller(data), self)

    def _create_dataset(self, h5_obj, key, data):
        """
        Mostly exists to be overridden
        :param h5_obj:
        :type h5_obj:
        :param key:
        :type key:
        :param data:
        :type data:
        :return:
        :rtype:
        """

        if data is None:
            return h5_obj.create_dataset(key, data=self.api.Empty("i"))
        else:
            # try:
            return h5_obj.create_dataset(key, data=data)
            # except:
            #     raise Exception(data.dtype, data)

    def _validate_datatype(self, data):
        return (
            data is None
            or isinstance(data, np.ndarray)
        )
    def _prune_existing(self, h5_obj, key):
        try:
            del h5_obj[key]
        except OSError:
            raise IOError("failed to remove key {} from {}".format(key, h5_obj))
    def _destroy_and_add(self, h5_obj, key, data):
        self._prune_existing(h5_obj, key)
        try:
            ds = self._create_dataset(h5_obj, key, data)
        except TypeError:
            raise TypeError("can't coerce {}:{} to HDF5 format".format(key, data))
    def _write_data(self, h5_obj, key, data):
        """
        Writes a numpy array into a group
        :param h5_group:
        :type h5_group: h5py.Group
        :param key:
        :type key: str
        :param data:
        :type data: np.ndarray
        :return:
        :rtype:
        """
        if not self._validate_datatype(data):
            raise TypeError('trying to write non-numpy data {} to key "{}"'.format(data, key))

        if data is not None:
            # clean up string datatypes
            dtype_name = str(data.dtype)
            if '<U' in dtype_name:
                # If so, convert the array to one with bytes
                data = data.astype(dtype=dtype_name.replace('<U', '|S'))

        try:
            ds = h5_obj[key] #type: h5py.Dataset
        except KeyError as e:
            if e.args[0] in {
                "Unable to open object (bad flag combination for message)",
                "Unable to open object (message type not found)",
            }:
                raise KeyError("failed to load key {} in {}".format(key, h5_obj))
            else:
                if isinstance(h5_obj, self.api.Dataset):
                    h5_obj = h5_obj.parent
                # self._destroy_and_add(h5_obj, key, data)
                try:
                    ds = self._create_dataset(h5_obj, key, data)
                except TypeError:
                    raise TypeError("can't coerce {}:{} to HDF5 format".format(key, data))
        except ValueError:
            if isinstance(h5_obj, self.api.Dataset):
                h5_obj = h5_obj.parent
            self._destroy_and_add(h5_obj, key, data)
            # self._prune_existing(h5_obj, key)
            # try:
            #     ds = self._create_dataset(h5_obj, key, data)
            # except TypeError:
            #     raise TypeError("can't coerce {}:{} to HDF5 format".format(key, data))
        else:
            dt = ds.dtype
            if (
                    data is None
                    or dt != data.dtype and dt.names is not None
            ): # record arrays are a pain
                if data is None:
                    data = self.api.Empty("i")
                self._destroy_and_add(h5_obj, key, data)
            else:
                try:
                    ds[...] = data
                except TypeError:
                    self._destroy_and_add(h5_obj, key, data)
                except:
                    raise IOError("failed to write key '{}' to HDF5 dataset {}".format(key, ds))
        # no need to return stuff, since we're just serializing
    def _write_dict(self, h5_obj, data):
        """
        Writes a dict into a group
        :param h5_group:
        :type h5_group: h5py.Group
        :param data:
        :type data: dict
        :return:
        :rtype:
        """
        for k,v in data.items():
            # print(h5_obj, k)
            # we want to either make a new Group or write the array to the key
            if isinstance(v, dict):
                try:
                    new_grp = h5_obj[k]
                except KeyError:
                    new_grp = h5_obj.create_group(k)
                self._write_dict(new_grp, v)
                # print(new_grp)
            else:
                self._write_data(h5_obj, k, v)

    def serialize(self, file, data, **kwargs):
        if not isinstance(data, ConvertedData):
            data = self.convert(data)
        if not isinstance(file, (self.api.File, self.api.Group)):
            file = self.api.File(file, "a")
        data = data.data
        if isinstance(data, np.ndarray):
            key = "_data"
            self._write_data(file, key, data)
        else:
            self._write_dict(file, data)

    def deconvert(self, data):
        """
        Converts an HDF5 Dataset into a NumPy array or Group into a dict
        :param data:
        :type data:
        :return:
        :rtype:
        """
        if isinstance(data, self.api.Dataset):
            if data.dtype == self.api.Empty("i"):
                res = None
            else:
                res = np.empty(data.shape, dtype=data.dtype)
                if data.shape != (0,):
                    data.read_direct(res)
                    names = res.dtype.names
                    if names is not None and 'item_0' in names:
                        res = [ res['item_'+str(i)][0] for i in range(len(names))]

            res = self.marshaller.deconvert(res)
        else:
            res = self.marshaller.deconvert(data)

        return res

    def deserialize(self, file, key=None, **kwargs):
        if not isinstance(file, (self.api.File, self.api.Group)):
            file = self.api.File(file, "r")
        if key is not None:
            file = file[key]
        return self.deconvert(file)

class NumPySerializer(BaseSerializer):
    """
    A serializer that implements NPZ dumps
    """

    default_extension = ".npz"

    # we define a converter layer that will coerce everything to NumPy arrays
    atomic_types = (str, int, float)
    converter_dispatch = None
    @classmethod
    def get_default_converters(self):
        return OrderedDict((
        ((np.ndarray,), lambda data, cls: data),
        ('asarray', lambda data, cls: data.asarray()),
        (self.atomic_types, lambda x, cls: cls._literal_to_numpy(x)),
        ((dict,), lambda data, cls: cls._dict_to_numpy(data)),
        ((list, tuple), lambda data, cls: cls._iterable_to_numpy(data))
    ))

    @classmethod
    def get_converters(self):
        if self.converter_dispatch is None:
            return self.get_default_converters()
        else:
            return self.converter_dispatch


    @classmethod
    def _literal_to_numpy(cls, data):
        return np.array([data]).reshape(())

    @classmethod
    def _dict_to_numpy(cls, data):
        return {k: cls._convert(v) for k, v in data.items()}

    @classmethod
    def _iterable_to_numpy(cls, data):
        arr = np.array(data)
        if arr.dtype == np.dtype(object):
            # map iterable into a stack of datasets :|
            return dict({'_list_item_' + str(i): cls._convert(v) for i, v in enumerate(data)},
                        _list_numitems=cls._convert(len(data)))
        else:
            return arr

    @classmethod
    def _convert(cls, data):
        """
        Recursively loop through, test data, make sure NumPy compatible
        :param data:
        :type data:
        :return:
        :rtype:
        """
        converter = None
        for k, f in cls.get_converters().items():
            if isinstance(k, tuple):  # check if we're dispatching based on type
                if isinstance(data, k):
                    converter = f
                    break
            elif isinstance(k, str):  # check if we're duck typing based on attributes
                if hasattr(data, k):
                    converter = f
                    break
            elif k(data):  # assume dispatch key is a callable that tells us if data is compatible
                converter = f
                break

        if converter is None:
            raise TypeError("no registered converter to coerce {} into HDF5 compatible format".format(data))

        return converter(data, cls)

    dict_key_sep="::>|<::" # we pick a somewhat goofy separator to avoid conflicts in most cases
    def _flatten_dict(self, d, sep=None):
        """
        :param d:
        :type d: dict
        :param sep:
        :type sep: str | None
        :return:
        :rtype:
        """
        if sep is None:
            sep = self.dict_key_sep
        new = {}
        for k, v in d.items():
            if isinstance(v, dict):
                d2 = self._flatten_dict(v, sep=sep)
                new.update({k+sep+k2:v2 for k2,v2 in d2.items()})
            else:
                new[k] = v
        return new

    def convert(self, data):
        first_pass = self._convert(data)
        if isinstance(first_pass, dict):
            # we need to flatten out nested dictionaries
            # we're hoping that we won't have massively nested structures
            # since those will totally wreck performance
            data = self._flatten_dict(first_pass)
        else:
            data = first_pass
        return ConvertedData(data, self)

    def _deconvert_val(self, data):
        if isinstance(data, dict):
            # we check to make sure we don't have an implicitly encoded mixed-type list
            if '_list_numitems' in data:
                # actually an iterable but with inconsistent shape
                n_items = data['_list_numitems']
                res = [ data['_list_item_' + str(i)] for i in range(n_items) ]
            else:
                res = {k: self._deconvert_val(v) for k,v in data.items()}
        else:
            res = data
        return res

    def deconvert(self, data, sep=None):
        """
        Unflattens nested dictionary structures so that the original data
        can be recovered
        :param data:
        :type data:
        :param sep:
        :type sep: str | None
        :return:
        :rtype:
        """
        if sep is None:
            sep = self.dict_key_sep
        if isinstance(data, np.ndarray):
            return data
        else:
            data = dict(data.items())
            # now we have to _unflatten_ the flattened dicts -_-
            keys = list(data.keys())
            remapping = [(k.split(sep), k) for k in keys]
            new_data = {}
            for k_list, main_key in remapping:
                where_do_i_go = new_data
                for k in k_list[:-1]:
                    if k not in where_do_i_go:
                        where_do_i_go[k] = {}
                        where_do_i_go = where_do_i_go[k]
                    else:
                        where_do_i_go = where_do_i_go[k]
                where_do_i_go[k_list[-1]] = data[main_key]
            return self._deconvert_val(new_data)

    def serialize(self, file, data, **kwargs):
        if not isinstance(data, ConvertedData):
            data = self.convert(data)
        data = data.data
        if isinstance(data, np.ndarray):
            return np.save(file, data)
        else:
            return np.savez(file, **data)
    def deserialize(self, file, key=None, **kwargs):
        dat = np.load(file)
        dat = self.deconvert(dat)
        if isinstance(dat, np.ndarray):
            return dat
        if key is not None:
            if '/' in key:
                key = key.split("/")
                for k in key:
                    dat = dat[k]
            else:
                return dat[key]
        else:
            return dat

class ModuleSerializer(BaseSerializer):
    """
    A somewhat hacky serializer that supports module-based serialization.
    Writes all module parameters to a dict with a given attribute.
    Serialization doesn't support loading arbitrary python code, but deserialization does.
    Use at your own risk.
    """

    default_extension = ".py"

    default_loader = None
    default_attr = "config"
    def __init__(self, attr=None, loader=None):
        self._loader =loader
        self._attr = attr

    @property
    def loader(self):
        if self._loader is None:
            if self.default_loader is None:
                self.default_loader = self._get_loader()
            return self.default_loader
        else:
            return self._loader
    @property
    def attr(self):
        if self._attr is None:
            return self.default_attr
        else:
            return self._attr
    @classmethod
    def _get_loader(cls):
        from McUtils.Extensions import ModuleLoader
        return ModuleLoader(rootpkg="Configs")

    def convert(self, data):
        return ConvertedData(data, self)
    def deconvert(self, data):
        return data
    def serialize(self, file, data, **kwargs):
        if not isinstance(data, ConvertedData):
            data = self.convert(data)
        data = data.data
        jsoner = JSONSerializer()
        with io.StringIO() as fake:
            jsoner.serialize(fake, data)
            serialized = fake.getvalue()
        print(
            "{} = {}".format(self.attr, serialized),
            file=file
        )
    def deserialize(self, file, key=None, **kwargs):
        module = self.loader.load(file)
        dat = self.deconvert(getattr(module, self.attr))
        if key is not None:
            if '/' in key:
                key = key.split("/")
                for k in key:
                    dat = dat[k]
            else:
                return dat[key]
        else:
            return dat