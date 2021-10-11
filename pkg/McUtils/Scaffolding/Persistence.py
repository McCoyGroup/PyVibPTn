"""
Provides utilities for managing object persistence.
Two classes of persistence are provided.
 1. Config persistence: stores objects by turning them into a
    set of config variables & provides reloading
 2. File-backed objects: stores objects by making serializing core
    pieces of the data
"""

import os, shutil, tempfile as tf, weakref

from .Checkpointing import Checkpointer, NumPyCheckpointer
from .Configurations import Config

__all__ = [
    "PersistenceLocation",
    "PersistenceManager"
]

class PersistenceLocation:
    """
    An object that tracks a location to persist data
    and whether or not that data should be cleaned up on
    exit
    """
    _cache = weakref.WeakValueDictionary()
    def __init__(self, loc, name=None, delete=None):
        if name is None:
            name = os.path.basename(loc)
        self.name = name

        if delete is None:
            delete = not os.path.isdir(loc)

        absloc = os.path.abspath(loc)
        if not os.path.isdir(absloc):
            if absloc != loc:
                loc = os.path.join(tf.TemporaryDirectory().name, loc)
        else:
            loc = absloc

        # all it takes is a single location
        # saying "don't delete" for us to not
        # delete...note that if the location
        # dies and then is reborn as a deletable
        # then it will be deleted
        for k,v in self._cache.items():
            if v.loc == loc:
                if not delete:
                    v.delete = False
                elif not v.delete:
                    delete = False

        self.loc = loc
        self.delete = delete

        self._cache[loc] = self

    def __repr__(self):
        return "{}({}, {}, delete={})".format(
            type(self).__name__,
            self.name,
            self.loc,
            self.delete
        )

    def __del__(self):
        if self.delete:
            try:
                shutil.rmtree(self.loc)
            except OSError:
                pass

class PersistenceManager:
    """
    Defines a manager that can load configuration data from a directory
    or, maybe in the future, a SQL database or similar.
    Requires class that supports `from_config` to load and `to_config` to save.
    """
    def __init__(self, cls, persistence_loc=None):
        """
        :param cls:
        :type cls: type
        :param persistence_loc: location from which to load/save objects
        :type persistence_loc: str | None
        """
        self.cls = cls
        if persistence_loc is None:
            if not hasattr(cls, 'persistence_loc'):
                raise AttributeError("{me}: to support persistence either a '{ploc}'' must be passed or {cls} needs the attribute '{ploc}'".format(
                    me=type(self).__name__,
                    cls=cls.__name__,
                    ploc=persistence_loc
                ))
            persistence_loc = cls.persistence_loc
        self.loc = persistence_loc

    def obj_loc(self, key):
        return os.path.join(self.loc, key)

    def load_config(self, key, make_new=False, init=None):
        """
        Loads the config for the persistent structure named `key`
        :param key:
        :type key:
        :return:
        :rtype:
        """
        if self.contains(key):
            return Config(self.obj_loc(key), extra_params=init)
        elif make_new:
            return self.new_config(key, init=init)
        else:
            raise KeyError("{}: no persistent object {}".format(
                type(self).__name__,
                key
            ))

    def new_config(self, key, init=None):
        """
        Creates a new space and config for the persistent structure named `key`

        :param key: name for job
        :type key: str
        :param init: initial parameters
        :type init: str | dict | None
        :return:
        :rtype:
        """
        loc = self.obj_loc(key)
        if init is None:
            init = {}
        elif isinstance(init, str):
            if os.path.isdir(init):
                init_dir = init
                init = Config(init).opt_dict
                if 'initialization_directory' not in init:
                    init['initialization_directory'] = init_dir
            else:
                init = Config(init).opt_dict

        if not os.path.isdir(loc):
            if 'initialization_directory' in init:
                shutil.copytree(
                    init['initialization_directory'],
                    loc
                )
            else:
                os.makedirs(loc, exist_ok=True)

        # now we prune, because we don't want to preserve this forever...
        if 'initialization_directory' in init:
            del init['initialization_directory']

        if Config.find_config(loc) is None:
            return Config.new(loc, init=init)
        else:
            conf = Config(loc)
            if len(init) > 0:
                conf.update(**init)
            return conf

    def contains(self, key):
        """
        Checks if `key` is a supported persistent structure

        :param key:
        :type key:
        :return:
        :rtype:
        """
        return os.path.isdir(self.obj_loc(key))

    def load(self, key, make_new=False, strict=True, init=None):
        """
        Loads the persistent structure named `key`

        :param key:
        :type key:
        :return:
        :rtype:
        """
        cfg = self.load_config(key, make_new=make_new, init=init)
        try:
            loader = self.cls.from_config
        except AttributeError:
            loader = None
        if loader is None:
            raise AttributeError("{}.{}: to support persistence {} has to have a classmethod '{}'".format(
                type(self).__name__,
                'load',
                self.cls,
                'from_config'
            ))
        return cfg.apply(loader, strict=strict)
    def save(self, obj):
        """
        Saves requisite config data for a structure

        :param obj:
        :type obj:
        :return:
        :rtype:
        """
        if not isinstance(obj, self.cls):
            raise TypeError("{}.{}: object {} isn't of persistence type {}".format(
                type(self).__name__,
                'save',
                obj,
                self.cls.__name__
            ))

        try:
            loader = obj.to_config
        except AttributeError:
            loader = None
        if loader is None:
            raise AttributeError("{}.{}: to support persistence {} has to have a classmethod '{}'".format(
                type(self).__name__,
                'save',
                self.cls,
                'to_config'
            ))
        data = loader()

        key = data['name']
        cfg = self.load_config(key, make_new=True)
        cfg.update(**data)