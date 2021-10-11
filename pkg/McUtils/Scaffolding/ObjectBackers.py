"""
Provides interfaces that _support_ objects
and make it easier to build more reliable,
higher-functioning classes
"""

import abc, os

from .Persistence import PersistenceLocation
from .Checkpointing import NumPyCheckpointer

__all__ = ["BaseObjectManager", "FileBackedObjectManager"]

class BaseObjectManager(metaclass=abc.ABCMeta):
    """
    Defines the basic parameters of an object interface
    that can handle marshalling the core data behind
    and object attribute to disk or vice versa
    """

    def __init__(self, obj):
        self.obj = obj
        self._base_name = None

    def get_basename(self):
        if hasattr(self.obj, 'serialization_id'):
            obj_id = self.obj.serialization_id
        else:
            obj_id = id(self.obj)
        return "{}_{}".format(type(self.obj).__name__, obj_id)

    @property
    def basename(self):
        if self._base_name is None:
            self._base_name = self.get_basename()
        return self._base_name

    @abc.abstractmethod
    def save_attr(self, attr):
        """
        Saves some attribute of the object

        :param attr:
        :type attr:
        :return:
        :rtype:
        """
        raise NotImplementedError("abstract interface")

    @abc.abstractmethod
    def load_attr(self, attr):
        """
        Loads some attribute of the object

        :param attr:
        :type attr:
        :return:
        :rtype:
        """
        raise NotImplementedError("abstract interface")

    @abc.abstractmethod
    def del_attr(self, attr):
        """
        Deletes some attribute of the object

        :param attr:
        :type attr:
        :return:
        :rtype:
        """
        raise NotImplementedError("abstract interface")

class FileBackedObjectManager(BaseObjectManager):
    """
    Provides an interface to back an object with
    a serializer
    """

    default_directory=PersistenceLocation("file_backed_objects")
    def __init__(self,
                 obj,
                 chk=None,
                 loc=None,
                 checkpoint_class=NumPyCheckpointer
                 ):
        """
        :param obj: the object to back
        :type obj: object
        :param chk: a checkpointer to manage storing attributes
        :type chk: Checkpointer
        :param loc: the location where attributes should be stored
        :type loc: str
        :param checkpoint_class: a subclass of Checkpointer that implements the actual writing to disk
        :type checkpoint_class: Type[Checkpointer]
        """
        super().__init__(obj)
        if chk is None:
            if loc is None:
                loc = self.default_directory.loc
            obj_file = os.path.join(loc, self.basename+checkpoint_class.default_extension)
            chk = checkpoint_class(obj_file)

        self.chk = chk
        self._id = None
        self._cache = {}

    @property
    def basename(self):
        if self._tag is None:
            self._tag = self.get_basename()
        return self.basename
    @basename.setter
    def basename(self, v):
        self._tag = v

    def get_basename(self):
        if hasattr(self.obj, 'serialization_id'):
            obj_id = self.obj.serialization_id
        else:
            obj_id = id(self.obj)
        return "{}_{}".format(type(self.obj).__name__, obj_id)

    def save_attr(self, attr):
        with self.chk:
            self.chk[attr] = getattr(self.obj, attr)
        return FileBackedAttribute(self, attr)

    def load_attr(self, attr):
        with self.chk:
            return self.chk[attr]

class FileBackedAttribute:
    """
    A helper class to make it very clear that
    an attribute is backed by a file on disk
    """

    def __init__(self, manager, attr):
        self.manager = manager
        self.attr = attr