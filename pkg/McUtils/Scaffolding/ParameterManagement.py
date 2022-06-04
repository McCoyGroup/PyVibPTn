import os
from .ConfigManagement import ConfigSerializer

__all__ = [ "ParameterManager" ]

class ParameterManager:

    def __init__(self, *d, **ops):
        if len(d) > 0:
            if isinstance(d[0], dict):
                self.ops = d[0]
                self.ops.update(ops)
            else:
                self.ops = dict(d, **ops)
        else:
            self.ops = ops
    def __getattr__(self, item):
        return self.ops[item]
    def __setattr__(self, key, value):
        if key == "ops":
            super().__setattr__(key, value)
        else:
            self.ops[key] = value
    def __delattr__(self, item):
        del self.ops[item]
    def __hasattr__(self, key):
        return key in self.ops
    def update(self, **ops):
        self.ops.update(**ops)

    def keys(self):
        return self.ops.keys()
    def items(self):
        return self.ops.items()

    def save(self, file, mode=None, attribute=None):
        self.serialize(file)
    @classmethod
    def load(cls, file, mode=None, attribute=None):
        cls(cls.deserialize(file, mode=mode, attribute=attribute))

    def get_props(self, obj):
        try:
            props = obj.__props__
        except AttributeError:
            props = None

        if props is None:
            raise AttributeError("{}.{}: object {} needs an attribute {} to filter against")
        return props

    def bind(self, obj, props = None):
        if props is None:
            props = self.get_props(obj)
        for k in props:
            setattr(obj, k, self.ops[k])
    def filter(self, obj, props = None):
        if props is None:
            props = self.get_props(obj)
        new = {}
        ops = self.ops
        for k in props:
            if k in ops:
               new[k] = ops[k]
        return new

    def serialize(self, file, mode = None):
        return ConfigSerializer.serialize(file, self.ops, mode = mode)

    @classmethod
    def deserialize(cls, file, mode=None, attribute=None):
        return ConfigSerializer.deserialize(file, mode=mode, attribute=attribute)