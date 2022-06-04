"""
Basic layer for Schema validation that provides a superset of JSON schema validation
"""

__all__ = ["Schema"]

class Schema:
    """
    An object that represents a schema that can be used to test
    if an object matches that schema or not
    """

    def __init__(self, schema, optional_schema=None):
        self.schema = self.canonicalize_schema(schema)
        self.sub_schema = self.canonicalize_schema(optional_schema)

    @classmethod
    def canonicalize_schema(cls, schema):
        if schema is None:
            return None
        try:
            k0 = schema[0] # list-like duck typing
        except TypeError:
            pass
        else:
            schema = {k:None for k in schema}
        return schema

    def _validate_entry(self, obj, k, v, prop_getter, throw=False):
        match = True
        missing = False
        mistyped = False
        try:
            t = prop_getter(k)
        except (KeyError, AttributeError):
            missing = True
            match = False
            t = None
        except (TypeError,):
            mistyped = True
            match = False
            t = None
        else:
            if v is None:
                pass
            elif isinstance(v, (type, tuple)):
                match = isinstance(t, v)
            elif isinstance(v, Schema):
                match = v.validate(t, throw=throw)
            else:
                match = v(t)

        if not match and throw:
            if missing:
                raise KeyError("object {} doesn't match schema {}; key {} is missing".format(
                    obj, self, k
                ))
            elif mistyped:
                raise KeyError("object {} doesn't match schema {}; doesn't support attributes".format(
                    obj, self
                ))
            else:
                raise KeyError("object {} doesn't match schema {}; value {} doesn't match schema type {}".format(
                    obj, self, t, v
                ))
        return (t, match)

    def validate(self, obj, throw=True):
        """
        Validates that `obj` matches the provided schema
        and throws an error if not

        :param obj:
        :type obj:
        :param throw:
        :type throw:
        :return:
        :rtype:
        """

        prop_getter = ( lambda k,obj=obj:getattr(obj, k) ) if not hasattr(obj, '__getitem__') else obj.__getitem__
        for k,v in self.schema.items():
            if not self._validate_entry(obj, k, v, prop_getter, throw=throw)[1]:
                return False
        else:
            return True

    def to_dict(self, obj, throw=True):
        """
        Converts `obj` into a plain `dict` representation

        :param obj:
        :type obj:
        :return:
        :rtype:
        """

        res = {}
        prop_getter = ( lambda k,obj=obj:getattr(obj, k) ) if not hasattr(obj, '__getitem__') else obj.__getitem__
        for k, v in self.schema.items():
            t, m = self._validate_entry(obj, k, v, prop_getter, throw=throw)
            if not m:
                return None
            else:
                res[k] = t
        if self.sub_schema is not None:
            for k, v in self.sub_schema.items():
                t, m = self._validate_entry(obj, k, v, prop_getter, throw=False)
                if m:
                    res[k] = t
        return res

    def __repr__(self):
        return "{}({}, {})".format(type(self).__name__, self.schema, self.sub_schema)