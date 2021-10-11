"""
Misc utilities for debugging
"""

__all__ = [
    "ModificationTracker"
]

import enum, inspect
from ..Scaffolding import Logger

class ModificationType(enum.Enum):
    GetAttr="__getattr__"
    SetAttr="__setattr__"
    # need to also track the in-place modifications
    Add="__iadd__"
    Sub="__isub__"
    Div="__idiv__"
    Mul="__imul__"
    MatMul="__imatmul__"
    Any="any"

class ModificationTypeHandler(enum.Enum):
    Raise="raise"
    Log="log"

class ModificationTracker:
    """
    A simple class to wrap an object to track when it is accessed or
    modified
    """
    def __init__(self, obj,
                 handlers=ModificationTypeHandler.Log,
                 logger=None
                 ):
        self.obj = obj
        if not isinstance(handlers, dict):
            handlers = {ModificationType.Any:handlers}
        self.handlers = {
            ModificationType(k) if isinstance(k, str) else k:
                ModificationTypeHandler(h) if isinstance(h, str) else h for k, h in handlers.items()}
        self.logger = logger if logger is not None else Logger()

    @property
    def handler_dispatch(self):
        return {
            ModificationTypeHandler.Log: self.log_modification,
            ModificationTypeHandler.Raise: self.raise_modification
        }

    def log_modification(self, obj, handler_type, *args, **kwargs):
        """
        Logs on modification

        :param obj:
        :type obj:
        :param handler_type:
        :type handler_type:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        outer_frame = inspect.currentframe()
        for _ in range(5):
            outer_frame = outer_frame.f_back
        frameinfo = inspect.getframeinfo(outer_frame)

        self.logger.log_print("{handler_type} of {obj} at line {lineno} in {fname}. Args: {args} Kwargs {kwargs}",
                              handler_type=handler_type,
                              obj=obj,
                              fname=frameinfo.filename,
                              args=args,
                              kwargs=kwargs,
                              lineno=frameinfo.lineno
                              )

    def raise_modification(self, obj, handler_type, *args, **kwargs):
        """
        Raises an error on modification

        :param obj:
        :type obj:
        :param handler_type:
        :type handler_type:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        raise RuntimeError(
            "{handler_type} of {obj}. Args: {args} Kwargs {kwargs}".format(
                handler_type=handler_type,
                obj=obj,
                args=args,
                kwargs=kwargs
            )
        )

    def _dispatch_handler(self, handler_type, *args, **kwargs):
        handlers = self.handlers
        handler = None
        if handler_type in handlers:
            handler = self._get_canonical_handler(handlers[handler_type])
        elif ModificationType.Any in handlers:
            handler = self._get_canonical_handler(handlers[ModificationType.Any])

        res = True
        if handler is not None:
            res = handler(
                self.obj,
                ModificationType.GetAttr,
                *args,
                **kwargs
            )

        return res

    def __getattr__(self, item):
        """
        Handler to intercept `getattr` requests
        :param item:
        :type item:
        :return:
        :rtype:
        """
        handle_flag = self._dispatch_handler(ModificationType.GetAttr, item)
        if handle_flag:
            return getattr(self.obj, item)

    def __setattr__(self, item, val):
        """
        Handler to intercept `setattr` requests

        :param item:
        :type item:
        :param val:
        :type val:
        :return:
        :rtype:
        """
        handle_flag = self._dispatch_handler(ModificationType.SetAttr, item, val)
        if handle_flag:
            return setattr(self.obj, item, val)

    def __iadd__(self, other):
        """
        Handler to intercept `add` requests

        :param item:
        :type item:
        :param val:
        :type val:
        :return:
        :rtype:
        """
        handle_flag = self._dispatch_handler(ModificationType.Add, other)
        if handle_flag:
            return self.obj.__iadd__(other)

    def __isub__(self, other):
        """
        Handler to intercept `sub` requests

        :param item:
        :type item:
        :param val:
        :type val:
        :return:
        :rtype:
        """
        handle_flag = self._dispatch_handler(ModificationType.Sub, other)
        if handle_flag:
            return self.obj.__isub__(other)

    def __imul__(self, other):
        """
        Handler to intercept `div` requests

        :param item:
        :type item:
        :param val:
        :type val:
        :return:
        :rtype:
        """
        handle_flag = self._dispatch_handler(ModificationType.Mul, other)
        if handle_flag:
            return self.obj.__imul__(other)

    def __idiv__(self, other):
        """
        Handler to intercept `div` requests

        :param item:
        :type item:
        :param val:
        :type val:
        :return:
        :rtype:
        """
        handle_flag = self._dispatch_handler(ModificationType.Div, other)
        if handle_flag:
            return self.obj.__idiv__(other)

    def __imatmul__(self, other):
        """
        Handler to intercept `matmul` requests

        :param item:
        :type item:
        :param val:
        :type val:
        :return:
        :rtype:
        """
        handle_flag = self._dispatch_handler(ModificationType.MatMul, other)
        if handle_flag:
            return self.obj.__imatmul__(other)
