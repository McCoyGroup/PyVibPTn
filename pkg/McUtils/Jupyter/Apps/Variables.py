import uuid

import numpy as np, weakref
from ..JHTML import JHTML, DefaultOutputArea, JupyterAPIs

__all__ = [
    "Var",
    "InterfaceVars",
    "VariableSynchronizer",
    "VariableNamespace",
    "WidgetControl"
]

__reload_hook__ = ['..JHTML']

class SettingChecker:
    int_types = (int, np.integer)
    numerics_types = (int, np.integer, float, np.floating)
    control_type = None
    @classmethod
    def check(self, **props):
        return False
    checkers = []
class CheckboxChecker(SettingChecker):
    control_type = "Checkbox"
    @classmethod
    def check(self, value=None, **rest):
        return isinstance(value, bool)
SettingChecker.checkers.append(CheckboxChecker)
class DropdownChecker(SettingChecker):
    control_type = "Dropdown"
    @classmethod
    def check(self, options=None, **rest):
        return options is not None
SettingChecker.checkers.append(DropdownChecker)
class TextChecker(SettingChecker):
    control_type = "Text"
    @classmethod
    def check(self, value=None, **rest):
        return isinstance(value, str)
SettingChecker.checkers.append(TextChecker)
class IntSliderChecker(SettingChecker):
    control_type = "IntSlider"
    @classmethod
    def check(self, value=None, min=None, max=None, **rest):
        return isinstance(value, self.int_types) and isinstance(min, self.int_types) and isinstance(max, self.int_types)
SettingChecker.checkers.append(IntSliderChecker)
class FloatSliderChecker(SettingChecker):
    control_type = "FloatSlider"
    @classmethod
    def check(self, value=None, min=None, max=None, **rest):
        return (
                isinstance(value, self.numerics_types)
                and isinstance(min, self.numerics_types)
                and isinstance(max, self.numerics_types)
        )
SettingChecker.checkers.append(FloatSliderChecker)
class IntRangeChecker(SettingChecker):
    control_type = "IntRangeSlider"
    @classmethod
    def check(self, value=None, min=None, max=None, **rest):
        return (
                len(value) == 2
                and isinstance(value[0], self.int_types)
                and isinstance(value[1], self.int_types)
                and isinstance(min, self.int_types) and isinstance(max, self.int_types)
        )
SettingChecker.checkers.append(IntRangeChecker)
class FloatRangeChecker(SettingChecker):
    control_type = "FloatRangeSlider"
    @classmethod
    def check(self, value=None, min=None, max=None, **rest):
        return (
                len(value) == 2
                and isinstance(value[0], self.numerics_types)
                and isinstance(value[1], self.numerics_types)
                and isinstance(min, self.numerics_types) and isinstance(max, self.numerics_types)
        )
SettingChecker.checkers.append(FloatRangeChecker)

class InterfaceVars:
    _cache_stack = []
    def __init__(self, *vars):
        vars = [Var(x) for x in vars]
        self._var_set = set(vars)
        self.var_list = vars
    @classmethod
    def active_vars(cls):
        if len(cls._cache_stack) > 0:
            return cls._cache_stack[-1]
        else:
            return None
    @property
    def dict(self):
        return {v.name:v.value for v in self.var_list}
    @property
    def items(self):
        return [(v.name, v.value) for v in self.var_list]
    def __iter__(self):
        return iter(self.var_list)
    def add(self, var):
        if var not in self._var_set:
            self.var_list.append(var)
            self._var_set.add(var)
    def __enter__(self):
        self._cache_stack.append(self)
        return self.var_list
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cache_stack.pop()
class VariableNamespace:
    _namespace_cache = weakref.WeakValueDictionary()
    def __init__(self, name=None, dedupe=True):
        if name is None:
            name = uuid.uuid4()
        self.name = name
        self._old_space = None
        if dedupe:
            if name in self._namespace_cache:
                self._var_cache = self._namespace_cache[name]._var_cache
            else:
                self._var_cache = weakref.WeakValueDictionary()
                self._namespace_cache[name] = self
        else:
            self._var_cache = weakref.WeakValueDictionary()
    def __repr__(self):
        return "{}({}, {})".format(
            type(self).__name__,
            self.name,
            list(self._var_cache.values())
        )
    @classmethod
    def create(cls, name):
        if isinstance(name, VariableNamespace):
            return name
        else:
            if name not in cls._namespace_cache:
                this = VariableNamespace(name)
                cls._namespace_cache[name] = this # hold a reference...
            return cls._namespace_cache[name]
    def __contains__(self, item):
        return self._var_cache.__contains__(item)
    def __getitem__(self, item):
        return self._var_cache[item]
    def __setitem__(self, key, value):
        self._var_cache[key] = value
    def __enter__(self):
        self._old_space = VariableSynchronizer.current_namespace
        VariableSynchronizer.current_namespace = self
    def __exit__(self, exc_type, exc_val, exc_tb):
        VariableSynchronizer.current_namespace = self._old_space
        self._old_space = None
class VariableSynchronizer:
    current_namespace = VariableNamespace.create('globals')
    def __init__(self, name, namespace=None, value=None, callbacks=(), output_pane=None):
        self._name = name
        self.namespace = namespace
        self._value = value
        self.callbacks = weakref.WeakSet(callbacks)
        self.output_pane = DefaultOutputArea.get_default() if output_pane is None else output_pane
        self._watchers = weakref.WeakSet()
    def __repr__(self):
        return "{}({}, {!r})".format(
            type(self).__name__,
            self._name,
            self._value
        )
    @classmethod
    def create_var(cls, var):
        var_cache = InterfaceVars.active_vars()
        if isinstance(var, VariableSynchronizer):
            if var_cache is not None:
                var_cache.add(var)
            return var
        else:
            if var not in cls.current_namespace:
                this_var = VariableSynchronizer(var, namespace=cls.current_namespace)
                cls.current_namespace[var] = this_var
            if var_cache is not None:
                var_cache.add(cls.current_namespace[var])
            return cls.current_namespace[var]
    @property
    def name(self):
        return self._name
    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, v):
        self.set_value(v)
    def set_value(self, v, caller=None):
        with self.output_pane:
            old = self._value
            try:
                check = old != v
            except TypeError:
                check = old is not v
            if check:
                self._value = v
                for c in self.callbacks:
                    c({'var': self, 'old': old, 'new': self._value})
                for w in self._watchers:
                    if w is not caller:
                        w.value = self._value
    def link(self, widget):
        if hasattr(widget, 'value'):
            self.set_value(widget.value, caller=widget)
            widget.observe(lambda d: self.set_value(widget.value, caller=widget), names=['value'])
        self._watchers.add(widget)
def Var(name, namespace=None):
    if namespace is not None:
        with VariableNamespace.create(namespace):
            return VariableSynchronizer.create_var(name)
    else:
        return VariableSynchronizer.create_var(name)
class WidgetControl:
    def __init__(self, var, control_type=None, widget=None, **settings):
        self.var = VariableSynchronizer.create_var(var)
        self.settings = settings
        if widget is None:
            widget = self._build_widget(control_type, settings)
        self.widget = widget
    @classmethod
    def _build_widget(cls, control_type, settings):
        if control_type is None:
            for checker in SettingChecker.checkers:
                if checker.check(**settings):
                    control_type = getattr(JupyterAPIs.get_widgets_api(), checker.control_type)
                    break
            else:
                control_type = JHTML.OutputArea
        return control_type(**settings)

    def to_widget(self):
        self.var.link(self.widget)
        return self.widget