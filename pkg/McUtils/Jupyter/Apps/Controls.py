
import abc, uuid, numpy as np
from ..JHTML import JHTML

from .Interfaces import Component, ListGroup, Dropdown, Progress
from .Variables import Var, InterfaceVars

__all__ = [
    "Control",
    "InputField",
    "StringField",
    "Slider",
    "Checkbox",
    "RadioButton",
    "Switch",
    "TextArea",
    "Selector",
    "VariableDisplay",
    "FunctionDisplay",
    # "DropdownMenu",
    "MenuSelect",
    "DropdownSelect",
    "ProgressBar"
]

__reload_hook__ = ['.Interfaces', '.Variables']

#region JHTML Controls
class Control(Component):
    layout_orientation = 'row'
    def __init__(self, var):
        self.var = Var(var)
        self._widget_cache = None
        super().__init__()
    def to_widget(self, parent=None):
        needs_link = self._widget_cache is None
        widg = super().to_widget(parent=parent)
        if needs_link:
            val = self.var.value
            self.var.link(self)
            self.var.value = val
            self.set_value()
        return widg
    @abc.abstractmethod
    def set_value(self):
        ...
    @abc.abstractmethod
    def get_value(self):
        ...
    @property
    def value(self):
        return self.get_value()
    @value.setter
    def value(self, v):
        self.var.set_value(v, caller=self)
        self.set_value()
    def observe(self, fn, names=None):
        if self._widget_cache is None:
            raise ValueError("not initialized")
        return self._widget_cache.to_widget().observe(fn, names=names)

    control_types = {}
    @classmethod
    def construct(cls, var, control_type=None, field_type=None, value=None, **attrs):
        if control_type is None:
            control_type = cls.infer_control(field_type=field_type, value=value, **attrs)
        if isinstance(control_type, str):
            control_type = cls.control_types[control_type]
        if value is not None:
            value = str(value)
        return control_type(var, value=value, **attrs)
    @classmethod
    def infer_control(cls, field_type=None, value=None, **ignored):
        if field_type is None and value is not None:
            field_type = type(value)
        if field_type is not None:
            if issubclass(field_type, str):
                return cls.control_types['StringField']
            elif issubclass(field_type, (bool,)):
                return cls.control_types['Checkbox']
            elif issubclass(field_type, (int, np.integer, float, np.floating)):
                return cls.control_types['Slider']
            else:
                raise NotImplementedError("can't infer control type for 'field_type' {}".format(field_type))
        else:
            raise NotImplementedError("can't infer control type without 'field_type'")

class ValueWidget(Control):
    def __init__(self, var, value=None):
        super().__init__(var)
        if value is not None:
            self.var.value = value
        if self.var.value is None:
            self.var.value = ""
    def get_value(self):
        if self._widget_cache is not None:
            val = self._widget_cache.value
            return "" if val is None else val
        else:
            return self.var.value
    def set_value(self):
        if self._widget_cache is not None:
            self._widget_cache.value = self.var.value
    def update(self, e):
        if self._widget_cache is not None:
            self.var.value = self._widget_cache.value
class InputField(ValueWidget):
    base_cls = ['form-control']
    def __init__(self, var, value=None, tag='input', track_value=True, continuous_update=False, base_cls=None, cls=None, **attrs):
        super().__init__(var, value=value)
        if base_cls is not None:
            self.base_cls = base_cls
        attrs['cls'] = self.base_cls + JHTML.manage_class(cls)
        attrs['tag'] = tag
        attrs['track_value'] = track_value
        attrs['continuous_update'] = continuous_update
        self.attrs = attrs
    def to_jhtml(self):
        # value = self.var.value
        # if value is not None and not isinstance(value, str):
        #     value = str(value)
        # self._attrs['value'] = value
        field = JHTML.Input(**self.attrs)
        return field
class StringField(InputField):
    # base_cls = ['form-text']
    def __init__(self, var, type='text', **attrs):
        super().__init__(var, type=type, **attrs)
Control.control_types['StringField'] = StringField
class Slider(InputField):
    base_cls = ['form-range']
    def __init__(self, var, type='range', value=None, **attrs):
        if value is not None and not isinstance(value, str):
            value = str(value)
        super().__init__(var, type=type, value=value, **attrs)
    def get_value(self):
        if self._widget_cache is not None:
            val = self._widget_cache.value
            try:
                val = int(val)
            except (ValueError, TypeError):
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    ...
            return val
    def set_value(self):
        if self._widget_cache is not None:
            v = self.var.value
            if not isinstance(v, str):
                v = str(v)
            self._widget_cache.value = v
Control.control_types['Slider'] = Slider
class Checkbox(InputField):
    base_cls = ['form-check-input']
    def __init__(self, var, type='checkbox', **attrs):
        super().__init__(var, type=type, **attrs)
    def get_value(self):
        if self._widget_cache is not None:
            val = self._widget_cache.value
            return isinstance(val, str) and val == "true"
    def set_value(self):
        if self._widget_cache is not None:
            if self.var.value:
                self._widget_cache.value = 'true'
            else:
                self._widget_cache.value = 'false'
Control.control_types['Checkbox'] = Checkbox
class RadioButton(Checkbox):
    base_cls = ['form-check-input']
    def __init__(self, var, type='radio', **attrs):
        super().__init__(var, type=type, **attrs)
Control.control_types['RadioButton'] = RadioButton
class Switch(Checkbox):
    base_cls = ['form-check-input']
    def __init__(self, var, type='checkbox', role='switch', **attrs):
        super().__init__(var, type=type, role=role, **attrs)
    def get_value(self):
        if self._widget_cache is not None:
            val = self._widget_cache.get_child(0).value
            return isinstance(val, str) and val == "true"
    def set_value(self):
        if self._widget_cache is not None:
            if self.var.value:
                self._widget_cache.get_child(0).value = 'true'
            else:
                self._widget_cache.get_child(0).value = 'false'
    def to_jhtml(self):
        return JHTML.Div(super().to_jhtml(), cls=['form-switch'])
Control.control_types['Switch'] = Switch
class TextArea(InputField):
    base_cls = ['form-control']
    def __init__(self, var, tag='textarea', **attrs):
        super().__init__(var, tag=tag, **attrs)
    def to_jhtml(self):
        field = JHTML.Textarea(**self.attrs)
        return field
Control.control_types['TextArea'] = TextArea
class ChangeTracker(ValueWidget):
    base = None
    base_cls = []
    def __init__(self, var, base=None, value=None, track_value=True, continuous_update=False, base_cls=None, cls=None, **attrs):
        super().__init__(var, value=value)
        base = self.base if base is None else base
        self.base = getattr(JHTML, base) if isinstance(base, str) else base
        if base_cls is not None:
            self.base_cls = base_cls
        attrs['cls'] = self.base_cls + JHTML.manage_class(cls)
        attrs['track_value'] = track_value
        attrs['continuous_update'] = continuous_update
        self.attrs = attrs
class Selector(ChangeTracker):
    base=JHTML.Select
    base_cls = ['form-select']
    def __init__(self, var, options=None, value=None, multiple=False, **attrs):
        self._options = self.canonicalize_options(options)
        if not multiple and value is None and len(self._options) > 0:
            value = self._options[0][1]
        if multiple:
            attrs['multiple']=True
        super().__init__(var, value=value, **attrs)
    @property
    def multiple(self):
        if self._widget_cache is not None:
            try:
                mult = self._widget_cache['multiple']
            except KeyError:
                mult = False
            return mult is True or mult == 'true'
        else:
            return 'multiple' in self.attrs and self.attrs['multiple']
    def get_value(self):
        if self._widget_cache is not None:
            val = super().get_value()
            if self.multiple:
                if val is None or val == "":
                    val = []
                else:
                    val = val.split("&&")
            elif val == "":
                val = None
            return val
    def set_value(self):
        if self._widget_cache is not None:
            if self.multiple:
                v = self.var.value
                if not isinstance(v, str):
                    v = "&&".join(v)
                self._widget_cache.value = v
            else:
                super().set_value()
    @classmethod
    def canonicalize_options(cls, options):
        ops = []
        for k in options:
            if isinstance(k, str):
                v = k
            else:
                try:
                    k, v = k
                except ValueError:
                    v = k
            ops.append((k,v))
        return tuple(ops)
    def _build_options_list(self):
        if self.multiple:
            vals = self.var.value
            if vals is None:
                vals = []
            elif isinstance(vals, str):
                vals = vals.split("&&")
            opts = [JHTML.Option(k, value=v, selected="true") if v in vals else JHTML.Option(k, value=v) for k,v in self._options]
        else:
            val = self.var.value
            opts = [JHTML.Option(k, value=v, selected=(val is not None and v == val)) for k,v in self._options]
        return opts
    def to_jhtml(self):
        field = self.base(*self._build_options_list(), **self.attrs)
        return field
Control.control_types['Selector'] = Selector
class VariableDisplay(Control):
    def __init__(self, var, value=None, pane=None, autoclear=True, **attrs):
        super().__init__(var)
        if pane is None:
            pane = JHTML.OutputArea(autoclear=autoclear, **attrs)
        self.out = pane
        if value is not None:
            self.var.value = value
    def get_value(self):
        return self.var.value
    def set_value(self):
        val = self.var.value
        if val is not None:
            if isinstance(val, str):
                if val == "":
                    self.out.clear()
                else:
                    self.out.print(val)
            else:
                self.out.show_output(val)
        else:
            self.out.clear()
    def update(self, e):
        self.set_value()
    def to_jhtml(self):
        self.set_value()
        return self.out
class FunctionDisplay(Component):
    def __init__(self, fn, vars, pane=None, autoclear=True, **attrs):
        super().__init__()
        if pane is None:
            pane = JHTML.OutputArea(autoclear=autoclear, **attrs)
        self.out = pane
        self.fn = fn
        self.vars = InterfaceVars(*vars) if not isinstance(vars, InterfaceVars) else vars
    def to_widget(self, parent=None):
        needs_link = self._widget_cache is None
        widg = super().to_widget(parent=parent)
        if needs_link:
            self.update = self.update # weakref patch
            for v in self.vars:
                v.callbacks.add(self.update)
                v.link(self)
        return widg
    def observe(self, fn, names=None):
        if self._widget_cache is None:
            raise ValueError("not initialized")
        return self._widget_cache.to_widget().observe(fn, names=names)
    def update(self, e):
        with self.out:
            res = self.fn(event=e, pane=self, **self.vars.dict)
            if isinstance(res, str):
                self.out.print(res)
            elif res is not None:
                self.out.show_output(res)
    def to_jhtml(self):
        self.update(None)
        return self.out

class ProgressBar(Control):
    def __init__(self, var, bar=None, **attrs):
        super().__init__(var)
        if bar is None:
            bar = Progress(self.value, **attrs)
        self.bar = bar
    def get_value(self):
        val = self.var.value
        if val is None or val == "":
            self.var.value = 0
        elif isinstance(val, str):
            self.var.val = int(val)
        return self.var.value
    def set_value(self):
        self.bar.bar['width'] = str(self.value) + "%"
    def update(self, e):
        self.set_value()
    def to_jhtml(self):
        self.set_value()
        return self.bar.to_widget()

class MenuSelect(ValueWidget):
    menu_type = ListGroup
    def __init__(self, var, options, menu_type=None, **attrs):
        super().__init__(var)
        self._value_map, self.ops = self.canonicalize_options(options)
        if menu_type is not None:
            self.menu_type = menu_type
        self._obj = self.menu_type(self.ops, **attrs) #type: MenuComponent
        self._active_item = None
    def get_value(self):
        if self._active_item is not None:
            return self._value_map[self._active_item]
    def set_value(self):
        self.set_active(self.var.value)
    def update(self, e):
        self.set_active(self.value)
    def set_active(self, v):
        for k, test in self._value_map.items():
            if test == v:
                break
        else:
            k = None
        if k is not None:
            self.set_active_key(k)
    def set_active_key(self, k):
        if len(self._obj._item_map) > 0:
            if (
                    self._active_item is not None
                    and k != self._active_item
            ):
                cur = self._obj._item_map[self._active_item]
                if cur is not None:
                    cur.remove_class('active')
            self._active_item = k
            new = self._obj._item_map[k]
            if new is not None:
                new.add_class('active')
    def onclick(self, e, i, v):
        self.var.set_value(v, caller=self)
        # with self.logger_pane:
        self.set_active(v)

    def canonicalize_options(self, options):
        ops = []
        val_dict = {}
        for k in options:
            if not isinstance(k, dict):
                if isinstance(k, str):
                    v = k
                else:
                    try:
                        k, v = k
                    except (TypeError, ValueError):
                        v = k
                k = {
                    'body':k,
                    'value':v
                }
            if 'id' not in k:
                uid = str(uuid.uuid4()).replace("-", "")
                k['id'] = uid
            val_dict[k['id']] = k['value']
            k['event_handlers'] = {'click':lambda *e,i=k['id'],v=k['value']:self.onclick(e, i, v)}
            ops.append(k)
        return val_dict, tuple(ops)
    def to_jhtml(self):
        widg = self._obj.to_jhtml()
        init_key = next(iter(self._value_map.keys()))
        self.set_active(init_key)
        return widg

class DropdownSelect(ValueWidget):
    menu_type = Dropdown
    def __init__(self, var, options, name=None, menu_type=None, **attrs):
        super().__init__(var)
        if menu_type is not None:
            self.menu_type = menu_type
        self.selector = MenuSelect(var, options, menu_type=self.menu_type.List)
        if name is None:
            name = self.var.name
        self.name = name
        self.attrs = attrs

    def get_value(self):
        return self.selector.get_value()
    def set_value(self):
        self.selector.set_value()
    def update(self, e):
        self.selector.update(e)
    def to_jhtml(self):
        return self.menu_type(
            self.name,
            self.selector.to_widget(),
            **self.attrs
        ).to_jhtml()
