
import abc, weakref, uuid
import itertools

from ..JHTML import JHTML, DefaultOutputArea
from ..JHTML.WidgetTools import JupyterAPIs, frozendict

__all__ = [
    "WidgetInterface",
    "Component",
    "Container",
    "MenuComponent",
    "ListGroup",
    "Button",
    "Spinner",
    "Progress",
    "ButtonGroup",
    "Navbar",
    "Carousel",
    "Pagination",
    "Sidebar",
    "Dropdown",
    "DropdownList",
    "Tabs",
    "TabPane",
    "TabList",
    "Accordion",
    "AccordionHeader",
    "AccordionBody",
    "Opener",
    "OpenerHeader",
    "OpenerBody",
    "Modal",
    "ModalHeader",
    "ModalBody",
    "ModalFooter",
    "Offcanvas",
    "OffcanvasHeader",
    "OffcanvasBody",
    "Toast",
    "ToastBody",
    "ToastHeader",
    "ToastContainer",
    "Spacer",
    "Breadcrumb",
    "Card",
    "CardHeader",
    "CardBody",
    "CardFooter",
    "ModifierComponent",
    "Tooltip",
    "Popover",
    "Layout",
    "Grid",
    "Flex"
]
__reload_hook__ = ["..JHTML", "..WidgetTools"]

class WidgetInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_widget(self):
        ...
    # @abc.abstractmethod
    def initialize(self):
        ...
    def _ipython_display_(self):
        JupyterAPIs.get_display_api().display(self.to_widget())
        self.initialize()
    def display(self):
        self._ipython_display_()

class Component(WidgetInterface):
    """
    Provides an abstract base class for an interface element
    to allow for the easy construction of interesting interfaces
    """
    def __init__(self, dynamic=True, debug_pane=None, **attrs):
        self._parents = weakref.WeakSet()
        self._widget_cache = None
        attrs['dynamic'] = dynamic
        self._attrs = attrs
        self.debug_pane = DefaultOutputArea.get_default() if debug_pane is None else debug_pane
    @property
    def attrs(self):
        return frozendict(self._attrs)
    @attrs.setter
    def attrs(self, value):
        self._attrs = value

    def get_attr(self, key):
        return self._attrs[key]
    def get_child(self, key):
        raise NotImplementedError("{} doesn't have children".format(
            type(self).__name__
        ))
    def __getitem__(self, item):
        if isinstance(item, str):
            return self.get_attr(item)
        else:
            return self.get_child(item)

    def set_attr(self, key, value):
        self._attrs[key] = value
    def update_widget_attr(self, key, value):
        self._widget_cache[key] = value
    def set_child(self, which, new):
        raise NotImplementedError("{} doesn't have children".format(
            type(self).__name__
        ))
    def update_widget_child(self, key, value):
        self._widget_cache[key] = value
    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.set_attr(key, value)
            if self._widget_cache is not None:
                self.update_widget_attr(key, value)
        else:
            self.set_child(key, value)
            if self._widget_cache is not None:
                self.update_widget_child(key, value)

    def del_attr(self, key,):
        del self._attrs[key]
    def del_widget_attr(self, key):
        del self._widget_cache[key]
    def del_child(self, which):
        raise NotImplementedError("{} doesn't have children".format(
            type(self).__name__
        ))
    def del_widget_child(self, key):
        del self._widget_cache[key]
    def __delitem__(self, key):
        if isinstance(key, str):
            self.del_attr(key)
            if self._widget_cache is not None:
                self.del_widget_attr(key)
        else:
            self.del_child(key)
            if self._widget_cache is not None:
                self.del_widget_child(key)

    def insert(self, where, new):
        self.insert_child(where, new)
        if self._widget_cache is not None:
            self.insert_widget_child(where, new)
    def append(self, child):
        return self.insert(None, child)

    def insert_child(self, where, child):
        raise NotImplementedError("{} doesn't have children".format(
            type(self).__name__
        ))
    def insert_widget_child(self, where, child):
        self._widget_cache.insert(where, child)

    def add_class(self, *cls):
        self.add_component_class(*cls)
        if self._widget_cache is not None:
            self.add_widget_class(*cls)
    def add_component_class(self, *cls):
        if not 'cls' in self._attrs:
            self._attrs['cls'] = []
        new = self._attrs['cls'].copy()
        for c in cls:
            for y in JHTML.manage_class(c):
                if y not in new:
                    new.append(y)
        self._attrs['cls'] = new
    def add_widget_class(self, *cls):
        return self._widget_cache.add_class(*cls)
    def remove_class(self, *cls):
        self.remove_component_class(*cls)
        if self._widget_cache is not None:
            self.remove_widget_class(*cls)
    def remove_component_class(self, *cls):
        if not 'cls' in self._attrs:
            self._attrs['cls'] = []
        new = self._attrs['cls'].copy()
        for c in cls:
            for y in JHTML.manage_class(c):
                try:
                    new.remove(y)
                except ValueError:
                    pass
        self._attrs['cls'] = new
    def remove_widget_class(self, *cls):
        return self._widget_cache.remove_class(*cls)

    @abc.abstractmethod
    def to_jhtml(self):
        ...
    def to_widget(self, parent=None):
        if parent is not None:
            self._parents.add(parent)
        if self._widget_cache is None:
            with DefaultOutputArea(self.debug_pane):
                self._widget_cache = self.to_jhtml()
                self._widget_cache.component = self
            # self._widget_cache.to_widget.observe(self.set_value, )
        return self._widget_cache
    def mutate(self, fn):
        fn(self)
        self.invalidate_cache()
    def invalidate_cache(self):
        self._widget_cache = None
        for w in self._parents:
            w.invalidate_cache()
class WrapperComponent(Component):
    wrapper = JHTML.Div
    wrapper_classes = []
    def __init__(self, items, wrapper=None, wrapper_classes=None, cls=None, **attrs):
        self.items, attrs = self.manage_items(items, attrs)
        if 'cls' in attrs:
            extra_classes = attrs['cls']
            del attrs['cls']
        else:
            extra_classes = []
        super().__init__(**attrs)
        if wrapper is not None:
            self.wrapper = wrapper
        if wrapper_classes is not None:
            self.wrapper_classes = wrapper_classes
        self.wrapper_classes = self.wrapper_classes + JHTML.manage_class(cls) + JHTML.manage_class(extra_classes)
    @classmethod
    def manage_items(cls, items, attrs):
        if isinstance(items, dict):
            attrs = dict(attrs, **items)
            del attrs['body']
            items = items['body']
        elif (
                isinstance(items, tuple)
                and len(items) == 2
                and isinstance(items[1], dict)
        ):
            attrs = dict(attrs, **items[1])
            items = items[0]
        if (
                isinstance(items, str)
                or hasattr(items, 'to_tree')
                or hasattr(items, 'to_widget')
        ):
            items = [items]
        elif items is None:
            items = []
        else:
            items = list(items)
        return items, attrs
    def get_child(self, key):
        return self.items[key]
    def set_child(self, which, new):
        self.items[which] = new
    def insert_child(self, where, child):
        if where is None:
            where = len(self.items)
        self.items.insert(where, child)

    def add_component_class(self, *cls):
        new = self.wrapper_classes.copy()
        for c in cls:
            for y in JHTML.manage_class(c):
                if y not in new:
                    new.append(y)
        self.wrapper_classes = new
    def remove_component_class(self, *cls):
        new = self.wrapper_classes.copy()
        for c in cls:
            for y in JHTML.manage_class(c):
                try:
                    new.remove(y)
                except ValueError:
                    pass
        self.wrapper_classes = new

    def wrap_items(self, items):
        return self.wrapper(*items, cls=self.wrapper_classes, **self.attrs)
    def to_jhtml(self, parent=None):
        return self.wrap_items(self.items)
class Container(WrapperComponent):
    subwrappers = None
    subwrapper_classes = None
    item = JHTML.Span
    item_classes = []
    def __init__(self, items,
                 wrapper=None, wrapper_classes=None,
                 subwrappers=None, subwrapper_classes=None,
                 item=None, item_classes=None, item_attrs=None,
                 cls=None,
                 **attrs):

        self._items = None
        items, attrs = self.manage_items(items, attrs)
        if subwrappers is None:
            subwrappers = self.subwrappers
        if subwrapper_classes is None:
            subwrapper_classes = self.subwrapper_classes
        if subwrappers is not None:
            if wrapper is None:
                wrapper = self.wrapper
            if wrapper_classes is not None:
                self.wrapper_classes = wrapper_classes
            if subwrapper_classes is None:
                subwrapper_classes = [None]*len(subwrappers)
            if len(subwrappers) != len(subwrapper_classes):
                raise ValueError("mismatch between number of subwrappers and wrapper classes")

            wrapper = JHTML.Compound(
                JHTML.Styled(wrapper, cls=self.wrapper_classes + JHTML.manage_class(cls), **attrs),
                *(
                    JHTML.Styled(sw, cls=JHTML.manage_class(sc))
                    for sw, sc in zip(subwrappers, subwrapper_classes)
                  )
            )
            self.wrapper_classes = JHTML.manage_class(subwrapper_classes[-1])
            super().__init__(None, wrapper=wrapper)
        else:
            super().__init__(None, cls=cls, wrapper=wrapper, wrapper_classes=wrapper_classes, **attrs)
        self._items = items

        if item is not None:
            self.item = item
        if item_classes is not None:
            self.item_classes = item_classes
        if item_attrs is None:
            item_attrs = {}
        if 'cls' in item_attrs:
            item_attrs = item_attrs.copy()
            self.item_classes = self.item_classes + JHTML.manage_class(item_attrs['cls'])
            del item_attrs['cls']
        self.item_attrs = item_attrs

    @property
    def items(self):
        return [self.create_item(i) for i in self._items]
    @items.setter
    def items(self, items):
        if self._items is not None:
            raise ValueError("can't set items")
    def _create_dict_item(self, body=None, cls=None, **extra):
        return self.item(body, cls=self.item_classes + JHTML.manage_class(cls), **dict(self.item_attrs, **extra))
    def _create_base_item(self, body):
        return self.item(body, cls=self.item_classes, **self.item_attrs)
    def create_item(self, i, **kw):
        if isinstance(i, dict):
            if 'raw' in i:
                return i['raw']
            if len(kw) > 0:
                return self._create_dict_item(**dict(i, **kw))
            else:
                return self._create_dict_item(**i)
        else:
            if len(kw) > 0:
                return self._create_dict_item(body=i, **kw)
            else:
               return self._create_base_item(i)

    def update_widget_child(self, key, value):
        super().update_widget_child(key, self.create_item(value))
    def insert_widget_child(self, where, child):
        super().insert_widget_child(where, self.create_item(child))

class ModifierComponent(Component):
    modifiers = None
    def __init__(self, base=None, **modifiers):
        base_mods = self.modifiers
        if base_mods is None:
            base_mods = {}
        modifiers = dict(base_mods, **modifiers)
        super().__init__(**modifiers)
        self.base = base
    def __call__(self, base):
        if self.base is None:
            self.base = base
        else:
            raise ValueError("{} already has a base object".format(
                self
            ))
        return self
    blacklist = {'dynamic'}
    def to_jhtml(self):
        base = self.base
        if hasattr(base, 'to_jhtml'):
            base = base.to_jhtml()
        for k,v in self.attrs.items():
            if k not in self.blacklist:
                base[k] = v
        return base

class Button(WrapperComponent):
    wrapper = JHTML.Bootstrap.Button
    def __init__(self, body, action=None, event_handlers=None, **kwargs):
        if event_handlers is None:
            event_handlers = {}
        self._action = action
        if isinstance(action, (str, dict, list)):
            event_handlers['click'] = action
        else:
            event_handlers['click'] = self._eval
        super().__init__(
            body,
            event_handlers=event_handlers,
            **kwargs
        )
    @property
    def action(self):
        return self._action
    @action.setter
    def action(self, a):
        self._action = a
        if isinstance(a, (str, dict, list)):
            self._attrs['event_handlers'] = {'click':a}
            if self._widget_cache is not None:
                self._widget_cache.add_event(click=a)
        else:
            self._attrs['event_handlers'] = {'click':self._eval}
            if self._widget_cache is not None:
                self._widget_cache.add_event(click=self._eval)
    def _eval(self, *args):
        if self.action is not None:
            w = self.to_widget()
            try:
                pane = w.debug_pane
            except AttributeError:
                pane = None
            if pane is None:
                self.action(*args)
            else:
                with pane:
                    self.action(*args)

class Spinner(WrapperComponent):
    def __init__(self, variant="border", body=None, cls=None, **attrs):
        """
        <div class="spinner-border text-primary" role="status">
  <span class="visually-hidden">Loading...</span>
</div>
        :param variant:
        :type variant:
        :param attrs:
        :type attrs:
        """
        cls = JHTML.manage_class(cls) + ['spinner-'+variant]
        if body is None:
            body = JHTML.Span("Loading...", cls="visually-hidden")
        super().__init__(body, cls=cls, **attrs)

class Progress(WrapperComponent):
    wrapper_classes = ['progress']
    subwrapper_classes = ['progress-bar']
    def __init__(self, value=0, label=None, cls=None, **attrs):
        items = []
        if label is True:
            label = str(value) + "%"
        elif not label:
            label = None
        if label is not None:
            items.append(label)
        super().__init__(
            items,
            wrapper=JHTML.Compound(
                ('container', JHTML.Styled(JHTML.Div, cls=JHTML.manage_class(cls) + self.wrapper_classes, **attrs)),
                ('bar', JHTML.Styled(JHTML.Div, width=str(value)+"%", dynamic=True))
            ),
            wrapper_classes = self.subwrapper_classes
        )
    @property
    def container(self):
        return self.to_widget().compound_wrapper_data["container"]
    @property
    def bar(self):
        return self.to_widget().compound_wrapper_data["bar"]
    def update_widget_attr(self, attr, val):
        self.container[attr] = val

#region Menus

class MenuComponent(Container):
    def __init__(self, items, item_attrs=None, cls=None, **attrs):
        super().__init__(items, item_attrs=item_attrs, cls=cls, **attrs)
        self._item_map = {}
    def create_item(self, item, **kw):
        item = super().create_item(item, **kw)
        if hasattr(item, 'id'):
            self._item_map[item.id] = item
        elif 'id' in item.attrs:
            self._item_map[item.attrs['id']] = item
        return item
class ListGroup(MenuComponent):
    item_classes = ['list-group-item', 'list-group-item-action']
    wrapper_classes = ['list-group', 'list-group-flush']
class ButtonGroup(MenuComponent):
    item = JHTML.Bootstrap.Button
    wrapper_classes = ['btn-group']
class DropdownList(MenuComponent):
    wrapper = JHTML.Ul
    wrapper_classes = ['dropdown-menu']
    item = JHTML.Compound(JHTML.Li, JHTML.Anchor)
    item_classes = ['dropdown-item']
    def create_item(self, item, **kw):
        item = super().create_item(item, **kw)
        if 'action' in item:
            item = item.copy()
            item['event_handlers'] = {'click':item['action']}
            del item['action']
        return item
class Dropdown(MenuComponent):
    wrapper_classes = ['dropdown']
    toggle_classes = ['dropdown-toggle']
    toggle = JHTML.Styled(JHTML.Button, data_bs_toggle='dropdown')
    List = DropdownList
    def __init__(self, header, actions, toggle_attrs=None, **attrs):
        self.header = header
        self.toggle_attrs = {} if toggle_attrs is None else toggle_attrs
        if 'cls' in self.toggle_attrs:
            self.toggle_attrs = self.toggle_attrs.copy()
            self.toggle_classes = self.toggle_classes + JHTML.manage_class(self.toggle_attrs['cls'])
            del self.toggle_attrs['cls']
        self.dlist = DropdownList(actions, **attrs) if not hasattr(actions, 'to_widget') else actions
        super().__init__([], **attrs)
    def wrap_items(self, items):
        items = [
            self.toggle(self.header, cls=self.toggle_classes, **self.toggle_attrs),
            self.dlist
        ]
        return super().wrap_items(items)

class Navbar(MenuComponent):
    wrapper = JHTML.Nav
    wrapper_classes = ['navbar', 'navbar-expand']
    subwrappers = [JHTML.Div, JHTML.Div]
    subwrapper_classes = [['container-fluid'], ['navbar-nav']]
    item = JHTML.Anchor
    item_classes = ['nav-link']
class Sidebar(MenuComponent):
    wrapper = JHTML.Nav
    wrapper_classes = ['nav', 'flex-column']
    # subwrappers = [JHTML.Div, JHTML.Div]
    # subwrapper_classes = [['container-fluid'], ['navbar-nav']]
    item = JHTML.Compound(
        JHTML.Styled(JHTML.Li, cls='nav-item'),
        JHTML.Anchor
    )
    item_classes = ['nav-link']

class Pagination(MenuComponent):
    wrapper = JHTML.Nav
    subwrappers = [JHTML.Ul]
    subwrapper_classes = [['pagination']]
    item = JHTML.Compound(
        JHTML.Styled(JHTML.Li, cls='page-item'),
        JHTML.Anchor
    )
    item_classes = ['page-link']


def short_uuid(len=6):
    return str(uuid.uuid4()).replace("-", "")[:len]

class Carousel(MenuComponent):
    wrapper = JHTML.Div
    wrapper_classes = ['carousel']
    subwrappers = [JHTML.Div]
    subwrapper_classes = ['carousel-inner']
    item = JHTML.Div
    item_classes = ['carousel-item', 'bg-secondary', 'text-light']
    def __init__(self, items, include_controls=True, data_bs_ride='carousel', **attrs):
        self.include_controls = include_controls
        self._active_made = False
        self.base_name = 'carousel-' + short_uuid()
        super().__init__(items, id=self.base_name, data_bs_ride=data_bs_ride, **attrs)
    def create_item(self, item, cls=None, data_bs_interval="10000000000", **kw):
        if not self._active_made:
            cls = JHTML.manage_class(cls) + ['active']
            self._active_made = True
        return super().create_item(item, cls=cls, data_bs_interval=data_bs_interval, **kw)
    def wrap_items(self, items):
        base = super().wrap_items(items)
        if self.include_controls:
            base.append(
                JHTML.Button(
                    JHTML.Span(cls='carousel-control-prev-icon'),
                    cls='carousel-control-prev', data_bs_target='#'+self.base_name, data_bs_slide='prev'
                )
            )

            base.append(
                JHTML.Button(
                    JHTML.Span(cls='carousel-control-next-icon'),
                    cls='carousel-control-next', data_bs_target='#'+self.base_name, data_bs_slide='next'
                )
            )
        return base


class TabList(MenuComponent):
    wrapper = JHTML.Ul
    wrapper_classes = ['nav', 'nav-tabs']
    item = JHTML.Compound(
        JHTML.Styled(JHTML.Li, cls='nav-item'),
        JHTML.Button
    )
    item_classes = ['nav-link']
    def __init__(self, *args, base_name=None, role="tablist", **kwargs):
        if base_name is None:
            base_name = 'tabs-' + str(uuid.uuid1()).replace("-", "")[:5]
        self.base_name = base_name
        self._active = None
        super().__init__(*args, role=role, **kwargs)
    def create_item(self, item, cls=None, **kw):
        item, _ = item
        item_id = self.base_name+"-"+item.replace(' ', '')
        if self._active is None:
            cls = JHTML.manage_class(cls) + ['active']
            self._active = item_id
        return super().create_item(item, id=item_id+'-tab', cls=cls, data_bs_target='#'+item_id, data_bs_toggle='tab', **kw)
class TabPane(MenuComponent):
    wrapper_classes = ['tab-content']
    item = JHTML.Div
    item_classes = ['tab-pane']
    def __init__(self, *args, base_name=None, **kwargs):
        if base_name is None:
            base_name = 'tabs-' + short_uuid()
        self.base_name = base_name
        self._active = None
        super().__init__(*args, **kwargs)
    def create_item(self, item, cls=None, **kw):
        key, item = item
        item_id = self.base_name+"-"+key.replace(' ', '')
        if self._active is None:
            cls = JHTML.manage_class(cls) + ['active']
            self._active = item_id
        item = super().create_item(item, id=item_id, role='tabpanel', cls=cls, **kw)
        return item
class Tabs(MenuComponent):
    wrapper_classes = []
    # toggle_classes = ['dropdown-toggle']
    # toggle = JHTML.Styled(JHTML.Bootstrap.Button, data_bs_toggle='dropdown')
    def __init__(self, tabs, base_name=None, **attrs):
        if base_name is None:
            base_name = 'tabs-' + str(uuid.uuid1()).replace("-", "")[:5]
        if isinstance(tabs, dict):
            tabs = tabs.items()
        self.tab_list = TabList(tabs, id=base_name, base_name=base_name)
        self.panes = TabPane(tabs, id=base_name+'Content', base_name=base_name)
        super().__init__([], **attrs)
    def wrap_items(self, items):
        items = [
            self.tab_list,
            self.panes
        ]
        return super().wrap_items(items)

class AccordionHeader(Container):
    wrapper_classes = ['accordion-header']
    item = JHTML.Button
    item_classes = ['accordion-button']
    def __init__(self, key, base_name=None, **kw):
        self.base_name = base_name
        super().__init__([key], id=self.base_name+'-heading', **kw)
    def create_item(self, i, **kw):
        return super().create_item(i, type='button', data_bs_toggle='collapse', data_bs_target='#'+self.base_name+'-collapse')
class AccordionBody(Container):
    wrapper_classes = ['accordion-collapse', 'collapse']
    item = JHTML.Div
    item_classes = ['accordion-body']
    def __init__(self, key, parent_name=None, base_name=None, **kw):
        self.base_name = base_name
        super().__init__([key], id=self.base_name+'-collapse', data_bs_parent='#'+parent_name, **kw)
class Accordion(MenuComponent):
    wrapper_classes = ['accordion']
    item = JHTML.Div
    item_classes = ['accordion-item']
    header_classes = ['h2']
    def __init__(self, items, base_name=None, header_classes=None, **attrs):
        if base_name is None:
            base_name = 'accordion-' + short_uuid()
        self.base_name = base_name
        if isinstance(items, dict):
            items = items.items()
        self._active = None
        if header_classes is not None:
            self.header_classes = JHTML.manage_class(header_classes)
        super().__init__(items, id=self.base_name, **attrs)
    def create_item(self, item, cls=None, **kw):
        key, item = item
        item_id = self.base_name + "-" + short_uuid(3)
        cls = JHTML.manage_class(cls)
        if self._active is None:
            cls = cls + ['show']
            self._active = item_id
            header_cls = None
        else:
            header_cls = ['collapsed']

        header = AccordionHeader(key, base_name=item_id, item_attrs={'cls':header_cls}, wrapper_classes=self.header_classes)
        body = AccordionBody(item, parent_name=self.base_name, base_name=item_id, cls=cls, **kw)
        return super().create_item([header, body])

class OpenerHeader(Container):
    wrapper_classes = ['collapse-header']
    item = JHTML.Button
    item_classes = ['accordion-button']
    def __init__(self, key, base_name=None, **kw):
        self.base_name = base_name
        super().__init__([key], id=self.base_name+'-heading', **kw)
    def create_item(self, i, **kw):
        return super().create_item(i, type='button', data_bs_toggle='collapse', data_bs_target='#'+self.base_name+'-collapse')
class OpenerBody(Container):
    wrapper_classes = ['collapse']
    item = JHTML.Div
    item_classes = ['collapse-body']
    def __init__(self, key, base_name=None, **kw):
        self.base_name = base_name
        super().__init__([key], id=self.base_name+'-collapse', **kw)#, data_bs_parent='#'+parent_name, **kw)
class Opener(MenuComponent):
    wrapper_classes = ['opener']
    item = JHTML.Div
    item_classes = ['opener-item']
    header_classes = []
    def __init__(self, items, base_name=None, header_classes=None, **attrs):
        if base_name is None:
            base_name = 'opener-' + short_uuid()
        self.base_name = base_name
        if isinstance(items, dict):
            items = items.items()
        # self._active = None
        if header_classes is not None:
            self.header_classes = JHTML.manage_class(header_classes)
        super().__init__(items, id=self.base_name, **attrs)
    def create_item(self, item, cls=None, **kw):
        key, item = item
        item_id = self.base_name + "-" + short_uuid(3)
        cls = JHTML.manage_class(cls)
        # if self._active is None:
        #     cls = cls + ['show']
        #     self._active = item_id
        #     header_cls = None
        # else:
        header_cls = ['collapsed']

        header = OpenerHeader(key, base_name=item_id,
                              item_attrs={'cls':self.header_classes + header_cls}
                              # wrapper_classes=self.header_classes
                              )
        body = OpenerBody(item, base_name=item_id, cls=cls, **kw)
        return super().create_item([header, body])

class Breadcrumb(MenuComponent):
    wrapper = JHTML.Nav
    subwrappers = [JHTML.Ol]
    subwrapper_classes = ['breadcrumb']
    item = JHTML.Li
    item_classes = ['breadcrumb-item']

class CardBody(WrapperComponent):
    wrapper = JHTML.Bootstrap.CardBody
class CardHeader(WrapperComponent):
    wrapper = JHTML.Bootstrap.CardHeader
class CardFooter(WrapperComponent):
    wrapper = JHTML.Bootstrap.CardHeader
class Card(WrapperComponent):
    wrapper = JHTML.Bootstrap.Card
    def __init__(self,
                 header=None,
                 body=None,
                 footer=None,
                 **attrs
                 ):
        items = []
        if header is not None:
            header = CardHeader(header)
            items.append(header)
        self.header = header
        if body is not None:
            body = CardBody(body)
            items.append(body)
        self.body = body
        if footer is not None:
            footer = CardFooter(footer)
            items.append(footer)
        self.footer = footer
        super().__init__(items, **attrs)

class Modal(Container):
    wrapper_classes = ['modal', 'fade']
    subwrappers = [JHTML.Div, JHTML.Div]
    subwrapper_classes = [['modal-dialog', 'modal-dialog-centered'], ['modal-content']]
    def __init__(self,
                 header=None,
                 body=None,
                 footer=None,
                 id=None,
                 tabindex=-1,
                 **attrs
                 ):
        items = []
        if header is not None:
            header = ModalHeader(header)
            items.append(header)
        self.header = header
        if body is not None:
            body = ModalBody(body)
            items.append(body)
        self.body = body
        if footer is not None:
            footer = ModalFooter(footer)
            items.append(footer)
        self.footer = footer
        if id is None:
            id = 'modal-'+short_uuid(6)
        self._id = id
        super().__init__(items, id=id, tabindex=tabindex, **attrs)

    trigger_class = JHTML.Bootstrap.Button
    def get_trigger(self, *items, trigger_class=None, data_bs_toggle='modal', data_bs_target=None, **attrs):
        if trigger_class is None:
            trigger_class = self.trigger_class
        if data_bs_target is None:
            data_bs_target = "#"+self._id
        return trigger_class(items, data_bs_toggle=data_bs_toggle, data_bs_target=data_bs_target, **attrs)
    @classmethod
    def close_button(self):
        return JHTML.Button(cls='btn-close', data_bs_dismiss='modal')
class ModalHeader(WrapperComponent):
    wrapper_classes = ['modal-header']
    def __init__(self, items, **attrs):
        items, attrs = self.manage_items(items, attrs)
        items.append(Modal.close_button())
        super().__init__(items, **attrs)
class ModalFooter(WrapperComponent):
    wrapper_classes = ['modal-footer']
class ModalBody(WrapperComponent):
    wrapper_classes = ['modal-body']

class Offcanvas(Container):
    wrapper_classes = ['offcanvas', 'ps-4', 'pt-5', 'pb-5']
    def __init__(self,
                 header=None,
                 body=None,
                 id=None,
                 tabindex=-1,
                 cls=None,
                 placement='start',
                 **attrs
                 ):
        items = []
        if header is not None:
            header = OffcanvasHeader(header)
            items.append(header)
        self.header = header
        if body is not None:
            body = OffcanvasBody(body)
            items.append(body)
        self.body = body
        if id is None:
            id = 'offcanvas-'+short_uuid(6)
        self._id = id
        cls = JHTML.manage_class(cls) + ['offcanvas-' + placement]
        super().__init__(items, id=id, tabindex=tabindex, cls=cls, **attrs)

    trigger_class = JHTML.Bootstrap.Button
    def get_trigger(self, *items, trigger_class=None, data_bs_toggle='offcanvas', data_bs_target=None, **attrs):
        if trigger_class is None:
            trigger_class = self.trigger_class
        if data_bs_target is None:
            data_bs_target = "#"+self._id
        return trigger_class(items, data_bs_toggle=data_bs_toggle, data_bs_target=data_bs_target, **attrs)
    @classmethod
    def close_button(self):
        return JHTML.Button(cls='btn-close', data_bs_dismiss='offcanvas')
class OffcanvasHeader(WrapperComponent):
    wrapper_classes = ['offcanvas-header', 'm-2', 'border-bottom']
    def __init__(self, items, **attrs):
        items, attrs = self.manage_items(items, attrs)
        items.append(Offcanvas.close_button())
        super().__init__(items, **attrs)
class OffcanvasBody(WrapperComponent):
    wrapper_classes = ['offcanvas-body']

class Spacer(WrapperComponent):
    wrapper = JHTML.Span
    wrapper_classes = ['me-auto']
    def __init__(self, items=None, **kwargs):
        if items is None:
            items = []
        super().__init__(items, **kwargs)

ToastAPI = JHTML.JavascriptAPI.loader(
    init="""widget.el.toast = new context.bootstrap.Toast(widget.el)""",
    showToast="""
let id = widget.getAttribute("data-bs-target");
let el = document.querySelector(id);
el.toast.show();
"""
)
class ToastBody(WrapperComponent):
    wrapper_classes = ['toast-body']
    def __init__(self, items, include_controls=False, cls=None, **attrs):
        if include_controls:
            cls = JHTML.manage_class(cls) + ['d-flex']
            items, attrs = self.manage_items(items, attrs)
            items.extend([Spacer(), Toast.close_button()])
        super().__init__(items, cls=cls, **attrs)
class ToastHeader(WrapperComponent):
    wrapper_classes = ['toast-header']
    def __init__(self, items, include_controls=True, **attrs):
        if include_controls:
            items, attrs = self.manage_items(items, attrs)
            items.extend([Spacer(), Toast.close_button()])
        super().__init__(items, **attrs)
class Toast(WrapperComponent):
    wrapper_classes = ['toast']
    def __init__(self,
                 header=None,
                 body=None,
                 role='alert',
                 hidden=True,
                 cls=None,
                 id=None,
                 javascript_handles=None,
                 onevents=None,
                 **attrs
                 ):
        attrs['role'] = role
        items = []
        only_body = header is not None and body is None
        if only_body:
            body = header
            header = None
        if header is not None:
            items.append(ToastHeader(header))
        if body is not None:
            items.append(ToastBody(body, include_controls=only_body))
        if id is None:
            id = 'toast-'+short_uuid(6)
        self._id = id
        if not hidden:
            cls = JHTML.manage_class(cls) + ['show']
        if javascript_handles is None:
            javascript_handles = ToastAPI.load()
        if onevents is None:
            onevents = {'initialize':'init'}
        super().__init__(items, cls=cls, javascript_handles=javascript_handles, onevents=onevents, id=id, **attrs)

    trigger_class = JHTML.Bootstrap.Button
    def get_trigger(self, *items, trigger_class=None, data_bs_toggle='toast', data_bs_target=None, **attrs):
        if trigger_class is None:
            trigger_class = self.trigger_class
        if data_bs_target is None:
            data_bs_target = "#" + self._id
        return trigger_class(items,
                             data_bs_toggle=data_bs_toggle,
                             data_bs_target=data_bs_target,
                             javascript_handles=self.to_widget().javascript_handles,
                             event_handlers={"click":"showToast"},
                             **attrs
                             )
    @classmethod
    def close_button(self):
        return JHTML.Button(cls='btn-close', data_bs_dismiss='toast')
    def show(self):
        self.add_class('show')
        self.remove_class('hide')
    def hide(self):
        self.remove_class('show')
        self.add_class('hide')
class ToastContainer(WrapperComponent):
    wrapper_classes = ['toast-container']
    def __init__(self, items=None, **kwargs):
        if items is None:
            items = []
        super().__init__(items, **kwargs)
    def create_toast(self, header=None, body=None, hidden=False, **kwargs):
        toast = Toast(header=header, body=body, hidden=hidden, **kwargs)
        self.append(toast)
        return toast


#endregion

#region Misc

class Tooltip(ModifierComponent):
    modifiers = dict(
        data_bs_toggle="tooltip",
        data_bs_placement='top',
        javascript_handles={
        "update": """
if (widget.tooltip !== null && widget.tooltip !== undefined) {
    let title = widget.getAttribute('title');
    if (!(title && typeof title === 'object') || title !== this.tooltipTitle) {
        this.callHandler("init", event);
    } else {
        this.tooltip.update();
        this.el.setAttribute("title", "");
    }
}
                """,
        "init": """
let shown=false;
if (widget.tooltip !== null && widget.tooltip !== undefined) {
    if (widget.tooltip.tip !== null) {
        widget.tooltipTitle = null;
        let tip = widget.tooltip.tip;
        shown = tip && tip.classList.contains("show");
        widget.tooltip.dispose();
    }
}
widget.tooltip = new context.bootstrap.Tooltip(widget.el);
widget.tooltip._setContent = widget.tooltip.setContent;
function wrapperSetContent(tip) {
    widget.tooltip._setContent(tip);
    let title = widget.getAttribute('title');
    if (title && typeof title === 'object') {
        if (!(widget.tooltip.tip && title == widget.tooltipTitle)) {
            widget.tooltipTitle = title;
            let bwrap = tip.querySelector('.tooltip-inner');
            if (bwrap) {
                widget.create_child_view(title).then(
                  (view)=>{
                        while (bwrap.lastChild) {
                            bwrap.removeChild(bwrap.lastChild);
                        };
                        bwrap.appendChild(view.el)
                    }
                );
            }
        }
    }
    tip.classList.add('jhtml');
    return tip;
}
widget.tooltip.setContent = wrapperSetContent;
if (shown) {
    let trigger = widget.el.getAttribute("data-bs-trigger");
    if (trigger && trigger.split(' ').includes('click')) {
        widget.el.click();
    } else {
        widget.tooltip.show()
    }
};
""",
        "destroy": """
    if (widget.tooltip !== null && widget.tooltip !== undefined) {
        if (widget.tooltip.tip !== null) {
            widget.tooltip.dispose();
        }
    }"""
        },
        onevents={
            'initialize': "init",
            'view-change:data-bs-trigger': 'init',
            'view-change:data-bs-placement': 'init',
            'view-change:title': 'update',
            'remove': "destroy"
        }
    )
    def __init__(self, base=None, title="tooltip", data_bs_html=None, **kwargs):
        if not isinstance(title, str):
            if hasattr(title, 'tostring'):
                title = title.tostring()
                if data_bs_html is None:
                    data_bs_html = True
            elif hasattr(title, 'to_widget'):
                title = title.to_widget()
                if hasattr(title, 'elem'):
                    title = title.elem
                kwargs['onevents'] = dict(self.modifiers['onevents'], change='init')
                if data_bs_html is None:
                    data_bs_html = True
            elif isinstance(title, JupyterAPIs.get_widgets_api().Widget):
                if data_bs_html is None:
                    data_bs_html = True
            elif hasattr(title, 'to_tree'):
                title = title.to_tree().tostring()
                if data_bs_html is None:
                    data_bs_html = True
            else:
                raise TypeError('tooltip title must be a string')
        if data_bs_html is not None:
            kwargs['data_bs_html'] = data_bs_html
        super().__init__(base, title=title, **kwargs)
class Popover(ModifierComponent):
    modifiers = dict(
        data_bs_toggle="popover",
        data_bs_placement='top',
        data_bs_container='body',
        tabindex="0",
        javascript_handles={
            "update":"""
if (widget.popover !== null && widget.popover !== undefined) {
    let body = widget.getAttribute('data-bs-content');
    if (!(body && typeof body === 'object') || body !== this.popoverBody) {
        this.callHandler("init", event);
    } else {
        let title = widget.getAttribute('title');
        if (!(title && typeof title === 'object') || title !== this.popoverTitle) {
            this.callHandler("init", event);
        } else {
            this.popover.update();
            this.el.setAttribute("title", "")
        }
    }
}
            """,
            "init":"""
let shown=false;
if (widget.popover !== null && widget.popover !== undefined) {
    if (widget.popover.tip !== null) {
        widget.popoverBody = null;
        widget.popoverTitle = null;
        let tip = widget.popover.tip;
        shown = tip && tip.classList.contains("show");
        widget.popover.dispose();
    }
}
widget.popover = new context.bootstrap.Popover(widget.el);
widget.popover._oldGetTipElement = widget.popover.getTipElement;
function wrapperGetElement() {
    let tip = widget.popover._oldGetTipElement();
    let body = widget.getAttribute('data-bs-content');
    if (body && typeof body === 'object') {
        if (body !== widget.popoverBody) {
            widget.popoverBody = body;
            let bwrap = widget.popover.tip.querySelector('.popover-body');
            if (bwrap) {
                widget.create_child_view(body).then(
                  (view)=>{
                        while (bwrap.lastChild) {
                            bwrap.removeChild(bwrap.lastChild);
                        };
                        bwrap.appendChild(view.el)
                    }
                );
            }
        }
    }
    let title = widget.getAttribute('title');
    if (title && typeof title === 'object') {
        if (title !== widget.popoverTitle) {
            widget.popoverTitle = title;
            let bwrap = widget.popover.tip.querySelector('.popover-header');
            if (bwrap) {
                widget.create_child_view(title).then(
                  (view)=>{
                        while (bwrap.lastChild) {
                            bwrap.removeChild(bwrap.lastChild);
                        };
                        bwrap.appendChild(view.el)
                    }
                );
            }
        }
    }
    tip.classList.add('jhtml');
    return tip;
}
widget.popover.getTipElement = wrapperGetElement;
if (shown) {
    const triggers = widget.el.getAttribute("data-bs-trigger").split(' ');
    if (triggers.includes('click')) {
        widget.el.click();
    } else {
        widget.popover.show()
    }
};
""",
                            "destroy":"""
if (widget.popover !== null && widget.popover !== undefined) {
    if (widget.popover.tip !== null) {
        widget.popover.dispose();
    }
}"""},
        onevents={
            'initialize':"init",
            'view-change:data-bs-trigger':'init',
            'view-change:data-bs-placement':'init',
            'view-change:title': 'update',
            'view-change:data-bs-content':'update',
            'remove':"destroy"
        }
    )
    def __init__(self, base=None, body="", data_bs_trigger="hover focus", data_bs_html=None, title=None, **kwargs):
        if title is not None:
            if not isinstance(title, str):
                if hasattr(title, 'tostring'):
                    title = title.tostring()
                    if data_bs_html is None:
                        data_bs_html = True
                elif hasattr(title, 'to_tree'):
                    title = title.to_tree().tostring()
                    if data_bs_html is None:
                        data_bs_html = True
                elif isinstance(title, JupyterAPIs.get_widgets_api().Widget):
                    if data_bs_html is None:
                        data_bs_html = True
                elif hasattr(title, 'to_widget'):
                    title = title.to_widget()
                    if hasattr(title, 'elem'):
                        title = title.elem
                    if data_bs_html is None:
                        data_bs_html = True
                else:
                    raise TypeError('popover title must be a string')
            kwargs['title'] = title
        if not isinstance(body, str):
            if hasattr(body, 'tostring'):
                body = body.tostring()
                if data_bs_html is None:
                    data_bs_html = True
            elif hasattr(body, 'to_tree'):
                body = body.to_tree().tostring()
                if data_bs_html is None:
                    data_bs_html = True
            elif isinstance(body, JupyterAPIs.get_widgets_api().Widget):
                if data_bs_html is None:
                    data_bs_html = True
            elif hasattr(body, 'to_widget'):
                body = body.to_widget()
                if hasattr(body, 'elem'):
                    body = body.elem
                if data_bs_html is None:
                    data_bs_html = True
            else:
                raise TypeError('popover body must be a string')
        if data_bs_trigger is not None:
            kwargs['data_bs_trigger'] = data_bs_trigger
        if data_bs_html is not None:
            kwargs['data_bs_html'] = data_bs_html
        super().__init__(base, data_bs_content=body, **kwargs)

#endregion

#region Layouts

class LayoutItem(Component):
    wrapper = JHTML.Div
    properties = []
    def __init__(self, item, **attrs):
        super().__init__()
        self.item = item
        self.attrs = attrs
    @abc.abstractmethod
    def get_layout_styles(self, **kwargs):
        raise NotImplementedError("LayoutItem is an abstract class")
    def to_jhtml(self):
        attrs = self.attrs
        style = self.get_layout_styles()
        if 'style' in self.attrs:
            style = dict(attrs['style'], **style)
            attrs = attrs.copy()
            del attrs['style']
        wat = self.wrapper(
            self.item,
            style=style,
            **attrs
        )
        return wat
class Layout(Component):
    wrapper = JHTML.Div
    Item = LayoutItem
    def __init__(self, elements, wrapper=None, item_attrs=None, style=None, **attrs):
        super().__init__()
        if item_attrs is None:
            item_attrs = {}
        self.layout_settings, self.elements = self.setup_layout(elements, item_attrs)
        self._style = {} if style is None else style
        self.attrs = attrs
        if wrapper is None:
            wrapper = self.wrapper
        self.wrapper = wrapper
    def wrap_item(self, e, attrs):
        return self.Item(e, **attrs)
    def setup_layout(self, elements, item_attrs):
        return None, [self.wrap_item(e, item_attrs) for e in elements]
    @abc.abstractmethod
    def get_layout_styles(self, **kwargs):
        raise NotImplementedError("Layout is an abstract class")
    @property
    def styles(self):
        return dict(self._style, **self.get_layout_styles())
    def to_jhtml(self):
        return self.wrapper(
            *self.elements,
            style=self.styles,
            **self.attrs
        )

class GridItem(LayoutItem):
    def __init__(self, item,
                 row=None, col=None,
                 row_span=None, col_span=None,
                 alignment=None, justification=None,
                 **attrs
                 ):
        super().__init__(item, **attrs)
        self.row = row
        self.col = col
        self.row_span = row_span
        self.col_span = col_span
        self.alignment = alignment
        self.justification = justification
        self.attrs = attrs
    @classmethod
    def get_grid_styles(cls,
                           row=None, row_span=None,
                           col=None, col_span=None,
                           alignment=None, justification=None
                           ):
        settings = {}
        if row is not None:
            settings['grid-row-start'] = row
            if row_span is not None:
                settings['grid-row-end'] = 'span ' + str(row_span)
        if col is not None:
            settings['grid-column-start'] = col
            if col_span is not None:
                settings['grid-column-end'] = 'span ' + str(col_span)
        if alignment is not None:
            settings['align-self'] = alignment
        if justification is not None:
            settings['justify-self'] = justification
        return settings
    def get_layout_styles(self):
        return self.get_grid_styles(
            row=self.row, row_span=self.row_span,
            col=self.col, col_span=self.col_span,
            alignment=self.alignment, justification=self.justification,
        )
class Grid(Layout):
    Item = GridItem
    def __init__(self, elements,
                 rows=None, cols=None,
                 alignment=None, justification=None,
                 row_spacing=None, col_spacing=None,
                 item_attrs=None,
                 row_height='1fr',
                 column_width='1fr',
                 **attrs
                 ):
        super().__init__(elements, item_attrs=item_attrs, **attrs)
        if rows is None:
            rows = self.layout_settings['rows']
        if cols is None:
            cols = self.layout_settings['cols']
        self.rows = rows
        self.cols = cols
        self.alignment = alignment
        self.justification = justification
        self.row_gaps = row_spacing
        self.col_gaps = col_spacing
        self.row_height = row_height
        self.col_width = column_width
    def setup_layout(self, grid, attrs):
        elements = []
        nrows = 0
        ncols = 0
        for i, row in enumerate(grid):
            for j, el in enumerate(row):
                if el is None:
                    continue
                elem = self.wrap_item(el, dict(attrs, row=i+1, col=j+1))
                n = elem.row
                if elem.row_span is not None:
                    n += elem.row_span
                if n > nrows:
                    nrows = n
                m = elem.col
                if elem.col_span is not None:
                    m += elem.col_span
                if m > ncols:
                    ncols = m
                elements.append(elem)
        return {'rows':nrows, 'cols':ncols}, elements
    def wrap_item(self, e, attrs):
        if not isinstance(e, self.Item):
            e = self.Item(e, **attrs)
        elif hasattr(e, 'items'):
            body = e['body']
            e = dict(e)
            del e['body']
            e = GridItem(body, **dict(attrs, **e))
        else:
            if e.row is None:
                e.row = attrs['row']
            if e.col is None:
                e.col = attrs['col']
        return e
    @classmethod
    def get_grid_styles(cls,
                        rows=None,  cols=None,
                        alignment=None, justification=None,
                        row_gap=None, col_gap=None,
                        row_height='1fr', col_width='1fr'
                        ):
        settings = {'display':'grid'}
        if rows is not None:
            if isinstance(row_height, str) and ' ' not in row_height:
                settings['grid-template-rows'] = 'repeat({rows}, {height})'.format(rows=rows, height=row_height) if isinstance(rows, int) else rows
            elif isinstance(row_height, str):
                settings['grid-template-rows'] = row_height
            else:
                settings['grid-template-rows'] = " ".join(row_height)

        if cols is not None:
            if isinstance(col_width, str) and ' ' not in col_width:
                settings['grid-template-columns'] = 'repeat({cols}, {width})'.format(cols=cols, width=col_width) if isinstance(cols, int) else cols
            elif isinstance(row_height, str):
                settings['grid-template-columns'] = col_width
            else:
                settings['grid-template-columns'] = " ".join(col_width)
        if alignment is not None:
            settings['align-items'] = alignment
        if justification is not None:
            settings['justify-items'] = justification
        if row_gap is not None:
            settings['row-gap'] = row_gap
        if row_gap is not None:
            settings['column-gap'] = col_gap
        return settings
    def get_layout_styles(self):
        return self.get_grid_styles(
            rows=self.rows, cols=self.cols,
            alignment=self.alignment, justification=self.justification,
            row_gap=self.row_gaps, col_gap=self.col_gaps,
            row_height=self.row_height, col_width=self.col_width
        )

class FlexItem(LayoutItem):
    def __init__(self,
                 item,
                 order=None, grow=None,
                 shrink=None, basis=None,
                 alignment=None,
                 **attrs
                 ):
        super().__init__(item, **attrs)
        self.order = order
        self.grow = grow
        self.shrink = shrink
        self.basis = basis
        self.alignment = alignment
    @classmethod
    def get_flex_styles(cls,
                        order=None, grow=None,
                        shrink=None, basis=None,
                        alignment=None
                        ):
        settings = {}
        if order is not None:
            settings['flex-order'] = order
        if grow is not None:
            settings['flex-grow'] = grow
        if shrink is not None:
            settings['flex-shrink'] = shrink
        if basis is not None:
            settings['flex-basis'] = basis
        if alignment is not None:
            settings['align-self'] = alignment
        return settings
    def get_layout_styles(self):
        return self.get_flex_styles(
            order=self.order, grow=self.grow,
            shrink=self.shrink, basis=self.basis,
            alignment=self.alignment
        )
class Flex(Layout):
    Item = FlexItem
    def __init__(self,
                 elements,
                 direction=None, wrap=None,
                 alignment=None, justification=None,
                 content_alignment=None,
                 **attrs
                 ):
        super().__init__(elements, **attrs)
        self.direction = direction
        self.wrap = wrap
        self.content_alignment = content_alignment
        self.alignment = alignment
        self.justification = justification
    @classmethod
    def get_flex_styles(cls,
                        direction=None, wrap=None,
                        alignment=None, justification=None,
                        content_alignment=None
                        ):
        settings = {'display': 'flex'}
        if direction is not None:
            settings['flex-direction'] = direction
        if wrap is not None:
            settings['flex-wrap'] = wrap
        if alignment is not None:
            settings['align-items'] = alignment
        if justification is not None:
            settings['justify-items'] = justification
        if content_alignment is not None:
            settings['align-content'] = content_alignment
        return settings
    def get_layout_styles(self):
        return self.get_flex_styles(
            direction=self.direction, wrap=self.wrap,
            alignment=self.alignment, justification=self.justification,
            content_alignment=self.content_alignment
        )

#endregion