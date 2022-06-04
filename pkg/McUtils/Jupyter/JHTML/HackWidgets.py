
# OLD AND DEPRECATED
class JHTMLShadowDOMElement:
    """
    Provides a shadow DOM tree that can makes it easier to update
    chunks of a tree constructed from JupyterHTMLWrappers
    """
    def __init__(self, widg, wrapper=None, parent=None):
        self.widg = widg
        self.wrapper = wrapper
        self.parent = parent.dom if hasattr(parent, 'dom') else parent
        self._tree_cache = None

    def link(self, dom_el, recursive=False):
        dom_el.on_update = self.Refresher(self)
        dom_el.shadow = self
        if recursive:
            for c in dom_el:
                if isinstance(c, HTML.XMLElement):
                    self.link(c, recursive=True)
    @property
    def tree(self):
        return self.to_tree()
    def to_tree(self, refresh=False):
        if refresh:
            self._invalidate_cache()
        if self._tree_cache is None:
            if hasattr(self.widg, 'children'):
                children = []
                for c in self.widg.children:
                    if hasattr(c, 'dom'):
                        el = c.dom.to_tree()
                        if el is not None:
                            children.append(el)
                    elif hasattr(c, 'value'):
                        shadow = type(self)(c, wrapper=None, parent=self)
                        # self.link(elem)
                        # elem.shadow = self
                        # elem.on_update = self.Refresher(self, which=len(children))
                        el = shadow.to_tree()
                        if el is not None:
                            children.append(el)
                attrs = {}
                if len(self.widg._dom_classes) > 0:
                    attrs['cls'] = self.widg._dom_classes
                dom_el = HTML.Div(children, **attrs)
            elif hasattr(self.widg, 'value'):
                dom_el = HTML.parse(self.widg.value, strict=False)
                self.link(dom_el, recursive=True)
            else:
                dom_el = None
            self._tree_cache = dom_el
            if dom_el is not None:
                self.link(dom_el, recursive=False)
                # dom_el.shadow = self
                # dom_el.on_update = self.Refresher(self)
        return self._tree_cache
    def _invalidate_cache(self):
        self._tree_cache = None
        if self.parent is not None:
            self.parent._invalidate_cache()

    class Refresher:
        def __init__(self, dom_el, which=None):
            self.dom_el = dom_el
            self.which = which
        def __call__(self, el, key, value):
            if self.which is not None:
                self.dom_el.refresh_child(el, key, value)
            else:
                self.dom_el.refresh(el, key, value)
    def refresh(self, el, key, value):
        if hasattr(self.widg, 'children'):
            if isinstance(key, str):
                if key=='elements':
                    raise NotImplementedError("haven't handle full elem updates yet")
                elif key == 'attributes':
                    k = list(value.keys())
                    if 'class' in value:
                        classes = value['class']
                        if isinstance(classes, str) or hasattr(classes, 'tostring'):
                            classes = [classes]
                        self.widg._dom_classes = [str(c) for c in classes]
                        k.remove('class')
                    if len(k) > 0:
                        raise NotImplementedError("haven't handled attr updates to keys {} yet".format(k))
                elif key == 'class':
                    if isinstance(value, str) or hasattr(value, 'tostring'):
                        value = [value]
                    self.widg._dom_classes = [str(c) for c in value]
                else:
                    raise NotImplementedError("can't update attribute {}")
            else:
                # print(self.widg.children[key])
                self.widg.children[key].value = value.tostring()
                self.widg.children[key].dom._invalidate_cache()
                # print(self.widg.children[key])
        else:
            self.widg.value = el.tostring()
        self._invalidate_cache()
    def refresh_child(self, el, key, value, which):
        raise NotImplementedError("ugh")
        subwidg = self.widg.children[which]
        if hasattr(subwidg, 'children'):
            raise NotImplementedError("ugh")
        else:
            subwidg.value = el.tostring()
        self._invalidate_cache()

    def get_parent(self, n):
        for i in range(n):
            self = self.parent
        return self

    def find(self, path, find_element=True):
        return self.tree.find(path, find_element=find_element)
    def findall(self, path, find_element=True):
        return self.tree.findall(path, find_element=find_element)
    def iterfind(self, path, find_element=True):
        return self.tree.iterfind(path, find_element=find_element)
    def find_by_id(self, id, mode='first', parent=None, find_element=True):
        return self.tree.find_by_id(id, mode=mode, parent=parent, find_element=find_element)
    def __getitem__(self, item):
        return self.to_tree()[item]
class JupyterHTMLWrapper:
    """
    Provides simple access to Jupyter display utilities
    """

    reset_attrs = [
        'background',
        'color',
        'font',
        'font-family'
    ]
    @classmethod
    def load_styles(cls):
        """
        Embeds widget restyle definitions into the active notebook

        :return:
        :rtype:
        """
        from IPython.core.display import HTML as IPyHTML
        custom_styles = [
            CSS.construct(
                '.jhtml-reset .jhtml-html',
                '.jhtml-reset .jhtml-parent',
                display='inline-block'
            )
        ]
        return IPyHTML(HTML.Style("\n".join(s.tostring() for s in custom_styles)).tostring())
    stripped_classes = ['lm-Widget', 'p-Widget', 'lm-Panel', 'p-Panel', 'jupyter-widgets', 'widget-container', 'widget-box', 'widget-html']
    protecting_classes = ['widget-slider']
    stripped_unprotected = ['widget-inline-hbox']
    @classmethod
    def get_class_stripper_js(cls):
        import uuid
        this = "stripper"+str(uuid.uuid1()).replace("-", "")[:8]
        script = """
        var {id} = document.getElementById('{id}');
        var {id}siblings = {id}.parentNode.parentNode.parentNode.childNodes;
        function {id}strip(node) {{
           if (typeof node.classList != 'undefined') {{
               node.classList.remove({stripped_classes});
               // ipython
               if (!{protecting_classes}.some(e=>node.classList.contains(e))) {{
                 node.classList.remove({stripped_unprotected});
               }}
           }}
           if (typeof node.classList != 'undefined') {{
               for (const child of node.childNodes) {{
                 {id}strip(child);
               }}
           }}
           // if (node.tagName === 'DIV') {{ 
           //    // something about chained conditions is giving as syntax error...
           //    if (node.classList.length == 1) {{
           //       if (node.classList[0] == 'jhtml-reset') {{
           //          node.replaceWith(...node.childNodes);
           //       }}
           //    }}
           // }}
        }}
        for (const sibling of {id}siblings) {{
            {id}strip(sibling);
        }}
        {id}.parentNode.parentNode.remove();
                    """.format(id=this,
                               stripped_classes=", ".join("'{}'".format(s) for s in cls.stripped_classes),
                               protecting_classes="[" +", ".join("'{}'".format(s) for s in cls.protecting_classes)+"]",
                               stripped_unprotected=", ".join("'{}'".format(s) for s in cls.stripped_unprotected)
                               )
        return script, this
    @classmethod
    def get_class_stripper(cls):
        stripper, this = cls.get_class_stripper_js()
        return JupyterAPIs.get_display_api().HTML('<script id="{this}">{script}</script>'.format(script=stripper, this=this))

    cls = None #for easy overloads later
    tag = None
    layout = None
    container = False # overloaded later
    def __init__(self, *elements,
                 tag=None,
                 event_handlers=None,
                 layout=None,
                 extra_classes=None,
                 cls=None,
                 debug_pane=None,
                 **styles
                 ):
        if len(elements) == 1 and isinstance(elements[0], (list, tuple)):
            elements = elements[0]
        self.elements = elements
        self.event_handlers = event_handlers
        if event_handlers is not None and not hasattr(self.event_handlers, 'items'):
            self.event_handlers = {
                'click':self.event_handlers,
                'mouseenter':self.event_handlers,
                'mouseleave':self.event_handlers,
                'keydown':self.event_handlers,
                'keyup':self.event_handlers
            }
        self.tag = tag if tag is not None else self.tag
        self.layout = self.layout if layout is None else layout
        self.styles = styles
        if extra_classes is None:
            extra_classes = self.cls
        if cls is not None:
            if hasattr(cls, 'value'):
                cls = cls.value
            if isinstance(cls, str):
                cls = cls.split()

            if extra_classes is None:
                extra_classes = cls
            else:
                if hasattr(extra_classes, 'value'):
                    extra_classes = extra_classes.value
                if isinstance(extra_classes, str):
                    extra_classes = extra_classes.split()
                extra_classes = list(extra_classes) + list(cls)
        if hasattr(extra_classes, 'value'):
            extra_classes = extra_classes.value
        if isinstance(extra_classes, str):
            extra_classes = extra_classes.split()
        if extra_classes is not None:
            extra_classes = [e.value if hasattr(e, 'value') else e for e in extra_classes]
        self.extra_classes = extra_classes

        self._parents = weakref.WeakSet()
        self._widget_cache = None
        if debug_pane is None:
            debug_pane = DefaultOutputArea.get_default()
        self.debug_pane = debug_pane

    @staticmethod
    def _handle_event(e, event_handlers, self, widg):
        try:
            handler = event_handlers[e['type']]
        except KeyError:
            pass
        else:
            handler(e, self, widg)
    def _event_handler(self, widg):
        def handler(e, output=self.debug_pane):
            with output:
                return self._handle_event(e, self.event_handlers, self, widg)
        return handler


    inherited_classes = ['d-inline', 'd-flex', 'd-inline-block', 'd-block', 'd-none'] # explicitly Bootstrap oriented for now...
    inherited_class_tag_map = { # this is _unlikely_ to be overwritten...
        'span':['d-inline-block'],
        None:['d-inline-block']
    }
    inherited_class_cls_map = { # again _unlikely_ to be overwritten
        'btn': ['d-inline-block']
    }
    def _convert(self, x, parent=None):
        widget_classes = ['jhtml-reset', 'jhtml-html']
        if isinstance(x, (HTML.XMLElement, HTML.ElementModifier)):
            # to get widget formatting right for container classes...
            try:
                mapped_widget = self.base_map[type(x)]
            except KeyError:
                mapped_widget = None
                needs_mods = False
            else:
                try:
                    needs_mods = mapped_widget.container
                except AttributeError:
                    needs_mods = False
                if mapped_widget.cls is None:
                    mapped_widget = type(x)
                if needs_mods:
                    x = x.remove_class(mapped_widget.cls)

            # now we propagate up inherited classes
            try:
                cls = x.attrs['class']
            except (AttributeError, KeyError):
                cls = None
            if cls is None:
                cls = []
            elif isinstance(cls, str):
                cls = cls.split()
            cls = list(cls)

            tag = x.tag.lower()
            if tag in self.inherited_class_tag_map:
                cls = list(self.inherited_class_tag_map[tag]) + cls
            for c in cls:
                if c in self.inherited_class_cls_map:
                    cls = list(self.inherited_class_cls_map[c]) + cls

            extras = [e for e in cls if e in self.inherited_classes]
            if len(extras) > 0:
                for e in extras:
                    widget_classes.append(e)
                x = x.remove_class(*extras)

            x = x.tostring()
        else:
            if isinstance(x, str):
                cls = list(self.inherited_class_tag_map[None])
                extras = [e for e in cls if e in self.inherited_classes]
                if len(extras) > 0:
                    for e in extras:
                        widget_classes.append(e)
            mapped_widget = None
            needs_mods = False

        if isinstance(x, str):
            w = JupyterAPIs().widgets_api.HTML(x)
            if needs_mods:
                cls = mapped_widget.cls
                if cls is None:
                    cls = []
                elif hasattr(cls, 'tostring'):
                    cls = cls.tostring
                if isinstance(cls, str):
                    cls = cls.split()
                extra_classes = list(cls)
                for c in extra_classes:
                    if parent.extra_classes is None or c not in parent.extra_classes:
                        # print(c, parent.extra_classes)
                        w.add_class(str(c))
            for c in widget_classes:
                w.add_class(c)
        elif isinstance(x, JupyterHTMLWrapper):
            w = x.to_widget(parent=self)
        else:
            w = x
        return w

    @classmethod
    def manage_styles(cls, styles, validate=True):
        layout_props = {
            'height',
            'width',
            'max_height',
            'max_width',
            'min_height',
            'min_width',
            'visibility',
            'display',
            'overflow',
            'model',
            'border',
            'margin',
            'padding',
            'top',
            'left',
            'bottom',
            'right',
            'order',
            'flex_flow',
            'align_items',
            'flex',
            'align_self',
            'align_content',
            'justify_content',
            'grid_auto_columns',
            'grid_auto_flow',
            'grid_auto_rows',
            'grid_gap',
            'grid_template',
            'grid_row',
            'grid_column'
        }
        style_props = {
            "color",
            "background",
            'font_family',
            'font_size',
            'font_style',
            'font_variant',
            'font_weight',
            'text_color',
            'text_decoration'
        }

        if styles is None:
            styles = {}

        layout_map = {}
        style_map = {}
        for k in styles:
            if k in layout_props:
                layout_map[k] = styles[k]
            elif k in style_props:
                style_map[k] = styles[k]
            elif validate:
                raise ValueError("as designed JupyterLab doesn't support the {} attribute".format(
                    k
                ))
        return {'style':style_map, 'layout':layout_map}

    def _invalidate_cache(self):
        if self._widget_cache is not None:
            self._widget_cache = None
            for p in tuple(self._parents):
                p._invalidate_cache()
                self._parents.remove(p)
    def to_widget(self, parent=None):
        if parent is not None:
            self._parents.add(parent)
        if self._widget_cache is None:
            layout = self.layout
            if isinstance(self.elements[0], (list, tuple)):
                widgets = [
                    [ self._convert(x, parent=self) for x in y ]
                    for y in self.elements
                ]
            else:
                widgets = [self._convert(x, parent=self) for x in self.elements]

            if 'style' in self.styles:
                props = self.manage_styles(self.styles['style'])
            elif len(self.styles) > 0:
                raise ValueError("JupyterLab doesn't support {} attributes on widgets".format(list(self.styles.keys())))
            else:
                props = {}

            if layout is None:
                layout = 'box'
            if layout == 'grid':
                widg = JupyterAPIs().widgets_api.GridBox(widgets, **props)
            elif layout == 'row':
                widg = JupyterAPIs().widgets_api.HBox(widgets, **props)
            elif layout == 'column':
                widg = JupyterAPIs().widgets_api.VBox(widgets, **props)
            else:
                def flatten(x, base):
                    if not isinstance(x, JupyterAPIs().widgets_api.Widget):
                        for y in x:
                            flatten(y, base)
                    else:
                        base.append(x)
                new = []; flatten(widgets, new)
                widg = JupyterAPIs().widgets_api.Box(new, **props)

            from IPython.core.display import display as IPyDisplay
            widg.add_class('jhtml-reset')
            widg.add_class('jhtml-wrapper')
            if self.extra_classes is not None:
                for c in self.extra_classes:
                    widg.add_class(c)
            if self.event_handlers is not None:
                listener = JupyterAPIs().events_api.Event(source=widg, watched_events=list(self.event_handlers.keys()))
                listener.on_dom_event(self._event_handler(widg))
                widg.listener = listener
                widg.add_class('jhtml-event-handler')
                # if self.extra_classes is None or len(self.extra_classes) == 0:
                for child in widg.children:
                    if not hasattr(child, 'listener'):
                        listener = JupyterAPIs().events_api.Event(source=child,
                                                                  watched_events=list(self.event_handlers.keys()))
                        listener.on_dom_event(self._event_handler(child))
                        child.listener = listener
                        child.add_class('jhtml-event-handler')
            html_widget = JupyterAPIs.get_widgets_api().HTML
            if (
                    len(widg.children)==1
                    and isinstance(widg.children[0], html_widget)
            ):
                widg.add_class('jhtml-parent')
                for c in widg.children[0]._dom_classes:
                    if c in self.inherited_classes:
                        widg.add_class(c)

            # for discoverability in callbacks
            widg.dom = JHTMLShadowDOMElement(widg, wrapper=self)
            if hasattr(widg, 'children'):
                for x in widg.children:
                    if not hasattr(x, 'dom'):
                        x.dom = JHTMLShadowDOMElement(x, wrapper=None, parent=widg.dom)
                    else:
                        x.dom.parent = widg.dom
            widg.display = lambda w=widg:self.display_widget(w)
            self._widget_cache = widg
        return self._widget_cache
    @classmethod
    def display_widget(cls, w):
        return JupyterAPIs.get_display_api().display(w, cls.get_class_stripper())
    def display(self):
        return JupyterAPIs.get_display_api().display(self.to_widget(), self.get_class_stripper())
    def _ipython_display_(self):
        return self.display()

    def find(self, path, find_element=True):
        return self.to_widget().dom.find(path, find_element=find_element)
    def findall(self, path, find_element=True):
        return self.to_widget().dom.findall(path, find_element=find_element)
    def iterfind(self, path, find_element=True):
        return self.to_widget().dom.iterfind(path, find_element=find_element)
    def find_by_id(self, id, mode='first', parent=None, find_element=True):
        return self.to_widget().dom.find_by_id(id, mode=mode, parent=parent, find_element=find_element)

    _widget_sources = []
    _base_map = None
    @classmethod
    def load_base_map(cls):
        if cls._base_map is None:
            cls._base_map = {}
            for src in cls._widget_sources:
                for v in src.__dict__.values():
                    if hasattr(v, 'base'):
                        cls._base_map[v.base] = v
        return cls._base_map
    @property
    def base_map(self):
        return self.load_base_map()
class JupyterHTMLWidgets:
    """
    Provides convenience constructors for HTML components
    """
    @classmethod
    def load(cls):
        return JupyterHTMLWrapper.load_styles()

    class WrappedElement(JupyterHTMLWrapper):
        base = None
        container = False
        def __init__(self, *elems, base=None, event_handlers=None, container=None, layout=None, extra_classes=None, **attrs):
            if base is None:
                base = self.base
            self.base = base
            self.container = self.container if container is None else container
            Widget = JupyterAPIs.get_widgets_api().Widget
            if any(isinstance(e, (JupyterHTMLWrapper, Widget)) for e in elems):
                elem = elems
                if hasattr(base, 'cls'):
                    cls = base.cls
                    if cls is None:
                        cls = []
                    elif isinstance(cls, str):
                        cls = cls.split()
                    extras = list(cls)

                    if extra_classes is None:
                        extra_classes = []
                    elif isinstance(extra_classes, str):
                        extra_classes = extra_classes.split()
                    extra_classes = list(extra_classes) + extras

                if not hasattr(base, 'tag') or base.tag != "div":
                    raise ValueError("{} can't be a proper widget base as JupyterLab is designed (i.e. it can't have widgets as children)".format(base))
            else:
                elem = base(*elems, **attrs)
                if self.container and hasattr(base, 'cls'):
                    cls = base.cls
                    if cls is None:
                        cls = []
                    elif isinstance(cls, str):
                        cls = cls.split()
                    extras = list(cls)

                    if extra_classes is None:
                        extra_classes = []
                    elif isinstance(extra_classes, str):
                        extra_classes = extra_classes.split()
                    extra_classes = list(extra_classes) + extras
                    elem = elem.remove_class(*extras)
                attrs = {}
            super().__init__(elem, event_handlers=event_handlers, layout=layout, extra_classes=extra_classes, **attrs)
        def copy(self):
            import copy
            new = copy.copy(self)
            new.elements = [(x.copy() if hasattr(x, 'copy') else x) for x in new.elements]
            new._widget_cache = None
            new._parents = weakref.WeakSet()
            return new
        def add_child_class(self, *cls, copy=True):
            if copy:
                new = self.copy()
                return new.add_child_class(*cls, copy=False)
            else:
                self.elements = [x.add_class(*cls) for x in self.elements]
                return self
        def add_class(self, *cls, copy=True):
            if copy:
                new = self.copy()
                return new.add_class(*cls, copy=False)
            else:
                if self.extra_classes is None:
                    self.extra_classes = [str(x) for x in cls]
                else:
                    self.extra_classes = [str(x) for x in self.extra_classes]
                for c in cls:
                    if c not in self.extra_classes:
                        self.extra_classes.append(str(c))
                self._invalidate_cache()
                return self
        def remove_class(self, *cls, copy=True):
            if copy:
                new = self.copy()
                return new.remove_class(*cls, copy=False)
            else:
                if self.extra_classes is not None:
                    self.extra_classes = [str(x) for x in self.extra_classes]
                for c in cls:
                    try:
                        self.extra_classes.remove(str(c))
                    except ValueError:
                        pass
                self._invalidate_cache()
                return self
        def add_styles(self, copy=True, **sty):
            if copy:
                new = self.copy()
                return new.add_styles(copy=False, **sty)
            else:
                self.elements = [x.add_styles(**sty) for x in self.elements]
                self._invalidate_cache()
                return self
    class ContainerWrapper(WrappedElement): container = True
    class Abbr(WrappedElement): base=HTML.Abbr
    class Address(WrappedElement): base=HTML.Address
    class Anchor(WrappedElement): base=HTML.Anchor
    A = Anchor
    class Area(WrappedElement): base=HTML.Area
    class Article(WrappedElement): base=HTML.Article
    class Aside(WrappedElement): base=HTML.Aside
    class Audio(WrappedElement): base=HTML.Audio
    class B(WrappedElement): base=HTML.B
    class Base(WrappedElement): base=HTML.Base
    class BaseList(WrappedElement): base=HTML.BaseList
    class Bdi(WrappedElement): base=HTML.Bdi
    class Bdo(WrappedElement): base=HTML.Bdo
    class Blockquote(WrappedElement): base=HTML.Blockquote
    class Body(WrappedElement): base=HTML.Body
    class Bold(WrappedElement): base=HTML.Bold
    class Br(WrappedElement): base=HTML.Br
    class Button(WrappedElement): base=HTML.Button
    class Canvas(WrappedElement): base=HTML.Canvas
    class Caption(WrappedElement): base=HTML.Caption
    class Cite(WrappedElement): base=HTML.Cite
    class ClassAdder(WrappedElement): base=HTML.ClassAdder
    class ClassRemover(WrappedElement): base=HTML.ClassRemover
    class Code(WrappedElement): base=HTML.Code
    class Col(WrappedElement): base=HTML.Col
    class Colgroup(WrappedElement): base=HTML.Colgroup
    class Data(WrappedElement): base=HTML.Data
    class Datalist(WrappedElement): base=HTML.Datalist
    class Dd(WrappedElement): base=HTML.Dd
    class Del(WrappedElement): base=HTML.Del
    class Details(WrappedElement): base=HTML.Details
    class Dfn(WrappedElement): base=HTML.Dfn
    class Dialog(WrappedElement): base=HTML.Dialog
    class Div(WrappedElement): base=HTML.Div
    class Dl(WrappedElement): base=HTML.Dl
    class Dt(WrappedElement): base=HTML.Dt
    class ElementModifier(WrappedElement): base=HTML.ElementModifier
    class Em(WrappedElement): base=HTML.Em
    class Embed(WrappedElement): base=HTML.Embed
    class Fieldset(WrappedElement): base=HTML.Fieldset
    class Figcaption(WrappedElement): base=HTML.Figcaption
    class Figure(WrappedElement): base=HTML.Figure
    class Footer(WrappedElement): base=HTML.Footer
    class Form(WrappedElement): base=HTML.Form
    class Head(WrappedElement): base=HTML.Head
    class Header(WrappedElement): base=HTML.Header
    class Heading(WrappedElement): base=HTML.Heading
    class Hr(WrappedElement): base=HTML.Hr
    class Iframe(WrappedElement): base=HTML.Iframe
    class Image(WrappedElement): base=HTML.Image
    class Img(WrappedElement): base=HTML.Img
    class Input(WrappedElement): base=HTML.Input
    class Ins(WrappedElement): base=HTML.Ins
    class Italic(WrappedElement): base=HTML.Italic
    class Kbd(WrappedElement): base=HTML.Kbd
    class Label(WrappedElement): base=HTML.Label
    class Legend(WrappedElement): base=HTML.Legend
    class Li(WrappedElement): base=HTML.Li
    class Link(WrappedElement): base=HTML.Link
    class List(WrappedElement): base=HTML.List
    class ListItem(WrappedElement): base=HTML.ListItem
    class Main(WrappedElement): base=HTML.Main
    class Map(WrappedElement): base=HTML.Map
    class Mark(WrappedElement): base=HTML.Mark
    class Meta(WrappedElement): base=HTML.Meta
    class Meter(WrappedElement): base=HTML.Meter
    class Nav(WrappedElement): base=HTML.Nav
    class Noscript(WrappedElement): base=HTML.Noscript
    class NumberedList(WrappedElement): base=HTML.NumberedList
    class Object(WrappedElement): base=HTML.Object
    class Ol(WrappedElement): base=HTML.Ol
    class Optgroup(WrappedElement): base=HTML.Optgroup
    class Option(WrappedElement): base=HTML.Option
    class Output(WrappedElement): base=HTML.Output
    class P(WrappedElement): base=HTML.P
    class Param(WrappedElement): base=HTML.Param
    class Picture(WrappedElement): base=HTML.Picture
    class Pre(WrappedElement): base=HTML.Pre
    class Progress(WrappedElement): base=HTML.Progress
    class Q(WrappedElement): base=HTML.Q
    class Rp(WrappedElement): base=HTML.Rp
    class Rt(WrappedElement): base=HTML.Rt
    class Ruby(WrappedElement): base=HTML.Ruby
    class S(WrappedElement): base=HTML.S
    class Samp(WrappedElement): base=HTML.Samp
    class Script(WrappedElement): base=HTML.Script
    class Section(WrappedElement): base=HTML.Section
    class Select(WrappedElement): base=HTML.Select
    class Small(WrappedElement): base=HTML.Small
    class Source(WrappedElement): base=HTML.Source
    class Span(WrappedElement): base=HTML.Span
    class Strong(WrappedElement): base=HTML.Strong
    class Style(WrappedElement): base=HTML.Style
    class StyleAdder(WrappedElement): base=HTML.StyleAdder
    class Sub(WrappedElement): base=HTML.Sub
    class SubHeading(WrappedElement): base=HTML.SubHeading
    class SubsubHeading(WrappedElement): base=HTML.SubsubHeading
    class SubsubsubHeading(WrappedElement): base=HTML.SubsubsubHeading
    class SubHeading5(WrappedElement): base=HTML.SubHeading5
    class SubHeading6(WrappedElement): base=HTML.SubHeading6
    class Summary(WrappedElement): base=HTML.Summary
    class Sup(WrappedElement): base=HTML.Sup
    class Svg(WrappedElement): base=HTML.Svg
    class Table(WrappedElement): base=HTML.Table
    class TableBody(WrappedElement): base=HTML.TableBody
    class TableHeading(WrappedElement): base=HTML.TableHeading
    class TableItem(WrappedElement): base=HTML.TableItem
    class TableRow(WrappedElement): base=HTML.TableRow
    class TagElement(WrappedElement): base=HTML.TagElement
    class Tbody(WrappedElement): base=HTML.Tbody
    class Td(WrappedElement): base=HTML.Td
    class Template(WrappedElement): base=HTML.Template
    class Text(WrappedElement): base=HTML.Text
    class Textarea(WrappedElement): base=HTML.Textarea
    class Tfoot(WrappedElement): base=HTML.Tfoot
    class Th(WrappedElement): base=HTML.Th
    class Thead(WrappedElement): base=HTML.Thead
    class Time(WrappedElement): base=HTML.Time
    class Title(WrappedElement): base=HTML.Title
    class Tr(WrappedElement): base=HTML.Tr
    class Track(WrappedElement): base=HTML.Track
    class U(WrappedElement): base=HTML.U
    class Ul(WrappedElement): base=HTML.Ul
    class Var(WrappedElement): base=HTML.Var
    class Video(WrappedElement): base=HTML.Video
    class Wbr(WrappedElement): base=HTML.Wbr

    class OutputArea(JupyterHTMLWrapper):
        def __init__(self, *elements, autoclear=False, event_handlers=None, layout=None, extra_classes=None, cls=None, **styles):
            if len(elements) == 1 and isinstance(elements, (list, tuple)):
                elements = elements[0]
            self.autoclear = autoclear
            self.output = JupyterAPIs.get_widgets_api().Output()
            elements = list(elements) + [self.output]
            super().__init__(*elements, tag='div', event_handlers=event_handlers, layout=layout, extra_classes=extra_classes, cls=cls, **styles)
        def print(self, *args, **kwargs):
            with self:
                print(*args, **kwargs)
        def display(self, *args):
            with self:
                JupyterAPIs().display_api.display(*args)
        def clear(self):
            self.output.clear_output()
        def __enter__(self):
            if self.autoclear:
                self.output.clear_output()
            return self.output.__enter__()
        def __exit__(self, exc_type, exc_val, exc_tb):
            return self.output.__exit__(exc_type, exc_val, exc_tb)
JupyterHTMLWrapper._widget_sources.append(JupyterHTMLWidgets)