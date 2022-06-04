
from .HTML import HTML, CSS
from .Bootstrap import Bootstrap
from .HTMLWidgets import ActiveHTMLWrapper, HTMLWidgets
from .BootstrapWidgets import BootstrapWidgets
from .WidgetTools import JupyterAPIs, DefaultOutputArea

import functools

__all__ = [ "JHTML" ]
__reload_hook__ = [".HTML", ".HTMLWidgets", ".Bootstrap", ".BootstrapWidgets", ".WidgetTools"]


class JHTML:
    """
    Provides dispatchers to either pure HTML components or Widget components based on whether interactivity
    is required or not
    """
    manage_class = HTML.manage_class
    manage_style = HTML.manage_styles
    extract_styles = HTML.extract_styles
    manage_attrs = HTML.manage_attrs
    @classmethod
    def load(cls, overwrite=False):
        from IPython.core.display import display

        elems = [
            HTMLWidgets.load(overwrite=overwrite)
            # BootstrapWidgets.load()
        ]
        display(*(e for e in elems if e is not None))

    APIs = JupyterAPIs
    DefaultOutputArea = DefaultOutputArea

    def __init__(self, context=None,
                 include_bootstrap=False,
                 expose_classes=False,
                 output_pane=True,
                 callbacks=None,
                 widgets=None
                 ):
        self._context = context
        self._additions = set()
        self._include_boostrap = include_bootstrap
        self.expose_classes = expose_classes
        self._callbacks = callbacks
        self._widgets = widgets
        self._output_pane = (
            DefaultOutputArea() if output_pane is True
            else DefaultOutputArea(output_pane) if output_pane is not None
            else output_pane
        )

    def _get_frame_vars(self):
        import inspect
        frame = inspect.currentframe()
        parent = frame.f_back.f_back.f_back
        # print(parent.f_locals)
        return parent.f_locals
    def insert_vars(self):
        globs = self._context
        if globs is None:
            globs = self._get_frame_vars()
        cls = type(self)
        for x in cls.__dict__.keys():
            if not x.startswith("_"):
                globs[x] = getattr(cls, x)
                self._additions.add(x)
        if self._include_boostrap:
            for x in cls.Bootstrap.__dict__.keys():
                if not x.startswith("_"):
                    globs[x] = getattr(cls.Bootstrap, x)
                    self._additions.add(x)

    def wrap_callbacks(self, c):
        if self._output_pane is None:
            callbacks = c
        else:
            callbacks = {}
            for k,v in c.items():
                @functools.wraps(v)
                def callback(*args, self=self, **kwargs):
                    with self._output_pane:
                        v(*args, **kwargs)
                callbacks[k] = callback
        return callbacks

    _callback_stack = []
    _widget_stack = []
    def __enter__(self):
        """
        To make writing HTML interactively a bit nicer

        :return:
        :rtype:
        """

        if self.expose_classes:
            self.insert_vars()
        if self._callbacks is not None:
            cls = type(self)
            self._callback_stack.append(cls.callbacks.copy())
            cls.callbacks = self.wrap_callbacks(self._callbacks)
        if self._output_pane is not None:
            self._output_pane.__enter__()
            if self._widgets is None:
                self._widgets = {'out':self._output_pane.obj}
            elif 'out' not in self._widgets:
                self._widgets['out'] = self._output_pane.obj
            else:
                self._widgets['default_output'] = self._output_pane.obj
        if self._widgets is not None:
            cls = type(self)
            self._widget_stack.append(cls.widgets.copy())
            cls.widgets = self._widgets

        return self
    @property
    def out(self):
        return self._output_pane.obj

    def prune_vars(self):
        globs = self._context
        if globs is None:
            globs = self._get_frame_vars()
        for x in self._additions:
            try:
                del globs[x]
            except KeyError:
                pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.expose_classes:
            self.prune_vars()
        if self._output_pane is not None:
            self._output_pane.__exit__(exc_type, exc_val, exc_tb)
        if self._callbacks is not None:
            cls = type(self)
            cls.callbacks = self._callback_stack.pop()
        if self._widgets is not None:
            cls = type(self)
            cls.widgets = self._widget_stack.pop()

    callbacks = {}
    @classmethod
    def parse_handlers(cls, handler_string):
        handlers = {}
        for x in handler_string.split(";"):
            k,v = x.split(":")
            k = k.strip()
            v = v.strip()
            if v in cls.callbacks:
                v = cls.callbacks[v]
            if isinstance(v, str):
                v = globals()[v]
            handlers[k] = v

        return handlers

    widgets = {} # should this be a weakset...?
    @classmethod
    def parse_widget(cls, uuid):
        if uuid in cls.widgets:
            return cls.widgets[uuid]
        else:        
            widgets = JupyterAPIs.get_widgets_api().Widget.widgets.values()
            map = {w.model_id: w for w in widgets}
            return map[uuid]
    @classmethod
    def convert(cls, etree, strip=True, converter=None, **extra_attrs):
        if converter is None:
            converter = cls.convert

        base = HTML.convert(etree, strip=strip, converter=converter, **extra_attrs)
        if base.tag.lower() == "jwidget":
            base = cls.parse_widget(base.attrs['id'])
        else:
            attrs = base.attrs
            handlers = cls.parse_handlers(attrs['event-handlers']) if 'event-handlers' in attrs else None
            track_value = attrs['track-value'].lower() == 'true' if 'track-value' in attrs else False
            dynamic = attrs['dynamic'].lower() == 'true' if 'dynamic' in attrs else False
            for k in ['event-handlers', 'track-value', 'dynamic']:
                try:
                    del attrs[k]
                except KeyError:
                    ...
            base.attrs = attrs
            if not dynamic:
                dynamic = track_value or handlers is not None
            if not dynamic:
                Widget = JupyterAPIs.get_widgets_api().Widget
                dynamic = any(isinstance(x, (ActiveHTMLWrapper, Widget)) for x in base.elems)
            if dynamic:
                base = HTMLWidgets.from_HTML(base, track_value=track_value, event_handlers=handlers)

        return base

    @classmethod
    def parse(cls, src, event_handlers=None, dynamic=None, track_value=None, strict=True, **attrs):
        base = HTML.parse(src, strict=strict, converter=cls.convert)
        _debugPrint = False if '_debugPrint' not in attrs else attrs['_debugPrint']
        trackInput = False if 'trackInput' not in attrs else attrs['trackInput']
        if isinstance(base, HTML.XMLElement) and (
                event_handlers is not None
                or track_value is True
                or trackInput is True
                or _debugPrint is True
                or dynamic is True
        ):
            base = HTMLWidgets.from_HTML(base,
                                        event_handlers=event_handlers,
                                        track_value=track_value,
                                        _debugPrint=_debugPrint,
                                        **attrs
                                        )

        return base

    @classmethod
    def _check_widg(cls, elems):
        Widget = JupyterAPIs.get_widgets_api().Widget
        if isinstance(elems, (list, tuple)) and len(elems) == 0:
            return False
        elif isinstance(elems, (ActiveHTMLWrapper, Widget)) or hasattr(elems, 'to_widget'):
            return True
        else:
            return any(
                cls._check_widg(e) if isinstance(e, (list, tuple)) else
                 (isinstance(e, (ActiveHTMLWrapper, Widget)) or hasattr(e, 'to_widget'))
                for e in elems
            )

    @classmethod
    def _resolve_source(jhtml, plain, widget, *elems,
                        event_handlers=None, dynamic=None,
                        track_value=None, trackInput=None, _debugPrint=None,
                        javascript_handles=None, oninitialize=None,
                        **attrs):
        if (
            event_handlers is not None
            or track_value is True
            or trackInput is True
            or _debugPrint is True
            or dynamic is True
            or javascript_handles is not None
            or oninitialize is not None
        ):
            return widget
        else:
            if jhtml._check_widg(elems):
                return widget
            else:
                return plain
    @classmethod
    def _dispatch(jhtml, plain, widget, *elements, event_handlers=None, dynamic=None, **styles):
        src = jhtml._resolve_source(plain, widget, *elements, event_handlers=event_handlers, dynamic=dynamic, **styles)
        if src is plain:
            return src(*elements, activator=widget.from_HTML, **styles)
        else:
            return src(*elements, event_handlers=event_handlers, **styles)

    def dispatcher(fn):
        name = fn.__name__
        @functools.wraps(fn)
        def dispatcher(jhtml, *elements, **attrs):
            return jhtml._dispatch(getattr(HTML, name), getattr(HTMLWidgets, name), *elements, **attrs)
        return dispatcher

    @classmethod
    @dispatcher
    def Abbr(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Address(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Anchor(jhtml, *elements, **styles): ...
    A = Anchor
    @classmethod
    @dispatcher
    def Area(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Article(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Aside(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Audio(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def B(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Base(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Bdi(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Bdo(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Blockquote(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Body(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Bold(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Br(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Button(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Canvas(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Caption(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Cite(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Code(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Col(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Colgroup(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Data(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Datalist(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Dd(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Del(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Details(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Dfn(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Dialog(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Div(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Dl(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Dt(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Em(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Embed(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Fieldset(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Figcaption(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Figure(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Footer(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Form(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Head(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Header(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Heading(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Hr(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Html(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Iframe(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Image(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Img(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Input(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Ins(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Italic(jhtml, *elements, **styles): ...
    I = Italic
    @classmethod
    @dispatcher
    def Kbd(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Label(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Legend(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Link(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def List(jhtml, *elements, **styles): ...
    Ul = List
    @classmethod
    @dispatcher
    def ListItem(jhtml, *elements, **styles): ...
    Li = ListItem
    @classmethod
    @dispatcher
    def Main(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Map(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Mark(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Meta(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Meter(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Nav(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Noscript(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def NumberedList(jhtml, *elements, **styles): ...
    Ol = NumberedList
    @classmethod
    @dispatcher
    def Object(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Optgroup(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Option(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Output(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Param(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Picture(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Pre(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Progress(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Q(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Rp(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Rt(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Ruby(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def S(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Samp(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Script(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Section(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Select(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Small(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Source(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Span(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Strong(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Style(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Sub(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def SubHeading(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def SubsubHeading(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def SubsubsubHeading(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def SubHeading5(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def SubHeading6(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Summary(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Sup(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Svg(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Table(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def TableBody(jhtml, *elements, **styles): ...
    Tbody = TableBody
    @classmethod
    @dispatcher
    def TableFooter(jhtml, *elements, **styles): ...
    Tfoot = TableFooter
    @classmethod
    @dispatcher
    def TableHeader(jhtml, *elements, **styles): ...
    Thead = TableHeader
    @classmethod
    @dispatcher
    def TableHeading(jhtml, *elements, **styles): ...
    Th = TableHeading
    @classmethod
    @dispatcher
    def TableItem(jhtml, *elements, **styles): ...
    Td = TableItem
    @classmethod
    @dispatcher
    def TableRow(jhtml, *elements, **styles): ...
    Tr = TableRow
    @classmethod
    @dispatcher
    def Template(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Text(jhtml, *elements, **styles): ...
    P = Text
    @classmethod
    @dispatcher
    def Textarea(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Time(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Title(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Track(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def U(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Var(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Video(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Wbr(jhtml, *elements, **styles): ...

    del dispatcher

    OutputArea = HTMLWidgets.OutputArea
    JavascriptAPI = HTMLWidgets.JavascriptAPI

    class Bootstrap:

        Class = Bootstrap.Class
        Variant = Bootstrap.Variant

        @classmethod
        def _dispatch(jhtml, *elems, **attrs):
            return JHTML._dispatch(*elems, **attrs)
        def dispatcher(fn):
            name = fn.__name__
            @functools.wraps(fn)
            def dispatcher(boots, *elements, **attrs):
                return boots._dispatch(getattr(Bootstrap, name), getattr(BootstrapWidgets, name), *elements, **attrs)
            return dispatcher

        @classmethod
        @dispatcher
        def Icon(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Alert(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Badge(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def PanelBody(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def PanelHeader(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Panel(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def CardBody(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def CardHeader(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def CardFooter(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def CardImage(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Card(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Jumbotron(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Col(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Row(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Container(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Button(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def ButtonGroup(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def LinkButton(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Table(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def ListGroup(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def ListGroupItem(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def FontAwesomeIcon(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def GlyphIcon(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Label(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Pill(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def ListComponent(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def ListItemComponent(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Breadcrumb(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def BreadcrumbItem(boots, *elements, **styles): ...


        @classmethod
        @dispatcher
        def Accordion(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def AccordionItem(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def AccordionCollapse(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def AccordionBody(boots, *elements, **styles): ...

        @classmethod
        @dispatcher
        def Carousel(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def CarouselInner(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def CarouselItem(boots, *elements, **styles): ...

        @classmethod
        @dispatcher
        def Collapse(boots, *elements, **styles): ...

        del dispatcher

    class Styled:
        def __init__(self, base, **attrs):
            self.attrs = attrs
            self.base = base
        def __call__(self, *args, **kwargs):
            return self.base(*args, **dict(self.attrs, **kwargs))
        def __repr__(self):
            return "{}({}, {})".format(
                type(self).__name__,
                self.base,
                self.attrs
            )
    class Compound:
        def __init__(self, *wrappers):
            self.base = wrappers[-1]
            self.classes = wrappers[:-1]
        @staticmethod
        def destructure_wrapper(wrapper):
            try:
                lw = len(wrapper)
            except TypeError:
                lw = 0
            if lw == 2 and isinstance(wrapper[0], str):
                name, wrapper = wrapper
            else:
                name = None
            return name, wrapper
        class CompoundWrapperData:
            __slots__ = ["list", "dict"]
            def __init__(self, list, dict):
                self.list = list
                self.dict = dict
            def __getitem__(self, item):
                if isinstance(item, str):
                    return self.dict[item]
                else:
                    return self.list[item]
        def __call__(self, *args, **kwargs):
            cache = []
            names = {}
            cwd = self.CompoundWrapperData(cache, names)
            n,w = self.destructure_wrapper(self.base)
            base = w(*args, **kwargs)
            if n is not None:
                names[n] = base
            cache.append(base)
            base.compound_wrapper_data = cwd
            for c in reversed(self.classes):
                n, c = self.destructure_wrapper(c)
                base = c(base)
                if n is not None:
                    names[n] = base
                cache.append(base)
                base.compound_wrapper_data = cwd
            return base