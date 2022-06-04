
from .HTML import HTML, CSS
from .BootstrapEnums import *

__all__ = ["Bootstrap"]
__reload_hook__ = [".HTML"]

class BootstrapBase(HTML):
    # Bootstrap components
    @classmethod
    def _add_cls(kls, base_cls, cls):
        if base_cls is None:
            base_cls = cls
        else:
            if isinstance(base_cls, str):
                base_cls = base_cls.split()
            try:
                base_cls = list(base_cls)
            except TypeError:
                base_cls = [str(base_cls)]
            if isinstance(cls, str):
                cls = cls.split()
            try:
                cls = list(cls)
            except TypeError:
                cls = [str(cls)]
            for c in cls:
                if c not in base_cls:
                    base_cls.append(c)
        return base_cls
    @classmethod
    def _manage_cls(kls, obj, cls, variant):
        added_cls = obj.cls
        if variant is None:
            variant = obj.base_style
        if variant is not None:
            if isinstance(variant, (str, BootstrapBase.Variant)):
                variant = [variant]
            added_cls = [obj.cls, *(obj.cls + "-" + str(v) for v in variant)]
        return BootstrapBase._add_cls(cls, added_cls)
    class DivComponent(HTML.Div):
        cls = None
        base_style = None
        def __init__(self, *elems, variant=None, cls=None, **attrs):
            cls = BootstrapBase._manage_cls(self, cls, variant)
            super().__init__(*elems, cls=cls, **attrs)
    class SpanComponent(HTML.Span):
        cls = None
        base_style = None
        def __init__(self, *elems, variant=None, cls=None, **attrs):
            cls = BootstrapBase._manage_cls(self, cls, variant)
            super().__init__(*elems, cls=cls, **attrs)
    class ListComponent(HTML.List):
        cls = None
        base_style = None
        def __init__(self, *elems, variant=None, cls=None, **attrs):
            cls = BootstrapBase._manage_cls(self, cls, variant)
            super().__init__(*elems, cls=cls, **attrs)
    class ListItemComponent(HTML.ListItem):
        cls = None
        base_style = None
        def __init__(self, *elems, variant=None, cls=None, **attrs):
            cls = BootstrapBase._manage_cls(self, cls, variant)
            super().__init__(*elems, cls=cls, **attrs)
    class ItalicComponent(HTML.Italic):
        cls = None
        base_style = None
        def __init__(self, *elems, variant=None, cls=None, **attrs):
            cls = BootstrapBase._manage_cls(self, cls, variant)
            super().__init__(*elems, cls=cls, **attrs)
    class TableComponent(HTML.Table):
        cls = None
        def __init__(self, *elems, variant=None, cls=None, **attrs):
            cls = BootstrapBase._manage_cls(self, cls, variant)
            super().__init__(*elems, cls=cls, **attrs)
    class ButtonComponent(HTML.Button):
        cls = None
        def __init__(self, *elems, variant=None, cls=None, **attrs):
            cls = BootstrapBase._manage_cls(self, cls, variant)
            super().__init__(*elems, cls=cls, **attrs)
    class AnchorComponent(HTML.Anchor):
        cls = None
        def __init__(self, *elems, variant=None, cls=None, **attrs):
            cls = BootstrapBase._manage_cls(self, cls, variant)
            super().__init__(*elems, cls=cls, **attrs)

    class BoostrapIcon(ItalicComponent):
        cls='bi'
        def __init__(self, icon_name, cls=None, **attrs):
            # cls = BootstrapBase._manage_cls(self, cls, icon_name)
            super().__init__(" ", variant=icon_name, cls=cls, **attrs)
    class FontAwesomeIcon(SpanComponent):
        cls='fa'
        def __init__(self, icon_name, **attrs):
            super().__init__(" ", variant=icon_name, **attrs)
    class GlyphIcon(SpanComponent):
        cls='glyphicon'
        def __init__(self, icon_name, **attrs):
            super().__init__(" ", variant=icon_name, **attrs)

    class Col(HTML.Div):
        cls='col'
        base_size=None
        base_width=None
        def __init__(self, *elems, width=None, size=None, cls=None, **attrs):
            added_cls = self.cls
            if size is None:
                size = self.base_size
            if size is not None:
                added_cls += '-'+str(size)
            if width is None:
                width = self.base_width
            if width is not None:
                added_cls += '-'+str(width)
            cls = BootstrapBase._add_cls(cls, added_cls)
            super().__init__(*elems, cls=cls, **attrs)
        def __repr__(self):
            return "{}({})".format(type(self).__name__, self.elems)
    class Row(DivComponent):
        cls = 'row'
        def __init__(self, *cols, item_attributes=None, **attrs):
            if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
                cols = cols[0]
            if item_attributes is None:
                item_attributes = {}
            super().__init__(
                [BootstrapBase.Col(x, **item_attributes) if not isinstance(x, BootstrapBase.Col) else x for x in cols],
                **attrs
            )
    class Container(DivComponent): cls='container'
    @staticmethod
    def Grid(rows, row_attributes=None, item_attributes=None, auto_size=False, **attrs):
        if row_attributes is None:
            row_attributes = {}
        if item_attributes is None:
            item_attributes = {}
        if auto_size:
            # most naive approach...
            width = round(100/max(len(r) for r in rows))
            item_attributes['style'] = CSS(width=str(width)+'%')
            attrs['style'] = CSS(width='100%')
        return BootstrapBase.Container(
            [BootstrapBase.Row(r, item_attributes=item_attributes, **row_attributes) for r in rows],
            **attrs
        )

    class Alert(DivComponent): cls='alert'
    class Breadcrumb(ListComponent): cls = ['breadcrumb']
    class BreadcrumbItem(ListItemComponent): cls = ['breadcrumb-item']
    class ListGroup(ListComponent): cls='list-group'
    class ListGroupItem(ListItemComponent): cls='list-group-item'

    class ButtonGroup(DivComponent):
        cls = 'btn-group'
    class Button(ButtonComponent):
        cls = 'btn'
        base_style = 'primary'
    class CloseButton(ButtonComponent):
        cls = 'btn-close'
    class LinkButton(AnchorComponent):
        cls = 'btn'
        base_style = 'default'
    class Table(TableComponent):
        cls = 'table'
        base_style = 'hover'

    class Label(SpanComponent):
        cls = 'label'
        base_style = 'default'
    class Badge(SpanComponent): cls = 'badge'

    Class = SemanticClass
    Variant = SemanticVariant

class Bootstrap3(BootstrapBase):
    Icon = BootstrapBase.GlyphIcon

    class Jumbotron(BootstrapBase.DivComponent): cls='jumbotron'

    class PanelBody(BootstrapBase.DivComponent): cls='panel-body'
    class PanelHeader(BootstrapBase.DivComponent): cls='panel-heading'
    class Panel(BootstrapBase.DivComponent):
        cls='panel'
        base_style='default'
        def __init__(self, *elems, header=None, **attrs):
            if header is None:
                super().__init__(*elems, **attrs)
            else:
                header = Bootstrap3.PanelHeader(header)
                super().__init__(header, Bootstrap3.PanelBody(*elems), **attrs)

class Bootstrap4(BootstrapBase):
    Icon = BootstrapBase.FontAwesomeIcon

    class Jumbotron(BootstrapBase.DivComponent): cls='jumbotron'

    class Pill(BootstrapBase.SpanComponent): cls = ['badge', 'badge-pill']

    class Collapse(BootstrapBase.DivComponent): cls='collapse'

    class CardBody(BootstrapBase.DivComponent): cls='card-body'
    class CardHeader(BootstrapBase.DivComponent): cls='card-header'
    class CardFooter(BootstrapBase.DivComponent): cls='card-footer'
    class CardImage(BootstrapBase.DivComponent): cls='card-img-top'
    class Card(BootstrapBase.DivComponent):
        cls='card'
        base_style=None
        def __init__(self, *elems, header=None, **attrs):
            if header is None:
                super().__init__(*elems, **attrs)
            else:
                header = Bootstrap4.CardHeader(header)
                super().__init__(header, Bootstrap4.CardBody(*elems), **attrs)

class Bootstrap5(BootstrapBase):
    Icon = BootstrapBase.BoostrapIcon

    class Pill(BootstrapBase.SpanComponent): cls = ['badge', 'rounded-pill']

    class Accordion(BootstrapBase.DivComponent): cls='accordion'
    class AccordionItem(BootstrapBase.DivComponent): cls='accordion-item'
    class AccordionCollapse(BootstrapBase.DivComponent): cls='accordion-collapse'
    class AccordionBody(BootstrapBase.DivComponent): cls='accordion-body'

    class Carousel(BootstrapBase.DivComponent): cls='carousel'
    class CarouselInner(BootstrapBase.DivComponent): cls='carousel-inner'
    class CarouselItem(BootstrapBase.DivComponent): cls='carousel-item'

    class Collapse(BootstrapBase.DivComponent): cls='collapse'

    class CardBody(BootstrapBase.DivComponent): cls='card-body'
    class CardHeader(BootstrapBase.DivComponent): cls='card-header'
    class CardFooter(BootstrapBase.DivComponent): cls='card-footer'
    class CardImage(BootstrapBase.DivComponent): cls='card-img-top'
    class Card(BootstrapBase.DivComponent):
        cls='card'
        base_style=None
        def __init__(self, *elems, header=None, **attrs):
            if header is None:
                super().__init__(*elems, **attrs)
            else:
                header = Bootstrap5.CardHeader(header)
                super().__init__(header, Bootstrap5.CardBody(*elems), **attrs)

Bootstrap = Bootstrap5