from .HTML import HTML
from .HTMLWidgets import ActiveHTMLWrapper, HTMLWidgets
from .Bootstrap import Bootstrap3, Bootstrap4, Bootstrap5

__all__ = ["BootstrapWidgets"]
__reload_hook__ = [".HTML", ".HTMLWidgets", ".Bootstrap"]

class BootstrapWidgetsBase:
    cdn_loader = None
    cdn_css = []
    cdn_js = []
    _cdn_cache = {}
    bootstrap_version = None

    @classmethod
    def load(cls):
        """
        Embeds Bootstrap style definitions into the active notebook

        :return:
        :rtype:
        """
        from IPython.core.display import HTML as IPyHTML
        from urllib.request import urlopen

        if cls.cdn_loader is not None:
            loader = cls.cdn_loader
        else:
            loader = ""
            for sheet in cls.cdn_css:
                if sheet in cls._cdn_cache:
                    styles = cls._cdn_cache[sheet]
                else:
                    styles = urlopen(sheet).read().decode()
                    cls._cdn_cache[sheet] = styles
                loader += "\n<style>"+styles+"</style>"
            for package in cls.cdn_js:
                if package in cls._cdn_cache:
                    js = cls._cdn_cache[package]
                else:
                    js = urlopen(package).read().decode()
                    cls._cdn_cache[package] = js
                loader += "\n<script>"+js+"</style>"
        loader_HTML = loader + """
        <div class="alert alert-info">Bootstrap loaded <p class="d-none" style="color:red">This text will be visible if Bootstrap isn't loaded</p></div>
        """

        # print(loader_HTML)
        return IPyHTML(loader_HTML)

    @classmethod
    def _monkey_patch(cls):
        for key, value in vars(cls.bootstrap_version).items():
            if (
                    not hasattr(cls, key)
                    and isinstance(value, type)
                    and issubclass(value, HTML.XMLElement)
            ): setattr(cls, key, type(key, (ActiveHTMLWrapper,), dict(base=value)) )

    @classmethod
    def Grid(cls, rows, row_attributes=None, item_attributes=None, auto_size=True, **attrs):
        return ActiveHTMLWrapper(rows, base=cls.bootstrap_version.Grid, row_attributes=row_attributes,
                                       item_attributes=item_attributes, auto_size=auto_size, **attrs)

class Bootstrap3Widgets(BootstrapWidgetsBase):
    cdn_loader = """
    <!-- Latest compiled and minified CSS -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@3.3.7/dist/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">
<!-- Optional theme -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@3.3.7/dist/css/bootstrap-theme.min.css" integrity="sha384-rHyoN1iRsVXV4nD0JutlnGaslCJuC7uwjduW9SVrLvRYooPp2bWYgmgJQIXwl/Sp" crossorigin="anonymous">
<!-- Latest compiled and minified JavaScript -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@3.3.7/dist/js/bootstrap.min.js" integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa" crossorigin="anonymous"></script>
"""
    bootstrap_version = Bootstrap3
    class Icon(ActiveHTMLWrapper): base = Bootstrap3.Icon
    class Alert(ActiveHTMLWrapper): base = Bootstrap3.Alert
    class Badge(ActiveHTMLWrapper): base = Bootstrap3.Badge
    class PanelBody(ActiveHTMLWrapper): base = Bootstrap3.PanelBody
    class PanelHeader(ActiveHTMLWrapper): base = Bootstrap3.PanelHeader
    class Panel(ActiveHTMLWrapper): base = Bootstrap3.Panel
    class Jumbotron(ActiveHTMLWrapper): base = Bootstrap3.Jumbotron
    class Col(ActiveHTMLWrapper): base = Bootstrap3.Col
    class Row(ActiveHTMLWrapper): base = Bootstrap3.Row
    class Container(ActiveHTMLWrapper): base = Bootstrap3.Container
    class Button(ActiveHTMLWrapper): base = Bootstrap3.Button
    class LinkButton(HTML.Anchor): base = Bootstrap3.LinkButton
    class Table(ActiveHTMLWrapper): base = Bootstrap3.Table
    class ListGroup(ActiveHTMLWrapper): base = Bootstrap3.ListGroup
    class ListGroupItem(ActiveHTMLWrapper): base = Bootstrap3.ListGroupItem
    class FontAwesomeIcon(ActiveHTMLWrapper): base = Bootstrap3.FontAwesomeIcon
    class GlyphIcon(ActiveHTMLWrapper): base = Bootstrap3.GlyphIcon
    class Label(ActiveHTMLWrapper): base = Bootstrap3.Label
    class ListComponent(ActiveHTMLWrapper): base = Bootstrap3.ListComponent
    class ListItemComponent(ActiveHTMLWrapper): base = Bootstrap3.ListItemComponent
    class Breadcrumb(ActiveHTMLWrapper): base = Bootstrap3.Breadcrumb
    class BreadcrumbItem(ActiveHTMLWrapper): base = Bootstrap3.BreadcrumbItem
Bootstrap3Widgets._monkey_patch()

class Bootstrap4Widgets(BootstrapWidgetsBase):
    cdn_loader = """
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.1/dist/css/bootstrap.min.css" integrity="sha384-zCbKRCUGaJDkqS1kPbPd7TveP5iyJE0EjAuZQTgFLD2ylzuqKfdKlfG/eSrtxUkn" crossorigin="anonymous">
            <script src="https://cdn.jsdelivr.net/npm/jquery@3.5.1/dist/jquery.slim.min.js" integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.1/dist/js/bootstrap.bundle.min.js" integrity="sha384-fQybjgWLrvvRgtW6bFlB7jaZrFsaBXjsOMm/tB9LTS58ONXgqbR9W8oWht/amnpF" crossorigin="anonymous"></script>
            """
    bootstrap_version = Bootstrap4
    class Icon(ActiveHTMLWrapper): base = Bootstrap4.Icon
    class Alert(ActiveHTMLWrapper): base = Bootstrap4.Alert
    class Badge(ActiveHTMLWrapper): base = Bootstrap4.Badge
    class CardBody(ActiveHTMLWrapper): base = Bootstrap4.CardBody
    class CardHeader(ActiveHTMLWrapper): base = Bootstrap4.CardHeader
    class CardFooter(ActiveHTMLWrapper): base = Bootstrap4.CardFooter
    class CardImage(ActiveHTMLWrapper): base = Bootstrap4.CardImage
    class Card(ActiveHTMLWrapper): base = Bootstrap4.Card
    class Jumbotron(ActiveHTMLWrapper): base = Bootstrap4.Jumbotron
    class Col(ActiveHTMLWrapper): base = Bootstrap4.Col
    class Row(ActiveHTMLWrapper): base = Bootstrap4.Row
    class Container(ActiveHTMLWrapper): base = Bootstrap4.Container
    class Button(ActiveHTMLWrapper): base = Bootstrap4.Button
    class LinkButton(HTML.Anchor): base = Bootstrap4.LinkButton
    class Table(ActiveHTMLWrapper): base = Bootstrap4.Table
    class ListGroup(ActiveHTMLWrapper): base = Bootstrap4.ListGroup
    class ListGroupItem(ActiveHTMLWrapper): base = Bootstrap4.ListGroupItem
    class FontAwesomeIcon(ActiveHTMLWrapper): base = Bootstrap4.FontAwesomeIcon
    class GlyphIcon(ActiveHTMLWrapper): base = Bootstrap4.GlyphIcon
    class Label(ActiveHTMLWrapper): base = Bootstrap4.Label
    class Pill(ActiveHTMLWrapper): base = Bootstrap4.Pill
    class ListComponent(ActiveHTMLWrapper): base = Bootstrap4.ListComponent
    class ListItemComponent(ActiveHTMLWrapper): base = Bootstrap4.ListItemComponent
    class Breadcrumb(ActiveHTMLWrapper): base = Bootstrap4.Breadcrumb
    class BreadcrumbItem(ActiveHTMLWrapper): base = Bootstrap4.BreadcrumbItem
Bootstrap4Widgets._monkey_patch()

class Bootstrap5Widgets(BootstrapWidgetsBase):
    cdn_loader = """
            <!-- CSS only -->
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
            <!-- JavaScript Bundle with Popper -->
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p" crossorigin="anonymous"></script>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
            """
    cdn_css = [
        "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css",
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css"
    ]
    cdn_js = [
        "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"
    ]
    bootstrap_version = Bootstrap5
    class Icon(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.Icon
    class Alert(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.Alert
    class Badge(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.Badge
    class CardBody(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.CardBody
    class CardHeader(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.CardHeader
    class CardFooter(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.CardFooter
    class CardImage(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.CardImage
    class Card(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.Card
    class Col(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.Col
    class Row(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.Row
    class Container(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.Container
    class ButtonGroup(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.ButtonGroup
    class Button(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.Button
    class CloseButton(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.CloseButton
    class LinkButton(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.LinkButton
    class Table(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.Table
    class ListGroup(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.ListGroup
    class ListGroupItem(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.ListGroupItem
    class FontAwesomeIcon(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.FontAwesomeIcon
    class GlyphIcon(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.GlyphIcon
    class Label(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.Label
    class Pill(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.Pill
    class ListComponent(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.ListComponent
    class ListItemComponent(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.ListItemComponent
    class Breadcrumb(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.Breadcrumb
    class BreadcrumbItem(HTMLWidgets.WrappedHTMLElement): base = Bootstrap5.BreadcrumbItem
Bootstrap5Widgets._monkey_patch()

BootstrapWidgets = Bootstrap5Widgets
# JupyterHTMLWrapper._widget_sources.append(BootstrapWidgets)