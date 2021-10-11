"""
Defines a helper class Styled to make it easier to style plots and stuff and a ThemeManager to handle all that shiz
"""

from collections import deque
from .Backends import Backends

__all__ = [
    "Styled",
    "ThemeManager"
]


class Styled:
    """
    Simple styling class
    """
    def __init__(self, *str, **opts):
        self.val = str
        self.opts = opts

class ThemeManager:
    """
    Simple manager class for plugging into themes in a semi-background agnostic way
    """
    extra_themes = {
        'mccoy': (
            ('seaborn-dark-palette'),
            {
                'axes.labelsize': 13,
                'xtick.labelsize':13,
                'ytick.labelsize':13,
                'padding': 50
            }
        )

    }
    _resolved_theme_cache = {

    }
    def __init__(self, *theme_names, backend=Backends.MPL, graphics_styles=None, **extra_styles):
        self.main_theme_names = theme_names
        self.extra_styles = extra_styles
        self.graphics_styles = graphics_styles
        self.backend=backend
        self.context_manager = None
    @classmethod
    def from_spec(cls, theme):
        if isinstance(theme, (str, dict)):
            theme = [theme]
        if len(theme) > 0:
            try:
                theme_names, theme_properties = theme
            except ValueError:
                theme_names = theme[0]
                theme_properties = {}
            if isinstance(theme_names, dict):
                theme_properties = theme_names
                theme_names = []
            elif isinstance(theme_names, str):
                theme_names = [theme_names]
        else:
            theme_names = []
            theme_properties = {}
        return cls(*theme_names, **theme_properties)
    def _test_rcparam(self, k):
        return '.' in k
    def __enter__(self):
        if self.backend == Backends.MPL:
            import matplotlib.pyplot as plt
            theme = self.resolve_theme(None, *self.main_theme_names, **self.extra_styles)
            self.validate_theme(*theme)
            name_list = list(theme[0])
            opts = {k:v for k,v in theme[1].items() if self._test_rcparam(k)}

            self.context_manager = plt.style.context(name_list+[opts])
            return self.context_manager.__enter__()
        # don't currently support any other backends...
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.context_manager is not None:
            self.context_manager.__exit__(exc_type, exc_val, exc_tb)
            self.context_manager = None
    @property
    def theme(self):
        if self.backend == Backends.MPL:
            import matplotlib.pyplot as plt
            return self.resolve_theme(None, *self.main_theme_names, **self.extra_styles)
        else:
            raise NotImplemented("Haven't implemented themes for anything other than MPL right now...")
    @classmethod
    def add_theme(self, theme_name, *base_theme, **extra_styles):
        """
        Adds a theme to the extra themes dict. At some future date we'll
        want to make it so that this does a level of validation, too.
        :param theme_name:
        :type theme_name:
        :param base_theme:
        :type base_theme:
        :param extra_styles:
        :type extra_styles:
        :return:
        :rtype:
        """
        self.extra_themes[theme_name] = (base_theme, extra_styles)
    @classmethod
    def resolve_theme(self, theme_name, *base_themes, **extra_styles):
        """
        Resolves a theme so that it only uses strings for built-in styles
        :return:
        :rtype:
        """
        if theme_name is not None:
            if theme_name in self._resolved_theme_cache:
                themes, styles = self._resolved_theme_cache[theme_name]
            elif theme_name in self.extra_themes:
                # recursively resolve the theme
                bases, extras = self.extra_themes[theme_name]
                if isinstance(bases, str):
                    bases = [bases]
                theme_stack = deque()
                style_stack = deque()
                for theme in bases:
                    if theme in self.extra_themes:
                        t, s = self.resolve_theme(theme_name)
                        theme_stack.appendleft(t)
                        style_stack.append(s)
                    else:
                        theme_stack.appendleft([theme])
                themes = tuple(x for y in theme_stack for x in y)
                styles = {}
                for s in style_stack:
                    styles.update(s)
                styles.update(extras)
                self._resolved_theme_cache[theme_name] = [themes, styles]
            else:
                themes = (theme_name,)
                styles = {}
        else:
            themes = ()
            styles = {}
        for b in base_themes:
            if b in self.extra_themes:
                t, s = self.resolve_theme(b)
                themes = tuple(t) + themes
                styles.update(s)
            else:
                themes = (b,) + themes
        styles.update(extra_styles)

        return [themes, styles]
    def validate_theme(self, theme_names, theme_styless):
        valid_names = set(self.backend_themes)
        for k in theme_names:
            if k not in valid_names:
                raise ValueError("{}.{}: theme '{}' isn't in supported set ({})".format(
                    type(self).__name__,
                    'validate_theme',
                    k,
                    valid_names
                ))

    @property
    def backend_themes(self):
        if self.backend == Backends.MPL: # default
            import matplotlib.style as sty
            theme_names = sty.available
        else:
            theme_names = ()
        return tuple(theme_names)
    @property
    def theme_names(self):
        return self.backend_themes + tuple(self.extra_themes.keys())

