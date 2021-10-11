"""
Convenience classes for hooking into the matplotlib animation framework
"""
from .Graphics import GraphicsBase, Graphics

__all__ = [
    "EventHandler",
    "Animator"
]

class EventHandler:
    def __init__(self,
                 figure,
                 on_click = None,
                 on_release = None,
                 on_draw = None,
                 on_key_press = None,
                 on_key_release = None,
                 on_move = None,
                 on_select = None,
                 on_resize = None,
                 on_scroll = None,
                 on_figure_entered = None,
                 on_figure_left = None,
                 on_axes_entered = None,
                 on_axes_left = None
                 ):
        """Creates an EventHandler on a Figure that handles most interactivity stuff

        :param figure:
        :type figure: GraphicsBase
        :param on_click:
        :type on_click:
        :param on_release:
        :type on_release:
        :param on_draw:
        :type on_draw:
        :param on_key_press:
        :type on_key_press:
        :param on_key_release:
        :type on_key_release:
        :param on_move:
        :type on_move:
        :param on_select:
        :type on_select:
        :param on_resize:
        :type on_resize:
        :param on_scroll:
        :type on_scroll:
        :param on_figure_entered:
        :type on_figure_entered:
        :param on_figure_left:
        :type on_figure_left:
        :param on_axes_entered:
        :type on_axes_entered:
        :param on_axes_left:
        :type on_axes_left:
        """

        self.figure = figure

        if not isinstance(on_click, self.Event):
            on_click = self.ButtonPressedEvent(on_click)
        if not isinstance(on_release, self.Event):
            on_release = self.ButtonReleasedEvent(on_release)
        if not isinstance(on_draw, self.Event):
            on_draw = self.DrawEvent(on_draw)
        if not isinstance(on_key_press, self.Event):
            on_key_press = self.KeyPressedEvent(on_key_press)
        if not isinstance(on_key_release, self.Event):
            on_key_release = self.KeyReleasedEvent(on_key_release)
        if not isinstance(on_move, self.Event):
            on_move = self.MoveEvent(on_move)
        if not isinstance(on_select, self.Event):
            on_select = self.SelectEvent(on_select)
        if not isinstance(on_scroll, self.Event):
            on_scroll = self.ScrollEvent(on_scroll)
        if not isinstance(on_figure_entered, self.Event):
            on_figure_entered = self.FigureEnterEvent(on_figure_entered)
        if not isinstance(on_figure_left, self.Event):
            on_figure_left = self.FigureLeaveEvent(on_figure_left)
        if not isinstance(on_axes_entered, self.Event):
            on_axes_entered = self.AxesEnterEvent(on_axes_entered)
        if not isinstance(on_axes_left, self.Event):
            on_axes_left = self.AxesEnterEvent(on_axes_left)

        self._handles = dict(
            button_press_event = on_click,
            button_release_event =on_release,
            draw_event = on_draw,
            key_press_event = on_key_press,
            key_release_event = on_key_release,
            motion_notify_event = on_move,
            pick_event = on_select,
            resize_event = on_resize,
            scroll_event = on_scroll,
            figure_enter_event = on_figure_entered,
            figure_leave_event = on_figure_left,
            axes_enter_event = on_axes_entered,
            axes_leave_event = on_axes_left
        )

        self.bind(**self._handles)
    def bind(self, **handlers):
        self._handles.update(**handlers)
        for handle, handler in handlers.items():
            self.figure.canvas.mpl_connect(handle, handler)

    @property
    def handlers(self):
        proc_handle = lambda h: self._handles[h].data if isinstance(self._handles[h], self.Event) else None
        return {
            h: proc_handle(h) for h in self._handles
        }

    class Event:
        def __init__(self,
                     event_handler,
                     handler,
                     filter = None,
                     update = True,
                     name = None
                     ):
            self.filter = filter
            self.handler = handler
            self.parent = event_handler
            self.update = update,
            self.name = name

        @property
        def data(self):
            return (
                self.handler,
                {
                    'filter':self.filter,
                    'update':self.update,
                    'name':self.name
                    }
            )

        def handle_event(self, event):
            res = None
            if self.handler is not None:
                if self.filter is not None:
                    handle = self.filter(event)
                else:
                    handle = True
                if handle:
                    res = self.handler(event)
                    if (res is not False) and (res is not None or self.update):
                        self.parent.figure.canvas.draw()
            return res

        def __call__(self, *args, **kwargs):
            self.handle_event(*args, **kwargs)

    def ButtonPressedEvent(self, handler, **kw):
        if isinstance(handler, tuple):
            handler, pars = kw
        else:
            pars = {}
        kw = dict(pars, **kw)
        kw = dict({'name' : 'ButtonPressed'}, **kw)
        return self.Event(self, handler, **kw)
    def ButtonReleasedEvent(self, handler, **kw):
        if isinstance(handler, tuple):
            handler, pars = kw
        else:
            pars = {}
        kw = dict(pars, **kw)
        kw = dict({'name' : 'ButtonReleased'}, **kw)
        return self.Event(self, handler, **kw)
    def DrawEvent(self, handler, **kw):
        if isinstance(handler, tuple):
            handler, pars = kw
        else:
            pars = {}
        kw = dict(pars, **kw)
        kw = dict({'name' : 'Draw'}, **kw)
        return self.Event(self, handler, **kw)
    def KeyPressedEvent(self, handler, **kw):
        if isinstance(handler, tuple):
            handler, pars = kw
        else:
            pars = {}
        kw = dict(pars, **kw)
        if isinstance(handler, dict):
            filt = lambda e, h=handler: e.key in h
            handler = lambda e, h = handler: h[e.key](e)
        else:
            filt = None
        kw = dict({'name' : 'KeyPressed', 'filter':filt}, **kw)
        return self.Event(self, handler, **kw)
    def KeyReleasedEvent(self, handler, **kw):
        if isinstance(handler, tuple):
            handler, pars = kw
        else:
            pars = {}
        kw = dict(pars, **kw)
        if isinstance(handler, dict):
            filt = lambda e, h=handler: e.key in h
            handler = lambda e, h = handler: h[e.key](e)
        else:
            filt = None
        kw = dict({'name' : 'KeyReleased', 'filter':filt}, **kw)
        return self.Event(self, handler, **kw)
    def MoveEvent(self, handler, **kw):
        if isinstance(handler, tuple):
            handler, pars = kw
        else:
            pars = {}
        kw = dict(pars, **kw)
        kw = dict({'name' : 'Move'}, **kw)
        return self.Event(self, handler, **kw)
    def SelectEvent(self, handler, **kw):
        if isinstance(handler, tuple):
            handler, pars = kw
        else:
            pars = {}
        kw = dict(pars, **kw)
        kw = dict({'name' : 'Select'}, **kw)
        return self.Event(self, handler, **kw)
    def ScrollEvent(self, handler, **kw):
        if isinstance(handler, tuple):
            handler, pars = kw
        else:
            pars = {}
        kw = dict(pars, **kw)
        kw = dict({'name' : 'Scroll'}, **kw)
        return self.Event(self, handler, **kw)
    def FigureEnterEvent(self, handler, **kw):
        if isinstance(handler, tuple):
            handler, pars = kw
        else:
            pars = {}
        kw = dict(pars, **kw)
        kw = dict({'name' : 'FigureEnter'}, **kw)
        return self.Event(self, handler, **kw)
    def FigureLeaveEvent(self, handler, **kw):
        if isinstance(handler, tuple):
            handler, pars = kw
        else:
            pars = {}
        kw = dict(pars, **kw)
        kw = dict({'name' : 'FigureLeave'}, **kw)
        return self.Event(self, handler, **kw)
    def AxesEnterEvent(self, handler, **kw):
        if isinstance(handler, tuple):
            handler, pars = kw
        else:
            pars = {}
        kw = dict(pars, **kw)
        kw = dict({'name' : 'AxesEnter'}, **kw)
        return self.Event(self, handler, **kw)
    def AxesLeaveEvent(self, handler, **kw):
        if isinstance(handler, tuple):
            handler, pars = kw
        else:
            pars = {}
        kw = dict(pars, **kw)
        kw = dict({'name' : 'AxesLeave'}, **kw)
        return self.Event(self, handler, **kw)

class Animator:
    def __init__(self, figure, data_generator,
                 plot_method = None,
                 events = True,
                 update = False,
                 **anim_ops
                 ):

        from matplotlib.animation import FuncAnimation
        from matplotlib.figure import Figure
        from functools import wraps

        self.figure = figure
        if isinstance(figure, Figure):
            figure = Graphics(figure=figure, axes=figure.axes)
        self.data_generator = data_generator
        self.plotter = self._get_plot_method(figure, plot_method)
        self.update = update
        self._prev_objects = None

        @wraps(self.plotter)
        def anim_func(frame, *args, self = self, **kwargs):
            try:
                if data_generator is not None:
                    data = self.data_generator(frame, *args)
                else:
                    data = args
            except (TypeError, ValueError):
                self.plotter = self.data_generator
                self.data_generator = None
                data = args

            if not self.update:
                if self._prev_objects is None:
                    self.figure.clear()
                else:
                    for o in self._prev_objects:
                        o.remove()

            if data is None:
                self.plotter = self.data_generator
                self.data_generator = None
            else:
                self._prev_objects = self.plotter(self.figure, *data, frame = frame, **kwargs)

        self._animation = FuncAnimation(figure.figure, anim_func, **anim_ops)
        self._active = True

        if events:
            self.figure.bind_events(
                on_click = lambda *e:self.toggle()
            )

    def _get_plot_method(self, figure, plot_method):
        if plot_method is None:
            plot_method = figure.plot
            plot_method = lambda figure, *data, m = plot_method, **kwargs: m(*data)

        return plot_method

    @property
    def active(self):
        return self._active
    @active.setter
    def active(self, val):
        if val:
            self.start()
        else:
            self.stop()
    def start(self):
        if not self.active:
            self._active = True
            self._animation.event_source.start()
    def stop(self):
        if self.active:
            self._active = False
            self._animation.event_source.stop()
    def toggle(self):
        if self.active:
            self.stop()
        else:
            self.start()
    def show(self):
        return self.figure.show()

    def to_jshtml(self):
        """
        Delegates to the underlying animation
        :return:
        :rtype:
        """
        return self._animation.to_jshtml()
    def to_html5_video(self):
        """
        Delegates to the underlying animation
        :return:
        :rtype:
        """
        return self._animation.to_html5_video()

    def as_jupyter_animation(self, mode='javascript'):
        """
        Chains some stuff to make Jupyter animations work
        :return:
        :rtype:
        """

        from IPython.display import HTML
        if mode == 'javascript':
            buffer = self.to_jshtml()
        else:
            buffer = self.to_html5_video()
        return HTML(buffer)