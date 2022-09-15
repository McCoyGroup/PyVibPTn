"""
Provides various types of plots and plotting utilities
"""

from .Graphics import Graphics, Graphics3D, GraphicsGrid
import numpy as np
import matplotlib.figure
import matplotlib.axes

__all__ = [
    "Plot", "DataPlot", "ArrayPlot", "TensorPlot",
    "Plot2D", "ListPlot2D",
    "Plot3D", "ListPlot3D"
]

######################################################################################################
#
#                                    'adaptive' function sampling
#
#
# region function application
def _apply_f(f, grid):
    try:
        vals = f(grid)
    except:
        vals = np.vectorize(f)(grid)

    return vals

def _semi_adaptive_sample_func(f, xmin, xmax, npts=150, max_refines=10, der_cut=10^5):

    refines = 0
    der_good = False

    try:
        from ..Zachary import finite_difference
    except ImportError:
        pass
    else:
        while refines < max_refines and not der_good:
            grid = np.linspace(xmin, xmax, npts)
            vals = _apply_f(f, grid)
            ders = finite_difference(grid, vals, 1, 5)
            der_good = not np.any(abs(ders) > der_cut)
            npts *= 2
            refines += 1

    if refines == 0:  # edge case
        grid = np.linspace(xmin, xmax, npts)
        vals = _apply_f(f, grid)

    return grid, vals, npts, refines

def _semi_adaptive_sample_func2(f, xmin, xmax, ymin, ymax, npts=15, max_refines=10, der_cut=10 ^ 5):
    from ..Zachary import finite_difference

    refines = 0
    der_good = False
    try:
        from ..Zachary import finite_difference
    except ImportError:
        pass
    else:
        while refines < max_refines and not der_good:
            grid = np.array(np.meshgrid(np.linspace(xmin, xmax, npts), np.linspace(ymin, ymax, npts))).T
            vals = _apply_f(f, grid)
            ders = finite_difference(grid, vals, (1, 1), (5, 5))
            der_good = not np.any(abs(ders) > der_cut)
            npts *= 2
            refines += 1

    if refines == 0:  # edge case
        grid = np.linspace(xmin, xmax, npts)
        vals = _apply_f(f, grid)

    return grid, vals, npts, refines
# endregion


######################################################################################################
#
#                                    Plot data methods
#
#
# region plot data prep
def _interp2DData(gpts, **opts):
    from scipy.interpolate import griddata

    x = np.sort(gpts[:, 0])
    y = np.sort(gpts[:, 1])

    xmin = np.min(x); xmax = np.max(x)

    xdiffs = np.abs(np.diff(x)); xh = np.min(xdiffs[np.nonzero(xdiffs)])
    ymin = np.min(y); ymax = np.max(y)
    ydiffs = np.abs(np.diff(y)); yh = np.min(xdiffs[np.nonzero(ydiffs)])

    num_x = (xmin - xmax) / xh
    if 5 > num_x or num_x < 10000:  # don't want to get too wild
        num_x = 100  # okay but let's get a little wild
    num_y = (ymin - ymax) / yh
    if 5 > num_y or num_y < 10000:  # don't want to get too wild
        num_y = 100  # okay but let's get a little wild

    # import sys
    # print(num_x, num_y, file = sys.stderr)

    xmesh = np.linspace(xmin, xmax, num_x); ymesh = np.linspace(ymin, ymax, num_y)
    xmesh, ymesh = np.meshgrid(xmesh, ymesh)
    mesh = np.array((xmesh, ymesh)).T
    vals = griddata(gpts[:, (0, 1)], gpts[:, 2], mesh, **opts)

    return xmesh, ymesh, vals.T

def _get_2D_plotdata(func, xrange):
    if hasattr(func, 'subs'):
        from sympy import lambdify
        sym, xrange = xrange
        xrange = np.arange(*xrange)
        fvalues = lambdify([sym], func)(xrange)
    elif not callable(func):
        fvalues = xrange
        xrange = func
    else:
        if len(xrange) == 3 and abs(xrange[2]) < abs(xrange[1] - xrange[0]):
            xrange = np.arange(*xrange)
            fvalues = _apply_f(func, xrange)
        elif len(xrange) > 2:
            fvalues = _apply_f(func, xrange)
        else:
            res = _semi_adaptive_sample_func(func, *xrange)
            xrange = res[0]
            fvalues = res[1]
    return xrange, fvalues

def _get_3D_plotdata(func, xrange, yrange):
    if not callable(func):
        fvalues = yrange
        yrange = xrange
        xrange = func
    else:
        if len(xrange) == 3 and abs(xrange[2]) < abs(xrange[1] - xrange[0]):
            xrange = np.arange(*xrange)
            yrange = np.arange(*yrange)
            xrange, yrange = mesh = np.meshgrid(xrange, yrange)
            fvalues = _apply_f(func, mesh)
        elif len(xrange) > 2 and len(yrange) > 2:
            xrange, yrange = mesh = np.meshgrid(xrange, yrange)
            fvalues = _apply_f(func, mesh)
        else:
            res = _semi_adaptive_sample_func2(func, *xrange, *yrange)
            xrange, yrange = res[0].T
            fvalues = res[1]

    return xrange, yrange, fvalues
# endregion

######################################################################################################
#
#                                    Unified PlotBase class
#
#
# Never implemented this

######################################################################################################
#
#                                    2D Plots on 2D Axes
#
#
class Plot(Graphics):
    """
    The base plotting class to interface into matplotlib or (someday 3D) VTK.
    In the future hopefully we'll be able to make a general-purpose `PlottingBackend` class that doesn't need to be `matplotlib` .
    Builds off of the `Graphics` class to make a unified and convenient interface to generating plots.
    Some sophisticated legwork unfortunately has to be done vis-a-vis tracking constructed lines and other plotting artefacts,
    since `matplotlib` is designed to infuriate.
    """

    line_params = {
        "linewidth", "linestyle", "color", "marker", "markersize",
        "markeredgewidth", "markeredgecolor", "markerfacecolor", "markerfacecoloralt",
        "fillstyle", "antialiased", "dash_capstyle", "solid_capstyle",
        "dash_joinstyle", "solid_joinstyle", "pickradius", "drawstyle", "markevery"
    }
    patch_parms = {
        "agg_filter", "alpha", "animated", "antialiased", "capstyle",
        "clip_box", "clip_on", "clip_path", "color", "edgecolor", "facecolor",
        "figure", "fill", "gid", "hatch", "in_layout", "joinstyle",
         "label", "linestyle", "linewidth", "path_effects",
        "picker", "rasterized", "sketch_params", "snap",
        "transform", "url", "visible", "zorder"
    }

    opt_keys = Graphics.opt_keys | {"plot_style"}

    default_plot_style = {}
    style_mapping = {"format":"fmt"}
    known_styles = {"fmt"} | line_params
    method = "plot"
    def __init__(self,
                 *params,
                 method=None,
                 figure=None, axes=None, subplot_kw=None,
                 plot_style=None, theme=None,
                 **opts
                 ):
        """
        :param params: _empty_ or _x_, _y_ arrays or _function_, _xrange_
        :type params:
        :param plot_style: the plot styling options to be fed into the plot method
        :type plot_style: dict | None
        :param method: the method name as a string or functional form of the method to plot
        :type method: str | function
        :param figure: the Graphics object on which to plot (None means make a new one)
        :type figure: Graphics | None
        :param axes: the axes on which to plot (used in constructing a Graphics, None means make a new one)
        :type axes: None
        :param subplot_kw: the keywords to pass on when initializing the plot
        :type subplot_kw: dict | None
        :param colorbar: whether to use a colorbar or what options to pass to the colorbar
        :type colorbar: None | bool | dict
        :param opts: options to be fed in when initializing the Graphics
        :type opts:
        """

        self.graphics = None

        # we're gonna set things up so that we can have delayed evaluation of the plotting.
        # i.e. a Plot can be initialized but then do all its plotting later
        if plot_style is None:
            plot_style = {}
        for k,v in self.style_mapping.items():
            if k in opts:
                opts[v] = opts[k]
                del opts[k]
        for k in self.known_styles:
            if k in opts:
                plot_style[k] = opts[k]
        for k in self.default_plot_style:
            if k not in plot_style:
                plot_style[k] = self.default_plot_style[k]
        self._plot_style = plot_style
        self.plot_opts = opts
        self._initialized = False
        self._data = None

        super().__init__(figure=figure, axes=axes, theme=theme, subplot_kw=subplot_kw, **opts)
        self._init_opts['plot_style'] = plot_style
        if method is None:
           method = self.method
        if isinstance(method, str):
            method = getattr(self.axes, method)
        self._method = method

        if len(params) > 0:
            self.plot(*params)

    known_keys = Graphics.known_keys | {
        'method',
        'plot_style',
        'insert_default_styles'
    }
    @classmethod
    def filter_options(cls, opts, allowed=None):
        new = {}
        if allowed is None:
            allowed = cls.known_styles | cls.known_keys
        for k in opts.keys() & allowed:
                new[k] = opts[k]
        return new
    def _check_opts(self, opts):
        diff = opts.keys() - (self.known_styles | self.known_keys)
        if len(diff) > 0:
            raise ValueError("unknown options for {}: {}".format(
                type(self).__name__, list(diff)
            ))

    def _initialize(self):
        self._initialized = True
        self.set_options(**self.plot_opts)

    def _get_plot_data(self, func, xrange):
        xrange, fvalues = _get_2D_plotdata(func, xrange)
        return xrange, fvalues

    def _plot_data(self, *data, **plot_style):
        return self._method(*self._get_plot_data(*data), **plot_style)

    def plot(self, *params, insert_default_styles=True, **plot_style):
        """
        Plots a set of data & stores the result
        :return: the graphics that matplotlib made
        :rtype:
        """
        if insert_default_styles:
            plot_style = dict(self.plot_style, **plot_style)
        self._data = (params, plot_style)
        self.graphics = self._plot_data(*params, **plot_style)
        if not self._initialized:
            self._initialize()
        return self.graphics
    @property
    def artists(self):
        if self.graphics is None or isinstance(self.graphics, list):
            return self.graphics
        else:
            return [self.graphics]

    # def copy(self):
    #     return self.change_figure(None)
    def _change_figure(self, new, *init_args, **init_kwargs):
        """Creates a copy of the object with new axes and a new figure

        :return:
        :rtype:
        """
        # print(init_kwargs)
        # print(self._data[0], init_args)
        # print(init_kwargs, self._init_opts)
        return super()._change_figure(new, *self._data[0], *init_args, **init_kwargs)

    def clear(self):
        """
        Removes the plotted data
        """
        for g in self.graphics:
            self.axes.remove(g)
        self.graphics=None
    def restyle(self, **plot_style):
        """
        Replots the data with updated plot styling
        :param plot_style:
        :type plot_style:
        """
        self.clear()
        self.plot(*self.data, **plot_style)

    @property
    def data(self):
        """
        The data that we plotted
        """
        if self._data is None:
            raise ValueError("{} hasn't been plotted in the first place...")
        return self._data[0]
    @property
    def plot_style(self):
        """
        The styling options applied to the plot
        """
        if self._data is None:
            style = self._plot_style
        else:
            style = self._data[1]
        return style

    def add_colorbar(self, graphics = None, norm = None,  **kw):
        """
        Adds a colorbar to the plot
        """
        if self._initialized:
            if graphics is None and norm is None:
                graphics = self.graphics
            return super().add_colorbar(graphics=graphics, **kw)

    def set_graphics_properties(self, *which, **kw):
        with self.pyplot as plt:
            if isinstance(self.graphics, tuple):
                for n,g in enumerate(self.graphics):
                    if len(which) == 0 or n in which:
                        plt.setp(g, **kw)
            else:
                plt.setp(self.graphics, **kw)


    @classmethod
    def merge(cls, main, other, *rest, **kwargs):
        return CompositePlot(main, other, *rest, **kwargs)

    plot_classes = {}
    @classmethod
    def resolve_method(cls, mpl_name):
        return cls.plot_classes[mpl_name]
    # @classmethod
    # def merge_plots(cls, *plots, **styles):
    #     ...
    @classmethod
    def register(cls, plot_class):
        cls.plot_classes[plot_class.method] = plot_class
        return plot_class
Plot.register(Plot)

class CompositePlot:
    def __init__(self, main, other, *rest, **kwargs):
        self.kwargs = kwargs
        self.plots = [main, other, *rest]
    def merge(self, **kwargs):
        base = self.plots[0].change_figure(None, **kwargs)
        for p in self.plots[1:]:
            p.change_figure(base)
        return base
    def show(self):
        self._ref = self.merge(interactive=True, **self.kwargs)
        # self._ref.pyplot.mpl_connect()
        self._ref.show()
    def _ipython_display_(self):
        self.show()

@Plot.register
class FilledPlot(Plot):
    """
    Inherits from `Plot`.
    Plots a bunch of x values against a bunch of y values using the `scatter` method.
    """
    known_styles = { "where", "interpolate", "step", "data" } | Plot.patch_parms
    method = "fill_between"

@Plot.register
class ScatterPlot(Plot):
    """
    Inherits from `Plot`.
    Plots a bunch of x values against a bunch of y values using the `scatter` method.
    """
    known_styles = { "s", "c", "marker", "cmap", "norm", "vmin", "vmax", "alpha", "linewidths", "edgecolors", "plotnonfinite", "data"}
    style_mapping = {"color":"c"}
    method = "scatter"

class ListScatterPlot(ScatterPlot):
    """
    Inherits from `Plot`.
    Plots a bunch of (x, y) points using the `scatter` method.
    """
    def __init__(self, griddata, **opts):
        super().__init__(griddata[:, 0], griddata[:, 1], **opts)

@Plot.register
class ErrorBarPlot(Plot):
    """
    Inherits from `Plot`.
    Plots error bars using the `errorbar` method.
    """
    known_styles = {"yerr", "xerr", "fmt", "ecolor", "elinewidth", "capsize", "barsabove",
    "lolims", "uplims", "xlolims", "xuplims", "errorevery", "capthick", "data"} | Plot.known_styles
    method = "errorbar"
class ListErrorBarPlot(ErrorBarPlot):
    """A Plot that pulls the errorbar data from a list"""
    def __init__(self, griddata, **opts):
        super().__init__(griddata[:, 0], griddata[:, 1], **opts)

@Plot.register
class StickPlot(Plot):
    """A Plot object that plots sticks"""

    default_plot_style = {'basefmt': " ", 'use_line_collection':True, 'markerfmt': " "}
    known_styles = {"linefmt", "markerfmt", "basefmt", "bottom", "label", "use_line_collection", "orientation", "data"}
    method = "stem"
    def plot(self, *params, insert_default_styles=True, **plot_style):
        """
        Plots a set of data | stores the result
        :return: the graphics that matplotlib made
        :rtype:
        """
        if insert_default_styles:
            plot_style = dict(self.plot_style, **plot_style)
        # plot_style = dict(self.plot_style, **plot_style)
        if 'linewidth' in plot_style:
            lw = plot_style['linewidth']
            del plot_style['linewidth']
        else:
            lw = None
        lc = None
        if 'color' in plot_style:
            if 'linefmt' in plot_style:
                raise ValueError("modifying linefmt not currently supported")
            lc = plot_style['color']
            del plot_style['color']
        linefmt = ''
        if 'line_style' in plot_style:
            ls = plot_style['line_style']
            del plot_style['line_style']
            if ls == 'dashed':
                if 'linefmt' in plot_style:
                    raise ValueError("modifying passed linefmt not currently supported")
                linefmt+="--"
            elif ls == 'dotted':
                if 'linefmt' in plot_style:
                    raise ValueError("modifying passed linefmt not currently supported")
                linefmt+="-."
            plot_style['linefmt'] = linefmt
        super().plot(*params, insert_default_styles=False, **plot_style)
        if lw is not None:
            self.set_graphics_properties(1, linewidth=lw)
        if lc is not None:
            self.set_graphics_properties(1, color=lc)
        return self.graphics
class ListStickPlot(StickPlot):
    """A Plot object that plots sticks from a list"""
    def __init__(self, griddata, **opts):
        super().__init__(griddata[:, 0], griddata[:, 1], **opts)

@Plot.register
class DatePlot(Plot):
    method = 'plot_date'
    known_styles = {'fmt', 'tz', 'xdate', 'ydate', 'data'} | Plot.known_styles
@Plot.register
class StepPlot(Plot):
    method = 'step'
    known_styles = {'where', 'data'} | Plot.known_styles
@Plot.register
class LogLogPlot(Plot):
    method = 'loglog'
    # known_styles = {}
@Plot.register
class SemiLogXPlot(Plot):
    method = 'semilogx'
    # known_styles = {}
@Plot.register
class SemilogYPlot(Plot):
    method = 'semilogy'
    # known_styles = {}
@Plot.register
class HorizontalFilledPlot(Plot):
    method = 'fill_betweenx'
    known_styles = {'where', 'step', 'interpolate', 'data'} | Plot.patch_parms
@Plot.register
class BarPlot(Plot):
    method = 'bar'
    known_styles = {'height', 'width', 'bottom', 'align', 'data'} | Plot.patch_parms
@Plot.register
class HorizontalBarPlot(Plot):
    method = 'barh'
    known_styles = {'width', 'height', 'left', 'align'} | Plot.patch_parms
# @Plot.register
# class BarLabelPlot(Plot):
#     method = 'bar_label'
#     known_styles = {'container', 'labels', 'fmt', 'label_type', 'padding'}  | Plot.patch_parms
@Plot.register
class EventPlot(Plot):
    method = 'eventplot'
    known_styles = {'positions', 'orientation', 'lineoffsets', 'linelengths', 'linewidths', 'colors', 'linestyles', 'data'} | Plot.line_params
@Plot.register
class PiePlot(Plot):
    method = 'pie'
    known_styles = {'explode', 'labels', 'colors', 'autopct', 'pctdistance',
                    'shadow', 'labeldistance', 'startangle', 'radius', 'counterclock',
                    'wedgeprops', 'textprops', 'center', 'frame', 'rotatelabels', 'normalize',
                    'data'} | Plot.patch_parms
@Plot.register
class StackPlot(Plot):
    method = 'stackplot'
    known_styles = {'labels', 'colors', 'baseline', 'data'} | Plot.patch_parms
@Plot.register
class BrokenHorizontalBarPlot(Plot):
    method = 'broken_barh'
    known_styles = {'xranges', 'yrange', 'data'} | Plot.patch_parms
@Plot.register
class VerticalLinePlot(Plot):
    """
    Plots a bunch of vertical lines
    """
    known_styles = {'ymin', 'ymax', 'colors', 'linestyles', 'label', 'data'} | Plot.line_params
    method = 'vlines'
    def _get_plot_data(self, x, y=1.0):
        if isinstance(y, (int, float)):
            y = [0, y]
        return (x, y)
    def _plot_data(self, *data, **plot_style):
        x, y = data
        return self._method(x, *y, **plot_style)
@Plot.register
class HorizontalLinePlot(Plot):
    """
    Plots a bunch of vertical lines
    """
    known_styles = {'ymin', 'ymax', 'colors', 'linestyles', 'label', 'data'} | Plot.line_params
    method = 'vlines'
    def _get_plot_data(self, y, x=1.0):
        if isinstance(x, (int, float)):
            x = [0, x]
        return (x, y)
    def _plot_data(self, *data, **plot_style):
        x, y = data
        return self._method(x, *y, **plot_style)
#     known_styles = {'xmin', 'xmax', 'colors', 'linestyles', 'label', 'data'}
@Plot.register
class PolygonPlot(Plot):
    method = 'fill'
    known_styles = {'data'} | Plot.patch_parms

@Plot.register
class AxisHorizontalLinePlot(Plot):
    method = 'axhline'
    known_styles = {'xmin', 'xmax'} | Plot.line_params
@Plot.register
class AxisHorizontalSpanPlot(Plot):
    method = 'axhspan'
    known_styles = {'ymin', 'ymax', 'xmin', 'xmax'} | Plot.patch_parms
@Plot.register
class AxisVerticalLinePlot(Plot):
    method = 'axvline'
    known_styles = {'ymin', 'ymax'} | Plot.line_params
@Plot.register
class AxisVeticalSpanPlot(Plot):
    method = 'axvspan'
    known_styles = {'xmin', 'xmax', 'ymin', 'ymax'} | Plot.patch_parms
@Plot.register
class AxisLinePlot(Plot):
    method = 'axline'
    known_styles = {'xy1', 'xy2', 'slope'} | Plot.line_params

@Plot.register
class StairsPlot(Plot):
    method = 'stairs'
    known_styles = {'values', 'edges', 'orientation', 'baseline', 'fill', 'data'} | Plot.patch_parms

# class ClabelPlot(Plot):
#     method = 'clabel'
#     known_styles = {'CS', 'levels'}


# class AnnotatePlot(Plot):
#     method = 'annotate'
#     known_styles = {'text', 'xy'}
# class TextPlot(Plot):
#     method = 'text'
#     known_styles = {'s', 'fontdict'}
# class TablePlot(Plot):
#     method = 'table'
#     known_styles = {'cellText', 'cellColours', 'cellLoc', 'colWidths', 'rowLabels', 'rowColours', 'rowLoc', 'colLabels', 'colColours', 'colLoc',
#                     'loc', 'bbox', 'edges'}
# class ArrowPlot(Plot):
#     method = 'arrow'
#     known_styles = {'dx', 'dy'}
# class InsetAxesPlot(Plot):
#     method = 'inset_axes'
#     known_styles = {'bounds', 'transform', 'zorder'}
# class IndicateInsetPlot(Plot):
#     method = 'indicate_inset'
#     known_styles = {'bounds', 'inset_ax', 'transform', 'facecolor', 'edgecolor', 'alpha', 'zorder'}
# class IndicateInsetZoomPlot(Plot):
#     method = 'indicate_inset_zoom'
#     known_styles = {'inset_ax'}
# class SecondaryXaxisPlot(Plot):
#     method = 'secondary_xaxis'
#     known_styles = {'location', 'functions'}
# class SecondaryYaxisPlot(Plot):
#     method = 'secondary_yaxis'
#     known_styles = {'location', 'functions'}
# class BarbsPlot(Plot):
#     method = 'barbs'
#     known_styles = {'data'}


######################################################################################################
#
#                                    Pure Data Plots on 2D Axes
#
#
class DataPlot(Plot):
    """
    Makes a 2D plot of arbitrary data using a plot method that handles that data type
    """
    image_params = {'cmap', 'norm', 'aspect', 'interpolation', 'alpha', 'vmin', 'vmax', 'origin',
                    'extent', 'interpolation_stage', 'filternorm', 'filterrad', 'resample', 'url', 'data'}
    def __init__(self,
                 *params,
                 plot_style=None, method=None,
                 figure=None, axes=None, subplot_kw=None,
                 colorbar=None,
                 **opts
                 ):
        """
        :param params: _empty_ or _data_
        :type params:
        :param plot_style: the plot styling options to be fed into the plot method
        :type plot_style: dict | None
        :param method: the method name as a string
        :type method: str
        :param figure: the Graphics object on which to plot (None means make a new one)
        :type figure: Graphics | None
        :param axes: the axes on which to plot (used in constructing a Graphics, None means make a new one)
        :type axes: None
        :param subplot_kw: the keywords to pass on when initializing the plot
        :type subplot_kw: dict | None
        :param colorbar: whether to use a colorbar or what options to pass to the colorbar
        :type colorbar: None | bool | dict
        :param opts: options to be fed in when initializing the Graphics
        :type opts:
        """
        super().__init__(*params,
                         plot_style=plot_style, method=method,
                         colorbar=colorbar, figure=figure,
                         axes=axes, subplot_kw=subplot_kw,
                         **opts
                         )
    def _get_plot_data(self, data):
        return data,

@Plot.register
class HistogramPlot(DataPlot):
    """
    Makes a Histogram of data
    """
    method = 'hist'
    known_styles = {'bins', 'range', 'density', 'weights', 'cumulative',
                    'bottom', 'histtype', 'align', 'orientation', 'rwidth', 'log', 'color',
                    'label', 'stacked', 'data'}

@Plot.register
class HistogramPlot2D(DataPlot):
    """
    Makes a 2D histogram of data
    """
    method = 'hist2d'
    known_styles = {'bins', 'range', 'density', 'weights', 'cmin', 'cmax', 'data'}

@Plot.register
class SpectrogramPlot(DataPlot):
    method = 'specgram'
    known_styles = {'NFFT', 'Fs', 'Fc', 'detrend', 'window', 'noverlap',
                    'cmap', 'xextent', 'pad_to', 'sides', 'scale_by_freq', 'mode', 'scale', 'vmin',
                    'vmax', 'data'} | DataPlot.image_params
@Plot.register
class AutocorrelationPlot(DataPlot):
    method = 'acorr'
    known_styles = {'detrend', 'normed', 'usevlines', 'maxlags', 'linestyle', 'marker', 'data'} | Plot.known_styles
@Plot.register
class AngleSpectrumPlot(DataPlot):
    method = 'angle_spectrum'
    known_styles = {'Fs', 'Fc', 'window', 'pad_to', 'sides', 'data'}  | Plot.patch_parms
@Plot.register
class CoherencePlot(DataPlot):
    method = 'cohere'
    known_styles = {'NFFT', 'Fs', 'Fc', 'noverlap', 'pad_to', 'sides', 'scale_by_freq', 'data'} | Plot.known_styles
@Plot.register
class CrossSpectralDensityPlot(DataPlot):
    method = 'csd'
    known_styles = {'NFFT', 'Fs', 'Fc', 'detrend', 'window', 'noverlap', 'pad_to', 'sides', 'scale_by_freq', 'return_line', 'data'}  | Plot.known_styles
@Plot.register
class MagnitudeSpectrumPlot(DataPlot):
    method = 'magnitude_spectrum'
    known_styles = {'Fs', 'Fc', 'window', 'pad_to', 'sides', 'scale', 'data'} | Plot.known_styles
@Plot.register
class PhaseSpectrumPlot(DataPlot):
    method = 'phase_spectrum'
    known_styles = {'Fs', 'Fc', 'window', 'pad_to', 'sides', 'data'} | Plot.known_styles
@Plot.register
class PowerSpectralDensityPlot(DataPlot):
    method = 'psd'
    known_styles = {'NFFT', 'Fs', 'Fc', 'detrend', 'window', 'noverlap', 'pad_to', 'sides', 'scale_by_freq', 'return_line', 'data'} | Plot.known_styles
@Plot.register
class CrossCorrelationPlot(DataPlot):
    method = 'xcorr'
    known_styles = {'normed', 'usevlines', 'maxlags', 'data', 'linestyle', 'marker'} | Plot.known_styles
@Plot.register
class BoxPlot(DataPlot):
    method = 'boxplot'
    known_styles = {'notch', 'sym', 'vert', 'whis', 'positions', 'widths',
                    'patch_artist', 'bootstrap', 'usermedians', 'conf_intervals', 'meanline',
                    'showmeans', 'showcaps', 'showbox', 'showfliers', 'boxprops', 'labels',
                    'flierprops', 'medianprops', 'meanprops', 'capprops', 'whiskerprops',
                    'manage_ticks', 'autorange', 'zorder', 'data'}
@Plot.register
class ViolinPlot(DataPlot):
    method = 'violinplot'
    known_styles = {'dataset', 'positions', 'vert', 'widths', 'showmeans',
                    'showextrema', 'showmedians', 'quantiles', 'points', 'bw_method', 'data'}
# class ViolinPlot(Plot):
#     method = 'violin'
#     known_styles = {'vpstats', 'positions', 'vert', 'widths', 'showmeans',
#                     'showextrema', 'showmedians'}
@Plot.register
class BoxAndWhiskerPlot(DataPlot):
    method = 'bxp'
    known_styles = {'bxpstats', 'positions', 'widths', 'vert', 'patch_artist',
                    'shownotches', 'showmeans', 'showcaps', 'showbox', 'showfliers', 'boxprops',
                    'whiskerprops', 'flierprops', 'medianprops', 'capprops', 'meanprops',
                    'meanline', 'manage_ticks', 'zorder'}
@Plot.register
class HexagonalHistogramPlot(DataPlot):
    method = 'hexbin'
    known_styles = {'C', 'gridsize', 'bins', 'xscale', 'yscale', 'extent', 'cmap', 'norm', 'vmin',
                    'vmax', 'alpha', 'linewidths', 'edgecolors', 'mincnt', 'marginals', 'data', 'reduce_C_function'} | Plot.patch_parms

@Plot.register
@Plot.register
class QuiverPlot(DataPlot):
    method = 'quiver'
    known_styles = {"units", "angles", "scale", "scale_units", "width", "headwidth", "headlength",
                    "headaxislength", "minshaft", "minlength", "pivot", "color", "data"} | Plot.patch_parms
@Plot.register
class StreamPlot(DataPlot):
    method = 'streamplot'
    known_styles = {'density', 'linewidth', 'color', 'cmap', 'norm',
                    'arrowsize', 'arrowstyle', 'minlength', 'transform', 'zorder', 'start_points',
                    'maxlength', 'integration_direction', 'data'}

@Plot.register
class ArrayPlot(DataPlot):
    """
    Plots an array as an image
    """

    method = 'imshow'
    known_styles = DataPlot.image_params
    def __init__(self, *params,
                 plot_style=None, colorbar=None,
                 figure=None, axes=None, subplot_kw=None,
                 **opts
                 ):
        super().__init__(*params,
                         plot_style=plot_style,
                         colorbar=colorbar, figure=figure,
                         axes=axes, subplot_kw=subplot_kw,
                         **opts
                         )
    def _get_plot_data(self, data):
        if hasattr(data, 'toarray'):
            data = data.toarray()
        return data,
@Plot.register
class MatrixPlot(ArrayPlot):
    method = 'matshow'
@Plot.register
class SparsityPlot(ArrayPlot):
    method = 'spy'
    known_styles = {'precision', 'marker', 'markersize', 'aspect', 'origin'} | ArrayPlot.known_styles

class TensorPlot(GraphicsGrid):
    """
    Plots slices of a tensor as a grid
    """
    def __init__(self, tensor,
                 nrows=None, ncols=None,
                 plot_style=None, colorbar=None,
                 figure=None, axes=None, subplot_kw=None,
                 method='imshow',
                 **opts
                 ):
        from operator import mul
        from functools import reduce
        tensor_shape = tensor.shape
        total_dim = reduce(mul, tensor_shape[:-2], 1)
        if nrows is None or ncols is None:
            if len(tensor_shape) == 3:
                nrows = 1
                ncols = tensor_shape[0]
            elif len(tensor_shape) == 4:  # best case
                nrows, ncols = tensor_shape[:2]
            else:
                if nrows is not None:
                    ncols = total_dim // nrows
                elif ncols is not None:
                    nrows = total_dim // ncols
                else:
                    ncols = 5
                    nrows = total_dim // ncols
        super().__init__(nrows=nrows, ncols=ncols,
                         figure=figure,
                         axes=axes,
                         subplot_kw=subplot_kw
                         )

        tensor = tensor.reshape((total_dim,) + tensor_shape[-2:])
        for i in range(nrows):
            for j in range(ncols):
                graphics = self.axes[i][j]
                self.axes[i][j] = ArrayPlot(
                    tensor[nrows * i + j],
                    figure=graphics,
                    plot_style=plot_style,
                    colorbar=colorbar,
                    method=method,
                    **opts
                )

######################################################################################################
#
#                                    3D Plots on 2D Axes
#
#

class Plot2D(Plot):
    """
    A base class for plots of 3D data but plotted on 2D axes
    """
    known_styles = {"corner_mask", "colors", "alpha", "cmap", "norm", "vmin, vmax", "origin",
                    "extent", "locator", "extend", "xunits, yunits", "antialiased", "nchunk",
                    "linewidths", "linestyles", "hatches", "data"}
    method='contour'
    def __init__(self, *params,
                 plot_style=None,
                 colorbar=None,
                 figure=None,
                 axes=None,
                 subplot_kw=None,
                 **opts
                 ):
        """
        :param params: either _empty_ or _x_, _y_, _z_ arrays or _function_, _xrange_, _yrange_
        :type params:
        :param plot_style: the plot styling options to be fed into the plot method
        :type plot_style: dict | None
        :param method: the method name as a string
        :type method: str
        :param figure: the Graphics object on which to plot (None means make a new one)
        :type figure: Graphics | None
        :param axes: the axes on which to plot (used in constructing a Graphics, None means make a new one)
        :type axes: None
        :param subplot_kw: the keywords to pass on when initializing the plot
        :type subplot_kw: dict | None
        :param colorbar: whether to use a colorbar or what options to pass to the colorbar
        :type colorbar: None | bool | dict
        :param opts: options to be fed in when initializing the Graphics
        :type opts:
        """
        super().__init__(*params,
                         plot_style=plot_style,
                         colorbar=colorbar, figure=figure,
                         axes=axes, subplot_kw=subplot_kw,
                         **opts
                         )
    def _get_plot_data(self, func, xrange, yrange):
        return _get_3D_plotdata(func, xrange, yrange)
@Plot.register
class ContourPlot(Plot2D):
    method = 'contourf'
@Plot.register
class DensityPlot(Plot2D):
    method = 'pcolormesh'
    known_styles = {'alpha', 'norm', 'cmap', 'vmin', 'vmax', 'shading', 'antialiased', 'data'} | Plot.patch_parms
@Plot.register
class HeatmapPlot(Plot2D):
    method = 'pcolor'
    known_styles = {'shading', 'alpha', 'norm', 'cmap', 'vmin', 'vmax', 'data'} | ArrayPlot.known_styles
@Plot.register
class TriPlot(Plot2D):
    """A Plot object that plots a triangulation bars"""
    method = 'triplot'
    known_styles = {"triangles", "mask"} | Plot.known_styles
class ListTriPlot(Plot2D):
    """A Plot that pulls the triangulation data from a list"""
    def __init__(self, griddata, **opts):
        super().__init__(griddata[:, 0], griddata[:, 1], **opts)
@Plot.register
class TriDensityPlot(Plot2D):
    method = 'tripcolor'
    known_styles = {'alpha', 'norm', 'cmap', 'vmin', 'vmax', 'shading', 'facecolors'}
@Plot.register
class TriContourLinesPlot(Plot2D):
    method = 'tricontour'
    known_styles = {"triangles", "mask", "levels", "colors",
                    "alpha", "cmap", "norm", "vmin, vmax", "origin", "extent", "locator", "extend",
                    "xunits, yunits", "antialiased", "linewidths", "linestyles"}
@Plot.register
class TriContourPlot(Plot2D):
    method = 'tricontourf'
    known_styles = {"triangles", "mask", "levels", "colors",
                    "alpha", "cmap", "norm", "vmin, vmax", "origin", "extent", "locator", "extend",
                    "xunits, yunits", "antialiased", "hatches"}

class ListPlot2D(Plot2D):
    """
    Convenience class that handles the interpolation first
    """
    def __init__(self,
                 *params,
                 plot_style=None,
                 method='contour',
                 colorbar=None,
                 figure=None,
                 axes=None,
                 subplot_kw=None,
                 interpolate=True,
                 **opts
                 ):
        """
        :param params: either _empty_ or and array of (_x_, _y_, _z_) points
        :type params:
        :param plot_style: the plot styling options to be fed into the plot method
        :type plot_style: dict | None
        :param method: the method name as a string
        :type method: str
        :param figure: the Graphics object on which to plot (None means make a new one)
        :type figure: Graphics | None
        :param axes: the axes on which to plot (used in constructing a Graphics, None means make a new one)
        :type axes: None
        :param subplot_kw: the keywords to pass on when initializing the plot
        :type subplot_kw: dict | None
        :param colorbar: whether to use a colorbar or what options to pass to the colorbar
        :type colorbar: None | bool | dict
        :param interpolate: whether to interpolate the data or not
        :type interpolate: bool
        :param opts: options to be fed in when initializing the Graphics
        :type opts:
        """
        self.interpolate = interpolate
        super().__init__(*params,
                         plot_style=plot_style, method=method,
                         colorbar=colorbar, figure=figure,
                         axes=axes, subplot_kw=subplot_kw,
                         **opts
                         )

    def _get_plot_data(self, *griddata, interpolate=None):
        if interpolate is None:
            interpolate = self.interpolate
        if len(griddata) == 3:
            x, y, z = griddata
        elif interpolate:
            x, y, z = _interp2DData(griddata[0])
        else:
            x = griddata[0][:, 0]
            y = griddata[0][:, 1]
            z = griddata[0][:, 2]

        return (x, y, z)

class ListContourPlot(ContourPlot):
    _get_plot_data = ListPlot2D._get_plot_data
class ListDensityPlot(DensityPlot):
    _get_plot_data = ListPlot2D._get_plot_data
class ListTriContourPlot(TriContourPlot):
    _get_plot_data = ListPlot2D._get_plot_data
class ListTriDensityPlot(TriDensityPlot):
    _get_plot_data = ListPlot2D._get_plot_data


######################################################################################################
#
#                                    3D Plots on 3D Axes
#
#
class Plot3D(Graphics3D):  # basically a mimic of the Plot class but inheriting from Graphics3D
    """A base class for 3D plots"""

    default_plot_style = {}
    style_mapping = {"format": "fmt"}
    known_styles = {"fmt"} | Plot.line_params
    method = 'plot_surface'
    def __init__(self, *params,
                 plot_style=None,
                 method=None, colorbar=None,
                 figure=None, axes=None, subplot_kw=None,
                 **opts
                 ):
        """
        :param params: either _empty_ or _x_, _y_, _z_ arrays or _function_, _xrange_, _yrange_
        :type params:
        :param plot_style: the plot styling options to be fed into the plot method
        :type plot_style: dict | None
        :param method: the method name as a string
        :type method: str
        :param figure: the Graphics object on which to plot (None means make a new one)
        :type figure: Graphics | None
        :param axes: the axes on which to plot (used in constructing a Graphics, None means make a new one)
        :type axes: None
        :param subplot_kw: the keywords to pass on when initializing the plot
        :type subplot_kw: dict | None
        :param colorbar: whether to use a colorbar or what options to pass to the colorbar
        :type colorbar: None | bool | dict
        :param opts: options to be fed in when initializing the Graphics
        :type opts:
        """

        super().__init__(figure=figure, axes=axes, subplot_kw=subplot_kw)
        if method is None:
           method = self.method
        if isinstance(method, str):
            method = getattr(self.axes, method)
        self._method = method

        # we're gonna set things up so that we can have delayed evaluation of the plotting.
        # i.e. a Plot3D can be initialized but then do all its plotting later
        if plot_style is None:
            plot_style = {}
        for k,v in self.style_mapping.items():
            if k in opts:
                opts[v] = opts[k]
                del opts[k]
        for k in self.known_styles:
            if k in opts:
                plot_style[k] = opts[k]
        for k in self.default_plot_style:
            if k not in plot_style:
                plot_style[k] = self.default_plot_style[k]
        self._plot_style = plot_style
        self.plot_style = plot_style
        self.plot_opts = opts
        self._colorbar = colorbar
        self._initialized = False

        if len(params) > 0:
            self.plot(*params)

    def _initialize(self):
        self._initialized = True
        self.set_options(**self.plot_opts)
        if self.colorbar:
            self.add_colorbar()
        elif isinstance(self.colorbar, dict):
            self.add_colorbar(**self.colorbar)

    def _get_plot_data(self, func, xrange, yrange):
        return _get_3D_plotdata(func, xrange, yrange)

    def _plot_data(self, *data, **plot_style):
        return self._method(*self._get_plot_data(*data), **plot_style)

    def plot(self, *params, **plot_style):
        plot_style = dict(self.plot_style, **plot_style)
        self.graphics = self._plot_data(*params, **plot_style)
        if not self._initialized:
            self._initialize()
        return self.graphics
    def add_colorbar(self, **kw):
        if self._initialized:
            fig = self.figure  # type: matplotlib.figure.Figure
            ax = self.axes  # type: matplotlib.axes.Axes
            return fig.colorbar(self.graphics, **kw)
        else:
            self._colorbar = kw
    plot_classes = {}
    @classmethod
    def resolve_method(cls, mpl_name):
        return cls.plot_classes[mpl_name]
    # @classmethod
    # def merge_plots(cls, *plots, **styles):
    #     ...
    @classmethod
    def register(cls, plot_class):
        cls.plot_classes[plot_class.method] = plot_class
        return plot_class

@Plot3D.register
class ScatterPlot3D(Plot3D):
    """
    Creates a ScatterPlot of 3D data
    """
    method = 'scatter'

@Plot3D.register
class WireframePlot3D(Plot3D):
    """
    Creates a Wireframe mesh plot of 3D data
    """
    method = 'plot_wireframe'

@Plot3D.register
class ContourPlot3D(Plot3D):
    """
    Creates a 3D ContourPlot of 3D data
    """
    method = 'contourf'

class ListPlot3D(Plot3D):
    """
    Convenience 3D plotting class that handles the interpolation first
    """
    def __init__(self,
                 *params,
                 plot_style=None,
                 method='contour',
                 colorbar=None,
                 figure=None,
                 axes=None,
                 subplot_kw=None,
                 interpolate=True,
                 **opts
                 ):
        """
        :param params: either _empty_ or and array of (_x_, _y_, _z_) points
        :type params:
        :param plot_style: the plot styling options to be fed into the plot method
        :type plot_style: dict | None
        :param method: the method name as a string
        :type method: str
        :param figure: the Graphics object on which to plot (None means make a new one)
        :type figure: Graphics | None
        :param axes: the axes on which to plot (used in constructing a Graphics, None means make a new one)
        :type axes: None
        :param subplot_kw: the keywords to pass on when initializing the plot
        :type subplot_kw: dict | None
        :param colorbar: whether to use a colorbar or what options to pass to the colorbar
        :type colorbar: None | bool | dict
        :param interpolate: whether to interpolate the data or not
        :type interpolate: bool
        :param opts: options to be fed in when initializing the Graphics
        :type opts:
        """
        self.interpolate = interpolate
        super().__init__(*params,
                         plot_style=plot_style, method=method,
                         colorbar=colorbar, figure=figure,
                         axes=axes, subplot_kw=subplot_kw,
                         **opts
                         )

    def _get_plot_data(self, *griddata, interpolate=None):
        if interpolate is None:
            interpolate = self.interpolate
        if len(griddata) == 3:
            x, y, z = griddata
        elif interpolate:
            x, y, z = _interp2DData(griddata[0])
        else:
            x = griddata[0][:, 0]
            y = griddata[0][:, 1]
            z = griddata[0][:, 2]

        return (x, y, z)

class ListTriPlot3D(ListPlot3D):
    """
    Creates a triangulated surface plot in 3D
    """
    method = 'plot_trisurf'
    default_plot_style = {'interpolate':False}


# add classes to __all__
for c in Plot.plot_classes.values():
    if c.__name__ not in __all__:
        __all__.append(c.__name__)
for c in Plot3D.plot_classes.values():
    if c.__name__ not in __all__:
        __all__.append(c.__name__)