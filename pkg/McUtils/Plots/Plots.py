"""
Provides various types of plots and plotting utilities
"""

from .Graphics import Graphics, Graphics3D, GraphicsGrid
import numpy as np
import matplotlib.figure
import matplotlib.axes

__all__ = [
    "Plot", "ScatterPlot", "ErrorBarPlot", "ListErrorBarPlot", "StickPlot", "ListStickPlot", "TriPlot", "ListTriPlot",
    "DataPlot", "HistogramPlot", "HistogramPlot2D", "VerticalLinePlot", "ArrayPlot", "TensorPlot",
    "Plot2D", "ListPlot2D", "ContourPlot", "DensityPlot", "ListContourPlot", "ListDensityPlot",
    "ListTriContourPlot", "ListTriDensityPlot", "ListTriPlot3D",
    "Plot3D", "ListPlot3D", "ScatterPlot3D", "WireframePlot3D", "ContourPlot3D"
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


def _semi_adaptive_sample_func(f, xmin, xmax, npts=150, max_refines=10, der_cut=10 ^ 5):

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
    if not callable(func):
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
    default_plot_style = {}
    def __init__(self,
                 *params,
                 method='plot',
                 figure=None, axes=None, subplot_kw=None,
                 plot_style=None, theme=None,
                 **opts
                 ):
        """
        :param params: _empty_ or _x_, _y_ arrays or _function_, _xrange_
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

        self.graphics = None

        # we're gonna set things up so that we can have delayed evaluation of the plotting.
        # i.e. a Plot can be initialized but then do all its plotting later
        if plot_style is None:
            plot_style = {}
        for k in self.default_plot_style:
            if k not in plot_style:
                plot_style[k] = self.default_plot_style[k]
        self._plot_style = plot_style
        self.plot_opts = opts
        self._initialized = False
        self._data = None

        super().__init__(figure=figure, axes=axes, theme=theme, subplot_kw=subplot_kw, **opts)
        self.method = getattr(self.axes, method)

        if len(params) > 0:
            self.plot(*params)

    def _initialize(self):
        self._initialized = True
        self.set_options(**self.plot_opts)

    def _get_plot_data(self, func, xrange):
        xrange, fvalues = _get_2D_plotdata(func, xrange)
        return xrange, fvalues

    def _plot_data(self, *data, **plot_style):
        return self.method(*self._get_plot_data(*data), **plot_style)

    def plot(self, *params, **plot_style):
        """
        Plots a set of data & stores the result
        :return: the graphics that matplotlib made
        :rtype:
        """
        plot_style = dict(self.plot_style, **plot_style)
        self._data = (params, plot_style)
        self.graphics = self._plot_data(*params, **plot_style)
        if not self._initialized:
            self._initialize()
        return self.graphics

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
            return super().add_colorbar(graphics = graphics, **kw)

class ScatterPlot(Plot):
    """
    Inherits from `Plot`.
    Plots a bunch of x values against a bunch of y values using the `scatter` method.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, method="scatter", **kwargs)

class ListScatterPlot(ScatterPlot):
    """
    Inherits from `Plot`.
    Plots a bunch of (x, y) points using the `scatter` method.
    """
    def __init__(self, griddata, **opts):
        super().__init__(griddata[:, 0], griddata[:, 1], **opts)

class ErrorBarPlot(Plot):
    """
    Inherits from `Plot`.
    Plots error bars using the `errorbar` method.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, method="errorbar", **kwargs)


class ListErrorBarPlot(ErrorBarPlot):
    """A Plot that pulls the errorbar data from a list"""
    def __init__(self, griddata, **opts):
        super().__init__(griddata[:, 0], griddata[:, 1], **opts)


class StickPlot(Plot):
    """A Plot object that plots sticks"""

    default_plot_style = {'basefmt': " ", 'use_line_collection':True, 'markerfmt': " "}
    def __init__(self, *args, **kwargs):
        super().__init__(*args, method="stem", **kwargs)


class ListStickPlot(StickPlot):
    """A Plot object that plots sticks from a list"""
    def __init__(self, griddata, **opts):
        super().__init__(griddata[:, 0], griddata[:, 1], **opts)


class TriPlot(Plot):
    """A Plot object that plots a triangulation bars"""
    def __init__(self, *args, **opts):
        super().__init__(args, method='triplot', **opts)


class ListTriPlot(TriPlot):
    """A Plot that pulls the triangulation data from a list"""
    def __init__(self, griddata, **opts):
        super().__init__(griddata[:, 0], griddata[:, 1], **opts)


######################################################################################################
#
#                                    Pure Data Plots on 2D Axes
#
#
class DataPlot(Plot):
    """
    Makes a 2D plot of arbitrary data using a plot method that handles that data type
    """
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


class HistogramPlot(DataPlot):
    """
    Makes a Histogram of data
    """
    def __init__(*args, **kwargs):
        super().__init__(*args, method='hist', **kwargs)


class HistogramPlot2D(DataPlot):
    """
    Makes a 2D histogram of data
    """
    def __init__(*args, **kwargs):
        super().__init__(*args, method='hist2d', **kwargs)


class VerticalLinePlot(Plot):
    """
    Plots a bunch of vertical lines
    """
    def __init__(*args, **kwargs):
        super().__init__(*args, method='vlines', **kwargs)

    def _get_plot_data(self, x, y=1.0):
        if isinstance(y, (int, float)):
            y = [0, y]
        return (x, y)

    def _plot_data(self, *data, **plot_style):
        x, y = data
        return self.method(x, *y, **plot_style)


class ArrayPlot(DataPlot):
    """
    Plots an array as an image
    """
    def __init__(self, *params,
                 plot_style=None, colorbar=None,
                 figure=None, axes=None, subplot_kw=None,
                 method='imshow',
                 **opts
                 ):
        super().__init__(*params,
                         plot_style=plot_style, method=method,
                         colorbar=colorbar, figure=figure,
                         axes=axes, subplot_kw=subplot_kw,
                         **opts
                         )
    def _get_plot_data(self, data):
        if hasattr(data, 'toarray'):
            data = data.toarray()
        return data,


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
    def __init__(self, *params,
                 plot_style=None,
                 method='contour',
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
                         plot_style=plot_style, method=method,
                         colorbar=colorbar, figure=figure,
                         axes=axes, subplot_kw=subplot_kw,
                         **opts
                         )

    def _get_plot_data(self, func, xrange, yrange):
        return _get_3D_plotdata(func, xrange, yrange)


class ContourPlot(Plot2D):
    def __init__(self, *params, **opts):
        super().__init__(*params, method='contourf', **opts)


class DensityPlot(Plot2D):
    def __init__(self, *params, **opts):
        super().__init__(*params, method='pcolormesh', **opts)


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


class ListContourPlot(ListPlot2D):
    def __init__(self, griddata, **opts):
        super().__init__(griddata, method='contourf', **opts)


class ListDensityPlot(ListPlot2D):
    def __init__(self, griddata, **opts):
        super().__init__(griddata, method='pcolormesh', **opts)


class ListTriContourPlot(ListPlot2D):
    def __init__(self, griddata, **opts):
        super().__init__(griddata, method='tricontourf', interpolate=False, **opts)


class ListTriDensityPlot(ListPlot2D):
    def __init__(self, griddata, **opts):
        super().__init__(griddata, method='tripcolor', interpolate=False, **opts)


######################################################################################################
#
#                                    3D Plots on 3D Axes
#
#
class Plot3D(Graphics3D):  # basically a mimic of the Plot class but inheriting from Graphics3D
    """A base class for 3D plots"""
    def __init__(self, *params,
                 plot_style=None,
                 method='plot_surface', colorbar=None,
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
        self.method = getattr(self, method)

        # we're gonna set things up so that we can have delayed evaluation of the plotting.
        # i.e. a Plot3D can be initialized but then do all its plotting later
        if plot_style is None:
            plot_style = {}
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
        return self.method(*self._get_plot_data(*data), **plot_style)

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

class ScatterPlot3D(Plot3D):
    """
    Creates a ScatterPlot of 3D data
    """
    def __init__(self, *params, **opts):
        super().__init__(*params, method='scatter', **opts)


class WireframePlot3D(Plot3D):
    """
    Creates a Wireframe mesh plot of 3D data
    """
    def __init__(self, *params, **opts):
        super().__init__(*params, method='plot_wireframe', **opts)


class ContourPlot3D(Plot3D):
    """
    Creates a 3D ContourPlot of 3D data
    """
    def __init__(self, *params, **opts):
        super().__init__(*params, method='contourf', **opts)


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
    def __init__(self, *params, **opts):
        super().__init__(*params, method='plot_trisurf', interpolate=False, **opts)

