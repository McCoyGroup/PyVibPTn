"""
Handles all the nastiness of Matplotlib properties so that we can use a more classically python plotting method
"""

from .Styling import Styled

__all__ = [
    "GraphicsPropertyManager",
    "GraphicsPropertyManager3D"
]

class GraphicsPropertyManager:
    """
    Manages properties for Graphics objects so that concrete GraphicsBase instances don't need to duplicate code, but
    at the same time things that build off of GraphicsBase don't need to implement all of these properties
    """

    def __init__(self, graphics, figure, axes, managed=False):
        self.graphics = graphics
        self.managed = managed # if there's an external manager
        self.figure = figure
        self.axes = axes
        self._figure_label = None
        self._plot_label = None
        self._plot_legend = None
        self._axes_labels = None
        self._frame = None
        self._frame_style = None
        self._plot_range = None
        self._ticks = None
        self._scale = None
        self._ticks_style = None
        self._ticks_label_style = None
        self._aspect_ratio = None
        self._image_size = None
        self._padding = None
        self._background = None
        self._colorbar = None
        self._cbar_obj = None
        self._spacings = None

    @property
    def figure_label(self):
        return self._figure_label
    @figure_label.setter
    def figure_label(self, label):
        self._figure_label = label
        if not self.managed:
            if label is None:
                self.figure.suptitle("")
            elif isinstance(label, Styled):
                self.figure.suptitle(*label.val, **label.opts)
            else:
                self.figure.suptitle(label)

    @property
    def plot_label(self):
        return self._plot_label
    @plot_label.setter
    def plot_label(self, label):
        self._plot_label = label
        if label is None:
            self.axes.set_title("")
        elif isinstance(label, Styled):
            self.axes.set_title(*label.val, **label.opts)
        else:
            self.axes.set_title(label)

    # set plot legend
    @property
    def plot_legend(self):
        return self._plot_legend
    @plot_legend.setter
    def plot_legend(self, legend):
        self._plot_legend = legend
        if legend is None:
            self.axes.set_label("")
        elif isinstance(legend, Styled):
            self.axes.set_label(*legend.val, **legend.opts)
        else:
            self.axes.set_label(legend)

    # set axes labels
    @property
    def axes_labels(self):
        return self._axes_labels
    @axes_labels.setter
    def axes_labels(self, labels):
        if self._axes_labels is None:
            self._axes_labels = (self.axes.get_xlabel(), self.axes.get_ylabel())
        try:
            xlab, ylab = labels
        except ValueError:
            xlab, ylab = labels = (labels, self._axes_labels[1])

        self._axes_labels = tuple(labels)
        if xlab is None:
            self.axes.set_xlabel("")
        elif isinstance(xlab, Styled):
            self.axes.set_xlabel(*xlab.val, **xlab.opts)
        else:
            self.axes.set_xlabel(xlab)
        if ylab is None:
            self.axes.set_ylabel("")
        elif isinstance(ylab, Styled):
            self.axes.set_ylabel(*ylab.val, **ylab.opts)
        else:
            self.axes.set_ylabel(ylab)

    # set plot ranges
    @property
    def plot_range(self):
        if self._plot_range is None:
            pr = (self.axes.get_xlim(), self.axes.get_ylim())
        else:
            pr = self._plot_range
        return pr
    @plot_range.setter
    def plot_range(self, ranges):
        if self._plot_range is None:
            self._plot_range = (self.axes.get_xlim(), self.axes.get_ylim())
        try:
            x, y = ranges
        except ValueError:
            x, y = ranges = (self._plot_range[0], ranges)
        else:
            if isinstance(x, int) or isinstance(x, float):
                x, y = ranges = (self._plot_range[0], ranges)

        self._plot_range = tuple(ranges)

        if isinstance(x, Styled):  # name feels wrong here...
            self.axes.set_xlim(*x.val, **x.opts)
        elif x is not None:
            self.axes.set_xlim(x)
        if isinstance(y, Styled):
            self.axes.set_ylim(*y.val, **y.opts)
        elif y is not None:
            self.axes.set_ylim(y)

    # set plot ticks
    @property
    def ticks(self):
        return self._ticks
    def _set_ticks(self, x, set_ticks=None, set_locator=None, set_minor_locator=None, **opts):
        import matplotlib.ticker as ticks

        if isinstance(x, Styled):
            self._set_ticks(*x.val,
                            set_ticks=set_ticks,
                            set_locator=set_locator, set_minor_locator=set_minor_locator,
                            **x.opts
                            )
        elif isinstance(x, ticks.Locator):
            set_locator(x)
        elif isinstance(x, (list, tuple)):
            if len(x) == 2 and isinstance(x[0], (list, tuple)):
                self.axes.set_xticks(*x, **opts)
            elif len(x) == 2 and isinstance(x[0], ticks.Locator):
                set_locator(x[0])
                set_minor_locator(x[1])
        elif isinstance(x, (float, int)):
            set_ticks(ticks.MultipleLocator(x), **opts)
        elif x is not None:
            set_ticks(x, **opts)
    def _set_xticks(self, x, **opts):
        return self._set_ticks(x,
                               set_ticks=self.axes.set_xticks,
                               set_locator=self.axes.xaxis.set_major_locator,
                               set_minor_locator=self.axes.xaxis.set_minor_locator,
                               **opts
                               )

    def _set_yticks(self, y, **opts):
        return self._set_ticks(y,
                               set_ticks=self.axes.set_yticks,
                               set_locator=self.axes.yaxis.set_major_locator,
                               set_minor_locator=self.axes.yaxis.set_minor_locator,
                               **opts
                               )

    @ticks.setter
    def ticks(self, ticks):
        if self._ticks is None:
            self._ticks = (self.axes.get_xticks(), self.axes.get_yticks())
        try:
            x, y = ticks
        except (ValueError, TypeError):
            if isinstance(ticks, bool):
                x, y = ticks = (ticks, ticks)
            else:
                x, y = ticks = (self._ticks[0], ticks)

        self._ticks = ticks
        self._set_xticks(x)
        self._set_yticks(y)

    @property
    def ticks_style(self):
        return self._ticks_style
    @ticks_style.setter
    def ticks_style(self, ticks_style):
        if self._ticks_style is None:
            self._ticks_style = (None,) * 2
        if isinstance(ticks_style, dict):
            ticks_style = (ticks_style, ticks_style)
        try:
            x, y = ticks_style
        except ValueError:
            x, y = ticks_style = (ticks_style, ticks_style)
        self._ticks_style = ticks_style
        if x is not None:
            if x is True:
                x = dict(bottom=True, labelbottom=True)
            elif x is False:
                x = dict(bottom=False, top=False, labelbottom=False, labeltop=False)
            self.axes.tick_params(
                axis='x',
                **x
            )
        if y is not None:
            if y is True:
                y = dict(left=True, labelleft=True)
            elif y is False:
                y = dict(left=False, right=False, labelleft=False, labelright=False)
            self.axes.tick_params(
                axis='y',
                **y
            )
    @property
    def frame_style(self):
        return self._frame_style
    @frame_style.setter
    def frame_style(self, f_style):
        if self._frame_style is None:
            self._frame_style = ((None,) * 2)*2
        if isinstance(f_style, dict):
            f_style = ((f_style, f_style), (f_style, f_style))
        try:
            x, y = f_style
        except ValueError:
            x, y = f_style = (f_style, f_style)
        if isinstance(y, dict):
            y = (y, y)
        if isinstance(x, dict):
            x = (x, x)
        if len(y) == 2:
            b, t = y
        else:
            b = t = y
        if len(x) == 2:
            l, r = x
        else:
            l = r = x

        self._ticks_label_style = ((l, r), (b, t))
        lax, rax, bax, tax = self.axes.spines.values()
        for a,o in zip((lax, rax, bax, tax), (l, r, b, t)):
            if o is not None:
                a.set(**o)

    @property
    def ticks_label_style(self):
        return self._ticks_label_style
    @ticks_label_style.setter
    def ticks_label_style(self, ticks_style):
        if self._ticks_label_style is None:
            self._ticks_label_style = (None,) * 2
        try:
            x, y = ticks_style
        except ValueError:
            x, y = ticks_style = (ticks_style, ticks_style)
        self._ticks_label_style = ticks_style
        if x is not None:
            self.axes.set_xticklabels(
                self.axes.get_xticklabels(),
                **x
            )
        if y is not None:
            self.axes.set_yticklabels(
                self.axes.get_yticklabels(),
                **y
            )

    @property
    def aspect_ratio(self):
        return self._aspect_ratio
    @aspect_ratio.setter
    def aspect_ratio(self, ar):
        if isinstance(ar, (float, int)):
            a, b = self.plot_range
            cur_ar = abs(b[1] - b[0]) / abs(a[1] - a[0])
            targ_ar = ar / cur_ar
            self.axes.set_aspect(targ_ar)
        elif isinstance(ar, str):
            self.axes.set_aspect(ar)
        else:
            self.axes.set_aspect(ar[0], **ar[1])
        self._aspect_ratio = ar

    # set size
    @property
    def image_size(self):
        # im_size = self._image_size
        # if isinstance(self._image_size, (int, float)):
        #     im_size =
        return self._image_size
    @image_size.setter
    def image_size(self, wh):
        if self._image_size is None:
            self._image_size = tuple(s * 72. for s in self.figure.get_size_inches())

        try:
            w, h = wh
        except (TypeError, ValueError):
            ar = self.aspect_ratio
            if not isinstance(ar, (int, float)):
                try:
                    ar = self._image_size[1] / self._image_size[0]
                except TypeError:
                    ar = 1
            w, h = wh = (wh, ar * wh)

        if w is not None or h is not None:
            if w is None:
                w = self._image_size[0]
            if h is None:
                h = self._image_size[1]

            if w > 72:
                wi = w / 72
            else:
                wi = w
                w = 72 * w

            if h > 72:
                hi = h / 72
            else:
                hi = h
                h = 72 * h

            self._image_size = (w, h)
            if not self.managed:
                self.figure.set_size_inches(wi, hi)

    # set background color
    @property
    def background(self):
        if self._background is None:
            self._background = self.figure.get_facecolor()
        return self._background

    @background.setter
    def background(self, bg):
        self._background = bg
        if not self.managed:
            self.figure.set_facecolor(bg)
        self.axes.set_facecolor(bg)

    # set show_frame
    @property
    def frame(self):
        return self._frame
    @frame.setter
    def frame(self, fr):
        self._frame = fr
        if fr is True or fr is False:
            self.axes.set_frame_on(fr)
        else:
            lr, bt = fr
            if len(lr) == 2:
                l, r = lr
            else:
                l = lr; r = lr
            if len(bt) == 2:
                b, t = bt
            else:
                b = bt; t = bt
            self.axes.spines['left'].set_visible(l); self.axes.spines['right'].set_visible(r)
            self.axes.spines['bottom'].set_visible(b); self.axes.spines['top'].set_visible(t)

    @property
    def scale(self):
        return self._scale
    @scale.setter
    def scale(self, scales):
        if self._scale is None:
            self._scale = (self.axes.get_xscale(), self.axes.get_yscale())
        try:
            x, y = scales
        except ValueError:
            x, y = scales = (self._scale[0], scales)

        self._scale = tuple(scales)

        if isinstance(x, Styled):
            self.axes.set_xscale(*x.val, **x.opts)
        elif x is not None:
            self.axes.set_xscale(x)
        if isinstance(y, Styled):
            self.axes.set_yscale(*y.val, **y.opts)
        elif y is not None:
            self.axes.set_yscale(y)

    @property
    def padding(self):
        return self._padding
    @padding.setter
    def padding(self, padding):
        try:
            w, h = padding
        except (ValueError, TypeError):
            w = h = padding
        try:
            wx, wy = w
        except (ValueError, TypeError):
            wx = wy = w
        try:
            hx, hy = h
        except (ValueError, TypeError):
            hx = hy = h

        W, H = self.image_size
        if wx < 1:
            wx = round(wx * W)
        if wy < 1:
            wy = round(wy * W)
        if hx < 1:
            hx = round(hx * H)
        if hy < 1:
            hy = round(hy * H)
        self._padding = ((wx, wy), (hx, hy))
        wx = wx / W; wy = wy / W
        hx = hx / H; hy = hy / H
        if not self.managed:
            # print(wx, wy, hx, hy)
            self.figure.subplots_adjust(left=wx, right=1 - wy, bottom=hx, top=1 - hy)
    @property
    def padding_left(self):
        return self._padding[0][0]
    @padding_left.setter
    def padding_left(self, p):
        wx, wy = self._padding[0]
        hx, hy = self._padding[1]
        self.padding = ((p, wy), (hx, hy))
    @property
    def padding_right(self):
        return self._padding[0][1]
    @padding_right.setter
    def padding_right(self, p):
        wx, wy = self._padding[0]
        hx, hy = self._padding[1]
        self.padding = ((wx, p), (hx, hy))
    @property
    def padding_top(self):
        return self._padding[1][1]
    @padding_top.setter
    def padding_top(self, p):
        wx, wy = self._padding[0]
        hx, hy = self._padding[1]
        self.padding = ((wx, wy), (hx, p))
    @property
    def padding_bottom(self):
        return self._padding[1][0]
    @padding_bottom.setter
    def padding_bottom(self, p):
        wx, wy = self._padding[0]
        hx, hy = self._padding[1]
        self.padding = ((wx, wy), (p, hy))

    @property
    def spacings(self):
        return self._spacings
    @spacings.setter
    def spacings(self, spacings):
        try:
            w, h = spacings
        except ValueError:
            w = h = spacings

        W, H = self.image_size
        if w < 1:
            wp = round(w * W)
        else:
            wp = w
        if h < 1:
            hp = round(h * H)
        else:
            hp = h
        self._spacings = (wp, hp)

        w = wp / W
        h = hp / H
        if not self.managed:
            self.figure.subplots_adjust(wspace=w, hspace=h)


    @property
    def colorbar(self):
        return self._colorbar
    @colorbar.setter
    def colorbar(self, c):
        self._colorbar = c
        # if self._cbar_obj is not None:
        #     self.graphics.remove(self._cbar_obj)
        if self._cbar_obj is None:
            if self._colorbar is True:
                self._cbar_obj = self.graphics.add_colorbar()
            elif isinstance(self._colorbar, dict):
                self._cbar_obj = self.graphics.add_colorbar(**self.colorbar)
        elif self._colorbar is None:
            pass
            #self.graphics.remove(self._cbar_obj)

class GraphicsPropertyManager3D(GraphicsPropertyManager):
    def __init__(self, graphics, figure, axes, managed=False):
        super().__init__(graphics, figure, axes, managed=managed)
        self._view_settings = None

    @property
    def axes_labels(self):
        return self._axes_labels
    @axes_labels.setter
    def axes_labels(self, labels):
        if self._axes_labels is None:
            self._axes_labels = (self.axes.get_xlabel(), self.axes.get_ylabel(), self.axes.get_zlabel())
        try:
            xlab, ylab, zlab = labels
        except ValueError:
            xlab, ylab, zlab = labels = (labels, self._axes_labels[1], self._axes_labels[2])

        self._axes_labels = tuple(labels)
        if xlab is None:
            self.axes.set_xlabel("")
        elif isinstance(xlab, Styled):
            self.axes.set_xlabel(*xlab.val, **xlab.opts)
        else:
            self.axes.set_xlabel(xlab)

        if ylab is None:
            self.axes.set_ylabel("")
        elif isinstance(ylab, Styled):
            self.axes.set_ylabel(*ylab.val, **ylab.opts)
        else:
            self.axes.set_ylabel(ylab)

        if zlab is None:
            self.axes.set_zlabel("")
        elif isinstance(zlab, Styled):
            self.axes.set_zlabel(*zlab.val, **zlab.opts)
        else:
            self.axes.set_zlabel(zlab)

    @property
    def plot_range(self):
        if self._plot_range is None:
            pr = (self.axes.get_xlim(), self.axes.get_ylim())
        else:
            pr = self._plot_range
        return pr
    @plot_range.setter
    def plot_range(self, ranges):
        if self._plot_range is None:
            self._plot_range = (self.axes.get_xlim(), self.axes.get_ylim(), self.axes.get_zlim())

        try:
            x, y, z = ranges
        except ValueError:
            x, y, z = ranges = (self._plot_range[0], self._plot_range[1], ranges)
        else:
            if isinstance(x, int) or isinstance(x, float):
                x, y, z = ranges = (self._plot_range[0], self._plot_range[1], ranges)

        self._plot_range = tuple(ranges)

        if isinstance(x, Styled):  # name feels wrong here...
            self.axes.set_xlim(*x.val, **x.opts)
        elif x is not None:
            self.axes.set_xlim(x)
        if isinstance(y, Styled):
            self.axes.set_ylim(*y.val, **y.opts)
        elif y is not None:
            self.axes.set_ylim(y)
        if isinstance(z, Styled):
            self.axes.set_zlim(*z.val, **z.opts)
        elif z is not None:
            self.axes.set_zlim(z)

    def _set_xticks(self, x, **opts):
        return self._set_ticks(x,
                               set_ticks=self.axes.set_xticks,
                               set_locator=self.axes.xaxis.set_major_locator,
                               set_minor_locator=self.axes.xaxis.set_minor_locator,
                               **opts
                               )
    def _set_yticks(self, y, **opts):
        return self._set_ticks(y,
                               set_ticks=self.axes.set_yticks,
                               set_locator=self.axes.yaxis.set_major_locator,
                               set_minor_locator=self.axes.yaxis.set_minor_locator,
                               **opts
                               )
    def _set_zticks(self, z, **opts):
        return self._set_ticks(z,
                               set_ticks=self.axes.set_zticks,
                               set_locator=self.axes.zaxis.set_major_locator,
                               set_minor_locator=self.axes.zaxis.set_minor_locator,
                               **opts
                               )

    @property
    def ticks(self):
        return self._ticks
    @ticks.setter
    def ticks(self, ticks):
        if self._ticks is None:
            self._ticks = (self.axes.get_xticks(), self.axes.get_yticks(), self.axes.get_zticks())
        try:
            x, y, z = ticks
        except ValueError:
            x, y, z = ticks = (self._ticks[0], self._ticks[1], ticks)

        self._ticks = ticks

        self._set_xticks(x)
        self._set_yticks(y)
        self._set_zticks(z)

    @property
    def ticks_style(self):
        return self._ticks_style
    @ticks_style.setter
    def ticks_style(self, ticks_style):
        if self._ticks_style is None:
            self._ticks_style = (None,)*3
        try:
            x, y, z = ticks_style
        except ValueError:
            x, y, z = ticks_style = (self._ticks_style[0], self._ticks_style[1], ticks_style)
        self._ticks_style = ticks_style
        if x is not None:
            self.axes.tick_params(
                axis='x',
                **x
            )
        if y is not None:
            self.axes.tick_params(
                axis='y',
                **y
            )
        if z is not None:
            self.axes.tick_params(
                axis='z',
                **z
            )

    @property
    def view_settings(self):
        return {'elev': self.axes.elev, 'azim':self.axes.azim}
    @view_settings.setter
    def view_settings(self, value):
        if isinstance(value, dict):
            if 'elev' not in value:
                value['elev'] = self.axes.elev
            if 'azim' not in value:
                value['azim'] = self.axes.azim
        else:
            value = dict(zip(['elev', 'azim'], value))
        self.axes.view_init(elev=value['elev'], azim=value['azim'])