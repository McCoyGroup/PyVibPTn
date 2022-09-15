"""
Graphics primitives module
Provides stuff like Disk, Sphere, etc. and lets them figure out how to plot themselves
"""

__all__ = [
    "GraphicsPrimitive", "Sphere", "Cylinder", "Disk", "Line", "Text", "Arrow", "Inset"
]

import abc, numpy as np
from .VTKInterface import *

class GraphicsPrimitive(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def plot(self, axes, *args, graphics=None, **kwargs):
        """The one method that needs to be implemented, which takes the graphics and actually puts stuff on its axes

        :param axes:
        :type axes:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        pass

    @abc.abstractmethod
    def get_bbox(self):
        ...

class Disk(GraphicsPrimitive):
    def __init__(self, position=(0, 0), radius=1, **opts):
        self.pos = position
        self.rad = radius
        self.opts = opts
        self.prim = None

    def get_bbox(self):
        return [(self.pos[0]-self.rad, self.pos[1]-self.rad), (self.pos[0]+self.rad, self.pos[1]+self.rad)]

    def plot(self, axes, *args, graphics=None, zdir = None, **kwargs):
        if isinstance(axes.figure, VTKWindow):
            self.prim = VTKDisk(self.pos, self.rad, **self.opts)
            s = self.prim
            return s.plot(axes.figure)
        else:
            if hasattr(axes, 'axes'):
                axes = axes.axes
            pt = np.array([self.pos]).T
            kw = dict(edgecolors = [ [.95]*3 +[.5] ])
            kw = dict(kw, **self.opts)
            kw = dict(kw, s=[(10*self.rad)**2], **kwargs)
            s = axes.scatter(*pt, **kw)
            print(pt, axes)

        return s

class Line(GraphicsPrimitive):
    def __init__(self, pos1, pos2, *rest, radius=.1, **opts):
        self.pos1 = pos1
        self.pos2 = pos2
        self.rest = rest
        self.rad = 72*radius # this can't be configured nearly as cleanly as the circle stuff...
        self.opts = opts
    @property
    def points(self):
        return [self.pos1, self.pos2, *self.rest]
    def get_bbox(self):
        pos = np.array(self.points).T
        return [(np.min(pos[0]), np.min(pos[1])), (np.max(pos[0]), np.max(pos[1]))]
    def plot(self, axes, *args, graphics=None, **kwargs):
        if isinstance(axes.figure, VTKWindow):
            if len(self.rest) > 0:
                raise NotImplementedError("...")
            self.prim = VTKLine(self.pos1, self.pos2, **self.opts)
            s = self.prim
            return s.plot(axes)
        else:
            if hasattr(axes, 'axes'):
                axes = axes.axes
            pos = np.array(self.points).T
            kw = dict(self.opts, linewidth = self.rad, **kwargs)
            line = axes.plot(*pos, **kw)

            return line


class Text(GraphicsPrimitive):
    def __init__(self, txt, pos, bbox=((1, 1), (1, 1)), **opts):
        self.txt = txt
        self.pos = pos
        self.bbox = bbox
        self.opts = opts
    def get_bbox(self):
        return [
            (self.pos[0]-self.bbox[0][0], self.pos[1]-self.bbox[1][0]),
            (self.pos[0]+self.bbox[0][1], self.pos[1]+self.bbox[1][1])
        ]
    def plot(self, axes, *args, graphics=None, **kwargs):
        if isinstance(axes.figure, VTKWindow):
            raise NotImplemented
        else:
            if hasattr(axes, 'axes'):
                axes = axes.axes
            return axes.text(*self.pos, self.txt, **self.opts)

class Arrow(GraphicsPrimitive):
    def __init__(self, pos1, pos2, **opts):
        self.pos1 = pos1
        self.pos2 = pos2
        self.opts = opts
    def get_bbox(self):
        pos = np.array([self.pos1, self.pos2]).T
        return [(np.min(pos[0]), np.min(pos[1])), (np.max(pos[0]), np.max(pos[1]))]
    def plot(self, axes, *args, graphics=None, **kwargs):
        if isinstance(axes.figure, VTKWindow):
            raise NotImplemented
        else:
            if hasattr(axes, 'axes'):
                axes = axes.axes
            return axes.arrow(*self.pos1, *(self.pos2 - self.pos1), **self.opts)

class Sphere(GraphicsPrimitive):
    def __init__(self, position = (0, 0, 0), radius = 1, sphere_points = 48, **opts):
        self.pos = position
        self.rad = radius
        self.opts = opts
        self.sphere_points = sphere_points

    def get_bbox(self):
        raise NotImplementedError("...")

    def plot(self, axes, *args, sphere_points=None, graphics=None, **kwargs):

        if isinstance(axes.figure, VTKWindow):
            self.prim = VTKSphere(self.pos, self.rad, **self.opts)
            s = self.prim
            return s.plot(axes.figure)
        else:
            if hasattr(axes, 'axes'):
                axes = axes.axes
            # create basic sphere points
            if sphere_points is None:
                sphere_points = self.sphere_points
            u = np.linspace(0, 2 * np.pi, sphere_points)
            v = np.linspace(0, np.pi, sphere_points)
            x = self.rad * np.outer(np.cos(u), np.sin(v))
            y = self.rad * np.outer(np.sin(u), np.sin(v))
            z = self.rad * np.outer(np.ones(np.size(u)), np.cos(v))

            kw = dict(self.opts, **kwargs)
            return axes.plot_surface(x + self.pos[0], y + self.pos[1], z + self.pos[2], *args, **kw)

class Cylinder(GraphicsPrimitive):
    def __init__(self, p1, p2, radius, circle_points = 32, **opts):
        self.pos1 = p1
        self.pos2 = p2
        self.rad = radius
        self.opts = opts
        self.circle_points = circle_points

    def get_bbox(self):
        raise NotImplementedError("...")

    def plot(self, axes, *args, circle_points=None, graphics=None, **kwargs):
        if isinstance(axes.figure, VTKWindow):
            self.prim = VTKCylinder(self.pos1, self.pos2, self.rad, **self.opts)
            s = self.prim
            return s.plot(axes.figure)
        else:
            if hasattr(axes, 'axes'):
                axes = axes.axes
            # create basic sphere points
            if circle_points is None:
                circle_points = self.circle_points
            u = np.linspace(0, 2 * np.pi, circle_points)
            v = np.linspace(0, np.pi, circle_points)

            # pulled from SO: https://stackoverflow.com/a/32383775/5720002

            #vector in direction of axis
            v = self.pos2 - self.pos1
            #find magnitude of vector
            mag = np.linalg.norm(v)
            #unit vector in direction of axis
            v = v / mag
            #make some vector not in the same direction as v
            not_v = np.array([1, 0, 0])
            if (v == not_v).all():
                not_v = np.array([0, 1, 0])
            #make vector perpendicular to v
            n1 = np.cross(v, not_v)
            #normalize n1
            n1 /= np.linalg.norm(n1)
            #make unit vector perpendicular to v and n1
            n2 = np.cross(v, n1)
            #surface ranges over t from 0 to length of axis and 0 to 2*pi
            t = np.linspace(0, mag, circle_points)
            theta = np.linspace(0, 2 * np.pi, circle_points)
            #use meshgrid to make 2d arrays
            t, theta = np.meshgrid(t, theta)
            #generate coordinates for surface
            X, Y, Z = [self.pos1[i] + v[i] * t + self.rad * np.sin(theta) * n1[i] + self.rad * np.cos(theta) * n2[i] for i in [0, 1, 2]]

            kw = dict(self.opts, **kwargs)
            return axes.plot_surface(X, Y, Z, *args, **kw)

class Inset(GraphicsPrimitive):
    def __init__(self, prims, position, offset=(.5, .5), dimensions=None, plot_range=None, **opts):
        self.prims = prims
        self.pos = position
        self.opts = opts
        self._plot_range = plot_range
        self._dimensions = dimensions
        self.offset = offset
        self._prim_cache = {}

    @property
    def plot_range(self):
        if self._plot_range is None:
            return self.get_plot_range()
        else:
            return self._plot_range
    @plot_range.setter
    def plot_range(self, pr):
        ((_, _), (_, _)) = pr
        self.plot_range = pr
    def get_plot_range(self):
        if len(self.prims) == 0:
            return [[0, 1], [0, 1]]
        [(plx, prx), (pby, pty)] = [[np.inf, -np.inf], [np.inf, -np.inf]]
        for g in self.prims:
            ((lx, by), (rx, ty)) = g.get_bbox()
            plx = min(lx, plx)
            prx = max(rx, prx)
            pby = min(by, pby)
            pty = max(ty,pty)
        return [(plx, prx), (pby, pty)]

    @property
    def dimensions(self):
        if self._dimensions is None:
            ((lx, rx), (by, ty)) = self.plot_range
            return (rx - lx, ty - by)
        else:
            dims = self._dimensions
            if dims[0] is None:
                ((lx, rx), (by, ty)) = self.plot_range
                w = (rx - lx)/(ty - by)*dims[1]
                dims = (w, dims[1])
            elif dims[1] is None:
                ((lx, rx), (by, ty)) = self.plot_range
                h = (ty - by)/(rx - lx)*dims[0]
                dims = (dims[0], h)
            return dims
    def get_bbox(self, graphics=None, preserve_aspect=None):
        w, h = self.dimensions
        if preserve_aspect is None and self._dimensions is not None:
            preserve_aspect = self._dimensions[0] is None or self._dimensions[1] is None
        if preserve_aspect and graphics is not None:
            ((lx, rx), (by, ty)) = graphics.plot_range
            gw = (rx - lx)
            gh = (ty - by)
            ar = graphics.aspect_ratio
            ((slx, srx), (sby, sty)) = self.plot_range
            sar = (sty - sby) / (srx - slx)
            if isinstance(ar, str) and ar == 'auto':
                w1, h1 = graphics.image_size
                ar = h1 / w1
            art = (gh/gw) / ar # the ratio of plot_range aspect to true aspect
            if self._dimensions[1] is None:
                h = w * (sar * art)
            else:
                w = h / (sar * art)

        ox, oy = self.offset
        x, y = self.pos
        bbox = [
            [x - ox * w, y - oy * h],
            [x + (1 - ox) * w, y + (1 - oy) * h],
        ]
        return bbox

    def get_axes(self, graphics, bbox=None, **opts):
        if bbox is None:
            bbox = self.get_bbox()
        if graphics.figure in self._prim_cache:
            self._prim_cache[graphics.figure].close()
        self._prim_cache[graphics.figure] = graphics.create_inset(bbox, **opts)
        return self._prim_cache[graphics.figure]

    def plot(self, axes, *args, graphics=None, **kwargs):
        if isinstance(axes.figure, VTKWindow):
            raise NotImplemented
        else:
            if graphics is None:
                graphics = axes
            bbox = self.get_bbox(graphics=graphics)
            g = self.get_axes(graphics, bbox, **self.opts)
            prims = [p.change_figure(g) if hasattr(p, 'change_figure') else p.plot(g) for p in self.prims]
            return prims

    # def __del__(self):
    #     if self._prim is not None:
    #         self._prim.remove()