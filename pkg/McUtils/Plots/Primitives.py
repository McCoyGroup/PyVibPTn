"""
Graphics primitives module
Provides stuff like Disk, Sphere, etc. and lets them figure out how to plot themselves
"""

__all__ = [
    "GraphicsPrimitive", "Sphere", "Cylinder", "Disk", "Line"
]

import abc
from .VTKInterface import *

class GraphicsPrimitive(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def plot(self, graphics, *args, **kwargs):
        """The one method that needs to be implemented, which takes the graphics and actually puts stuff on its axes

        :param graphics:
        :type graphics:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        pass

class Disk(GraphicsPrimitive):
    def __init__(self, position = (0, 0), radius = 1, **opts):
        self.pos = position
        self.rad = (100*radius)**2
        self.opts = opts
        self.prim = None

    def plot(self, graphics, *args, zdir = None, **kwargs):

        if isinstance(graphics.figure, VTKWindow):
            self.prim = VTKDisk(self.pos, self.rad, **self.opts)
            s = self.prim
            return s.plot(graphics.figure)
        else:
            import numpy as np
            pt = np.array([self.pos]).T
            kw = dict(self.opts, s = [ self.rad ], edgecolors = [ [.95]*3 +[.5] ], **kwargs)
            s = graphics.scatter(*pt, **kw)

        return s

class Line(GraphicsPrimitive):
    def __init__(self, pos1, pos2, radius, **opts):
        self.pos1 = pos1
        self.pos2 = pos2
        self.rad = 72*radius # this can't be configured nearly as cleanly as the circle stuff...
        self.opts = opts
    def plot(self, graphics, *args, **kwargs):
        if isinstance(graphics.figure, VTKWindow):
            self.prim = VTKLine(self.pos1, self.pos2, **self.opts)
            s = self.prim
            return s.plot(graphics)
        else:
            import numpy as np
            pos = np.array([self.pos1, self.pos2]).T

            kw = dict(self.opts, linewidth = self.rad, **kwargs)
            line = graphics.plot(*pos, **kw)

            return line

class Text(GraphicsPrimitive):
    def __init__(self, txt, pos, **opts):
        self.txt = txt
        self.pos = pos
        self.opts = opts
    def plot(self, graphics, *args, **kwargs):
        if isinstance(graphics.figure, VTKWindow):
            raise NotImplemented
        else:
            return graphics.axes.arrow(self.txt, *self.pos, **self.opts)

class Arrow(GraphicsPrimitive):
    def __init__(self, pos1, pos2, **opts):
        self.pos1 = pos1
        self.pos2 = pos2
        self.opts = opts
    def plot(self, graphics, *args, **kwargs):
        if isinstance(graphics.figure, VTKWindow):
            raise NotImplemented
        else:
            return graphics.axes.arrow(*self.pos1, *(self.pos2 - self.pos1), **self.opts)

class Sphere(GraphicsPrimitive):
    def __init__(self, position = (0, 0, 0), radius = 1, sphere_points = 48, **opts):
        self.pos = position
        self.rad = radius
        self.opts = opts
        self.sphere_points = sphere_points

    def plot(self, graphics, *args, sphere_points = None, **kwargs):
        if isinstance(graphics.figure, VTKWindow):
            self.prim = VTKSphere(self.pos, self.rad, **self.opts)
            s = self.prim
            return s.plot(graphics.figure)
        else:
            import numpy as np
            # create basic sphere points
            if sphere_points is None:
                sphere_points = self.sphere_points
            u = np.linspace(0, 2 * np.pi, sphere_points)
            v = np.linspace(0, np.pi, sphere_points)
            x = self.rad * np.outer(np.cos(u), np.sin(v))
            y = self.rad * np.outer(np.sin(u), np.sin(v))
            z = self.rad * np.outer(np.ones(np.size(u)), np.cos(v))

            kw = dict(self.opts, **kwargs)
            return graphics.plot_surface(x+self.pos[0], y+self.pos[1], z+self.pos[2], *args, **kw)

class Cylinder(GraphicsPrimitive):
    def __init__(self, p1, p2, radius, circle_points = 32, **opts):
        self.pos1 = p1
        self.pos2 = p2
        self.rad = radius
        self.opts = opts
        self.circle_points = circle_points

    def plot(self, graphics, *args, circle_points = None, **kwargs):
        if isinstance(graphics.figure, VTKWindow):
            self.prim = VTKCylinder(self.pos1, self.pos2, self.rad, **self.opts)
            s = self.prim
            return s.plot(graphics.figure)
        else:
            import numpy as np
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
            return graphics.plot_surface(X, Y, Z, *args, **kw)