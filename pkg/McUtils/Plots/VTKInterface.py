"""
Provides an interface to VTK that implements all necessary components with a matplotlib compatible API
"""

__all__ = [
    "VTKWindow",
    "VTKPrimitive",
    "VTKGeometricPrimitive",
    "VTKSphere",
    "VTKDisk",
    "VTKCylinder",
    "VTKLine",
    "VTKObject"
]

#####################################################################################################################
#
#                                           VTKObject
#
class VTKObject:
    """A general wrapper for _any_ vtk object that provides a more pythonic interface and can be extended to be add \
    layers of functionality

    :param obj: any kind of low-level vtk object that exposes a Set/Get interface
    :type obj: Any
    """

    def __init__(self, obj):
        if isinstance(obj, str):
            import vtk
            obj = getattr(vtk, 'vtk' + obj)()
        self.obj = obj

    def chain_to(self, src):
        self.obj.SetInputConnection(src.GetOutputPort())

    def __getattr__(self, item):

        attr = None
        retrieved = False
        try:
            attr = getattr(self.obj, item)
        except AttributeError:
            pass
        else:
            retrieved = True

        if not retrieved:
            raise AttributeError('{} has no attribute {}'.format(type(self).__name__, item))
        else:
            return attr


    @staticmethod
    def _HTMLColorToRGB(colorString):
        '''
        Convert #RRGGBB to a [R, G, B] list.
        :param: colorString a string in the form: #RRGGBB where RR, GG, BB are hexadecimal.
        The elements of the array rgb are unsigned chars (0..255).
        :return: The red, green and blue components as a list.
        '''
        colorString = colorString.strip()
        if colorString[0] == '#': colorString = colorString[1:]
        if len(colorString) != 6:
            raise ValueError("Input #%s is not in #RRGGBB format" % colorString)
        r, g, b = colorString[:2], colorString[2:4], colorString[4:]
        r, g, b = [int(n, 16) for n in (r, g, b)]

        return [r/255, g/255, b/255]

    @classmethod
    def color_tuple(cls, c):
        import vtk

        if isinstance(c, str) and c.startswith("#"):
            c = cls._HTMLColorToRGB(c)
        elif isinstance(c, str):
            colors = vtk.vtkNamedColors()
            c = colors.GetColor3d(c)
        else:
            if any(x > 1 for x in c):
                c = [ x / 255 for x in c ]
        return c

    def _get_prop(self, item, default_prefix = "Get"):
        """To simplify the API a bit we add property-like retrievers for a general Get/Set attribute structure"""

        retrieved = False
        prefixed = False
        attr = None
        try:
            attr = getattr(self.obj, item)
        except AttributeError:
            if not item.startswith(default_prefix):
                to_camel = item.split("_")
                to_camel = default_prefix + ''.join([(x[0].upper() + x[1:] if len(x) > 0 else x) for x in to_camel])
                try:
                    attr = getattr(self.obj, to_camel)
                except AttributeError:
                    pass
                else:
                    retrieved = True
                    prefixed = True
        else:
            retrieved = True

        return attr, retrieved, prefixed

    # defines a dict-like API to make it easy to set and get properties
    def get_prop(self, item):
        attr, retrieved, prefixed = self._get_prop(item, "Get")
        if prefixed and callable(attr): # call the getter
            attr = attr()

        return attr, retrieved
    def set_prop(self, item, val):
        attr, retrieved, prefixed = self._get_prop(item, "Set")
        if prefixed:
            attr(val)

        return attr, retrieved
    def _get_color(self):
        return self.obj.GetColor()
    def _get_background(self):
        return self.obj.GetBackground()
    _prop_getters = {
        'color': _get_color,
        'background': _get_background
    }
    def _set_color(self, c):
        try:
            self.obj.SetColor(self.color_tuple(c))
        except AttributeError:
            self["property"].SetColor(self.color_tuple(c))
    def _set_background(self, c):
        self.obj.SetBackground(self.color_tuple(c))
    _prop_setters = {
        'color': _set_color,
        'background': _set_background
    }
    def __getitem__(self, item):
        if item in self._prop_getters:
            return self._prop_getters[item](self)
        else:
            attr, retrieved = self.get_prop(item)
            if not retrieved:
                raise AttributeError('{} has no attribute {}'.format(type(self).__name__, item))
            else:
                return attr
    def __setitem__(self, key, value):
        if key in self._prop_setters:
            return self._prop_setters[key](self, value)
        else:
            self.set_prop(key, value)

class VTKRenderer(VTKObject):
    def __init__(self):
        super().__init__("Renderer")
class VTKRenderWindow(VTKObject):
    def __init__(self):
        super().__init__("RenderWindow")
class VTKRenderWindowInteractor(VTKObject):
    def __init__(self):
        super().__init__("RenderWindowInteractor")

class VTKActor(VTKObject):
    def __init__(self, actor = "", name = None):
        if isinstance(actor, str):
            import vtk

            name = actor if name is None else name
            actor = getattr(vtk, 'vtk' + actor + "Actor")()
        super().__init__(actor)
        self.name = name

class VTKWrapper:
    """Provides something that *looks* like a lower-level actor but is not
    Requires a setter, getter, and a property object
    """
    def __init__(self, get_val, set_val, prop):
        self._setter = set_val
        self._getter = get_val
        self._prop = prop

    @property
    def val(self):
        return self._getter()
    @val.setter
    def val(self, v):
        self._setter(v)

    def __getattr__(self, item):
        return getattr(self._prop, item)

class VTKTicks:
    def __init__(self, major_ticks, minor_ticks):
        ...

class VTKAxis:
    def __init__(self,
                 lines, title, label
                 ):
        self.lines = lines
        self.title = title
        self.label = label
        self.things = (self.lines, self.title, self.label)
    def set_prop(self, c, v):
        for thing in self.things:
            thing[c] = v
    def get_prop(self, c):
        return tuple(thing[c] for thing in self.things)
    def __getitem__(self, item):
        return self.get_prop(item)
    def __setitem__(self, key, value):
        return self.set_prop(key, value)

class VTKCubeAxes(VTKActor):
    def __init__(self):
        super().__init__("CubeAxes")
        self.SetFlyModeToStaticEdges()

        self._xaxis = None
        self._yaxis = None
        self._zaxis = None
        self.color = 'black'
    @property
    def x_axis(self):
        if self._xaxis is None:
            self._xaxis = VTKAxis(
                VTKObject(self["x_axes_lines_property"]),
                VTKObject(VTKWrapper(self.GetXTitle, self.SetXTitle, self.GetTitleTextProperty(0))),
                VTKObject(VTKWrapper(self.GetXLabelFormat, self.SetXLabelFormat, self.GetLabelTextProperty(0)))
            )
        return self._xaxis
    @property
    def y_axis(self):
        if self._yaxis is None:
            self._yaxis = VTKAxis(
                VTKObject(self["y_axes_lines_property"]),
                VTKObject(VTKWrapper(self.GetYTitle, self.SetYTitle, self.GetTitleTextProperty(1))),
                VTKObject(VTKWrapper(self.GetYLabelFormat, self.SetYLabelFormat, self.GetLabelTextProperty(1)))
            )
        return self._yaxis
    @property
    def z_axis(self):
        if self._zaxis is None:
            self._zaxis = VTKAxis(
                VTKObject(self["z_axes_lines_property"]),
                VTKObject(VTKWrapper(self.GetZTitle, self.SetZTitle, self.GetTitleTextProperty(2))),
                VTKObject(VTKWrapper(self.GetZLabelFormat, self.SetZLabelFormat, self.GetLabelTextProperty(2)))
            )
        return self._zaxis
    @property
    def axes(self):
        return [self.x_axis, self.y_axis, self.z_axis]
    @property
    def color(self):
        return [self.x_axis['color'][0], self.y_axis['color'][0], self.z_axis['color'][0]]
    @color.setter
    def color(self, colors):
        if isinstance(colors, str):
            colors = [colors] * 3
        self.x_axis['color'] = colors[0]
        self.y_axis['color'] = colors[1]
        self.z_axis['color'] = colors[2]

########################################################################################################################
#
#                                               VTKWindow
#
class VTKWindow:
    """Handles all communication with a vtkRenderWindow object
    Creates a vtkRenderer and vtkRenderInteractor and the rest of it to manage this

    """
    # this will take care of all of the low level communication with the window object

    def __init__(self,
                 title = None,
                 legend = None,
                 window = None,
                 cube = None,
                 use_axes = True,
                 interactor = None,
                 renderer = None,
                 background = 'white',
                 image_size = (640, 480),
                 viewpoint = (5, 5, 5),
                 focalpoint = (0, 0, 0),
                 scale = (1, 1, 1)
                 ):

        self._window = VTKObject("RenderWindow") if window is None else window
        self._axes = None
        self._objects = set()
        self._interactor = interactor
        self._renderer = renderer
        self._title = title
        self._legend = None
        self._size = image_size
        self._viewpoint = viewpoint
        self._focus = focalpoint
        self._scale = scale
        if background is not None:
            self.set_background(background)

        self.use_axes = use_axes
        self._axes = VTKCubeAxes() if cube is None else cube
        self.add_object(self._axes)
        if not use_axes:
            self._axes['x_axis_visibility'] = False
            self._axes['y_axis_visibility'] = False
            self._axes['z_axis_visibility'] = False

    def add_object(self, thing):
        if thing not in self._objects:
            self._objects.add(thing)

            if isinstance(thing, VTKObject):
                actors = [ thing ]
            else:
                try:
                    actors = thing.actor[0]
                except (TypeError, IndexError, KeyError):
                    actors = [thing.actor]

            for actor in actors:
                self.renderer.AddActor(actor.obj)

    def remove_object(self, thing):
        if thing in self._objects:
            self._objects.remove(thing)

            if isinstance(thing, VTKObject):
                actors = [ thing ]
            else:
                try:
                    actors = thing.actor[0]
                except (TypeError, IndexError, KeyError):
                    actors = [thing.actor]

            for actor in actors:
                self.renderer.RemoveActor(actor.obj)

    @property
    def window(self):
        return self._window

    @property
    def interactor(self):
        return self.setup_interactor()
    def setup_interactor(self):

        if self._interactor is None:
            self._interactor = VTKRenderWindowInteractor()
            self._interactor.SetRenderWindow(self.window.obj)

        return self._interactor

    @property
    def renderer(self):
        return self.setup_renderer()
    def setup_renderer(self):

        if self._renderer is None:
            self._renderer = VTKRenderer()
            self.window.AddRenderer(self.renderer.obj)

        return self._renderer

    def _not_imped_warning(self, method):
        import warnings
        warnings.warn("{} doesn't implement {} yet".format(type(self).__name__, method))

    #MPL-like API
    def set_size(self, w, h):
        import math
        self._size = [math.floor(w), math.floor(h)]
        self.window.SetSize(*self._size)
    def set_size_inches(self, wi, hi):
        return self.set_size(wi * 72, hi * 72)
    def get_size_inches(self):
        w,h = self.window.GetSize()
        return w/72, h/72

    def get_title(self):
        return self._title
    def set_title(self, title):
        self._title = title
        return self.window.SetWindowTitle(title)

    def get_lims(self):
        return self._axes.GetBounds()
    def get_xlim(self):
        # self._not_imped_warning('get_xlim')
        return self._axes.GetXAxisRange()
    def set_xlim(self, x):
        self._axes.SetXAxisRange(*x)
        bounds = list(self.get_lims())
        bounds[0] = x[0]
        bounds[1] = x[1]
        self._axes.SetBounds(*bounds)
    def get_ylim(self):
        # self._not_imped_warning('get_ylim')
        return self._axes.GetYAxisRange()
    def set_ylim(self, y):
        self._axes.SetYAxisRange(*y)
        bounds = list(self.get_lims())
        bounds[2] = y[0]
        bounds[3] = y[1]
        self._axes.SetBounds(*bounds)
    def get_zlim(self):
        # self._not_imped_warning('get_zlim')
        return self._axes.GetZAxisRange()
    def set_zlim(self, z):
        self._axes.SetZAxisRange(*z)
        bounds = list(self.get_lims())
        bounds[4] = z[0]
        bounds[5] = z[1]
        self._axes.SetBounds(*bounds)

    def get_xlabel(self):
        # self._not_imped_warning('get_xlabel')
        return self._axes.x_axis.label.val
    def set_xlabel(self, lab, **ops):
        # self._not_imped_warning('set_xlabel')
        x = self._axes.x_axis.label
        x.val = lab
        for k,v in ops.items():
            x[k] = v
    def get_ylabel(self):
        return self._axes.y_axis.label.val
    def set_ylabel(self, lab, **ops):
        x = self._axes.y_axis.label
        x.val = lab
        for k,v in ops.items():
            x[k] = v
    def get_zlabel(self):
        return self._axes.z_axis.label.val
    def set_zlabel(self, lab, **ops):
        x = self._axes.z_axis.label
        x.val = lab
        for k,v in ops.items():
            x[k] = v

    def get_xticks(self):
        # this is actually pretty tough to do...
        # self._not_imped_warning('get_xticks')
        return None
    def set_xticks(self, lab):
        # self._not_imped_warning('set_xticks')
        return None
    def get_yticks(self):
        # self._not_imped_warning('get_yticks')
        return None
    def set_yticks(self, lab):
        # self._not_imped_warning('set_yticks')
        return None
    def get_zticks(self):
        # self._not_imped_warning('get_zticks')
        return None
    def set_zticks(self, lab):
        # self._not_imped_warning('set_zticks')
        return None

    def set_model_matrix(self):
        # transform types
        scale = self._scale

        transform = VTKObject('Transform')
        transform.Scale(*scale)

        # handle other affine transforms here

        self.camera['model_tranform_matrix'] = transform['matrix']

    def get_xscale(self):
        # self._not_imped_warning('get_xscale')
        return self._scale[0]
    def set_xscale(self, lab):
        # self._not_imped_warning('set_xscale')
        self._scale[0] = lab
    def get_yscale(self):
        # self._not_imped_warning('get_yscale')
        return self._scale[1]
    def set_yscale(self, lab):
        # self._not_imped_warning('set_yscale')
        self._scale[1] = lab
    def get_zscale(self):
        # self._not_imped_warning('get_zscale')
        return self._scale[2]
    def set_zscale(self, lab):
        # self._not_imped_warning('set_zscale')
        self._scale[2] = lab

    def get_legend(self):
        return self._legend
    def set_legend(self, l):
        return self._not_imped_warning('set_legend')

    def set_facecolor(self, bg):
        self.renderer["background"] = bg
    @property
    def camera(self):
        return self.renderer["active_camera"]
    @camera.setter
    def camera(self, cam):
        self.renderer["active_camera"] = cam
    def set_viewpoint(self, vp):
        if vp is not None:
            self.camera.SetPosition(*vp)
    def set_focalpoint(self, fp):
        if fp is not None:
            self.camera.SetFocalPoint(*fp)
    def set_background(self, bg):
        self.set_facecolor(bg)

    def show(self):
        if self.use_axes:
            self._axes["camera"] = self.renderer["active_camera"]
        self.window.Render()
        self.set_size(*self._size)
        self.set_viewpoint(self._viewpoint)
        self.set_focalpoint(self._focus)
        self.interactor.Start()

#####################################################################################################################
#
#                                           VTKPrimitives
#
class VTKPrimitive:
    def __init__(self,
                 get_mapper,
                 name,
                 color = None,
                 parent = None
                 # we'll add other props as we come along them
                 ):
        self.name = name
        self.parent = parent
        self._mapper = None
        self._actor = None
        self._get_mapper = get_mapper
        self._color = None
        if color is not None:
            self.actor['color'] = color

    @property
    def actor(self):
        return self.get_actor()

    @staticmethod
    def _setup_actor(mapper):

        actor = VTKActor()
        actor['mapper'] = mapper.obj

        return actor
    @staticmethod
    def _setup_mapper(source):
        mapper = VTKObject("PolyDataMapper")
        mapper.chain_to(source)
        return mapper

    def get_actor(self):
        if self._actor is None:
            self._actor = self._setup_actor(self.get_mapper())
        return self._actor
    def get_mapper(self):
        if self._mapper is None:
            self._mapper = self._get_mapper()
        return self._mapper


    def plot(self, window):
        window.add_object(self)
        return [self]

class VTKGeometricPrimitive(VTKPrimitive):
    def __init__(self, source, name, **opts):
        self.source = source
        super().__init__(self._get_mapper, name, **opts)
        self.name = name

    def _get_mapper(self):
        return self._setup_mapper(self.source)

class VTKDisk(VTKGeometricPrimitive):
    def __init__(self, pos, rad, **opts):

        super().__init__(VTKObject("DiskSource"), "Disk", **opts)

        src = self.source
        # src.SetCenter(*pos)
        src.SetRadius(rad)

class VTKLine(VTKGeometricPrimitive):
    def __init__(self, pt1, pt2, **opts):

        super().__init__(VTKObject("LineSource"), "Line", **opts)

        self.source.SetPoint1(*pt1)
        self.source.SetPoint2(*pt2)


class VTKCylinder(VTKGeometricPrimitive):
    def __init__(self, pt1, pt2, rad, cylinder_points = 24, **opts):

        self._line = line = VTKObject("LineSource")
        line.SetPoint1(*pt1)
        line.SetPoint2(*pt2)

        tf = VTKObject("TubeFilter")
        tf.chain_to(line)
        tf["radius"] = rad
        tf['number_of_sides']  = cylinder_points
        tf.Update()
        super().__init__(tf, "Cylinder", **opts)

class VTKSphere(VTKGeometricPrimitive):
    def __init__(self, pos, rad,
                 theta_points = 24,
                 phi_points = 24,
                 **opts):

        super().__init__(
            VTKObject("SphereSource"),
            "Sphere",
            **opts
        )

        sphereSource = self.source
        sphereSource.SetCenter(*pos)
        sphereSource.SetRadius(rad)
        # Make the surface smooth.
        sphereSource.SetPhiResolution(phi_points)
        sphereSource.SetThetaResolution(theta_points)
