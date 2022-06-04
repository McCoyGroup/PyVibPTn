"""
A plotting framework that builds off of `matplotlib`, but potentially could use a different backend.
The design is intended to mirror the `Graphics` framework in Mathematica and where possible option
names have been chosen to be the same.
Difficulties with `matplotlib` prevent a perfect mirror but the design is consistent.
There are a few primary divisions:
1. `Graphics`/`Graphics3D`/`GraphicsGrid` provide basic access to `matplotlib.figure` and `matplotlib.axes`
    they also hold a `GraphicsPropertyManager`/`GraphicsPropertyManager3D` that manages all properties
    (`image_size`, `axes_label`, `ticks_style`, etc.).
    The full lists can be found on the relevant reference pages and are bound as `properties` on the
    `Graphics`/`Graphics3D` instances.
2. `Plot/Plot3D` and everything in the `Plots` subpackage provide concrete instances of common plots
    with nice names/consistent with Mathematica for discoverability but primarily fall back onto
    `matplotlib` built-in methods and then allow for restyling/data reuse, etc.
3. `Primitives` provide direct access to the shapes that are actually plotted on screen (i.e. `matplotlib.Patch` objects)
    in a convenient way to add on to existing plots
4. `Styling` provides access to theme management/construction

Image/animation support and other back end support for 3D graphics (`VTK`) are provided at the experimental level.
"""

from .Graphics import *
from .Plots import *
from .Primitives import *
from .Interactive import *
from .Styling import *
from .Image import *
from .Properties import *

__all__ = []
from .Graphics import __all__ as exposed
__all__ += exposed
from .Plots import __all__ as exposed
__all__ += exposed
from .Primitives import __all__ as exposed
__all__ += exposed
from .Interactive import __all__ as exposed
__all__ += exposed
from .Styling import __all__ as exposed
__all__ += exposed
from .Image import __all__ as exposed
__all__ += exposed
from .Properties import __all__ as exposed
__all__ += exposed
