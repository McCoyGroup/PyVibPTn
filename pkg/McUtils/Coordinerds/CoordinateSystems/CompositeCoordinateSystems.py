

class CompositeCoordinateSystem:
    """
    Defines a coordinate system that comes from applying a transformation
    to another coordinate system
    """

    def __init__(self, base_system, conversion):
        ...