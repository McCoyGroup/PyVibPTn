"""
Provides a QuantityArray class to manage data & units simultaneously
"""

import numpy as np, io

__all__ = [
    "QuantityArray"
]

class QuantityArrayException(Exception):
    ...

class QuantityArray:
    """
    A little helper for working with NumPy arrays with units.
It's mostly just a safety mechanism for imported data, but also helps keep you from messing up when you
  do addition and multiplication and stuff.
    """
    def __init__(self, array, units):
        """
        :param array: array data
        :type array: np.ndarray
        :param unit: list of units for the array
        :type unit: str | Iterable[str]
        """
        if isinstance(units, str):
            units = [units]
        if len(units) != self.array.ndim:
            raise QuantityArrayException("units {} can't correspond to an array of shape {}".format(
                units,
                array.shape
            ))
        self.array = array
        self.units = tuple(units)
    @property
    def shape(self):
        return self.array.shape
    @property
    def dtype(self):
        return self.array.dtype

    @classmethod
    def raise_unit_mismatch(cls, u1, u2):
        raise QuantityArrayException("units '{}' and '{}' are incompatible".format(
            u1, u2
        ))
    def __neg__(self):
        """Implements -a"""
        return type(self)(-self.array, self.units)
    def __pos__(self):
        """Implements +a"""
        return type(self)(+self.array, self.units)
    def __add__(self, other):
        """Implements a+b"""
        if isinstance(other, (np.array, int, float, np.integer, np.floating)):
            return type(self)(other + self.array, self.units)
        elif isinstance(other, QuantityArray):
            if other.units != self.units:
                self.raise_unit_mismatch(self.units, other.units)
            return self.__add__(other.array)
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(
                type(self),
                type(other)
            ))
    def __sub__(self, other):
        """Implements a-b"""
        if isinstance(other, (np.array, int, float, np.integer, np.floating)):
            return type(self)(other - self.array, self.units)
        elif isinstance(other, QuantityArray):
            if other.units != self.units:
                self.raise_unit_mismatch(self.units, other.units)
            return self.__mul__(other.array)
        else:
            raise TypeError("unsupported operand type(s) for -: '{}' and '{}'".format(
                type(self),
                type(other)
            ))
    def __mul__(self, other):
        """Implements a*b"""
        if isinstance(other, (np.array, int, float, np.integer, np.floating)):
            return type(self)(other * self.array, self.units)
        elif isinstance(other, QuantityArray):
            if other.units != self.units:
                self.raise_unit_mismatch(self.units, other.units)
            return self.__mul__(other.array)
        else:
            raise TypeError("unsupported operand type(s) for *: '{}' and '{}'".format(
                type(self),
                type(other)
            ))
    def __truediv__(self, other):
        """Implements a/b"""
        if isinstance(other, (np.array, int, float, np.integer, np.floating)):
            return type(self)(other / self.array, self.units)
        elif isinstance(other, QuantityArray):
            if other.units != self.units:
                self.raise_unit_mismatch(self.units, other.units)
            return self.__mul__(other.array)
        else:
            raise TypeError("unsupported operand type(s) for /: '{}' and '{}'".format(
                type(self),
                type(other)
            ))
    def __divmod__(self, other):
        """Implements a//b"""
        if isinstance(other, (np.array, int, float, np.integer, np.floating)):
            return type(self)(other // self.array, self.units)
        elif isinstance(other, QuantityArray):
            if other.units != self.units:
                self.raise_unit_mismatch(self.units, other.units)
            return self.__mul__(other.array)
        else:
            raise TypeError("unsupported operand type(s) for //: '{}' and '{}'".format(
                type(self),
                type(other)
            ))
    def convert(self, units):
        """
        Converts the array from units A to units B
        :param units:
        :type units:
        :return:
        :rtype:
        """
        from ..Data import UnitsData

        if isinstance(units, str):
            units = [units]
        units = tuple(units)
        if units == self.units:
            return self
        if len(units) != len(self.units):
            raise QuantityArrayException("{}.{}: can't convert from units {} to {}".format(
                type(self).__name__,
                'convert',
                units,
                self.units
            ))

        factor = 1
        for old, new in zip(self.units, units):
            if old != new:
                factor *= UnitsData.convert(old, new)

        return type(self)(self.array*factor, units)
    def save(self, file):
        """
        Saves the QuantityArray to a file

        :param file:
        :type file:
        :return:
        :rtype:
        """
        np.savez(file, array=self.array, units=np.array(self.units, dtype=str))
    @classmethod
    def load(cls, file):
        """
        Loads a QuantityArray from file

        :param file:
        :type file:
        :return:
        :rtype:
        """
        dat = np.load(file)
        return cls(dat['array'], dat['units'])
    def format_header(self):
        return "Units: {} | Shape: {}".format(
            self.units,
            self.shape
        )
    @classmethod
    def parse_header(cls, line):
        unit, shape = line.split("|")
        # this kind of thing should never be done in legit production
        # code since it's fundamentally insecure...but we also do
        # dynamic loading of modules which is _also_ fundamentally insecure
        # so like we're already garbage
        unit = eval(unit.strip("Units: "))
        shape = eval(shape.strip("Shape: "))
        return (unit, shape)
    def savetxt(self, file):
        """
        Saves the QuantityArray to a text file
        :param file:
        :type file: file
        :return:
        :rtype:
        """
        opened_file = False
        try:
            if isinstance(file, str):
                file = open(file, "w")
                opened_file = True
            file.writelines([self.format_header()])
            array = self.array
            if self.array.ndim > 2:
                new_shape = (np.prod(array.shape[:-2]), array.shape[-1])
                array = array.reshape(new_shape)
            np.savetxt(file, array)
        finally:
            if opened_file:
                file.close()
    @classmethod
    def loadtxt(cls, file):
        """
        Loads a QuantityArray from a text file

        :param file:
        :type file:
        :return:
        :rtype:
        """
        opened_file = False
        try:
            if isinstance(file, str):
                file = open(file, "r")
                opened_file = True
            line_0 = file.readline()
            units, shape = cls.parse_header(line_0)
            array = np.loadtxt(file)
            return cls(array.reshape(shape), units)
        finally:
            if opened_file:
                file.close()
    def __repr__(self):
        return "{}(units={}, shape={}, dtype={})".format(
            type(self).__name__,
            self.units,
            self.shape,
            self.dtype
        )


