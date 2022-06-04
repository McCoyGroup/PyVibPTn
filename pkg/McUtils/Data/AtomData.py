"""
Provides a class for handling a compiled set of atomic data
"""
from .CommonData import DataHandler

__all__ = [ "AtomData", "AtomDataHandler" ]
__reload_hook__ = [".CommonData"]

class AtomDataHandler(DataHandler):
    """
    A DataHandler that's built for use with the atomic data we've collected.
    Usually used through the `AtomData` object.
    """
    def __init__(self):
        super().__init__("AtomData", alternate_keys=("Name", "Symbol", "CanonicalSymbol"))
    def __getitem__(self, item):
        """
        Special cases the default getitem so tuples are mapped
        :param item:
        :type item:
        :return:
        :rtype:
        """
        if isinstance(item, tuple):
            val = super().__getitem__(item[0])
            for k in item[1:]:
                val = val[k]
        else:
            val = super().__getitem__(item)
        return val
    def load(self):
        # now update by max IsotopeFraction
        super().load()
        maxIsos = {}
        for v in self._data.values():
            num = v["Number"]
            if num not in maxIsos or v["IsotopeFraction"] > maxIsos[num][0]:
                maxIsos[num] = (v["IsotopeFraction"], v)
        self._data.update(((k, v[1]) for k, v in maxIsos.items()))
AtomData = AtomDataHandler()
AtomData.__doc__ = """An instance of AtomDataHandler that can be used for looking up atom data"""
AtomData.__name__ = "AtomData"