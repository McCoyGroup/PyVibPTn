from .CommonData import DataHandler, DataError

__all__ = [ "BondData", "BondDataHandler" ]
__reload_hook__ = [".CommonData"]

class BondDataHandler(DataHandler):
    """
    A DataHandler that's built for use with the bond data we've collected.
    Usually used through the `BondData` object.
    """
    def __init__(self):
        super().__init__("BondData", getter=lambda d,k:self.get_distance(k))
    def load(self):
        super().load()
        # now we create a back mapping
        for k in tuple(self._data.keys()):
            sub = self._data[k]
            for a in sub.keys():
                if a not in self._data:
                    self._data[a] = {}
                da = self._data[a]
                for n,v in sub[a].items():
                    if k not in da:
                        da[k] = {n:v}
                    else:
                        da[k][n] = v
    def get_distance(self, key, default=None):
        if isinstance(key, str):
            a1 = key
            a2 = None
            w = None
        elif len(key) == 1:
            a1 = key[0]
            a2 = None
            w = None
        else:
            try:
                a1, a2, w = key
            except (TypeError, ValueError):
                a1, a2 = key
                w = "first"

        d = self.data
        if a1 not in d:
            if default is None:
                raise DataError("Bond data for {} is unknown".format(a1))
            else:
                return default
        d = d[a1]
        if a2 is None:
            return d
        if a2 not in d:
            if default is None:
                raise DataError("Bond data for a {}-{} bond is unknown".format(a1, a2))
            else:
                return default
        d = d[a2]

        if w is None:
            return d
        if isinstance(w, int):
            if w == 1:
                w = "Single"
            elif w == 2:
                w = "Double"
            elif w == 3:
                w = "Triple"
        if isinstance(w, str):
            w = w.capitalize()

        if w not in d:
            if default is None:
                raise DataError("Bond data for a {}-{} {} bond is unknown".format(a1, a2, w))
            else:
                return default

        return d[w]


BondData=BondDataHandler()
BondData.__doc__ = """An instance of BondDataHandler that can be used for looking up bond distances"""
BondData.__name__ = "BondData"