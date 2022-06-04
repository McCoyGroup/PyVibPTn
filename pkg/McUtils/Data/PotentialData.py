from .CommonData import DataHandler, DataRecord

__all__ = [ "PotentialData" ]
__reload_hook__ = [".CommonData"]

class PotentialDataHandler(DataHandler):
    def __init__(self):
        super().__init__("PotentialData", record_type=PotentialDataRecord)
    def __getitem__(self, item):
        """
        :param item:
        :type item: str
        :return:
        :rtype: PotentialDataRecord
        """
        return super().__getitem__(item)
class PotentialDataRecord(DataRecord):
    """
    Represents a simple callable wavefunction...
    """
    def __init__(self, data_handler, key, records):
        super().__init__(data_handler, key, records)
        self.parameters = dict(self['Defaults'])
    def __call__(self, *args, **kwargs):
        kwargs = dict(self.parameters, **kwargs)
        return self['Function'](*args, **kwargs)
    def __repr__(self):
        return "{}({})".format(self.key, ", ".join("{}={}".format(k,v) for k,v in self.parameters.items()))
PotentialData=PotentialDataHandler()
PotentialData.__doc__ = """An instance of PotentialDataHandler that can be used for looking up data on potentials"""
PotentialData.__name__ = "PotentialData"