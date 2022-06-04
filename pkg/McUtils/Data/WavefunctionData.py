from .CommonData import DataHandler, DataRecord

__all__ = [ "WavefunctionData" ]
__reload_hook__ = [".CommonData"]

class WavefunctionDataHandler(DataHandler):
    def __init__(self):
        super().__init__("WavefunctionData", record_type=WavefunctionDataRecord)

class WavefunctionDataRecord(DataRecord):
    """
    Represents a simple callable wavefunction...
    """
    def __call__(self, *args, **kwargs):
        return self['Wavefunction'](*args, **kwargs)
WavefunctionData=WavefunctionDataHandler()
WavefunctionData.__doc__ = """An instance of WavefunctionDataHandler that can be used for looking up data on wavefunctions"""
WavefunctionData.__name__ = "WavefunctionData"