"""
Implements an importer for Gaussian output formats
"""

import numpy as np, re, math, io
from .GaussianLogComponents import GaussianLogComponents, GaussianLogDefaults, GaussianLogOrdering
from .GaussianFChkComponents import FormattedCheckpointComponents, FormattedCheckpointCommonNames
from ..Parsers import FileStreamReader, FileStreamCheckPoint, FileStreamReaderException

__all__ = ["GaussianFChkReader", "GaussianLogReader", "GaussianLogReaderException", "GaussianFChkReaderException"]
__reload_hook__ = [ '.GaussianFChkComponents', ".GaussianLogComponents" ]

########################################################################################################################
#
#                                           GaussianLogReader
#
class GaussianLogReaderException(FileStreamReaderException):
    """
    A class for holding exceptions that occur in the course of reading from a log file
    """
    pass

class GaussianLogReader(FileStreamReader):
    """
    Implements a stream based reader for a Gaussian .log file.
    This is inherits from the `FileStreamReader` base, and takes a two pronged approach to getting data.
    First, a block is found in a log file based on a pair of tags.
    Next, a function (usually based on a `StringParser`) is applied to this data to convert it into a usable data format.
    The goal is to move toward wrapping all returned data in a `QuantityArray` so as to include data type information, too.

    You can see the full list of available keys in the `GaussianLogComponents` module, but currently they are:
    * `"Header"`: the header for the Gaussian job
    * `"InputZMatrix"`: the string of the input Z-matrix
    * `"CartesianCoordinates"`: all the Cartesian coordinates in the file
    * `"ZMatCartesianCoordinates"`: all of the Cartesian coordinate in Z-matrix orientation
    * `"StandardCartesianCoordinates"`: all of the Cartesian coordinates in 'standard' orientation
    * `"InputCartesianCoordinates"`: all of the Cartesian coordinates in 'input' orientation
    * `"ZMatrices"`: all of the Z-matrices
    * `"OptimizationParameters"`: all of the optimization parameters
    * `"MullikenCharges"`: all of the Mulliken charges
    * `"MultipoleMoments"`: all of the multipole moments
    * `"DipoleMoments"`: all of the dipole moments
    * `"OptimizedDipoleMoments"`: all of the dipole moments from an optimized scan
    * `"ScanEnergies"`: the potential surface information from a scan
    * `"OptimizedScanEnergies"`: the PES from an optimized scan
    * `"XMatrix"`: the anharmonic X-matrix from Gaussian's style of perturbation theory
    * `"Footer"`: the footer from a calculation

    You can add your own types, too.
    If you need something we don't have, give `GaussianLogComponents` a look to see how to add it in.

    """

    registered_components = GaussianLogComponents
    default_keys = GaussianLogDefaults
    default_ordering = GaussianLogOrdering

    def parse(self, keys = None, num = None, reset = False):
        """The main function we'll actually use. Parses bits out of a .log file.

        :param keys: the keys we'd like to read from the log file
        :type keys: str or list(str)
        :param num: for keys with multiple entries, the number of entries to pull
        :type num: int or None
        :return: the data pulled from the log file, strung together as a `dict` and keyed by the _keys_
        :rtype: dict
        """
        if keys is None:
            keys = self.get_default_keys()
        # important for ensuring correctness of what we pull
        if isinstance(keys, str):
            keys = (keys,)
        keys = sorted(keys,
                      key = lambda k: (
                          -1 if (self.registered_components[k]["mode"] == "List") else (
                              self.default_ordering[k] if k in self.default_ordering else 0
                          )
                      )
                      )

        res = {}
        if reset:
            with FileStreamCheckPoint(self):
                for k in keys:
                    comp = self.registered_components[k]
                    res[k] = self.parse_key_block(**comp, num=num)
        else:
            for k in keys:
                comp = self.registered_components[k]
                try:
                    res[k] = self.parse_key_block(**comp, num=num)
                except:
                    raise GaussianLogReaderException("failed to parse block for key '{}'".format(k))
        return res

    job_default_keys = {
        "opt":{
            "p": ("StandardCartesianCoordinates", "OptimizedScanEnergies", "OptimizedDipoleMoments"),
            "_": ("StandardCartesianCoordinates", "OptimizedScanEnergies")
        },
        "popt": {
            "p": ("StandardCartesianCoordinates", "OptimizedScanEnergies", "OptimizedDipoleMoments"),
            "_": ("StandardCartesianCoordinates", "OptimizedScanEnergies")
        },
        "scan": ("StandardCartesianCoordinates", "ScanEnergies")
    }
    def get_default_keys(self):
        """
        Tries to get the default keys one might be expected to want depending on the type of job as determined from the Header
        Currently only supports 'opt', 'scan', and 'popt' as job types.

        :return: key listing
        :rtype: tuple(str)
        """
        header = self.parse("Header", reset=True)["Header"]

        header_low = {k.lower() for k in header.job}
        for k in self.job_default_keys:
            if k in header_low:
                sub = self.job_default_keys[k]
                if isinstance(sub, dict):
                    for k in sub:
                        if k in header_low:
                            defs = sub[k]
                            break
                    else:
                        defs = sub["_"]
                else:
                    defs = sub
                break
        else:
            raise GaussianLogReaderException("unclear what default keys should be used if not a scan and not a popt")

        return ("Header", ) + tuple(defs) + ("Footer",)

    @classmethod
    def read_props(cls, file, keys):
        with cls(file) as reader:
            parse = reader.parse(keys)
        if isinstance(keys, str):
            parse = parse[keys]
        return parse

########################################################################################################################
#
#                                           GaussianFChkReader
#
class GaussianFChkReaderException(FileStreamReaderException):
    pass

class GaussianFChkReader(FileStreamReader):
    """Implements a stream based reader for a Gaussian .fchk file. Pretty generall I think. Should be robust-ish.
    One place to change things up is convenient parsers for specific commonly pulled parts of the fchk

    """

    GaussianFChkReaderException = GaussianFChkReaderException
    registered_components = FormattedCheckpointComponents
    common_names = {to_:from_ for from_, to_ in FormattedCheckpointCommonNames.items()}
    to_common_name = FormattedCheckpointCommonNames
    def read_header(self):
        """Reads the header and skips the stream to where we want to be

        :return: the header
        :rtype: str
        """
        return self.get_tagged_block(None, "Number of atoms")

    fchk_re_pattern = r"^(.+?)\s+(I|R|C|H)\s+(N=)?\s+(.+)\s+" # matches name, type, num (if there), and val
    fchk_re = re.compile(fchk_re_pattern)
    def get_next_block_params(self):
        """Pulls the tag of the next block, the type, the number of bytes it'll be,
        and if it's a single-line block it'll also spit back the block itself

        :return:
        :rtype: dict
        """
        with FileStreamCheckPoint(self):
            tag_line = self.readline()
            if tag_line == b'' or tag_line == '':
                return None
            match = re.match(self.fchk_re, tag_line)
            if match is None:
                with FileStreamCheckPoint(self):
                    for i in range(4):
                        prev_lines = self.rfind("\n")
                        self.seek(prev_lines)
                    lines = "".join(self.readline() for i in range(4))
                    raise GaussianFChkReaderException("{}.{}: line '{}' couldn't be read as a tag line (in '{}')".format(
                        type(self).__name__,
                        "get_next_block_params",
                        tag_line,
                        lines
                    ))
            jump = self.tell()
        self.seek(jump)

        name, btype, numQ, val = gg = match.groups()
        # print(gg)
        if numQ:
            byte_count = 0
            shits = int(val)
            # hard coded these block formats since they're documented by Gaussian and thus unlikely to change
            if btype == "I":
                byte_per_shit = 12
                shits_per_line = 6
                btype = int
            elif btype == "R":
                byte_per_shit = 16
                shits_per_line = 5
                btype = float
            elif btype == "C":
                byte_per_shit = 12
                shits_per_line = 5
                btype = str
            elif btype == "L":
                byte_per_shit = 12
                shits_per_line = 5
                btype = bool
            byte_count = shits * byte_per_shit + math.ceil( shits / shits_per_line ) # each newline needs a byte
        else:
            byte_count = None
            if btype == "I":
                val = int(val)
            elif btype == "R":
                val = float(val)
            elif btype == "L":
                val = bool(int(val))

        return {
            "name": name,
            "dtype": btype,
            "byte_count": byte_count,
            "value": val
        }

    def get_block(self, name = None, dtype = None, byte_count = None, value = None):
        """Pulls the next block by first pulling the block tag

        :return:
        :rtype:
        """

        if byte_count is not None:

            block_str = self.read(byte_count)
            if dtype in {int, float}:
                block_str = io.StringIO(block_str.replace("\n", "")) # flatten it out
                value = np.loadtxt(block_str)
                if dtype == int:
                    value = value.astype(np.int64)
            else:
                value = block_str

        try:
            parser = self.registered_components[name]
            value = parser(value)
        except KeyError:
            pass

        return value

    def skip_block(self, name = None, dtype = None, byte_count = None, value = None):
        """Skips the next block

        :return:
        :rtype:
        """

        if byte_count is not None:
            self.seek(self.tell() + byte_count)

    def parse(self, keys = None, default='raise'):
        if keys is None:
            keys_to_go = None
        else:
            if isinstance(keys, str):
                keys = (keys,)
            keys_original = set(keys)
            keys_to_go = { (self.common_names[k] if k in self.common_names else k) for k in keys }

        parse_results = {}
        header = self.read_header()

        parse_results['Header'] = header
        if keys_to_go is None:
            while True: # I'll just break once I've exhausted everything
                next_block = self.get_next_block_params()
                if next_block is None:
                    break
                tag = next_block["name"]
                parse_results[tag] = self.get_block(**next_block)
        else:
            while len(keys_to_go)>0:
                # try to skip malformatted blocks...
                try:
                    next_block = self.get_next_block_params()
                except GaussianFChkReaderException:
                    fp = self.find("\n")
                    if fp == -1:
                        next_block = None
                    else:
                        next_block = ""
                        self.seek(fp + 1)
                    while next_block is not None and next_block == "":
                        try:
                            next_block = self.get_next_block_params()
                        except GaussianFChkReaderException:
                            fp = self.find("\n")
                            if fp == -1:
                                next_block = None
                            else:
                                self.seek(fp + 1)

                if next_block is None:
                    if isinstance(default, str) and default=='raise':
                        raise GaussianFChkReaderException("{}.{}: couldn't find keys {}".format(
                            type(self).__name__,
                            "parse",
                            keys_to_go
                            )
                        )
                    else:
                        for tag in keys_to_go:
                            if tag not in keys_original:
                                tag = self.to_common_name[tag]
                            parse_results[tag] = default
                        break
                tag = next_block["name"]
                if tag in keys_to_go:
                    keys_to_go.remove(tag)
                    if tag not in keys_original:
                        tag = self.to_common_name[tag]
                    parse_results[tag] = self.get_block(**next_block)
                else:
                    self.skip_block(**next_block)

        return parse_results

    @classmethod
    def read_props(cls, file, keys):
        with cls(file) as reader:
            parse = reader.parse(keys)
        if isinstance(keys, str):
            parse = parse[keys]
        return parse

