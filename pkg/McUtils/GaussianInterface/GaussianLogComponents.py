"""
This lists the types of readers and things available to the GaussianLogReader
"""

from ..Parsers import *
from collections import namedtuple, OrderedDict

########################################################################################################################
#
#                                           GaussianLogComponents
#
# region GaussianLogComponents
GaussianLogComponents = OrderedDict()  # we'll register on this bit by bit
# each registration should look like:

# GaussianLogComponents["Name"] = {
#     "description" : string, # used for docmenting what we have
#     "tag_start"   : start_tag, # starting delimeter for a block
#     "tag_end"     : end_tag, # ending delimiter for a block None means apply the parser upon tag_start
#     "parser"      : parser, # function that'll parse the returned list of blocks (for "List") or block (for "Single")
#     "mode"        : mode # "List" or "Single"
# }

########################################################################################################################
#
#                                           Header
#

tag_start = "******************************************"
tag_end   = FileStreamerTag(
    """ --------""",
    follow_ups = (""" -----""",)
)

HeaderPercentBlockParser = StringParser(
    NonCapturing(
        ("%", Capturing(Word, dtype=str), "=", Capturing((Word, Optional(Any), Word), dtype=str) ),
        dtype=str
    )
)
HeaderHashBlockLine = RegexPattern((
    Capturing(Optional((Word, "="), dtype=str)),
    Capturing(Repeating(Alternatives((WordCharacter, "\(", "\)", "\/", "\-")), dtype=str))
    ))
HeaderHashBlockLineParser = StringParser(HeaderHashBlockLine)
HeaderHashBlockParser = StringParser(
    RegexPattern((
        "#",
         Optional(Whitespace),
         Repeating(
             Named(
                Repeating(HeaderHashBlockLine),
                "Command",
                suffix=Optional(Whitespace),
                dtype=str,
                default=""
             )
        )
    ))
)

def header_parser(header):
    # regex = HeaderHashBlockParser.regex #type: RegexPattern
    header_percent_data = HeaderPercentBlockParser.parse_all(header)
    runtime_options = {}
    for k, v in header_percent_data.array:
        runtime_options[k] = v

    header_hash_data = HeaderHashBlockParser.parse_all(header)
    all_keys=" ".join(header_hash_data["Command"].array.flatten())
    raw_data = HeaderHashBlockLineParser.parse_all(all_keys).array

    job_options = {}
    for k, v in raw_data:
        if k.endswith("="):
            job_options[k.strip("=").lower()] = v.strip("()").split(",")
        else:
            job_options[v] = []

    return namedtuple("HeaderData", ["config", 'job'])(runtime_options, job_options)

mode = "Single"

GaussianLogComponents["Header"] = {
    "tag_start": tag_start,
    "tag_end"  : tag_end,
    "parser"   : header_parser,
    "mode"     : mode
}

########################################################################################################################
#
#                                           InputZMatrix
#

# region InputZMatrix
tag_start = "Z-matrix:"
tag_end   = """ 
"""

def parser(zmat):
    return zmat

mode = "Single"

GaussianLogComponents["InputZMatrix"] = {
    "tag_start": tag_start,
    "tag_end"  : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

# endregion

########################################################################################################################
#
#                                           CartesianCoordinates
#

# region CartesianCoordinates

 # the region thing is just a PyCharm hack to collapse the boilerplate here... Could also have done 5000 files

cart_delim = """ --------------------------------------------------------------"""
cartesian_start_tag = FileStreamerTag(
    """Center     Atomic      Atomic             Coordinates (Angstroms)""",
    follow_ups= cart_delim
)
cartesian_end_tag = cart_delim

CartParser = StringParser(
    Repeating(
        (
            Named(
                Repeating(
                    Capturing(PositiveInteger),
                    min=3, max=3,
                    prefix=Optional(Whitespace),
                    suffix=Whitespace
                ),
                "GaussianStuff", handler=StringParser.array_handler(dtype=int)
            ),
            Named(
                Repeating(
                    Capturing(Number),
                    min = 3,
                    max = 3,
                    prefix=Optional(Whitespace),
                    joiner = Whitespace
                ),
                "Coordinates", handler=StringParser.array_handler(dtype=float)
            )
        ),
        suffix = Optional(Newline)
    )
)

# raise Exception(CartParser.regex)

def cartesian_coordinates_parser(strs):
    strss = "\n\n".join(strs)

    parse = CartParser.parse_all(strss)

    coords = (
        parse["GaussianStuff", 0],
        parse["Coordinates"].array
    )

    return coords

GaussianLogComponents["CartesianCoordinates"] = {
    "tag_start": cartesian_start_tag,
    "tag_end"  : cartesian_end_tag,
    "parser"   : cartesian_coordinates_parser,
    "mode"     : "List"
}
GaussianLogComponents["ZMatCartesianCoordinates"] = {
    "tag_start": FileStreamerTag('''Z-Matrix orientation:''', follow_ups = (cart_delim, cart_delim)),
    "tag_end"  : cartesian_end_tag,
    "parser"   : cartesian_coordinates_parser,
    "mode"     : "List"
}
GaussianLogComponents["StandardCartesianCoordinates"] = {
    "tag_start": FileStreamerTag('''Standard orientation:''', follow_ups = (cart_delim, cart_delim)),
    "tag_end"  : cartesian_end_tag,
    "parser"   : cartesian_coordinates_parser,
    "mode"     : "List"
}
GaussianLogComponents["InputCartesianCoordinates"] = {
    "tag_start": FileStreamerTag('''Input orientation:''', follow_ups = (cart_delim, cart_delim)),
    "tag_end"  : cartesian_end_tag,
    "parser"   : cartesian_coordinates_parser,
    "mode"     : "List"
}

# endregion

########################################################################################################################
#
#                                           ZMatrices
#

# region ZMatrices
tag_start = """Z-MATRIX (ANGSTROMS AND DEGREES)
   CD    Cent   Atom    N1       Length/X        N2       Alpha/Y        N3        Beta/Z          J
 ---------------------------------------------------------------------------------------------------"""
tag_end   = " ---------------------------------------------------------------------"

ZMatParser = StringParser(
    Repeating(
        (
            Named(
                Repeating(Capturing(PositiveInteger), min=1, max=2, prefix=Optional(Whitespace), suffix=Whitespace),
                "GaussianInts"
            ),
            Named(
                Capturing(AtomName),
                "AtomNames",
                suffix=Whitespace
            ),
            Named(
                Repeating(
                    (
                        Capturing(PositiveInteger),
                        Capturing(Number),
                        Parenthesized(PositiveInteger, prefix=Whitespace)
                    ),
                    min = None,
                    max = 3,
                    prefix=Optional(Whitespace),
                    joiner = Whitespace,
                ),
                "Coordinates"
            )
        ),
        suffix = Optional(RegexPattern((
            Optional((Whitespace, PositiveInteger)),
            Optional(Newline)
        )))
    )
)

def parser(strs):

    strss = '\n\n'.join(strs)
    fak = ZMatParser.parse_all(strss)
    coords = [
        (
            fak["GaussianInts", 0],
            fak["AtomNames", 0]
        ),
        fak["Coordinates", 0, 0],
        fak["Coordinates", 1]
        ]

    return coords
mode = "List"

GaussianLogComponents["ZMatrices"] = {
    "tag_start": tag_start,
    "tag_end"  : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

# endregion

########################################################################################################################
#
#                                           OptimizationParameters
#

# region OptimizationParameters

tag_start  = "Optimization "
tag_end    = """                        !
 ------------------------------------------------------------------------
"""


def parser(pars):
    """Parses a optimizatioon parameters block"""
    did_opts = [ "Non-Optimized" not in par for par in pars]
    return did_opts, pars


mode = "List"

GaussianLogComponents["OptimizationParameters"] = {
    "tag_start": tag_start,
    "tag_end"  : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

# endregion

########################################################################################################################
#
#                                           MullikenCharges
#

#region MullikenCharges
tag_start = "Mulliken charges:"
tag_end   = "Sum of Mulliken charges"


def parser(charges):
    """Parses a Mulliken charges block"""
    return charges
mode = "List"

GaussianLogComponents["MullikenCharges"] = {
    "tag_start": tag_start,
    "tag_end"  : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

# endregion

########################################################################################################################
#
#                                           MultipoleMoments
#

#region MultipoleMoments
tag_start  = "Dipole moment ("
tag_end    = " N-N="


def parser(moms):
    """Parses a multipole moments block"""
    return moms


mode = "List"

GaussianLogComponents["MultipoleMoments"] = {
    "tag_start": tag_start,
    "tag_end"  : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

# endregion

########################################################################################################################
#
#                                           DipoleMoments
#

# region DipoleMoments
tag_start  = "Dipole moment ("
tag_end    = "Quadrupole moment ("

dips_parser = StringParser(
    RegexPattern(
        (
            "X=", Capturing(Number),
            "Y=", Capturing(Number),
            "Z=", Capturing(Number)
        ),
        joiner=Whitespace,
        dtype=(float, (3,))
    )
)
def parser(moms):
    """Parses a multipole moments block"""

    res = dips_parser.parse_all("\n".join(moms))
    return res.array

mode = "List"
GaussianLogComponents["DipoleMoments"] = {
    "tag_start" : tag_start,
    "tag_end"   : tag_end,
    "parser"    : parser,
    "mode"      : mode
}

# endregion

########################################################################################################################
#
#                                           OptimizedDipoleMoments
#

# region DipoleMoments
tag_start  = " Dipole        ="
tag_end    = " Optimization"


def convert_D_number(a, **kw):
    import numpy as np
    return np.array([float(s.replace("D", "E")) for s in a])
DNumberPattern = RegexPattern((Number, "D", Integer), dtype=float)
OptimizedDipolesParser = StringParser(
    RegexPattern(
        (
            "Dipole", "=",
            Repeating(
                Capturing(DNumberPattern, handler=convert_D_number),
                min=3,
                max=3,
                suffix=Optional(Whitespace)
            )
        ),
        joiner=Whitespace
    )
)

def parser(mom):
    """Parses dipole block, but only saves the dipole of the optimized structure"""
    import numpy as np

    mom = "Dipole  =" + mom
    # print(">>>>>", mom)
    grps = OptimizedDipolesParser.parse_iter(mom)
    match = None
    for match in grps:
        pass

    if match is None:
        return np.array([])
    return match.value.array
    # else:
    #     grp = match.value
    #     dip_list = [x.replace("D", "E") for x in grp]
    #     dip_array = np.asarray(dip_list)
    #     return dip_array.astype("float64")

mode       = "List"
parse_mode = "Single"

GaussianLogComponents["OptimizedDipoleMoments"] = {
    "tag_start" : tag_start,
    "tag_end"   : tag_end,
    "parser"    : parser,
    "mode"      : mode,
    "parse_mode": parse_mode
}

# endregion

########################################################################################################################
#
#                                           ScanEnergies
#

# region ScanEnergies

tag_start = """ Summary of the potential surface scan:"""
tag_end = """Normal termination of"""

# Number = '(?:[\\+\\-])?\\d*\\.\\d+'
# block_pattern = "\s*"+Number+"\s*"+Number+"\s*"+Number+"\s*"+Number+"\s*"+Number
# block_re = re.compile(block_pattern)

ScanEnergiesParser = StringParser(
    RegexPattern(
        (
            Named(
                Repeating(Capturing(Word), prefix=Whitespace),
                "Keys",
                suffix=NonCapturing([Whitespace, Newline])
            ),
            Named(
                Repeating(
                    Repeating(
                        Alternatives([PositiveInteger, Number], dtype=float),
                        prefix=Whitespace
                        ),
                    suffix=Newline
                ),
                "Coords",
                prefix=Newline,
                handler=StringParser.array_handler(dtype=float)
            ),
        ),
        joiner=NonCapturing(Repeating([Whitespace, Repeating(["-"])]))
    )
)

def parser(block):
    """Parses the scan summary block"""
    import re

    if block is None:
        raise KeyError("key '{}' not in .log file".format('ScanEnergies'))

    r = ScanEnergiesParser.regex # type: RegexPattern
    parse=ScanEnergiesParser.parse(block)
    
    return namedtuple("ScanEnergies", ["coords", "energies"])(
        parse["Keys"].array,
        parse["Coords"].array
    )

mode = "Single"

GaussianLogComponents["ScanEnergies"] = {
    "tag_start": tag_start,
    "tag_end"  : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

# endregion

########################################################################################################################
#
#                                           OptimizedScanEnergies
#

# region OptimizedScanEnergies

tag_start = """ Summary of Optimized Potential Surface Scan"""
tag_end = FileStreamerTag(
    tag_alternatives = (
        """ Largest change from initial coordinates is atom """,
        """-"""*25
    )
)

eigsPattern = RegexPattern(
    (
        "Eigenvalues --",
        Repeating(Capturing(Number), suffix=Optional(Whitespace))
    ),
    joiner=Whitespace
)

coordsPattern = RegexPattern(
    (
        Capturing(VariableName),
        Repeating(Capturing(Number), suffix=Optional(Whitespace))
    ),
    prefix=Whitespace,
    joiner=Whitespace
)

OptScanPat = StringParser(
    RegexPattern(
        (
            Named(eigsPattern,
                  "Eigenvalues"
                  #parser=lambda t: np.array(Number.findall(t), 'float')
                  ),
            Named(Repeating(coordsPattern, suffix=Optional(Newline)), "Coordinates")
        ),
        joiner=Newline
    )
)

# Gaussian16 started subtracting any uniform shift off of the energies
# we'd like to get it back

eigsShift = RegexPattern(
    (
        "add",
        Whitespace,
        Named(Number, "Shift")
    ),
    joiner=Whitespace
)

EigsShiftPat = StringParser(eigsShift)

def parser(pars):
    """Parses the scan summary block and returns only the energies"""
    from collections import OrderedDict
    import numpy as np

    if pars is None:
        return None

    try:
        shift = EigsShiftPat.parse(pars)["Shift"]
        #TODO: might need to make this _not_ be a `StructuredTypeArray` at some point, but seems fine for now
    except StringParserException:
        shift = 0
    par_data = OptScanPat.parse_all(pars)
    energies_array = np.concatenate(par_data["Eigenvalues"]).flatten()\
    # when there isn't a value, for shape reasons we get extra nans
    energies_array = energies_array[np.logical_not(np.isnan(energies_array))] + shift
    coords = OrderedDict()
    cdz = [a.array for a in par_data["Coordinates"].array]
    for coord_names, coord_values in zip(*cdz):
        for k, v in zip(coord_names, coord_values):
            if k not in coords:
                coords[k] = [v]
            else:
                coords[k].append(v)
    for k in coords:
        coords[k] = np.concatenate(coords[k]).flatten()
        coords[k] = coords[k][np.logical_not(np.isnan(coords[k]))]

    return namedtuple("OptimizedScanEnergies", ["energies", "coords"])(
        energies_array,
        coords
    )

mode = "Single"

GaussianLogComponents["OptimizedScanEnergies"] = {
    "tag_start": tag_start,
    "tag_end"  : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

# endregion

########################################################################################################################
#
#                                           X-Matrix
#

# region X-Matrix

tag_start = FileStreamerTag(
    """Total Anharmonic X Matrix (in cm^-1)""",
    follow_ups=("""-"""*25,)
)
tag_end = FileStreamerTag(
    tag_alternatives = (
        """ ================================================== """,
        """-"""*25
    )
)

def parser(pars):
    """Parses the X matrix block and returns stuff --> huge pain in the ass function"""
    import numpy as np

    energies = np.array([x.replace("D", "E") for x in DNumberPattern.findall(pars)])
    l = len(energies)
    n = int( (-1 + np.sqrt(1 + 8*l))/2 )
    X = np.empty((n, n))
    # gaussian returns the data as blocks of 5 columns in the lower-triangle, annoyingly,
    # so we need to rearrange the indices so that they are sorted to make this work
    i, j = np.tril_indices_from(X)
    energies_taken = 0
    blocks = int(np.ceil(n/5))
    for b in range(blocks):
        sel = np.where((b*5-1 < j) * (j < (b+1)*5))[0]
        e_new=energies_taken+len(sel)
        e = energies[energies_taken:e_new]
        energies_taken=e_new
        ii = i[sel]
        jj = j[sel]
        X[ii, jj] = e
        X[jj, ii] = e

    return X

mode = "Single"

GaussianLogComponents["XMatrix"] = {
    "tag_start": tag_start,
    "tag_end"  : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

# endregion


tag_start =  "Job cpu time"
tag_end = "Normal termination"

def parser(block, start=tag_start):
    return " " + start + block

mode = "Single"

GaussianLogComponents["Footer"] = {
    "tag_start": tag_start,
    "tag_end"  : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

# endregion

########################################################################################################################
#
#                                           GaussianLogDefaults
#
# region GaussianLogDefaults
GaussianLogDefaults = (
    "StartDateTime",
    "InputZMatrix",
    "ScanTable",
    "Blurb",
    "ComputerTimeElapsed",
    "EndDateTime"
)
# endregion

########################################################################################################################
#
#                                           GaussianLogOrdering
#
# region GaussianLogOrdering
# defines the ordering in a GaussianLog file
glk = ( # this must be sorted by what appears when
    "Header",
    "StartDateTime",
    "CartesianCoordinates",
    "ZMatCartesianCoordinates",
    "StandardCartesianCoordinates",
    "CartesianCoordinateVectors",
    "MullikenCharges",
    "MultipoleMoments",
    "DipoleMoments",
    "OptimizedDipoleMoments",
    "QuadrupoleMoments",
    "OctapoleMoments",
    "HexadecapoleMoments",
    "IntermediateEnergies",
    "InputZMatrix",
    "InputZMatrixVariables",
    "ZMatrices",
    "ScanEnergies",
    "OptimizedScanEnergies",
    "OptimizationScan",
    "Blurb",
    "Footer"
)
list_type = { k:-1 for k in GaussianLogComponents if GaussianLogComponents[k]["mode"] == "List" }
GaussianLogOrdering = { k:i for i, k in enumerate([k for k in glk if k not in list_type]) }
GaussianLogOrdering.update(list_type)
del glk
del list_type
# endregion
