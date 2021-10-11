"""Defines components of an .fchk file that are already known and parseable"""

from ..Parsers.RegexPatterns import *
from .FChkDerivatives import *
import numpy as np

########################################################################################################################
#
#                                           FormattedCheckpointComponents
#
#region FormattedCheckpointComponents
FormattedCheckpointComponents = { } # we'll register on this bit by bit
# each registration should look like:

# FormattedCheckpointComponents["Name"] = parser

########################################################################################################################
#
#                                          Int Atom Types
#

#region IInt Atom Types

def get_names(atom_ints):
    from ..Data import AtomData
    return [ AtomData[x, "Symbol"] for x in atom_ints ]
FormattedCheckpointComponents["Int Atom Types"] = get_names

#endregion

########################################################################################################################
#
#                                          Current cartesian coordinates
#

#region Current cartesian coordinates

def reformat(coords):
    import numpy as np

    ncoords = len(coords)
    return np.reshape(coords, (int(ncoords/3), 3))
FormattedCheckpointComponents["Current cartesian coordinates"] = reformat

#endregion

########################################################################################################################
#
#                                           Cartesian Force Constants
#

#region Cartesian Force Constants

FormattedCheckpointComponents["Cartesian Force Constants"] = FchkForceConstants

#endregion

########################################################################################################################
#
#                                           Cartesian 3rd/4th derivatives
#

#region Cartesian 3rd/4th derivatives

FormattedCheckpointComponents["Cartesian 3rd/4th derivatives"] = FchkForceDerivatives

#endregion

########################################################################################################################
#
#                                           Dipole Derivatives
#

#region Dipole Derivatives

FormattedCheckpointComponents["Dipole Derivatives"] = FchkDipoleDerivatives

#endregion

########################################################################################################################
#
#                                           Dipole Derivatives num derivs
#

#region Dipole Derivatives num derivs

FormattedCheckpointComponents["Dipole Moment num derivs"] = FchkDipoleNumDerivatives

#region Dipole Derivatives num derivs

FormattedCheckpointComponents["Dipole Derivatives num derivs"] = FchkDipoleHigherDerivatives

#endregion

########################################################################################################################
#
#                                           Vib-Modes
#

#region Vib-Modes

def split_vib_modes(mcoeffs):
    """Pulls the mode vectors from the coeffs
    There should be 3N-6 modes where each vector is 3N long so N = (1 + sqrt(1 + l/9))

    :param mcoeffs:
    :type mcoeffs:
    :return:
    :rtype:
    """
    import numpy as np

    l = len(mcoeffs)
    n = int(1 + np.sqrt(1 + l/9))
    return np.reshape(mcoeffs, (3*n-6, 3*n))
FormattedCheckpointComponents["Vib-Modes"] = split_vib_modes

#endregion

########################################################################################################################
#
#                                           Vib-E2
#

#region Vib-E2

def split_vib_e2(e2):
    """Pulls the vibrational data out of the file

    :param e2:
    :type e2:
    :return:
    :rtype:
    """

    l = len(e2)
    n = 1 + np.sqrt(1 + l/9) # I thought this was the way it was defined but...seems like not exactly
    if n != int(n):
        n = l/14
        if n != int(n):
            raise ValueError("Gaussian FChk Vib-E2 block malformatted")
    n = int(n)

    freq = e2[:n]
    red_m = e2[n:2*n]
    frc_const = e2[2*n:3*n]
    intense = e2[3*n:4*n]

    return {
        "Frequencies"    : freq,
        "ReducedMasses"  : red_m,
        "ForceConstants" : frc_const,
        "Intensities"    : intense
    }
FormattedCheckpointComponents["Vib-E2"] = split_vib_e2

#endregion


########################################################################################################################
#
#                                           CommonNames
#

#region CommonNames

FormattedCheckpointCommonNames = {

    "Atomic numbers": "AtomicNumbers",
    "Current cartesian coordinates":"Coordinates",
    "Cartesian Gradient": "Gradient",
    "Cartesian Force Constants" : "ForceConstants",
    "Cartesian 3rd/4th derivatives" : "ForceDerivatives",
    "Dipole Moment" : "DipoleMoment",
    "Dipole Derivatives" : "DipoleDerivatives",
    "Dipole Moment num derivs" : "DipoleNumDerivatives",
    "Dipole Derivatives num derivs" : "DipoleHigherDerivatives",
    "Vib-E2" : "VibrationalData",
    "Vib-Modes" : "VibrationalModes",
    "Vib-AtMass" : "VibrationalAtomicMasses",
    "Real atomic weights" : "AtomicMasses"

}

#endregion