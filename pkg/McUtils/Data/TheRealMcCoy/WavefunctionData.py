import scipy.special as funcs, numpy as np

data = {
    "HarmonicOscillator": {
        "Name": "HarmonicOscillator",
        "Parameters": {
            "n": "Quantum number",
            "re": "Equilibrium bond length",
            "freq": "Frequency of the oscillator",
            "hb": "Reduced Planck Constant in appropriate units"
        },
        "Energy": lambda n, freq=1, hb=1: hb*freq*(n+1/2),
        "Wavefunction": lambda n,re=0: lambda r,H=funcs.hermite(n): H(r-re)*np.exp(-((r-re)**2))
    },
    "MorseOscillator": {
        "Name": "MorseOscillator",
        "Parameters": {
            "re": "Equilibrium bond length",
            "De": "Dissociation energy",
            "alpha": "Width of the well",
            "mass": "Mass of the oscillator",
            "hb": "Reduced Planck Constant in appropriate units"
        },
        "Energy": lambda n, re=0, De=1, alpha=1, mass=1, hb=1: (-((np.sqrt(2*mass*De)/(alpha*hb)-n-1/2)*alpha*hb)**2/(2*mass)),
        "Wavefunction": lambda n,re=0: "No" # never got around to implementing it since it was so nasty
        }
}