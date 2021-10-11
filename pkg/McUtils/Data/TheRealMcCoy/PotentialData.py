import scipy.special as funcs, numpy as np

data = {
    "HarmonicPotential":{
        "Name":"HarmonicPotential",
        "Parameters":{
            "re":"Equilibrium bond length",
            "freq":"Frequency of the oscillator",
            "mass":"Mass of the oscillator"
        },
        "Defaults":{
            "re":1,
            "freq":1,
            "mass":1
        },
        "Function": lambda r, re=1,freq=1,mass=1:1/2*(mass*(freq**2))*(r-re)**2
    },
    "MorsePotential":{
        "Name":"MorsePotential",
        "Parameters":{
            "re":"Equilibrium bond length",
            "De":"Dissociation energy",
            "alpha":"Width of the well"
        },
        "Defaults":{
            "re":1,
            "De":1,
            "alpha":1
        },
        "Function": lambda r, re=1,De=1,alpha=1:De*(1-np.exp(-alpha*(r-re)))**2
    }
}