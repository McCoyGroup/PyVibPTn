"""
A growing package of assorted functionality that finds use across many different packages, but doesn't attempt to
provide a single unified interface for doing certain types of projects.

All of the McUtils packages stand mostly on their own, but there will be little calls into one another here and there.

The more scientifically-focused `Psience` package makes significant use of `McUtils` as do various packages that have
been written over the years.
"""

import McUtils.Data
import McUtils.Parsers
import McUtils.GaussianInterface
import McUtils.Misc
import McUtils.Jupyter
import McUtils.Plots
import McUtils.Zachary
import McUtils.Coordinerds
import McUtils.ExternalPrograms
import McUtils.Extensions

__all__ = [
    "Parsers",
    "Extensions",
    "Plots",
    "ExternalPrograms",
    "Zachary",
    "Data",
    "Coordinerds",
    "GaussianInterface",
    "Numputils",
    "Scaffolding",
    "Parallelizers",
    "Jupyter",
    "Misc"
]