"""
Utilities for writing parsers of structured text.
An entirely standalone package which is used extensively by `GaussianInterface`.
Three main threads are handled:
1. A `FileStreamer` interface which allows for efficient searching for blocks of text
   in large files with no pattern matching
2. A `Regex` interface that provides declarative tools for building and manipulating a regular expression
   as a python tree
3. A `StringParser`/`StructuredTypeArray` interface that takes the `Regex` tools and allows for automatic
   construction of complicated `NumPy`-backed arrays from the parsed data. Generally works well but the
   problem is complicated and there are no doubt many unhandled edge cases.
   This is used extensively with (1.) to provide efficient parsing of data from Gaussian `.log` files by
   using a streamer to match chunks and a parser to extract data from the matched chunks.
"""

from .FileStreamer import *
from .StringParser import *
from .RegexPatterns import *
from .StructuredType import *
from .Parsers import *

from .FileStreamer import __all__ as FS__all__
from .StringParser import __all__ as StringParser__all__
from .RegexPatterns import __all__ as RegexPatterns__all__
from .StructuredType import __all__ as StructuredType__all__
from .Parsers import __all__ as Parsers__all__

__all__ = (
        FS__all__  +
        RegexPatterns__all__ +
        StringParser__all__ +
        StructuredType__all__ +
        Parsers__all__
)