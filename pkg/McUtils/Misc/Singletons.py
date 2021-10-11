"""
Provides a set of singleton objects that can declare their purpose a little bit better than None can
"""

import enum

__all__ = [ "Default" ]

class DefaultType:
    """
    A type for declaring an argument should use its default value (for when `None` has meaning)
    """
Default=DefaultType()