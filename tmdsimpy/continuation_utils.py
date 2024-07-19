"""
Depricated module, remove on a future release. 

This is here for backwards compatibility right now.
"""


from warnings import warn

warn(f'The module {__name__} is depricated, '
      + 'use tmdsimpy.utils.continuation instead.',
      FutureWarning, stacklevel=2)

from .utils.continuation import *
