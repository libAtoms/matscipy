try:
    from . import _matscipy
except ImportError:
    import _matscipy
    from warnings import warn as _warn
    _warn("importing top-level _matscipy")

from _matscipy import *
