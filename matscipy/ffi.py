#
# Copyright 2022 Lucas Frérot (U. Freiburg)
#
# matscipy - Materials science with Python at the atomic-scale
# https://github.com/libAtoms/matscipy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Foreign function interface module.

Depending on the build-system (particularly pip version), the compiled extension
_matscipy.<cpython>.so may be installed in site-packages/ or in matscipy/, with
the latter being the intended destination. This module abstracts away the
import of symbols from the extension.

Example usage:
--------------

>>> from .ffi import first_neighbours  # imports a function from extension
>>> from . import ffi   # import as module

"""

try:
    from ._matscipy import *  # noqa
except ModuleNotFoundError:
    from _matscipy import *  # noqa
    from warnings import warn as _warn
    _warn("importing top-level _matscipy")

