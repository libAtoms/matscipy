#
# Copyright 2015, 2017 Lars Pastewka (U. Freiburg)
#           2015 Till Junge (EPFL)
#           2014-2015 James Kermode (Warwick U.)
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
#

# Generic stuff may go here.

from matscipy.logger import screen
from .distributed_computation import BaseWorker, BaseResultManager
###

def has_parameter(name):
    """
    Test if a parameter has been provided in params.py.

    Parameters
    ----------
    name : str
        Name of the parameter.

    Returns
    -------
    value : bool
        Returns True if parameter exists.
    """
    import sys
    for x in ['.', '..']:
        if x not in sys.path:
            sys.path += [x]
    import params
    return name in params.__dict__


def parameter(name, default=None, logger=screen):
    """
    Read parameter from params.py control file.

    Parameters
    ----------
    name : str
        Name of the parameter.
    default : optional
        Default value. Will be returned if parameter is not present.

    Returns
    -------
    value
        Value of the parameter.
    """
    import sys
    for x in ['.', '..']:
        if x not in sys.path:
            sys.path += [x]
    import params
    try:
        value = params.__dict__[name]
        logger.pr('(user value)      {0} = {1}'.format(name, value))
    except KeyError:
        if default is not None:
            value = default
            logger.pr('(default value)   {0} = {1}'.format(name, value))
        else:
            raise
    return value

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
