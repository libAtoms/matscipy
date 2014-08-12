# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) James Kermode, King's College London
#                  Lars Pastewka, Karlsruhe Institute of Technology
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
# ======================================================================

import re

import numpy as np

###

def savetbl(fn, **kwargs):
    """
    Save tabulated data and write column header strings.
    
    Example:
        savetbl('file.dat', time=time, strain=strain, energy=energy)
    
    Parameters
    ----------
    fn : str
        Name of file to write to.
    kwargs : dict
        Keyword argument pass data and column header names.
    """
    header = reduce(lambda s, (i, x): '{0} {1}:{2}'.format(s, i+1, x),
                    enumerate(kwargs.iterkeys()), '')
    data = np.transpose([x for x in kwargs.itervalues()])
    np.savetxt(fn, data, header=header)


def loadtbl(fn, usecols=None):
    """
    Load tabulated data from column header strings.

    Example data file:
        # time strain energy
        1.0 0.01 283
        2.0 0.02 398
        ...
       
        strain, energy = loadtbl('file.dat', usecols=['strain', 'energy'])
    
    Parameters
    ----------
    fn : str
        Name of file to load.
    usecols : list of strings
        List of column names.
        
    Returns
    -------
    data : tuple of arrays
        Return tuple of array with data for each colume in usecols if
        usecols specified. For usecols=None, return dictionary with header
        keys and arrays as data entries.
    """
    f = open(fn)
    line = f.readline()
    while line.startswith('#'):
        column_labels = [ s.strip() for s in re.split('[\s,]+', line)[1:] ]
        line = f.readline()
    f.close()
    
    sep_i = [ x.find(':') for x in column_labels ]
    column_labels = map(lambda s,i: s[i+1:] if i >= 0 else s, column_labels,
                        sep_i)
    if usecols is None:
        data = np.loadtxt(fn, unpack=True)
        return { s: d for s, d in zip(column_labels, data) }
    else:
        column_i = [ column_labels.index(s) for s in usecols ]
        return np.loadtxt(fn, usecols=column_i, unpack=True)
