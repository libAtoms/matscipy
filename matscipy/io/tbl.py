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
    sorted_kwargs = sorted(kwargs.items(), key=lambda t: t[0])
    header = ''
    for i, (x, y) in enumerate(sorted_kwargs):
        header = '{0} {1}:{2}'.format(header, i+1, x)
    data = np.transpose([y for x, y in sorted_kwargs])
    np.savetxt(fn, data, header=header)


def loadtbl(fn, usecols=None, converters=None, fromfile=False, **kwargs):
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
    fromfile : bool
        Use numpy.fromfile instead of numpy.loadtxt if set to True. Can be
        faster in some circumstances.

    Returns
    -------
    data : tuple of arrays
        Return tuple of array with data for each colume in usecols if
        usecols specified. For usecols=None, return dictionary with header
        keys and arrays as data entries.
    """
    f = open(fn)
    line = f.readline()
    column_labels = None
    while line.startswith('#'):
        line = line[1:].strip()
        column_labels = [s.strip() for s in re.split('[\s,]+', line)]
        line = f.readline()
    if column_labels is None:
        f.close()
        raise RuntimeError("No header found in file '{}'".format(fn))

    sep_i = [x.find(':') for x in column_labels]
    column_labels = [s[i+1:] if i >= 0 else s for s, i
                     in zip(column_labels, sep_i)]

    if fromfile:
        data = np.fromfile(f, sep=' ')
        f.close()
        data.shape = (-1, len(column_labels))
        if usecols is None:
            return dict((s, d) for s, d in zip(column_labels, data.T))
        else:
            return [data[:, column_labels.index(s)] for s in usecols]
    else:
        f.close()
        converters_i = None
        if converters is not None:
            converters_i = {column_labels.index(s): c for s, c
                            in converters.items()}
        if usecols is None:
            data = np.loadtxt(fn, converters=converters_i, unpack=True,
                              **kwargs)
            return dict((s, d) for s, d in zip(column_labels, data))
        else:
            column_i = [column_labels.index(s) for s in usecols]
            return np.loadtxt(fn, usecols=column_i, converters=converters_i,
                              unpack=True, **kwargs)
