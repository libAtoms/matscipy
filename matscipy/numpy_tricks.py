#
# Copyright 2021 Lars Pastewka (U. Freiburg)
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

import itertools

import numpy as np


def mabincount(x, weights, minlength, axis=0):
    """
    Multi-axis bin count. Executes a bin count online a long a specific axis.
    (`numpy.bincount` only works on flattened arrays.)

    Parameters
    ----------
    x : array_like
        Array containing bin indices.
    weights : array_like
        Weights to be binned, dimension `axis` must have same size as x.
    minlength : int
        A minimum number of bins for the output array.
    axis : int, optional
        Axis along which the bin counting is performed. (Default: 0)

    Returns
    -------
    out : np.ndarray
        Array containing the counted data. Array has same dimensions as
        weights, with the exception of dimension `axis` that has is of at least
        `minlength` length.
    """

    # Construct shapes of result array and iterator
    result_shape = list(weights.shape)
    result_shape[axis] = minlength
    iter_shape = list(weights.shape)
    del iter_shape[axis]

    # Initialize result array to zero
    result = np.zeros(result_shape, dtype=weights.dtype)

    # Loop over all trailing dimensions and perform bin count
    for c in itertools.product(*(range(s) for s in iter_shape)):
        axis_slice = list(c)
        axis_slice.insert(axis, slice(None))
        axis_slice = tuple(axis_slice)
        result[axis_slice] = np.bincount(x, weights=weights[axis_slice], minlength=minlength)

    # Return results
    return result
