#
# Copyright 2014-2015, 2017, 2021 Lars Pastewka (U. Freiburg)
#           2018, 2020 Jan Griesser (U. Freiburg)
#           2014, 2020 James Kermode (Warwick U.)
#           2018 Jacek Golebiowski (Imperial College London)
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

"""Deprecated module."""

from numpy import deprecate
from .numerical import numerical_hessian


@deprecate(new_name="numerical.numerical_hessian")
def fd_hessian(atoms, dx=1e-5, indices=None):
    """

    Compute the hessian matrix from Jacobian of forces via central differences.

    Parameters
    ----------
    atoms: ase.Atoms
        Atomic configuration in a local or global minima.

    dx: float
        Displacement increment

    indices:
        Compute the hessian only for these atom IDs

    """
    return numerical_hessian(atoms, dx=dx, indices=indices)
