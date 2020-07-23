# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014-2017) James Kermode, Warwick University
#                       Lars Pastewka, Karlsruhe Institute of Technology
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

from __future__ import absolute_import

from .eam import EAM
from .pair_potential import PairPotential
from .supercell_calculator import SupercellCalculator

try:
    import scipy.sparse as sp
except ImportError:
    warnings.warn('Warning: no scipy')

def fd_hessian(atoms, dx=1e-5, indices=None):
    """
    Finite difference hessian from Jacobian of forces
    """
    if indices is None:
        indices = range(len(atoms))
    I = []
    J = []
    Z = []
    for i, ii in enumerate(indices):
        for j in range(3):
            atoms.positions[ii, j] += dx
            fp = atoms.get_forces()[indices].reshape(-1)
            atoms.positions[ii, j] -= 2 * dx
            fm = atoms.get_forces()[indices].reshape(-1)
            atoms.positions[ii, j] -= dx
            dH = -(fp - fm) / (2 * dx)
            for k, kk in enumerate(indices):
                for l in range(3):
                    I.append(3 * ii + j)
                    J.append(3 * kk + l)
                    Z.append(dH[3 * k + j])

    return sp.coo_matrix((Z, (I, J)),
                         shape=(3 * len(atoms), 3 * len(atoms)))