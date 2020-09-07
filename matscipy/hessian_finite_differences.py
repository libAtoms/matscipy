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

from __future__ import division

import numpy as np

from scipy.sparse import coo_matrix

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

    nat = len(atoms)
    if indices is None:
        indices = range(nat)

    row = []
    col = []
    H = []
    for i, AtomId1 in enumerate(indices):
        for direction in range(3):
            atoms.positions[AtomId1, direction] += dx
            fp_nc = atoms.get_forces().reshape(-1)
            atoms.positions[AtomId1, direction] -= 2 * dx
            fn_nc = atoms.get_forces().reshape(-1)
            atoms.positions[AtomId1, direction] += dx
            dH_nc = (fn_nc - fp_nc) / (2 * dx)

            if indices is None:
                for j, AtomId2 in enumerate(indices):
                    for l in range(3):
                        H.append(dH_nc[3 * AtomId2 + l])
                        row.append(3 * AtomId1 + direction)  
                        col.append(3 * AtomId2 + l) 

            else:
                for j, AtomId2 in enumerate(range(nat)):
                    for l in range(3):     
                        H.append(dH_nc[3 * j + l])
                        row.append(3 * i + direction)  
                        col.append(3 * AtomId2 + l) 

    return coo_matrix((H, (row, col)),
                          shape=(3 * len(indices), 3 * len(atoms)))



