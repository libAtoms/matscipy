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

try:
    import scipy.sparse as sp
except ImportError:
    warnings.warn('Warning: no scipy')

def fd_hessian(atoms, dx=1e-5, indices=None, H_format="dense"):
    """

    Compute the hessian matrix from Jacobian of forces via central differences.
    Finite difference hessian from Jacobian of forces

    Parameters
    ----------
    atoms: ase.Atoms
        Atomic configuration in a local or global minima.

    dx: float
        Displacement increment  
   
    indices: 
        Compute the hessian only for these atom IDs

    H_format: "dense" or "sparse"
        Output format of the hessian matrix.
        The format "sparse" is only possible if matscipy was build with scipy.

    """

    if H_format == "sparse":
        try:
            from scipy.sparse import coo_matrix
        except ImportError:
            raise ImportError(
                "Import error: Can not output the hessian matrix since scipy.sparse could not be loaded!")

    if indices is None:
        indices = range(len(atoms))

    if H_format == "sparse":
        row = []
        col = []
        H = []

        for i, AtomId1 in enumerate(indices):
            for direction in range(3):
                atoms.positions[AtomId1, direction] += dx
                fp_nc = atoms.get_forces()[indices].reshape(-1)
                atoms.positions[AtomId1, direction] -= 2 * dx
                fn_nc = atoms.get_forces()[indices].reshape(-1)
                atoms.positions[AtomId1, direction] += dx
                dH_nc = (fn_nc - fp_nc) / (2 * dx)
                if indices is None:
                    for j, AtomId2 in enumerate(indices):
                        for l in range(3):
                            H.append(dH_nc[3 * AtomId2 + l])
                            row.append(3 * AtomId1 + direction)  
                            col.append(3 * AtomId2 + l) 
                else:
                    for j, AtomId2 in enumerate(indices):
                        for l in range(3):     
                            H.append(dH_nc[3 * j + l])
                            row.append(3 * i + direction)  
                            col.append(3 * AtomId2 + l) 

        return coo_matrix((H, (row, col)),
                          shape=(3 * len(indices), 3 * len(atoms)))

    if H_format == "dense":
        if indices is None:
            H = np.zeros((3 * len(atoms), 3 * len(atoms)))
        else:
            H = np.zeros((3 * len(indices), 3 * len(atoms)))

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
                        H[3*AtomId1+direction,3*AtomId2:3*AtomId2+3] = dH_nc[3*AtomId2:3*AtomId2+3]
                else:
                    nat = len(atoms)
                    for j, AtomId2 in enumerate(range(nat)):
                        H[3*i+direction,3*AtomId2:3*AtomId2+3] = dH_nc[3*j:3*j+3]
 
        return H

