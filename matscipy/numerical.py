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

"""Numerical algorithms for force, stress, hessian, etc."""

import numpy as np

from scipy.sparse import coo_matrix
from ase.calculators.calculator import Calculator


def numerical_forces(atoms, d=1e-3):
    """
    Compute numerical forces using finite differences.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration in a local or global minumum.
    d : float
        Displacement increment.
    """
    return Calculator().calculate_numerical_forces(atoms, d=d)


def numerical_stress(atoms, d=1e-6, voigt=True):
    """
    Compute numerical stresses using finite differences.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration in a local or global minumum.
    d : float
        Displacement increment.
    voigt : bool
        Return results in Voigt notation.
    """
    return Calculator().calculate_numerical_stress(atoms, d=d, voigt=voigt)


def numerical_hessian(atoms, dx=1e-5, indices=None):
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
            fp_nc = atoms.get_forces().ravel()
            atoms.positions[AtomId1, direction] -= 2 * dx
            fn_nc = atoms.get_forces().ravel()
            atoms.positions[AtomId1, direction] += dx
            dH_nc = (fn_nc - fp_nc) / (2 * dx)

            for j, AtomId2 in enumerate(range(nat)):
                for k in range(3):
                    H.append(dH_nc[3 * j + k])
                    row.append(3 * i + direction)
                    col.append(3 * AtomId2 + k)

    return coo_matrix(
        (H, (row, col)), shape=(3 * len(indices), 3 * len(atoms))
    )


def numerical_nonaffine_forces(atoms, d=1e-6):
    """
    Calculate numerical non-affine forces using central finite differences.

    This is done by deforming the box, rescaling atoms and measure the force.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration in a local or global minima.
    d : float
        Finite difference step size.

    """
    nat = len(atoms)
    cell = atoms.cell.copy()
    fna_ncc = np.zeros((nat, 3, 3, 3))

    for i in range(3):
        # Diagonal
        x = np.eye(3)
        x[i, i] += d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        fplus = atoms.get_forces()

        x[i, i] -= 2 * d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        fminus = atoms.get_forces()

        fna_ncc[..., i, i] = (fplus - fminus) / (2 * d)

        # Off diagonal
        j = i - 2
        x[i, j] = x[j, i] = d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        fplus = atoms.get_forces()

        x[i, j] = x[j, i] = -d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        fminus = atoms.get_forces()

        fna_ncc[..., i, j] = fna_ncc[..., j, i] = (fplus - fminus) / (4 * d)

    return fna_ncc
