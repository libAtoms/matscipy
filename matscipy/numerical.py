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

import ase

from scipy.sparse import coo_matrix

from ase.calculators.calculator import Calculator


def numerical_forces(atoms: ase.Atoms, d: float = 1e-5):
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


def numerical_stress(atoms: ase.Atoms, d: float = 1e-5, voigt: bool = True):
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


def numerical_hessian(atoms: ase.Atoms, d: float = 1e-5, indices=None) -> coo_matrix:
    """
    Compute the hessian matrix from Jacobian of forces using central differences.

    Parameters
    ----------
    atoms: ase.Atoms
        Atomic configuration in a local or global minima.
    d: float
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
            atoms.positions[AtomId1, direction] += d
            fp_nc = atoms.get_forces().ravel()
            atoms.positions[AtomId1, direction] -= 2 * d
            fn_nc = atoms.get_forces().ravel()
            atoms.positions[AtomId1, direction] += d
            dH_nc = (fn_nc - fp_nc) / (2 * d)

            for j, AtomId2 in enumerate(range(nat)):
                for k in range(3):
                    H.append(dH_nc[3 * j + k])
                    row.append(3 * i + direction)
                    col.append(3 * AtomId2 + k)

    return coo_matrix(
        (H, (row, col)), shape=(3 * len(indices), 3 * len(atoms))
    )


def numerical_nonaffine_forces(atoms: ase.Atoms, d: float = 1e-5):
    """
    Calculate numerical non-affine forces using central differences.

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
        x = np.eye(3)
        j = i - 2
        x[i, j] = x[j, i] = d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        fplus = atoms.get_forces()

        x[i, j] = x[j, i] = -d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        fminus = atoms.get_forces()

        fna_ncc[..., i, j] = fna_ncc[..., j, i] = (fplus - fminus) / (4 * d)

    return fna_ncc


def numerical_nonaffine_forces_reference(atoms: ase.Atoms, d: float = 1e-5):
    """
    Compute nonaffine forces in the reference configuration using finite differences.
    """
    fna_ncc = np.zeros([len(atoms)] + 3 * [3])
    pos = atoms.positions

    for i in range(len(atoms)):
        for dim in range(3):
            pos[i, dim] += d
            fna_ncc[i, dim] = atoms.get_stress(voigt=False)
            pos[i, dim] -= 2 * d
            fna_ncc[i, dim] -= atoms.get_stress(voigt=False)
            pos[i, dim] += d  # reset position

    fna_ncc *= -atoms.get_volume() / (2 * d)
    return fna_ncc

def get_derivative_volume(atoms: ase.Atoms, d: float = 1e-5):
    """
    Calculate the derivative of the volume with respect to strain using central differences.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration in a local or global minima.
    d : float
        Finite difference step size.

    """
    cell = atoms.cell.copy()
    dvol = np.zeros((3, 3))

    for i in range(3):
        # Diagonal
        x = np.eye(3)
        x[i, i] += d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        Vplus = atoms.get_volume()

        x[i, i] -= 2 * d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        Vminus = atoms.get_volume()

        derivative_volume = (Vplus - Vminus) / (2 * d)

        dvol[i, i] = derivative_volume

        # Off diagonal
        j = i - 2
        x[i, j] = d
        x[j, i] = d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        Vplus = atoms.get_volume()

        x[i, j] = -d
        x[j, i] = -d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        Vminus = atoms.get_volume()

        derivative_volume = (Vplus - Vminus) / (4 * d)
        dvol[i, j] = derivative_volume
        dvol[j, i] = derivative_volume

    return dvol

def get_derivative_wave_vector(atoms: ase.Atoms, d: float = 1e-5):
    """
    Calculate the derivative of a wave vector with respect to strain using central differences.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration in a local or global minima.
    d : float
        Finite difference step size.

    """
    cell = atoms.cell.copy()

    e = np.ones(3)
    initial_k = 2 * np.pi * np.dot(np.linalg.inv(cell), e)

    dk = np.zeros((3, 3, 3))

    for i in range(3):
        # Diagonal
        x = np.eye(3)
        x[i, i] += d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        k_pos = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), e)

        x[i, i] -= 2 * d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        k_minus = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), e)
        derivative_k = (k_pos - k_minus) / (2 * d)
        dk[:, i, i] = derivative_k

        # Off diagonal --> xy, xz, yz
        j = i - 2
        x[i, j] = d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        k_pos = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), e)

        x[i, j] = -d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        k_minus = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), e)

        derivative_k = (k_pos - k_minus) / (2 * d)
        dk[:, i, j] = derivative_k

        # Odd diagonal --> yx, zx, zy
        x[j, i] = d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        k_pos = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), e)

        x[j, i] = -d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        k_minus = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), e)

        derivative_k = (k_pos - k_minus) / (2 * d)
        dk[:, j, i] = derivative_k

    return dk
