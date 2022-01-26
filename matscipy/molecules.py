#
# Copyright 2022 Lucas Frérot (U. Freiburg)
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

"""Classes that deal with interactions defined by connectivity."""

import numpy as np

from ase import Atoms
from ase.geometry import find_mic, get_angles, get_dihedrals


class Molecules:
    """Similar to ase.Atoms, but for molecular data."""

    _dtypes = {
        "bonds": np.dtype([('type', np.int), ('atoms', np.int, 2)]),
        "angles": np.dtype([('type', np.int), ('atoms', np.int, 3)]),
        "dihedrals": np.dtype([('type', np.int), ('atoms', np.int, 4)]),
    }

    def __init__(self, atoms: Atoms,
                 bonds_connectivity: np.ndarray = None,
                 bonds_types: np.ndarray = None,
                 angles_connectivity: np.ndarray = None,
                 angles_types: np.ndarray = None,
                 dihedrals_connectivity: np.ndarray = None,
                 dihedrals_types: np.ndarray = None):
        """
        Initialize with connectivity data.

        Parameters
        ----------
        bonds_connectivity: np.ndarray
            Array defining bonds with atom ids.
            Expected shape is ``(nbonds, 2)``.
        bonds_types: np.ndarray
            Array defining the bond types. Expected shape is ``nbonds``.
        angles_connectivity: np.ndarray
            Array defining angles with atom ids.
            Expected shape is ``(nangles, 3)``.
        angles_types: np.ndarray
            Array defining the angle types. Expected shape is ``nangles``.
        dihedrals_connectivity: np.ndarray
            Array defining angles with atom ids.
            Expected shape is ``(ndihedrals, 3)``.
        dihedrals_types: np.ndarray
            Array defining the dihedral types.
            Expected shape is ``ndihedrals``.
        """
        self.atoms = atoms

        # Defining data arrays
        for data, dtype in self._dtypes.items():
            self.__dict__[data] = np.array([], dtype=dtype)

        if bonds_connectivity is not None:
            self.bonds.resize(len(bonds_connectivity))
            self.bonds["atoms"][:] = bonds_connectivity
            self.bonds["type"][:] = bonds_types \
                if bonds_types is not None else 1

        if angles_connectivity is not None:
            self.angles.resize(len(angles_connectivity))
            self.angles["atoms"][:] = angles_connectivity
            self.angles["type"][:] = angles_types \
                if angles_types is not None else 1

        if dihedrals_connectivity is not None:
            self.dihedrals.resize(len(dihedrals_connectivity))
            self.dihedrals["atoms"][:] = dihedrals_connectivity
            self.dihedrals["type"][:] = dihedrals_types \
                if dihedrals_types is not None else 1

    def get_distances(self):
        """Compute distances for all bonds."""
        positions = [
            self.atoms.get_positions()[self.bonds["atoms"][:, i]]
            for i in range(2)
        ]

        # Return distances only
        return find_mic(positions[1] - positions[0],
                        self.atoms.cell, self.atoms.pbc)[1]

    def get_angles(self):
        """Compute angles for all angles."""
        positions = [
            self.atoms.get_positions()[self.angles["atoms"][:, i]]
            for i in range(3)
        ]

        # WARNING: returns angles in degrees
        return get_angles(positions[1] - positions[0],
                          positions[2] - positions[1],
                          self.atoms.cell, self.atoms.pbc)

    def get_dihedrals(self):
        """Compute angles for all dihedrals."""
        positions = [
            self.atoms.get_positions()[self.dihedrals["atoms"][:, i]]
            for i in range(4)
        ]

        return get_dihedrals(positions[1] - positions[0],
                             positions[2] - positions[1],
                             positions[3] - positions[2],
                             self.atoms.cell, self.atoms.pbc)
