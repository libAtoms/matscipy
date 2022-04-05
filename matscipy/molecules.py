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

import re
import numpy as np

from ase.geometry import find_mic, get_angles, get_dihedrals
from ase import Atoms


class Molecules:
    """Similar to ase.Atoms, but for molecular data."""

    _dtypes = {
        "bonds": np.dtype([('type', np.int32), ('atoms', np.int32, 2)]),
        "angles": np.dtype([('type', np.int32), ('atoms', np.int32, 3)]),
        "dihedrals": np.dtype([('type', np.int32), ('atoms', np.int32, 4)]),
    }

    def __init__(self,
                 bonds_connectivity=None,
                 bonds_types=None,
                 angles_connectivity=None,
                 angles_types=None,
                 dihedrals_connectivity=None,
                 dihedrals_types=None):
        """
        Initialize with connectivity data.

        Parameters
        ----------
        bonds_connectivity : ArrayLike
            Array defining bonds with atom ids.
            Expected shape is ``(nbonds, 2)``.
        bonds_types : ArrayLike
            Array defining the bond types. Expected shape is ``nbonds``.
        angles_connectivity : ArrayLike
            Array defining angles with atom ids.
            Expected shape is ``(nangles, 3)``.
        angles_types : ArrayLike
            Array defining the angle types. Expected shape is ``nangles``.
        dihedrals_connectivity : ArrayLike
            Array defining angles with atom ids.
            Expected shape is ``(ndihedrals, 3)``.
        dihedrals_types : ArrayLike
            Array defining the dihedral types.
            Expected shape is ``ndihedrals``.
        """
        default_type = 1

        # Defining data arrays
        for data, dtype in self._dtypes.items():
            self.__dict__[data] = np.array([], dtype=dtype)

        if bonds_connectivity is not None:
            self.bonds.resize(len(bonds_connectivity))
            self.bonds["atoms"][:] = bonds_connectivity
            self.bonds["type"][:] = bonds_types \
                if bonds_types is not None else default_type

        if angles_connectivity is not None:
            self.angles.resize(len(angles_connectivity))
            self.angles["atoms"][:] = angles_connectivity
            self.angles["type"][:] = angles_types \
                if angles_types is not None else default_type

        if dihedrals_connectivity is not None:
            self.dihedrals.resize(len(dihedrals_connectivity))
            self.dihedrals["atoms"][:] = dihedrals_connectivity
            self.dihedrals["type"][:] = dihedrals_types \
                if dihedrals_types is not None else default_type

    def get_distances(self, atoms) -> np.ndarray:
        """Compute distances for all bonds."""
        positions = [
            atoms.positions[self.bonds["atoms"][:, i]]
            for i in range(2)
        ]

        # Return distances only
        return find_mic(positions[1] - positions[0],
                        atoms.cell, atoms.pbc)[1]

    def get_angles(self, atoms) -> np.ndarray:
        """Compute angles (degrees) for all angles."""
        positions = [
            atoms.positions[self.angles["atoms"][:, i]]
            for i in range(3)
        ]

        # WARNING: returns angles in degrees
        return get_angles(positions[1] - positions[0],
                          positions[2] - positions[1],
                          atoms.cell, atoms.pbc)

    def get_dihedrals(self, atoms) -> np.ndarray:
        """Compute angles (degrees) for all dihedrals."""
        positions = [
            atoms.positions[self.dihedrals["atoms"][:, i]]
            for i in range(4)
        ]

        return get_dihedrals(positions[1] - positions[0],
                             positions[2] - positions[1],
                             positions[3] - positions[2],
                             atoms.cell, atoms.pbc)

    @staticmethod
    def from_atoms(atoms: Atoms):
        """Construct a Molecules object from ase.Atoms object."""
        kwargs = {}

        def parse_tuples(regex, permutation, label):
            all_tuples = np.zeros((0, len(permutation)), np.int32)
            types = np.array([], np.int32)

            tuples = atoms.arrays[label]
            bonded = np.where(tuples != '_')[0]

            for i, per_atom in zip(bonded, tuples[bonded]):
                per_atom = np.array(regex.findall(per_atom), np.int32)
                new_tuples = np.array([
                    np.full(per_atom.shape[0], i, np.int32),
                    *(per_atom[:, :-1].T)
                ])

                all_tuples = np.append(all_tuples,
                                       new_tuples[permutation, :].T,
                                       axis=0)
                types = np.append(types, per_atom[:, -1])

            kwargs[f'{label}_connectivity'] = all_tuples
            kwargs[f'{label}_types'] = types

        if 'bonds' in atoms.arrays:
            bre = re.compile(r'(\d+)\((\d+)\)')
            parse_tuples(bre, (0, 1), 'bonds')
        if 'angles' in atoms.arrays:
            are = re.compile(r'(\d+)-(\d+)\((\d+)\)')
            parse_tuples(are, (1, 0, 2), 'angles')
        if 'dihedrals' in atoms.arrays:
            dre = re.compile(r'(\d+)-(\d+)-(\d+)\((\d+)\)')
            parse_tuples(dre, (0, 1, 2, 3), 'dihedrals')

        return Molecules(**kwargs)
