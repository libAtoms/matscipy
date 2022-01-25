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

"""Calculator and classes defining bonded interactions."""

import typing as ts
from ase.calculators.calculator import Calculator
from ase.geometry import find_mic, get_angles, get_dihedrals
from ..molecules import Molecules


class HarmonicPotential:
    """U(r) = 1/2 * k * (r - r0)**2."""

    def __init__(self, k, r0):
        self.k = k
        self.r0 = r0

    def __call__(self, r):
        return 0.5 * self.k * (r - self.r0)**2

    def first_derivative(self, r):
        return -self.k * (r - self.r0)

    def second_derivative(self, r):
        return self.k

    def derivative(self, n):
        if n == 1:
            return self.first_derivative
        elif n == 2:
            return self.second_derivative
        raise ValueError(
            "Don't know how to compute {}-th derivative.".format(n))


class MolecularCalculator(Calculator):
    """Base class for calculators based on bonded interactions."""

    implemented_properties = ["energy", "free_energy"]

    def __init__(self, molecules: Molecules,
                 interactions: ts.Mapping[int, ts.Any]):
        """Initialize calculator."""
        super().__init__()
        self.molecules = molecules
        self.interactions = interactions

    def calculate(self, atoms, properties, system_changes):
        """Calculate all the bonded interactions."""
        super().calculate(atoms, properties, system_changes)

        data = self.get_data(atoms)  # can be distance, angles, etc.
        epot = 0.
        for interaction_type, potential in self.interactions.items():
            mask = getattr(self.molecules, self.interaction_label)["type"] \
                == interaction_type
            epot += potential(data[mask]).sum()

        self.results = {"energy": epot,
                        "free_energy": epot}


class BondsCalculator(MolecularCalculator):
    """Calculator class for bonded interactions."""

    interaction_label = "bonds"

    def get_data(self, atoms):
        """Compute distances for all bonds."""
        positions = [
            atoms.get_positions()[self.molecules.bonds["atoms"][:, i]]
            for i in range(2)
        ]

        # Return distances only
        return find_mic(positions[1] - positions[0], atoms.cell, atoms.pbc)[1]


class AnglesCalculator(MolecularCalculator):
    """Calculator class for angle interactions."""

    interaction_label = "angles"

    def get_data(self, atoms):
        """Compute angles for all angles."""
        positions = [
            atoms.get_positions()[self.molecules.angles["atoms"][:, i]]
            for i in range(3)
        ]

        # WARNING: returns angles in degrees
        return get_angles(positions[1] - positions[0],
                          positions[2] - positions[1],
                          atoms.cell, atoms.pbc)


class DihedralsCalculator(MolecularCalculator):
    """Calculator class for dihedral interactions."""

    interaction_label = "dihedrals"

    def get_data(self, atoms):
        """Compute angles for all dihedrals."""
        positions = [
            atoms.get_positions()[self.molecules.dihedrals["atoms"][:, i]]
            for i in range(4)
        ]

        return get_dihedrals(positions[1] - positions[0],
                             positions[2] - positions[1],
                             positions[3] - positions[2],
                             atoms.cell, atoms.pbc)
