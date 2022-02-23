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

import unittest
import numpy as np

from ase import Atoms
from ase.calculators.mixing import SumCalculator
from matscipy.molecules import Molecules
from matscipy.calculators.molecular_calculator import (
    BondsCalculator, AnglesCalculator, DihedralsCalculator, HarmonicPotential
)

import matscipytest


def test_flat():
    atoms = Atoms("CO2", positions=[[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
    molecules = Molecules(bonds_connectivity=[[0, 1], [1, 2]],
                            angles_connectivity=[[0, 1, 2]])

    r0, theta0 = 0.5, np.pi / 3

    bonds_c = BondsCalculator(molecules,
                                {1: HarmonicPotential(1, r0)})
    angles_c = AnglesCalculator(molecules,
                                {1: HarmonicPotential(1, theta0)})

    atoms.calc = SumCalculator([bonds_c, angles_c])

    epot = atoms.get_potential_energy()
    epot_ref = 2 * (0.5 * r0**2) + 0.5 * theta0**2
    assert np.abs(epot - epot_ref) < 1e-15

def test_bent():
    r0, theta0 = 0.5, np.pi / 3
    atoms = Atoms("CO2", positions=[[-1, 0, 0],
                                    [0, 0, 0],
                                    [np.cos(theta0), np.sin(theta0), 0]])
    molecules = Molecules(bonds_connectivity=[[0, 1], [1, 2]],
                            angles_connectivity=[[0, 1, 2]])

    bonds_c = BondsCalculator(molecules,
                                {1: HarmonicPotential(1, r0)})
    angles_c = AnglesCalculator(molecules,
                                {1: HarmonicPotential(1, 0)})

    atoms.calc = SumCalculator([bonds_c, angles_c])

    epot = atoms.get_potential_energy()
    epot_ref = 2 * (0.5 * r0**2) + 0.5 * (360 * theta0 / (2 * np.pi))**2
    assert np.abs(epot - epot_ref) < 1e-12

def test_dihedral():
    atoms = Atoms("OCCO", positions=[
        [-1, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    molecules = Molecules(dihedrals_connectivity=[[0, 1, 2, 3]])

    atoms.calc = DihedralsCalculator(molecules,
                                        {1: HarmonicPotential(1, 0)})
    epot = atoms.get_potential_energy()
    epot_ref = 0.5 * 90**2
    assert np.abs(epot - epot_ref) < 1e-15
