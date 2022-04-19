#
# Copyright 2021 Lars Pastewka (U. Freiburg)
#           2021 Jan Griesser (U. Freiburg)
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

import pytest

import numpy as np 
import numpy.testing as nt

from ase import Atoms 

from matscipy.numerical import numerical_forces, numerical_stress, numerical_hessian

from matscipy.calculators.manybody.newmb import Manybody

from matscipy.calculators.manybody.potentials import (
    HarmonicPair, HarmonicAngle, ZeroPair
)

from matscipy.molecules import Molecules
from matscipy.neighbours import MolecularNeighbourhood

from collections import defaultdict

@pytest.fixture(params=[1.0])
def distance(request):
	return request.param 

@pytest.fixture
def configuration(distance):
	atoms = Atoms(
		"H"*4,
		positions=[
		     (0, 0, 0),
		     (1, 0, 0), 
		     (0, 1, 0),
		     (0, 1, 1)
		],
		cell=[10, 10, 10])

	atoms.positions[:] *= distance

	return atoms

def molecule():
    # Get all combinations of eight atoms, delete pairs with i == j
    bonds = np.array(np.meshgrid(np.linspace(0, 3, 4), np.linspace(0, 3, 4))).T.reshape(-1, 2)
    bonds = bonds[bonds[:, 0] != bonds[:, 1]]
    
    # Get all combinations of eight atoms, delete pairs with i == j == k
    angles = np.array(np.meshgrid(np.linspace(0, 3, 4), np.linspace(0, 3, 4), np.linspace(0, 3, 4))).T.reshape(-1, 3)
    angles = angles[angles[:, 0] != angles[:, 1]]
    angles = angles[angles[:, 0] != angles[:, 2]]
    angles = angles[angles[:, 1] != angles[:, 2]]

    return MolecularNeighbourhood(
        Molecules(
            bonds_connectivity=bonds,
            angles_connectivity=angles
        )
    )  

# Potentiales to be tested
potentials = [
    ({1: HarmonicPair(1, 1)}, {1: HarmonicAngle(1, np.pi/4)}, molecule())
]

@pytest.fixture(params=potentials)
def potential(request):
	return request.param

def test_properties(distance, configuration, potential):
    calc = Manybody(*potential)
    configuration.calc = calc

    # Testing forces
    f_ana = configuration.get_forces()
    f_num = numerical_forces(configuration, d=1e-6)
    nt.assert_allclose(f_ana, f_num, rtol=1e-6)   

    # Testing stresses
    s_ana = configuration.get_stress()
    s_num = numerical_stress(configuration, d=1e-6)
    nt.assert_allclose(s_ana, s_num, rtol=1e-6)

    # Testing Born elastic constants
    C_ana = configuration.calc.get_property("born_constants")
