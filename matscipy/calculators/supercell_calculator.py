#
# Copyright 2017 Lars Pastewka (U. Freiburg)
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

"""
Run calculation on a supercell of the atomic structure.
"""

import numpy as np

import ase
from ase.calculators.calculator import Calculator

###

class SupercellCalculator(Calculator):
    implemented_properties = ['energy', 'stress', 'forces']
    default_parameters = {}
    name = 'EAM'

    def __init__(self, calc, supercell):
        Calculator.__init__(self)
        self.calc = calc
        self.supercell = supercell

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        atoms = self.atoms.copy()
        atoms.set_constraint(None)
        atoms *= self.supercell
        atoms.set_calculator(self.calc)

        energy = atoms.get_potential_energy()
        stress = atoms.get_stress()
        forces = atoms.get_forces()

        self.results = {'energy': energy/np.prod(self.supercell),
                        'stress': stress,
                        'forces': forces[:len(self.atoms)]}
