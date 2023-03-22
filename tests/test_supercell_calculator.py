#
# Copyright 2014-2015, 2017, 2020-2021 Lars Pastewka (U. Freiburg)
#           2014 James Kermode (Warwick U.)
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

# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014-2017) James Kermode, Warwick University
#                       Lars Pastewka, Karlsruhe Institute of Technology
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

import unittest

from ase.build import bulk

import matscipytest
from matscipy.calculators import EAM, SupercellCalculator

###

class TestSupercellCalculator(matscipytest.MatSciPyTestCase):

    def test_eam(self):
        for calc in [EAM('Au-Grochola-JCP05.eam.alloy')]:
            a = bulk('Au')
            a *= (2, 2, 2)
            a.rattle(0.1)
            a.calc = calc
            e = a.get_potential_energy()
            f = a.get_forces()
            s = a.get_stress()

            a.set_calculator(SupercellCalculator(calc, (3, 3, 3)))
            self.assertAlmostEqual(e, a.get_potential_energy())
            self.assertArrayAlmostEqual(f, a.get_forces())
            self.assertArrayAlmostEqual(s, a.get_stress())

###

if __name__ == '__main__':
    unittest.main()
