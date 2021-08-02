#
# Copyright 2014-2015, 2020 Lars Pastewka (U. Freiburg)
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

import unittest

import numpy as np

import ase
import ase.io as io
from ase.lattice.cubic import Diamond

import matscipytest
from matscipy.hydrogenate import hydrogenate
from matscipy.neighbours import neighbour_list

###

class TestNeighbours(matscipytest.MatSciPyTestCase):

    def test_hydrogenate(self):
        a = Diamond('Si', size=[2,2,1])
        b = hydrogenate(a, 2.85, 1.0, mask=[True,True,False], vacuum=5.0)
        # Check if every atom is fourfold coordinated
        syms = np.array(b.get_chemical_symbols())
        c = np.bincount(neighbour_list('i', b, 2.4))
        self.assertTrue((c[syms!='H']==4).all())

###

if __name__ == '__main__':
    unittest.main()

