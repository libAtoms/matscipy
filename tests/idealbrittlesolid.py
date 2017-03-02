#! /usr/bin/env python

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
import ase.lattice.hexagonal
from ase.lattice import bulk
from ase.structure import molecule

import matscipytest
from matscipy.fracture_mechanics.idealbrittlesolid import find_triangles_2d

###

class TestNeighbours(matscipytest.MatSciPyTestCase):

    def test_two_triangles(self):
        a = ase.Atoms('4Xe', [[0,0,0], [1,0,0], [1,1,0], [0,1,0]])
        c1, c2, c3 = find_triangles_2d(a, 1.1)
        self.assertArrayAlmostEqual(np.transpose([c1, c2, c3]), [[0,1,2], [0,1,3], [0,2,3], [1,2,3]])

###

if __name__ == '__main__':
    unittest.main()

