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

import ase.structure

import matscipytest
from matscipy.neighbours import neighbour_list
from _matscipy import distance_map, find_sp_rings

###

class TestNeighbours(matscipytest.MatSciPyTestCase):

    def test_single_ring(self):
        a = ase.structure.molecule('C6H6')
        a = a[a.numbers==6]
        a.center(vacuum=5)

        i, j, r = neighbour_list('ijD', a, 1.85)
        d = distance_map(i, j)

        self.assertEqual(d.shape, (6,6))

        self.assertArrayAlmostEqual(d-d.T, np.zeros_like(d))

        dcheck = np.arange(len(a))
        dcheck = np.abs(dcheck.reshape(1,-1)-dcheck.reshape(-1,1))
        dcheck = np.where(dcheck > len(a)/2, len(a)-dcheck, dcheck)
        self.assertArrayAlmostEqual(d, dcheck)

        r = find_sp_rings(i, j, r, d)
        self.assertArrayAlmostEqual(r, [0,0,0,0,0,0,1])

###

if __name__ == '__main__':
    unittest.main()

