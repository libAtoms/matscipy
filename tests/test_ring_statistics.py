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

import ase.build
import ase.io
import ase.lattice.hexagonal

import matscipytest
from matscipy.neighbours import neighbour_list
from matscipy.rings import ring_statistics
from matscipy.ffi import distances_on_graph, find_sp_rings

###

class TestNeighbours(matscipytest.MatSciPyTestCase):

    def test_single_ring(self):
        a = ase.build.molecule('C6H6')
        a = a[a.numbers==6]
        a.center(vacuum=5)

        i, j, r = neighbour_list('ijD', a, 1.85)
        d = distances_on_graph(i, j)

        self.assertEqual(d.shape, (6,6))

        self.assertArrayAlmostEqual(d-d.T, np.zeros_like(d))

        dcheck = np.arange(len(a))
        dcheck = np.abs(dcheck.reshape(1,-1)-dcheck.reshape(-1,1))
        dcheck = np.where(dcheck > len(a)/2, len(a)-dcheck, dcheck)
        self.assertArrayAlmostEqual(d, dcheck)

        r = find_sp_rings(i, j, r, d)
        self.assertArrayAlmostEqual(r, [0,0,0,0,0,0,1])

    def test_two_rings(self):
        a = ase.build.molecule('C6H6')
        a = a[a.numbers==6]
        a.center(vacuum=10)
        b = a.copy()
        b.translate([5.0,5.0,5.0])
        a += b

        r = ring_statistics(a, 1.85)
        self.assertArrayAlmostEqual(r, [0,0,0,0,0,0,2])

    def test_many_rings(self):
        a = ase.lattice.hexagonal.Graphite('C', latticeconstant=(2.5, 10.0),
                                           size=[2,2,1])
        r = ring_statistics(a, 1.85)
        self.assertArrayAlmostEqual(r, [0,0,0,0,0,0,8])

    def test_pbc(self):
        r = np.arange(6)
        r = np.transpose([r,np.zeros_like(r),np.zeros_like(r)])
        a = ase.Atoms('6C', positions=r, cell=[6,6,6], pbc=True)
        r = ring_statistics(a, 1.5)
        self.assertEqual(len(r), 0)

    def test_aC(self):
        a = ase.io.read('aC.cfg')
        r = ring_statistics(a, 1.85, maxlength=16)
        self.assertArrayAlmostEqual(r, [0,0,0,0,4,813,2678,1917,693,412,209,89,
                                        21,3])

###

if __name__ == '__main__':
    unittest.main()

