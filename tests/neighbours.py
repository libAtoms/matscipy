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

import matscipytest
from matscipy.neighbours import mic, neighbour_list

###

class TestNeighbours(matscipytest.MatSciPyTestCase):

    def test_neighbour_list(self):
        a = io.read('aC.traj')
        j, dr, i, abs_dr = neighbour_list("jDid", a, 1.85)

        self.assertTrue((np.bincount(i) == np.bincount(j)).all())

        r = a.get_positions()
        dr_direct = mic(r[i]-r[j], a.cell)

        abs_dr_from_dr = np.sqrt(np.sum(dr*dr, axis=1))
        abs_dr_direct = np.sqrt(np.sum(dr_direct*dr_direct, axis=1))

        self.assertTrue(np.all(np.abs(abs_dr-abs_dr_from_dr) < 1e-12))
        self.assertTrue(np.all(np.abs(abs_dr-abs_dr_direct) < 1e-12))

        self.assertTrue(np.all(np.abs(dr-dr_direct) < 1e-12))

    def test_small_cell(self):
        a = ase.Atoms('C', positions=[[0.5, 0.5, 0.5]], cell=[1, 1, 1],
                      pbc=True)
        i, j, dr, shift = neighbour_list("ijDS", a, 1.1)
        assert np.bincount(i)[0] == 6
        assert (dr == shift).all()

        i, j = neighbour_list("ij", a, 1.5)
        assert np.bincount(i)[0] == 18

        a.set_pbc(False)
        i = neighbour_list("i", a, 1.1)
        assert i == []

        a.set_pbc([True, False, False])
        i = neighbour_list("i", a, 1.1)
        assert np.bincount(i)[0] == 2

        a.set_pbc([True, False, True])
        i = neighbour_list("i", a, 1.1)
        assert np.bincount(i)[0] == 4

###

if __name__ == '__main__':
    unittest.main()

