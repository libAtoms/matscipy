#
# Copyright 2015, 2020-2021 Lars Pastewka (U. Freiburg)
#           2017 Thomas Reichenbach (Fraunhofer IWM)
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
import ase.lattice.hexagonal

import matscipytest
from matscipy.neighbours import neighbour_list
from matscipy.angle_distribution import angle_distribution

###

class TestAngleDistribution(matscipytest.MatSciPyTestCase):

    def test_no_angle(self):
        a = ase.Atoms('CC', positions=[[0.5, 0.5, 0.5], [0.5, 0.5, 1.0]],
                      cell=[2, 2, 2], pbc=True)
        i, j, dr = neighbour_list("ijD", a, 1.1)
        hist = angle_distribution(i, j, dr, 20, 1.1)
        self.assertEqual(hist.sum(), 0)

    def test_single_angle(self):
        a = ase.Atoms('CCC', positions=[[0.5, 0.5, 0.5], [0.5, 0.5, 1.0],
                                       [0.5, 1.0, 1.0]],
                      cell=[2, 2, 2], pbc=True)
        i, j, dr = neighbour_list("ijD", a, 0.6)
        hist = angle_distribution(i, j, dr, 20, 0.6)
        #                                                 v 45 degrees
        self.assertArrayAlmostEqual(hist, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #                                  ^ 90 degrees


    def test_single_angle_reversed_order(self):
        a = ase.Atoms('CCC', positions=[[0.5, 0.5, 0.5], [0.5, 1.0, 1.0],
                                       [0.5, 0.5, 1.0]],
                      cell=[2, 2, 2], pbc=True)
        i, j, dr = neighbour_list("ijD", a, 0.6)
        hist = angle_distribution(i, j, dr, 20, 0.6)
        #                                                 v 45 degrees
        self.assertArrayAlmostEqual(hist, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #                                  ^ 90 degrees

    def test_three_angles(self):
        a = ase.Atoms('CCC', positions=[[0.5, 0.5, 0.5], [0.5, 0.5, 1.0],
                                       [0.5, 1.0, 1.0]],
                      cell=[2, 2, 2], pbc=True)
        i, j, dr = neighbour_list("ijD", a, 1.1)
        hist = angle_distribution(i, j, dr, 20)
        #                                                 v 45 degrees
        self.assertArrayAlmostEqual(hist, [0, 0, 0, 0, 0, 4, 0, 0, 0, 0,
                                           2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #                                  ^ 90 degrees
###

if __name__ == '__main__':
    unittest.main()

