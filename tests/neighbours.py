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

import ase.io as io

import matscipytest
from matscipy.neighbors import neighbor_list

###

class TestNeighbors(matscipytest.MatSciPyTestCase):

    def test_neighbor_list(self):
        a = io.read('aC.traj')
        i, j = neighbor_list(a, 1.85)

        print len(i), len(j)
        print i[0], j[0]
        print i[-1], j[-1]

###

if __name__ == '__main__':
    unittest.main()

