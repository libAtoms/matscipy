#! /usr/bin/env python

# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) James Kermode, King's College London
#                  Lars Pastewka, Karlsruhe Institute of Technology
#                  Adrien Gola, Karlsruhe Institute of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# ======================================================================

import unittest

import numpy as np

import os
from matscipy.io import loadtbl, savetbl

import matscipytest

###

class TestEAMIO(matscipytest.MatSciPyTestCase):

    def test_savetbl_loadtbl(self):
        n = 123
        a = np.random.random(n)
        b = np.random.random(n)
        poe = np.random.random(n)
        savetbl('test.out', a=a, b=b, poe=poe)

        data = loadtbl('test.out')
        self.assertArrayAlmostEqual(a, data['a'])
        self.assertArrayAlmostEqual(b, data['b'])
        self.assertArrayAlmostEqual(poe, data['poe'])
            
###

if __name__ == '__main__':
    unittest.main()
