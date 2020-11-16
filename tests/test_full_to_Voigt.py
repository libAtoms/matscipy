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

import matscipytest
from matscipy.elasticity import (full_3x3_to_Voigt_6_index, 
                                 Voigt_6x6_to_full_3x3x3x3,
                                 full_3x3x3x3_to_Voigt_6x6)

###

class TestFullToVoigt(matscipytest.MatSciPyTestCase):

    def test_full_3x3_to_Voigt_6_index(self):
        self.assertTrue(full_3x3_to_Voigt_6_index(1, 1) == 1)
        self.assertTrue(full_3x3_to_Voigt_6_index(1, 2) == 3)
        self.assertTrue(full_3x3_to_Voigt_6_index(2, 1) == 3)
        self.assertTrue(full_3x3_to_Voigt_6_index(0, 2) == 4)
        self.assertTrue(full_3x3_to_Voigt_6_index(0, 1) == 5)

    def test_stiffness_conversion(self):
        C6 = np.random.random((6,6))
        # Should be symmetric
        C6 = (C6+C6.T)/2
        C3x3 = Voigt_6x6_to_full_3x3x3x3(C6)
        C6_check = full_3x3x3x3_to_Voigt_6x6(C3x3)
        self.assertArrayAlmostEqual(C6, C6_check)

###

if __name__ == '__main__':
    unittest.main()

