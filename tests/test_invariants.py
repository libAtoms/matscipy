#
# Copyright 2014, 2020-2021 Lars Pastewka (U. Freiburg)
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

import matscipytest
from matscipy.elasticity import invariants

###

class TestInvariants(matscipytest.MatSciPyTestCase):

    def test_return_shape(self):
        sxx = np.array([[[10,11,12,13,14,15],[20,21,22,23,24,25]]], dtype=float)
        sxy = np.array([[[0,0,0,0,0,0],[0,0,0,0,0,0]]], dtype=float)
        P, tau, J3 = invariants(sxx, sxx, sxx, sxy, sxy, sxy)
        assert P.shape == sxx.shape
        assert tau.shape == sxx.shape
        assert J3.shape == sxx.shape

        assert np.abs(sxx+P).max() < 1e-12

###

if __name__ == '__main__':
    unittest.main()

