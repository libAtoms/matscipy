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

from matscipy.elasticity import (full_3x3_to_Voigt_6_index,
                                 Voigt_6x6_to_full_3x3x3x3,
                                 full_3x3x3x3_to_Voigt_6x6,
                                 Voigt_6_to_full_3x3_strain,
                                 full_3x3_to_Voigt_6_strain,
                                 Voigt_6_to_full_3x3_stress,
                                 full_3x3_to_Voigt_6_stress)


###

def test_full_3x3_to_Voigt_6_index():
    assert full_3x3_to_Voigt_6_index(1, 1) == 1
    assert full_3x3_to_Voigt_6_index(1, 2) == 3
    assert full_3x3_to_Voigt_6_index(2, 1) == 3
    assert full_3x3_to_Voigt_6_index(0, 2) == 4
    assert full_3x3_to_Voigt_6_index(0, 1) == 5


def test_stiffness_conversion():
    C6 = np.random.random((6, 6))
    # Should be symmetric
    C6 = (C6 + C6.T) / 2
    C3x3 = Voigt_6x6_to_full_3x3x3x3(C6)
    C6_check = full_3x3x3x3_to_Voigt_6x6(C3x3)
    np.testing.assert_array_almost_equal(C6, C6_check)


def test_strain_conversion():
    voigt0 = np.random.random(6)
    full1 = Voigt_6_to_full_3x3_strain(voigt0)
    voigt1 = full_3x3_to_Voigt_6_strain(full1)
    np.testing.assert_array_almost_equal(voigt0, voigt1)


def test_stress_conversion():
    voigt0 = np.random.random(6)
    full2 = Voigt_6_to_full_3x3_stress(voigt0)
    voigt2 = full_3x3_to_Voigt_6_stress(full2)
    np.testing.assert_array_almost_equal(voigt0, voigt2)
