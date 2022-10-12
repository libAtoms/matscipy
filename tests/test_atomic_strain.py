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

import ase.io as io
from ase.lattice.cubic import Diamond

import matscipytest
from matscipy.neighbours import mic, neighbour_list
from matscipy.atomic_strain import (get_delta_plus_epsilon_dgesv,
                                    get_delta_plus_epsilon,
                                    get_D_square_min)

###

class TestAtomicStrain(matscipytest.MatSciPyTestCase):

    def test_dsygv_dgelsd(self):
        a = Diamond('C', size=[4,4,4])
        b = a.copy()
        b.positions += (np.random.random(b.positions.shape)-0.5)*0.1
        i, j = neighbour_list("ij", b, 1.85)

        dr_now = mic(b.positions[i] - b.positions[j], b.cell)
        dr_old = mic(a.positions[i] - a.positions[j], a.cell)

        dgrad1 = get_delta_plus_epsilon_dgesv(len(b), i, dr_now, dr_old)
        dgrad2 = get_delta_plus_epsilon(len(b), i, dr_now, dr_old)

        self.assertArrayAlmostEqual(dgrad1, dgrad2)

###

if __name__ == '__main__':
    unittest.main()

