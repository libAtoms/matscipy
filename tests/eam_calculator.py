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

from __future__ import print_function

import random
import unittest

import numpy as np

import ase.io as io
from ase.calculators.test import numeric_force

import matscipytest
from matscipy.eam.calculator import EAM

###

class TestEAMCalculator(matscipytest.MatSciPyTestCase):

    disp = 1e-6
    tol = 1e-6

    def test_forces(self):
        for calc in [EAM('Au-Grochola-JCP05.eam.alloy')]:
            a = io.read('Au_923.xyz')
            a.center(vacuum=10.0)
            a.set_calculator(calc)
            f = a.get_forces()
            random.seed()
            for dummy in range(10):
                i = random.randrange(len(a))
                d = random.randrange(3)
                self.assertTrue((numeric_force(a, i, d, self.disp)-f[i, d]) <
                                self.tol)

###

if __name__ == '__main__':
    unittest.main()
