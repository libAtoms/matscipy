#! /usr/bin/env pytho

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
from numpy.linalg import norm

import ase.io as io
from ase.constraints import StrainFilter, UnitCellFilter
from ase.lattice.compounds import B1, B2, L1_0, L1_2
from ase.lattice.cubic import FaceCenteredCubic
from ase.lattice.hexagonal import HexagonalClosedPacked
from ase.optimize import FIRE
from ase.units import GPa

import matscipytest
from matscipy.calculators.pair_potential import PairPotential
from matscipy.elasticity import fit_elastic_constants, Voigt_6x6_to_cubic

###

class TestPairPotentialCalculator(matscipytest.MatSciPyTestCase):

    disp = 1e-6
    tol = 2e-6

    def test_forces(self):
        for calc in [EAM('Au-Grochola-JCP05.eam.alloy')]:
            a = io.read('Au_923.xyz')
            a.center(vacuum=10.0)
            a.set_calculator(calc)
            f = a.get_forces()
            fn = calc.calculate_numerical_forces(a)
            self.assertArrayAlmostEqual(f, fn, tol=self.tol)


###

if __name__ == '__main__':
    unittest.main()
