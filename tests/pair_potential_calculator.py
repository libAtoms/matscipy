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
import sys

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
from matscipy.calculators.pair_potential import PairPotential, LennardJonesCut, LennardJonesQuadratic, get_dynamical_matrix
from matscipy.elasticity import fit_elastic_constants, Voigt_6x6_to_cubic

###

class TestPairPotentialCalculator(matscipytest.MatSciPyTestCase):

    disp = 1e-8
    tol = 8e-3

    def test_forces(self):
        for calc in [PairPotential({(1,1): LennardJonesCut(1, 1, 3), (1,2): LennardJonesCut(1.5, 0.8, 2.4), (2, 2): LennardJonesCut(0.5, 0.88, 2.64 )})]:
            a = io.read('KA_108.xyz')
            a.center(vacuum=20.0)
            a.set_calculator(calc)
            f = a.get_forces()
            fn = calc.calculate_numerical_forces(a)
            self.assertArrayAlmostEqual(f, fn, tol=self.tol)

    def test_symmetry(self):
        for calc in [PairPotential({(1,1): LennardJonesCut(1, 1, 3), (1,2): LennardJonesCut(1.5, 0.8, 2.4), (2, 2): LennardJonesCut(0.5, 0.88, 2.64 )})]:
            a = io.read('KA_108.xyz')
            a.center(vacuum=20.0)
            a.set_calculator(calc)
            D = get_dynamical_matrix({(1,1): LennardJonesCut(1, 1, 3), (1,2): LennardJonesCut(1.5, 0.8, 2.4), (2, 2): LennardJonesCut(0.5, 0.88, 2.64 )}, a)
            self.assertArrayAlmostEqual(np.sum(np.abs(D.toarray()-D.toarray().T)), 0, tol = 0)
###

if __name__ == '__main__':
    unittest.main()