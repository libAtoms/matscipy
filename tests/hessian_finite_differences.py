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
from ase.phonons import Phonons

import matscipytest
from matscipy.calculators.pair_potential import PairPotential, LennardJonesCut, LennardJonesQuadratic
import matscipy.calculators.pair_potential as calculator
from matscipy.elasticity import fit_elastic_constants, Voigt_6x6_to_cubic
from matscipy.hessian_finite_differences import fd_hessian

###


class TestPairPotentialCalculator(matscipytest.MatSciPyTestCase):

    tol = 1e-4

    def test_hessian(self):
        for calc in [{(1, 1): LennardJonesCut(1, 1, 3)}]:
            atoms = io.read("FCC_LJcut.xyz")
            atoms.center(vacuum=5.0)
            a = calculator.PairPotential(calc)
            atoms.set_calculator(a)
            # Analytical
            H_analytical = a.calculate_hessian_matrix(atoms, "dense")
            # Numerical
            H_numerical = fd_hessian(atoms, dx=1e-5, indices=None, H_format="dense")
            self.assertArrayAlmostEqual(H_analytical, H_numerical, tol=self.tol)



###


if __name__ == '__main__':
    unittest.main()
