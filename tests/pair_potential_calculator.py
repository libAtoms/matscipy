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

###


class TestPairPotentialCalculator(matscipytest.MatSciPyTestCase):

    tol = 1e-4

    def test_forces(self):
        for calc in [PairPotential({(1, 1): LennardJonesQuadratic(1, 1, 3), (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.4), (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.64)})]:
            a = io.read('KA256.xyz')
            a.center(vacuum=5.0)
            a.set_calculator(calc)
            f = a.get_forces()
            fn = calc.calculate_numerical_forces(a, d=0.0001)
            self.assertArrayAlmostEqual(f, fn, tol=self.tol)

    def test_symmetry(self):
        for calc in [{(1, 1): LennardJonesQuadratic(1, 1, 3), (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.4), (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.64)}]:
            a = io.read('KA256_Min.xyz')
            a.center(vacuum=5.0)
            b = calculator.PairPotential(calc)
            H = b.calculate_hessian_matrix(a, "dense")
            self.assertArrayAlmostEqual(np.sum(np.abs(H-H.T)), 0, tol=0)

    def test_hessian(self):
        for calc in [{(1, 1): LennardJonesQuadratic(1, 1, 3), (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.4), (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.64)}]:
            atoms = io.read("KA256_Min.xyz")
            atoms.center(vacuum=5.0)
            b = calculator.PairPotential(calc)
            H_analytical = b.calculate_hessian_matrix(atoms, "dense")
            # Numerical
            ph = Phonons(atoms, b, supercell=(1, 1, 1), delta=0.001)
            ph.run()
            ph.read(acoustic=False)
            ph.clean()
            H_numerical = ph.get_force_constant()[0, :, :]
            self.assertArrayAlmostEqual(H_analytical, H_numerical, tol=0.03)
###


if __name__ == '__main__':
    unittest.main()
