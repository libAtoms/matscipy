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
from ase import Atoms
from ase.constraints import StrainFilter, UnitCellFilter
from ase.lattice.compounds import B1, B2, L1_0, L1_2
from ase.lattice.cubic import FaceCenteredCubic
from ase.lattice.hexagonal import HexagonalClosedPacked
from ase.optimize import FIRE
from ase.units import GPa
from ase.phonons import Phonons

import matscipytest
from matscipy.calculators.polydisperse import IPL, Polydisperse
import matscipy.calculators.polydisperse as calculator
from matscipy.elasticity import fit_elastic_constants, Voigt_6x6_to_cubic

###


class TestPolydisperseCalculator(matscipytest.MatSciPyTestCase):

    tol = 1e-4

    def test_forces(self):
        d = 1.2
        L = 10 
        atomic_configuration = Atoms("HH", 
                                     positions=[(L/2, L/2, L/2), (L/2 + d, L/2, L/2)],
                                     cell=[L, L, L],
                                     pbc=[1, 1, 1]
                                     )
        atomic_configuration.set_array("size", np.array([1.3, 2.22]), dtype=float)
        atomic_configuration.set_masses(masses=np.repeat(1.0, len(atomic_configuration)))
        calc = Polydisperse(IPL(1.0, 1.4, 0.1, 3, 1, 2.22))
        atomic_configuration.set_calculator(calc)
        f = atomic_configuration.get_forces()
        fn = calc.calculate_numerical_forces(atomic_configuration, d=0.0001)
        self.assertArrayAlmostEqual(f, fn, tol=self.tol)

    def test_symmetry_dense(self):
        """
        Test the symmetry of the dense Hessian matrix 
        """

    def test_symmetry_sparse(self):
        """
        Test the symmetry of the sparse Hessian matrix 
        """

    def test_hessian(self):
        """
        Test the computation of the Hessian matrix 
        """

###


if __name__ == '__main__':
    unittest.main()
