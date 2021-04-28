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
from matscipy.calculators.pair_potential import PairPotential, LennardJonesCut, LennardJonesQuadratic
from matscipy.elasticity import fit_elastic_constants, elastic_moduli, full_3x3x3x3_to_Voigt_6x6
import matscipy.calculators.pair_potential as calculator
from matscipy.hessian_finite_differences import fd_hessian

###


class TestPairPotentialCalculator(matscipytest.MatSciPyTestCase):

    tol = 1e-4
    """
    def test_forces(self):
        for calc in [PairPotential({(1, 1): LennardJonesQuadratic(1, 1, 3), (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.4), (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.64)})]:
            a = io.read('KA256.xyz')
            a.center(vacuum=5.0)
            a.set_calculator(calc)
            f = a.get_forces()
            fn = calc.calculate_numerical_forces(a, d=0.0001)
            self.assertArrayAlmostEqual(f, fn, tol=self.tol)

    def test_symmetry_dense(self):
        for calc in [{(1, 1): LennardJonesQuadratic(1, 1, 3), (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.4), (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.64)}]:
            a = io.read('KA256_Min.xyz')
            a.center(vacuum=5.0)
            b = calculator.PairPotential(calc)
            H = b.calculate_hessian_matrix(a, "dense")
            self.assertArrayAlmostEqual(np.sum(np.abs(H-H.T)), 0, tol=0)

    def test_symmetry_sparse(self):
        for calc in [{(1, 1): LennardJonesQuadratic(1, 1, 3), (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.4), (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.64)}]:
            a = io.read('KA256_Min.xyz')
            a.center(vacuum=5.0)
            b = calculator.PairPotential(calc)
            H = b.calculate_hessian_matrix(a, "sparse")
            H = H.todense()
            self.assertArrayAlmostEqual(np.sum(np.abs(H-H.T)), 0, tol=0)

    def test_hessian(self):
        for calc in [{(1, 1): LennardJonesQuadratic(1, 1, 3), (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.4), (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.64)}]:
            atoms = io.read("KA256_Min.xyz")
            atoms.center(vacuum=5.0)
            a = calculator.PairPotential(calc)
            atoms.set_calculator(a)
            H_analytical = a.calculate_hessian_matrix(atoms, "dense")
            H_numerical = fd_hessian(atoms, dx=1e-5, indices=None)
            H_numerical = H_numerical.todense()
            self.assertArrayAlmostEqual(H_analytical, H_numerical, tol=self.tol)

    def test_hessian_split(self):
        for calc in [{(1, 1): LennardJonesQuadratic(1, 1, 3), (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.4), (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.64)}]:
            atoms = io.read("KA256_Min.xyz")
            atoms.center(vacuum=5.0)
            b = calculator.PairPotential(calc)
            H_full = b.calculate_hessian_matrix(atoms, "dense")
            H_0to128 = b.calculate_hessian_matrix(
                atoms, "dense", limits=[0, 128])
            H_128to256 = b.calculate_hessian_matrix(
                atoms, "dense", limits=[128, 256])
            self.assertArrayAlmostEqual(
                np.sum(np.sum(H_full-H_0to128-H_128to256, axis=1)), 0, tol=0)
    """
    def test_elastic_born_crystal(self):
        for calc in [{(1, 1): calculator.LennardJonesQuadratic(1.0, 1.0, 2.5)}]:
            b = calculator.PairPotential(calc)
            atoms = FaceCenteredCubic('H', size=[4,4,4], latticeconstant=1.56)
            atoms.set_calculator(b)
            Cnum, Cerr_num = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=11, delta=1e-4, optimizer=None, fmax=1e-5, verbose=False)
            Cana = b.elastic_constants_born(atoms)
            Cana_voigt = full_3x3x3x3_to_Voigt_6x6(Cana)
            print(atoms.get_stress())
            print(Cnum)
            print(Cana_voigt)
            self.assertArrayAlmostEqual(Cnum, Cana_voigt, tol=2)
"""
    def test_elastic_born_crystal(self):
        for calc in [{(1, 1): LennardJonesQuadratic(1, 1, 3), (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.4), (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.64)}]:
            atoms = io.read('KA256_Min.xyz')
            atoms.center(vacuum=5.0)
            b = calculator.PairPotential(calc)
            atoms.set_calculator(b)
            naForces_num = b.numerical_non_affine_forces(atoms, d=1e-6)
            print(naForces_num[0,:,:,:])
            naForces_ana = b.non_affine_forces(atoms)    
            print(naForces_ana[0,:,:,:])    
            self.assertArrayAlmostEqual(naForces_num, naForces_ana, tol=2)     
"""
###


if __name__ == '__main__':
    unittest.main()
