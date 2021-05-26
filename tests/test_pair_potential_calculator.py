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
from matscipy.elasticity import fit_elastic_constants, elastic_moduli, full_3x3x3x3_to_Voigt_6x6, measure_triclinic_elastic_constants
import matscipy.calculators.pair_potential as calculator
from matscipy.hessian_finite_differences import fd_hessian

###

def measure_triclinic_elastic_constants_2nd(a, delta=0.001):
    r0 = a.positions.copy()

    cell = a.cell.copy()
    volume = a.get_volume()
    e0 = a.get_potential_energy()

    C = np.zeros((3, 3, 3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            a.set_cell(cell, scale_atoms=True)
            a.set_positions(r0)

            e = np.zeros((3, 3))
            e[i, j] += 0.5*delta
            e[j, i] += 0.5*delta
            F = np.eye(3) + e
            a.set_cell(np.matmul(F, cell.T).T, scale_atoms=True)
            ep = a.get_potential_energy()

            e = np.zeros((3, 3))
            e[i, j] -= 0.5*delta
            e[j, i] -= 0.5*delta
            F = np.eye(3) + e
            a.set_cell(np.matmul(F, cell.T).T, scale_atoms=True)
            em = a.get_potential_energy()

            C[:, :, i, j] = (ep + em - 2*e0) / (delta ** 2)

    a.set_cell(cell, scale_atoms=True)
    a.set_positions(r0)

    return C

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

    def test_stress(self):
        for calc in [PairPotential({(1, 1): LennardJonesQuadratic(1, 1, 3), (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.4), (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.64)})]:
            a = io.read('KA256.xyz')
            a.center(vacuum=5.0)
            a.set_calculator(calc)
            s = a.get_stress()
            sn = calc.calculate_numerical_stress(a, d=0.0001)
            self.assertArrayAlmostEqual(s, sn, tol=self.tol)

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

    def test_elastic_born_crystal_stress_free(self):
        for calc in [{(1, 1): calculator.LennardJonesQuadratic(1.0, 1.0, 2.5)}]:
            b = calculator.PairPotential(calc)
            # Stress is minimal at latticeconstant=1.5695. For larger/smaller stresses at elastic constants deviate
            atoms = FaceCenteredCubic('H', size=[6,6,6], latticeconstant=1.5695)
            atoms.set_calculator(b)
            Cnum, Cerr_num = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=11, delta=1e-4, optimizer=None, verbose=False)
            Cana = b.get_born_elastic_constants(atoms)
            Cana_voigt = full_3x3x3x3_to_Voigt_6x6(Cana)
            #print("Stress: \n", atoms.get_stress())
            #print("Numeric: \n", Cnum)
            #print("Analytic: \n", Cana_voigt)
            #print("Absolute Difference: \n", Cnum-Cana_voigt)
            self.assertArrayAlmostEqual(Cnum, Cana_voigt, tol=2)
    """

    def test_elastic_born_crystal_stress(self):
        class TestPotential():
            def __init__(self, cutoff):
                self.cutoff = cutoff

            def __call__(self, r):
                """
                Return function value (potential energy).
                """
                return r - self.cutoff
                #return np.ones_like(r)

            def get_cutoff(self):
                return self.cutoff

            def first_derivative(self, r):
                return np.ones_like(r)
                #return np.zeros_like(r)

            def second_derivative(self, r):
                return np.zeros_like(r)

            def derivative(self, n=1):
                if n == 1:
                    return self.first_derivative
                elif n == 2:
                    return self.second_derivative
                else:
                    raise ValueError(
                        "Don't know how to compute {}-th derivative.".format(n))

        for calc in [{(1, 1): calculator.LennardJonesQuadratic(1.0, 1.0, 2.5)}]:
        #for calc in [{(1, 1): TestPotential(2.5)}]:
            b = calculator.PairPotential(calc)
            atoms = FaceCenteredCubic('H', size=[6,6,6], latticeconstant=1.2)
            # Randomly deform the cell
            strain = np.random.random([3, 3]) * 0.02
            atoms.set_cell(np.matmul(np.identity(3) + strain, atoms.cell), scale_atoms=True)
            atoms.set_calculator(b)
            Cnum, Cerr_num = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=11, delta=1e-4, optimizer=None, verbose=False)
            Cnum2_voigt = full_3x3x3x3_to_Voigt_6x6(measure_triclinic_elastic_constants(atoms), tol=10)
            #Cnum3_voigt = full_3x3x3x3_to_Voigt_6x6(measure_triclinic_elastic_constants_2nd(atoms), tol=10)
            Cana = b.get_born_elastic_constants(atoms)
            Cana_voigt = full_3x3x3x3_to_Voigt_6x6(Cana, tol=10)
            #print(atoms.get_stress())
            #print(Cnum)
            #print(Cana_voigt)
            np.set_printoptions(precision=3)
            print("Stress: \n", atoms.get_stress())
            print("Numeric (fit_elastic_constants): \n", Cnum)
            print("Numeric (measure_triclinic_elastic_constants): \n", Cnum2_voigt)
            #print("Numeric (measure_triclinic_elastic_constants_2nd): \n", Cnum3_voigt)
            print("Analytic: \n", Cana_voigt)
            print("Absolute Difference (fit_elastic_constants): \n", Cnum-Cana_voigt)
            print("Absolute Difference (measure_triclinic_elastic_constants): \n", Cnum2_voigt-Cana_voigt)
            print("Difference between numeric results: \n", Cnum-Cnum2_voigt)
            self.assertArrayAlmostEqual(Cnum, Cana_voigt, tol=10)
    """
    def test_non_affine_forces_glass(self):
        for calc in [{(1, 1): LennardJonesQuadratic(1, 1, 3), (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.4), (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.64)}]:
            atoms = io.read('KA256_Min.xyz')
            atoms.center(vacuum=5.0)
            b = calculator.PairPotential(calc)
            atoms.set_calculator(b)
            print(atoms.get_forces()[0,:])
            dyn = FIRE(atoms)
            dyn.run(fmax=1e-7)
            print(atoms.get_forces()[0,:])
            naForces_num = b.numerical_non_affine_forces(atoms, d=1e-6)
            print(naForces_num[0,:,:,:])
            naForces_ana = b.non_affine_forces(atoms)    
            print(naForces_ana[0,:,:,:])    
            self.assertArrayAlmostEqual(naForces_num, naForces_ana, tol=0.01)   

    def test_elastic_constants(self):
        for calc in [{(1, 1): LennardJonesQuadratic(1, 1, 3), (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.4), (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.64)}]:
            atoms = io.read('KA256_Min.xyz')
            atoms.center(vacuum=0.01)
            b = calculator.PairPotential(calc)
            atoms.set_calculator(b)
            dyn = FIRE(atoms)
            dyn.run(fmax=1e-7)
            C_num, Cerr_num = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=7, delta=1e-2, optimizer=FIRE, fmax=1e-6, verbose=False)
            Cna_ana = b.elastic_constants_non_affine_contribution(atoms)
            Cb = b.elastic_constants_born(atoms)
            print("C_num: \n", C_num)
            print("C_ana: \n", full_3x3x3x3_to_Voigt_6x6(Cna_ana + Cb))
            #self.assertArrayAlmostEqual(naForces_num, naForces_ana, tol=0.01)   
    """
###


if __name__ == '__main__':
    unittest.main()
