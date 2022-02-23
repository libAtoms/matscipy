#
# Copyright 2018-2021 Jan Griesser (U. Freiburg)
#           2014, 2020-2021 Lars Pastewka (U. Freiburg)
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

import random

import pytest

import sys

import numpy as np

from numpy.linalg import norm

from scipy.linalg import eigh

import ase
import ase.io as io
import ase.constraints
from ase.optimize import FIRE
from ase.units import GPa
from ase.lattice.cubic import FaceCenteredCubic

from matscipy.calculators.pair_potential import PairPotential, LennardJonesQuadratic, LennardJonesLinear
from matscipy.elasticity import fit_elastic_constants, elastic_moduli, full_3x3x3x3_to_Voigt_6x6, measure_triclinic_elastic_constants
from matscipy.calculators.calculator import MatscipyCalculator
from matscipy.numerical import numerical_hessian

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

def test_forces():
    """
    Test the computation of forces for a crystal and a glass
    """
    calc = {(1, 1): LennardJonesQuadratic(1, 1, 2.5)}
    atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=1.0) 
    atoms.rattle(0.01)
    b = PairPotential(calc)
    atoms.calc = b
    f = atoms.get_forces()
    fn = b.calculate_numerical_forces(atoms, d=0.0001)
    np.testing.assert_allclose(f, fn, atol=1e-2)

    calc = {(1, 1): LennardJonesQuadratic(1, 1, 2.5), 
            (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.0),
            (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.2)}
    atoms = io.read('glass_min.xyz')
    atoms.rattle(0.01)
    b = PairPotential(calc)
    atoms.calc = b
    f = atoms.get_forces()
    fn = b.calculate_numerical_forces(atoms, d=0.0001)
    np.testing.assert_allclose(f, fn, atol=1e-2)

@pytest.mark.parametrize('a0', [1.0, 1.5, 2.0, 2.5, 3.0])
def test_crystal_stress(a0):
    """
    Test the computation of stresses for a crystal 
    """
    calc = {(1, 1): LennardJonesQuadratic(1, 1, 2.5)}
    atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=a0) 
    b = PairPotential(calc)
    atoms.calc = b
    s = atoms.get_stress()
    sn = b.calculate_numerical_stress(atoms, d=0.0001)
    np.testing.assert_allclose(s, sn, atol=1e-4)

def test_amorphous_stress():
    """
    Test the computation of stresses for a glass
    """
    calc = {(1, 1): LennardJonesQuadratic(1, 1, 2.5), 
            (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.0),
            (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.2)}
    atoms = io.read('glass_min.xyz')
    b = PairPotential(calc)
    atoms.calc = b
    s = atoms.get_stress()
    sn = b.calculate_numerical_stress(atoms, d=0.0001)
    np.testing.assert_allclose(s, sn, atol=1e-4)

def test_hessian():
    """
    Test the computation of the Hessian matrix 
    """
    calc = {(1, 1): LennardJonesQuadratic(1, 1, 2.5), 
            (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.0),
            (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.2)}
    atoms = io.read("glass_min.xyz")
    b = PairPotential(calc)
    atoms.calc = b
    FIRE(atoms, logfile=None).run(fmax=1e-5)
    H_numerical = numerical_hessian(atoms, dx=1e-5, indices=None)
    H_numerical = H_numerical.todense()
    H_analytical = b.get_hessian(atoms, "dense")
    np.testing.assert_allclose(H_analytical, H_numerical, atol=1e-4)

    H_analytical = b.get_hessian(atoms, "sparse")
    H_analytical = H_analytical.todense()
    np.testing.assert_allclose(H_analytical, H_numerical, atol=1e-4)

def test_symmetry_dense():
    """
    Test the symmetry of the dense Hessian matrix 
    """
    calc = {(1, 1): LennardJonesQuadratic(1, 1, 2.5), 
            (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.0),
            (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.2)}
    a = io.read('glass_min.xyz')
    b = PairPotential(calc)
    a.calc = b
    FIRE(a, logfile=None).run(fmax=1e-5)
    H = b.get_hessian(a, "dense")
    np.testing.assert_allclose(np.sum(np.abs(H-H.T)), 0, atol=1e-10)

def test_symmetry_sparse():
    """
    Test the symmetry of the dense Hessian matrix 
    """
    calc = {(1, 1): LennardJonesQuadratic(1, 1, 2.5), 
            (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.0),
            (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.2)}
    a = io.read('glass_min.xyz')
    b = PairPotential(calc)
    a.calc = b
    FIRE(a, logfile=None).run(fmax=1e-5)
    H = b.get_hessian(a, "sparse")
    H = H.todense()
    np.testing.assert_allclose(np.sum(np.abs(H-H.T)), 0, atol=1e-10)

def test_hessian_divide_by_masses():
    """
    Test the computation of the Dynamical matrix 
    """
    calc = {(1, 1): LennardJonesQuadratic(1, 1, 2.5), 
            (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.0),
            (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.2)}
    atoms = io.read("glass_min.xyz")
    b = PairPotential(calc)
    atoms.calc = b
    FIRE(atoms, logfile=None).run(fmax=1e-5)
    masses_n = np.random.randint(1, 10, size=len(atoms))
    atoms.set_masses(masses=masses_n)  
    D_analytical = b.get_hessian(atoms, "sparse", divide_by_masses=True)
    D_analytical = D_analytical.todense()
    H_analytical = b.get_hessian(atoms, "sparse", divide_by_masses=False)
    H_analytical = H_analytical.todense()  
    masses_p = masses_n.repeat(3)
    H_analytical /= np.sqrt(masses_p.reshape(-1, 1) * masses_p.reshape(1, -1))
    np.testing.assert_allclose(H_analytical, D_analytical, atol=1e-4)

    D_analytical = b.get_hessian(atoms, "dense", divide_by_masses=True)
    np.testing.assert_allclose(H_analytical, D_analytical, atol=1e-4)

def test_non_affine_forces_glass():
    """
    Test the computation of the non-affine forces 
    """
    calc = {(1, 1): LennardJonesQuadratic(1, 1, 2.5), 
            (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.0),
            (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.2)}
    atoms = io.read("glass_min.xyz")
    b = PairPotential(calc)
    atoms.calc = b
    FIRE(atoms, logfile=None).run(fmax=1e-5)    
    naForces_num = b.get_numerical_non_affine_forces(atoms, d=1e-5)
    naForces_ana = b.get_nonaffine_forces(atoms)    
    np.testing.assert_allclose(naForces_num, naForces_ana, atol=0.1) 

@pytest.mark.parametrize('a0', [1.0, 1.5, 2.0, 2.5, 3.0])
def test_crystal_birch_elastic_constants(a0):
    """
    Test the Birch elastic constants for a crystalline system
    """
    calc = {(1, 1): LennardJonesLinear(1, 1, 2.5)}
    atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=a0) 
    b = PairPotential(calc)
    atoms.calc = b
    FIRE(ase.constraints.UnitCellFilter(atoms, mask=[0, 0, 0, 1, 1, 1]), logfile=None).run(fmax=1e-5)    
    C_num, Cerr = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=7, delta=1e-4, optimizer=None, verbose=False)
    C_ana = full_3x3x3x3_to_Voigt_6x6(b.get_birch_coefficients(atoms))
    np.testing.assert_allclose(C_num, C_ana, atol=0.1)

def test_amorphous_birch_elastic_constants():
    """
    Test the Birch elastic constants for an amorphous system
    """
    calc = {(1, 1): LennardJonesQuadratic(1, 1, 2.5), 
            (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.0),
            (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.2)}
    atoms = io.read("glass_min.xyz")
    b = PairPotential(calc)
    atoms.calc = b
    FIRE(ase.constraints.UnitCellFilter(atoms, mask=[1, 1, 1, 1, 1, 1]), logfile=None).run(fmax=1e-5)   
    C_num, Cerr = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=7, delta=1e-4, optimizer=None, verbose=False)
    C_ana = full_3x3x3x3_to_Voigt_6x6(b.get_birch_coefficients(atoms))
    np.testing.assert_allclose(C_num, C_ana, atol=0.1)

@pytest.mark.parametrize('a0', [1.0, 1.5, 2.0, 2.5, 3.0])
def test_non_affine_elastic_constants_crystal(a0):
    """
    Test the computation of Birch elastic constants and correction due to non-affine displacements
    """
    calc = {(1, 1): LennardJonesLinear(1, 1, 2.5)}
    atoms = FaceCenteredCubic('H', size=[3,3,3], latticeconstant=a0) 
    b = PairPotential(calc)
    atoms.calc = b  
    FIRE(ase.constraints.UnitCellFilter(atoms, mask=[0, 0, 0, 1, 1, 1]), logfile=None).run(fmax=1e-5)   
    C_num, Cerr = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=7, delta=1e-4, optimizer=FIRE, fmax=1e-5, verbose=False)
    anaC_na = full_3x3x3x3_to_Voigt_6x6(b.get_non_affine_contribution_to_elastic_constants(atoms))
    anaC_af = full_3x3x3x3_to_Voigt_6x6(b.get_birch_coefficients(atoms))
    np.testing.assert_allclose(C_num, anaC_af + anaC_na, atol=0.1)

def test_non_affine_elastic_constants_glass():
    """
    Test the computation of Birch elastic constants and correction due to non-affine displacements
    """
    calc = {(1, 1): LennardJonesQuadratic(1, 1, 2.5), 
            (1, 2): LennardJonesQuadratic(1.5, 0.8, 2.0),
            (2, 2): LennardJonesQuadratic(0.5, 0.88, 2.2)}          
    atoms = io.read("glass_min.xyz")
    b = PairPotential(calc)
    atoms.calc = b    
    FIRE(ase.constraints.UnitCellFilter(atoms, mask=[1, 1, 1, 1, 1, 1]), logfile=None).run(fmax=1e-5)   
    C_num, Cerr = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=5, delta=1e-4, optimizer=FIRE, fmax=1e-5, verbose=False)
    Cana_af = full_3x3x3x3_to_Voigt_6x6(b.get_birch_coefficients(atoms))
    Cana_na = full_3x3x3x3_to_Voigt_6x6(b.get_non_affine_contribution_to_elastic_constants(atoms))
    np.testing.assert_allclose(C_num, Cana_na + Cana_af, atol=0.1)
    
    H_nn = b.get_hessian(atoms, "sparse").todense()
    eigenvalues, eigenvectors = eigh(H_nn, subset_by_index=[3,3*len(atoms)-1])
    Cana2_na = full_3x3x3x3_to_Voigt_6x6(b.get_non_affine_contribution_to_elastic_constants(atoms, eigenvalues, eigenvectors))
    np.testing.assert_allclose(C_num, Cana2_na + Cana_af, atol=0.1)

def test_elastic_born_crystal_stress():
    class TestPotential():
        def __init__(self, cutoff):
            self.cutoff = cutoff

        def __call__(self, r):
            # Return function value (potential energy).

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

    for calc in [{(1, 1): LennardJonesQuadratic(1.0, 1.0, 2.5)}]:
    #for calc in [{(1, 1): TestPotential(2.5)}]:
        b = PairPotential(calc)
        atoms = FaceCenteredCubic('H', size=[6,6,6], latticeconstant=1.2)
        # Randomly deform the cell
        strain = np.random.random([3, 3]) * 0.02
        atoms.set_cell(np.matmul(np.identity(3) + strain, atoms.cell), scale_atoms=True)
        atoms.calc = b
        FIRE(ase.constraints.StrainFilter(atoms, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=1e-5)   
        Cnum, Cerr_num = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=11, delta=1e-4, optimizer=None, verbose=False)
        Cnum2_voigt = full_3x3x3x3_to_Voigt_6x6(measure_triclinic_elastic_constants(atoms), tol=10)
        #Cnum3_voigt = full_3x3x3x3_to_Voigt_6x6(measure_triclinic_elastic_constants_2nd(atoms), tol=10)
        Cana = b.get_birch_coefficients(atoms)
        Cana_voigt = full_3x3x3x3_to_Voigt_6x6(Cana, tol=10)
        #print(atoms.get_stress())
        #print(Cnum)
        #print(Cana_voigt)
        np.set_printoptions(precision=3)
        #print("Stress: \n", atoms.get_stress())
        #print("Numeric (fit_elastic_constants): \n", Cnum)
        #print("Numeric (measure_triclinic_elastic_constants): \n", Cnum2_voigt)
        #print("Numeric (measure_triclinic_elastic_constants_2nd): \n", Cnum3_voigt)
        #print("Analytic: \n", Cana_voigt)
        #print("Absolute Difference (fit_elastic_constants): \n", Cnum-Cana_voigt)
        #print("Absolute Difference (measure_triclinic_elastic_constants): \n", Cnum2_voigt-Cana_voigt)
        #print("Difference between numeric results: \n", Cnum-Cnum2_voigt)
        np.testing.assert_allclose(Cnum, Cana_voigt, atol=10)

