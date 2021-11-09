#
# Copyright 2018-2021 Jan Griesser (U. Freiburg)
#           2014, 2020-2021 Lars Pastewka (U. Freiburg)
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
from ase.lattice.hexagonal import HexagonalFactory
from ase.lattice.cubic import SimpleCubicFactory

from matscipy.calculators.ewald import Ewald, BKS_ewald
from matscipy.elasticity import fit_elastic_constants, elastic_moduli, full_3x3x3x3_to_Voigt_6x6, measure_triclinic_elastic_constants
from matscipy.hessian_finite_differences import fd_hessian
from matscipy.numpy_tricks import mabincount

###

class alpha_quartz(HexagonalFactory):
    """
    Factory to creata an alpha quartz crystal structure
    """
    xtal_name = "alpha_quartz"
    bravais_basis = [[0, 0.4763, 0.6667], [0.4763, 0, 0.3333], [0.5237, 0.5237, 0],
        [0.1588, 0.7439, 0.4612], [0.2561, 0.4149, 0.7945], [0.4149, 0.2561, 0.2055],
        [0.5851, 0.8412, 0.1279], [0.7439, 0.1588, 0.5388], [0.8412, 0.5851, 0.8721]]
    element_basis = (0, 0, 0, 1, 1, 1, 1 ,1, 1)

class beta_cristobalite(SimpleCubicFactory):
    """
    Factory to create a beta cristobalite crystal structure
    """
    xtal_name = "beta_cristobalite"
    bravais_basis = [[0.98184000, 0.51816000,0.48184000],
                    [0.51816000, 0.48184000, 0.98184000], 
                    [0.48184000, 0.98184000, 0.51816000], 
                    [0.01816000, 0.01816000, 0.01816000],   
                    [0.72415800, 0.77584200, 0.22415800], 
                    [0.77584200, 0.22415800, 0.72415800],  
                    [0.22415800, 0.72415800, 0.77584200],  
                    [0.27584200, 0.27584200, 0.27584200],  
                    [0.86042900, 0.34389100, 0.55435200],  
                    [0.36042900, 0.15610900, 0.44564800],  
                    [0.13957100, 0.84389100, 0.94564800],  
                    [0.15610900, 0.44564800, 0.36042900],  
                    [0.94564800, 0.13957100, 0.84389100],  
                    [0.44564800, 0.36042900, 0.15610900],  
                    [0.34389100, 0.55435200, 0.86042900],  
                    [0.55435200, 0.86042900, 0.34389100],  
                    [0.14694300, 0.14694300, 0.14694300],  
                    [0.35305700, 0.85305700, 0.64694300],  
                    [0.64694300, 0.35305700, 0.85305700],  
                    [0.85305700, 0.64694300, 0.35305700],  
                    [0.63957100, 0.65610900, 0.05435200],  
                    [0.05435200, 0.63957100, 0.65610900],  
                    [0.65610900, 0.05435200, 0.63957100],  
                    [0.84389100, 0.94564800, 0.13957100]] 
    element_basis = (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

calc_alpha_quartz = {(14, 14): BKS_ewald(0, 1, 0, 1.28, 2.9, 10.43, np.array([8, 11, 9]), 1e-6),
                     (14, 8): BKS_ewald(18003.7572, 4.87318, 133.5381, 1.28, 2.9, 10.43, np.array([8, 11, 9]), 1e-6),
                     (8, 8): BKS_ewald(1388.7730, 2.76000, 175.0000, 1.28, 2.9, 10.43, np.array([8, 11, 9]), 1e-6)} 

calc_beta_cristobalite = {(14, 14): BKS_ewald(0, 1, 0, 0.96, 3.8, 7, np.array([9, 9, 9]), 1e-6),
                          (14, 8): BKS_ewald(18003.7572, 4.87318, 133.5381, 0.96, 3.8, 7, np.array([9, 9, 9]), 1e-6),
                          (8, 8): BKS_ewald(1388.7730, 2.76000, 175.0000, 0.96, 3.8, 7, np.array([9, 9, 9]), 1e-6)} 

### 

# Test for alpha quartz

@pytest.mark.parametrize('a0', [4.7, 4.9, 5.1])
def test_stress_alpha_quartz(a0):
    """
    Test the computation of stress
    """
    structure = alpha_quartz()
    atoms = structure(["Si", "O"], size=[1, 1, 1], latticeconstant={"a": a0, "b": a0, "c": 5.4})
    charges = np.zeros(len(atoms))
    for i in range(len(atoms)):
        if atoms.get_chemical_symbols()[i] == "Si":
            charges[i] = +2.4
        elif atoms.get_chemical_symbols()[i] == "O":
            charges[i] = -1.2
    atoms.set_array("charge", charges, dtype=float)

    b = Ewald(calc_alpha_quartz)
    atoms.calc = b

    s = atoms.get_stress()
    sn = b.calculate_numerical_stress(atoms, d=0.001)

    print("Stress ana: \n", s)
    print("Stress num: \n", sn)

    np.testing.assert_allclose(s, sn, atol=1e-3)

@pytest.mark.parametrize('a0', [4.7, 4.9, 5.1])
def test_forces_alpha_quartz(a0):
    """
    Test the computation of forces
    """
    structure = alpha_quartz()
    atoms = structure(["Si", "O"], size=[1, 1, 1], latticeconstant={"a": a0, "b": a0, "c": 5.4, "gamma": 120})
    charges = np.zeros(len(atoms))
    for i in range(len(atoms)):
        if atoms.get_chemical_symbols()[i] == "Si":
            charges[i] = +2.4
        elif atoms.get_chemical_symbols()[i] == "O":
            charges[i] = -1.2
    atoms.set_array("charge", charges, dtype=float)

    b = Ewald(calc_alpha_quartz)
    atoms.calc = b

    f = atoms.get_forces()
    fn = b.calculate_numerical_forces(atoms, d=0.0001)

    print("f_ana: \n", f[:5,:])
    print("f_num: \n", fn[:5,:])

    np.testing.assert_allclose(f, fn, atol=1e-3)

def test_hessian_alpha_quartz():
    """
    Test the computation of the Hessian matrix
    """
    structure = alpha_quartz()
    atoms = structure(["Si", "O"], size=[1, 1, 1], latticeconstant={"a": 4.9, "b": 4.9, "c": 5.4, "gamma": 120})
    charges = np.zeros(len(atoms))
    for i in range(len(atoms)):
        if atoms.get_chemical_symbols()[i] == "Si":
            charges[i] = +2.4
        elif atoms.get_chemical_symbols()[i] == "O":
            charges[i] = -1.2
    atoms.set_array("charge", charges, dtype=float)  

    b = Ewald(calc_alpha_quartz)
    atoms.calc = b
    FIRE(atoms, logfile=None).run(fmax=1e-2)

    H_num = fd_hessian(atoms, dx=1e-5, indices=None)
    H_ana = b.get_hessian(atoms)

    print("H_num: \n", H_num.todense()[:6,:6])
    print("H_ana: \n", H_ana[:6,:6])

    np.testing.assert_allclose(H_num.todense(), H_ana, atol=1e-3)

def test_non_affine_forces_alpha_quartz():
    """
    Test the computation of non-affine forces 
    """
    structure = alpha_quartz()
    atoms = structure(["Si", "O"], size=[1, 1, 1], latticeconstant={"a": 4.9, "b": 4.9, "c": 5.4, "gamma": 120})
    charges = np.zeros(len(atoms))
    for i in range(len(atoms)):
        if atoms.get_chemical_symbols()[i] == "Si":
            charges[i] = +2.4
        elif atoms.get_chemical_symbols()[i] == "O":
            charges[i] = -1.2
    atoms.set_array("charge", charges, dtype=float)  

    b = Ewald(calc_alpha_quartz)
    atoms.calc = b

    FIRE(atoms, logfile=None).run(fmax=1e-2)

    naForces_ana = b.get_nonaffine_forces(atoms)
    naForces_num = b.get_numerical_non_affine_forces(atoms, d=1e-5)

    print("Num: \n", naForces_num[:1])
    print("Ana: \n", naForces_ana[:1])

    np.testing.assert_allclose(naForces_num, naForces_ana, atol=1e-2)

def test_birch_coefficients_alpha_quartz():
    """
    Test the computation of the affine elastic constants + stresses
    """
    structure = alpha_quartz()
    atoms = structure(["Si", "O"], size=[1, 1, 1], latticeconstant={"a": 4.9, "b": 4.9, "c": 5.4, "gamma": 120})
    charges = np.zeros(len(atoms))
    for i in range(len(atoms)):
        if atoms.get_chemical_symbols()[i] == "Si":
            charges[i] = +2.4
        elif atoms.get_chemical_symbols()[i] == "O":
            charges[i] = -1.2
    atoms.set_array("charge", charges, dtype=float)  

    b = Ewald(calc_alpha_quartz)
    atoms.calc = b

    FIRE(atoms, logfile=None).run(fmax=1e-2)

    C_num, Cerr = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=11, delta=1e-5, optimizer=None, verbose=False)
    C_ana = full_3x3x3x3_to_Voigt_6x6(b.get_birch_coefficients(atoms), check_symmetry=False)

    print("Stress: \n", atoms.get_stress() / GPa) 
    print("C_num: \n", C_num / GPa)
    print("C_ana: \n", C_ana / GPa)

    np.testing.assert_allclose(C_num, C_ana, atol=1e-1)

def test_full_elastic_alpha_quartz():
    """
    Test the computation of the affine elastic constants + stresses + non-affine elastic constants
    """
    structure = alpha_quartz()
    atoms = structure(["Si", "O"], size=[1, 1, 1], latticeconstant={"a": 4.94, "b": 4.94, "c": 5.44, "gamma": 120})
    charges = np.zeros(len(atoms))
    for i in range(len(atoms)):
        if atoms.get_chemical_symbols()[i] == "Si":
            charges[i] = +2.4
        elif atoms.get_chemical_symbols()[i] == "O":
            charges[i] = -1.2
    atoms.set_array("charge", charges, dtype=float)  

    b = Ewald(calc_alpha_quartz)
    atoms.calc = b
    FIRE(ase.constraints.UnitCellFilter(atoms, mask=[1, 1, 1, 0, 0, 0]), logfile=None).run(fmax=1e-3)    

    C_num, Cerr = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=11, delta=1e-3, optimizer=FIRE, fmax=1e-3, logfile=None, verbose=False)
    C_ana = full_3x3x3x3_to_Voigt_6x6(b.get_birch_coefficients(atoms), check_symmetry=False)
    C_na = full_3x3x3x3_to_Voigt_6x6(b.get_non_affine_contribution_to_elastic_constants(atoms))

    print("stress: \n", atoms.get_stress())
    print("C_num: \n", C_num )
    print("C_ana: \n", C_ana + C_na) 

    np.testing.assert_allclose(C_num, C_ana+C_na, atol=1e-1)

"""
def test_real_elastic_alpha_quartz():
    # This test uses a larger cutoff and cell in order to verify the precise experimental measured values of C
    # Test the computation of the affine elastic constants + stresses + non-affine elastic constants
    structure = alpha_quartz()
    atoms = structure(["Si", "O"], size=[2, 2, 2], latticeconstant={"a": 4.94, "b": 4.94, "c": 5.44, "gamma": 120})
    charges = np.zeros(len(atoms))
    for i in range(len(atoms)):
        if atoms.get_chemical_symbols()[i] == "Si":
            charges[i] = +2.4
        elif atoms.get_chemical_symbols()[i] == "O":
            charges[i] = -1.2
    atoms.set_array("charge", charges, dtype=float)  


    calc_alpha_quartz = {(14, 14): BKS_ewald(0, 1, 0, None, 9, None, np.array([None, None, None]), 1e-5),
            (14, 8): BKS_ewald(18003.7572, 4.87318, 133.5381, None, 9, None, np.array([None, None, None]), 1e-5),
            (8, 8): BKS_ewald(1388.7730, 2.76000, 175.0000, None, 9, None, np.array([None, None, None]), 1e-5)} 

    b = Ewald(calc_alpha_quartz)
    atoms.calc = b
    FIRE(ase.constraints.UnitCellFilter(atoms, mask=[1, 1, 1, 0, 0, 0]), logfile=None).run(fmax=1e-3)    
    #FIRE(atoms, logfile=None).run(fmax=1e-4)

    C_num, Cerr = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=11, delta=1e-3, optimizer=FIRE, fmax=1e-3, logfile=None, verbose=False)
    C_ana = full_3x3x3x3_to_Voigt_6x6(b.get_birch_coefficients(atoms), check_symmetry=False)
    C_na = full_3x3x3x3_to_Voigt_6x6(b.get_non_affine_contribution_to_elastic_constants(atoms))

    print("stress: ", atoms.get_stress())
    print("C_num: \n", C_num )
    print("C_ana: \n", C_ana + C_na) 

    np.testing.assert_allclose(C_num, C_ana+C_na, atol=1e-1)
"""


###

# Beta crisotbalite

@pytest.mark.parametrize('a0', [6, 7, 8, 9])
def test_stress_beta_cristobalite(a0):
    """
    Test the computation of stress
    """
    structure = beta_cristobalite()
    atoms = structure(["Si", "O"], size=[1, 1, 1], latticeconstant=a0)
    charges = np.zeros(len(atoms))
    for i in range(len(atoms)):
        if atoms.get_chemical_symbols()[i] == "Si":
            charges[i] = +2.4
        elif atoms.get_chemical_symbols()[i] == "O":
            charges[i] = -1.2
    atoms.set_array("charge", charges, dtype=float)

    b = Ewald(calc_beta_cristobalite)
    atoms.calc = b

    s = atoms.get_stress()
    sn = b.calculate_numerical_stress(atoms, d=0.0001)

    print("Stress ana: \n", s)
    print("Stress num: \n", sn)

    np.testing.assert_allclose(s, sn, atol=1e-3)

@pytest.mark.parametrize('a0', [7, 9])
def test_forces_beta_cristobalite(a0):
    """
    Test the computation of forces
    """
    structure = beta_cristobalite()
    atoms = structure(["Si", "O"], size=[1, 1, 1], latticeconstant=a0)
    charges = np.zeros(len(atoms))
    for i in range(len(atoms)):
        if atoms.get_chemical_symbols()[i] == "Si":
            charges[i] = +2.4
        elif atoms.get_chemical_symbols()[i] == "O":
            charges[i] = -1.2
    atoms.set_array("charge", charges, dtype=float)

    b = Ewald(calc_beta_cristobalite)
    atoms.calc = b

    f = atoms.get_forces()
    fn = b.calculate_numerical_forces(atoms, d=1e-5)

    print("forces ana: \n", f[:5,:])
    print("forces num: \n", fn[:5,:])

    np.testing.assert_allclose(f, fn, atol=1e-3)

def test_hessian_beta_cristobalite():
    """
    Test the computation of the Hessian matrix
    """
    structure = beta_cristobalite()
    atoms = structure(["Si", "O"], size=[1, 1, 1], latticeconstant=7)
    charges = np.zeros(len(atoms))
    for i in range(len(atoms)):
        if atoms.get_chemical_symbols()[i] == "Si":
            charges[i] = +2.4
        elif atoms.get_chemical_symbols()[i] == "O":
            charges[i] = -1.2
    atoms.set_array("charge", charges, dtype=float)

    b = Ewald(calc_beta_cristobalite)
    atoms.calc = b

    FIRE(atoms, logfile=None).run(fmax=1e-2)

    H_num = fd_hessian(atoms, dx=1e-5, indices=None)
    H_ana = b.get_hessian(atoms)

    np.testing.assert_allclose(H_num.todense(), H_ana, atol=1e-3)

def test_non_affine_forces_beta_cristobalite():
    """
    Test the computation of non-affine forces 
    """
    structure = beta_cristobalite()
    atoms = structure(["Si", "O"], size=[1, 1, 1], latticeconstant=7)
    charges = np.zeros(len(atoms))
    for i in range(len(atoms)):
        if atoms.get_chemical_symbols()[i] == "Si":
            charges[i] = +2.4
        elif atoms.get_chemical_symbols()[i] == "O":
            charges[i] = -1.2
    atoms.set_array("charge", charges, dtype=float)

    b = Ewald(calc_beta_cristobalite)
    atoms.calc = b

    FIRE(atoms, logfile=None).run(fmax=1e-2)

    naForces_ana = b.get_nonaffine_forces(atoms)
    naForces_num = b.get_numerical_non_affine_forces(atoms, d=1e-5)

    print("Num: \n", naForces_num[:1])
    print("Ana: \n", naForces_ana[:1])

    np.testing.assert_allclose(naForces_num, naForces_ana, atol=1e-2)

def test_birch_coefficients_beta_cristobalite():
    """
    Test the computation of the affine elastic constants + stresses
    """
    structure = beta_cristobalite()
    atoms = structure(["Si", "O"], size=[1, 1, 1], latticeconstant=7)
    charges = np.zeros(len(atoms))
    for i in range(len(atoms)):
        if atoms.get_chemical_symbols()[i] == "Si":
            charges[i] = +2.4
        elif atoms.get_chemical_symbols()[i] == "O":
            charges[i] = -1.2
    atoms.set_array("charge", charges, dtype=float)

    b = Ewald(calc_beta_cristobalite)
    atoms.calc = b

    FIRE(atoms, logfile=None).run(fmax=1e-2)

    C_num, Cerr = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=11, delta=1e-5, optimizer=None, verbose=False)
    C_ana = full_3x3x3x3_to_Voigt_6x6(b.get_birch_coefficients(atoms), check_symmetry=False)

    print("C_num: \n", C_num)
    print("C_ana: \n", C_ana)

    np.testing.assert_allclose(C_num, C_ana, atol=1e-1)

def test_non_affine_elastic_beta_cristobalite():
    """
    Test the computation of the affine elastic constants + stresses + non-affine elastic constants
    """
    structure = beta_cristobalite()
    atoms = structure(["Si", "O"], size=[1, 1, 1], latticeconstant=7)
    charges = np.zeros(len(atoms))
    for i in range(len(atoms)):
        if atoms.get_chemical_symbols()[i] == "Si":
            charges[i] = +2.4
        elif atoms.get_chemical_symbols()[i] == "O":
            charges[i] = -1.2
    atoms.set_array("charge", charges, dtype=float)

    b = Ewald(calc_beta_cristobalite)
    atoms.calc = b

    FIRE(ase.constraints.UnitCellFilter(atoms, mask=[1, 1, 1, 0, 0, 0]), logfile=None).run(fmax=1e-3)  

    C_num, Cerr = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=11, delta=1e-3, optimizer=FIRE, fmax=1e-3, logfile=None, verbose=False)
    C_ana = full_3x3x3x3_to_Voigt_6x6(b.get_birch_coefficients(atoms), check_symmetry=False)
    C_na = full_3x3x3x3_to_Voigt_6x6(b.get_non_affine_contribution_to_elastic_constants(atoms))

    print("stress: \n", atoms.get_stress())
    print("C_num: \n", C_num)
    print("C_ana: \n", C_ana + C_na) 

    np.testing.assert_allclose(C_num, C_ana+C_na, atol=1e-1)
