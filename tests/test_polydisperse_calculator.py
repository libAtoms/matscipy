#
# Copyright 2020-2021 Jan Griesser (U. Freiburg)
#           2020-2021 Lars Pastewka (U. Freiburg)
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

import ase.io as io
import ase.constraints
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
from ase.optimize import FIRE

import matscipy.calculators.polydisperse as calculator
from matscipy.elasticity import (
    full_3x3x3x3_to_Voigt_6x6,
    measure_triclinic_elastic_constants,
    nonaffine_elastic_contribution
    )
from matscipy.calculators.polydisperse import InversePowerLawPotential, Polydisperse
from matscipy.numerical import (
    numerical_forces,
    numerical_stress,
    numerical_hessian,
    numerical_nonaffine_forces,
)

def test_forces_dimer():
    """
    Check forces for a dimer
    """
    d = 1.2
    L = 10 
    atomic_configuration = Atoms("HH", 
                                 positions=[(L/2, L/2, L/2), (L/2 + d, L/2, L/2)],
                                 cell=[L, L, L],
                                 pbc=[1, 1, 1]
                                 )
    atomic_configuration.set_array("size", np.array([1.3, 2.22]), dtype=float)
    atomic_configuration.set_masses(masses = np.repeat(1.0, len(atomic_configuration)))
    calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))
    atomic_configuration.calc = calc
    f = atomic_configuration.get_forces()
    fn = numerical_forces(atomic_configuration, d=1e-5)
    np.testing.assert_allclose(f, fn, atol=1e-4, rtol=1e-4)

def test_forces_crystal():
    """
    Check forces for a crystalline configuration
    """
    atoms = FaceCenteredCubic('H', size=[2, 2, 2], latticeconstant=2.37126)
    calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))
    atoms.set_masses(masses = np.repeat(1.0, len(atoms)))       
    atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
    atoms.calc = calc
    f = atoms.get_forces()
    fn = numerical_forces(atoms, d=1e-5)
    np.testing.assert_allclose(f, fn, atol=1e-4, rtol=1e-4)

@pytest.mark.parametrize('a0', [2.0, 2.5, 3.0])
def test_crystal_stress(a0):
    """
    Test the computation of stresses for a crystal
    """
    calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))
    atoms = FaceCenteredCubic('H', size=[2, 2, 2], latticeconstant=a0) 
    atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
    atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
    atoms.calc = calc
    s = atoms.get_stress()
    sn = numerical_stress(atoms, d=1e-5)
    np.testing.assert_allclose(s, sn, atol=1e-4, rtol=1e-4)

def test_glass_stress():
    """
    Test the computation of stresses for a glass
    """
    calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))
    atoms = io.read('glass_min.xyz')
    atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
    atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
    atoms.set_atomic_numbers(np.repeat(1.0, len(atoms)))   
    atoms.calc = calc
    s = atoms.get_stress()
    sn = numerical_stress(atoms, d=1e-5)
    np.testing.assert_allclose(s, sn, atol=1e-4, rtol=1e-4)

def test_symmetry_sparse():
    """
    Test the symmetry of the dense Hessian matrix 
    """
    atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=2.37126)
    calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))
    atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
    atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
    atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=1e-5)
    H = calc.get_property('hessian', atoms).todense()
    np.testing.assert_allclose(np.sum(np.abs(H-H.T)), 0, atol=1e-6, rtol=1e-6)

def test_hessian_random_structure():
    """
    Test the computation of the Hessian matrix
    """ 
    atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=2.37126)
    calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))
    atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
    atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
    atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=1e-5)
    H_analytical = atoms.calc.get_property('hessian', atoms).todense()
    H_numerical = numerical_hessian(atoms, dx=1e-5, indices=None).todense()
    np.testing.assert_allclose(H_analytical, H_numerical, atol=1e-4, rtol=1e-6)

def test_hessian_divide_by_masses():
    """
    Test the computation of the Hessian matrix 
    """
    atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=2.37126)     
    atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
    masses_n = np.random.randint(1, 10, size=len(atoms))
    atoms.set_masses(masses=masses_n)
    calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))
    atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=1e-5)
    D_analytical = calc.get_property('dynamical_matrix', atoms).todense()
    H_analytical = calc.get_property('hessian', atoms).todense()
    masses_nc = masses_n.repeat(3)
    H_analytical /= np.sqrt(masses_nc.reshape(-1,1)*masses_nc.reshape(1,-1))
    np.testing.assert_allclose(H_analytical, D_analytical, atol=1e-6, rtol=1e-6)

@pytest.mark.parametrize('a0', [2.5, 3.0])
def test_crystal_non_affine_forces(a0):
    """
    Test the computation of the non-affine forces for a crystal
    """ 
    calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))
    atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=a0) 
    atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
    atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
    atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=1e-6)
    naForces_num = numerical_nonaffine_forces(atoms, d=1e-6)
    naForces_ana = calc.get_property('nonaffine_forces', atoms)  
    np.testing.assert_allclose(naForces_num, naForces_ana, atol=1e-2) 

def test_glass_non_affine_forces():
    """
    Test the computation of the non-affine forces for a glass
    """ 
    calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))
    atoms = io.read("glass_min.xyz")
    atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
    atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
    atoms.set_atomic_numbers(np.repeat(1.0, len(atoms))) 
    atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=1e-6)
    naForces_num = numerical_nonaffine_forces(atoms, d=1e-6)
    naForces_ana = calc.get_property('nonaffine_forces', atoms)  
    np.testing.assert_allclose(naForces_num, naForces_ana, atol=1) 