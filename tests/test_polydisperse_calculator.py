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
import unittest
import sys

import numpy as np
from numpy.linalg import norm

from scipy.linalg import eigh

import ase.io as io
from ase import Atoms
import ase.constraints
from ase.lattice.compounds import B1, B2, L1_0, L1_2
from ase.lattice.cubic import FaceCenteredCubic
from ase.optimize import FIRE
from ase.units import GPa
from ase.phonons import Phonons


import matscipytest
import matscipy.calculators.polydisperse as calculator
from matscipy.elasticity import fit_elastic_constants, elastic_moduli, full_3x3x3x3_to_Voigt_6x6, measure_triclinic_elastic_constants
from matscipy.calculators.polydisperse import InversePowerLawPotential, Polydisperse
from matscipy.hessian_finite_differences import fd_hessian

###


class TestPolydisperseCalculator(matscipytest.MatSciPyTestCase):

    tol = 1e-4

    def test_forces_dimer(self):
        d = 1.2
        L = 10 
        atomic_configuration = Atoms("HH", 
                                     positions=[(L/2, L/2, L/2), (L/2 + d, L/2, L/2)],
                                     cell=[L, L, L],
                                     pbc=[1, 1, 1]
                                     )
        atomic_configuration.set_array("size", np.array([1.3, 2.22]), dtype=float)
        atomic_configuration.set_masses(masses=np.repeat(1.0, len(atomic_configuration)))
        calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))
        atomic_configuration.set_calculator(calc)
        f = atomic_configuration.get_forces()
        fn = calc.calculate_numerical_forces(atomic_configuration, d=0.0001)
        self.assertArrayAlmostEqual(f, fn, tol=self.tol)

    def test_forces_random_structure(self):
        atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=2.37126)
        calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))
        atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
        atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
        atoms.set_calculator(calc)
        f = atoms.get_forces()
        fn = calc.calculate_numerical_forces(atoms, d=0.0001)
        self.assertArrayAlmostEqual(f, fn, tol=self.tol)

    def test_stress(self):
        # Test the computation of stresses for a crystal and a glass
        calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))
        for a0 in [1.0, 1.5, 2.0, 2.5, 3.0]:
            atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=a0) 
            atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
            atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
            atoms.set_calculator(calc)
            s = atoms.get_stress()
            sn = calc.calculate_numerical_stress(atoms, d=0.0001)
            #print(s)
            #print(sn)
            np.allclose(s, sn, atol=self.tol)

        atoms = io.read('glass_min.xyz')
        atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
        atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
        atoms.set_atomic_numbers(np.repeat(1.0, len(atoms)))   
        atoms.set_calculator(calc)
        s = atoms.get_stress()
        sn = calc.calculate_numerical_stress(atoms, d=0.0001)
        np.allclose(s, sn, atol=self.tol)

    def test_non_affine_forces(self):
        # Test the computation of the non-affine forces 
        calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))

        atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=2.37126) 
        atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
        atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
        atoms.rattle(0.1)  
        atoms.set_calculator(calc)
        FIRE(ase.constraints.StrainFilter(atoms, mask=[1,1,1,0,0,0]),
            logfile=None).run(fmax=1e-5)
        naForces_num = calc.get_numerical_non_affine_forces(atoms, d=1e-5)
        naForces_ana = calc.get_nonaffine_forces(atoms)  
        np.allclose(naForces_num, naForces_ana, atol=0.1) 

        atoms = io.read("glass_min.xyz")
        atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
        atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
        atoms.set_atomic_numbers(np.repeat(1.0, len(atoms))) 
        atoms.set_calculator(calc)
        FIRE(ase.constraints.StrainFilter(atoms, mask=[1,1,1,0,0,0]),
            logfile=None).run(fmax=1e-5)
        naForces_num = calc.get_numerical_non_affine_forces(atoms, d=1e-5)
        naForces_ana = calc.get_nonaffine_forces(atoms)    
        np.allclose(naForces_num, naForces_ana, atol=0.1) 

    def test_birch_elastic_constants(self):
        # Test the Birch elastic constants
        calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))

        for a0 in [1.0, 1.5, 2.0, 2.5, 3.0]:
            atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=2.37126) 
            atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
            atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float) 
            atoms.set_calculator(calc)
            FIRE(ase.constraints.StrainFilter(atoms, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.1)
            FIRE(atoms, logfile=None).run(fmax=1e-5)
            C_num, Cerr = fit_elastic_constants(atoms, symmetry="cubic", N_steps=7, delta=1e-4, optimizer=None, verbose=False)
            C_ana = full_3x3x3x3_to_Voigt_6x6(calc.get_birch_coefficients(atoms))
            np.allclose(C_num, C_ana, atol=0.1)

        atoms = io.read("glass_min.xyz")
        atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
        atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
        atoms.set_atomic_numbers(np.repeat(1.0, len(atoms))) 
        atoms.set_calculator(calc)
        FIRE(ase.constraints.StrainFilter(atoms, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.1)
        FIRE(atoms, logfile=None).run(fmax=1e-5)
        C_num, Cerr = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=7, delta=1e-4, optimizer=None, verbose=False)
        C_ana = full_3x3x3x3_to_Voigt_6x6(calc.get_birch_coefficients(atoms))
        #print("C_ana: \n", C_ana)
        #print("C_num: \n", C_num)
        np.allclose(C_num, C_ana, atol=0.1)

    def test_non_affine_elastic_constants(self):
        # Test the computation of Birch elastic constants and correction due to non-affine displacements
        calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))

        atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=2.37126) 
        atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
        atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
        atoms.rattle(0.1)  
        atoms.set_calculator(calc)
        dyn = FIRE(atoms, logfile=None).run(fmax=1e-5)
        C_num, Cerr = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=7, delta=1e-4, optimizer=FIRE, fmax=1e-5, verbose=False)
        anaC_na = full_3x3x3x3_to_Voigt_6x6(calc.get_non_affine_contribution_to_elastic_constants(atoms, tol=1e-5))
        anaC_af = full_3x3x3x3_to_Voigt_6x6(calc.get_birch_coefficients(atoms))
        np.allclose(C_num, anaC_af + anaC_na, atol=0.1)
       
        atoms = io.read("glass_min.xyz")
        atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
        atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
        atoms.set_atomic_numbers(np.repeat(1.0, len(atoms))) 
        atoms.set_calculator(calc)
        dyn = FIRE(atoms, logfile=None).run(fmax=1e-5)   
        C_num, Cerr = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=7, delta=1e-4, optimizer=FIRE, fmax=1e-5, verbose=False)
        Cana_af = full_3x3x3x3_to_Voigt_6x6(calc.get_birch_coefficients(atoms))
        Cana_na = full_3x3x3x3_to_Voigt_6x6(calc.get_non_affine_contribution_to_elastic_constants(atoms, tol=1e-5), tol=0.1)
        np.allclose(C_num, Cana_na + Cana_af, atol=0.1)
        # 
        H_nn = calc.get_hessian(atoms, "sparse").todense()
        eigenvalues, eigenvectors = eigh(H_nn, subset_by_index=[3,3*len(atoms)-1])
        Cana2_na = full_3x3x3x3_to_Voigt_6x6(calc.get_non_affine_contribution_to_elastic_constants(atoms, eigenvalues, eigenvectors), tol=0.1)
        np.allclose(C_num, Cana2_na + Cana_af, atol=0.1)

    def test_symmetry_sparse(self):
        #Test the symmetry of the dense Hessian matrix 
        atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=2.37126)
        calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))
        atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
        atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
        atoms.set_calculator(calc)
        dyn = FIRE(atoms, logfile=None).run(fmax=1e-5)
        H = calc.get_hessian(atoms)
        H = H.todense()
        self.assertArrayAlmostEqual(np.sum(np.abs(H-H.T)), 0, tol=1e-5)

    def test_hessian_random_structure(self):
        #Test the computation of the Hessian matrix 
        atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=2.37126)
        calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))
        atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
        atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
        atoms.set_calculator(calc)
        dyn = FIRE(atoms, logfile=None).run(fmax=1e-5)
        H_analytical = calc.get_hessian(atoms)
        H_analytical = H_analytical.todense()
        H_numerical = fd_hessian(atoms, dx=1e-5, indices=None)
        H_numerical = H_numerical.todense()
        self.assertArrayAlmostEqual(H_analytical, H_numerical, tol=self.tol)

    def test_hessian_divide_by_masses(self):
        #Test the computation of the Hessian matrix 
        atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=2.37126)     
        atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
        masses_n = np.random.randint(1, 10, size=len(atoms))
        atoms.set_masses(masses=masses_n)
        calc = Polydisperse(InversePowerLawPotential(1.0, 1.4, 0.1, 3, 1, 2.22))
        atoms.set_calculator(calc)
        dyn = FIRE(atoms, logfile=None).run(fmax=1e-5)
        D_analytical = calc.get_hessian(atoms, divide_by_masses=True)
        D_analytical = D_analytical.todense()
        H_analytical = calc.get_hessian(atoms)
        H_analytical = H_analytical.todense()
        masses_nc = masses_n.repeat(3)
        H_analytical /= np.sqrt(masses_nc.reshape(-1,1)*masses_nc.reshape(1,-1))
        self.assertArrayAlmostEqual(H_analytical, D_analytical, tol=self.tol)

###


if __name__ == '__main__':
    unittest.main()
