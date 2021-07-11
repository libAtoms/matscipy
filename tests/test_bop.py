#! /usr/bin/env python

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

import unittest

import numpy as np

import ase

import matscipytest

import ase.io as io
from matscipy.calculators.bop_sw import AbellTersoffBrennerStillingerWeber 
import matscipy.calculators.bop_sw.explicit_forms.stillinger_weber as sw
import matscipy.calculators.bop_sw.explicit_forms.kumagai as kum
import matscipy.calculators.bop_sw.explicit_forms.tersoff3 as t3
from matscipy.calculators.bop_sw.explicit_forms import KumagaiTersoff, TersoffIII, StillingerWeber
from ase import Atoms
import ase.io
from matscipy.hessian_finite_differences import fd_hessian
from ase.lattice.cubic import Diamond
from matscipy.elasticity import fit_elastic_constants, elastic_moduli, full_3x3x3x3_to_Voigt_6x6, measure_triclinic_elastic_constants
from matscipy.calculators.calculator import MatscipyCalculator
from ase.units import GPa

###

class TestAbellTersoffBrennerStillingerWeber(matscipytest.MatSciPyTestCase):

    def test_born_elastic_constants(self):
        atoms = Diamond('Si', size=[4,4,4], latticeconstant=5.431)
        io.write("cSi.xyz", atoms)
        kumagai_potential = kum.kumagai
        calculator = AbellTersoffBrennerStillingerWeber(**KumagaiTersoff(kumagai_potential))
        atoms.set_calculator(calculator)
        C_num, Cerr = fit_elastic_constants(atoms, symmetry="triclinic", N_steps=11, delta=1e-2, optimizer=None)
        #C_ana = calculator.get_birch_coefficients(atoms)
        C_ana = calculator.get_born_elastic_constants_from_second_derivative(atoms)
        naC_ana = calculator.get_non_affine_contribution_to_elastic_constants(atoms)
        print("Stress: \n", atoms.get_stress())
        print("C_num: \n", -C_num)
        print("C_ana: \n", full_3x3x3x3_to_Voigt_6x6(C_ana))
        print("naC_ana: \n", full_3x3x3x3_to_Voigt_6x6(naC_ana))
        self.assertArrayAlmostEqual(-C_num, full_3x3x3x3_to_Voigt_6x6(C_ana), tol=1) 

    """
    def test_non_affine_forces(self):
        atoms = Diamond('Si', size=[1,1,1], latticeconstant=5.431)
        io.write("cSi.xyz", atoms)
        kumagai_potential = kum.kumagai
        calculator = AbellTersoffBrennerStillingerWeber(**KumagaiTersoff(kumagai_potential))
        atoms.set_calculator(calculator)
        naF_ana = calculator.get_non_affine_forces_from_second_derivative(atoms)
        naF_num = calculator.get_numerical_non_affine_forces(atoms, d=1e-5)
        print("naF_ana: \n", naF_ana[0])
        print("naF_ana: \n", naF_num[0])
        self.assertArrayAlmostEqual(naF_ana, naF_num, tol=1e-2)

    def test_non_affine_forces(self):
        atoms = Diamond('Si', size=[1,1,1], latticeconstant=5.431)
        io.write("cSi.xyz", atoms)
        kumagai_potential = kum.kumagai
        calculator = AbellTersoffBrennerStillingerWeber(**KumagaiTersoff(kumagai_potential))
        atoms.set_calculator(calculator)
        naF_ana = calculator.get_non_affine_forces_from_second_derivative(atoms)
        naF_num = calculator.get_numerical_non_affine_forces(atoms, d=1e-5)
        print("naF_ana: \n", naF_ana[0])
        print("naF_ana: \n", naF_num[0])
        self.assertArrayAlmostEqual(naF_ana, naF_num, tol=1e-2)


    def test_computation_of_hessian(self):
        atoms = Diamond('Si', size=[1,1,1], latticeconstant=5.431)
        #io.write("cSi.xyz", atoms)
        kumagai_potential = kum.kumagai
        calculator = AbellTersoffBrennerStillingerWeber(**KumagaiTersoff(kumagai_potential))
        atoms.set_calculator(calculator)
        H_ana = calculator.get_hessian(atoms).todense()
        H_ana2 = calculator.get_hessian_from_second_derivative(atoms)

        self.assertArrayAlmostEqual(H_ana, H_ana2)

    def test_hessian_divide_by_masses(self):

        #Test the computation of dynamical matrix

        atoms = ase.io.read('aSi.cfg')
        masses_n = np.random.randint(1, 10, size=len(atoms))
        atoms.set_masses(masses=masses_n)
        kumagai_potential = kum.kumagai
        calc = AbellTersoffBrennerStillingerWeber(**KumagaiTersoff(kumagai_potential))
        D_ana = calc.get_hessian(atoms, divide_by_masses=True).todense()
        H_ana = calc.get_hessian(atoms).todense()
        masses_nc = masses_n.repeat(3)
        H_ana /= np.sqrt(masses_nc.reshape(-1,1)*masses_nc.reshape(1,-1))
        self.assertArrayAlmostEqual(D_ana, H_ana, tol=1e-4)

    def test_kumagai_tersoff(self):

        #Test forces and hessian matrix for Kumagai  

        kumagai_potential = kum.kumagai
        for d in np.arange(1.0, 2.3, 0.15):
            small = Atoms([14]*4, [(d, 0, d/2), (0, 0, 0), (d, 0, 0), (0, 0, d)], cell=(100, 100, 100))
            small.center(vacuum=10.0)
            small2 = Atoms([14]*5, [(d, 0, d/2), (0, 0, 0), (d, 0, 0), (0, 0, d), (0, d, d)], cell=(100, 100, 100))
            small2.center(vacuum=10.0)

            self.compute_forces_and_hessian(small, KumagaiTersoff(kumagai_potential))
            self.compute_forces_and_hessian(small2, KumagaiTersoff(kumagai_potential))

        aSi = ase.io.read('aSi.cfg')
        self.compute_forces_and_hessian(aSi, KumagaiTersoff(kumagai_potential))

    def test_tersoffIII(self):

        #Test forces and hessian matrix for Tersoff3

        T3_Si_potential = t3.tersoff3_Si
        for d in np.arange(1.0, 2.3, 0.15):        
            small = Atoms([14]*4, [(d, 0, d/2), (0, 0, 0), (d, 0, 0), (0, 0, d)], cell=(100, 100, 100))
            small.center(vacuum=10.0)
            small2 = Atoms([14]*5, [(d, 0, d/2), (0, 0, 0), (d, 0, 0), (0, 0, d), (0, d, d)], cell=(100, 100, 100))
            small2.center(vacuum=10.0)
        
            self.compute_forces_and_hessian(small, TersoffIII(T3_Si_potential))
            self.compute_forces_and_hessian(small2, TersoffIII(T3_Si_potential))

        aSi = ase.io.read('aSi.cfg')
        self.compute_forces_and_hessian(aSi, TersoffIII(T3_Si_potential))

    def test_stillinger_weber(self):

        #Test forces and hessian matrix for Stillinger-Weber

        SW_potential = sw.original_SW
        for d in np.arange(1.0, 1.8, 0.15):  
            small = Atoms([14]*4, [(d, 0, d/2), (0, 0, 0), (d, 0, 0), (0, 0, d)], cell=(100, 100, 100))
            small.center(vacuum=10.0)
            small2 = Atoms([14]*5, [(d, 0, d/2), (0, 0, 0), (d, 0, 0), (0, 0, d), (0, d, d)], cell=(100, 100, 100))
            small2.center(vacuum=10.0)
        
            self.compute_forces_and_hessian(small, StillingerWeber(SW_potential))
            self.compute_forces_and_hessian(small2, StillingerWeber(SW_potential))

        aSi = ase.io.read('aSi.cfg')
        self.compute_forces_and_hessian(aSi, StillingerWeber(SW_potential))

    def test_generic_potential_form(self):
        self.test_cutoff = 2.4
        d = 2.0  # Si2 bondlength
        small = Atoms([14]*4, [(d, 0, d/2), (0, 0, 0), (d, 0, 0), (0, 0, d)], cell=(100, 100, 100))
        small.center(vacuum=10.0)
        small2 = Atoms([14]*5, [(d, 0, d/2), (0, 0, 0), (d, 0, 0), (0, 0, d), (0, d, d)], cell=(100, 100, 100))
        small2.center(vacuum=10.0)
        self.compute_forces_and_hessian(small, self.term1())
        self.compute_forces_and_hessian(small, self.term4())
        self.compute_forces_and_hessian(small, self.d11_term5())
        self.compute_forces_and_hessian(small, self.d22_term5())

        self.compute_forces_and_hessian(small2, self.term1())
        self.compute_forces_and_hessian(small2, self.term4())
        self.compute_forces_and_hessian(small2, self.d11_term5())
        self.compute_forces_and_hessian(small2, self.d22_term5())

    # 0 - Tests Hessian term #4 (with all other terms turned off)
    def term4(self):
        return {'F': lambda x, y: x,
        'G': lambda x, y: np.ones_like(x[:, 0]),
        'd1F': lambda x, y: np.ones_like(x),
        'd11F': lambda x, y: np.zeros_like(x),
        'd2F': lambda x, y: np.zeros_like(y),
        'd22F': lambda x, y: np.zeros_like(y),
        'd12F': lambda x, y: np.zeros_like(y),
        'd1G': lambda x, y: np.zeros_like(y),
        'd2G': lambda x, y: np.zeros_like(y),
        'd11G': lambda x, y: 0*x.reshape(-1,3,1)*y.reshape(-1,1,3), #if beta <= 1 else beta*(beta-1)*x.**(beta-2) * y[:, 2]**gamma,
        'd12G': lambda x, y: 0*x.reshape(-1,3,1)*y.reshape(-1,1,3),
        'd22G': lambda x, y: 0*x.reshape(-1,3,1)*y.reshape(-1,1,3),
        'cutoff': self.test_cutoff}
    
    # 1 - Tests Hessian term #1 (and #4, with all other terms turned off)
    def term1(self):
        return {'F': lambda x, y: x**2,
        'G': lambda x, y: np.ones_like(x[:, 0]),
        'd1F': lambda x, y: 2*x,
        'd11F': lambda x, y: 2*np.ones_like(x),
        'd2F': lambda x, y: np.zeros_like(y),
        'd22F': lambda x, y: np.zeros_like(y),
        'd12F': lambda x, y: np.zeros_like(y),
        'd1G': lambda x, y: np.zeros_like(x),
        'd2G': lambda x, y: np.zeros_like(y),
        'd11G': lambda x, y: 0*x.reshape(-1,3,1)*y.reshape(-1,1,3),
        'd12G': lambda x, y: 0*x.reshape(-1,3,1)*y.reshape(-1,1,3),
        'd22G': lambda x, y: 0*x.reshape(-1,3,1)*y.reshape(-1,1,3),
        'cutoff': self.test_cutoff}

    # 2 - Tests D_11 parts of Hessian term #5
    def d11_term5(self):
        return {
        'F': lambda x, y: y,
        'G': lambda x, y: np.sum(x**2, axis=1),
        'd1F': lambda x, y: np.zeros_like(x),
        'd11F': lambda x, y: np.zeros_like(x),
        'd2F': lambda x, y: np.ones_like(x),
        'd22F': lambda x, y: np.zeros_like(x),
        'd12F': lambda x, y: np.zeros_like(x),
        'd1G': lambda x, y: 2*x,
        'd2G': lambda x, y: np.zeros_like(y),
        'd11G': lambda x, y: np.array([2*np.eye(3)]*x.shape[0]),#np.ones_like(x).reshape(-1,3,1)*np.ones_like(y).reshape(-1,1,3), #if beta <= 1 else beta*(beta-1)*x.**(beta-2) * y[:, 2]**gamma,
        'd12G': lambda x, y: 0*x.reshape(-1,3,1)*y.reshape(-1,1,3),
        'd22G': lambda x, y: 0*x.reshape(-1,3,1)*y.reshape(-1,1,3),
        'cutoff': self.test_cutoff}
    
    # 3 - Tests D_22 parts of Hessian term #5
    def d22_term5(self):
        return {
        'F': lambda x, y: y,
        'G': lambda x, y: np.sum(y**2, axis=1),
        'd1F': lambda x, y: np.zeros_like(x),
        'd11F': lambda x, y: np.zeros_like(x),
        'd2F': lambda x, y: np.ones_like(x),
        'd22F': lambda x, y: np.zeros_like(x),
        'd12F': lambda x, y: np.zeros_like(x),
        'd2G' : lambda x, y: 2*y,
        'd1G' : lambda x, y: np.zeros_like(x),
        'd22G': lambda x, y: np.array([2*np.eye(3)]*x.shape[0]),#np.ones_like(x).reshape(-1,3,1)*np.ones_like(y).reshape(-1,1,3), #if beta <= 1 else beta*(beta-1)*x.**(beta-2) * y[:, 2]**gamma,
        'd12G': lambda x, y: 0*x.reshape(-1,3,1)*y.reshape(-1,1,3),
        'd11G': lambda x, y: 0*x.reshape(-1,3,1)*y.reshape(-1,1,3),
        'cutoff': self.test_cutoff}
    """

    def compute_forces_and_hessian(self, a, par):

        #function to test the bop AbellTersoffBrenner class on
            #a potential given by the form defined in par

        #Parameters
        #----------
        #a : ase atoms object
        #    passes an atomic configuration as an ase atoms object
        #par : bop explicit form
         #   defines the explicit form of the bond order potential
        

        calculator = AbellTersoffBrennerStillingerWeber(**par)
        a.set_calculator(calculator)

        ana_forces = a.get_forces()
        num_forces = calculator.calculate_numerical_forces(a, d=1e-5)
        #print('num\n', num_forces)
        #print('ana\n', ana_forces)
        assert np.allclose(ana_forces, num_forces, rtol=1e-3)
        
        ana_hessian = calculator.get_hessian(a).todense()
        num_hessian = fd_hessian(a, dx=1e-5, indices=None).todense()
        #print('ana\n', ana_hessian)
        #print('num\n', num_hessian)
        #print('ana - num\n', (np.abs(ana_hessian - num_hessian) > 1e-6).astype(int))
        assert np.allclose(ana_hessian, ana_hessian.T, atol=1e-6)
        assert np.allclose(ana_hessian, num_hessian, atol=1e-4)

###

if __name__ == '__main__':
    unittest.main()
