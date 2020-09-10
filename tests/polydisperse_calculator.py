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

from ase import Atoms
from ase.constraints import StrainFilter, UnitCellFilter
from ase.lattice.compounds import B1, B2, L1_0, L1_2
from ase.lattice.cubic import FaceCenteredCubic
from ase.optimize import FIRE
from ase.units import GPa
from ase.phonons import Phonons


import matscipytest
import matscipy.calculators.polydisperse as calculator
from matscipy.calculators.polydisperse import IPL, Polydisperse
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
        calc = Polydisperse(IPL(1.0, 1.4, 0.1, 3, 1, 2.22))
        atomic_configuration.set_calculator(calc)
        f = atomic_configuration.get_forces()
        fn = calc.calculate_numerical_forces(atomic_configuration, d=0.0001)
        self.assertArrayAlmostEqual(f, fn, tol=self.tol)

    def test_forces_random_structure(self):
        atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=2.37126)
        calc = Polydisperse(IPL(1.0, 1.4, 0.1, 3, 1, 2.22))
        atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
        atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
        atoms.set_calculator(calc)
        f = atoms.get_forces()
        fn = calc.calculate_numerical_forces(atoms, d=0.0001)
        self.assertArrayAlmostEqual(f, fn, tol=self.tol)

    def test_symmetry_sparse(self):
        """
        Test the symmetry of the dense Hessian matrix 

        """
        atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=2.37126)
        calc = Polydisperse(IPL(1.0, 1.4, 0.1, 3, 1, 2.22))
        atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
        atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
        atoms.set_calculator(calc)
        dyn = FIRE(atoms)
        dyn.run(fmax=1e-5)
        H = calc.hessian_matrix(atoms)
        H = H.todense()
        self.assertArrayAlmostEqual(np.sum(np.abs(H-H.T)), 0, tol=1e-5)

    def test_hessian_random_structure(self):
        """
        Test the computation of the Hessian matrix 
        """
        atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=2.37126)
        calc = Polydisperse(IPL(1.0, 1.4, 0.1, 3, 1, 2.22))
        atoms.set_masses(masses=np.repeat(1.0, len(atoms)))       
        atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
        atoms.set_calculator(calc)
        dyn = FIRE(atoms)
        dyn.run(fmax=1e-5)
        H_analytical = calc.hessian_matrix(atoms)
        H_analytical = H_analytical.todense()
        H_numerical = fd_hessian(atoms, dx=1e-5, indices=None)
        H_numerical = H_numerical.todense()
        self.assertArrayAlmostEqual(H_analytical, H_numerical, tol=self.tol)

    def test_hessian_divide_by_masses(self):
        """
        Test the computation of the Hessian matrix 
        """
        atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=2.37126)     
        atoms.set_array("size", np.random.uniform(1.0, 2.22, size=len(atoms)), dtype=float)
        masses_n = np.random.randint(1, 10, size=len(atoms))
        atoms.set_masses(masses=masses_n)
        calc = Polydisperse(IPL(1.0, 1.4, 0.1, 3, 1, 2.22))
        atoms.set_calculator(calc)
        dyn = FIRE(atoms)
        dyn.run(fmax=1e-5)
        D_analytical = calc.hessian_matrix(atoms, divide_by_masses=True)
        D_analytical = D_analytical.todense()
        H_analytical = calc.hessian_matrix(atoms)
        H_analytical = H_analytical.todense()
        masses_nc = masses_n.repeat(3)
        H_analytical /= np.sqrt(masses_nc.reshape(-1,1)*masses_nc.reshape(1,-1))
        self.assertArrayAlmostEqual(H_analytical, D_analytical, tol=self.tol)

###


if __name__ == '__main__':
    unittest.main()
