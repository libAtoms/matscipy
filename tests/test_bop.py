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

from matscipy.calculators.bop import AbellTersoffBrenner 
from matscipy.calculators.bop.explicit_forms import KumagaiTersoff
from ase import Atoms
from ase import io
from matscipy.hessian_finite_differences import fd_hessian

class TestAbellTersoffBrenner(matscipytest.MatSciPyTestCase):

    def test_kumagai_tersoff(self):
        d = 2.0  # Si2 bondlength
        small = Atoms([14]*4, [(d, 0, d/2), (0, 0, 0), (d, 0, 0), (0, 0, d)], cell=(100, 100, 100))
        small.center(vacuum=10.0)
        small2 = Atoms([14]*5, [(d, 0, d/2), (0, 0, 0), (d, 0, 0), (0, 0, d), (0, d, d)], cell=(100, 100, 100))
        small2.center(vacuum=10.0)
        self.compute_forces_and_hessian(small, KumagaiTersoff())

        self.compute_forces_and_hessian(small2, KumagaiTersoff())

        aSi = io.read('aSi.structure_minimum_65atoms_pot_energy.nc')
        self.compute_forces_and_hessian(aSi, KumagaiTersoff())

    def compute_forces_and_hessian(self, a, par):
        """ function to test the bop AbellTersoffBrenner class on
            a potential given by the form defined in par

        Parameters
        ----------
        a : ase atoms object
            passes an atomic configuration as an ase atoms object
        par : bop explicit form
            defines the explicit form of the bond order potential
        
        """
        calculator = AbellTersoffBrenner(**par)
        a.set_calculator(calculator)

        print('FORCES')
        ana_forces = a.get_forces()
        num_forces = calculator.calculate_numerical_forces(a, d=1e-5)
        print('num\n', num_forces)
        print('ana\n', ana_forces)
        assert np.allclose(ana_forces, num_forces, rtol=1e-3)
        
        print('HESSIAN')
        ana_hessian = calculator.calculate_hessian_matrix(a).todense()
        num_hessian = fd_hessian(a, dx=1e-5, indices=None).todense()
        print('ana\n', ana_hessian)
        print('num\n', num_hessian)
        print('ana - num\n', (np.abs(ana_hessian - num_hessian) > 1e-6).astype(int))
        assert np.allclose(ana_hessian, ana_hessian.T, atol=1e-6)
        assert np.allclose(ana_hessian, num_hessian, atol=1e-3)