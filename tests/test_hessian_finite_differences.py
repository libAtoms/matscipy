#
# Copyright 2020-2021 Lars Pastewka (U. Freiburg)
#           2020 Jan Griesser (U. Freiburg)
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

import ase.io as io
from ase.constraints import StrainFilter, UnitCellFilter
from ase.lattice.compounds import B1, B2, L1_0, L1_2
from ase.lattice.cubic import FaceCenteredCubic
from ase.optimize import FIRE
from ase.units import GPa
from ase.build import bulk 

import matscipytest
from matscipy.calculators.pair_potential import PairPotential, LennardJonesCut
import matscipy.calculators.pair_potential as calculator
from matscipy.numerical import numerical_hessian

###


class TestPairPotentialCalculator(matscipytest.MatSciPyTestCase):

    tol = 1e-4

    def test_hessian_sparse(self):
        for calc in [{(1, 1): LennardJonesCut(1, 1, 3)}]:
            atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=1.550)
            a = calculator.PairPotential(calc)
            atoms.calc = a
            H_analytical = a.get_hessian(atoms, "sparse")
            H_numerical = numerical_hessian(atoms, dx=1e-5, indices=None)
            self.assertArrayAlmostEqual(H_analytical.todense(), H_numerical.todense(), tol=self.tol)

    def test_symmetry_sparse(self):
        for calc in [{(1, 1): LennardJonesCut(1, 1, 3)}]:
            atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=1.550)
            a = calculator.PairPotential(calc)
            atoms.calc = a
            H_numerical = numerical_hessian(atoms, dx=1e-5, indices=None)
            H_numerical = H_numerical.todense()
            self.assertArrayAlmostEqual(np.sum(np.abs(H_numerical-H_numerical.T)), 0, tol=1e-5)

    def test_hessian_sparse_split(self):
        for calc in [{(1, 1): LennardJonesCut(1, 1, 3)}]:
            atoms = FaceCenteredCubic('H', size=[2,2,2], latticeconstant=1.550)
            nat = len(atoms)
            a = calculator.PairPotential(calc)
            atoms.calc = a
            H_analytical = a.get_hessian(atoms, "sparse")
            H_numerical_split1 = numerical_hessian(atoms, dx=1e-5, indices=np.arange(0, np.int32(nat/2), 1))
            H_numerical_split2 = numerical_hessian(atoms, dx=1e-5, indices=np.arange(np.int32(nat/2), nat, 1))
            H_numerical_splitted = np.concatenate((H_numerical_split1.todense(), H_numerical_split2.todense()), axis=0)
            self.assertArrayAlmostEqual(H_analytical.todense(), H_numerical_splitted, tol=self.tol)


###


if __name__ == '__main__':
    unittest.main()
