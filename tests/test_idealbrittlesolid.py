#
# Copyright 2014-2015, 2017, 2020-2021 Lars Pastewka (U. Freiburg)
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

import unittest

import numpy as np

import ase
import ase.lattice.hexagonal

import matscipytest
from matscipy.fracture_mechanics.idealbrittlesolid import (
    find_triangles_2d,
    IdealBrittleSolid,
    triangular_lattice_slab,
)
from matscipy.numerical import numerical_forces, numerical_stress

###

class TestNeighbours(matscipytest.MatSciPyTestCase):

    tol = 1e-6

    def test_forces_and_virial(self):
        a = triangular_lattice_slab(1.0, 2, 2)
        calc = IdealBrittleSolid(rc=1.2, beta=0.0)
        a.set_calculator(calc)
        a.rattle(0.1)
        f = a.get_forces()
        fn = numerical_forces(a)
        self.assertArrayAlmostEqual(f, fn, tol=self.tol)
        self.assertArrayAlmostEqual(a.get_stress(),
                                    numerical_stress(a),
                                    tol=self.tol)

    def test_forces_linear(self):
        a = triangular_lattice_slab(1.0, 1, 1)
        calc = IdealBrittleSolid(rc=1.2, beta=0.0, linear=True)
        calc.set_reference_crystal(a)
        a.set_calculator(calc)
        a.rattle(0.01)
        f = a.get_forces()
        fn = numerical_forces(a)
        self.assertArrayAlmostEqual(f, fn, tol=self.tol)

    def test_two_triangles(self):
        a = ase.Atoms('4Xe', [[0,0,0], [1,0,0], [1,1,0], [0,1,0]])
        a.center(vacuum=10)
        c1, c2, c3 = find_triangles_2d(a, 1.1)
        self.assertArrayAlmostEqual(np.transpose([c1, c2, c3]), [[0,1,2], [0,1,3], [0,2,3], [1,2,3]])

###

if __name__ == '__main__':
    unittest.main()

