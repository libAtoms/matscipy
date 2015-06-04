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

import unittest

import numpy as np

# from ase.calculators.eam import EAM
# from ase.constraints import StrainFilter
# from ase.lattice.cubic import Diamond, FaceCenteredCubic
# from ase.optimize import FIRE
# from ase.units import GPa

import matscipytest
from matscipy.elasticity import (rotate_elastic_constants,
                                 elastic_moduli)

###


class TestElasticModuli(matscipytest.MatSciPyTestCase):

    fmax = 1e-6
    delta = 1e-6

    def test_rotation(self):
        n_atoms = 3
        for atom in range(n_atoms):
            C = np.random.randint(300, size=(6, 6))
            C = (C.T+C)/2

            for directions in [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                               [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                               [[1, 1, 0], [0, 0, 1], [1, -1, 0]],
                               [[1, 1, 1], [-1, -1, 2], [1, -1, 0]]]:
                a, b, c = directions

                directions = np.array([np.array(x)/np.linalg.norm(x)
                                       for x in directions])

                E, nu, Gm, B, K = elastic_moduli(C)
                E1, nu1, Gm1, B1, K1 = elastic_moduli(C, R=directions)

                Cr = rotate_elastic_constants(C, directions)
                Er, nur, Gmr, Br, Kr = elastic_moduli(Cr)
                E1r, nu1r, Gm1r, B1r, K1r = elastic_moduli(
                                              Cr,
                                              R=np.transpose(directions)
                                              )

                self.assertArrayAlmostEqual(E1, Er, tol=1e-6)
                self.assertArrayAlmostEqual(E, E1r, tol=1e-6)
                self.assertArrayAlmostEqual(nu1, nur, tol=1e-6)
                self.assertArrayAlmostEqual(nu, nu1r, tol=1e-6)
                self.assertArrayAlmostEqual(Gm1, Gmr, tol=1e-6)
                self.assertArrayAlmostEqual(Gm, Gm1r, tol=1e-6)
                self.assertArrayAlmostEqual(B1, Br, tol=1e-6)
                self.assertArrayAlmostEqual(B, B1r, tol=1e-6)
                self.assertArrayAlmostEqual(K1, Kr, tol=1e-6)
                self.assertArrayAlmostEqual(K, K1r, tol=1e-6)

###

if __name__ == '__main__':
    unittest.main()
