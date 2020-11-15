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
from ase.units import GPa

import matscipytest
from matscipy.elasticity import (rotate_elastic_constants,
                                 elastic_moduli)

###


class TestElasticModuli(matscipytest.MatSciPyTestCase):

    fmax = 1e-6
    delta = 1e-6

    def test_rotation(self):
        Cs = [
             np.array([
             [165.7, 63.9, 63.9, 0, 0, 0],
             [63.9, 165.7, 63.9, 0, 0, 0],
             [63.9, 63.9, 165.7, 0, 0, 0],
             [0, 0, 0, 79.6, 0, 0],
             [0, 0, 0, 0, 79.6, 0],
             [0, 0, 0, 0, 0, 79.6],
             ])
             ]
        l = [np.array([1, 0, 0]), np.array([1, 1, 0])]
        EM = [
              {'E': np.array([130, 130, 130]),
              'nu': np.array([0.28, 0.28, 0.28]),
              'G': np.array([79.6, 79.6, 79.6])},
              {'E': np.array([169, 169, 130]),
              'nu': np.array([0.36, 0.28, 0.064]),
              'G': np.array([79.6, 79.6, 50.9])}
              ]
        for C in Cs:
            for i, directions in enumerate(l):
                directions = directions/np.linalg.norm(directions)

                E, nu, Gm, B, K = elastic_moduli(C, l=directions)

                nu_v = np.array([nu[1, 2], nu[2, 0], nu[0, 1]])
                G = np.array([Gm[1, 2], Gm[2, 0], Gm[0, 1]])

                self.assertArrayAlmostEqual(E, EM[i]['E'], tol=1e0)
                self.assertArrayAlmostEqual(nu_v, EM[i]['nu'], tol=1e-2)
                self.assertArrayAlmostEqual(G, EM[i]['G'], tol=1e-1)

###

if __name__ == '__main__':
    unittest.main()
