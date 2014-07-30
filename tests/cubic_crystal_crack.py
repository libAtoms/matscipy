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

import math
import unittest

import numpy as np

from matscipy.fracture_mechanics.crack import CubicCrystalCrack
from matscipy.fracture_mechanics.crack import \
    isotropic_modeI_crack_tip_displacement_field

import matplotlib.pyplot as plt

###

class TestCubicCrystalCrack(unittest.TestCase):

    def test_isotropic_near_field_solution(self):
        """
        Check if we recover the near field solution for isotropic cracks.
        """

        E = 100
        nu = 0.3

        K = E/(3.*(1-2*nu))
        C44 = E/(2.*(1+nu))
        C11 = K+4.*C44/3.
        C12 = K-2.*C44/3.
        kappa = 3-4*nu
        #kappa = 4./(1+nu)-1

        crack = CubicCrystalCrack(C11, C12, C44, [1,0,0], [0,1,0])
   
        #r = np.random.random(10)*10
        #theta = np.random.random(10)*2*math.pi

        theta = np.linspace(0.0, math.pi, 101)
        r = 1.0*np.ones_like(theta)
        
        k = 1.0

        u, v = crack.crack.displacements(r, theta, k)
        ref_u, ref_v = isotropic_modeI_crack_tip_displacement_field(k, C44, nu,
                                                                    r, theta)
                                                                    
        self.assertTrue(np.all(np.abs(u-ref_u) < 1e-8))
        self.assertTrue(np.all(np.abs(v-ref_v) < 1e-8))

###

if __name__ == '__main__':
    unittest.main()
