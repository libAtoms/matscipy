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

import ase.io
from ase.constraints import FixAtoms
from ase.optimize import FIRE
from ase.lattice.cubic import FaceCenteredCubic, SimpleCubic

import matscipy.fracture_mechanics.clusters as clusters
from matscipy.elasticity import measure_triclinic_elastic_moduli
from matscipy.elasticity import Voigt_6x6_to_cubic
from matscipy.fracture_mechanics.crack import CubicCrystalCrack
from matscipy.fracture_mechanics.crack import \
    isotropic_modeI_crack_tip_displacement_field

try:
    import atomistica
except:
    atomistica = None

###

class TestCubicCrystalCrack(unittest.TestCase):

    delta = 1e-6

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
        
        k = crack.crack.k1g(1.0)

        u, v = crack.crack.displacements(r, theta, k)
        ref_u, ref_v = isotropic_modeI_crack_tip_displacement_field(k, C44, nu,
                                                                    r, theta)

        self.assertTrue(np.all(np.abs(u-ref_u) < 1e-6))
        self.assertTrue(np.all(np.abs(v-ref_v) < 1e-6))


    def test_anisotropic_near_field_solution(self):
        """
        Run an atomistic calculation of a harmonic solid and compare to
        continuum solution.
        """

        if not atomistica:
            print 'Atomistica not available. Skipping test.'

        for nx in [ 8, 16, 32, 64 ]:
            for calc, a, C11, C12, C44, surface_energy, bulk_coordination in [
                ( atomistica.DoubleHarmonic(k1=1.0, r1=1.0, k2=1.0,
                                            r2=math.sqrt(2), cutoff=1.6),
                  clusters.sc('He', 1.0, [nx,nx,1], [1,0,0], [0,1,0]),
                  3, 1, 1, 0.05, 6 ),
                ( atomistica.Harmonic(k=1.0, r0=1.0, cutoff=1.3, shift=True),
                  clusters.fcc('He', math.sqrt(2.0), [nx,nx,1], [1,0,0],
                               [0,1,0]),
                  math.sqrt(2), 1.0/math.sqrt(2), 1.0/math.sqrt(2), 0.05, 12)
                ]:
                crack = CubicCrystalCrack(C11, C12, C44, [1,0,0], [0,1,0])

                a.center(vacuum=20.0, axis=0)
                a.center(vacuum=20.0, axis=1)
                a.set_calculator(calc)

                sx, sy, sz = a.cell.diagonal()
                tip_x = sx/2
                tip_y = sy/2

                k1g = crack.k1g(surface_energy)
                r0 = a.positions.copy()

                u, v = crack.displacements(a.positions[:,0], a.positions[:,1],
                                           tip_x, tip_y, k1g)
                a.positions[:,0] += u
                a.positions[:,1] += v

                g = a.get_array('groups')
                a.set_constraint(FixAtoms(mask=g==0))

                #ase.io.write('initial_{}.xyz'.format(nx), a, format='extxyz')

                x1, y1, z1 = a.positions.copy().T
                FIRE(a, logfile=None).run(fmax=1e-3)
                x2, y2, z2 = a.positions.T

                # Get coordination numbers and find properly coordinated atoms
                coord = calc.nl.get_coordination_numbers(calc.particles, 1.1)
                mask=coord == bulk_coordination

                residual = np.sqrt(((x2-x1)/u)**2 + ((y2-y1)/v)**2)
                
                #a.set_array('residual', residual)
                #ase.io.write('final_{}.xyz'.format(nx), a, format='extxyz')

                self.assertTrue(np.max(residual[mask]) < 0.21)

###

if __name__ == '__main__':
    unittest.main()
