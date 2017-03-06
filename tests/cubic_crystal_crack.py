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

import math
import unittest

import numpy as np
from scipy.integrate import quadrature

import ase.io
from ase.constraints import FixAtoms
from ase.optimize import FIRE
from ase.lattice.cubic import FaceCenteredCubic, SimpleCubic

import matscipytest
import matscipy.fracture_mechanics.clusters as clusters
from matscipy.elasticity import measure_triclinic_elastic_constants
from matscipy.elasticity import Voigt_6x6_to_cubic
from matscipy.fracture_mechanics.crack import CubicCrystalCrack
from matscipy.fracture_mechanics.crack import \
    isotropic_modeI_crack_tip_displacement_field

try:
    import atomistica
except:
    atomistica = None

###

class TestCubicCrystalCrack(matscipytest.MatSciPyTestCase):

    delta = 1e-6
    #             C11, C12, C44, surface energy
    materials = [(1.0, 0.5, 0.3, 1.0),
                 (1.0, 0.5, 0.3, 10.0)]

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

        crack = CubicCrystalCrack([1,0,0], [0,1,0], C11, C12, C44)

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
            print('Atomistica not available. Skipping test.')
            return

        for nx in [ 8, 16, 32, 64 ]:
            for calc, a, C11, C12, C44, surface_energy, bulk_coordination in [
                #( atomistica.DoubleHarmonic(k1=1.0, r1=1.0, k2=1.0,
                #                            r2=math.sqrt(2), cutoff=1.6),
                #  clusters.sc('He', 1.0, [nx,nx,1], [1,0,0], [0,1,0]),
                #  3, 1, 1, 0.05, 6 ),
                ( atomistica.Harmonic(k=1.0, r0=1.0, cutoff=1.3, shift=True),
                  clusters.fcc('He', math.sqrt(2.0), [nx,nx,1], [1,0,0],
                               [0,1,0]),
                  math.sqrt(2), 1.0/math.sqrt(2), 1.0/math.sqrt(2), 0.05, 12)
                ]:
                clusters.set_groups(a, (nx, nx, 1), 0.5, 0.5)
                crack = CubicCrystalCrack([1,0,0], [0,1,0], C11, C12, C44)

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

                #print(np.max(residual[mask]))
                self.assertTrue(np.max(residual[mask]) < 0.2)

    def test_consistency_of_deformation_gradient_and_stress(self):
        for C11, C12, C44, surface_energy in self.materials:
            crack = CubicCrystalCrack([1,0,0], [0,1,0], C11, C12, C44)
            k = crack.k1g(surface_energy)
            for i in range(10):
                x = np.random.uniform(-10, 10)
                y = np.random.uniform(-10, 10)
                F = crack.deformation_gradient(x, y, 0, 0, k)
                eps = (F+F.T)/2-np.eye(2)
                sig_x, sig_y, sig_xy = crack.stresses(x, y, 0, 0, k)
                eps_xx = crack.crack.a11*sig_x + \
                         crack.crack.a12*sig_y + \
                         crack.crack.a16*sig_xy
                eps_yy = crack.crack.a12*sig_x + \
                         crack.crack.a22*sig_y + \
                         crack.crack.a26*sig_xy
                # Factor 1/2 because elastic constants and matrix product are
                # expressed in Voigt notation.
                eps_xy = (crack.crack.a16*sig_x + \
                          crack.crack.a26*sig_y + \
                          crack.crack.a66*sig_xy)/2
                self.assertAlmostEqual(eps[0, 0], eps_xx)
                self.assertAlmostEqual(eps[1, 1], eps_yy)
                self.assertAlmostEqual(eps[0, 1], eps_xy)

    def test_consistency_of_deformation_gradient_and_displacement(self):
        eps = 1e-6
        for C11, C12, C44, surface_energy in self.materials:
            crack = CubicCrystalCrack([1,0,0], [0,1,0], C11, C12, C44)
            k = crack.k1g(surface_energy)
            for i in range(10):
                x = np.random.uniform(-10, 10)
                y = np.random.uniform(-10, 10)
                F = crack.deformation_gradient(x, y, 0, 0, k)
                # Finite difference approximation of deformation gradient
                u, v = crack.displacements(x, y, 0, 0, k)
                ux0, vx0 = crack.displacements(x-eps, y, 0, 0, k)
                uy0, vy0 = crack.displacements(x, y-eps, 0, 0, k)
                ux1, vx1 = crack.displacements(x+eps, y, 0, 0, k)
                uy1, vy1 = crack.displacements(x, y+eps, 0, 0, k)
                du_dx = (ux1-ux0)/(2*eps)
                du_dy = (uy1-uy0)/(2*eps)
                dv_dx = (vx1-vx0)/(2*eps)
                dv_dy = (vy1-vy0)/(2*eps)
                du_dx += np.ones_like(du_dx)
                dv_dy += np.ones_like(dv_dy)
                F_num = np.transpose([[du_dx, du_dy], [dv_dx, dv_dy]])
                self.assertArrayAlmostEqual(F, F_num)

###

if __name__ == '__main__':
    unittest.main()
