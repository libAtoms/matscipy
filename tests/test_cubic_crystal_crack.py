#
# Copyright 2014-2015, 2017, 2020-2021 Lars Pastewka (U. Freiburg)
#           2020 Johannes Hoermann (U. Freiburg)
#           2014, 2017 James Kermode (Warwick U.)
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

import math
import unittest

import numpy as np
try:
    from scipy.integrate import quadrature
except:
    quadrature = None

import ase.io
from ase.constraints import FixAtoms
from ase.optimize import FIRE
from ase.lattice.cubic import FaceCenteredCubic, SimpleCubic

import matscipytest
import matscipy.fracture_mechanics.clusters as clusters
from matscipy.neighbours import neighbour_list
from matscipy.elasticity import measure_triclinic_elastic_constants
from matscipy.elasticity import Voigt_6x6_to_cubic
from matscipy.fracture_mechanics.crack import CubicCrystalCrack
from matscipy.fracture_mechanics.crack import \
    isotropic_modeI_crack_tip_displacement_field
from matscipy.fracture_mechanics.idealbrittlesolid import IdealBrittleSolid

###

class TestCubicCrystalCrack(matscipytest.MatSciPyTestCase):

    delta = 1e-6

    def setUp(self):
        E = 100
        nu = 0.3

        K = E/(3.*(1-2*nu))
        C44 = E/(2.*(1+nu))
        C11 = K+4.*C44/3.
        C12 = K-2.*C44/3.

        #             C11, C12, C44, surface energy, k1
        self.materials = [(1.0, 0.5, 0.3, 1.0, 1.0),
                          (1.0, 0.7, 0.3, 1.3, 1.0),
                          (C11, C12, C44, 1.77, 1.0)]

        #self.materials = [(C11, C12, C44, 1.0, 1.0)]

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
        r = np.linspace(0.5, 10.0, 101)

        k = crack.crack.k1g(1.0)

        u, v = crack.crack.displacements(r, theta, k)
        ref_u, ref_v = isotropic_modeI_crack_tip_displacement_field(k, C44, nu,
                                                                    r, theta)

        self.assertArrayAlmostEqual(u, ref_u)
        self.assertArrayAlmostEqual(v, ref_v)

    def test_anisotropic_near_field_solution(self):
        """
        Run an atomistic calculation of a harmonic solid and compare to
        continuum solution.
        """

        for nx in [ 8, 16, 32, 64 ]:
            for calc, a, C11, C12, C44, surface_energy, bulk_coordination in [
                #( atomistica.DoubleHarmonic(k1=1.0, r1=1.0, k2=1.0,
                #                            r2=math.sqrt(2), cutoff=1.6),
                #  clusters.sc('He', 1.0, [nx,nx,1], [1,0,0], [0,1,0]),
                #  3, 1, 1, 0.05, 6 ),
                (
                  #atomistica.Harmonic(k=1.0, r0=1.0, cutoff=1.3, shift=True),
                  IdealBrittleSolid(k=1.0, a=1.0, rc=1.3),
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

                #ase.io.write('initial_{}.xyz'.format(nx), a, format='extxyz', write_results=False)

                x1, y1, z1 = a.positions.copy().T
                FIRE(a, logfile=None).run(fmax=1e-3)
                x2, y2, z2 = a.positions.T

                # Get coordination numbers and find properly coordinated atoms
                i = neighbour_list("i", a, 1.1)
                coord = np.bincount(i, minlength=len(a))
                mask=coord == bulk_coordination

                residual = np.sqrt(((x2-x1)/u)**2 + ((y2-y1)/v)**2)

                #a.set_array('residual', residual)
                #ase.io.write('final_{}.xyz'.format(nx), a, format='extxyz')

                #print(np.max(residual[mask]))
                self.assertTrue(np.max(residual[mask]) < 0.2)

    def test_consistency_of_deformation_gradient_and_stress(self):
        for C11, C12, C44, surface_energy, k1 in self.materials:
            crack = CubicCrystalCrack([1,0,0], [0,1,0], C11, C12, C44)
            k = crack.k1g(surface_energy)*k1
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
        eps = 1e-3
        for C11, C12, C44, surface_energy, k1 in self.materials:
            crack = CubicCrystalCrack([1,0,0], [0,1,0], C11, C12, C44)
            k = crack.k1g(surface_energy)*k1
            for i in range(10):
                x = i+1
                y = i+1
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
                self.assertArrayAlmostEqual(F, F_num, tol=1e-4)

    def test_elastostatics(self):
        eps = 1e-3
        for C11, C12, C44, surface_energy, k1 in self.materials:
            crack = CubicCrystalCrack([1,0,0], [0,1,0], C11, C12, C44)
            k = crack.k1g(surface_energy)*k1
            for i in range(10):
                x = i+1
                y = i+1
                # Finite difference approximation of the stress divergence
                sxx0x, syy0x, sxy0x = crack.stresses(x-eps, y, 0, 0, k)
                sxx0y, syy0y, sxy0y = crack.stresses(x, y-eps, 0, 0, k)
                sxx1x, syy1x, sxy1x = crack.stresses(x+eps, y, 0, 0, k)
                sxx1y, syy1y, sxy1y = crack.stresses(x, y+eps, 0, 0, k)
                divsx = (sxx1x-sxx0x)/(2*eps) + (sxy1y-sxy0y)/(2*eps)
                divsy = (sxy1x-sxy0x)/(2*eps) + (syy1y-syy0y)/(2*eps)
                # Check that divergence of stress is zero (elastostatic
                # equilibrium)
                self.assertAlmostEqual(divsx, 0.0, places=3)
                self.assertAlmostEqual(divsy, 0.0, places=3)

    def test_J_integral(self):
        if quadrature is None:
            print('No scipy.integrate.quadrature. Skipping J-integral test.')
            return

        for C11, C12, C44, surface_energy, k1 in self.materials:
            crack = CubicCrystalCrack([1,0,0], [0,1,0], C11, C12, C44)
            k = crack.k1g(surface_energy)*k1

            def polar_path(theta, r=1, x0=0, y0=0):
                nx = np.cos(theta)
                ny = np.sin(theta)
                n = np.transpose([nx, ny])
                return r*n-np.array([x0, y0]), n, r

            def elliptic_path(theta, r=1, x0=0, y0=0):
                rx, ry = r
                x = rx*np.cos(theta)
                y = ry*np.sin(theta)
                nx = ry*np.cos(theta)
                ny = rx*np.sin(theta)
                ln = np.sqrt(nx**2+ny**2)
                nx /= ln
                ny /= ln
                ds = np.sqrt((rx*np.sin(theta))**2 + (ry*np.cos(theta))**2)
                return np.transpose([x-x0, y-y0]), np.transpose([nx, ny]), ds

            def rectangular_path(t, r):
                x = -r*np.ones_like(t)
                y = r*(t-8)
                nx = -np.ones_like(t)
                ny = np.zeros_like(t)
                x = np.where(t < 7, r*(6-t), x)
                y = np.where(t < 7, -r*np.ones_like(t), y)
                nx = np.where(t < 7, np.zeros_like(t), nx)
                ny = np.where(t < 7, -np.ones_like(t), ny)
                x = np.where(t < 5, r*np.ones_like(t), x)
                y = np.where(t < 5, r*(4-t), y)
                nx = np.where(t < 5, np.ones_like(t), nx)
                ny = np.where(t < 5, np.zeros_like(t), ny)
                x = np.where(t < 3, r*(t-2), x)
                y = np.where(t < 3, r*np.ones_like(t), y)
                nx = np.where(t < 3, np.zeros_like(t), nx)
                ny = np.where(t < 3, np.ones_like(t), ny)
                x = np.where(t < 1, -r*np.ones_like(t), x)
                y = np.where(t < 1, r*t, y)
                nx = np.where(t < 1, -np.ones_like(t), nx)
                ny = np.where(t < 1, np.zeros_like(t), ny)
                return np.transpose([x, y]), np.transpose([nx, ny]), r

            def J(t, path_func=polar_path):
                # Position, normal to path, length
                pos, n, ds = path_func(t)
                x, y = pos.T
                nx, ny = n.T
                # Stress tensor
                sxx, syy, sxy = crack.stresses(x, y, 0, 0, k)
                s = np.array([[sxx, sxy], [sxy, syy]]).T
                # Deformation gradient and strain tensor
                F = crack.deformation_gradient(x, y, 0, 0, k)
                eps = (F+F.swapaxes(1, 2))/2 - np.identity(2).reshape(1, 2, 2)
                # Strain energy density
                W = (s*eps).sum(axis=2).sum(axis=1)/2
                # Note dy == nx*ds
                retval = (W*nx - np.einsum('...j,...jk,...k->...', n, s, F[..., 0, :]))*ds
                return retval

            eps = 1e-6
            for r in [1, 10, 100]:
                # Polar path
                J_val, J_err = quadrature(J, -np.pi+eps, np.pi-eps,
                                          args=(lambda t: polar_path(t, r)), )
                self.assertAlmostEqual(J_val, 2*surface_energy, places=2)
                # Elliptic path
                J_val, J_err = quadrature(J, -np.pi+eps, np.pi-eps,
                                          args=(lambda t: elliptic_path(t, (3*r, r)), ),
                                          maxiter=200)
                self.assertAlmostEqual(J_val, 2*surface_energy, places=2)
                # Elliptic path, shifted in x and y directions off-center
                J_val, J_err = quadrature(J, -np.pi+eps, np.pi-eps,
                                          args=(lambda t: elliptic_path(t, (r, 1.5*r), 0, 0.5), ),
                                          maxiter=200)
                self.assertAlmostEqual(J_val, 2*surface_energy, places=2)
                #J_val, J_err = quadrature(J, eps, 8-eps, args=(r, rectangular_path))
                #print('rectangular: J =', J_val, J_err)
                #self.assertAlmostEqual(J_val, surface_energy, places=2)

###

if __name__ == '__main__':
    unittest.main()
