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

from __future__ import division

from math import pi, sqrt

import unittest

import numpy as np

import matscipytest
import matscipy.contact_mechanics.greens_function as gf
import matscipy.contact_mechanics.Hertz as Hertz

###

class TestGreensFunction(matscipytest.MatSciPyTestCase):

    def test_Hertz_displacements_square(self):
        nx = 256 # Grid size
        a = 32. # Contact radius
        G, x, y = gf.real_to_reciprocal_space(
            nx, nx, gf=gf.square_pressure__nonperiodic,
            coordinates=True)

        r_sq = (x**2 + y**2)/a**2
        P = np.where(r_sq > 1., np.zeros_like(r_sq), np.sqrt(1.-r_sq))
        u = -np.fft.ifft2(G*np.fft.fft2(P)).real
        # Note: contact modulus Es = 2 for the GF
        uref = a/2. * Hertz.surface_displacements(np.sqrt(r_sq))

        # Only the middle section of displacements is correct in for the
        # free-space Green's function.
        u = u[:nx//4, :nx//4]
        uref = uref[:nx//4, :nx//4]
        #np.savetxt('u.out', u)
        #np.savetxt('u0.out', np.transpose([y[0,:nx//4], u[0,:]]))
        #np.savetxt('uref.out', uref)
        #np.savetxt('uref0.out', np.transpose([y[0,:nx//4], uref[0,:]]))
        #np.savetxt('u_uref.out', u/uref)
        self.assertTrue(np.max(np.abs((u-uref)/uref)) < 0.01)

    def test_Hertz_stress(self):
        nx = 256 # Grid size
        a = 32. # Contact radius

        # z: Depth at which to compute stress
        # nu: Poisson
        for z, nu in [ ( 16., 0.5 ), ( 32., 0.5 ), ( 16., 0.3 ) ]:
            ( Gxx, Gyy, Gzz, Gyz, Gxz, Gxy ), x, y = \
                gf.real_to_reciprocal_space(
                    nx, nx, gf=lambda x, y:
                    gf.point_traction__nonperiodic('Z', x, y, z, nu=nu),
                    coordinates=True)

            r_sq = (x**2 + y**2)/a**2
            P = np.where(r_sq > 1., np.zeros_like(r_sq), np.sqrt(1.-r_sq))
            sxx = -np.fft.ifft2(Gxx*np.fft.fft2(P)).real
            syy = -np.fft.ifft2(Gyy*np.fft.fft2(P)).real
            szz = -np.fft.ifft2(Gzz*np.fft.fft2(P)).real
            syz = -np.fft.ifft2(Gyz*np.fft.fft2(P)).real
            sttref, srrref, szzref, srzref = \
                Hertz.stress(np.sqrt(r_sq), z/a, nu=nu)

            #np.savetxt('s0.out', np.transpose([y[0,:nx//4], sxx[0,:nx//4],
            #                                   syy[0,:nx//4], szz[0,:nx//4],
            #                                   syz[0,:nx//4]]))
            #np.savetxt('sref0.out', np.transpose([y[0,:nx//4], sttref[0,:nx//4],
            #                                     srrref[0,:nx//4], szzref[0,:nx//4],
            #                                     srzref[0,:nx//4]]))
            self.assertTrue(np.max(np.abs((sxx[0,1:nx//4]-sttref[0,1:nx//4])/
                                           sttref[0,1:nx//4])) < 1e-2)
            self.assertTrue(np.max(np.abs((syy[0,1:nx//4]-srrref[0,1:nx//4])/
                                           srrref[0,1:nx//4])) < 1e-2)
            self.assertTrue(np.max(np.abs((szz[0,1:nx//4]-szzref[0,1:nx//4])/
                                           szzref[0,1:nx//4])) < 1e-2)
            self.assertTrue(np.max(np.abs((syz[0,1:nx//4]-srzref[0,1:nx//4])/
                                           srzref[0,1:nx//4])) < 1e-2)

    def test_min_ccg(self):
        nx = 256 # Grid size
        R = 256. # sphere radius
        d = R/20. # penetration depth
        G, x, y = gf.real_to_reciprocal_space(
            nx, nx, gf=gf.square_pressure__nonperiodic,
            coordinates=True)

        r_sq = (x**2 + y**2)/R**2
        h = np.where(r_sq > 1., (R-d)*np.ones_like(r_sq), (R-d)-R*np.sqrt(1.-r_sq))

        # Contact radius
        a = R*sqrt(d/R)
        p0 = 2/pi*sqrt(d/R)
        r_sq = (x**2 + y**2)/a**2
        p_analytic, sr, stheta = Hertz.surface_stress(np.sqrt(r_sq), nu=0.5)
        p_analytic *= p0

        u, p = gf.min_ccg(h, G)
        p /= 2 # E* of GF is 2

        assert (p-p_analytic).max() < 1e-3

###

if __name__ == '__main__':
    unittest.main()

