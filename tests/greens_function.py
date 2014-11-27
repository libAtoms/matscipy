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

import matscipytest
import matscipy.contact_mechanics.greens_function as gf
import matscipy.contact_mechanics.Hertz as Hertz

###

class TestGreensFunction(matscipytest.MatSciPyTestCase):

    def test_Hertz_displacements(self):
        nx = 256 # Use 128 x 128 grid
        a = 32. # Contact radius
        G, x, y = gf.reciprocal_grid(nx, nx, gf=gf.gf_displacement_nonperiodic,
                                     coordinates=True)

        r_sq = (x**2 + y**2)/a**2
        P = np.where(r_sq > 1., np.zeros_like(r_sq), np.sqrt(1.-r_sq))
        u = -np.fft.ifft2(G*np.fft.fft2(P)).real
        # Note: contact modulus Es = 2 for the GF
        uref = a/2. * Hertz.surface_displacements(np.sqrt(r_sq))

        # Only the middle section of displacements is correct in for the
        # free-space Green's function.
        u = u[:nx/4, :nx/4]
        uref = uref[:nx/4, :nx/4]
        #np.savetxt('u.out', u)
        #np.savetxt('u0.out', np.transpose([y[0,:nx/4], u[0,:]]))
        #np.savetxt('uref.out', uref)
        #np.savetxt('uref0.out', np.transpose([y[0,:nx/4], uref[0,:]]))
        #np.savetxt('u_uref.out', u/uref)
        self.assertTrue(np.max(np.abs((u-uref)/uref)) < 0.01)

###

if __name__ == '__main__':
    unittest.main()

