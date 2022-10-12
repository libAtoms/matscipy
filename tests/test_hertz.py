#
# Copyright 2014-2015, 2020 Lars Pastewka (U. Freiburg)
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

import matscipytest
import matscipy.contact_mechanics.Hertz as Hertz

###

class TestHertz(matscipytest.MatSciPyTestCase):

    def test_Hertz_centerline_stress(self):
        z = np.linspace(0.0, 5.0, 101)
        for nu in [ 0.3, 0.5 ]:
            srr1, szz1 = Hertz.centerline_stress(z, poisson=nu)
            stt2, srr2, szz2, srz2 = Hertz.stress(np.zeros_like(z), z, poisson=nu)

            self.assertTrue(np.max(np.abs(srr1-srr2)) < 1e-6)
            self.assertTrue(np.max(np.abs(srr1-stt2)) < 1e-6)
            self.assertTrue(np.max(np.abs(szz1-szz2)) < 1e-6)


    def test_Hertz_surface_stress(self):
        r = np.linspace(0.0, 5.0, 101)
        for nu in [ 0.3, 0.5 ]:
            pzz1, srr1, stt1 = Hertz.surface_stress(r, poisson=nu)
            stt2, srr2, szz2, srz2 = Hertz.stress(r, np.zeros_like(r), poisson=nu)

            self.assertTrue(np.max(np.abs(pzz1+szz2)) < 1e-6)
            self.assertTrue(np.max(np.abs(srr1-srr2)) < 1e-6)
            self.assertTrue(np.max(np.abs(stt1-stt2)) < 1e-6)

    def test_Hertz_subsurface_stress(self):
        nx = 51 # Grid size
        a = 32. # Contact radius

        y = np.linspace(0, 3*a, nx)
        z = np.linspace(0, 3*a, nx)

        y, z = np.meshgrid(y, z)

        x = np.zeros_like(y)

        # nu: Poisson
        for nu in [0.3, 0.5]:
            sxx, syy, szz, syz, sxz, sxy = \
                Hertz.stress_Cartesian(x/a, y/a, z/a, poisson=nu)

            r_sq = (x**2 + y**2)/a**2
            stt2, srr2, szz2, srz2 = \
                Hertz.stress(np.sqrt(r_sq), z/a, poisson=nu)

            self.assertTrue(np.max(np.abs(sxx-stt2)) < 1e-12)
            self.assertTrue(np.max(np.abs(syy-srr2)) < 1e-12)
            self.assertTrue(np.max(np.abs(szz-szz2)) < 1e-12)
            self.assertTrue(np.max(np.abs(syz-srz2)) < 1e-12)

###

if __name__ == '__main__':
    unittest.main()
