#
# Copyright 2014-2016, 2021 Lars Pastewka (U. Freiburg)
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
#                  Adrien Gola, Karlsruhe Institute of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# ======================================================================

import unittest

import numpy as np

from matscipy.io import loadtbl, savetbl

import matscipytest

###

class TestEAMIO(matscipytest.MatSciPyTestCase):

    def test_savetbl_loadtbl(self):
        n = 123
        a = np.random.random(n)
        b = np.random.random(n)
        poe = np.random.random(n)
        savetbl('test.out', a=a, b=b, poe=poe)

        data = loadtbl('test.out')
        self.assertArrayAlmostEqual(a, data['a'])
        self.assertArrayAlmostEqual(b, data['b'])
        self.assertArrayAlmostEqual(poe, data['poe'])
    
    def test_savetbl_loadtbl_text(self):
        n = 12
        a = np.random.random(n)
        b = np.random.random(n)
        t = ['a'*(i+1) for i in range(n)]
        savetbl('test2.out', a=a, b=b, t=t)

        a2, t2, b2 = loadtbl('test2.out', usecols=['a', 't', 'b'], types={'t': np.str_})
        self.assertArrayAlmostEqual(a, a2)
        self.assertArrayAlmostEqual(b, b2)
        assert (t == t2).all()

            
###

if __name__ == '__main__':
    unittest.main()
