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

import os
import unittest

import numpy as np

import ase
from ase.lattice.cubic import Diamond

import matscipytest
from matscipy.checkpoint import Checkpoint

###

class TestCheckpoint(matscipytest.MatSciPyTestCase):

    def op1(self, a, m):
        a[1].position += m*np.array([0.1, 0.2, 0.3])
        return a

    def op2(self, a, m):
        a += ase.Atom('C', m*np.array([0.2, 0.3, 0.1]))
        return a

    def test_sqlite(self):
        print 'test_single_file'

        try:
            os.remove('checkpoints.db')
        except OSError:
            pass

        cp = Checkpoint('checkpoints.db')
        a = Diamond('Si', size=[2,2,2])
        a = cp(self.op1, a, 1.0)
        op1a = a.copy()
        a = cp(self.op2, a, 2.0)
        op2a = a.copy()

        cp = Checkpoint('checkpoints.db')
        a = Diamond('Si', size=[2,2,2])
        a = cp(self.op1, a, 1.0)
        self.assertAtomsAlmostEqual(a, op1a)
        a = cp(self.op2, a, 2.0)
        self.assertAtomsAlmostEqual(a, op2a)

###

if __name__ == '__main__':
    unittest.main()
