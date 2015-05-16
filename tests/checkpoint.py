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

import os
import unittest

import numpy as np

import ase
from ase.lattice.cubic import Diamond

import matscipytest
from matscipy.checkpoint import Checkpoint, CheckpointCalculator
from matscipy.logger import screen
from matscipy.fracture_mechanics.idealbrittlesolid import IdealBrittleSolid, triangular_lattice_slab

###

class TestCheckpoint(matscipytest.MatSciPyTestCase):

    def op1(self, a, m):
        a[1].position += m*np.array([0.1, 0.2, 0.3])
        return a

    def op2(self, a, m):
        a += ase.Atom('C', m*np.array([0.2, 0.3, 0.1]))
        return a, a.positions[0]

    def test_sqlite(self):
        print('test_single_file')

        try:
            os.remove('checkpoints.db')
        except OSError:
            pass

        CP = Checkpoint('checkpoints.db')
        a = Diamond('Si', size=[2,2,2])
        a = CP(self.op1)(a, 1.0)
        op1a = a.copy()
        a, ra = CP(self.op2)(a, 2.0)
        op2a = a.copy()
        op2ra = ra.copy()

        CP = Checkpoint('checkpoints.db')
        a = Diamond('Si', size=[2,2,2])
        a = CP(self.op1)(a, 1.0)
        self.assertAtomsAlmostEqual(a, op1a)
        a, ra = CP(self.op2)(a, 2.0)
        self.assertAtomsAlmostEqual(a, op2a)
        self.assertArrayAlmostEqual(ra, op2ra)

class TestCheckpointCalculator(matscipytest.MatSciPyTestCase):

    def rattle_calc(self, atoms, calc):
        try:
            os.remove('checkpoints.db')
        except OSError:
            pass

        orig_atoms = atoms.copy()        
        
        # first do a couple of calculations
        np.random.seed(0)
        atoms.rattle()
        cp_calc_1 = CheckpointCalculator(calc, logger=screen)
        atoms.set_calculator(cp_calc_1)
        e11 = atoms.get_potential_energy()
        f11 = atoms.get_forces()
        atoms.rattle()
        e12 = atoms.get_potential_energy()
        f12 = atoms.get_forces()
        
        # then re-read them from checkpoint file
        atoms = orig_atoms
        np.random.seed(0)
        atoms.rattle()
        cp_calc_2 = CheckpointCalculator(calc, logger=screen)
        atoms.set_calculator(cp_calc_2)
        e21 = atoms.get_potential_energy()
        f21 = atoms.get_forces()
        atoms.rattle()
        e22 = atoms.get_potential_energy()
        f22 = atoms.get_forces()

        self.assertAlmostEqual(e11, e21)
        self.assertAlmostEqual(e12, e22)
        self.assertArrayAlmostEqual(f11, f21)
        self.assertArrayAlmostEqual(f12, f22)
    
    def test_new_style_interface(self):
        calc = IdealBrittleSolid()
        atoms = triangular_lattice_slab(1.0, 5, 5)
        self.rattle_calc(atoms, calc)

    def test_old_style_interface(self):
        try:
            import atomistica
        except ImportError:
            print('atomistica not available, skipping test')
            return
        atoms = Diamond('Si', size=[2,2,2])
        calc = atomistica.Kumagai()
        self.rattle_calc(atoms, calc)
        

###

if __name__ == '__main__':
    unittest.main()
