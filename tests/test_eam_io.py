#
# Copyright 2015, 2020-2021 Lars Pastewka (U. Freiburg)
#           2020 Wolfram G. Nöhring (U. Freiburg)
#           2015 Adrien Gola (KIT)
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

import os
from matscipy.calculators.eam.io import (read_eam,
                                         write_eam,
                                         mix_eam)
try:
    from scipy import interpolate
    from matscipy.calculators.eam import EAM
except:
    print('Warning: No scipy')
    interpolate = False


import ase.io as io
from ase.calculators.test import numeric_force
from ase.constraints import StrainFilter, UnitCellFilter
from ase.lattice.compounds import B1, B2, L1_0, L1_2
from ase.lattice.cubic import FaceCenteredCubic
from ase.optimize import FIRE

import matscipytest

###

class TestEAMIO(matscipytest.MatSciPyTestCase):

    tol = 1e-6

    def test_eam_read_write(self):
        source,parameters,F,f,rep = read_eam("Au_u3.eam",kind="eam")
        write_eam(source,parameters,F,f,rep,"Au_u3_copy.eam",kind="eam")
        source1,parameters1,F1,f1,rep1 = read_eam("Au_u3_copy.eam",kind="eam")
        os.remove("Au_u3_copy.eam")
        for i,p in enumerate(parameters):
            try:
                diff = p - parameters1[i]
            except:
                diff = None
            if diff is None:
                self.assertTrue(p == parameters1[i])
            else:
                print(i, p, parameters1[i], diff, self.tol, diff < self.tol)
                self.assertTrue(diff < self.tol)
        self.assertTrue((F == F1).all())
        self.assertTrue((f == f1).all())
        self.assertArrayAlmostEqual(rep, rep1)
    
    def test_eam_alloy_read_write(self):
        source,parameters,F,f,rep = read_eam("CuAgNi_Zhou.eam.alloy",kind="eam/alloy")
        write_eam(source,parameters,F,f,rep,"CuAgNi_Zhou.eam.alloy_copy",kind="eam/alloy")
        source1,parameters1,F1,f1,rep1 = read_eam("CuAgNi_Zhou.eam.alloy_copy",kind="eam/alloy")
        os.remove("CuAgNi_Zhou.eam.alloy_copy")
        fail = 0
        for i,p in enumerate(parameters):
            try:
              for j,d in enumerate(p):
                  if d != parameters[i][j]:
                      fail+=1
            except:
                if p != parameters[i]:
                    fail +=1
        self.assertTrue(fail == 0)
        self.assertTrue((F == F1).all())
        self.assertTrue((f == f1).all())
        for i in range(len(rep)):
            for j in range(len(rep)):
                if j < i :
                    self.assertTrue((rep[i,j,:] == rep1[i,j,:]).all())
                    
    def test_eam_fs_read_write(self):
        source,parameters,F,f,rep = read_eam("CuZr_mm.eam.fs",kind="eam/fs")
        write_eam(source,parameters,F,f,rep,"CuZr_mm.eam.fs_copy",kind="eam/fs")
        source1,parameters1,F1,f1,rep1 = read_eam("CuZr_mm.eam.fs_copy",kind="eam/fs")
        os.remove("CuZr_mm.eam.fs_copy")
        fail = 0
        for i,p in enumerate(parameters):
            try:
              for j,d in enumerate(p):
                  if d != parameters[i][j]:
                      fail+=1
            except:
                if p != parameters[i]:
                    fail +=1
        self.assertTrue(fail == 0)
        self.assertTrue((F == F1).all())
        for i in range(f.shape[0]):
            for j in range(f.shape[0]):
                self.assertTrue((f[i,j,:] == f1[i,j,:]).all())
        for i in range(len(rep)):
            for j in range(len(rep)):
                if j < i :
                    self.assertTrue((rep[i,j,:] == rep1[i,j,:]).all())
         
    def test_mix_eam_alloy(self):
        if False:
            source,parameters,F,f,rep = read_eam("CuAu_Zhou.eam.alloy",kind="eam/alloy")
            source1,parameters1,F1,f1,rep1 = mix_eam(["Cu_Zhou.eam.alloy","Au_Zhou.eam.alloy"],"eam/alloy","weight")
            write_eam(source1,parameters1,F1,f1,rep1,"CuAu_mixed.eam.alloy",kind="eam/alloy")

            calc0 = EAM('CuAu_Zhou.eam.alloy')
            calc1 = EAM('CuAu_mixed.eam.alloy')
       
            a = FaceCenteredCubic('Cu', size=[2,2,2])
            a.set_calculator(calc0)
            FIRE(StrainFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e0 = a.get_potential_energy()/len(a)
            a = FaceCenteredCubic('Cu', size=[2,2,2])
            a.set_calculator(calc1)
            FIRE(StrainFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e1 = a.get_potential_energy()/len(a)
            self.assertTrue(e0-e1 < 0.0005)

            a = FaceCenteredCubic('Au', size=[2,2,2])
            a.set_calculator(calc0)
            FIRE(StrainFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e0 = a.get_potential_energy()/len(a)
            a = FaceCenteredCubic('Au', size=[2,2,2])
            a.set_calculator(calc1)
            FIRE(StrainFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e1 = a.get_potential_energy()/len(a)
            self.assertTrue(e0-e1 < 0.0005)

            a = L1_2(['Au', 'Cu'], size=[2,2,2], latticeconstant=4.0)
            a.set_calculator(calc0)
            FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e0 = a.get_potential_energy()/len(a)
            a = L1_2(['Au', 'Cu'], size=[2,2,2], latticeconstant=4.0)
            a.set_calculator(calc1)
            FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e1 = a.get_potential_energy()/len(a)
            self.assertTrue(e0-e1 < 0.0005)

            a = L1_2(['Cu', 'Au'], size=[2,2,2], latticeconstant=4.0)
            a.set_calculator(calc0)
            FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e0 = a.get_potential_energy()/len(a)
            a = L1_2(['Cu', 'Au'], size=[2,2,2], latticeconstant=4.0)
            a.set_calculator(calc1)
            FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e1 = a.get_potential_energy()/len(a)
            self.assertTrue(e0-e1 < 0.0005)

            a = B1(['Au', 'Cu'], size=[2,2,2], latticeconstant=4.0)
            a.set_calculator(calc0)
            FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e0 = a.get_potential_energy()/len(a)
            a = B1(['Au', 'Cu'], size=[2,2,2], latticeconstant=4.0)
            a.set_calculator(calc1)
            FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e1 = a.get_potential_energy()/len(a)
            self.assertTrue(e0-e1 < 0.0005)
          
            os.remove("CuAu_mixed.eam.alloy")
            
###

if __name__ == '__main__':
    unittest.main()
